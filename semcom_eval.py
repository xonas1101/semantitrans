"""Semantic communication over a simulated noisy wireless channel.

Compares two transmission schemes at a range of channel SNRs:

  TRADITIONAL  transmit the raw speech waveform (16 kHz x 16-bit PCM) through
               an AWGN channel; the receiver runs ASR + translation.
               Bits/message = n_samples * 16 (~256 kbit per second of speech).

  SEMANTIC     the SENDER runs ASR + idiom resolution and transmits only the
               compact semantic message (idiom-resolved English text) as UTF-8
               bits, BPSK-modulated over the same AWGN channel (bit-flip
               probability BER = 0.5*erfc(sqrt(SNR))); the receiver decodes
               the bits and translates to Hindi.
               Bits/message = len(text) * 8 (~1000x fewer bits).

  LEARNED      (auto-enabled once models/semcodec exists — train_semcodec.py)
               OUR from-scratch semantic channel codec: a transformer encodes
               the semantic message into K float symbols per token, the symbols
               cross the same AWGN channel, and OUR decoder reconstructs the
               sentence before translation. Trained across random SNRs, so it
               degrades gracefully instead of collapsing with the BER curve.

Meaning preservation is scored with chrF between each scheme's received Hindi
and its own clean-channel Hindi (so both start at 1.0 and the curves show pure
channel robustness). If the manifest's gold `hindi_reference` column is filled,
pass --gold to score both schemes against it instead (absolute meaning accuracy).

Outputs:
  data/testset/semcom_snr.csv   SNR_dB, BER, chrF per scheme, bits per scheme
  data/testset/semcom_snr.png   meaning-vs-SNR curves + bits-per-message bars

Usage:
  python semcom_eval.py
  python semcom_eval.py --snr 20 10 5 0 -5 --max-clips 10
  python semcom_eval.py --gold          # score against gold hindi_reference
  python semcom_eval.py --channel rayleigh   # flat Rayleigh fading (perfect CSI)
  python semcom_eval.py --sem-noise 0 0.1 0.2 0.3  # sender-side semantic noise sweep
  python semcom_eval.py --selfcheck
"""

from __future__ import annotations

import argparse
import collections
import csv
import logging
import math
import tempfile
from pathlib import Path

import config
from noise_eval import add_awgn

logger = logging.getLogger("semcom_eval")


# --------------------------------------------------------------------------- #
# Channel model
# --------------------------------------------------------------------------- #
def bpsk_ber(snr_db: float) -> float:
    """Bit error rate of BPSK over AWGN at the given Eb/N0 (dB)."""
    return 0.5 * math.erfc(math.sqrt(10 ** (snr_db / 10)))


def rayleigh_ber(snr_db: float) -> float:
    """Average BPSK BER over flat Rayleigh fading (analytic, for the CSV)."""
    g = 10 ** (snr_db / 10)
    return 0.5 * (1 - math.sqrt(g / (1 + g)))


def fading_snr_db(snr_db: float, rng) -> float:
    """Quasi-static flat Rayleigh fading with perfect CSI: draw one |h|^2 ~ Exp(1)
    per message; the whole message then sees an AWGN channel at the faded SNR.
    """
    return snr_db + 10 * math.log10(max(rng.exponential(1.0), 1e-12))


def semantic_noise(text: str, p: float, rng, pool: list[str] | None = None) -> str:
    """Semantic noise: each word independently replaced (prob p) by a random
    word from `pool` (default: the sentence itself) — models sender-side
    meaning corruption (misheard/ambiguous words), as distinct from channel noise.
    """
    if p <= 0:
        return text
    words = text.split()
    pool = pool or words
    return " ".join(pool[int(rng.integers(len(pool)))] if rng.random() < p else w
                    for w in words)


def transmit_bits(data: bytes, ber: float, rng) -> bytes:
    """Flip each bit of `data` independently with probability `ber`."""
    import numpy as np

    if ber <= 0 or not data:
        return data
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    flips = rng.random(bits.size) < ber
    return np.packbits(bits ^ flips).tobytes()


def transmit_text(text: str, snr_db: float, rng, rep3: bool = False) -> str:
    """Send UTF-8 text over the BPSK/AWGN channel; undecodable bytes -> U+FFFD.

    rep3=True models a rate-1/3 repetition code with majority-vote decoding
    (3x the bits, residual bit-error prob 3p^2 - 2p^3) — a fair *coded*
    digital baseline instead of raw uncoded BPSK.
    """
    ber = bpsk_ber(snr_db)
    if rep3:
        ber = 3 * ber**2 - 2 * ber**3
    received = transmit_bits(text.encode("utf-8"), ber, rng)
    return received.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Meaning metric: chrF (character n-gram F-score, standard for Hindi MT)
# --------------------------------------------------------------------------- #
def chrf(ref: str, hyp: str, max_n: int = 6, beta: float = 2.0) -> float:
    ref, hyp = ref.replace(" ", ""), hyp.replace(" ", "")

    def ngrams(s: str, n: int):
        return collections.Counter(s[i : i + n] for i in range(len(s) - n + 1))

    precs, recs = [], []
    for n in range(1, max_n + 1):
        rn, hn = ngrams(ref, n), ngrams(hyp, n)
        if not rn or not hn:
            continue
        overlap = sum((rn & hn).values())
        precs.append(overlap / sum(hn.values()))
        recs.append(overlap / sum(rn.values()))
    if not precs:
        return 1.0 if ref == hyp else 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + r == 0:
        return 0.0
    return (1 + beta**2) * p * r / (beta**2 * p + r)


def _selfcheck() -> int:
    import numpy as np

    rng = np.random.default_rng(0)
    assert transmit_bits(b"hello", 0.0, rng) == b"hello"
    assert transmit_bits(b"hello", 1.0, rng) == bytes(b ^ 0xFF for b in b"hello")
    assert chrf("same text", "same text") == 1.0
    assert chrf("abcdef", "uvwxyz") == 0.0
    assert 0.0 < chrf("the cat sat", "the cat sit") < 1.0
    assert bpsk_ber(10) < 1e-5 < bpsk_ber(0) < bpsk_ber(-5) < 0.5
    p = bpsk_ber(0)
    assert 3 * p**2 - 2 * p**3 < p  # rep-3 majority vote always reduces BER
    assert rayleigh_ber(10) > bpsk_ber(10)  # fading is strictly worse than AWGN
    assert rayleigh_ber(-5) < 0.5
    fades = [fading_snr_db(0, rng) for _ in range(20000)]
    assert abs(sum(10 ** (f / 10) for f in fades) / len(fades) - 1.0) < 0.05  # E|h|^2 = 1
    assert semantic_noise("a b c", 0.0, rng) == "a b c"
    assert semantic_noise("a b c d e f g h", 1.0, rng, pool=["x"]) == "x x x x x x x x"
    print("semcom_eval selfcheck OK")
    return 0


def main() -> int:
    config.configure_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snr", type=float, nargs="+", default=[10, 5, 2, 0, -2, -5])
    ap.add_argument("--max-clips", type=int, default=25,
                    help="clips to evaluate (CPU translation is the bottleneck)")
    ap.add_argument("--gold", action="store_true",
                    help="score against manifest hindi_reference instead of clean-channel output")
    ap.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn",
                    help="rayleigh = quasi-static flat fading with perfect CSI")
    ap.add_argument("--sem-noise", type=float, nargs="+", metavar="P",
                    help="sweep sender-side semantic noise (word-corruption probs) "
                         "on a clean channel instead of the SNR sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return _selfcheck()

    import numpy as np
    import librosa
    import soundfile as sf

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)  # add_awgn uses the global generator

    rows = list(csv.DictReader(config.MANIFEST_PATH.open(encoding="utf-8")))
    rows = [r for r in rows if r.get("audio_path") and r.get("english_source", "").strip()]
    if args.gold:
        rows = [r for r in rows if r.get("hindi_reference", "").strip()]
        if not rows:
            raise SystemExit("--gold: no rows with hindi_reference filled in the manifest.")
    rows = rows[: args.max_clips]

    from semantitrans.pipeline import Pipeline

    pipe = Pipeline(idiom_mode="substitute")  # one model set; baseline uses its parts directly

    codec = None
    codec_dir = config.ROOT_DIR / "models" / "semcodec"
    if codec_dir.exists():
        from semantitrans.semcodec import SemCodec

        codec = SemCodec.load(codec_dir)
        logger.info("Learned semantic codec loaded from %s", codec_dir)

    def asr_translate(wav) -> str:
        """Receiver of the traditional scheme: ASR then plain translation."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            sf.write(tf.name, wav, 16000)
            tmp = tf.name
        try:
            return pipe.translator.translate(pipe.asr.transcribe(tmp))
        finally:
            Path(tmp).unlink(missing_ok=True)

    # ---- sender side (clean): audio, semantic message, clean references ----
    clips = []
    for r in rows:
        path = config.TESTSET_DIR / r["audio_path"]
        if not path.exists():
            continue
        x, _ = librosa.load(str(path), sr=16000, mono=True)
        sem_msg = pipe.resolver.resolve(pipe.asr.transcribe(str(path))).output_text
        if args.sem_noise and not args.gold:
            ref_trad = ""  # traditional scheme is not part of the semantic-noise sweep
        else:
            ref_trad = r["hindi_reference"] if args.gold else asr_translate(x)
        ref_sem = r["hindi_reference"] if args.gold else pipe.translator.translate(sem_msg)
        ref_codec = ""
        if codec:
            ref_codec = (r["hindi_reference"] if args.gold
                         else pipe.translator.translate(codec.reconstruct(sem_msg, None)))
        clips.append({"x": x, "sem_msg": sem_msg, "ref_trad": ref_trad,
                      "ref_sem": ref_sem, "ref_codec": ref_codec})
    logger.info("Loaded %d clips", len(clips))
    if not clips:
        raise SystemExit("No usable clips.")

    bits_trad = float(np.mean([len(c["x"]) * 16 for c in clips]))
    bits_sem = float(np.mean([len(c["sem_msg"].encode("utf-8")) * 8 for c in clips]))
    bits_codec = float(np.mean([codec.bits_per_message(c["sem_msg"]) for c in clips])) if codec else 0.0
    logger.info("Mean bits/message: traditional=%.0f semantic=%.0f codec=%.0f",
                bits_trad, bits_sem, bits_codec)

    from noise_eval import wer

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = ("_rayleigh" if args.channel == "rayleigh" else "") + ("_gold" if args.gold else "")
    chan_label = "Rayleigh fading" if args.channel == "rayleigh" else "AWGN"

    # fixed series style shared by every figure (identity never changes color)
    SCHEMES = [("traditional (waveform bits)", "o-"),
               ("semantic (text bits, uncoded)", "s-"),
               ("semantic (text bits, rep-3 coded)", "d-"),
               ("semantic (our learned codec)", "^-")]

    def line_fig(xs, series, xlabel, ylabel, title, out_png, ylim=None, invert=False):
        """series = list of (label, fmt, ys) with ys possibly all-None (skipped)."""
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, fmt, ys in series:
            if any(y is not None for y in ys):
                ax.plot(xs, ys, fmt, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if invert:
            ax.invert_xaxis()
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        logger.info("Wrote %s", out_png)

    # ---- semantic-noise sweep (clean channel, sender-side corruption) ----
    if args.sem_noise:
        pool = sorted({w for c in clips for w in c["sem_msg"].split()})
        results = []  # (p, chrf_sem, wer_sem, chrf_codec, wer_codec)
        for p in args.sem_noise:
            cs, ws, cc, wc = [], [], [], []
            for c in clips:
                noised = semantic_noise(c["sem_msg"], p, rng, pool)
                hi_sem = pipe.translator.translate(noised)
                cs.append(chrf(c["ref_sem"], hi_sem))
                ws.append(wer(c["ref_sem"], hi_sem))
                if codec:
                    hi_codec = pipe.translator.translate(codec.reconstruct(noised, None))
                    cc.append(chrf(c["ref_codec"], hi_codec))
                    wc.append(wer(c["ref_codec"], hi_codec))
            results.append((p, float(np.mean(cs)), float(np.mean(ws)),
                            float(np.mean(cc)) if cc else None,
                            float(np.mean(wc)) if wc else None))
            r = results[-1]
            logger.info("sem-noise p=%.2f  chrF sem=%.3f codec=%s  WER sem=%.3f codec=%s",
                        p, r[1], f"{r[3]:.3f}" if r[3] is not None else "n/a",
                        r[2], f"{r[4]:.3f}" if r[4] is not None else "n/a")

        gtag = "_gold" if args.gold else ""
        out_csv = config.TESTSET_DIR / f"semcom_semnoise{gtag}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["p_word_corrupt", "chrF_semantic", "WER_semantic",
                        "chrF_codec", "WER_codec", "n"])
            for p, cs_, ws_, cc_, wc_ in results:
                w.writerow([p, f"{cs_:.4f}", f"{ws_:.4f}",
                            f"{cc_:.4f}" if cc_ is not None else "",
                            f"{wc_:.4f}" if wc_ is not None else "", len(clips)])
        logger.info("Wrote %s", out_csv)

        xs = [r[0] for r in results]
        line_fig(xs, [("semantic (text)", "s-", [r[1] for r in results]),
                      ("semantic (our learned codec)", "^-", [r[3] for r in results])],
                 "word corruption probability p", "meaning preservation (chrF)",
                 f"Semantic noise at the sender, clean channel (n={len(clips)})",
                 config.TESTSET_DIR / f"semcom_semnoise{gtag}.png", ylim=(0, 1.05))
        line_fig(xs, [("semantic (text)", "s-", [r[2] for r in results]),
                      ("semantic (our learned codec)", "^-", [r[4] for r in results])],
                 "word corruption probability p", "WER of received Hindi",
                 f"Semantic noise at the sender, clean channel (n={len(clips)})",
                 config.TESTSET_DIR / f"semcom_semnoise{gtag}_wer.png")
        print(f"\nDone. Plots: semcom_semnoise{gtag}.png / _wer.png")
        return 0

    # ---- SNR sweep over the selected channel ----
    ber_fn = bpsk_ber if args.channel == "awgn" else rayleigh_ber
    results = []  # (snr, ber, [4 chrF], [4 WER])
    for snr in args.snr:
        ber = ber_fn(snr)
        cf = [[], [], [], []]
        we = [[], [], [], []]
        for c in clips:
            # one fade per message, shared by all schemes (paired comparison)
            eff = snr if args.channel == "awgn" else fading_snr_db(snr, rng)
            hyps = [asr_translate(add_awgn(c["x"], eff)),
                    pipe.translator.translate(transmit_text(c["sem_msg"], eff, rng)),
                    pipe.translator.translate(transmit_text(c["sem_msg"], eff, rng, rep3=True)),
                    pipe.translator.translate(codec.reconstruct(c["sem_msg"], eff)) if codec else None]
            refs = [c["ref_trad"], c["ref_sem"], c["ref_sem"], c["ref_codec"]]
            for i, (ref, hyp) in enumerate(zip(refs, hyps)):
                if hyp is not None:
                    cf[i].append(chrf(ref, hyp))
                    we[i].append(wer(ref, hyp))
        mc = [float(np.mean(s)) if s else None for s in cf]
        mw = [float(np.mean(s)) if s else None for s in we]
        results.append((snr, ber, mc, mw))
        logger.info("SNR=%g dB (%s)  BER=%.2e  chrF trad=%.3f sem=%.3f coded=%.3f codec=%s",
                    snr, chan_label, ber, mc[0], mc[1], mc[2],
                    f"{mc[3]:.3f}" if codec else "n/a")

    out_csv = config.TESTSET_DIR / f"semcom_snr{tag}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SNR_dB", "BER", "chrF_traditional", "chrF_semantic",
                    "chrF_semantic_rep3", "chrF_codec",
                    "WER_traditional", "WER_semantic", "WER_semantic_rep3", "WER_codec",
                    "bits_traditional", "bits_semantic", "bits_semantic_rep3",
                    "bits_codec", "n"])
        for snr, ber, mc, mw in results:
            w.writerow([snr, f"{ber:.3e}",
                        *[f"{v:.4f}" if v is not None else "" for v in mc],
                        *[f"{v:.4f}" if v is not None else "" for v in mw],
                        f"{bits_trad:.0f}", f"{bits_sem:.0f}", f"{3 * bits_sem:.0f}",
                        f"{bits_codec:.0f}" if codec else "", len(clips)])
    logger.info("Wrote %s", out_csv)

    pts = sorted(results)
    xs = [p[0] for p in pts]

    # chrF figure keeps the bits-per-message bar panel (the headline figure)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]})
    for i, (label, fmt) in enumerate(SCHEMES):
        ys = [p[2][i] for p in pts]
        if any(y is not None for y in ys):
            ax1.plot(xs, ys, fmt, label=label)
    ax1.set_xlabel("channel SNR (dB)")
    ax1.set_ylabel("meaning preservation (chrF)")
    ax1.set_title(f"Semantic vs traditional transmission, {chan_label} (n={len(clips)})")
    ax1.invert_xaxis()
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    labels, vals = ["traditional", "semantic", "sem rep-3"], [bits_trad, bits_sem, 3 * bits_sem]
    if codec:
        labels.append("codec")
        vals.append(bits_codec)
    ax2.bar(labels, vals)
    ax2.set_yscale("log")
    ax2.set_ylabel("bits per message (log)")
    ax2.set_title(f"{bits_trad / bits_sem:.0f}x fewer bits")
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = config.TESTSET_DIR / f"semcom_snr{tag}.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    logger.info("Wrote %s", out_png)

    line_fig(xs, [(label, fmt, [p[3][i] for p in pts])
                  for i, (label, fmt) in enumerate(SCHEMES)],
             "channel SNR (dB)", "WER of received Hindi (lower = better)",
             f"WER vs SNR, {chan_label} (n={len(clips)})",
             config.TESTSET_DIR / f"semcom_wer{tag}.png", invert=True)
    print(f"\nDone. Plots: {out_png} + semcom_wer{tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def transmit_bits(data: bytes, ber: float, rng) -> bytes:
    """Flip each bit of `data` independently with probability `ber`."""
    import numpy as np

    if ber <= 0 or not data:
        return data
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    flips = rng.random(bits.size) < ber
    return np.packbits(bits ^ flips).tobytes()


def transmit_text(text: str, snr_db: float, rng) -> str:
    """Send UTF-8 text over the BPSK/AWGN channel; undecodable bytes -> U+FFFD."""
    received = transmit_bits(text.encode("utf-8"), bpsk_ber(snr_db), rng)
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

    results = []  # (snr, ber, chrf_trad, chrf_sem, chrf_codec)
    for snr in args.snr:
        ber = bpsk_ber(snr)
        st, ss, sc = [], [], []
        for c in clips:
            hi_trad = asr_translate(add_awgn(c["x"], snr))
            hi_sem = pipe.translator.translate(transmit_text(c["sem_msg"], snr, rng))
            st.append(chrf(c["ref_trad"], hi_trad))
            ss.append(chrf(c["ref_sem"], hi_sem))
            if codec:
                hi_codec = pipe.translator.translate(codec.reconstruct(c["sem_msg"], snr))
                sc.append(chrf(c["ref_codec"], hi_codec))
        results.append((snr, ber, float(np.mean(st)), float(np.mean(ss)),
                        float(np.mean(sc)) if sc else None))
        logger.info("SNR=%g dB  BER=%.2e  chrF trad=%.3f sem=%.3f codec=%s",
                    snr, ber, results[-1][2], results[-1][3],
                    f"{results[-1][4]:.3f}" if codec else "n/a")

    out_csv = config.TESTSET_DIR / "semcom_snr.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SNR_dB", "BER", "chrF_traditional", "chrF_semantic", "chrF_codec",
                    "bits_traditional", "bits_semantic", "bits_codec", "n"])
        for snr, ber, ct, cs, cc in results:
            w.writerow([snr, f"{ber:.3e}", f"{ct:.4f}", f"{cs:.4f}",
                        f"{cc:.4f}" if cc is not None else "",
                        f"{bits_trad:.0f}", f"{bits_sem:.0f}",
                        f"{bits_codec:.0f}" if codec else "", len(clips)])
    logger.info("Wrote %s", out_csv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = sorted(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot([p[0] for p in pts], [p[2] for p in pts], "o-", label="traditional (waveform bits)")
    ax1.plot([p[0] for p in pts], [p[3] for p in pts], "s-", label="semantic (text bits, BPSK)")
    if codec:
        ax1.plot([p[0] for p in pts], [p[4] for p in pts], "^-", label="semantic (our learned codec)")
    ax1.set_xlabel("channel SNR (dB)")
    ax1.set_ylabel("meaning preservation (chrF)")
    ax1.set_title(f"Semantic vs traditional transmission (n={len(clips)})")
    ax1.invert_xaxis()
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    labels, vals = ["traditional", "semantic"], [bits_trad, bits_sem]
    if codec:
        labels.append("codec")
        vals.append(bits_codec)
    ax2.bar(labels, vals)
    ax2.set_yscale("log")
    ax2.set_ylabel("bits per message (log)")
    ax2.set_title(f"{bits_trad / bits_sem:.0f}x fewer bits")
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = config.TESTSET_DIR / "semcom_snr.png"
    fig.savefig(out_png, dpi=130)
    logger.info("Wrote %s", out_png)
    print(f"\nDone. Plot: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

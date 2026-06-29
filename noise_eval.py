"""Checkpoint #6 — noisy-channel robustness: WER vs SNR.

The test-set audio is clean (TTS). This simulates a noisy transmission channel
by adding additive white Gaussian noise (AWGN) at a range of SNRs, re-runs the
Whisper ASR (stage 1) on the degraded audio, and plots word error rate (WER)
against SNR. Pure evaluation — no training.

Reference text = the `english_source` column of the manifest (what the TTS said).

Outputs:
  data/testset/wer_snr.csv   SNR_dB, mean_WER, n
  data/testset/wer_snr.png   the plot

Usage:
  python noise_eval.py
  python noise_eval.py --snr 20 10 5 0 -5 --max-clips 20
  python noise_eval.py --selfcheck      # validate the WER function only
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import tempfile
from pathlib import Path

import config

logger = logging.getLogger("noise_eval")


def _words(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def wer(ref: str, hyp: str) -> float:
    """Word error rate via word-level Levenshtein distance / reference length."""
    r, h = _words(ref), _words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def add_awgn(x, snr_db: float):
    """Add white Gaussian noise to signal x at the given SNR (dB)."""
    import numpy as np

    ps = float((x ** 2).mean())
    if ps == 0:
        return x
    pn = ps / (10 ** (snr_db / 10))
    noise = np.sqrt(pn) * np.random.randn(len(x)).astype("float32")
    return (x + noise).astype("float32")


def _selfcheck() -> int:
    assert wer("a b c", "a b c") == 0.0
    assert abs(wer("a b c", "a b") - 1 / 3) < 1e-9          # one deletion
    assert abs(wer("a b c", "a x c") - 1 / 3) < 1e-9        # one substitution
    assert wer("", "anything") == 1.0
    print("noise_eval selfcheck OK")
    return 0


def main() -> int:
    config.configure_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snr", type=float, nargs="+", default=[30, 20, 10, 5, 0, -5],
                    help="SNR levels in dB (descending reads left→right)")
    ap.add_argument("--no-clean", action="store_true", help="skip the clean (no-noise) reference point")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return _selfcheck()

    import numpy as np
    import librosa
    import soundfile as sf

    np.random.seed(args.seed)

    if not config.MANIFEST_PATH.exists():
        raise SystemExit(f"{config.MANIFEST_PATH} not found — run build_testset.py first.")
    rows = list(csv.DictReader(config.MANIFEST_PATH.open(encoding="utf-8")))
    rows = [r for r in rows if r.get("audio_path") and r.get("english_source", "").strip()]
    if args.max_clips:
        rows = rows[: args.max_clips]
    if not rows:
        raise SystemExit("No usable rows (need audio_path + english_source).")

    config.log_device()
    from semantitrans.asr import WhisperASR

    asr = WhisperASR()

    # cache clean audio once
    clips = []
    for r in rows:
        ap_path = config.TESTSET_DIR / r["audio_path"]
        if not ap_path.exists():
            continue
        x, _ = librosa.load(str(ap_path), sr=16000, mono=True)
        clips.append((x, r["english_source"]))
    logger.info("Loaded %d clips", len(clips))

    conditions = list(args.snr) + ([] if args.no_clean else [None])  # None = clean
    results = []  # (label_snr_or_None, mean_wer, n)
    for snr in conditions:
        wers = []
        for x, ref in clips:
            y = x if snr is None else add_awgn(x, snr)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                sf.write(tf.name, y, 16000)
                tmp = tf.name
            try:
                hyp = asr.transcribe(tmp)
            finally:
                Path(tmp).unlink(missing_ok=True)
            wers.append(wer(ref, hyp))
        mean = sum(wers) / len(wers)
        results.append((snr, mean, len(wers)))
        logger.info("SNR=%s dB  mean WER=%.3f (n=%d)", "clean" if snr is None else snr, mean, len(wers))

    # write CSV
    out_csv = config.TESTSET_DIR / "wer_snr.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SNR_dB", "mean_WER", "n"])
        for snr, mean, n in results:
            w.writerow(["clean" if snr is None else snr, f"{mean:.4f}", n])
    logger.info("Wrote %s", out_csv)

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = sorted([(s, m) for s, m, _ in results if s is not None])
    xs = [s for s, _ in pts]
    ys = [m for _, m in pts]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, "o-", label="noisy channel")
    clean = next((m for s, m, _ in results if s is None), None)
    if clean is not None:
        ax.axhline(clean, ls="--", color="gray", label=f"clean ({clean:.2f})")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("WER")
    ax.set_title(f"ASR robustness: WER vs SNR (n={len(clips)})")
    ax.invert_xaxis()  # noisier (low SNR) on the right
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png = config.TESTSET_DIR / "wer_snr.png"
    fig.savefig(out_png, dpi=130)
    logger.info("Wrote %s", out_png)
    print(f"\nDone. Plot: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

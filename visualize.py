"""Checkpoint #5 — result visualizations.

Produces the figures we can make from artifacts that already exist (no gold
Hindi annotation needed):

  --detector   confusion matrix + P/R/F1 of the trained figurative/literal
               detector on a held-out MAGPIE split  -> models/figs/detector_cm.png
  --testset    composition of the idiom test set (usage split, KB coverage,
               most frequent idioms)                 -> models/figs/testset.png

The baseline-vs-idiom-aware metric chart lives in the eval (#7/#8) because it
needs the gold hindi_reference column filled.

Usage:
  python visualize.py                 # all available figures
  python visualize.py --detector --max-eval 1500
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random

import config

logger = logging.getLogger("visualize")

DETECTOR_DIR = config.ROOT_DIR / "models" / "idiom-detector"
MAGPIE_PATH = config.DATA_DIR / "magpie.jsonl"
FIGS_DIR = config.ROOT_DIR / "models" / "figs"
LABEL2ID = {"l": 0, "i": 1}  # must match train_idiom_detector.py


def _load_magpie_rows():
    rows = []
    for line in MAGPIE_PATH.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("label") not in LABEL2ID:
            continue
        ctx = " ".join(r.get("context", [])).strip()
        idiom = (r.get("idiom") or "").strip()
        if ctx and idiom:
            rows.append((idiom, ctx, LABEL2ID[r["label"]]))
    return rows


def viz_detector(max_eval: int | None):
    """Evaluate the trained detector on the SAME held-out split training used."""
    if not DETECTOR_DIR.exists():
        logger.warning("No detector at %s; skipping. Run train_idiom_detector.py.", DETECTOR_DIR)
        return
    if not MAGPIE_PATH.exists():
        logger.warning("No %s; skipping detector figure.", MAGPIE_PATH)
        return

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = _load_magpie_rows()
    random.Random(0).shuffle(rows)               # same seed as training
    n_val = max(1, int(0.1 * len(rows)))
    val = rows[:n_val]                            # held-out (not trained on)
    if max_eval:
        val = val[:max_eval]
    logger.info("Evaluating detector on %d held-out examples", len(val))

    tok = AutoTokenizer.from_pretrained(str(DETECTOR_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(DETECTOR_DIR)).eval()

    preds, labels = [], []
    bs = 32
    for i in range(0, len(val), bs):
        chunk = val[i : i + bs]
        enc = tok([c[0] for c in chunk], [c[1] for c in chunk],
                  truncation=True, max_length=128, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(logits.argmax(-1).tolist())
        labels.extend(c[2] for c in chunk)

    preds, labels = np.array(preds), np.array(labels)
    # confusion matrix: rows = true (literal, figurative), cols = pred
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    tp = cm[1, 1]; fp = cm[0, 1]; fn = cm[1, 0]; tn = cm[0, 0]
    acc = (tp + tn) / cm.sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    logger.info("acc=%.3f  precision=%.3f  recall=%.3f  f1(figurative)=%.3f", acc, prec, rec, f1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    names = ["literal", "figurative"]
    ax.set_xticks([0, 1], labels=names)
    ax.set_yticks([0, 1], labels=names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Idiom detector (held-out n={cm.sum()})\nacc={acc:.2f}  F1(fig)={f1:.2f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGS_DIR / "detector_cm.png"
    fig.savefig(out, dpi=130)
    logger.info("Wrote %s", out)


def viz_testset():
    if not config.MANIFEST_PATH.exists():
        logger.warning("No %s; skipping test-set figure. Run build_testset.py.", config.MANIFEST_PATH)
        return
    import collections

    rows = list(csv.DictReader(config.MANIFEST_PATH.open(encoding="utf-8")))
    usage = collections.Counter(r["usage"] for r in rows)
    in_kb = collections.Counter("in KB" if r["in_kb"] == "1" else "not in KB" for r in rows)
    top = collections.Counter(r["idiom"] for r in rows).most_common(10)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar(usage.keys(), usage.values(), color=["#4c72b0", "#dd8452"])
    axes[0].set_title("usage")
    axes[1].bar(in_kb.keys(), in_kb.values(), color=["#55a868", "#c44e52"])
    axes[1].set_title("KB coverage")
    if top:
        labels = [t[0] for t in top][::-1]
        vals = [t[1] for t in top][::-1]
        axes[2].barh(labels, vals, color="#8172b3")
        axes[2].set_title("most frequent idioms")
    fig.suptitle(f"Idiom test set composition (n={len(rows)})")
    fig.tight_layout()
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGS_DIR / "testset.png"
    fig.savefig(out, dpi=130)
    logger.info("Wrote %s", out)


def main() -> int:
    config.configure_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", action="store_true")
    ap.add_argument("--testset", action="store_true")
    ap.add_argument("--max-eval", type=int, default=2000, help="cap held-out examples for the detector figure")
    args = ap.parse_args()

    do_all = not (args.detector or args.testset)
    if args.detector or do_all:
        viz_detector(args.max_eval)
    if args.testset or do_all:
        viz_testset()
    print(f"\nFigures in {FIGS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

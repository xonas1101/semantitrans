"""Stage 2 — train OUR idiom-usage detector (figurative vs literal).

The rule-based resolver substitutes a literal gloss whenever an idiom STRING
matches. That's wrong when the same words are meant literally ("he kicked the
bucket across the yard"). This model decides figurative-vs-literal so the
resolver only injects glosses for genuinely idiomatic usages.

It is a fine-tune of distilbert-base-uncased on the MAGPIE corpus
(Haagsma et al. 2020, CC-BY-4.0): input = (idiom, sentence-context), label =
figurative ('i') vs literal ('l'). Output: models/idiom-detector/.

Usage:
  python train_idiom_detector.py                # all data, 3 epochs
  python train_idiom_detector.py --max-samples 2000 --epochs 1   # quick
"""

from __future__ import annotations

import argparse
import json
import logging
import random

import config

logger = logging.getLogger("train_idiom_detector")

BASE_MODEL = "distilbert-base-uncased"
OUT_DIR = config.ROOT_DIR / "models" / "idiom-detector"
MAGPIE_PATH = config.DATA_DIR / "magpie.jsonl"
# label ids: 1 = figurative (idiomatic), 0 = literal
LABEL2ID = {"l": 0, "i": 1}


def load_magpie(max_samples: int | None):
    if not MAGPIE_PATH.exists():
        raise SystemExit(
            f"{MAGPIE_PATH} not found. Download with:\n"
            "  curl -sL -o data/magpie.jsonl "
            "https://raw.githubusercontent.com/hslh/magpie-corpus/master/"
            "MAGPIE_filtered_split_typebased.jsonl"
        )
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
        if not ctx or not idiom:
            continue
        rows.append((idiom, ctx, LABEL2ID[r["label"]]))
    random.Random(0).shuffle(rows)
    if max_samples:
        rows = rows[:max_samples]
    pos = sum(1 for _, _, y in rows if y == 1)
    logger.info("Loaded %d examples (%d figurative / %d literal)", len(rows), pos, len(rows) - pos)
    return rows


def main() -> int:
    config.configure_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=128)
    args = ap.parse_args()

    device = config.log_device()

    import numpy as np
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    rows = load_magpie(args.max_samples)
    n_val = max(1, int(0.1 * len(rows)))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    def to_ds(rs):
        return Dataset.from_dict(
            {"idiom": [r[0] for r in rs], "ctx": [r[1] for r in rs], "label": [r[2] for r in rs]}
        )

    def preprocess(b):
        return tok(b["idiom"], b["ctx"], truncation=True, max_length=args.max_len)

    cols = ["idiom", "ctx"]
    train_ds = to_ds(train_rows).map(preprocess, batched=True, remove_columns=cols)
    val_ds = to_ds(val_rows).map(preprocess, batched=True, remove_columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label={0: "literal", 1: "figurative"}, label2id={"literal": 0, "figurative": 1}
    )

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean()
        # F1 on the figurative (positive) class
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {"accuracy": acc, "f1_figurative": f1}

    targs = TrainingArguments(
        output_dir=str(OUT_DIR.parent / "_detector_tmp"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        fp16=(device == "cuda"),
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics,
    )
    trainer.train()
    print("Final eval:", trainer.evaluate())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))
    tok.save_pretrained(str(OUT_DIR))
    logger.info("Saved idiom detector -> %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

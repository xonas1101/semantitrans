"""OPTIONAL lightweight LoRA adapter on opus-mt-en-hi for idiom-containing input.

This is the only optional *trained* component. It is a LoRA adapter (a few
hundred steps), designed to finish in tens of minutes on an RTX 4060 — NOT a
from-scratch or full fine-tune. Report it honestly as "a LoRA adapter on
Helsinki-NLP/opus-mt-en-hi".

Training data: an EN-HI parallel corpus (default cfilt/iitb-english-hindi,
CC-BY-NC for IIT-B — check terms for your use) filtered to pairs whose English
contains a known idiom (from the gloss KB), so the adapter specializes on the
idiomatic slice. With --literalize, the English side is first passed through the
stage-2 resolver so the adapter is trained on the SAME distribution the pipeline
feeds at inference (gloss-injected English).

Output: config.LORA_ADAPTER_DIR. Enable at inference with run.py --lora.

Usage:
  python train_lora.py --max-pairs 4000 --epochs 1
  python train_lora.py --max-pairs 4000 --literalize append
"""

from __future__ import annotations

import argparse
import logging

import config
from semantitrans.idiom_resolver import MODE_APPEND, MODE_OFF, MODE_SUBSTITUTE, IdiomResolver

logger = logging.getLogger("train_lora")


def build_pairs(dataset_name: str, max_pairs: int, literalize: str):
    from datasets import load_dataset

    resolver = IdiomResolver(mode=literalize) if literalize != MODE_OFF else None
    # idiom inventory keys for filtering
    inv = IdiomResolver(mode=MODE_OFF)._index  # noqa: SLF001 - reuse index
    keys = set(inv.keys())

    logger.info("Loading %s ...", dataset_name)
    ds = load_dataset(dataset_name, split="train", streaming=True)

    import re

    def has_idiom(text: str) -> bool:
        toks = tuple(m.group(0).lower() for m in re.finditer(r"\w+", text))
        for n in range(1, 7):
            for i in range(len(toks) - n + 1):
                if toks[i : i + n] in keys:
                    return True
        return False

    src, tgt = [], []
    for row in ds:
        pair = row.get("translation", row)
        en, hi = pair.get("en", ""), pair.get("hi", "")
        if not en or not hi or not has_idiom(en):
            continue
        if resolver is not None:
            en = resolver.literalize(en)
        src.append(en)
        tgt.append(hi)
        if len(src) >= max_pairs:
            break
    logger.info("Collected %d idiom-containing pairs", len(src))
    return src, tgt


def main() -> int:
    config.configure_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="cfilt/iitb-english-hindi")
    ap.add_argument("--max-pairs", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--literalize", choices=[MODE_OFF, MODE_SUBSTITUTE, MODE_APPEND], default=MODE_OFF)
    ap.add_argument("--max-len", type=int, default=128)
    args = ap.parse_args()

    device = config.log_device()
    if device != "cuda":
        logger.warning("No GPU detected — LoRA training on CPU is slow. Proceeding anyway.")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    from datasets import Dataset

    src, tgt = build_pairs(args.dataset, args.max_pairs, args.literalize)
    if not src:
        logger.error("No training pairs found; aborting.")
        return 1

    tok = AutoTokenizer.from_pretrained(config.TRANSLATOR_MODEL)
    base = AutoModelForSeq2SeqLM.from_pretrained(config.TRANSLATOR_MODEL)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    def preprocess(batch):
        enc = tok(batch["src"], max_length=args.max_len, truncation=True)
        lbl = tok(text_target=batch["tgt"], max_length=args.max_len, truncation=True)
        enc["labels"] = lbl["input_ids"]
        return enc

    ds = Dataset.from_dict({"src": src, "tgt": tgt}).map(
        preprocess, batched=True, remove_columns=["src", "tgt"]
    )

    collator = DataCollatorForSeq2Seq(tok, model=model)
    targs = Seq2SeqTrainingArguments(
        output_dir=str(config.LORA_ADAPTER_DIR.parent / "_train_tmp"),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=20,
        save_strategy="no",
        fp16=(device == "cuda"),
        report_to=[],
    )
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    config.LORA_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.LORA_ADAPTER_DIR))
    tok.save_pretrained(str(config.LORA_ADAPTER_DIR))
    logger.info("Saved LoRA adapter -> %s", config.LORA_ADAPTER_DIR)
    print("Enable it at inference with:  python run.py audio.wav --lora")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

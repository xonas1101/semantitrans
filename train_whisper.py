"""Stage 1 — fine-tune OUR Whisper checkpoint.

A demonstrative fine-tune of openai/whisper-tiny (English transcription) on a
small LibriSpeech slice, using the standard HF Whisper recipe. This is NOT a
from-scratch ASR and won't beat the pretrained base on general audio; it exists
so stage 1 has a component we trained, loaded automatically by asr.py from
config.WHISPER_FT_DIR.

Cited: OpenAI Whisper (Radford et al. 2022, MIT); LibriSpeech (Panayotov et al.
2015, CC-BY-4.0).

Usage:
  python train_whisper.py                       # tiny dummy set, quick
  python train_whisper.py --dataset librispeech_asr --config clean \\
      --split "train.100" --max-samples 2000 --epochs 1   # heavier, real
"""

from __future__ import annotations

import argparse
import io
import logging
from dataclasses import dataclass

import config

logger = logging.getLogger("train_whisper")

BASE_MODEL = "openai/whisper-tiny"
OUT_DIR = config.WHISPER_FT_DIR


@dataclass
class Collator:
    processor: object

    def __call__(self, features):
        import torch

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # drop the BOS the model adds itself
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def load_audio_text(dataset, cfg, split, max_samples):
    import librosa
    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset(dataset, cfg, split=split) if cfg else load_dataset(dataset, split=split)
    ds = ds.cast_column("audio", Audio(decode=False))  # avoid torchcodec; decode ourselves
    text_col = "text" if "text" in ds.column_names else "sentence"
    out = []
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        a = row["audio"]
        if a.get("bytes"):
            arr, sr = sf.read(io.BytesIO(a["bytes"]))
        else:
            arr, sr = sf.read(a["path"])
        if sr != 16000:
            arr = librosa.resample(arr.astype("float32"), orig_sr=sr, target_sr=16000)
        out.append((arr, row[text_col]))
    logger.info("Loaded %d audio/text pairs", len(out))
    return out


def main() -> int:
    config.configure_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hf-internal-testing/librispeech_asr_dummy")
    ap.add_argument("--config", default="clean")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    device = config.log_device()

    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    pairs = load_audio_text(args.dataset, args.config, args.split, args.max_samples)
    if not pairs:
        logger.error("No data loaded; aborting.")
        return 1

    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="en", task="transcribe")

    def make_example(arr, text):
        feat = processor.feature_extractor(arr, sampling_rate=16000).input_features[0]
        labels = processor.tokenizer(text.lower()).input_ids
        return {"input_features": feat, "labels": labels}

    from datasets import Dataset

    ds = Dataset.from_list([make_example(a, t) for a, t in pairs])

    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    targs = Seq2SeqTrainingArguments(
        output_dir=str(OUT_DIR.parent / "_whisper_tmp"),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="no",
        fp16=(device == "cuda"),
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(processor),
    )
    trainer.train()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))
    processor.save_pretrained(str(OUT_DIR))
    logger.info("Saved fine-tuned Whisper -> %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

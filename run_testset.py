"""Run the pipeline over the test-set manifest and write predictions.

For each manifest row it produces a Hindi prediction and writes it to a column
named `pred_<mode>` (e.g. pred_substitute, pred_off). Run it once per mode you
want to compare; evaluate.py then scores all present prediction columns.

By default it runs the FULL pipeline from audio. Use --from-text to skip ASR and
translate the manifest's english_source directly (faster; isolates stages 2-3).

Usage:
  python run_testset.py --mode off                  # baseline cascade
  python run_testset.py --mode substitute           # idiom-aware (span sub)
  python run_testset.py --mode append --from-text   # idiom-aware (gloss hint)
  python run_testset.py --mode substitute --lora    # with LoRA adapter
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

import config
from semantitrans.idiom_resolver import MODE_APPEND, MODE_OFF, MODE_SUBSTITUTE
from semantitrans.pipeline import Pipeline

logger = logging.getLogger("run_testset")


def main() -> int:
    config.configure_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=[MODE_SUBSTITUTE, MODE_APPEND, MODE_OFF], default=MODE_SUBSTITUTE)
    ap.add_argument("--from-text", action="store_true", help="skip ASR; translate english_source")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--no-lemmas", action="store_true")
    ap.add_argument("--manifest", default=str(config.MANIFEST_PATH))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, dtype=str).fillna("")
    if args.limit:
        df = df.head(args.limit)

    pipe = Pipeline(idiom_mode=args.mode, use_lora=args.lora, use_lemmas=not args.no_lemmas)

    col = f"pred_{args.mode}" + ("_lora" if args.lora else "")
    inter_col = f"inter_{args.mode}"
    preds, inters = [], []
    for _, row in df.iterrows():
        if args.from_text:
            res = pipe.translate_text(row["english_source"])
        else:
            audio = config.TESTSET_DIR / row["audio_path"]
            if not row["audio_path"] or not audio.exists():
                logger.warning("missing audio for %s; falling back to text", row["id"])
                res = pipe.translate_text(row["english_source"])
            else:
                res = pipe.run(str(audio))
        preds.append(res.hindi)
        inters.append(res.intermediate)
        logger.info("%s: %s", row["id"], res.hindi)

    df[inter_col] = inters
    df[col] = preds
    df.to_csv(args.manifest, index=False)
    logger.info("Wrote column '%s' -> %s", col, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

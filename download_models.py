"""Pre-download the pretrained models so the first real run isn't slow.

  python download_models.py                 # whisper + translator
  python download_models.py --whisper medium
"""

from __future__ import annotations

import argparse
import logging

import config

logger = logging.getLogger("download_models")


def main() -> int:
    config.configure_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--whisper", default=config.WHISPER_MODEL)
    ap.add_argument("--translator", default=config.TRANSLATOR_MODEL)
    args = ap.parse_args()

    config.log_device()

    logger.info("Downloading Whisper '%s' ...", args.whisper)
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    WhisperProcessor.from_pretrained(args.whisper)
    WhisperForConditionalGeneration.from_pretrained(args.whisper)

    logger.info("Downloading translator '%s' ...", args.translator)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    AutoTokenizer.from_pretrained(args.translator)
    AutoModelForSeq2SeqLM.from_pretrained(args.translator)

    logger.info("Done. Models cached for offline use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

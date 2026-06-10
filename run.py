"""CLI entry point: English audio in, Hindi text out.

Examples:
  python run.py path/to/audio.wav
  python run.py audio.wav --mode append           # gloss-hint injection
  python run.py audio.wav --mode off              # baseline cascade (no idioms)
  python run.py audio.wav --lora                  # use LoRA adapter if present
  python run.py audio.wav --json                  # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import config
from semantitrans.idiom_resolver import MODE_APPEND, MODE_OFF, MODE_SUBSTITUTE
from semantitrans.pipeline import Pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="path to an English audio file (wav/mp3/m4a/...)")
    ap.add_argument(
        "--mode",
        choices=[MODE_SUBSTITUTE, MODE_APPEND, MODE_OFF],
        default=MODE_SUBSTITUTE,
        help="idiom injection mode (default: substitute; 'off' = baseline)",
    )
    ap.add_argument("--lora", action="store_true", help="apply LoRA adapter if present")
    ap.add_argument("--no-lemmas", action="store_true", help="disable spaCy lemma matching")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    config.configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    if not Path(args.audio).exists():
        print(f"error: audio file not found: {args.audio}", file=sys.stderr)
        return 2

    pipe = Pipeline(
        idiom_mode=args.mode,
        use_lora=args.lora,
        use_lemmas=not args.no_lemmas,
    )
    result = pipe.run(args.audio)

    if args.json:
        print(json.dumps(
            {
                "english": result.english,
                "intermediate": result.intermediate,
                "hindi": result.hindi,
                "idiom_mode": result.idiom_mode,
                "detections": [
                    {"idiom": d.idiom, "surface": d.surface, "literal": d.literal}
                    for d in result.detections
                ],
                "timings": result.timings,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    print("\n=== Stage 1: English transcript ===")
    print(result.english)
    print(f"\n=== Stage 2: literalized English (mode={result.idiom_mode}) ===")
    print(result.intermediate)
    if result.detections:
        print("  idioms resolved:")
        for d in result.detections:
            print(f"    - {d.surface!r} -> {d.literal!r}")
    print("\n=== Stage 3: Hindi ===")
    print(result.hindi)
    if result.timings:
        print(
            "\n[timing] asr={asr:.2f}s resolve={resolve:.3f}s "
            "translate={translate:.2f}s total={total:.2f}s".format(**result.timings)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

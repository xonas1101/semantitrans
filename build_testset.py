"""Build the idiom test set: MAGPIE figurative sentences -> TTS audio -> manifest.

Mirrors the protocol used by idiom speech-translation work (Zaitova et al. ACL
2025) which synthesizes test audio with TTS. We sample FIGURATIVE-usage MAGPIE
sentences, synthesize English speech for each, and write a manifest with empty
`hindi_reference` and `literal_trap_hi` columns for the user (a Hindi speaker)
to fill as the gold standard.

Columns:
  id, idiom, in_kb, usage, english_source, audio_path,
  hindi_reference     (EMPTY -> fill with the correct figurative Hindi),
  literal_trap_hi     (EMPTY -> optional: Hindi word(s) that would indicate a
                       LITERAL mistranslation of the idiom; used by evaluate.py
                       for the literal-translation-error metric)

Usage:
  python build_testset.py --n 60
  python build_testset.py --n 60 --prefer-kb        # bias toward KB-covered idioms
  python build_testset.py --n 60 --no-tts           # manifest only, skip audio
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random

import config

logger = logging.getLogger("build_testset")


def kb_idiom_keys() -> set[tuple[str, ...]]:
    import re

    if not config.GLOSS_KB_PATH.exists():
        return set()
    with open(config.GLOSS_KB_PATH, encoding="utf-8") as f:
        kb = json.load(f)
    return {
        tuple(m.group(0).lower() for m in re.finditer(r"\w+", k))
        for k, v in kb.get("glosses", {}).items()
        if v.get("literal")
    }


def norm_key(idiom: str) -> tuple[str, ...]:
    import re

    return tuple(m.group(0).lower() for m in re.finditer(r"\w+", idiom))


def synthesize(text: str, out_path) -> bool:
    try:
        from gtts import gTTS

        gTTS(text=text, lang="en").save(str(out_path))
        return True
    except Exception as e:  # network / dependency issues
        logger.warning("TTS failed for %r: %s", text[:40], e)
        return False


def main() -> int:
    config.configure_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60, help="number of test sentences")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--prefer-kb", action="store_true", help="bias toward KB-covered idioms")
    ap.add_argument("--no-tts", action="store_true", help="write manifest without audio")
    ap.add_argument("--max-words", type=int, default=30, help="skip overly long sentences")
    args = ap.parse_args()

    from datasets import load_dataset

    logger.info("Loading %s ...", config.MAGPIE_DATASET)
    ds = load_dataset(config.MAGPIE_DATASET, split="train")

    kb_keys = kb_idiom_keys()
    rows = []
    for row in ds:
        if (row.get("usage") or "").lower() != "figurative":
            continue
        sent = (row.get("sentence") or "").strip()
        idiom = (row.get("idiom") or "").strip()
        if not sent or not idiom:
            continue
        if len(sent.split()) > args.max_words:
            continue
        rows.append({"idiom": idiom, "sentence": sent, "in_kb": norm_key(idiom) in kb_keys})

    logger.info("Figurative candidates: %d (%d in KB)", len(rows), sum(r["in_kb"] for r in rows))

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.prefer_kb:
        rows.sort(key=lambda r: not r["in_kb"])  # KB-covered first, stable
    selected = rows[: args.n]

    manifest_rows = []
    for i, r in enumerate(selected):
        rid = f"t{i:04d}"
        audio_rel = f"audio/{rid}.mp3"
        audio_abs = config.TESTSET_DIR / audio_rel
        if not args.no_tts:
            ok = synthesize(r["sentence"], audio_abs)
            if not ok:
                audio_rel = ""
        manifest_rows.append(
            {
                "id": rid,
                "idiom": r["idiom"],
                "in_kb": int(r["in_kb"]),
                "usage": "figurative",
                "english_source": r["sentence"],
                "audio_path": audio_rel,
                "hindi_reference": "",
                "literal_trap_hi": "",
            }
        )

    fields = ["id", "idiom", "in_kb", "usage", "english_source", "audio_path", "hindi_reference", "literal_trap_hi"]
    with open(config.MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest_rows)

    logger.info("Wrote %d rows -> %s", len(manifest_rows), config.MANIFEST_PATH)
    print(f"\nNext: fill the 'hindi_reference' column in {config.MANIFEST_PATH}")
    print("(optional) fill 'literal_trap_hi' to enable the literal-translation-error metric.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

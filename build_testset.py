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
    ap.add_argument("--n", type=int, default=60, help="number of FIGURATIVE test sentences")
    ap.add_argument("--n-literal", type=int, default=0,
                    help="number of LITERAL-usage sentences (for the gated-detector analysis)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--prefer-kb", action="store_true", help="bias toward KB-covered idioms")
    ap.add_argument("--no-tts", action="store_true", help="write manifest without audio")
    ap.add_argument("--max-words", type=int, default=30, help="skip overly long sentences")
    args = ap.parse_args()

    magpie_path = config.DATA_DIR / "magpie.jsonl"
    if not magpie_path.exists():
        raise SystemExit(
            f"{magpie_path} not found. Download with:\n"
            "  curl -sL -o data/magpie.jsonl "
            "https://raw.githubusercontent.com/hslh/magpie-corpus/master/"
            "MAGPIE_filtered_split_typebased.jsonl"
        )
    logger.info("Loading %s ...", magpie_path)

    def best_sentence(idiom: str, context: list[str]) -> str:
        """Pick the context sentence with the most idiom-token overlap."""
        want = set(norm_key(idiom))
        best, score = "", -1
        for s in context:
            toks = set(norm_key(s))
            ov = len(want & toks)
            if ov > score:
                best, score = s.strip(), ov
        return best

    kb_keys = kb_idiom_keys()
    # MAGPIE label -> usage; 'i' = figurative/idiomatic, 'l' = literal
    label2usage = {"i": "figurative", "l": "literal"}
    by_usage: dict[str, list] = {"figurative": [], "literal": []}
    for line in magpie_path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = label2usage.get(r.get("label"))
        if usage is None:
            continue
        idiom = (r.get("idiom") or "").strip()
        sent = best_sentence(idiom, r.get("context", []))
        if not sent or not idiom:
            continue
        if len(sent.split()) > args.max_words:
            continue
        by_usage[usage].append(
            {"idiom": idiom, "sentence": sent, "in_kb": norm_key(idiom) in kb_keys, "usage": usage}
        )

    for u, lst in by_usage.items():
        logger.info("%s candidates: %d (%d in KB)", u, len(lst), sum(x["in_kb"] for x in lst))

    rng = random.Random(args.seed)
    selected = []
    for usage, want in (("figurative", args.n), ("literal", args.n_literal)):
        pool = by_usage[usage]
        rng.shuffle(pool)
        if args.prefer_kb:
            pool.sort(key=lambda r: not r["in_kb"])  # KB-covered first, stable
        take = pool[:want]
        if len(take) < want:
            logger.warning("Only %d %s items available (< requested %d)", len(take), usage, want)
        selected.extend(take)

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
                "usage": r["usage"],
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

"""Expand / audit the idiom gloss knowledge base against the MAGPIE inventory.

The resolver (semantitrans/idiom_resolver.py) consumes
data/glosses/idiom_glosses.json. This script:

  * reports how much of the MAGPIE idiom inventory the current KB covers,
  * optionally merges glosses from an external JSON source (you must verify and
    record that source's license), and
  * optionally emits a template of UNGLOSSED MAGPIE idioms for you to fill in.

It never invents glosses. MAGPIE provides idioms, not literal paraphrases, so
the literal side must come from the seed set, a licensed dictionary, or manual
authoring. Keep provenance honest via the per-entry "source" tag.

Usage:
  python build_glosses.py --audit
  python build_glosses.py --merge external_glosses.json --merge-source ode --merge-license "CC-BY-SA"
  python build_glosses.py --emit-template unglossed_template.json --limit 500
"""

from __future__ import annotations

import argparse
import json
import logging

import config

logger = logging.getLogger("build_glosses")


def load_kb() -> dict:
    if config.GLOSS_KB_PATH.exists():
        with open(config.GLOSS_KB_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"metadata": {"sources": []}, "glosses": {}}


def save_kb(kb: dict) -> None:
    with open(config.GLOSS_KB_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    logger.info("Wrote KB -> %s (%d glosses)", config.GLOSS_KB_PATH, len(kb["glosses"]))


def magpie_idioms() -> set[str]:
    from datasets import load_dataset

    logger.info("Loading %s (idiom inventory)...", config.MAGPIE_DATASET)
    ds = load_dataset(config.MAGPIE_DATASET, split="train")
    idioms = {row["idiom"].strip().lower() for row in ds if row.get("idiom")}
    logger.info("MAGPIE unique idioms: %d", len(idioms))
    return idioms


def normalized_key(idiom: str) -> tuple[str, ...]:
    # mirror the resolver's surface normalization for fair coverage stats
    import re

    return tuple(m.group(0).lower() for m in re.finditer(r"\w+", idiom))


def audit(kb: dict) -> None:
    inv = magpie_idioms()
    glossed_keys = {normalized_key(k) for k, v in kb["glosses"].items() if v.get("literal")}
    inv_keys = {normalized_key(i) for i in inv}
    covered = inv_keys & glossed_keys
    print(f"KB glossed idioms : {len(glossed_keys)}")
    print(f"MAGPIE inventory  : {len(inv_keys)}")
    print(f"Covered by KB     : {len(covered)} ({100*len(covered)/max(1,len(inv_keys)):.1f}%)")
    print(f"KB-only (non-MAGPIE): {len(glossed_keys - inv_keys)}")


def emit_template(kb: dict, path: str, limit: int | None) -> None:
    inv = sorted(magpie_idioms())
    have = {normalized_key(k) for k in kb["glosses"]}
    missing = [i for i in inv if normalized_key(i) not in have]
    if limit:
        missing = missing[:limit]
    template = {i: {"literal": "", "source": "TODO"} for i in missing}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %d unglossed idioms to %s for manual/licensed filling", len(template), path)


def merge(kb: dict, source_path: str, source_tag: str, license_str: str) -> None:
    with open(source_path, encoding="utf-8") as f:
        incoming = json.load(f)
    added = 0
    for idiom, entry in incoming.items():
        literal = entry.get("literal") if isinstance(entry, dict) else entry
        if not literal:
            continue
        kb["glosses"][idiom.strip().lower()] = {"literal": literal, "source": source_tag}
        added += 1
    srcs = kb.setdefault("metadata", {}).setdefault("sources", [])
    srcs.append({"tag": source_tag, "name": source_path, "license": license_str})
    logger.info("Merged %d glosses from %s (license: %s)", added, source_path, license_str)
    save_kb(kb)


def main() -> None:
    config.configure_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", action="store_true", help="report MAGPIE coverage")
    ap.add_argument("--emit-template", metavar="PATH", help="write unglossed MAGPIE idioms")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--merge", metavar="JSON", help="merge external glosses (verify license!)")
    ap.add_argument("--merge-source", default="external")
    ap.add_argument("--merge-license", default="UNKNOWN - VERIFY")
    args = ap.parse_args()

    kb = load_kb()
    if args.merge:
        merge(kb, args.merge, args.merge_source, args.merge_license)
    if args.emit_template:
        emit_template(kb, args.emit_template, args.limit)
    if args.audit or not (args.merge or args.emit_template):
        audit(kb)


if __name__ == "__main__":
    main()

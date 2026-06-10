"""Score predictions against the Hindi gold references.

Metrics:
  * chrF / chrF++  (sacrebleu) — surface overlap, language-agnostic, the primary
    metric here since it needs no extra models.
  * Literal-translation-error rate (LTE) — inspired by Baziotis et al. (EACL
    2023): the fraction of idiom items whose Hindi output contains a LITERAL
    (word-for-word) rendering of the idiom rather than its figurative meaning.
    Computed only for rows where the optional `literal_trap_hi` column is filled
    (Hindi trap word(s), '|'-separated). Lower is better.
  * COMET (optional, --comet) — neural quality, ~2GB download + GPU recommended.

It scores every `pred_*` column present in the manifest so you can compare modes
(e.g. pred_off vs pred_substitute vs pred_append) side by side.

Usage:
  python evaluate.py
  python evaluate.py --comet            # also run COMET (heavy)
  python evaluate.py --manifest data/testset/manifest.csv
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

import config

logger = logging.getLogger("evaluate")


def chrf_scores(hyps: list[str], refs: list[str]) -> dict:
    from sacrebleu.metrics import CHRF

    metric = CHRF(word_order=2)  # chrF++
    corpus = metric.corpus_score(hyps, [refs]).score
    per_item = [metric.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]
    return {"chrF++_corpus": corpus, "chrF++_mean": sum(per_item) / max(1, len(per_item))}


def lte_rate(hyps: list[str], traps: list[str]) -> float | None:
    pairs = [(h, t) for h, t in zip(hyps, traps) if t.strip()]
    if not pairs:
        return None
    hits = 0
    for h, t in pairs:
        trap_words = [w.strip() for w in t.split("|") if w.strip()]
        if any(w in h for w in trap_words):
            hits += 1
    return 100.0 * hits / len(pairs)


def comet_scores(srcs: list[str], hyps: list[str], refs: list[str]) -> float:
    from comet import download_model, load_from_checkpoint

    logger.info("Loading COMET (this downloads ~2GB on first run)...")
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    gpus = 1 if config.get_device() == "cuda" else 0
    out = model.predict(data, batch_size=8, gpus=gpus)
    return float(out["system_score"])


def main() -> int:
    config.configure_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(config.MANIFEST_PATH))
    ap.add_argument("--comet", action="store_true", help="also compute COMET (heavy)")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, dtype=str).fillna("")

    if "hindi_reference" not in df.columns or not df["hindi_reference"].str.strip().any():
        print("No filled 'hindi_reference' values found. Fill the gold column first.")
        return 1

    scored = df[df["hindi_reference"].str.strip() != ""].copy()
    pred_cols = [c for c in scored.columns if c.startswith("pred_")]
    if not pred_cols:
        print("No prediction columns (pred_*). Run run_testset.py first.")
        return 1

    refs = scored["hindi_reference"].tolist()
    srcs = scored["english_source"].tolist()
    traps = scored["literal_trap_hi"].tolist() if "literal_trap_hi" in scored.columns else [""] * len(scored)

    print(f"\nScoring {len(scored)} items with filled references.\n")
    rows = []
    for col in sorted(pred_cols):
        hyps = scored[col].tolist()
        res = {"system": col}
        res.update(chrf_scores(hyps, refs))
        lte = lte_rate(hyps, traps)
        res["LTE%"] = round(lte, 1) if lte is not None else None
        if args.comet:
            res["COMET"] = round(comet_scores(srcs, hyps, refs), 4)
        rows.append(res)

    out = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(out.to_string(index=False))

    # in-KB vs out-of-KB breakdown if available
    if "in_kb" in scored.columns:
        in_kb = scored[scored["in_kb"] == "1"]
        if 0 < len(in_kb) < len(scored):
            print("\nchrF++ on KB-covered idioms only:")
            for col in sorted(pred_cols):
                s = chrf_scores(in_kb[col].tolist(), in_kb["hindi_reference"].tolist())
                print(f"  {col}: {s['chrF++_corpus']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train OUR semantic channel codec from scratch (no pretrained weights).

Trains the transformer encoder/decoder in semantitrans/semcodec.py to compress
an English sentence into K channel symbols, survive AWGN at a random SNR drawn
per batch, and reconstruct the sentence at the receiver. Training text: the
context sentences of the local MAGPIE dump (data/magpie.jsonl).

Outputs models/semcodec/{model.pt, vocab.json}; semcom_eval.py picks the
directory up automatically as the "semantic (learned codec)" scheme.

Usage:
  python train_semcodec.py                       # full run (CPU-friendly)
  python train_semcodec.py --max-sents 2000 --epochs 2   # quick pass
  python train_semcodec.py --selfcheck           # overfit 32 sentences
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random

import config
from semantitrans.semcodec import MAX_LEN, PAD, SemCodec, SemCodecModel, Vocab, tokenize

logger = logging.getLogger("train_semcodec")

SEMCODEC_DIR = config.ROOT_DIR / "models" / "semcodec"


def load_sentences(max_sents: int) -> list[str]:
    seen, out = set(), []
    with (config.DATA_DIR / "magpie.jsonl").open(encoding="utf-8") as f:
        for line in f:
            for s in json.loads(line).get("context", []):
                s = s.strip()
                if s in seen or not (4 <= len(tokenize(s)) <= MAX_LEN - 2):
                    continue
                seen.add(s)
                out.append(s)
                if len(out) >= max_sents:
                    return out
    return out


def build_vocab(sents: list[str], size: int) -> Vocab:
    counts = collections.Counter(w for s in sents for w in tokenize(s))
    return Vocab([w for w, _ in counts.most_common(size)])


def batches(ids_list, batch_size, shuffle=True):
    import torch

    order = list(range(len(ids_list)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [ids_list[j] for j in order[i : i + batch_size]]
        width = max(len(c) for c in chunk)
        yield torch.tensor([c + [PAD] * (width - len(c)) for c in chunk])


def train(sents: list[str], vocab: Vocab, epochs: int, batch_size: int, lr: float,
          snr_range=(-4.0, 12.0), log_every: int = 50, model=None):
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    ids_list = [vocab.encode(s) for s in sents]
    n_val = max(1, len(ids_list) // 20)
    val, tr = ids_list[:n_val], ids_list[n_val:]

    if model is None:
        model = SemCodecModel(vocab_size=len(vocab.itos))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    def run_epoch(data, training: bool) -> float:
        model.train(training)
        total, nb = 0.0, 0
        with torch.set_grad_enabled(training):
            for step, batch in enumerate(batches(data, batch_size, shuffle=training)):
                snr = random.uniform(*snr_range)
                logits = model(batch, snr)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
                if training:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    if step % log_every == 0:
                        logger.info("  step %d loss %.3f (snr %.1f dB)", step, loss.item(), snr)
                total += loss.item()
                nb += 1
        return total / max(nb, 1)

    for ep in range(1, epochs + 1):
        tr_loss = run_epoch(tr, True)
        val_loss = run_epoch(val, False)
        logger.info("epoch %d/%d  train %.3f  val %.3f", ep, epochs, tr_loss, val_loss)
        codec = SemCodec(model, vocab)
        for s in sents[:2]:
            logger.info("  @5dB  %r -> %r", s, codec.reconstruct(s, 5.0))
        model.train()
    return model


def _selfcheck() -> int:
    config.configure_logging()
    sents = load_sentences(32)
    vocab = build_vocab(sents, 4000)
    model = train(sents, vocab, epochs=300, batch_size=8, lr=1e-3, snr_range=(20.0, 20.0))
    codec = SemCodec(model, vocab)
    probe = sents[-1]  # sents[0] is held out as validation inside train()
    rec = codec.reconstruct(probe, 20.0)
    ref = " ".join(tokenize(probe))
    logger.info("ref: %r\nrec: %r", ref, rec)
    overlap = len(set(rec.split()) & set(ref.split())) / max(len(set(ref.split())), 1)
    assert overlap > 0.5, f"codec failed to overfit 32 sentences (overlap {overlap:.2f})"
    print("train_semcodec selfcheck OK")
    return 0


def main() -> int:
    config.configure_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-sents", type=int, default=20000)
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", action="store_true",
                    help="continue training from the saved models/semcodec checkpoint")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return _selfcheck()

    import torch

    sents = load_sentences(args.max_sents)
    logger.info("Training on %d sentences", len(sents))
    model = None
    if args.resume:
        # must reuse the saved vocab or the embedding indices won't line up
        vocab = Vocab.load(SEMCODEC_DIR / "vocab.json")
        codec = SemCodec.load(SEMCODEC_DIR)
        model = codec.model.train()
        logger.info("Resumed from %s (vocab %d)", SEMCODEC_DIR, len(vocab.itos))
    else:
        vocab = build_vocab(sents, args.vocab_size)
    model = train(sents, vocab, args.epochs, args.batch_size, args.lr, model=model)

    SEMCODEC_DIR.mkdir(parents=True, exist_ok=True)
    vocab.save(SEMCODEC_DIR / "vocab.json")
    torch.save(model.state_dict(), SEMCODEC_DIR / "model.pt")
    logger.info("Saved codec -> %s", SEMCODEC_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

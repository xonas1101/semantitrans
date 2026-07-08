"""OUR semantic channel codec, trained from scratch (see train_semcodec.py).

DeepSC-style joint semantic/channel coding (Xie et al. 2021 — cite the idea):
a transformer encoder compresses a sentence into K power-normalized channel
symbols, the symbols cross an AWGN channel at a given SNR, and a transformer
decoder reconstructs the sentence at the receiver. All weights are trained by
us from random init on MAGPIE context sentences; no pretrained backbone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn as nn

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
MAX_LEN = 48


def tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z']+|[0-9]+|[^\w\s]", s.lower())


class Vocab:
    def __init__(self, words: list[str]):
        self.itos = SPECIALS + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, s: str) -> list[int]:
        return [BOS] + [self.stoi.get(w, UNK) for w in tokenize(s)][: MAX_LEN - 2] + [EOS]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids if i > UNK)

    def save(self, path: Path):
        path.write_text(json.dumps(self.itos[len(SPECIALS):]), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(path.read_text(encoding="utf-8")))


class SemCodecModel(nn.Module):
    """K channel symbols PER TOKEN (as in DeepSC), not per sentence."""

    def __init__(self, vocab_size: int, d_model: int = 128, k_symbols: int = 16,
                 nhead: int = 4, layers: int = 2, ff: int = 256):
        super().__init__()
        self.k_symbols = k_symbols
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_LEN, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, ff, batch_first=True), layers)
        self.to_channel = nn.Linear(d_model, k_symbols)
        self.from_channel = nn.Linear(k_symbols, d_model)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, ff, batch_first=True), layers)
        self.out = nn.Linear(d_model, vocab_size)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.size(1), device=ids.device)
        return self.emb(ids) + self.pos(pos)

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """Token ids -> (batch, seq, K) unit-average-power channel symbols."""
        pad_mask = ids == PAD
        h = self.encoder(self._embed(ids), src_key_padding_mask=pad_mask)
        keep = (~pad_mask).unsqueeze(-1).float()
        z = self.to_channel(h) * keep  # pad positions are not transmitted
        n = (keep.sum(dim=(1, 2)) * z.size(-1)).clamp_min(1.0)  # non-pad symbol count
        power = (z.pow(2).sum(dim=(1, 2)) / n).sqrt().clamp_min(1e-8)
        return z / power.view(-1, 1, 1)

    @staticmethod
    def channel(z: torch.Tensor, snr_db: float | None) -> torch.Tensor:
        """AWGN at the given SNR; None = clean channel."""
        if snr_db is None:
            return z
        pn = 1.0 / (10 ** (snr_db / 10))  # signal power is normalized to 1
        return z + (pn ** 0.5) * torch.randn_like(z)

    def decode_logits(self, tgt_ids: torch.Tensor, z: torch.Tensor,
                      mem_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        memory = self.from_channel(z)
        t = tgt_ids.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(t, device=tgt_ids.device)
        h = self.decoder(self._embed(tgt_ids), memory,
                         tgt_mask=causal, tgt_key_padding_mask=tgt_ids == PAD,
                         memory_key_padding_mask=mem_pad_mask)
        return self.out(h)

    def forward(self, ids: torch.Tensor, snr_db: float | None) -> torch.Tensor:
        z = self.channel(self.encode(ids), snr_db)
        # teacher forcing; predict ids[:, 1:]
        return self.decode_logits(ids[:, :-1], z, mem_pad_mask=ids == PAD)


class SemCodec:
    """Inference wrapper: text -> symbols -> AWGN -> text."""

    def __init__(self, model: SemCodecModel, vocab: Vocab):
        self.model = model.eval()
        self.vocab = vocab

    @torch.no_grad()
    def reconstruct(self, text: str, snr_db: float | None) -> str:
        ids = torch.tensor([self.vocab.encode(text)])
        z = self.model.channel(self.model.encode(ids), snr_db)
        out = [BOS]
        for _ in range(MAX_LEN - 1):
            logits = self.model.decode_logits(torch.tensor([out]), z)
            nxt = int(logits[0, -1].argmax())
            if nxt == EOS:
                break
            out.append(nxt)
        return self.vocab.decode(out)

    def bits_per_message(self, text: str) -> int:
        return len(self.vocab.encode(text)) * self.model.k_symbols * 32  # float32 symbols

    @classmethod
    def load(cls, dirpath: Path) -> "SemCodec":
        vocab = Vocab.load(dirpath / "vocab.json")
        state = torch.load(dirpath / "model.pt", map_location="cpu", weights_only=True)
        model = SemCodecModel(vocab_size=len(vocab.itos),
                              k_symbols=state["to_channel.weight"].size(0))
        model.load_state_dict(state)
        return cls(model, vocab)

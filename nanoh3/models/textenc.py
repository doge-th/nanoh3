"""Text conditioning.

``NanoTextEncoder`` is a small byte-level causal transformer trained jointly with
the denoiser. Byte-level is deliberate: it needs no tokenizer download, so the
whole pipeline runs offline, and it keeps the text tower under 10 M parameters
which matters when the total budget is 8 GB.

For the full-quality model the text tower is replaced by a frozen CLIP text
encoder (see ``docs/DESIGN.md``); the denoiser only ever sees a
``(B, L, text_dim)`` tensor, so the swap is a load-time change, not a code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class TextEncoderConfig:
    vocab_size: int = 256  # byte level
    width: int = 256
    layers: int = 4
    heads: int = 4
    max_len: int = 77
    dropout: float = 0.0


def encode_prompts(prompts: list[str], max_len: int = 77) -> Tuple[Tensor, Tensor]:
    """Byte-encode prompts -> (token_ids, attention_mask), right-padded."""
    ids = torch.zeros(len(prompts), max_len, dtype=torch.long)
    mask = torch.zeros(len(prompts), max_len, dtype=torch.long)
    for i, text in enumerate(prompts):
        raw = text.encode("utf-8")[: max_len - 1]
        ids[i, : len(raw)] = torch.tensor(list(raw), dtype=torch.long)
        mask[i, : len(raw) + 1] = 1  # keep one trailing token so no row is fully masked
    return ids, mask


class Block(nn.Module):
    def __init__(self, cfg: TextEncoderConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.width)
        self.attn = nn.MultiheadAttention(cfg.width, cfg.heads, batch_first=True, dropout=cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.width)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.width, cfg.width * 4), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.width * 4, cfg.width)
        )

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        h = self.ln1(x)
        causal = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), diagonal=1)
        a, _ = self.attn(h, h, h, attn_mask=causal, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class NanoTextEncoder(nn.Module):
    def __init__(self, cfg: TextEncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TextEncoderConfig()
        c = self.cfg
        self.embed = nn.Embedding(c.vocab_size, c.width)
        self.pos = nn.Parameter(torch.zeros(1, c.max_len, c.width))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.ln_f = nn.LayerNorm(c.width)
        self.out_dim = c.width

    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        b, l = tokens.shape
        x = self.embed(tokens.clamp(max=self.cfg.vocab_size - 1)) + self.pos[:, :l]
        for block in self.blocks:
            x = block(x, torch.empty(0))
        x = self.ln_f(x)
        if mask is not None:
            x = x * mask[..., None].to(x.dtype)
        return x, (mask if mask is not None else torch.ones(b, l, dtype=torch.long, device=tokens.device))

    def embed_prompts(self, prompts: list[str], device: torch.device) -> Tuple[Tensor, Tensor]:
        tokens, mask = encode_prompts(prompts, self.cfg.max_len)
        return self.forward(tokens.to(device), mask.to(device))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["NanoTextEncoder", "TextEncoderConfig", "encode_prompts"]

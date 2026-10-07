"""3D rotary position embeddings for video transformers.

The head dimension is split across the three axes (time, height, width) in
proportion to the latent grid, and each slice gets an independent rotary
frequency band. Tables are cached per grid shape because a training run sees the
same handful of shapes thousands of times.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn

Grid = Tuple[int, int, int]  # (frames, height, width) in latent tokens


def split_head_dim(head_dim: int, grid: Grid) -> Tuple[int, int, int]:
    """Split head_dim into three even slices, biased towards the longest axis."""
    longest_axis = max(range(3), key=lambda i: grid[i])
    total = sum(grid)
    dims: List[int] = [2 * max(1, round(head_dim * (g / total) / 2)) for g in grid]
    dims[longest_axis] += head_dim - sum(dims)
    if any(d < 2 for d in dims):
        # Degenerate tiny grids: fall back to an even split so the model still runs.
        base = max(2, 2 * (head_dim // 6))
        dims = [base, base, head_dim - 2 * base]
    return dims[0], dims[1], dims[2]


def _axis_sin_cos(channels: int, rows: int, base: float, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    """cos/sin of shape (rows, channels // 2) for one axis owning `channels` dims."""
    pos = torch.arange(rows, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, channels, 2, device=device, dtype=torch.float32) / channels))
    angles = pos[:, None] * inv_freq[None, :]
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


class Rotary3D(nn.Module):
    """Cached 3D rotary embedding over a (T, H, W) latent grid."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        self.head_dim = head_dim
        self.base = base
        self._cache: Dict[Tuple[Grid, str, torch.dtype], Tuple[Tensor, Tensor]] = {}

    def _tables(self, grid: Grid, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor, List[int]]:
        key = (grid, str(device), dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        dims = split_head_dim(self.head_dim, grid)
        halves = [d // 2 for d in dims]
        # All three axis tables share `max(grid)` rows so they can be concatenated
        # into one (max(grid), head_dim // 2) table; each axis is indexed only at
        # its own coordinates, which are always < grid[i] <= max(grid).
        rows = max(grid)
        per_axis = [_axis_sin_cos(dims[i], rows, self.base, device, dtype) for i in range(3)]
        cos = torch.cat([c for c, _ in per_axis], dim=-1)
        sin = torch.cat([s for _, s in per_axis], dim=-1)
        self._cache[key] = (cos, sin, halves)
        return cos, sin, halves

    def forward(self, grid: Grid, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        """cos/sin of shape (T*H*W, head_dim // 2), ordered row-major over (t, h, w)."""
        cos, sin, halves = self._tables(grid, device, dtype)
        t, h, w = grid
        idx = torch.arange(t * h * w, device=device)
        coords = torch.stack(
            [
                torch.div(idx, h * w, rounding_mode="floor"),
                torch.div(idx, w, rounding_mode="floor") % h,
                idx % w,
            ],
            dim=-1,
        )  # (N, 3)
        out_c: List[Tensor] = []
        out_s: List[Tensor] = []
        offset = 0
        for axis, half in enumerate(halves):
            out_c.append(cos[:, offset : offset + half][coords[:, axis]])
            out_s.append(sin[:, offset : offset + half][coords[:, axis]])
            offset += half
        return torch.cat(out_c, dim=-1), torch.cat(out_s, dim=-1)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to ``x`` of shape (B, heads, N, head_dim).

    Interleaved-pair convention: channels ``2i`` and ``2i+1`` are rotated against
    each other, so ``cos``/``sin`` carry ``head_dim // 2`` values per token.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


__all__ = ["Rotary3D", "apply_rotary", "split_head_dim", "Grid"]

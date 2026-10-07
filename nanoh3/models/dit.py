"""NanoH3 denoiser: a rectified-flow video DiT with 3D RoPE, adaLN-Zero and text cross-attention.

Design notes relevant to the 8 GB target
-----------------------------------------
* Every residual branch is zero-initialised (adaLN-Zero), so depth can be pruned
  at inference time with a bounded quality loss -- ``set_active_depth`` is the hook
  the engine uses when the memory budget does not allow a full forward pass.
* Attention goes through ``scaled_dot_product_attention`` so FlashAttention is
  selected automatically on CUDA.
* Weights are plain ``nn.Linear`` so the quantiser in :mod:`nanoh3.engine.quant`
  can swap them for int8/fp8 storage without touching model code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rope import Grid, Rotary3D, apply_rotary


@dataclass
class DiTConfig:
    """Shape of the denoiser. Parameter count follows directly from these fields."""

    latent_channels: int = 16
    patch_size: int = 2
    hidden_size: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    text_dim: int = 384
    max_text_len: int = 77
    rope_base: float = 10000.0
    qk_norm: bool = True
    # Depth-pruning schedule: how many blocks survive per "speed setting".
    # Index 0 is the full model; each entry removes one block.
    layer_skip: Sequence[int] = field(default_factory=tuple)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


def timestep_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Sinusoidal timestep features, (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t.dtype)


class SelfAttention(nn.Module):
    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(cfg.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(cfg.head_dim) if cfg.qk_norm else nn.Identity()

    def forward(self, x: Tensor, rotary: Rotary3D, grid: Grid) -> Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = rotary(grid, x.device, q.dtype)
        q, k = apply_rotary(self.q_norm(q), cos, sin), apply_rotary(self.k_norm(k), cos, sin)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, self.heads * self.head_dim)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Text cross-attention. Skipped entirely when no text conditioning is present."""

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.q = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.kv = nn.Linear(cfg.text_dim, 2 * cfg.hidden_size, bias=False)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x: Tensor, context: Optional[Tensor], mask: Optional[Tensor]) -> Tensor:
        if context is None:
            return torch.zeros_like(x)
        b, n, _ = x.shape
        l = context.shape[1]
        q = self.q(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).view(b, l, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn_mask = None
        if mask is not None:
            # (B, L) -> (B, 1, 1, L) boolean mask for SDPA.
            attn_mask = mask[:, None, None, :].to(torch.bool)
        out = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, n, self.heads * self.head_dim)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_size * cfg.mlp_ratio)
        self.fc1 = nn.Linear(cfg.hidden_size, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, cfg.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class DiTBlock(nn.Module):
    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross = CrossAttention(cfg)
        self.norm3 = nn.LayerNorm(cfg.hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(cfg)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cfg.hidden_size, 6 * cfg.hidden_size, bias=False))
        nn.init.zeros_(self.modulation[1].weight)  # adaLN-Zero: identity at init

    def modulation_params(self, cond: Tensor) -> Tuple[Tensor, ...]:
        m = self.modulation(cond)[:, None, :].chunk(6, dim=-1)
        return m  # shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        context: Optional[Tensor],
        mask: Optional[Tensor],
        rotary: Rotary3D,
        grid: Grid,
    ) -> Tensor:
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = self.modulation_params(cond)
        h = self.norm1(x) * (1 + scale_sa) + shift_sa
        x = x + gate_sa * self.attn(h, rotary, grid)
        x = x + self.cross(self.norm2(x), context, mask)
        h = self.norm3(x) * (1 + scale_mlp) + shift_mlp
        return x + gate_mlp * self.mlp(h)


class NanoDiT(nn.Module):
    """Velocity predictor for rectified flow over 3D latents.

    Input/output shape: ``(B, latent_channels, T, H, W)`` in latent units.
    """

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch = cfg.patch_size
        self.x_embedder = nn.Conv3d(cfg.latent_channels, cfg.hidden_size, (1, self.patch, self.patch), (1, self.patch, self.patch))
        self.t_embedder = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size), nn.SiLU(), nn.Linear(cfg.hidden_size, cfg.hidden_size)
        )
        self.text_embedder = nn.Sequential(
            nn.Linear(cfg.text_dim, cfg.hidden_size), nn.SiLU(), nn.Linear(cfg.hidden_size, cfg.hidden_size)
        )
        self.blocks = nn.ModuleList([DiTBlock(cfg) for _ in range(cfg.depth)])
        self.rotary = Rotary3D(cfg.head_dim, cfg.rope_base)
        self.norm_out = nn.LayerNorm(cfg.hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation_out = nn.Sequential(nn.SiLU(), nn.Linear(cfg.hidden_size, 2 * cfg.hidden_size, bias=False))
        self.proj_out = nn.Linear(cfg.hidden_size, self.patch * self.patch * cfg.latent_channels, bias=False)
        nn.init.zeros_(self.modulation_out[1].weight)
        nn.init.zeros_(self.proj_out.weight)
        self._active_blocks: Optional[List[int]] = None

    # ---- depth pruning -------------------------------------------------
    @property
    def active_blocks(self) -> List[int]:
        return list(range(len(self.blocks))) if self._active_blocks is None else list(self._active_blocks)

    def set_active_depth(self, depth: int) -> None:
        """Keep the first ``depth`` blocks; used by the 8 GB speed profile."""
        depth = max(1, min(depth, len(self.blocks)))
        self._active_blocks = list(range(depth))

    def reset_active_depth(self) -> None:
        self._active_blocks = None

    # ---- forward -------------------------------------------------------
    def forward(
        self,
        z: Tensor,
        timestep: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        b, c, t, h, w = z.shape
        p = self.patch
        if h % p or w % p:
            raise ValueError(f"latent spatial dims {(h, w)} must be divisible by patch size {p}")
        x = self.x_embedder(z)  # (B, D, T, H/p, W/p)
        grid: Grid = (t, h // p, w // p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        cond = self.t_embedder(timestep_embedding(timestep, self.cfg.hidden_size))
        if context is not None:
            if mask is None:
                mask = torch.ones(context.shape[:2], device=context.device, dtype=torch.long)
            pooled = (context * mask[..., None].to(context.dtype)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            cond = cond + self.text_embedder(pooled.to(cond.dtype))

        for i in self.active_blocks:
            x = self.blocks[i](x, cond, context, mask, self.rotary, grid)

        shift, scale = self.modulation_out(cond)[:, None, :].chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + scale) + shift
        x = self.proj_out(x)  # (B, N, p*p*C): velocity in latent space, same shape as `z`
        x = x.view(b, t, grid[1], grid[2], p, p, c).permute(0, 6, 1, 4, 2, 5, 3)
        return x.reshape(b, c, t, grid[1] * p, grid[2] * p)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["DiTConfig", "NanoDiT", "timestep_embedding"]

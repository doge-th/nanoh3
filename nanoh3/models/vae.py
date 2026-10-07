"""NanoVAE: a small, strictly causal 3D autoencoder for video latents.

Compression matches the MiniMax-H3 family where it matters for the token budget:
4x temporal, 8x spatial by default, 16 latent channels, and a ``patch_size=2``
token grid on top -- i.e. an effective 16x spatial reduction per token, against
H3's 32x.  H3 reaches 32x by spending 16x in the VAE *and* 2x in the patchifier
with 24 latent channels; ``configs/h3_matched.yaml`` reproduces that combination.

"Strictly causal" is the property that lets us stream and, more importantly here,
lets the whole encode/decode path run with a bounded memory window:

    encode(clip_with_extra_future_frames)[:, :k] == encode(clip)[:, :k]

i.e. the latent at index k is a function of input frames <= 4k+3 only.  That is
what ``tests/test_model.py::test_vae_is_causal`` asserts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def causal_pad(kernel: Sequence[int], dilation: Sequence[int], stride: Sequence[int]) -> List[int]:
    """Left-only padding for a causal conv.

    Padding ``(k - 1) * d`` samples on the left and nothing on the right gives
    both properties we need at once:

    * causality -- output ``n`` sees inputs ``[n*s - d(k-1), n*s]``, i.e. never a
      future frame;
    * length -- ``floor((L - 1) / s) + 1 == ceil(L / s)``, so a downsample by ``s``
      produces exactly the expected number of latent steps.

    Returns a ``F.pad`` argument list in ``(W_left, W_right, H_left, H_right,
    T_left, T_right)`` order.
    """
    pad: List[int] = []
    for k, d in zip(reversed(kernel), reversed(dilation)):
        pad.extend([(k - 1) * d, 0])
    return pad


def plan_strides(spatial_compression: int, temporal_compression: int, num_downs: int) -> List[Tuple[int, int, int]]:
    """Split ``num_downs`` downsampling convs between the time and spatial axes.

    Temporal reduction goes first (early layers), so a decoder can upsample
    frames before paying for full-resolution pixels.
    """
    t_downs = int(round(math.log2(temporal_compression)))
    s_downs = int(round(math.log2(spatial_compression)))
    if 2**t_downs != temporal_compression or 2**s_downs != spatial_compression:
        raise ValueError("compression factors must be powers of two")
    if t_downs + s_downs > num_downs:
        raise ValueError(
            f"{t_downs} temporal + {s_downs} spatial downsamples need {t_downs + s_downs} stages, got {num_downs}"
        )
    strides: List[Tuple[int, int, int]] = []
    for i in range(num_downs):
        stride_t = 2 if i < t_downs else 1
        stride_s = 2 if i >= t_downs else 1
        strides.append((stride_t, stride_s, stride_s))
    return strides


class CausalConv3d(nn.Module):
    """Conv3d with left-only padding.

    ``torch.nn.Conv3d`` only accepts symmetric padding, so the asymmetric causal
    padding is applied explicitly with ``F.pad`` before the convolution.
    ``replicate`` mode keeps the first frame from being darkened by a zero border.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: Sequence[int] = (3, 3, 3),
        stride: Sequence[int] = (1, 1, 1),
        dilation: Sequence[int] = (1, 1, 1),
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.pad = causal_pad(kernel, dilation, stride)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            tuple(kernel),
            stride=tuple(stride),
            dilation=tuple(dilation),
            padding=0,  # the asymmetric causal pad is applied explicitly above
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, self.pad, mode="replicate") if any(self.pad) else x)


class FrameGroupNorm(nn.Module):
    """GroupNorm applied independently to every frame.

    Plain ``nn.GroupNorm`` normalises over ``(C/G, T, H, W)``, so its output at
    frame ``t`` depends on the mean and variance of the *entire* clip -- a silent
    future-frame leak that survives any amount of careful causal padding. Folding
    time into the batch dimension here makes the statistics per-frame, which is
    both causal and the right behaviour for streaming.
    """

    def __init__(self, num_channels: int, num_groups: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        y = F.group_norm(y, self.groups, self.weight, self.bias, self.eps)
        return y.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)


class ResBlock3D(nn.Module):
    def __init__(self, channels: int, kernel: Tuple[int, int, int] = (3, 3, 3)) -> None:
        super().__init__()
        self.norm1 = FrameGroupNorm(channels, min(8, channels))
        self.conv1 = CausalConv3d(channels, channels, kernel)
        self.norm2 = FrameGroupNorm(channels, min(8, channels))
        self.conv2 = CausalConv3d(channels, channels, kernel)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


@dataclass
class VAEConfig:
    in_channels: int = 3
    base_channels: int = 32
    latent_channels: int = 16
    channel_multipliers: Sequence[int] = (1, 2, 3, 4, 4, 4)  # len = num_downs + 1
    num_res_blocks: int = 1
    spatial_compression: int = 8
    temporal_compression: int = 4
    groups: int = 8

    @property
    def num_downs(self) -> int:
        return len(self.channel_multipliers) - 1

    @property
    def channels(self) -> List[int]:
        return [self.base_channels * m for m in self.channel_multipliers]

    @property
    def strides(self) -> List[Tuple[int, int, int]]:
        return plan_strides(self.spatial_compression, self.temporal_compression, self.num_downs)


class NanoVAE(nn.Module):
    def __init__(self, cfg: VAEConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or VAEConfig()
        self.cfg = cfg
        chans = cfg.channels
        strides = cfg.strides
        groups = min(cfg.groups, chans[0])

        enc: List[nn.Module] = [CausalConv3d(cfg.in_channels, chans[0])]
        for i, stride in enumerate(strides):
            for _ in range(cfg.num_res_blocks):
                enc.append(ResBlock3D(chans[i]))
            enc.append(CausalConv3d(chans[i], chans[i + 1], (3, 3, 3), stride=stride))
        self.encoder = nn.Sequential(*enc)
        # Bottleneck at the widest width, then project to 2x latent channels
        # (first half mean, second half log-variance).
        self.mid = nn.Sequential(ResBlock3D(chans[-1]), ResBlock3D(chans[-1]))
        self.enc_to_latent = CausalConv3d(chans[-1], 2 * cfg.latent_channels)

        self.latent_to_dec = CausalConv3d(cfg.latent_channels, chans[-1])
        self.dec_res = nn.Sequential(*[ResBlock3D(chans[-1]) for _ in range(cfg.num_res_blocks)])
        # One up-stage per encoder down-stage, applied in reverse order.
        self.up_stages = nn.ModuleList(
            [
                nn.Sequential(ResBlock3D(chans[i + 1]), CausalConv3d(chans[i + 1], chans[i]))
                for i in range(cfg.num_downs - 1, -1, -1)
            ]
        )
        self.dec_out = nn.Sequential(
            FrameGroupNorm(chans[0], groups),
            nn.SiLU(),
            CausalConv3d(chans[0], cfg.in_channels),
        )
        self._decode_order = strides  # ups run in reverse of the encoder

    # ---- shapes ----
    def latent_shape(self, frames: int, height: int, width: int) -> Tuple[int, int, int]:
        c = self.cfg
        return (
            -(-frames // c.temporal_compression),
            -(-height // c.spatial_compression),
            -(-width // c.spatial_compression),
        )

    # ---- codec ----
    def encode(self, video: Tensor) -> Tensor:
        """(B, C, F, H, W) pixels in [-1, 1] -> (B, 2*latent, ceil(F/4), ceil(H/8), ceil(W/8))."""
        return self.enc_to_latent(self.mid(self.encoder(video)))

    def decode(self, latent: Tensor, output_size: Tuple[int, int, int] | None = None) -> Tensor:
        """latent -> pixels, cropped to ``output_size`` = (F, H, W) when given."""
        f, h, w = output_size or tuple(
            s * c for s, c in zip(latent.shape[2:], (self.cfg.temporal_compression, self.cfg.spatial_compression, self.cfg.spatial_compression))
        )
        x = self.dec_res(self.latent_to_dec(latent))
        for stage, (stride_t, stride_s, _) in zip(self.up_stages, reversed(self._decode_order)):
            x = F.interpolate(x, scale_factor=(stride_t, stride_s, stride_s), mode="nearest")
            x = stage(x)
        x = self.dec_out(x)
        return x[:, :, :f, :h, :w]

    def forward(self, video: Tensor, sample: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns (reconstruction, sampled latent, posterior std)."""
        posterior = self.encode(video)
        std = self.posterior_std(posterior)
        latent = self.sample_latent(posterior) if sample else posterior.chunk(2, dim=1)[0]
        recon = self.decode(latent, (video.shape[2], video.shape[3], video.shape[4]))
        return recon, latent, std

    def posterior_std(self, posterior: Tensor, min_std: float = 1e-3) -> Tensor:
        logvar = self._logvar(posterior)
        return torch.exp(0.5 * logvar).clamp(min=min_std, max=1e3)

    def sample_latent(self, posterior: Tensor) -> Tensor:
        mean, logvar = posterior.chunk(2, dim=1)
        return mean + torch.exp(0.5 * self._logvar(posterior)) * torch.randn_like(mean)

    @staticmethod
    def _logvar(posterior: Tensor) -> Tensor:
        """Clamp log-variance to a sane band.

        The upper clamp matters more than it looks. Without a KL term early in
        training the log-variance head runs away (observed: +242), and
        ``exp(0.5 * 242)`` turns every sampled latent into noise several orders of
        magnitude larger than the mean, which silently poisons the denoiser's
        targets. Capping at ``+6`` bounds the posterior std at e^3.
        """
        return posterior.chunk(2, dim=1)[1].clamp(-30.0, 6.0)

    def kl_divergence(self, posterior: Tensor) -> Tensor:
        """KL(q(z|x) || N(0,1)) per element, in nats."""
        mean = posterior.chunk(2, dim=1)[0]
        logvar = self._logvar(posterior)
        return 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar).mean()

    def recon_loss(self, video: Tensor, kl_weight: float = 1e-4) -> Tuple[Tensor, dict]:
        """L1 + multi-scale gradient + a small KL term.

        The KL weight is deliberately tiny: it only has to stop the variance head
        from running away, which is a stability constraint rather than a desire
        for a maximally compressed latent. Reconstruction uses the posterior mean
        so the two loss terms are comparable.
        """
        posterior = self.encode(video)
        latent = posterior.chunk(2, dim=1)[0]
        rec = self.decode(latent, (video.shape[2], video.shape[3], video.shape[4]))
        l1 = F.l1_loss(rec, video)
        grad = video.new_zeros(())
        for m in (2, 4):
            a = F.avg_pool3d(video, m, m)
            b = F.avg_pool3d(rec, m, m)
            grad = grad + F.l1_loss(a, b) / m
        kl = self.kl_divergence(posterior)
        total = l1 + 0.25 * grad + kl_weight * kl
        return total, {
            "l1": float(l1.detach()),
            "grad": float(grad.detach()),
            "kl": float(kl.detach()),
            "rec": float(total.detach()),
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["NanoVAE", "VAEConfig", "CausalConv3d", "causal_pad", "plan_strides", "ResBlock3D"]

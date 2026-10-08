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


def centered_pad(kernel: Sequence[int], dilation: Sequence[int]) -> List[int]:
    """Symmetric padding, preserving length -- for the *decoder*.

    The decoder must not be causal. With nearest-neighbour temporal upsampling,
    a causal decoder convolution looks back over three neighbours that are mostly
    the *previous* latent: for output frame 4 the three taps come from
    ``[latent0, latent0, latentent1]``. Measured on this model, that skews the
    gradient so hard that ``|d out[4] / d latent0| = 1670`` against
    ``|d out[4] / d latent1| = 31`` -- and "copy the first latent and repeat"
    becomes the cheapest way to minimise any reconstruction loss, which is
    exactly the static output this design was fighting.

    Centring the taps on the current frame (``[latent0, latent1, latent1]``)
    restores the balance. MiniMax-H3's own VAE makes the same choice:
    its config states ``causal_encoder: true, causal_decoder: false``.
    """
    pad: List[int] = []
    for k, d in zip(reversed(kernel), reversed(dilation)):
        reach = (k - 1) * d
        pad.extend([reach // 2, reach - reach // 2])
    return pad


def plan_strides(
    spatial_compression: int, temporal_compression: int, num_downs: int, order: str = "time_first"
) -> List[Tuple[int, int, int]]:
    """Split ``num_downs`` downsampling convs between the time and spatial axes.

    ``order`` controls where the temporal reduction happens:

    * ``"time_first"`` -- all temporal downsampling in the earliest stages, so the
      decoder can expand frames before paying for full-resolution pixels. Cheap,
      but it averages motion *at full spatial resolution*, where a moving object
      occupies a handful of pixels out of 64x64, and the measured result was that
      the encoder wrote motion into the latent at 0.07 % of its magnitude.
    * ``"interleaved"`` -- alternate temporal and spatial downsampling. Motion is
      reduced once the features are already spatial, and each latent frame ends up
      with a comparable share of the motion signal.
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
    if order == "time_first":
        for i in range(num_downs):
            stride_t = 2 if i < t_downs else 1
            stride_s = 2 if i >= t_downs else 1
            strides.append((stride_t, stride_s, stride_s))
        return strides
    if order != "interleaved":
        raise ValueError(f"order must be 'time_first' or 'interleaved', got {order!r}")
    # Spread the two reductions across the available stages, temporal first.
    for i in range(num_downs):
        stride_t = 2 if i < t_downs else 1
        stride_s = 2 if i < t_downs + s_downs else 1
        strides.append((stride_t, stride_s, stride_s))
    return strides


class CausalConv3d(nn.Module):
    """Conv3d whose padding is either causal (encoder) or centred (decoder).

    ``torch.nn.Conv3d`` only accepts symmetric padding, so the asymmetric causal
    padding is applied explicitly with ``F.pad`` before the convolution.
    ``replicate`` mode keeps the first frame from being darkened by a zero border.

    Set ``causal=False`` for decoder layers: see :func:`centered_pad` for the
    measurement that makes this necessary rather than merely tidy.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: Sequence[int] = (3, 3, 3),
        stride: Sequence[int] = (1, 1, 1),
        dilation: Sequence[int] = (1, 1, 1),
        bias: bool = True,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.pad = causal_pad(kernel, dilation, stride) if causal else centered_pad(kernel, dilation)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            tuple(kernel),
            stride=tuple(stride),
            dilation=tuple(dilation),
            padding=0,  # the pad is applied explicitly above
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, self.pad, mode="replicate") if any(self.pad) else x)


class ClipOrFrameNorm(nn.Module):
    """GroupNorm whose statistics are either per-frame or per-clip.

    Plain ``nn.GroupNorm`` normalises over ``(C/G, T, H, W)``, so its output at
    frame ``t`` depends on the mean and variance of the *entire* clip -- a silent
    future-frame leak that survives any amount of careful causal padding. Folding
    time into the batch dimension makes the statistics per-frame, which fixes
    causality and is what streaming needs.

    **The cost is real and it is not obvious.** Per-frame normalisation also
    discards each frame's own mean and variance. If the encoder represents motion
    as a per-frame offset or contrast change -- which is the cheapest way to
    encode it -- that information is annihilated, and a decoder fed latents that
    differ only in offset emits *bit-identical* frames. Measured on a trained
    model: flipping the latent between +5 and -5 changed the output by exactly
    0.0.

    So this is a switch, not a free win:

    * ``"frame"`` (default) -- strictly causal, but per-frame contrast cannot pass.
    * ``"clip"`` -- per-frame contrast survives, at the cost of the norm being
      non-causal, which breaks the streaming guarantee and the causality test.

    See ``docs/DESIGN.md`` for the measurement that led to making this explicit.
    """

    def __init__(self, num_channels: int, num_groups: int, eps: float = 1e-6, mode: str = "frame") -> None:
        super().__init__()
        if mode not in ("frame", "clip"):
            raise ValueError(f"norm mode must be 'frame' or 'clip', got {mode!r}")
        self.groups = num_groups
        self.eps = eps
        self.mode = mode
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "clip":
            y = F.group_norm(x, self.groups, self.weight, self.bias, self.eps)
            return y
        b, c, t, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        y = F.group_norm(y, self.groups, self.weight, self.bias, self.eps)
        return y.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)


class ResBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel: Tuple[int, int, int] = (3, 3, 3),
        norm_mode: str = "frame",
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = ClipOrFrameNorm(channels, min(8, channels), mode=norm_mode)
        self.conv1 = CausalConv3d(channels, channels, kernel, causal=causal)
        self.norm2 = ClipOrFrameNorm(channels, min(8, channels), mode=norm_mode)
        self.conv2 = CausalConv3d(channels, channels, kernel, causal=causal)

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
    norm_mode: str = "frame"  # "frame" = strictly causal, "clip" = keeps per-frame contrast
    downsample_order: str = "time_first"  # "time_first" | "interleaved"
    # The decoder is deliberately NOT causal by default, matching MiniMax-H3's own
    # VAE ("causal_encoder: true, causal_decoder: false"). A causal decoder after
    # nearest-neighbour temporal upsampling skews its receptive field backwards and
    # makes "repeat the first latent" the cheapest reconstruction. Set True only if
    # you need a strictly streaming decoder and accept the static-output failure.
    causal_decoder: bool = False

    @property
    def num_downs(self) -> int:
        return len(self.channel_multipliers) - 1

    @property
    def channels(self) -> List[int]:
        return [self.base_channels * m for m in self.channel_multipliers]

    @property
    def strides(self) -> List[Tuple[int, int, int]]:
        return plan_strides(
            self.spatial_compression, self.temporal_compression, self.num_downs, self.downsample_order
        )


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
                enc.append(ResBlock3D(chans[i], norm_mode=cfg.norm_mode))
            enc.append(CausalConv3d(chans[i], chans[i + 1], (3, 3, 3), stride=stride))
        self.encoder = nn.Sequential(*enc)
        # Bottleneck at the widest width, then project to 2x latent channels
        # (first half mean, second half log-variance).
        self.mid = nn.Sequential(ResBlock3D(chans[-1], norm_mode=cfg.norm_mode), ResBlock3D(chans[-1], norm_mode=cfg.norm_mode))
        self.enc_to_latent = CausalConv3d(chans[-1], 2 * cfg.latent_channels)

        dec_causal = cfg.causal_decoder
        self.latent_to_dec = CausalConv3d(cfg.latent_channels, chans[-1], causal=dec_causal)
        self.dec_res = nn.Sequential(
            *[
                ResBlock3D(chans[-1], norm_mode=cfg.norm_mode, causal=dec_causal)
                for _ in range(cfg.num_res_blocks)
            ]
        )
        # One up-stage per encoder down-stage, applied in reverse order.
        self.up_stages = nn.ModuleList(
            [
                nn.Sequential(
                    ResBlock3D(chans[i + 1], norm_mode=cfg.norm_mode, causal=dec_causal),
                    CausalConv3d(chans[i + 1], chans[i], causal=dec_causal),
                )
                for i in range(cfg.num_downs - 1, -1, -1)
            ]
        )
        self.dec_out = nn.Sequential(
            ClipOrFrameNorm(chans[0], groups, mode=cfg.norm_mode),
            nn.SiLU(),
            CausalConv3d(chans[0], cfg.in_channels, causal=dec_causal),
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

    def recon_loss(
        self,
        video: Tensor,
        kl_weight: float = 1e-4,
        temporal_weight: float = 10.0,
        temporal_relative: bool = True,
        temporal_form: str = "motion_weighted",
    ) -> Tuple[Tensor, dict]:
        """L1 + multi-scale gradient + motion-weighted temporal gradient + small KL.

        The temporal term is not optional, and its *form* is the part that matters.
        Frame-to-frame differences in a video are typically 10-30x smaller than
        per-pixel contrast, so plain L1 is minimised by emitting one averaged frame
        repeated N times -- the decoder reaches a good loss while producing **no
        motion at all**. Measured before this term existed: reconstruction
        frame-to-frame delta of exactly 0.0, over 600 steps of overfitting four
        clips.

        Unweighted temporal losses are not enough either. In a typical clip more
        than 90 % of pixels sit on a static background, where ``d_rec - d_ref`` is
        exactly zero and the term contributes no gradient, so the gradient that
        could teach motion is diluted to nothing. Weighting by ``|d_ref|`` per
        pixel concentrates it on the pixels that actually move. See the comment on
        ``temporal_form`` below for the measurement.

        ``temporal_relative`` (default) divides by the mean squared motion, making
        the term scale-free: a static decoder scores ~1.0 however fast the clip
        moves. Set ``temporal_relative=False`` for the absolute form.
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
        if video.shape[2] > 1:
            d_rec = rec[:, :, 1:] - rec[:, :, :-1]
            d_ref = video[:, :, 1:] - video[:, :, :-1]
            if temporal_form == "motion_weighted":
                # Two details here are what make motion appear at all, and both were
                # found by measurement rather than by reasoning.
                #
                # 1. Weight each pixel by how much it actually moves. In a typical
                #    video >90% of pixels sit on a static background where
                #    ``d_rec - d_ref`` is exactly zero, so an unweighted term has
                #    almost no gradient where motion lives. Same seed, 25 steps:
                #    unweighted MSE and Charbonnier both stayed at a reconstruction
                #    frame-delta of exactly 0.00000; this reached 0.32.
                # 2. Normalise **per clip**, not per batch. Clips in the same batch
                #    differ by more than an order of magnitude in how much they
                #    move, so a batch-level mean makes the loss jump by ~20x between
                #    steps and the run never settles: measured reconstruction L1
                #    oscillating between 0.05 and 1.27 across consecutive batches.
                per_clip = (d_ref.abs() * (d_rec - d_ref).pow(2)).flatten(1).sum(1)
                per_clip_w = d_ref.abs().flatten(1).sum(1)
                scale = d_ref.pow(2).flatten(1).mean(1).clamp(min=1e-8)
                raw = per_clip / per_clip_w.clamp(min=1e-8)
                if temporal_relative:
                    raw = raw / scale
                # A clip with no motion at all should not be asked to move.
                temporal = torch.where(per_clip_w > 1e-6, raw, torch.zeros_like(raw)).mean()
            elif temporal_form == "l1":
                temporal = F.l1_loss(d_rec, d_ref)
            elif temporal_form == "mse":
                temporal = F.mse_loss(d_rec, d_ref)
            else:
                raise ValueError(f"unknown temporal_form {temporal_form!r}")
            if temporal_relative and temporal_form != "motion_weighted":
                # Scale-free: a static decoder scores ~1.0 whatever the clip does.
                temporal = temporal / d_ref.pow(2).mean().clamp(min=1e-8)
        else:
            temporal = l1.new_zeros(())
        kl = self.kl_divergence(posterior)
        total = l1 + 0.25 * grad + temporal_weight * temporal + kl_weight * kl
        return total, {
            "l1": float(l1.detach()),
            "grad": float(grad.detach()),
            "temporal": float(temporal.detach()),
            "kl": float(kl.detach()),
            "rec": float(total.detach()),
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["NanoVAE", "VAEConfig", "CausalConv3d", "ClipOrFrameNorm", "causal_pad", "plan_strides", "ResBlock3D"]

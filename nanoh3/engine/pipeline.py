"""End-to-end text-to-video pipeline: prompt -> latents -> frames -> mp4."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ..models import DiTConfig, NanoDiT, NanoTextEncoder, NanoVAE, TextEncoderConfig, VAEConfig
from .offload import OffloadStats, SequentialOffload
from .quant import QuantMode, quantize_module_
from .sampler import SamplerConfig, add_noise, make_noise, sample


@dataclass
class GenerationConfig:
    prompt: str = "a red ball rolling to the right"
    negative_prompt: str = ""
    width: int = 256
    height: int = 256
    frames: int = 16
    steps: int = 4
    guidance: float = 1.0
    shift: float = 3.0
    seed: int = 0
    fps: int = 8
    batch: int = 1
    decode_chunk: int = 0  # 0 = decode the whole clip at once
    offload: bool = False
    offload_vae: bool = False
    active_depth: int = 0  # 0 = every block
    progress: bool = False

    def latent_shape(self, vae: NanoVAE) -> Tuple[int, int, int]:
        return vae.latent_shape(self.frames, self.height, self.width)


@dataclass
class ModelBundle:
    vae: NanoVAE
    dit: NanoDiT
    text: NanoTextEncoder
    dit_config: DiTConfig = field(default_factory=DiTConfig)
    vae_config: VAEConfig = field(default_factory=VAEConfig)
    text_config: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    meta: Dict[str, object] = field(default_factory=dict)

    def parameter_counts(self) -> Dict[str, int]:
        return {
            "dit": self.dit.num_parameters(),
            "vae": self.vae.num_parameters(),
            "text": self.text.num_parameters(),
            "total": self.dit.num_parameters() + self.vae.num_parameters() + self.text.num_parameters(),
        }


class NanoH3Pipeline:
    """Owns the three modules and the knobs that make them fit in 8 GB."""

    def __init__(self, bundle: ModelBundle, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        self.bundle = bundle
        self.device = torch.device(device)
        self.dtype = dtype
        self.vae.to(self.device, dtype)
        self.dit.to(self.device)
        self.text.to(self.device, dtype)
        self.dit.eval()
        self.vae.eval()
        self.text.eval()
        self.last_offload: Optional[OffloadStats] = None
        self.last_timing: Dict[str, float] = {}

    # ---- construction -------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | str = "cpu",
        quant: QuantMode = "none",
        dtype: torch.dtype = torch.float32,
    ) -> "NanoH3Pipeline":
        path = Path(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        dit_cfg = DiTConfig(**state["dit_config"])
        vae_cfg = VAEConfig(**state["vae_config"])
        text_cfg = TextEncoderConfig(**state["text_config"])
        dit = NanoDiT(dit_cfg)
        vae = NanoVAE(vae_cfg)
        text = NanoTextEncoder(text_cfg)
        dit.load_state_dict(state["dit"])
        vae.load_state_dict(state["vae"])
        text.load_state_dict(state["text"])
        if quant not in ("none", "bf16"):
            quantize_module_(dit, quant)
            quantize_module_(vae, quant)
        bundle = ModelBundle(vae, dit, text, dit_cfg, vae_cfg, text_cfg, state.get("meta", {}))
        return cls(bundle, device=device, dtype=dtype)

    def save_checkpoint(self, path: str | Path, meta: Optional[Dict[str, object]] = None) -> None:
        bundle = self.bundle
        torch.save(
            {
                "dit": {k: v for k, v in bundle.dit.state_dict().items() if not k.endswith(".err")},
                "vae": bundle.vae.state_dict(),
                "text": bundle.text.state_dict(),
                "dit_config": asdict(bundle.dit_config),
                "vae_config": asdict(bundle.vae_config),
                "text_config": asdict(bundle.text_config),
                "meta": meta or bundle.meta,
            },
            path,
        )

    # ---- helpers ------------------------------------------------------
    def encode_prompt(self, prompt: str) -> Tuple[Tensor, Tensor]:
        context, mask = self.text.embed_prompts([prompt], self.device)
        return context.to(self.dtype), mask

    def _model_fn(self, cfg: GenerationConfig, context: Tensor, mask: Tensor, negative: Optional[Tuple[Tensor, Tensor]]):
        def fn(x: Tensor, t: Tensor) -> Tensor:
            v = self.dit(x.to(self.dtype), t, context.to(x.dtype), mask)
            if cfg.guidance != 1.0 and negative is not None:
                neg_ctx, neg_mask = negative
                v_neg = self.dit(x.to(self.dtype), t, neg_ctx.to(x.dtype), neg_mask)
                v = v_neg + cfg.guidance * (v - v_neg)
            return v
        return fn

    def decode(self, latents: Tensor, frames: int, height: int, width: int, chunk: int = 0) -> Tensor:
        """Decode latents to pixels, optionally in temporal chunks to cap activation memory."""
        if chunk and chunk < latents.shape[2]:
            pieces: List[Tensor] = []
            for start in range(0, latents.shape[2], chunk):
                end = min(start + chunk, latents.shape[2])
                f0, f1 = start * self.bundle.vae_config.temporal_compression, end * self.bundle.vae_config.temporal_compression
                f1 = min(f1, frames)
                pieces.append(self.bundle.vae.decode(latents[:, :, start:end], (f1 - f0, height, width)))
            return torch.cat(pieces, dim=2)
        return self.bundle.vae.decode(latents, (frames, height, width))

    # ---- main entry point ---------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        cfg: GenerationConfig,
        callback=None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Returns (video in [-1, 1] shaped (B, 3, F, H, W), timing dict)."""
        vae = self.bundle.vae
        timing: Dict[str, float] = {}
        t0 = time.perf_counter()

        context, mask = self.encode_prompt(cfg.prompt)
        negative = self.encode_prompt(cfg.negative_prompt) if cfg.guidance != 1.0 and cfg.negative_prompt else None
        timing["text"] = time.perf_counter() - t0

        f_lat, h_lat, w_lat = vae.latent_shape(cfg.frames, cfg.height, cfg.width)
        latents = make_noise([cfg.batch, vae.cfg.latent_channels, f_lat, h_lat, w_lat], self.device, self.dtype, cfg.seed)
        model_fn = self._model_fn(cfg, context, mask, negative)

        if cfg.active_depth:
            self.bundle.dit.set_active_depth(cfg.active_depth)
        t1 = time.perf_counter()
        offload_ctx = (
            SequentialOffload(self.bundle.dit.blocks, self.device, enabled=cfg.offload)
            if cfg.offload
            else _NullContext()
        )
        with offload_ctx as offload:
            denoised = sample(
                model_fn,
                latents,
                SamplerConfig(steps=cfg.steps, shift=cfg.shift, guidance=cfg.guidance),
                progress=cfg.progress,
            )
        if cfg.active_depth:
            self.bundle.dit.reset_active_depth()
        if cfg.offload:
            self.last_offload = offload.stats
            timing["offload_moves"] = float(offload.stats.moves)
        timing["denoise"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        vae_offload = SequentialOffload(vae.up_stages, self.device, enabled=cfg.offload_vae) if cfg.offload_vae else _NullContext()
        with vae_offload:
            video = self.decode(denoised, cfg.frames, cfg.height, cfg.width, cfg.decode_chunk).float()
        timing["decode"] = time.perf_counter() - t2
        timing["total"] = time.perf_counter() - t0
        if callback is not None:
            callback(video, timing)
        return video.clamp(-1, 1), timing


class _NullContext:
    """Stand-in for SequentialOffload when offload is disabled."""

    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def save_video(video: Tensor, path: str | Path, fps: int = 8) -> Path:
    """Write a (B, 3, F, H, W) tensor in [-1, 1] as an H.264 mp4."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clip = video[0].detach().float().clamp(-1, 1).add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8)
    frames_np = clip.permute(1, 2, 3, 0).cpu().numpy()  # (F, H, W, C)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in frames_np:
            writer.append_data(frame)
    return path


__all__ = ["NanoH3Pipeline", "GenerationConfig", "ModelBundle", "save_video", "add_noise"]

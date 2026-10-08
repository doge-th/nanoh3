"""Two-stage training: the autoencoder first, then the denoiser on frozen latents.

    python -m nanoh3.training.train --config configs/demo.yaml --stage vae
    python -m nanoh3.training.train --config configs/demo.yaml --stage dit

Stage separation is not ceremony: the denoiser only ever sees normalised latents,
so the VAE can be re-fit later without invalidating the denoiser as long as the
latent statistics are re-measured (``--stage dit --refresh-stats``).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor

from ..data.procedural import ProceduralClips
from ..engine.pipeline import ModelBundle, save_video
from ..engine.sampler import add_noise
from ..models import DiTConfig, NanoDiT, NanoTextEncoder, NanoVAE, TextEncoderConfig, VAEConfig


@dataclass
class TrainConfig:
    name: str = "demo"
    out_dir: str = "runs/demo"
    device: str = "auto"
    seed: int = 0
    # data
    frames: int = 9
    size: int = 64
    batch_size: int = 4
    dataset_size: int = 256
    # vae
    vae_steps: int = 300
    vae_lr: float = 2e-4
    vae_kl_weight: float = 1e-4
    vae_temporal_weight: float = 10.0
    vae_temporal_form: str = "motion_weighted"
    vae_temporal_relative: bool = True
    # dit
    dit_steps: int = 400
    dit_lr: float = 2e-4
    text_lr: float = 2e-4
    text_loss_weight: float = 0.05
    sample_latent: bool = False
    # model
    dit: Dict[str, object] = field(default_factory=dict)
    vae: Dict[str, object] = field(default_factory=dict)
    text: Dict[str, object] = field(default_factory=dict)
    # misc
    log_every: int = 25
    save_every: int = 200
    grad_clip: float = 1.0
    max_seconds: float = 0.0  # 0 = no wall-clock cap
    preview_every: int = 0  # 0 = no preview videos

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**data)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_models(cfg: TrainConfig, device: torch.device) -> ModelBundle:
    dit_cfg = DiTConfig(**cfg.dit)
    vae_cfg = VAEConfig(**cfg.vae)
    text_cfg = TextEncoderConfig(**cfg.text)
    return ModelBundle(
        vae=NanoVAE(vae_cfg),
        dit=NanoDiT(dit_cfg),
        text=NanoTextEncoder(text_cfg),
        dit_config=dit_cfg,
        vae_config=vae_cfg,
        text_config=text_cfg,
    )


def save_bundle(bundle: ModelBundle, path: Path, meta: Optional[Dict[str, object]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dit": {k: v for k, v in bundle.dit.state_dict().items()},
            "vae": bundle.vae.state_dict(),
            "text": bundle.text.state_dict(),
            "dit_config": asdict(bundle.dit_config),
            "vae_config": asdict(bundle.vae_config),
            "text_config": asdict(bundle.text_config),
            "meta": meta or {},
        },
        path,
    )


def load_bundle(path: str | Path, device: torch.device) -> ModelBundle:
    state = torch.load(path, map_location="cpu", weights_only=False)
    bundle = ModelBundle(
        vae=NanoVAE(VAEConfig(**state["vae_config"])),
        dit=NanoDiT(DiTConfig(**state["dit_config"])),
        text=NanoTextEncoder(TextEncoderConfig(**state["text_config"])),
        dit_config=DiTConfig(**state["dit_config"]),
        vae_config=VAEConfig(**state["vae_config"]),
        text_config=TextEncoderConfig(**state["text_config"]),
        meta=state.get("meta", {}),
    )
    bundle.vae.load_state_dict(state["vae"])
    bundle.dit.load_state_dict(state["dit"])
    bundle.text.load_state_dict(state["text"])
    return _move_bundle(bundle, device)


def _move_bundle(bundle: ModelBundle, device: torch.device) -> ModelBundle:
    bundle.vae.to(device)
    bundle.dit.to(device)
    bundle.text.to(device)
    return bundle


# ---------------------------------------------------------------- stage 1: VAE


def train_vae(cfg: TrainConfig, bundle: ModelBundle, device: torch.device, data: ProceduralClips) -> Dict[str, float]:
    opt = torch.optim.AdamW(bundle.vae.parameters(), lr=cfg.vae_lr, betas=(0.9, 0.99), weight_decay=0.01)
    bundle.vae.train()
    stats = {"first": float("nan"), "last": float("nan"), "steps": 0}
    start = time.perf_counter()
    for step in range(1, cfg.vae_steps + 1):
        video, _ = data.batch(cfg.batch_size)
        video = video.to(device)
        loss, metrics = bundle.vae.recon_loss(
            video,
            kl_weight=cfg.vae_kl_weight,
            temporal_weight=cfg.vae_temporal_weight,
            temporal_relative=cfg.vae_temporal_relative,
            temporal_form=cfg.vae_temporal_form,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bundle.vae.parameters(), cfg.grad_clip)
        opt.step()
        if step == 1:
            stats["first"] = metrics["l1"]
        if step % cfg.log_every == 0 or step == 1:
            print(
                f"[vae] step {step:5d}/{cfg.vae_steps} l1={metrics['l1']:.4f} "
                f"motion={metrics['temporal']:.4f} kl={metrics['kl']:.3f} total={metrics['rec']:.4f}",
                flush=True,
            )
        stats["last"] = metrics["l1"]
        stats["steps"] = step
        if cfg.max_seconds and time.perf_counter() - start > cfg.max_seconds:
            print(f"[vae] wall-clock cap hit at step {step}", flush=True)
            break
    return stats


# ---------------------------------------------------------------- stage 2: DiT


def dit_loss(
    bundle: ModelBundle,
    latents: Tensor,
    context: Tensor,
    mask: Tensor,
    timesteps: Tensor,
    noise: Tensor,
    text: Optional[Tensor] = None,
    text_mask: Optional[Tensor] = None,
    text_weight: float = 0.0,
) -> Tuple[Tensor, Dict[str, float]]:
    x_t, target = add_noise(latents, noise, timesteps)
    pred = bundle.dit(x_t, timesteps, context, mask)
    loss = F.mse_loss(pred, target)
    metrics = {"flow": float(loss.detach())}
    if text is not None and text_weight > 0:
        rebuilt, _ = bundle.text(text, text_mask)
        t_loss = F.mse_loss(rebuilt, context.detach())
        loss = loss + text_weight * t_loss
        metrics["text"] = float(t_loss.detach())
    return loss, metrics


def train_dit(
    cfg: TrainConfig,
    bundle: ModelBundle,
    device: torch.device,
    data: ProceduralClips,
    stats: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    bundle.vae.eval()
    for p in bundle.vae.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(
        [
            {"params": bundle.dit.parameters(), "lr": cfg.dit_lr},
            {"params": bundle.text.parameters(), "lr": cfg.text_lr},
        ],
        betas=(0.9, 0.99),
        weight_decay=0.01,
    )
    bundle.dit.train()
    bundle.text.train()
    mean = float(stats.get("latent_mean", 0.0)) if stats else 0.0
    std = float(stats.get("latent_std", 1.0)) if stats else 1.0
    out: Dict[str, float] = {"first": float("nan"), "last": float("nan"), "steps": 0}
    start = time.perf_counter()
    for step in range(1, cfg.dit_steps + 1):
        video, prompts = data.batch(cfg.batch_size)
        video = video.to(device)
        with torch.no_grad():
            posterior = bundle.vae.encode(video)
            # The posterior mean is the training target by default. Sampling from
            # the posterior is the textbook choice for a VAE with a real KL term;
            # here the variance head is only weakly regularised, so a sample is a
            # large random perturbation and it corrupts the flow-matching target.
            latent = bundle.vae.sample_latent(posterior) if cfg.sample_latent else posterior.chunk(2, dim=1)[0]
        latent = ((latent - mean) / std).float()
        context, mask = bundle.text.embed_prompts(prompts, device)
        noise = torch.randn_like(latent)
        t = torch.rand(cfg.batch_size, device=device)
        loss, metrics = dit_loss(bundle, latent, context, mask, t, noise, text_weight=cfg.text_loss_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(bundle.dit.parameters()) + list(bundle.text.parameters()), cfg.grad_clip
        )
        opt.step()
        if step == 1:
            out["first"] = metrics["flow"]
        if step % cfg.log_every == 0 or step == 1:
            print(f"[dit] step {step:5d}/{cfg.dit_steps} flow={metrics['flow']:.4f}", flush=True)
        if cfg.preview_every and step % cfg.preview_every == 0:
            preview_path = Path(cfg.out_dir) / f"preview_{step:05d}.mp4"
            write_preview(bundle, device, data, preview_path, mean, std, steps=8)
        out["last"] = metrics["flow"]
        out["steps"] = step
        if cfg.max_seconds and time.perf_counter() - start > cfg.max_seconds:
            print(f"[dit] wall-clock cap hit at step {step}", flush=True)
            break
    return out


@torch.no_grad()
def write_preview(
    bundle: ModelBundle,
    device: torch.device,
    data: ProceduralClips,
    path: Path,
    mean: float = 0.0,
    std: float = 1.0,
    steps: int = 8,
    shift: float = 3.0,
) -> Path:
    """Sample one training prompt end-to-end and write it to mp4."""
    from ..engine.sampler import SamplerConfig, sample

    video, prompt = data[0]
    context, mask = bundle.text.embed_prompts([prompt], device)
    f, h, w = bundle.vae.latent_shape(video.shape[1], video.shape[2], video.shape[3])
    x = torch.randn(1, bundle.vae_config.latent_channels, f, h, w, device=device)

    def fn(xx: Tensor, tt: Tensor) -> Tensor:
        return bundle.dit(xx, tt, context, mask)

    latent = sample(fn, x, SamplerConfig(steps=steps, shift=shift))
    frames = bundle.vae.decode(latent * std + mean, (video.shape[1], video.shape[2], video.shape[3]))
    return save_video(frames.clamp(-1, 1), path, fps=8)


def latent_statistics(bundle: ModelBundle, device: torch.device, data: ProceduralClips, n: int = 32) -> Dict[str, float]:
    """Mean/std of VAE latents, needed to normalise the denoiser's targets."""
    vals: List[Tensor] = []
    with torch.no_grad():
        for i in range(min(n, len(data))):
            video, _ = data[i]  # (C, F, H, W)
            vals.append(bundle.vae.encode(video.unsqueeze(0).to(device)).chunk(2, dim=1)[0].float())
    flat = torch.cat([v.flatten() for v in vals])
    return {"latent_mean": float(flat.mean()), "latent_std": float(flat.std().clamp(min=1e-4))}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train NanoH3")
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--stage", choices=["vae", "dit", "all"], default="all")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=None, help="override both stage step counts")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = TrainConfig.load(args.config)
    if args.steps:
        cfg.vae_steps = args.steps
        cfg.dit_steps = args.steps
    if args.max_seconds is not None:
        cfg.max_seconds = args.max_seconds
    if args.out:
        cfg.out_dir = args.out
    device = resolve_device(args.device or cfg.device)
    torch.manual_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = ProceduralClips(frames=cfg.frames, size=cfg.size, seed=cfg.seed)
    bundle = build_models(cfg, device)
    _move_bundle(bundle, device)
    counts = bundle.parameter_counts()
    print(f"device={device} params={json.dumps(counts)}", flush=True)

    report: Dict[str, object] = {"config": asdict(cfg), "params": counts, "device": str(device)}
    if args.stage in ("vae", "all"):
        report["vae"] = train_vae(cfg, bundle, device, data)
        save_bundle(bundle, out_dir / "stage1_vae.pt", {"stage": "vae", "train": asdict(cfg)})
    if args.stage in ("dit", "all"):
        stats = latent_statistics(bundle, device, data)
        report["latent_stats"] = stats
        report["dit"] = train_dit(cfg, bundle, device, data, stats)
        save_bundle(bundle, out_dir / "model.pt", {"stage": "dit", "train": asdict(cfg), "latent_stats": stats})
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report.get("dit", report.get("vae", {})), indent=2), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

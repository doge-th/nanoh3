"""Few-step distillation (LCM-style trajectory skipping).

The <60 s target is a *sampler* problem, not a model-size problem: at 4 steps a
320 M denoiser already runs in a few seconds on an 8 GB card, while the 20-30 step
schedule that photorealism needs is what blows the budget. This module shrinks the
schedule instead of the network.

Method
------
1. Roll the teacher out with a fine Euler schedule, recording ``x_i`` at each step.
2. Sample a skip ``(i, j = i + k)``.
3. Train the student to land on the teacher's endpoint ``x_j`` in one jump.

The student keeps the same parameterisation (it still predicts velocity); we just
convert to the x0 prediction inside the loss, ``x0_hat = x_t - t * v`` (exact for
rectified flow at ``t=1``, and a well-behaved target for smaller ``t``). This is
progressive step distillation: repeat with smaller ``k`` to push the model to 2
steps.

    python -m nanoh3.training.distill --config configs/demo.yaml --input runs/demo/model.pt
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import copy

from torch import Tensor

from ..engine.sampler import euler_step, timesteps
from .train import TrainConfig, load_bundle, resolve_device, save_bundle


@dataclass
class DistillConfig:
    name: str = "demo-distill"
    out_dir: str = "runs/demo/distilled"
    steps: int = 300
    lr: float = 2e-5
    batch_size: int = 2
    teacher_schedule: int = 16  # fine Euler steps used to build the trajectory
    skip: int = 4  # k: jump k teacher steps per student update
    t_min: float = 0.15  # do not train on the very end of the schedule
    grad_clip: float = 1.0
    log_every: int = 25
    seed: int = 0
    max_seconds: float = 0.0
    # Must match how the teacher was trained. Sampling from the posterior when
    # the teacher was fit on the mean injects a large random perturbation into
    # every distillation target and the loss diverges.
    sample_latent: bool = False


def velocity_to_x0(x_t: Tensor, v: Tensor, t: Tensor) -> Tensor:
    """x0 = x_t - t * v for the rectified-flow parameterisation."""
    return x_t - t.view(-1, *([1] * (x_t.dim() - 1))) * v


@torch.no_grad()
def teacher_trajectory(
    teacher,
    latents: Tensor,
    context: Tensor,
    mask: Tensor,
    schedule: int,
    shift: float,
) -> Tuple[List[Tensor], Tensor]:
    """Return the list of states along the teacher trajectory and its t-schedule."""
    ts = timesteps(schedule, shift, latents.device)
    states: List[Tensor] = [latents]
    x = latents
    for i in range(schedule):
        t = torch.full((x.shape[0],), float(ts[i]), device=x.device, dtype=x.dtype)
        v = teacher(x, t, context, mask)
        x = euler_step(x, v, float(ts[i]), float(ts[i + 1]))
        states.append(x)
    return states, ts


def distill_step(
    teacher,
    student,
    latents: Tensor,
    context: Tensor,
    mask: Tensor,
    cfg: DistillConfig,
) -> Tuple[Tensor, Dict[str, float]]:
    """One distillation update: teach the student a longer jump than the teacher takes.

    The trajectory must start from **noise** at ``t = 1``, not from the data. That
    is the direction the sampler actually runs in; starting from the data and
    integrating "backwards" from t=1 produces a trajectory the student will never
    be asked to follow, and the loss diverges (observed: x0_mse ~1e3).
    """
    noise = torch.randn_like(latents)
    states, ts = teacher_trajectory(teacher, noise, context, mask, cfg.teacher_schedule, shift=3.0)
    b = latents.shape[0]
    idx = torch.randint(0, cfg.teacher_schedule - cfg.skip + 1, (b,), device=latents.device)
    # `states[j]` already carries the batch dimension, so gather rows rather than
    # stacking whole batches (which would add a spurious leading axis).
    x_t = torch.stack([states[int(j)][k] for k, j in enumerate(idx)])
    t_i = ts[idx]
    target_x0 = torch.stack([states[int(j) + cfg.skip][k] for k, j in enumerate(idx)])

    # Below t_min the teacher endpoint is unreliable, so fall back to the real data.
    keep = (t_i >= cfg.t_min).float().view(-1, *([1] * (latents.dim() - 1)))
    target_x0 = latents * (1 - keep) + target_x0 * keep

    v = student(x_t, t_i, context, mask)
    pred_x0 = velocity_to_x0(x_t, v, t_i)
    # Normalise by the number of *masked elements*, not by the batch size:
    # `keep` broadcasts over the latent dimensions, so `keep.sum()` counts one
    # per sample while the numerator sums over every element. Dividing by the
    # batch size inflates the loss (and the gradient) by the element count.
    sq = (pred_x0 - target_x0).pow(2) * keep
    loss = sq.sum() / keep.expand_as(sq).sum().clamp(min=1.0)
    return loss, {"t_mean": float(t_i.mean()), "x0_mse": float(loss.detach())}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Distill NanoH3 to few steps")
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--input", required=True, help="checkpoint from nanoh3.training.train")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--skip", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args(argv)

    train_cfg = TrainConfig.load(args.config)
    cfg = DistillConfig(out_dir=args.out or f"{train_cfg.out_dir}/distilled")
    if args.steps:
        cfg.steps = args.steps
    if args.skip:
        cfg.skip = args.skip
    if args.max_seconds is not None:
        cfg.max_seconds = args.max_seconds
    device = resolve_device(args.device or train_cfg.device)
    torch.manual_seed(cfg.seed)

    from ..data.procedural import ProceduralClips

    data = ProceduralClips(frames=train_cfg.frames, size=train_cfg.size, seed=train_cfg.seed)
    bundle = load_bundle(args.input, device)
    # The teacher must be a *separate*, frozen copy. Sharing parameters would mean
    # freezing the student's weights too, and the optimiser would then have
    # nothing to update.
    teacher = copy.deepcopy(bundle.dit).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = bundle.dit.train()

    stats = bundle.meta.get("latent_stats", {}) if isinstance(bundle.meta, dict) else {}
    mean = float(stats.get("latent_mean", 0.0))
    std = float(stats.get("latent_std", 1.0))
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, betas=(0.9, 0.99), weight_decay=0.0)

    report: Dict[str, object] = {"config": asdict(cfg), "device": str(device)}
    first, last = float("nan"), float("nan")
    start = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        video, prompts = data.batch(cfg.batch_size)
        with torch.no_grad():
            posterior = bundle.vae.encode(video.to(device))
            latent = (
                bundle.vae.sample_latent(posterior)
                if cfg.sample_latent
                else posterior.chunk(2, dim=1)[0]
            )
            latent = ((latent - mean) / std).float()
        context, mask = bundle.text.embed_prompts(prompts, device)
        loss, metrics = distill_step(teacher, student, latent, context, mask, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        opt.step()
        if step == 1:
            first = metrics["x0_mse"]
        if step % cfg.log_every == 0 or step == 1:
            print(f"[distill] step {step:5d}/{cfg.steps} x0_mse={metrics['x0_mse']:.4f}", flush=True)
        last = metrics["x0_mse"]
        if cfg.max_seconds and time.perf_counter() - start > cfg.max_seconds:
            print(f"[distill] wall-clock cap hit at step {step}", flush=True)
            break
    report.update({"first_loss": first, "last_loss": last})
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_bundle(bundle, out_dir / "model.pt", {"stage": "distilled", "distill": asdict(cfg), "latent_stats": {"latent_mean": mean, "latent_std": std}})
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

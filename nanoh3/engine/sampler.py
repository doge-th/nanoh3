"""Few-step samplers for rectified-flow models.

The flow convention throughout NanoH3 is ``t = 1`` (noise) -> ``t = 0`` (data)::

    x_t = (1 - t) * x_data + t * noise
    v_target = noise - x_data

so sampling is a plain Euler integration of ``dx/dt = v_theta(x, t)`` downwards.
A timestep *shift* redistributes the steps towards high noise, which is where a
video model's error budget actually lives; ``shift > 1`` spends more steps early.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class SamplerConfig:
    steps: int = 4
    shift: float = 3.0
    guidance: float = 1.0
    method: str = "euler"  # euler | heun | multistep
    eta: float = 0.0


def timesteps(steps: int, shift: float = 1.0, device: torch.device | str = "cpu") -> Tensor:
    """Descending t schedule from 1 to 0, optionally shifted towards high noise."""
    t = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    if shift != 1.0:
        t = shift * t / (1.0 + (shift - 1.0) * t)
    return t


def euler_step(x: Tensor, v: Tensor, t_cur: float, t_next: float) -> Tensor:
    return x + (t_next - t_cur) * v


def heun_step(
    model_fn: Callable[[Tensor, Tensor], Tensor], x: Tensor, t_cur: float, t_next: float, t: Tensor
) -> Tensor:
    v = model_fn(x, t)
    x_next = euler_step(x, v, t_cur, t_next)
    if t_next <= 0.0:
        return x_next
    v2 = model_fn(x_next, t_next)
    return x_next + 0.5 * (t_next - t_cur) * (v + v2)


@torch.no_grad()
def sample(
    model_fn: Callable[[Tensor, Tensor], Tensor],
    latents: Tensor,
    cfg: SamplerConfig,
    generator: Optional[torch.Generator] = None,
    progress: bool = False,
) -> Tensor:
    """Integrate the flow from ``latents`` (t=1) down to data (t=0).

    ``model_fn(x, t)`` must return the predicted velocity with the same shape as ``x``.
    """
    ts = timesteps(cfg.steps, cfg.shift, latents.device)
    x = latents
    iterator = range(cfg.steps)
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc="sampling", leave=False)  # type: ignore[assignment]
        except ImportError:  # pragma: no cover - tqdm is optional
            pass
    for i in iterator:
        t_cur, t_next = float(ts[i]), float(ts[i + 1])
        t_batch = torch.full((x.shape[0],), t_cur, device=x.device, dtype=x.dtype)
        if cfg.method == "heun":
            x = heun_step(model_fn, x, t_cur, t_next, t_batch)
        elif cfg.method == "multistep":
            # Distilled models are trained so the direct velocity output is already
            # the x0 prediction; a plain Euler step in the shifted schedule suffices.
            x = euler_step(x, model_fn(x, t_batch), t_cur, t_next)
        else:
            x = euler_step(x, model_fn(x, t_batch), t_cur, t_next)
    return x


def make_noise(shape: List[int], device: torch.device, dtype: torch.dtype, seed: Optional[int] = None) -> Tensor:
    """Latent-space noise; a fixed seed makes generation reproducible."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=dtype)


def add_noise(x0: Tensor, noise: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
    """Forward process for flow matching. Returns (x_t, target_velocity)."""
    tt = t.view(-1, *([1] * (x0.dim() - 1)))
    x_t = (1.0 - tt) * x0 + tt * noise
    return x_t, noise - x0


__all__ = [
    "SamplerConfig",
    "timesteps",
    "euler_step",
    "heun_step",
    "sample",
    "make_noise",
    "add_noise",
]

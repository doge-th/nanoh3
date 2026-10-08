"""Procedurally generated clips.

The demonstration model is trained on these instead of scraped video for three
reasons: they are generated on the fly at any resolution, every factor
(colour / shape / motion / background) is known so the text conditioning has to
actually work, and the whole dataset is a few hundred lines of code rather than a
download. Real training data enters through :mod:`nanoh3.data.teacher` instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import torch
from torch import Tensor

COLORS: dict = {
    "red": (0.90, 0.18, 0.16),
    "green": (0.20, 0.75, 0.30),
    "blue": (0.20, 0.40, 0.92),
    "yellow": (0.95, 0.83, 0.20),
    "purple": (0.62, 0.30, 0.86),
    "white": (0.92, 0.92, 0.92),
    "orange": (0.95, 0.55, 0.15),
}
BACKGROUNDS: dict = {
    "black": (0.04, 0.04, 0.06),
    "navy": (0.06, 0.10, 0.25),
    "grey": (0.30, 0.30, 0.32),
    "cream": (0.90, 0.88, 0.80),
}
SHAPES = ("circle", "square", "triangle")
MOTIONS = (
    "still",
    "right",
    "left",
    "up",
    "down",
    "zoom",
    "pulse",
    # Camera motion: a pan translates the *whole frame*, so the temporal gradient
    # is large across every pixel instead of only at a small object's edges.
    # Real video is full of this; a small shape drifting on a flat background is
    # close to the worst case a video model can be handed, and it is what made
    # the demo VAE collapse to a still image.
    "pan_right",
    "pan_left",
    "sweep",
)


@dataclass(frozen=True)
class ClipSpec:
    color: str
    shape: str
    motion: str
    background: str
    size: float

    def prompt(self) -> str:
        if self.motion == "pan_right":
            clause = "while the camera pans right"
        elif self.motion == "pan_left":
            clause = "while the camera pans left"
        else:
            clause = "that " + {
                "still": "stays still",
                "right": "moves to the right",
                "left": "moves to the left",
                "up": "moves upward",
                "down": "moves downward",
                "zoom": "grows larger",
                "pulse": "pulses in size",
                "sweep": "sweeps across the frame",
            }[self.motion]
        return f"a {self.color} {self.shape} on a {self.background} background {clause}"


def random_specs(count: int, seed: int = 0) -> List[ClipSpec]:
    import random

    rng = random.Random(seed)
    names = sorted(COLORS)
    bgs = sorted(BACKGROUNDS)
    return [
        ClipSpec(
            color=names[rng.randrange(len(names))],
            shape=SHAPES[rng.randrange(len(SHAPES))],
            motion=MOTIONS[rng.randrange(len(MOTIONS))],
            background=bgs[rng.randrange(len(bgs))],
            size=round(rng.uniform(0.22, 0.46), 3),
        )
        for _ in range(count)
    ]


def _shape_mask(shape: str, ys: Tensor, xs: Tensor, cx: Tensor, cy: Tensor, r: Tensor) -> Tensor:
    """Normalised coordinates in [-1, 1] -> boolean mask of the shape."""
    ny = (ys - cy) / r
    nx = (xs - cx) / r
    if shape == "circle":
        return (nx * nx + ny * ny) <= 1.0
    if shape == "square":
        return (nx.abs() <= 0.9) & (ny.abs() <= 0.9)
    if shape == "triangle":
        return (ny <= 0.85) & (ny >= -0.85) & (nx.abs() <= 0.9 * (1.0 - (ny + 0.85) / 1.7))
    raise ValueError(f"unknown shape {shape!r}")


def _stripe_pattern(size: int, period: int, offset: float) -> Tensor:
    """A repeating vertical stripe pattern, used to make a camera pan visible.

    A pan of a *flat* background is invisible, so camera-motion clips are drawn
    with stripes: shifting them produces a temporal gradient across the entire
    frame, which is exactly the signal the demo VAE was failing to learn.
    """
    xs = torch.linspace(0.0, 1.0, size)
    return ((xs * (size / period) + offset).floor() % 2).to(torch.bool)


def render_clip(spec: ClipSpec, frames: int, size: int = 64) -> Tensor:
    """Render one clip to (3, frames, size, size) in [-1, 1]."""
    device = torch.device("cpu")
    ys = torch.linspace(-1, 1, size, device=device)[:, None].expand(size, size)
    xs = torch.linspace(-1, 1, size, device=device)[None, :].expand(size, size)
    color = torch.tensor(COLORS[spec.color], device=device).view(3, 1)
    bg = torch.tensor(BACKGROUNDS[spec.background], device=device).view(3, 1, 1)
    panning = spec.motion in ("pan_right", "pan_left")
    out = torch.empty(3, frames, size, size, device=device)
    for f in range(frames):
        p = f / max(frames - 1, 1)
        cx, cy, r = 0.0, 0.0, spec.size
        if spec.motion == "right":
            cx = -0.6 + 1.2 * p
        elif spec.motion == "left":
            cx = 0.6 - 1.2 * p
        elif spec.motion == "up":
            cy = 0.6 - 1.2 * p
        elif spec.motion == "down":
            cy = -0.6 + 1.2 * p
        elif spec.motion == "zoom":
            r = spec.size * (0.5 + p)
        elif spec.motion == "pulse":
            r = spec.size * (0.75 + 0.45 * math.sin(p * math.pi * 2.0))
        elif spec.motion == "sweep":
            cx = -0.7 + 1.4 * p
            cy = 0.35 * math.sin(p * math.pi)
        mask = _shape_mask(spec.shape, ys, xs, torch.tensor(cx), torch.tensor(cy), torch.tensor(r))
        frame = bg.expand(3, size, size).clone()
        if panning:
            shift = (0.6 - 1.2 * p) if spec.motion == "pan_left" else (-0.6 + 1.2 * p)
            stripes = _stripe_pattern(size, period=6, offset=shift * 3.0)
            n = int(stripes.sum())
            tint = color.view(3, 1, 1).expand(3, size, n)
            frame[:, :, stripes] = frame[:, :, stripes] * 0.35 + tint
        frame[:, mask] = color.expand(3, int(mask.sum()))
        out[:, f] = frame
    return out * 2.0 - 1.0


class ProceduralClips:
    """Infinite stream of (video, prompt) pairs, shuffled per epoch by seed."""

    def __init__(self, frames: int = 9, size: int = 64, seed: int = 0) -> None:
        self.frames, self.size, self.seed = frames, size, seed
        self.specs = random_specs(256, seed)

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> Tuple[Tensor, str]:
        spec = self.specs[index % len(self.specs)]
        return render_clip(spec, self.frames, self.size), spec.prompt()

    def batch(self, batch_size: int, generator: torch.Generator | None = None) -> Tuple[Tensor, List[str]]:
        idx = torch.randint(0, len(self), (batch_size,), generator=generator)
        videos, prompts = [], []
        for i in idx.tolist():
            video, prompt = self[i]
            videos.append(video)
            prompts.append(prompt)
        return torch.stack(videos), prompts

    def __iter__(self) -> Iterator[Tuple[Tensor, str]]:
        for i in range(len(self)):
            yield self[i]


__all__ = ["ClipSpec", "ProceduralClips", "render_clip", "random_specs", "COLORS", "SHAPES", "MOTIONS"]

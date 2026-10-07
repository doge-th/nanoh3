"""Reference-clip preparation for distilling a permitted teacher into NanoH3.

The teacher is invoked as an *external command* and only its pixels are kept, so
the licence of the weights and of the inference code stays the operator's
decision and nothing proprietary is vendored into this repository.

**Licence guard.** The obvious candidate teacher is MiniMax-H3, and it is the one
teacher this module refuses by default. The MiniMax H3 Community License
Agreement section V.3 states you "may not use the MiniMax H3 Works or any of their
Outputs or results to improve any other artificial intelligence model". Training
NanoH3 on H3 output is exactly that, so :func:`check_teacher_rights` blocks it
unless the operator explicitly asserts a different grant. The default path is
self-distillation (distil a long-schedule model of ours into a few-step one) or a
permissively licensed dataset.

The default backend is ``dryrun``, which only writes the prompt manifest. That is
the mode used in CI, and it is also the mode you use first to sanity-check the
prompt set before spending GPU hours.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import Tensor


class TeacherRightsError(RuntimeError):
    """Raised when a teacher's licence forbids the use we would make of it."""


#: repo id -> why it is blocked by default
RESTRICTED_TEACHERS: Dict[str, str] = {
    "minimaxai/minimax-h3": (
        "MiniMax H3 Community License Agreement section V.3 forbids using the H3 works or their "
        "outputs to improve any other AI model. Train NanoH3 from permissively licensed data, "
        "self-distill, or pass --assert-rights if you hold a separate grant."
    ),
    "minimax/minimax-h3": (
        "MiniMax H3 Community License Agreement section V.3 forbids using the H3 works or their "
        "outputs to improve any other AI model."
    ),
    "comfy-org/minimax-h3": "Derivative of MiniMax-H3; inherits the same section V.3 restriction.",
}


def check_teacher_rights(repo: str, assert_rights: bool = False) -> str:
    """Return a note about the teacher's licence, or raise if the use is not allowed."""
    key = repo.strip().lower()
    if key in RESTRICTED_TEACHERS and not assert_rights:
        raise TeacherRightsError(f"refusing to use {repo!r} as a data source: {RESTRICTED_TEACHERS[key]}")
    if key in RESTRICTED_TEACHERS:
        return "asserted by operator -- confirm your grant covers model improvement"
    return f"{repo}: no restriction known to this tool; check the licence yourself"

DEFAULT_PROMPTS: List[str] = [
    "a paper boat drifting on a calm river at sunrise",
    "a neon city street in the rain at night, reflections on the asphalt",
    "a red fox walking through dry grass, close up",
    "a satellite orbiting above the earth, clouds slowly moving",
    "hands kneading dough on a wooden table, top down",
    "a train crossing a bridge over a river at dusk",
    "a jellyfish drifting through dark water, bioluminescence",
    "an astronaut walking across a dusty plain, low sun",
    "a busy night market with steam and lanterns",
    "leaves falling slowly in front of a window, afternoon light",
    "a skateboarder rolling along a concrete wall, golden hour",
    "a whale swimming past a small boat, deep blue water",
]


@dataclass
class TeacherConfig:
    """How to reach the teacher and at which preset."""

    backend: str = "dryrun"  # dryrun | command
    command: str = ""  # e.g. "python infer.py --prompt {prompt} --out {out}"
    repo: str = "MiniMaxAI/MiniMax-H3"
    # Lowest-resolution preset the research report identified as supported.
    width: int = 480
    height: int = 480
    frames: int = 33
    fps: int = 24
    steps: int = 20
    seed: int = 1234
    target_width: int = 256
    target_height: int = 256
    target_frames: int = 16
    out_dir: Path = Path("data/teacher")
    timeout_s: int = 3600
    assert_rights: bool = False
    extra_args: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        d = {k: v for k, v in self.__dict__.items()}
        d["out_dir"] = str(self.out_dir)
        return d


def build_manifest(cfg: TeacherConfig, prompts: Sequence[str]) -> List[Dict[str, object]]:
    """One record per clip: prompt, teacher preset, expected output path."""
    licence_note = check_teacher_rights(cfg.repo, cfg.assert_rights)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, object]] = []
    for i, prompt in enumerate(prompts):
        name = f"clip_{i:04d}"
        manifest.append(
            {
                "id": name,
                "prompt": prompt,
                "seed": cfg.seed + i,
                "teacher": {
                    "repo": cfg.repo,
                    "width": cfg.width,
                    "height": cfg.height,
                    "frames": cfg.frames,
                    "fps": cfg.fps,
                    "steps": cfg.steps,
                },
                "licence_note": licence_note,
                "raw_path": str(cfg.out_dir / f"{name}_raw.mp4"),
                "target": {"width": cfg.target_width, "height": cfg.target_height, "frames": cfg.target_frames},
            }
        )
    return manifest


def write_manifest(manifest: Iterable[Dict[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(manifest)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_backend(cfg: TeacherConfig, record: Dict[str, object], dry_run: bool = False) -> Optional[Path]:
    """Invoke the teacher for one record. Returns the produced file, or None if dry."""
    out = Path(str(record["raw_path"]))  # type: ignore[index]
    if dry_run or cfg.backend == "dryrun":
        return None
    if cfg.backend != "command":
        raise ValueError(f"unknown backend {cfg.backend!r}")
    if not cfg.command:
        raise ValueError("backend 'command' requires a --command template with {prompt} and {out}")
    cmd = cfg.command.format(prompt=shlex.quote(str(record["prompt"])), out=shlex.quote(str(out)))
    for key, value in cfg.extra_args.items():
        cmd += f" --{key} {shlex.quote(value)}"
    cmd += f" --seed {record['seed']}"
    subprocess.run(cmd, shell=True, check=True, timeout=cfg.timeout_s)
    if not out.exists():
        raise FileNotFoundError(f"teacher produced no file at {out}")
    return out


def load_video_tensor(path: str | Path) -> Tensor:
    """Read a video file into (3, F, H, W) in [-1, 1]."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    frames = [f for f in reader]
    reader.close()
    array = torch.from_numpy(__import__("numpy").stack(frames)).float().permute(3, 0, 1, 2) / 127.5 - 1.0
    return array


def fit_clip(video: Tensor, frames: int, height: int, width: int, mode: str = "uniform") -> Tensor:
    """Resize/crop a teacher clip onto the student's grid.

    ``uniform`` picks ``frames`` evenly spaced source frames (keeps motion speed),
    ``head`` takes the first ``frames``.
    """
    import torch.nn.functional as F

    if video.shape[1] >= frames:
        if mode == "uniform":
            idx = torch.linspace(0, video.shape[1] - 1, frames).round().long()
        else:
            idx = torch.arange(frames)
        video = video[:, idx]
    elif video.shape[1] < frames:
        pad = video[:, -1:].expand(-1, frames - video.shape[1], -1, -1)
        video = torch.cat([video, pad], dim=1)
    video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
    return video.clamp(-1, 1)


def save_clip(video: Tensor, path: str | Path) -> Path:
    from ..engine.pipeline import save_video

    return save_video(video.unsqueeze(0), path)


def postprocess(record: Dict[str, object], source: str | Path) -> Tensor:
    """Teacher mp4 -> student-sized clip tensor."""
    target = record["target"]  # type: ignore[index]
    video = load_video_tensor(source)
    return fit_clip(video, int(target["frames"]), int(target["height"]), int(target["width"]))  # type: ignore[index]


__all__ = [
    "TeacherConfig",
    "TeacherRightsError",
    "RESTRICTED_TEACHERS",
    "check_teacher_rights",
    "DEFAULT_PROMPTS",
    "build_manifest",
    "write_manifest",
    "run_backend",
    "load_video_tensor",
    "fit_clip",
    "save_clip",
    "postprocess",
]

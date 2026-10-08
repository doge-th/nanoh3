#!/usr/bin/env python
"""Score a trained NanoH3 checkpoint against the demo acceptance criteria.

    python scripts/eval_demo.py --checkpoint runs/demo/model.pt

Prints one number per criterion and exits non-zero if any of them fails, so the
loop "train -> measure -> fix" has a definition of done that is not a vibe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# MPS has no native conv3d kernel on this build; the CPU fallback is what the
# error message itself recommends, and for a 24-clip eval it costs seconds.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from nanoh3.data.procedural import ProceduralClips  # noqa: E402
from nanoh3.engine.pipeline import GenerationConfig, NanoH3Pipeline  # noqa: E402

# Acceptance thresholds, deliberately conservative.
MIN_RECON_L1 = 0.25  # VAE reconstruction, training clips
MIN_MOTION_RATIO = 0.50  # generated motion / reference clip motion
MIN_COLOR_CONTRAST = 0.15  # object-vs-background contrast in the output


def recon_metrics(bundle, data: ProceduralClips, n: int = 24) -> dict:
    vae = bundle.vae
    l1s, motion, ref_motion = [], [], []
    with torch.no_grad():
        for i in range(min(n, len(data))):
            video, _ = data[i]
            video = video.unsqueeze(0)
            # Keep the clip on the same device as the model; mixing devices here
            # silently produced NaN through the MPS CPU fallback path.
            video = video.to(next(vae.parameters()).device)
            latent = vae.encode(video).chunk(2, dim=1)[0]
            rec = vae.decode(latent, (video.shape[2], video.shape[3], video.shape[4]))[0]
            l1s.append(float((rec - video[0]).abs().mean()))
            motion.append(sum(float((rec[:, f] - rec[:, f - 1]).abs().mean()) for f in range(1, 9)) / 8)
            ref_motion.append(sum(float((video[0, :, f] - video[0, :, f - 1]).abs().mean()) for f in range(1, 9)) / 8)
    return {
        "recon_l1": sum(l1s) / len(l1s),
        "recon_motion": sum(motion) / len(motion),
        "ref_motion": sum(ref_motion) / len(ref_motion),
    }


def prompt_color_contrast(video: Tensor, prompt: str) -> float:
    """How much brighter is the prompt's colour than the rest of the frame.

    A crude but honest proxy for "did the model draw the thing": a flat dark blur
    scores near zero, a red circle on grey does not.
    """
    palette = {
        "red": (0.90, 0.18, 0.16),
        "green": (0.20, 0.75, 0.30),
        "blue": (0.20, 0.40, 0.92),
        "yellow": (0.95, 0.83, 0.20),
        "purple": (0.62, 0.30, 0.86),
        "white": (0.92, 0.92, 0.92),
        "orange": (0.95, 0.55, 0.15),
    }
    colour = None
    for name, rgb in palette.items():
        if prompt.split()[1:2] == [name]:
            colour = torch.tensor(rgb, device=video.device).view(3, 1, 1)
            break
    if colour is None:
        return float("nan")
    frame = video.mean(1)  # (F, H, W) in [-1, 1]
    dist = ((frame.unsqueeze(1) - colour.view(1, 3, 1, 1)) ** 2).sum(1).sqrt()  # (F, H, W)
    best = dist.min().item()
    typical = dist.median().item()
    return float(typical - best)  # >0 means something resembles the colour


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default="a red circle on a grey background that moves to the right")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    pipe = NanoH3Pipeline.from_checkpoint(args.checkpoint, device=args.device)
    data = ProceduralClips(frames=9, size=64, seed=0)
    recon = recon_metrics(pipe.bundle, data)
    cfg = GenerationConfig(
        prompt=args.prompt, width=args.width, height=args.height, frames=args.frames, steps=args.steps, seed=0
    )
    video, _ = pipe.generate(cfg)
    a = video[0]
    gen_motion = sum(float((a[:, f] - a[:, f - 1]).abs().mean()) for f in range(1, args.frames)) / (args.frames - 1)
    contrast = prompt_color_contrast(a, args.prompt)

    # The reference motion at the generated resolution, for the same prompt.
    ref = ProceduralClips(frames=args.frames, size=args.width, seed=0)
    matching = [i for i in range(len(ref)) if ref[i][1] == args.prompt]
    if matching:
        clip_ref = ref[matching[0]][0]
        ref_motion = (
            sum(float((clip_ref[:, f] - clip_ref[:, f - 1]).abs().mean()) for f in range(1, args.frames))
            / (args.frames - 1)
        )
    else:
        ref_motion = recon["ref_motion"]

    checks = {
        "recon_l1": (recon["recon_l1"], MIN_RECON_L1, recon["recon_l1"] < MIN_RECON_L1),
        "motion_ratio": (
            gen_motion / max(ref_motion, 1e-8),
            MIN_MOTION_RATIO,
            gen_motion / max(ref_motion, 1e-8) >= MIN_MOTION_RATIO,
        ),
        "colour_contrast": (contrast, MIN_COLOR_CONTRAST, contrast == contrast and contrast >= MIN_COLOR_CONTRAST),
    }
    report = {
        "checkpoint": args.checkpoint,
        "recon": recon,
        "generated_motion": gen_motion,
        "reference_motion": ref_motion,
        "colour_contrast": contrast,
        "checks": {k: {"value": round(v, 5), "threshold": t, "pass": ok} for k, (v, t, ok) in checks.items()},
        "all_pass": all(ok for _, _, ok in checks.values()),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
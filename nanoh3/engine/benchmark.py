"""Benchmark harness: measure here, project there -- and label which is which.

Every number this module prints is tagged:

* ``VERIFIED``  -- measured on the machine that ran the command.
* ``PROJECTED`` -- derived from a counted operation count divided by an assumed
  effective throughput. The assumption is printed next to the result; change it
  with ``--throughput`` and the projection changes with it.
* ``UNKNOWN``   -- not derivable without hardware we do not have.

That separation is the whole point. A "<60 s" claim for an 8 GB NVIDIA card is
unverifiable on a laptop, so it is reported as a projection with its assumptions
attached rather than as a measurement.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .budget import GIB, PROFILES, Profile
from .pipeline import GenerationConfig, NanoH3Pipeline

#: Effective bf16 matmul throughput assumed for a mid-range 8 GB NVIDIA card.
#: This is the single number every PROJECTED figure depends on.
DEFAULT_GPU_TFLOPS = 100.0


def device_report() -> Dict[str, object]:
    info: Dict[str, object] = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / GIB, 2)
    elif torch.backends.mps.is_available():
        info["gpu"] = "Apple Silicon (MPS)"
    else:
        info["gpu"] = "CPU"
    return info


def dit_flops(profile: Profile) -> float:
    """Multiply-accumulate count for one denoiser forward pass.

    Linear layers dominate: ~2 * (non-embedding parameters) per token. Attention
    adds 4 * N^2 * head_dim per layer on top, which is why this is reported
    separately -- it is the term that explodes at high resolution.
    """
    d = profile.hidden_size
    layers = profile.active_depth or profile.dit_depth
    n = profile.num_tokens * profile.batch
    mlp = int(d * profile.mlp_ratio) if hasattr(profile, "mlp_ratio") else 4 * d
    # qkv + out + cross-attn q/kv/proj + mlp, per layer, in multiply-accumulates
    per_token = 2 * (3 * d * d + d * d + d * d + 2 * d * d + d * d + d * mlp + mlp * d)
    linear = layers * n * per_token
    attn = layers * 2 * 2 * (profile.num_tokens**2) * (d // max(1, profile.dit_depth // layers or 1))
    return float(linear + attn)


def vae_flops(profile: Profile) -> float:
    """Rough decoder cost: a few conv3d passes over the full-resolution volume."""
    volume = profile.batch * profile.frames * profile.height * profile.width * profile.vae_base_channels
    # 27 MACs per 3x3x3 conv per output element, ~6 such layers at full resolution
    return float(volume * 27 * 6)


def measure(
    checkpoint: str,
    prompt: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    seed: int,
    device: str,
    quant: str,
    repeats: int,
    offload: bool,
) -> Dict[str, object]:
    resolved = device
    if resolved == "auto":
        resolved = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    pipeline = NanoH3Pipeline.from_checkpoint(checkpoint, device=resolved, quant=quant)  # type: ignore[arg-type]
    cfg = GenerationConfig(
        prompt=prompt, width=width, height=height, frames=frames, steps=steps, seed=seed, offload=offload
    )
    runs: List[Dict[str, float]] = []
    for i in range(repeats + 1):  # first pass is a warm-up and is discarded
        t0 = time.perf_counter()
        _, timing = pipeline.generate(cfg)
        timing["wall"] = time.perf_counter() - t0
        if i:
            runs.append(timing)
    stats = {
        key: round(statistics.median(r[key] for r in runs), 3)
        for key in runs[0]
    } if runs else {}
    return {
        "measured": stats,
        "repeats": repeats,
        "checkpoint": checkpoint,
        "prompt": prompt,
        "shape": f"{width}x{height}x{frames}",
        "steps": steps,
        "quant": quant,
        "offload": offload,
    }


def project(profile: Profile, tflops: float) -> Dict[str, float]:
    """Projected wall clock for the DiT + VAE decode on an assumed GPU."""
    dit = dit_flops(profile) * profile.steps
    vae = vae_flops(profile)
    rate = tflops * 1e12
    return {
        "total_tflop": round((dit + vae) / 1e12, 2),
        "dit_tflop": round(dit / 1e12, 2),
        "vae_tflop": round(vae / 1e12, 2),
        "projected_denoise_s": round(dit / rate, 2),
        "projected_decode_s": round(vae / rate, 2),
        "projected_total_s": round((dit + vae) / rate, 2),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="NanoH3 benchmark")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default="a red circle moving to the right")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quant", default="none", choices=["none", "bf16", "int8", "fp8"])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--throughput", type=float, default=DEFAULT_GPU_TFLOPS, help="assumed effective TFLOP/s for projections")
    ap.add_argument("--profile", default="8gb", help="budget profile to project")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None, help="write the report to this path")
    args = ap.parse_args(argv)

    report: Dict[str, object] = {"device": device_report(), "assumed_tflops": args.throughput}
    report["VERIFIED"] = measure(
        args.checkpoint,
        args.prompt,
        args.width,
        args.height,
        args.frames,
        args.steps,
        args.seed,
        args.device,
        args.quant,
        args.repeats,
        args.offload,
    )
    profile = PROFILES[args.profile]
    report["PROJECTED"] = {
        "profile": profile.name,
        "shape": f"{profile.width}x{profile.height}x{profile.frames}",
        "steps": profile.steps,
        "assumed_effective_tflops": args.throughput,
        **project(profile, args.throughput),
        "vram_budget_gib": round(profile.estimate()["total"] / GIB, 2),
    }
    report["UNKNOWN"] = [
        "CUDA wall-clock on an 8 GB card (no NVIDIA GPU in this environment)",
        "quantised int8/fp8 speedup on CUDA (fake-quant dequantises per matmul here)",
        "offload bandwidth cost on PCIe (offload is a no-op on CPU/MPS)",
    ]

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

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

#: Assumed fixed cost of one denoising step on the target GPU: kernel launches
#: plus weight traffic, independent of how small the model is. This is the term
#: that actually binds at NanoH3's scale, and it is an assumption, not a
#: measurement. Override with --step-overhead-ms after measuring your own card.
DEFAULT_GPU_STEP_OVERHEAD_MS = 25.0


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


def per_step_overhead(measured_denoise_s: float, steps: int) -> float:
    """Seconds of fixed cost per denoising step on the machine that was measured.

    At these model sizes the arithmetic is essentially free -- a 4-step 256x256
    pass is ~2 TFLOP, which is milliseconds on any modern GPU. What actually
    costs time is launching the kernels and streaming the weights, and that cost
    does not shrink with the model. Measuring it here gives the floor that the
    projection has to respect.
    """
    return measured_denoise_s / max(1, steps)


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


def project(profile: Profile, tflops: float, step_overhead_s: float) -> Dict[str, float]:
    """Projected wall clock on an assumed GPU.

    Two terms, and the larger one wins:

    * **compute** -- counted MACs divided by the assumed effective throughput.
    * **overhead** -- ``steps x per-step fixed cost``. Kernel launches and weight
      traffic do not shrink with the model, and at this size they dominate.

    Reporting only the compute term would produce a number in the tens of
    milliseconds for a 4-step pass, which is not what a real run does: the local
    measurement below shows ~19 ms per step *at 48 tokens*, where the arithmetic
    is negligible. The projection takes the max so it cannot flatter itself.
    """
    dit = dit_flops(profile) * profile.steps
    vae = vae_flops(profile)
    rate = tflops * 1e12
    compute_denoise = dit / rate
    overhead_denoise = step_overhead_s * profile.steps
    compute_decode = vae / rate
    return {
        "total_tflop": round((dit + vae) / 1e12, 2),
        "dit_tflop": round(dit / 1e12, 2),
        "vae_tflop": round(vae / 1e12, 2),
        "compute_denoise_s": round(compute_denoise, 3),
        "overhead_denoise_s": round(overhead_denoise, 3),
        "denoise_s": round(max(compute_denoise, overhead_denoise), 3),
        "decode_s": round(compute_decode, 3),
        "total_s": round(max(compute_denoise, overhead_denoise) + compute_decode, 3),
        "binding_term": "overhead" if overhead_denoise > compute_denoise else "compute",
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
    ap.add_argument(
        "--step-overhead-ms",
        type=float,
        default=DEFAULT_GPU_STEP_OVERHEAD_MS,
        help="assumed fixed cost per denoising step on the target GPU (kernel launches, weight traffic)",
    )
    ap.add_argument("--profile", default="8gb", help="budget profile to project")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None, help="write the report to this path")
    args = ap.parse_args(argv)

    measured = measure(
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
    local_per_step = per_step_overhead(
        float(measured["measured"].get("denoise", 0.0)) if measured.get("measured") else 0.0,  # type: ignore[union-attr]
        args.steps,
    )
    report: Dict[str, object] = {
        "device": device_report(),
        "assumed_tflops": args.throughput,
        "assumed_step_overhead_ms": args.step_overhead_ms,
        "VERIFIED": measured,
    }
    report["VERIFIED_local_per_step_ms"] = round(local_per_step * 1000, 2)
    profile = PROFILES[args.profile]
    report["PROJECTED"] = {
        "profile": profile.name,
        "shape": f"{profile.width}x{profile.height}x{profile.frames}",
        "steps": profile.steps,
        "assumed_effective_tflops": args.throughput,
        "assumed_step_overhead_ms": args.step_overhead_ms,
        **project(profile, args.throughput, args.step_overhead_ms / 1000.0),
        "vram_budget_gib": round(profile.estimate()["total"] / GIB, 2),
    }
    report["UNKNOWN"] = [
        "CUDA wall-clock on an 8 GB card (no NVIDIA GPU in this environment)",
        "quantised int8/fp8 speedup on CUDA (fake-quant dequantises per matmul here)",
        "offload bandwidth cost on PCIe (offload is a no-op on CPU/MPS)",
        "the per-step overhead constant, which is the binding term: it is assumed, not measured",
    ]

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

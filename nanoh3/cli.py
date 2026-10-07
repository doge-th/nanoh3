"""Command line interface.

    python -m nanoh3 generate --checkpoint runs/demo/model.pt --prompt "a red circle ..." -o out.mp4
    python -m nanoh3 info --checkpoint runs/demo/model.pt
    python -m nanoh3 budget --profile 8gb
    python -m nanoh3 benchmark --checkpoint runs/demo/model.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import torch

from .engine.budget import PROFILES
from .engine.pipeline import GenerationConfig, NanoH3Pipeline, save_video
from .models import DiTConfig, NanoDiT, NanoVAE, TextEncoderConfig, VAEConfig


def _load_pipeline(args: argparse.Namespace) -> NanoH3Pipeline:
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" and str(device) != "cpu" else torch.float32
    return NanoH3Pipeline.from_checkpoint(args.checkpoint, device=device, quant=args.quant, dtype=dtype)


def cmd_generate(args: argparse.Namespace) -> int:
    pipeline = _load_pipeline(args)
    cfg = GenerationConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        frames=args.frames,
        steps=args.steps,
        guidance=args.guidance,
        shift=args.shift,
        seed=args.seed,
        fps=args.fps,
        offload=args.offload,
        offload_vae=args.offload_vae,
        decode_chunk=args.decode_chunk,
        active_depth=args.active_depth,
        progress=not args.quiet,
    )
    t0 = time.perf_counter()
    video, timing = pipeline.generate(cfg)
    out = Path(args.out)
    save_video(video, out, fps=args.fps)
    if not args.quiet:
        print(
            json.dumps(
                {
                    "output": str(out),
                    "shape": list(video.shape),
                    "seconds": round(time.perf_counter() - t0, 2),
                    "timing": {k: round(v, 3) for k, v in timing.items()},
                    "offload": pipeline.last_offload.summary() if pipeline.last_offload else "off",
                },
                indent=2,
            )
        )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dit = NanoDiT(DiTConfig(**state["dit_config"]))
    vae = NanoVAE(VAEConfig(**state["vae_config"]))
    text_cfg = TextEncoderConfig(**state["text_config"])
    print(
        json.dumps(
            {
                "params": {
                    "dit": dit.num_parameters(),
                    "vae": vae.num_parameters(),
                    "text": __import__("nanoh3.models", fromlist=["NanoTextEncoder"]).NanoTextEncoder(text_cfg).num_parameters(),
                },
                "dit_config": state["dit_config"],
                "vae_config": state["vae_config"],
                "text_config": state["text_config"],
                "meta": state.get("meta", {}),
            },
            indent=2,
        )
    )
    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    from .engine.budget import main as budget_main

    argv = ["--profile", args.profile]
    if args.json:
        argv.append("--json")
    return budget_main(argv)


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .engine.benchmark import main as bench_main

    argv = ["--checkpoint", args.checkpoint]
    if args.prompt:
        argv += ["--prompt", args.prompt]
    if args.offload:
        argv.append("--offload")
    return bench_main(argv)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nanoh3", description="NanoH3: small open text-to-video model")
    sub = ap.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", required=True)
    common.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    common.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    common.add_argument("--quant", default="none", choices=["none", "bf16", "int8", "fp8"])

    gen = sub.add_parser("generate", parents=[common], help="text -> mp4")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--negative-prompt", default="")
    gen.add_argument("-o", "--out", default="out.mp4")
    gen.add_argument("--width", type=int, default=256)
    gen.add_argument("--height", type=int, default=256)
    gen.add_argument("--frames", type=int, default=16)
    gen.add_argument("--steps", type=int, default=4)
    gen.add_argument("--guidance", type=float, default=1.0)
    gen.add_argument("--shift", type=float, default=3.0)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--fps", type=int, default=8)
    gen.add_argument("--offload", action="store_true", help="stream DiT blocks through host RAM")
    gen.add_argument("--offload-vae", action="store_true")
    gen.add_argument("--decode-chunk", type=int, default=0)
    gen.add_argument(
        "--active-depth",
        type=int,
        default=0,
        help="keep only the first N denoiser blocks (0 = all). Fewer blocks is linear speed and memory saving.",
    )
    gen.add_argument("--quiet", action="store_true")
    gen.set_defaults(func=cmd_generate)

    info = sub.add_parser("info", parents=[common], help="print model sizes and config")
    info.set_defaults(func=cmd_info)

    budget = sub.add_parser("budget", help="analytic VRAM budget")
    budget.add_argument("--profile", default="8gb", choices=list(PROFILES) + ["all"])
    budget.add_argument("--json", action="store_true")
    budget.set_defaults(func=cmd_budget)

    bench = sub.add_parser("benchmark", parents=[common], help="measure stage timings")
    bench.add_argument("--prompt", default=None)
    bench.add_argument("--offload", action="store_true")
    bench.set_defaults(func=cmd_benchmark)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Analytic VRAM / host-RAM budget.

The point of this module is that the 8 GB claim is *checked by code*, not asserted
in a README. Every term below is either an exact tensor byte count or a documented
upper bound on one; ``assert_fits`` turns the sum into a pass/fail so a config that
would not fit cannot be shipped quietly.

Run standalone::

    python -m nanoh3.engine.budget --profile 8gb
    python -m nanoh3.engine.budget --profile 8gb --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MIB = 1024 * 1024
GIB = 1024 * MIB


@dataclass
class Profile:
    """A generation request plus the knobs that change its memory profile."""

    name: str
    width: int = 256
    height: int = 256
    frames: int = 16
    batch: int = 1
    dit_params: int = 320_000_000
    text_params: int = 8_000_000
    vae_params: int = 24_000_000
    latent_channels: int = 16
    patch_size: int = 2
    hidden_size: int = 768
    dit_depth: int = 24
    vae_base_channels: int = 48
    text_len: int = 77
    quant: str = "int8"  # none | bf16 | int8 | fp8
    compute_dtype: str = "bf16"  # dtype of activations
    steps: int = 4
    active_depth: int = 0  # 0 = all blocks
    resident_blocks: int = 2  # blocks kept in VRAM when offload is on
    offload_dit: bool = True
    offload_vae: bool = True
    vae_chunk_frames: int = 8  # temporal chunking of the decoder
    latent_upscale: int = 1  # optional 2x latent refine pass
    notes: List[str] = field(default_factory=list)

    # ---- derived geometry ----
    @property
    def spatial_compression(self) -> int:
        return 8

    @property
    def temporal_compression(self) -> int:
        return 4

    @property
    def latent_shape(self) -> tuple:
        c = self.temporal_compression
        s = self.spatial_compression
        return (-(-self.frames // c), -(-self.height // s), -(-self.width // s))

    @property
    def token_grid(self) -> tuple:
        f, h, w = self.latent_shape
        p = self.patch_size
        return (f, h // p, w // p)

    @property
    def num_tokens(self) -> int:
        f, h, w = self.token_grid
        return f * h * w

    @property
    def weight_bytes(self) -> float:
        table = {"none": 4.0, "bf16": 2.0, "int8": 1.0, "fp8": 1.0}
        return (self.dit_params + self.text_params + self.vae_params) * table[self.quant]

    # ---- terms ----
    def dit_weight_bytes(self) -> float:
        """Bytes of denoiser weights that must be resident in VRAM.

        With offload on, only ``resident_blocks`` blocks stay resident plus one
        block in flight on the copy stream; the rest lives in host RAM. When
        ``active_depth`` is set (the speed profile) only those blocks exist at all.
        """
        table = {"none": 4.0, "bf16": 2.0, "int8": 1.0, "fp8": 1.0}
        total = self.dit_params * table[self.quant]
        if self.offload_dit and self.resident_blocks > 0:
            blocks = max(1, min(self.active_depth or self.dit_depth, self.dit_depth))
            resident = total * min(1.0, self.resident_blocks / blocks)
            return resident + total / blocks
        return total

    def dit_activation_bytes(self) -> float:
        """Flash-attention forward peak. Coefficients are documented per-term."""
        act = 2 if self.compute_dtype == "bf16" else 4
        n, d = self.num_tokens * self.batch, self.hidden_size
        per_token_resident = act * d * 4  # qkv(3d) + attn out(d) + residual(q,k,v slices) ~
        mlp = act * d * 4 * 4  # hidden expansion in the MLP
        cross = act * self.text_len * d * 2  # k,v of the text cross-attention
        return n * (per_token_resident + mlp) + self.batch * cross

    def vae_bytes(self) -> float:
        act = 2 if self.compute_dtype == "bf16" else 4
        ch = self.vae_base_channels
        frames = min(self.vae_chunk_frames, self.frames)
        # Decoder activations dominate: chunked decode keeps the peak at `frames`.
        dec = self.batch * frames * self.height * self.width * ch * act * 6
        if self.offload_vae:
            dec = min(dec, self.batch * self.vae_chunk_frames * self.height * self.width * ch * act * 6)
        enc = self.batch * frames * self.height * self.width * ch * act * 3
        weights = self.vae_params * (1.0 if self.quant in ("int8", "fp8") else 2.0)
        return dec + enc + weights

    def latent_bytes(self) -> float:
        act = 2 if self.compute_dtype == "bf16" else 4
        f, h, w = self.latent_shape
        latent = self.batch * self.latent_channels * f * h * w * act
        # two denoising states (x_t and the model's output) plus a noise buffer
        return latent * 3

    def decode_output_bytes(self) -> float:
        return self.batch * self.frames * self.height * self.width * 3 * 4

    def workspace_bytes(self) -> float:
        """CUDA context + kernel scratch, a fixed floor for any torch CUDA process."""
        return 900 * MIB

    def estimate(self) -> Dict[str, float]:
        terms = {
            "dit_weights": self.dit_weight_bytes(),
            "dit_activations": self.dit_activation_bytes(),
            "vae": self.vae_bytes(),
            "latents": self.latent_bytes(),
            "decode_output": self.decode_output_bytes(),
            "workspace": self.workspace_bytes(),
        }
        terms["total"] = sum(terms.values())
        terms["headroom_8gib"] = 8 * GIB - terms["total"]
        return terms

    def host_ram_bytes(self) -> float:
        """System RAM needed: weights that do not fit in VRAM, plus staging buffers."""
        resident = self.dit_weight_bytes() if self.offload_dit else self.weight_bytes
        return max(0.0, self.weight_bytes - resident + self.dit_activation_bytes() * 2)


PROFILES: Dict[str, Profile] = {
    "8gb": Profile(
        name="8gb-reference",
        width=256,
        height=256,
        frames=16,
        dit_params=320_000_000,
        text_params=8_000_000,
        vae_params=24_000_000,
        hidden_size=768,
        quant="int8",
        compute_dtype="bf16",
        steps=4,
        active_depth=0,
        resident_blocks=2,
        offload_dit=True,
        offload_vae=True,
        vae_chunk_frames=4,
        notes=["4-step sampler", "int8 weights", "2 resident DiT blocks", "chunked VAE decode"],
    ),
    "8gb-tiny": Profile(
        name="8gb-tiny",
        width=256,
        height=256,
        frames=16,
        dit_params=130_000_000,
        text_params=6_000_000,
        vae_params=12_000_000,
        hidden_size=512,
        dit_depth=20,
        quant="int8",
        compute_dtype="bf16",
        steps=4,
        active_depth=0,
        resident_blocks=4,
        vae_chunk_frames=4,
        notes=["smallest honest configuration"],
    ),
    "8gb-512": Profile(
        name="8gb-512p",
        width=512,
        height=288,
        frames=24,
        dit_params=320_000_000,
        hidden_size=768,
        quant="int8",
        steps=4,
        resident_blocks=2,
        vae_chunk_frames=2,
        notes=["16:9 at roughly 512x288"],
    ),
    "24gb": Profile(
        name="24gb-comfy",
        width=512,
        height=288,
        frames=24,
        dit_params=320_000_000,
        hidden_size=768,
        quant="bf16",
        compute_dtype="bf16",
        steps=8,
        offload_dit=False,
        offload_vae=False,
        notes=["no offload, bf16, 8 steps"],
    ),
    "demo": Profile(
        name="local-demo",
        width=128,
        height=128,
        frames=9,
        dit_params=12_000_000,
        text_params=3_000_000,
        vae_params=2_000_000,
        hidden_size=192,
        dit_depth=8,
        quant="none",
        compute_dtype="fp32",
        steps=8,
        offload_dit=False,
        offload_vae=False,
        notes=["runs on a laptop CPU/MPS in seconds"],
    ),
}


def assert_fits(profile: Profile, limit_bytes: int = 8 * GIB) -> Dict[str, float]:
    """Raise if the profile exceeds ``limit_bytes``; return the estimate either way."""
    est = profile.estimate()
    if est["total"] > limit_bytes:
        raise MemoryError(
            f"profile {profile.name} needs {est['total'] / GIB:.2f} GiB > {limit_bytes / GIB:.2f} GiB limit; "
            f"largest term: {max((k for k in est if k not in ('total', 'headroom_8gib')), key=lambda k: est[k])}"
        )
    return est


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="NanoH3 memory budget")
    ap.add_argument("--profile", default="8gb", help="profile name or 'all'")
    ap.add_argument("--limit-gib", type=float, default=8.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--quant", choices=["none", "bf16", "int8", "fp8"])
    args = ap.parse_args(argv)

    names = list(PROFILES) if args.profile == "all" else [args.profile]
    report = {}
    failures = 0
    for name in names:
        if name not in PROFILES:
            ap.error(f"unknown profile {name!r}; known: {', '.join(PROFILES)}")
        profile = PROFILES[name]
        for attr in ("frames", "width", "height"):
            value = getattr(args, attr)
            if value:
                setattr(profile, attr, value)
        if args.quant:
            profile.quant = args.quant
        est = profile.estimate()
        fits = est["total"] <= args.limit_gib * GIB
        failures += 0 if fits else 1
        report[name] = {
            "fits_limit": fits,
            "shape": f"{profile.width}x{profile.height}x{profile.frames}",
            "steps": profile.steps,
            "gib": {k: round(v / GIB, 3) for k, v in est.items() if k != "headroom_8gib"},
            "host_ram_gib": round(profile.host_ram_bytes() / GIB, 2),
            "notes": profile.notes,
        }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for name, row in report.items():
            mark = "PASS" if row["fits_limit"] else "FAIL"
            print(f"[{mark}] {name:12s} {row['shape']:>16s} {row['steps']} steps  total={row['gib']['total']:.2f} GiB")
            for key, value in row["gib"].items():
                if key != "total":
                    print(f"           {key:16s} {value:6.3f} GiB")
            print(f"           host RAM        {row['host_ram_gib']:6.2f} GiB   {'; '.join(row['notes'])}")
            print()
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["Profile", "PROFILES", "assert_fits", "MIB", "GIB"]

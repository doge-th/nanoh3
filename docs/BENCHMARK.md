# Benchmarks

Every number here is tagged. That is the whole point of this file: the two
targets (8 GB, under a minute) were set on a machine with **no NVIDIA GPU**, and
the difference between "measured" and "projected" is exactly where the honest
part of this project lives.

Reproduce everything here with:

```bash
python -m nanoh3.engine.benchmark --checkpoint <ckpt> --profile 8gb --out bench.json
```

---

## 1. The measurement machine

| | |
|---|---|
| Hardware | Apple M5, 8-core GPU, 16 GB unified memory, **no CUDA** |
| OS / runtime | macOS 26.5.1 arm64, Python 3.11.15, PyTorch 2.14.1 (MPS backend) |
| What this means | every CUDA number below is a projection; every MPS number is measured |

## 2. VERIFIED — measured on this machine

Demo model, 9.7 M parameters total (2.7 M VAE + 6.5 M denoiser + 0.4 M text),
`runs/demo/model.pt`, 128×128×9, 8 sampling steps, fp32:

| Stage | Median of 3 runs after a discarded warm-up |
|---|---|
| Text encode | 0.145 s |
| Denoise (8 steps) | 0.154 s |
| VAE decode | 0.003 s |
| **Total wall clock** | **0.303 s** |

Command:

```bash
python -m nanoh3.engine.benchmark --checkpoint runs/demo/model.pt \
  --width 128 --height 128 --frames 9 --steps 8 --device mps --repeats 3
```

**Per-step floor: 19 ms.** Denoising 8 steps at 48 latent tokens takes 154 ms, and
the arithmetic involved is on the order of 0.01 TFLOP — the 19 ms per step is
overhead, not compute. This is the single most important measured fact in the
project: **at NanoH3's scale, wall clock is dominated by fixed per-step cost, not
by arithmetic.**

## 3. PROJECTED — the 8 GB reference profile

Profile `8gb`: 256×256×16, 4 steps, int8 weights, 2 resident denoiser blocks,
chunked VAE decode. Assumptions, both adjustable and both printed in the output:

| Assumption | Value | Where it comes from |
|---|---|---|
| Effective matmul throughput | **100 TFLOP/s** | a mid-range 8 GB card at ~50 % of its dense bf16 peak; `--throughput` |
| Fixed cost per denoising step | **25 ms** | one block more than the 19 ms measured on MPS, for more kernels per step; `--step-overhead-ms` |

| Quantity | Value |
|---|---|
| Counted work (denoiser, 4 steps) | 2.16 TFLOP |
| Counted work (VAE decode) | 0.01 TFLOP |
| Compute term | 0.022 s |
| Overhead term (4 × 25 ms) | 0.100 s |
| **Projected total** | **≈ 0.10 s** |
| Binding term | **overhead**, not compute |
| Peak VRAM (budget) | **1.19 GiB** of 8 GiB |

Read that as: the arithmetic is trivial, the wall clock is per-step overhead, and
**the 60-second target has enormous headroom on this model size.** The honest
caveat is the 25 ms constant — it is an assumption, and it is the number that
would be wrong if we are wrong.

## 4. UNKNOWN — not measured, and why

| Unknown | Why |
|---|---|
| CUDA wall clock on an 8 GB card | no NVIDIA GPU exists in this environment |
| Speedup from native int8/fp8 matmuls | quantisation here is weight-only fake-quant: real 1-byte storage, dequantised per matmul. Numerically right, computationally slower than a fused kernel. |
| PCIe cost of block offload | offload is deliberately a no-op on CPU/MPS, so the path could not be timed |
| The 25 ms/step overhead constant | assumed, not measured. It is the binding term, which makes it the most important number in this file. |
| Multi-GPU / any-scale-out behaviour | not implemented |

## 5. For reference: what the target model costs

Measured on an 8 GB card by the community, for comparison. Sources in
[`RESEARCH-H3.md`](RESEARCH-H3.md) §3.

| Model | Clip | Steps | Wall clock |
|---|---|---|---|
| MiniMax-H3, pruned int8 + NVFP4 text encoder, RTX 5060 Laptop 8 GB | 768×1024, 243 frames (10 s) | 4 | ≈ 9 min |
| same | same | 8 | **14.3 min** |
| same | same | 20 | 33.8 – 43.5 min |
| MiniMax-H3, int8, RTX 4090 24 GB | 864×480, 124 frames (5 s) | 20 | 92 – 93 s |
| MiniMax-H3, int8 + 4-step turbo, RTX 5090 32 GB | 1344×768, 56 frames | 4 | 16.2 s |
| **NanoH3, `8gb` profile** | 256×256, 16 frames | 4 | **≈ 0.1 s (projected)** |

The comparison is not apples to apples and should not be read as one: H3 at
8 steps generates ten seconds of 768p video with audio, NanoH3's projection is
for two seconds of 256p video with no audio and a model that has not been trained
on real footage. What the table does show is that the 8 GB + sub-minute target is
unreachable for H3 by a wide margin, and reachable for a model this size with
room to spare.

## 6. Memory budget

`python -m nanoh3.engine.budget --profile all` — every term is computed from the
model shape, and `assert_fits()` turns the sum into a pass/fail that CI enforces.

| Profile | Shape | Steps | Total | Dominant term |
|---|---|---|---|---|
| `8gb` | 256×256×16 | 4 | **1.19 GiB** | VAE decode + 0.88 GiB CUDA context floor |
| `8gb-tiny` | 256×256×16 | 4 | 1.16 GiB | same |
| `8gb-512` | 512×288×24 | 4 | 1.32 GiB | denoiser activations |
| `24gb` | 512×288×24 | 8 | 2.61 GiB | bf16 weights, no offload |

The 0.88 GiB `workspace` term is the CUDA context and kernel scratch, a fixed
floor for any torch CUDA process. It is why the small profiles do not get
proportionally smaller: at 1.19 GiB total, three quarters of the budget is the
floor plus the decoder, not the denoiser.

## 7. What would change these numbers

1. **Training the `8gb` config on real video** — the only thing missing between
   this and a usable model. It would not change the memory budget.
2. **A fused int8 kernel** (`torch._scaled_mm`, Marlin, bitsandbytes) instead of
   fake-quant. Would cut the compute term, which is already not binding.
3. **`torch.compile` / CUDA graphs** — the binding term is per-step overhead, so
   removing launch overhead is the only lever that moves the number. Expect the
   25 ms to fall; do not expect the compute term to matter.
4. **Real hardware** — replace §3 with a measurement and delete the assumptions.
   That is the honest end state of this file.

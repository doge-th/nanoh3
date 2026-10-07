# Running NanoH3 on an 8 GB card

Everything here is written for a single NVIDIA GPU with 8 GB of VRAM (RTX 3050 /
4060 / 5060 class). The important part is that **8 GB of VRAM is not the
constraint for this model** — the reference profile needs 1.29 GiB — so the
runbook is mostly about the constraint that *is* real: getting a trained
checkpoint, and not wasting time on the sampler.

---

## 0. Before anything else: you need weights

This repository ships code, not a trained video model. The demo checkpoint that
`configs/demo.yaml` produces learns coloured shapes; it will not generate
photoreal footage. To get a usable model you have to train one:

| Stage | Command | Time |
|---|---|---|
| Autoencoder | `python -m nanoh3.training.train --config configs/8gb.yaml --stage vae` | ~1 GPU-day on a 24 GB card |
| Denoiser | `python -m nanoh3.training.train --config configs/8gb.yaml --stage dit` | ~3–5 GPU-days |
| Few-step distill | `python -m nanoh3.training.distill --config configs/8gb.yaml --input runs/8gb/model.pt` | ~6–12 GPU-hours |

Training needs more than 8 GB (24 GB is the comfortable floor for the 8 GB
config), so rent for training and run the result on the 8 GB card you have.

---

## 1. Install

```bash
git clone <repo> && cd nanoh3
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

PyTorch with CUDA 12.4+ for a Blackwell card, 12.1 for Ada, 11.8 for Ampere:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify the environment sees the card:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 2. Generate

The default invocation is already the 8 GB profile:

```bash
python -m nanoh3 generate \
  --checkpoint runs/8gb/model.pt \
  --prompt "a paper boat drifting on a calm river at sunrise" \
  -o boat.mp4 \
  --width 256 --height 256 --frames 16 --steps 4 \
  --quant int8 --offload --decode-chunk 4
```

| Flag | Why |
|---|---|
| `--quant int8` | weights to 1 byte each. Measured round-trip error < 0.02 absolute. |
| `--steps 4` | the post-distillation schedule. 4 steps vs H3's 20–50 is a 5–12× compute cut. |
| `--offload` | streams denoiser blocks through host RAM. Unnecessary at 1.29 GiB, cheap insurance if you raise resolution. |
| `--decode-chunk 4` | caps VAE decoder activations at 4 latent frames. The decoder is the largest activation consumer. |
| `--device cuda --dtype bf16` | explicit is faster than `auto` on a first run, because you see the device it chose. |

### If you run out of memory, in this order

1. `--decode-chunk 2` — halves the decoder's peak activation. Usually enough.
2. `--width 256 --height 144` — token count is what dominates the denoiser.
3. `--frames 12` — temporal cost is linear in frames.
4. `--offload` — moves the denoiser to host RAM block by block.
5. `--quant int8` — if you were running fp32/bf16.

Check the budget before you run, rather than after an OOM:

```bash
python -m nanoh3.engine.budget --profile 8gb --frames 24 --width 384 --height 216
```

It exits non-zero if the configuration does not fit.

---

## 3. Measure on your own card

The projections in `BENCHMARK.md` were computed without an NVIDIA GPU. Replace
them with real numbers:

```bash
python -m nanoh3.engine.benchmark \
  --checkpoint runs/8gb/model.pt \
  --width 256 --height 256 --frames 16 --steps 4 \
  --quant int8 --repeats 5 \
  --throughput <your measured TFLOP/s> \
  --out docs/BENCHMARK.local.md
```

Watch for the usual failure modes on small cards:

- **Host RAM, not VRAM.** If the process is paging, the answer is more system
  RAM (32 GB minimum, 64 GB comfortable) or a smaller clip — not a smaller
  quantisation step. This is the same lesson the H3 community learned: on an
  8 GB card running a streamed 20 GB denoiser, the wall was 15 GB of system RAM.
- **Thermal throttling.** A laptop GPU at 30 W with 99 % utilisation is
  throttling, not working. Compare `nvidia-smi` power draw against a known-good
  run before concluding the model is slow.
- **First call is not representative.** The harness discards a warm-up run; if
  you time it by hand, throw the first one away.

---

## 4. Trading speed for quality

The knobs, roughly in order of how much they cost per unit of time saved:

| Knob | Effect | Cost |
|---|---|---|
| `--steps 4` → 2 | 2× faster | needs a 2-step distillation run; quality drops visibly |
| `--active-depth` (API) | linear in blocks removed | a 24-block model at depth 16 is ~1.5× faster |
| `--frames` | linear | the strongest quality knob: 16 frames is 2 s at 8 fps, 24 is 3 s |
| `--quant fp8` | same speed here, half the bytes | fp8 error is ~10× int8; use on Ada/Hopper/Blackwell only |

There is no CFG to disable — like H3, the recommended model is
guidance-distilled and runs one forward pass per step.

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `assert_fits` raises in CI | a profile exceeds 8 GiB | read the largest term it prints; it names the offending component |
| CUDA OOM during `decode` | decoder activations | `--decode-chunk 2` |
| CUDA OOM at the first denoising step | something is not quantised | check `--quant` actually took effect with `nanoh3 info` |
| Video is noise | checkpoint is undertrained, or `latent_mean`/`latent_std` do not match the VAE | the stats live in the checkpoint's `meta.latent_stats`; re-run `--stage dit` after changing the VAE |
| Video is static / no motion | the sampler is running too few steps for the schedule | try `--steps 8` before assuming the model is broken |
| Prompt has no effect | the text tower is the demo one, not CLIP | expected; see README limitation 5 |
| macOS `ModuleNotFoundError: No module named 'encodings'` | something exported `PYTHONHOME`/`PYTHONPATH` | run `unset PYTHONHOME PYTHONPATH` first |

---

## 6. What this runbook cannot tell you

The one number everyone wants — **"how many seconds will my clip take on my
card?"** — is not in here, because it was written on a machine with no NVIDIA
GPU. `BENCHMARK.md` has the projection and the arithmetic; run §3 and replace it
with a measurement. If you do, the projection is wrong somewhere, and the
measurement is the thing to trust.

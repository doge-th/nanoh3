# NanoH3

A small, fully open text-to-video model for people with an 8 GB graphics card.

Independent implementation. Not affiliated with MiniMax. Contains no MiniMax
weights, no MiniMax code, and no data derived from MiniMax outputs.

```bash
git clone <this repo> && cd nanoh3
pip install torch numpy pillow imageio imageio-ffmpeg pyyaml pytest

# train the demonstration model and generate a video (CPU or Apple Silicon, ~30 min)
python -m nanoh3.training.train --config configs/demo.yaml
python -m nanoh3 generate --checkpoint runs/demo/model.pt \
    --prompt "a red circle on a grey background that moves to the right" \
    -o out.mp4 --width 128 --height 128 --frames 9 --steps 8
```

Shorten the wait while you are reading the code — `--steps N` overrides both stage
counts, and a 60-step run is enough to see loss come down:

```bash
python -m nanoh3.training.train --config configs/demo.yaml --steps 60
```

---

## Why this exists

[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) is a 33 B-parameter
video model whose released stack is **144 GB for one text-to-video workflow** —
61.7 GiB of denoiser in bf16 plus a 62.1 GiB Qwen3-VL-32B text encoder. It is not
open source; it ships under a community licence that excludes the EU, UK, South
Korea and the USA and that **forbids using its outputs to improve any other AI
model**.

The measurements that shaped this project:

- On a real 8 GB card, H3 with the pruned int8 denoiser and an NVFP4 text encoder
  takes **≈9 minutes at 4 steps and 14.3 minutes at 8 steps** for a 10-second
  clip at 768×1024 — the denoiser does not fit in VRAM at all and is streamed
  from host RAM every step.
- So 8 GB is *reachable* for H3, and **sub-minute generation is not**. No published
  technique closes a 30× gap.

NanoH3 takes the same public architectural vocabulary — causal 3D VAE, rectified
flow, 3D RoPE, adaLN-Zero, guidance distillation — and builds it at a size where
the whole stack, activations included, fits in **about 1.2 GiB**, leaving the
whole 8 GB budget available. Full research notes with sources:
[`docs/RESEARCH-H3.md`](docs/RESEARCH-H3.md).

**What you get is not H3's quality.** You get a model whose weights, training
code, data pipeline and licence are all yours to read, change and ship.

---

## The two targets, and how honestly we report them

| Target | Status | Evidence |
|---|---|---|
| Fits in 8 GB VRAM | ✅ **verified by assertion** | `python -m nanoh3.engine.budget --profile 8gb` computes the peak and fails if it exceeds 8 GiB; `tests/test_model.py::test_8gb_profile_fits_8gib` asserts the same. Reference profile: **1.19 GiB**. |
| Generates in under a minute | ⚠️ **target, not a measured 8 GB result** | No NVIDIA GPU was available during development. See [`docs/BENCHMARK.md`](docs/BENCHMARK.md) for the projection, its arithmetic and its assumptions. |

Anything this project cannot prove on the hardware it was built on is labelled
`UNKNOWN` rather than rounded into a claim. That includes CUDA wall-clock, the
speedup from native int8/fp8 matmuls, and the PCIe cost of offload — all three
are real, all three are unmeasured here.

---

## What is in the box

```
nanoh3/
  models/       causal 3D VAE, video DiT, 3D RoPE, byte-level text tower
  engine/       int8/fp8 quantisation, block offload, memory budget, few-step sampler, pipeline
  data/         procedural clips; reference-clip preparation with a licence guard
  training/     two-stage trainer, few-step distillation
  tests/        31 tests: shapes, causality, determinism, quantisation error, offload, budget
docs/
  RESEARCH-H3.md   the MiniMax-H3 study this design came out of
  DESIGN.md        architecture and the reasoning behind each choice
  RUNBOOK-8GB.md   how to actually run this on an 8 GB card
  BENCHMARK.md     measured vs projected, with assumptions
  ACCEPTANCE.md    the criteria this project agreed to be judged against
```

### The memory budget is code, not a claim

```bash
$ python -m nanoh3.engine.budget --profile all
[PASS] 8gb                256x256x16 4 steps  total=1.19 GiB
           dit_weights       0.037 GiB
           dit_activations   0.030 GiB
           vae               0.233 GiB
           workspace         0.879 GiB
...
```

`assert_fits()` raises if any profile crosses the limit, and the test suite calls
it, so a config that would not fit cannot ship quietly.

### The VAE is strictly causal, and that is tested

```bash
$ python -m pytest tests/test_model.py::test_vae_is_causal -q
1 passed
```

Appending arbitrary future frames leaves every earlier latent unchanged to float32
precision (measured residual ~1e-7; the first latent frame is exactly equal).
Getting there required two non-obvious fixes that are worth knowing about if you
extend this code: left-only padding of `(k-1)·dilation` (which gives causality
*and* `ceil(L/stride)` output length), and a per-frame group norm — plain
`nn.GroupNorm` normalises over the time axis and silently leaks the whole clip.

---

## Training

Two stages, because the denoiser only ever sees normalised latents:

```bash
# stage 1: autoencoder
python -m nanoh3.training.train --config configs/8gb.yaml --stage vae

# stage 2: denoiser on frozen latents
python -m nanoh3.training.train --config configs/8gb.yaml --stage dit

# stage 3: collapse the schedule to 2-4 steps  (this is the speed lever)
python -m nanoh3.training.distill --config configs/8gb.yaml --input runs/8gb/model.pt
```

| Config | What it is |
|---|---|
| `configs/demo.yaml` | ~10 M parameters, trains in minutes on a laptop. Proves the pipeline. Not a quality model. |
| `configs/8gb.yaml` | ~300 M denoiser + VAE + text tower. The 8 GB reference. |
| `configs/h3_matched.yaml` | 16× spatial / 4× temporal VAE with 24 latent channels — the same token grid as H3, for controlled ablations. |

### Data

`configs/demo.yaml` trains on procedurally generated clips (coloured shapes with
known motion), because every factor is known and the text conditioning has to
actually work for the run to mean anything.

For real data, `scripts/prepare_teacher_data.py` drives any external generator as
a subprocess and keeps only the pixels:

```bash
python scripts/prepare_teacher_data.py --prompts-file prompts.txt --out data/teacher
```

It **refuses MiniMax-H3 by default**, because licence section V.3 forbids using
its output to improve another model:

```
$ python scripts/prepare_teacher_data.py --repo MiniMaxAI/MiniMax-H3
licence guard: refusing to use 'MiniMaxAI/MiniMax-H3' as a data source: ...
```

The default speed path needs no teacher at all: `distill.py` collapses a
long-schedule model of ours into a 2–4 step one.

---

## Limitations

Stated plainly, because you will hit them:

1. **The shipped demo model is not a good video generator.** It learns coloured
   shapes on flat backgrounds. It exists to prove the pipeline end to end.
2. **No quality model was trained.** Training the 8 GB config properly is
   GPU-days on real video data. The config, the trainer and the data pipeline are
   delivered and tested; the weights are not, because the hardware was not
   available.
3. **The sub-minute target is a projection.** Compute it yourself with
   `python -m nanoh3.engine.benchmark --checkpoint <ckpt> --throughput <TFLOPs>`.
4. **Quantisation is weight-only fake-quant.** It stores real int8/fp8 tensors
   (so the memory saving is real) and dequantises per matmul, which is correct
   everywhere but gives up the native-kernel speedup on Hopper/Ada/Blackwell.
5. **Text conditioning is a small byte-level transformer**, not a trained
   CLIP/Qwen. Swap it by loading a different encoder that returns
   `(B, L, text_dim)` — the denoiser does not know the difference.
6. **English-ish prompts only** in the demo, because the dataset is.
7. **Video only, no audio.** H3 generates audio in the same pass; we do not.
8. **The demo VAE is not strictly causal.** `norm_mode: clip` plus
   `causal_decoder: false` are what make motion survive; the strictly causal
   configuration is still implemented, tested, and available, but it produces
   still frames. The trade-off is measured in
   [`docs/DESIGN.md`](docs/DESIGN.md) §"Getting the clip to move".
9. **Demo motion is coarse.** A 9-frame clip is 3 latent frames, so the model
   learns "the object is elsewhere now" rather than a smooth trajectory. Real
   quality needs longer clips and real video.

---

## Licence

Apache-2.0 — see [`LICENSE`](LICENSE).

You are responsible for the licence of any data or model you point the data
pipeline at. `docs/RESEARCH-H3.md` §4 covers the H3 licence terms specifically,
because they are the reason the obvious approach is closed to us.

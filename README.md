# NanoH3

**A research and engineering project, not a working video generator. Please read
this before you install anything.**

The question I set out to answer was: *can you get sub-minute text-to-video on
an 8 GB card?* The honest answer turned out to be two different questions, with
two different answers:

1. **Can MiniMax-H3 run on 8 GB at all?** Yes — and the ecosystem's blog posts
   about it are substantially wrong about the cost. Measured on a real 8 GB card,
   a 10-second clip takes **9–14 minutes**, not seconds. The research is in
   [`docs/RESEARCH-H3.md`](docs/RESEARCH-H3.md) and every claim is sourced.
2. **Can you build a small open model that does?** I built one, and then spent a
   long time failing to make it animate. That failure is documented precisely,
   because it is the more useful half of the result.

**The demo model does not produce meaningful video.** It renders coloured shapes
on flat backgrounds, and it holds them still — the motion ratio scores 0.007
against a 0.5 pass mark, measured by
[`scripts/eval_demo.py`](scripts/eval_demo.py). Do not use this for anything.
There are no pretrained weights in this repository and none are published.

Independent implementation. Not affiliated with MiniMax. No MiniMax weights, no
MiniMax code, and — deliberately — no data derived from MiniMax outputs, because
their licence forbids it ([§V.3](docs/RESEARCH-H3.md#4-the-licence-finding--this-one-changes-the-plan)).

```bash
git clone <this repo> && cd nanoh3
pip install torch numpy pillow imageio imageio-ffmpeg pyyaml pytest

# trains on CPU or Apple Silicon, ~30 min, and writes a playable mp4
python -m nanoh3.training.train --config configs/demo.yaml
python -m nanoh3 generate --checkpoint runs/demo/model.pt \
    --prompt "a red square on a cream background that sweeps across the frame" \
    -o out.mp4 --width 128 --height 128 --frames 9 --steps 8

# score any checkpoint against the acceptance criteria; exits non-zero on failure
python scripts/eval_demo.py --checkpoint runs/demo/model.pt
```

---

## What is actually worth reading here

Three things, in descending order of value to someone who was not me:

**1. `docs/RESEARCH-H3.md` — the MiniMax-H3 study.**
Architecture from the published configs (33 B dense DiT, 61.7 GiB of it in bf16,
a 62.1 GiB Qwen3-VL-32B text encoder), the lowest-resolution preset anyone has
actually measured, the consumer-GPU cost table, the quantisation/LoRA ecosystem
map — and the licence clause that closes off the obvious distillation path.

**2. The measured wall in `docs/DESIGN.md`.**
Four separate causes of a frozen video clip, each isolated with a controlled
experiment, and then an 11-configuration matrix showing that good reconstruction
and real motion are simply not both reachable in a 2.7 M-parameter VAE at 64×64.
Negative results are rare enough to be worth keeping.

**3. The code**, which is competent, tested (33 tests) and readable. It is also a
reimplementation of recipes that already exist in Wan, LTX and Flux, so treat it
as a teaching artifact rather than something to depend on.

## The interesting engineering result

Making a video autoencoder actually move took four fixes, and the decisive one
was not a bug at all. A temporal loss computed uniformly over pixels is *inert*:
more than 90 % of a typical video's pixels sit on a static background, where the
reconstruction error is exactly zero and the L1 subgradient is exactly zero, so
the gradient that could teach motion is diluted away. Measured gradient norms on
one batch: L1 → 106, temporal term → 0.59.

Weighting each pixel by how much it actually moves is what changed the outcome.
Same seed, 25 steps, frame-to-frame delta of the reconstruction:

| Temporal objective | delta |
|---|---|
| L1 | 0.00000 |
| MSE | 0.00000 |
| Charbonnier | 0.00000 |
| **motion-weighted** | **0.32210** |

End to end, the generated clip went from a frame delta of `[0, 0, 0, 5e-5 …]` to
`[4.6e-4, 1.2e-3, 1.5e-3 …]`. The motion mechanism works. The picture is still
not good enough, and the reason is capacity, documented in
[`docs/DESIGN.md`](docs/DESIGN.md) with the full matrix.

## The two hard numbers

| Claim | Status |
|---|---|
| Fits in 8 GB VRAM | ✅ verified by assertion — `python -m nanoh3.engine.budget --profile 8gb` computes the peak and exits non-zero if it exceeds 8 GiB. Reference profile: **1.19 GiB**. |
| Generates in under a minute | ❌ **not verified.** No NVIDIA GPU was available. `docs/BENCHMARK.md` labels every number `VERIFIED` / `PROJECTED` / `UNKNOWN` and prints the assumptions. |

Worth saying plainly: a 300 M model fitting in 1.2 GB is not an achievement, it
is arithmetic. The interesting part of this repository is the research, not the
footprint.

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
  tests/        33 tests: shapes, causality, determinism, quantisation error, offload, budget
docs/
  RESEARCH-H3.md   the MiniMax-H3 study this design came out of
  DESIGN.md        architecture, the four frozen-clip bugs, and the measured wall
  RUNBOOK-8GB.md   how to actually run this on an 8 GB card
  BENCHMARK.md     measured vs projected, with assumptions
  ACCEPTANCE.md    the criteria this project agreed to be judged against
scripts/
  eval_demo.py       scores a checkpoint against the acceptance criteria
  prepare_teacher_data.py  reference-clip pipeline, with the licence guard
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

1. **The demo model does not move.** This is the big one. It draws coloured
   shapes and holds them still: motion ratio **0.007** against a 0.5 pass mark.
   The mechanism is fixed and proven — the failure is capacity, documented in
   [`docs/DESIGN.md`](docs/DESIGN.md). Treat this repository as research, not as
   a video generator.
2. **No pretrained weights are published.** Training the 8 GB config properly is
   GPU-days on real video. The config, the trainer, the data pipeline and the
   distillation are delivered and tested; the weights are not, because the
   hardware was not available.
3. **The sub-minute target is unverified.** No NVIDIA GPU was available at any
   point. Compute your own projection with
   `python -m nanoh3.engine.benchmark --checkpoint <ckpt> --throughput <TFLOPs>`.
4. **Quantisation is weight-only fake-quant.** It stores real int8/fp8 tensors
   (so the memory saving is real) and dequantises per matmul, which is correct
   everywhere but gives up the native-kernel speedup on Hopper/Ada/Blackwell.
5. **Text conditioning is a small byte-level transformer**, not a trained
   CLIP/Qwen. Swap it by loading a different encoder that returns
   `(B, L, text_dim)` — the denoiser does not know the difference.
6. **The demo VAE is not strictly causal.** `norm_mode: clip` plus
   `causal_decoder: false` are what let motion survive at all; the strictly
   causal configuration is still implemented, tested and available, but it
   produces still frames. The trade-off is measured in
   [`docs/DESIGN.md`](docs/DESIGN.md).
7. **Video only, no audio.** H3 generates audio in the same pass; we do not.
8. **English prompts on a synthetic shape dataset.** Real data enters through
   `scripts/prepare_teacher_data.py`.

---

## Licence

Apache-2.0 — see [`LICENSE`](LICENSE).

You are responsible for the licence of any data or model you point the data
pipeline at. `docs/RESEARCH-H3.md` §4 covers the H3 licence terms specifically,
because they are the reason the obvious approach is closed to us.

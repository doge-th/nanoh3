# Design

Why the model looks the way it does, and what each decision bought. The
MiniMax-H3 research that motivated the constraints is in
[`RESEARCH-H3.md`](RESEARCH-H3.md); this document is about our side of the trade.

---

## The constraint chain

Everything follows from one measurement: on an 8 GB card, H3 needs **9–14
minutes** for a 10-second clip even with a pruned int8 denoiser and a 4-step
turbo LoRA, because the model is streamed from host RAM every step. There is no
30× speedup available in the ecosystem.

So the target is not "run H3 faster". The target is "a video model that fits, and
finishes, on the hardware the user has". That reverses the usual order of
optimisation:

```
memory is not the constraint  ->  time is the constraint
                              ->  spend the memory headroom on speed
```

The reference profile lands at **1.19 GiB of 8 GiB**. That 6-7x headroom is not
wasted; it is what makes depth pruning, aggressive quantisation and a 4-step
schedule affordable without argument.

---

## Architecture

```
text  ->  byte-level transformer  ->  (B, 77, text_dim)
                                       |
noise ->  patchify 2x2  ->  (B, N, D)  |
                   3D RoPE + adaLN-Zero + cross-attention
                                       |
                     velocity in latent space
                                       |
                     causal 3D VAE decode -> frames -> mp4
```

### Causal 3D VAE — 8× spatial, 4× temporal, 16 latent channels

The compression ratio is not arbitrary. It is what keeps the token count small
enough that attention is not the bottleneck, and 4× temporal is what makes the
encoder strictly causal in a useful way: latent frame *k* depends only on input
frames `≤ 4k+3`.

Two things had to be right for the causality to be real, and both are easy to get
wrong:

**Padding.** `nn.Conv3d` only accepts symmetric padding, so causal padding is
applied explicitly with `F.pad` before a `padding=0` convolution. The amount is
`(k-1)·dilation` on the left and zero on the right. That single choice gives
both properties we need:

- output *n* sees inputs `[n·s - d(k-1), n·s]`, never a future frame;
- `floor((L-1)/s) + 1 == ceil(L/s)`, so a downsample by *s* yields exactly the
  expected number of latent steps.

The natural-looking alternative, `(s-1)·k` of total padding, gives neither: it
loses two samples per layer at stride 1 and produces an off-by-one latent count
at stride 2.

**Normalisation.** `nn.GroupNorm` computes statistics over `(C/G, T, H, W)`, so
its output at frame *t* depends on the mean and variance of the *entire clip*.
Causal convolutions do not save you: the norm is a global mixing step wearing a
convolution's clothes. `ClipOrFrameNorm` in its default `"frame"` mode folds time into the batch dimension so
the statistics are per-frame. Measured leak before the fix: **0.032** on the
encoder and **1.09** on the decoder. After: ~1e-7 on both — float32 reassociation
from the shape-dependent kernel choice, not leakage (the first latent frame is
exactly 0).

The decoder up-samples with `nearest` interpolation rather than transposed
convolutions, because the interpolation is exact in shape and the transposed
convolution's padding is a well-known source of checkerboard artefacts. It is
also cheap, and the decoder is a real memory consumer at output resolution.

### Denoiser — rectified-flow DiT with adaLN-Zero

Standard modern recipe, chosen for parts that pay off under our constraints:

| Choice | What it buys here |
|---|---|
| **adaLN-Zero, all gates zero-initialised** | the model starts as an exact identity, and depth can be cut at inference without the surviving blocks being mis-calibrated |
| **`set_active_depth`** | the 8 GB "fast" profile drops blocks rather than resolution; bounded quality loss, and it is one line to test |
| **3D RoPE, per-axis frequency bands** | head dims split across (t, h, w) in proportion to the grid, so the time axis gets its own frequency spectrum instead of competing with the spatial axes |
| **`scaled_dot_product_attention`** | FlashAttention on CUDA for free, and it avoids materialising an `N²` matrix at 1024+ tokens |
| **Text cross-attention with a mask** | padded prompt positions are exactly zero, not just unused |
| **Interleaved rotary pairs** | `apply_rotary` pairs channels `2i`/`2i+1`; the frequency table is built from *channel* dims, and all three axis tables share `max(grid)` rows so they concatenate |

`tests/test_model.py` asserts the awkward parts: that `split_head_dim` always
sums to `head_dim` and stays even, that token 0 is an identity rotation, and
that an untrained model predicts exactly zero (which is what adaLN-Zero means).

### Text tower — byte-level, trained jointly

A 4-layer, 256-wide causal transformer over raw UTF-8 bytes. Not a compromise so
much as a fit: H3's Qwen3-VL-32B conditioner is 62.1 GiB, larger than its own
quantised denoiser, and it runs once per prompt. Ours is ~8 M parameters, needs
no tokenizer download, and trains with the denoiser.

The denoiser only ever sees `(B, L, text_dim)`, so replacing this with a frozen
CLIP or Qwen tower is a load-time change, not a code change.

---

## The engine

### int8 / fp8 weight-only quantisation

`int8` is symmetric with a per-output-channel scale; `fp8` is
`torch.float8_e4m3fn` with a per-tensor scale. Both store the *quantised*
tensor, so the memory saving is real, and both dequantise per matmul.

That last choice costs speed and buys correctness: fake-quant numerics are
identical to a real int8 kernel's, so a measurement taken on a laptop is a
measurement of the thing that will run on the GPU, and the same code path works
on CPU, MPS and CUDA. The native-kernel speedup is listed as `UNKNOWN` in the
benchmark rather than assumed.

Measured round-trip error, `tests/test_model.py::test_quant_roundtrip_error_is_small`:
int8 **< 0.02**, fp8 **< 0.20** absolute on N(0,1) weights. The fp8 bound is
loose because e4m3 has 3 mantissa bits — the absolute error is set by the largest
weight, not by the format's nominal precision.

### Block offload

`SequentialOffload` replaces a `ModuleList`'s entries with wrappers that copy
weights in before each block and back out after, prefetching the next block on a
worker thread. Blocks are *replaced*, not wrapped in place, so the parent module
still owns the same parameter objects — no submodule cycle, and checkpoints and
optimizers are unaffected.

On CPU and MPS this is a no-op, which is deliberate: it means the code path is
identical everywhere and the memory numbers in the budget are the ones that
matter on CUDA.

### Few-step sampler

Rectified flow, `t = 1` (noise) → `t = 0` (data), integrated with Euler; Heun is
available for the last step. A `shift` parameter redistributes steps towards high
noise — H3 uses 12 for its 20–50 step schedule, but a 4-step schedule wants a
different shift, so it is a config knob rather than a constant.

Distillation (`training/distill.py`) is LCM-style trajectory skipping: roll the
teacher out with a fine schedule, then train the student to land on the teacher's
endpoint in one jump. Repeat with a smaller skip to reach 2 steps.

This is the actual speed lever. A 4-step schedule is a **12.5× reduction** in
denoising compute against H3's 50-step default, and it needs no licence grant,
because the teacher is our own long-schedule model.

### Memory budget

`engine/budget.py` computes peak VRAM as a sum of documented terms: resident
denoiser weights (after quantisation and offload), denoiser activations from the
token count, VAE decode activations (capped by `vae_chunk_frames`), latents, the
output buffer, and a 0.88 GiB floor for the CUDA context and kernel scratch.

`assert_fits()` turns the sum into a pass/fail, the CLI exits non-zero when a
profile does not fit, and a test calls it. The claim "fits in 8 GB" is therefore
a property of the code, not of a README sentence.

---

## What we deliberately did not do

- **No sparse or windowed attention.** At 1024 tokens with FlashAttention it is
  not the bottleneck, and it would cost quality we cannot measure here.
- **No audio.** H3 generates video and audio in one pass through two sigma
  schedules (12 and 3). For an 8 GB card that is two more models for no gain on
  the stated target.
- **No adversarial or LPIPS loss on the VAE.** A proper autoencoder needs one.
  L1 plus multi-scale gradient plus a temporal term is enough to prove the
  pipeline and is honest about what it is.
- **No int4.** The SVDQuant/NVFP4 ladder is where H3's ecosystem went, and
  implementing a 4-bit kernel is a project of its own. int8 already leaves us at
  1.19 GiB of an 8 GiB budget; there is nothing to buy.

---

## Getting the clip to move: four bugs, in the order they had to be found

The first demo checkpoint generated one frame and repeated it nine times. This
section is the post-mortem, because every fix was found by measurement rather
than by reasoning, and three of the four fixes were individually insufficient.

**Symptom.** Frame-to-frame delta on the output: `0.0` to `3e-4`, against
`1e-2`…`3e-2` in the training clips. It is a video that does not move.

**First, prove it is not the architecture.** Overfit *four* clips for 600 steps —
if the model cannot memorise motion, nothing about data or schedule will help.
It could not: `motion` stayed pinned at exactly `1.0000` and the reconstruction
frame delta at exactly `0.0000`. Overfitting is the right first experiment
because it removes capacity and generalisation from the list of suspects.

### Bug 1 — the decoder's receptive field pointed backwards

Measured with a gradient probe, `|∂ out[4] / ∂ latent[k]|`:

| decoder | ∂/∂latent[0] | ∂/∂latent[1] | ratio |
|---|---|---|---|
| causal (before) | 1670 | 31 | **144×** |
| centred (after) | 3.29 | 3.25 | **1.0×** |

Output frame 4 should be built from latent 1. It was dominated by latent 0 by two
and a half orders of magnitude, because nearest-neighbour temporal upsampling
followed by a *causal* convolution gives `out[4]` the taps
`[latent0, latent0, latent1]`. With that skew, "copy the first latent and repeat"
wins every reconstruction loss, which is exactly the frozen output.

Fixed with `causal_decoder=False` and centred padding — which is what MiniMax-H3
itself does (`causal_encoder: true, causal_decoder: false` in its VAE config).
Regression test: `test_decoder_receptive_field_is_centred_on_its_own_latent`.

### Bug 2 — per-frame normalisation erased per-frame contrast

Feed a trained decoder a latent that flips between +5 and −5:

| `norm_mode` | output change |
|---|---|
| `frame` | **exactly 0.0** |
| `clip` | `1e-4`…`4e-4` |

Per-frame statistics remove exactly the per-frame mean and scale that an encoder
will otherwise use to carry motion. The `frame` mode keeps strict causality; the
`clip` mode does not. The demo uses `clip`; both are available and
`test_clip_norm_mode_is_not_causal` documents the cost.

### Bug 3 — loss weight was not the lever

Raising the temporal weight 2 → 25 → 50 changed nothing: `motion` stayed at
exactly `1.0000` in every case, and at weight 25 reconstruction L1 degraded from
0.14 to 0.65+ for nothing.

### Bug 4 — the actual cause: the temporal gradient lived on 4 % of the pixels

In a video, most pixels sit on a static background. There `d_rec - d_ref` is
*exactly* zero, so an unweighted temporal term has essentially no gradient where
motion actually is. Measured gradient norms on the same batch: **L1 → 106, the
temporal term → 0.59**, i.e. 180× weaker, despite being 4× the loss value. No
weight could fix a gradient that was being cancelled.

The fix is to weight each pixel by how much it actually moves,
`Σ |d_ref| · (d_rec − d_ref)² / Σ |d_ref|`. Controlled comparison, same seed,
25 steps, frame delta of the reconstruction:

| temporal term | frame delta |
|---|---|
| L1 | 0.00000 |
| MSE | 0.00000 |
| Charbonnier | 0.00000 |
| **motion-weighted** | **0.32210** |

`recon_loss` implements this as `temporal_form="motion_weighted"` (the default),
with `"l1"` and `"mse"` kept for ablation, and prints the `motion=` term next to
`l1` in the training log so this can never be invisible again.

### What is still missing

- motion is now produced, but it is coarse: the demo's latent timeline is 3 frames
  for a 9-frame clip, so it learns "the object is somewhere else next" rather
  than smooth trajectories;
- a real video VAE adds a temporal-adversarial or motion-consistency term and
  trains on data where motion dominates the pixels — neither is in scope here;
- `norm_mode="clip"` and `causal_decoder=False` together mean **the demo VAE is
  not strictly causal**. The strict path is still available and tested; it is
  just not the one that produces motion.


---

## If you extend this

Three things we would do next, in order:

1. **Train the `configs/8gb.yaml` model on real video.** The pipeline is done and
   tested; only the GPU hours and the data are missing. Everything downstream
   (speed, memory, distillation) is already sized for it.
2. **Replace the text tower with a frozen CLIP.** The interface is a
   `(B, L, text_dim)` tensor and nothing else changes.
3. **Measure on real hardware and replace the projections in `BENCHMARK.md` with
   measurements.** That file is the weakest part of this project by construction;
   it was written without an NVIDIA GPU and says so in every section.

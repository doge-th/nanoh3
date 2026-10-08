# MiniMax-H3: research notes

Compiled 2026-10-07. Every number below carries the URL it came from; anything
that could not be verified is listed in §7 rather than guessed at.

This document exists because the rest of the repository makes design decisions
"because of H3", and those decisions have to be traceable to facts rather than to
a vibe. Read §1 and §5 first if you only want the parts that change what NanoH3
does.

---

## 1. What H3 actually is

| Field | Value | Source |
|---|---|---|
| Model | **MiniMax H3**, released 2026-08-02 | [model card](https://huggingface.co/MiniMaxAI/MiniMax-H3), [`LICENSE`](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE) line 2 |
| Weights | `MiniMaxAI/MiniMax-H3` (HF, ungated), `MiniMax/MiniMax-H3` (ModelScope) | [HF API](https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3) |
| Code | [github.com/MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) — 9,623 stars, **no licence file** | [GitHub API](https://api.github.com/repos/MiniMax-AI/MiniMax-H3) |
| ComfyUI repack | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) | HF API |
| Size on HF | **498 GB across 309 files** (the diffusers layout is present three times) | [HF tree API](https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3/tree/main?recursive=true) |
| Technical report | **Does not exist.** No arXiv paper, no training-data disclosure | arXiv API searched exhaustively; model card and README have no link |
| Licence | **Not open source.** "MiniMax H3 Community License Agreement" | [`LICENSE`](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE) |

### Architecture (from `transformer/config.json`)

```
num_attention_heads 56 · attention_head_dim 128 · hidden_size 5376
num_layers 50 · ffn_dim 14336 · patch_size [1, 2, 2] · in_channels 24
text_dim 5120 · rope_freq_dim 16 · rope_theta 10000
```

- **33 B parameters, dense, single-stream** — no MoE. The weight index has 638
  keys and zero `expert`/`gate`/`moe` keys.
- **~13 B of those sit in AdaLN branches.** The model card says they "can be
  precomputed and cached" and "do not need to be loaded for inference-only
  deployment". Measured independently: Comfy-Org's `pruned_bf16` file is
  40.226 GB against 66.280 GB for the full one — a 26.054 GB delta, i.e.
  13.03 B parameters at bf16.
- **Text encoder is Qwen3-VL-32B** (64 layers, hidden 5120, 262 k context).
  H3 reads the **unnormalised hidden state after layer 50**, so the whole
  released checkpoint is loaded with its language-model head unused. The encoder
  is *larger than the quantised denoiser* at every tier.
- **VAE is `f16t4d24`**: 16× spatial, 4× temporal, 24 latent channels, with a
  36-layer ViT decoder. Frame counts must be `17n + 5`.
- **Effective token reduction is 32× spatial / 4× temporal** (16× in the VAE × 2×2
  patchifier), not the 16× the VAE alone suggests.

### Scheduling

```json
{"_class_name": "MiniMaxH3Scheduler", "shift": 12.0}   // video
{"_class_name": "MiniMaxH3Scheduler", "shift": 3.0}    // audio
```

**There is no CFG.** The released weights are guidance-distilled: no negative
prompt, no `guidance_scale`, one forward pass per step. (GGUF runners must be
told `--cfg-scale 1.0` explicitly — the default of 7.0 aborts.)

---

## 2. The lowest-resolution open preset

The model card publishes **no minimum resolution**. It only ever shows 768-short-edge
(~1 MP) examples. The lowest resolution with a *published runtime measurement* is:

| Preset | Where | Setting |
|---|---|---|
| **864 × 480, 124 frames (5 s), 20 NFE** | [SGLang cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3), consumer-GPU table | 12 GB VRAM + 32 GB host RAM; 16.8–18.7 s/it |
| 960 × 544 | [diffusers docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3) | "runs about **2.3× faster per step** than the trained 1344×768" |
| ≈832 × 480 (0.4 MP) | Official ComfyUI T2V template | described as "a fast preview size" |

Hard constraints: dimensions must be multiples of **32**, frames must be
`17n + 5`, duration 5–15 s, short edge 768 by default, area capped at
768 × 1344 ≈ 1.03 MP.

The exact command the cookbook gives for that 864 × 480 row:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang serve --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va \
  --performance-mode memory \
  --layerwise-offload-components dit,text_encoder,vae \
  --layerwise-resident-layers video_vae=36
```

---

## 3. What 8 GB actually costs on real H3

This is the measurement that decided the NanoH3 design. From a community
reproduction on an **RTX 5060 Laptop 8 GB** (15.26 GiB RAM, ComfyUI 0.35,
pruned-int8 DiT + NVFP4 text encoder, 768×1024, 243 frames = 10 s):

| Steps | Wall clock |
|---|---|
| 4 (turbo LoRA) | ≈ **9 minutes** |
| 8 (turbo LoRA) | **857.7 s = 14.3 minutes** |
| 20 (base weights) | **33.8 – 43.5 minutes** |

Source: [FlowForgeLabAi/MiniMax-H3-8GB-Workflows](https://github.com/FlowForgeLabAi/MiniMax-H3-8GB-Workflows).
ComfyUI reports `19995MB Staged` for that stack: the DiT never fits in VRAM, it
is streamed from host RAM every step. Their single biggest operational finding
is that **system RAM is the binding constraint, not VRAM** — at 768×1344 on a
16 GB box the process pages, and after ~1.7 h of continuous sampling a run slows
from 0.9 s to 212 s per block.

**Conclusion: 8 GB is feasible for H3, and sub-minute generation is not.** A 30×
speedup over the 4-step measurement is not available from any published
technique. That is the entire reason this repository exists: the only way to
"8 GB and under a minute" is a much smaller model, not a better H3 setup.

### Where the memory goes

- **Peak VRAM at sampling is activations, not weights.** ComfyUI PR #15983
  measured the H3 working set at **≈151,000 bytes per packed sequence row** and
  had to recalibrate their memory factor from 0.114 to 0.24 because the old
  value "substantially underestimates H3's actual memory requirements". H3 denoises
  one packed sequence holding text, keyframes, audio and video tokens together
  ([PR #16460](https://github.com/Comfy-Org/ComfyUI/pull/16460)).
- **Largest single file is the text encoder**: 14.61 GiB (NVFP4) / 25.28 GiB
  (int8) / 47.97 GiB (bf16) — bigger than the pruned int8 DiT's 19.53 GiB. It is
  transient (loaded once per prompt, freed before the sampler runs), which is why
  `--backend te=cpu` is mandatory in the GGUF path.
- **VAE decode is the third consumer** and the one people forget: 4.85 GiB fp16 /
  2.62 GiB int8, running once at full output resolution.

### What the ecosystem has actually built

| Technique | Repo | Effect |
|---|---|---|
| AdaLN pruning (13.03 B → 0.04 B) | `lilcheaty/MiniMax-H3-NVFP4` documents it best | 61.72 → 37.46 GiB, no quality claim |
| int8 convrot | Comfy-Org | 37.46 → 19.53 GiB |
| W4A4 SVDQuant | `QuantFunc/…-4bit` | 11.52 GiB; **3.2 s/step vs 8.5 s int8 on a 4090** |
| NVFP4 | `lilcheaty/…` | 11.67 GiB; −8.0 GB VRAM, −12.4 % time vs int8 |
| Turbo LoRAs (4/8 NFE) | `lightx2v`, `larryvrh` | the main speed lever |
| PDMD 2-NFE / 4-NFE | `Iwannapose/…`, `Kijai/MiniMax-H3-experimental/loras` | fastest H3 mode in existence, **zero published timings** |
| Learned 10 %-attention predictor | `Veda-Sparse/…` | 3.12× end-to-end on a PRO 6000, 1.77–2.82× on a 4090 |
| GGUF Q2_K | `unsloth/…` | 6.26 GiB — the only denoiser that fits *inside* 8 GB, **never benchmarked at 8 GB** |

### A Chinese deployment guide, cross-checked

A widely-shared WeChat guide covers exactly this ground: Kijai's PDMD
`2-step rank 38 (381 MB)` and `4-step rank 57 (559 MB)` LoRAs, deployed through
ComfyUI with a pruned W4A8/W6A8 backbone, an int8-convrot video VAE and an NVFP4
text encoder. **Its file sizes cross-check against the HF tree** — 381 MB vs the
0.36 GiB `pdmd_2step_rank-38` file, 559 MB vs the 0.52 GiB `pdmd_4step_rank-57`
file — so the guide is describing real artefacts, not vapour. Its tuning red
lines also match the official docs independently: steps must equal the LoRA's
step count, and CFG stays at 1.0–1.5 because the weights are guidance-distilled.

Its card-by-card table is the useful part, and it is worth reading as a *tier
map* rather than a performance claim:

| Tier in the guide | Cards | Its own verdict |
|---|---|---|
| 12 GB | 5070 / 4070 / 3060 / 3080 12 GB | "核心受惠区" — the tier the recipe is designed for |
| 16 GB | 4070 Ti S / 4080 / 5080 / 4060 Ti 16 GB | native 768p |
| 24 GB | 3090 / 4090 / 5090 | full-fat |
| **8–10 GB** | 4060 8 GB / 3070 / 3080 10 GB | **"极客折腾梯队"** — requires extreme CPU/GPU slicing, and it warns of system-memory swapping |

So the guide's own framing puts **8 GB in the tinker tier, below the 12 GB tier it
was written for** — which matches the measured 14.3-minute 8 GB run in the table
above. Its headline speed claim is "12 GB, 960×544, 5 s, down to tens of seconds"
with the 2-step LoRA; note that is 12 GB and a 5-second clip, and the guide gives
no GPU model, no seed count and no per-step figure to check it against. Nothing
in it produces "8 GB, under a minute".

Real per-step cost, best available public data:

| GPU | Clip | Per step |
|---|---|---|
| RTX 4090 24 GB | 864×480, 124 fr, 20 steps | 3.7–4.0 s |
| RTX 5090 32 GB | 864×480, 124 fr, 20 steps | 2.6–2.9 s |
| RTX 5090 + 4-step turbo | 1344×768, 56 fr | 16.17 s total |
| RTX PRO 6000 + int8 + SLA sparse, 4 steps | 1152×640, 243 fr (10 s) | 76 s total |

---

## 4. The licence finding — this one changes the plan

Section **V.3** of the MiniMax H3 Community License Agreement:

> You may not use the MiniMax H3 Works or any of their Outputs or results to
> improve any other artificial intelligence model (other than MiniMax H3 or its
> Model Derivatives).

The obvious plan — distil H3's output into our own model — is **exactly what that
clause forbids.** So `nanoh3/data/teacher.py` refuses `MiniMaxAI/MiniMax-H3` and
its derivatives by default, and exits non-zero:

```
$ python scripts/prepare_teacher_data.py --repo MiniMaxAI/MiniMax-H3
licence guard: refusing to use 'MiniMaxAI/MiniMax-H3' as a data source:
MiniMax H3 Community License Agreement section V.3 forbids using the H3 works or
their outputs to improve any other AI model.
```

Other terms worth knowing before you build anything commercial on top of this
area: the licence **excludes the EU, UK, South Korea and the USA** from its
"Applicable Territory", requires written authorisation above **US$20 M/year**
revenue, and requires "MiniMax H3" in the UI of commercial products.

The compliant paths, both implemented in this repo:

1. **Self-distillation** (`nanoh3/training/distill.py`): train a long-schedule
   NanoH3, then distil it to 2–4 steps. This is the speed story, and it costs
   nothing in licence terms.
2. **Permissively licensed data** through the same teacher interface, pointed at
   a teacher whose licence permits model improvement.

`--assert-rights` exists for operators who hold a separate grant. It is an
assertion, not a licence, and the manifest records that it was asserted.

---

## 5. What NanoH3 borrows, and what it deliberately does not

| H3 | NanoH3 | Why |
|---|---|---|
| 4× temporal VAE compression | same | video cost is dominated by the time axis; also makes streaming possible |
| 16× spatial + 2×2 patch → 32× tokens | same (`configs/h3_matched.yaml`) | matching the token grid is what makes a controlled transfer or ablation possible |
| 3D RoPE over (t, h, w) | same, per-axis frequency bands | positional structure in three axes is the reason H3's motion looks coherent |
| AdaLN-Zero conditioning | same, and prunable at inference (`set_active_depth`) | H3 keeps ~13 B AdaLN parameters; ours is zero-init so depth can be cut |
| Guidance-distilled (no CFG) | same (CFG optional, off by default) | halves the per-step cost |
| Flow shift 12 | `shift` is a config knob, demo default 3.0 | H3's 12 is tuned to 768p and 20–50 steps; a 4-step schedule wants a different shift |
| 33 B dense DiT | ~300 M | this is the entire point |
| Qwen3-VL-32B conditioner (62 GiB) | ~8 M byte-level transformer | the text encoder is the largest file in the H3 stack and it runs once per prompt |
| Concurrent audio + video in one pass | video only | audio adds a second VAE, a second sigma schedule and a vocoder for no benefit to the 8 GB target |
| 20–50 NFE | 4 NFE after distillation | the sub-minute target |
| Context-IR prompt expansion (closed) | none | it is not open-sourced and we cannot reimplement what we cannot see |

**The honest framing:** NanoH3 is not a smaller H3. It is an independently
implemented model that uses the same *public architectural vocabulary* (causal
3D VAE, rectified-flow DiT, 3D RoPE, adaLN-Zero, guidance distillation) at a
scale where the whole thing — denoiser, text tower, VAE, activations — fits in
about 1.2 GiB. It will not match H3's quality and does not claim to.

---

## 6. The measurements that produced the 8 GB profile

`python -m nanoh3.engine.budget --profile all` — total for each profile, with the
denoiser weights already int8:

| Profile | Shape | Steps | Total | Dominant term |
|---|---|---|---|---|
| 8gb | 256×256×16 | 4 | **1.19 GiB** | VAE decode + 0.88 GiB CUDA workspace floor |
| 8gb-tiny | 256×256×16 | 4 | 1.16 GiB | same |
| 8gb-512 | 512×288×24 | 4 | 1.32 GiB | activations |
| 24gb | 512×288×24 | 8 | 2.61 GiB | bf16 weights, no offload |

**The 8 GB budget is met with roughly 6× headroom**, which is the honest and
slightly awkward result: at this model size the binding constraint is *time*,
not memory. That is why the project also ships a depth-pruning knob
(`--active-depth`) and a 4-step sampler — memory was never going to be the
problem, so we spend the savings on speed instead.

---

## 7. Not known / not verified

1. **No official VRAM requirement exists for H3.** The model card, Comfy-Org's
   README and the ComfyUI docs all omit it. Everything in §3 is third-party.
2. **No minimum supported resolution is published** — only the observed 864×480.
3. **4 s vs 5 s minimum duration** conflicts between the model card/SGLang (4–15 s)
   and the diffusers source (`min_duration = 5.0`).
4. **The exact AdaLN factorisation in the full checkpoint** is not documented;
   the pruned build's `adaln_t_table` of shape `[1025, 8]` is the known form.
5. **SGLang and ComfyUI both ship a smaller text encoder than the released one**
   (46.18 GiB and 51.5 GB against 66.7 GB) with no official explanation.
6. **No timings exist for the PDMD 2-NFE student** — the fastest H3 mode in
   existence is entirely unbenchmarked publicly.
7. **No 8 GB benchmark exists for the GGUF path.** The smallest denoiser is
   6.26 GiB, leaving almost no activation headroom.
8. **WeeLLM's "3.14 GB, no quantisation" claim for H3** has blank RAM, resolution
   and time fields in its own table, and the author has not reproduced it.
9. **Training data and compute for H3 are entirely undisclosed.**

---

## 8. Sources

* Model card, licence, QA doc: [README.md](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/README.md) · [`LICENSE`](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE) · [`QA-about-License.md`](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/docs/QA-about-License.md)
* Configs: [`transformer/config.json`](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/transformer/config.json) · [`text_encoder/config.json`](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/text_encoder/config.json) · [`vae/config.json`](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/vae/config.json) · [`scheduler_config.json`](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/scheduler/scheduler_config.json)
* File sizes: [HF tree API](https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3/tree/main?recursive=true) · [Comfy-Org tree API](https://huggingface.co/api/models/Comfy-Org/MiniMax-H3/tree/main?recursive=true)
* Deployment and consumer-GPU numbers: [SGLang cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3)
* API, memory tiers, CFG distillation: [diffusers docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3)
* Step counts, resolution cap, attention backends: [ComfyUI docs](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
* Activation-memory measurement: [ComfyUI PR #15983](https://github.com/Comfy-Org/ComfyUI/pull/15983) · [PR #16460](https://github.com/Comfy-Org/ComfyUI/pull/16460)
* The 8 GB measurement: [FlowForgeLabAi/MiniMax-H3-8GB-Workflows](https://github.com/FlowForgeLabAi/MiniMax-H3-8GB-Workflows) · [discussion #59](https://huggingface.co/Comfy-Org/MiniMax-H3/discussions/59) (six-GPU per-step comparison) · [discussion #26](https://huggingface.co/Comfy-Org/MiniMax-H3/discussions/26) (5090 turbo timing)
* Quantisation ladder: [lilcheaty/MiniMax-H3-NVFP4](https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4) · [QuantFunc 4-bit](https://huggingface.co/QuantFunc/Minimax-H3-Quantfunc-4bit) · [MATLOWAI fused turbo](https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot) · [unsloth GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF)
* Few-step: [lightx2v Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) · [larryvrh Turbo](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) · [PDMD 2-NFE](https://huggingface.co/Iwannapose/minimax_h3_pdmd_2nfe_comfyui) · [Veda sparse attention](https://huggingface.co/Veda-Sparse/Minimax-h3-T2VA-Veda-8NFE-600Step-Preview)

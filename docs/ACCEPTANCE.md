# NanoH3 — Acceptance Checklist

Derived from the original request:

> 先研究再干，把 minimax H3 的最低分辨率的开源版，进行研究，研发出自己的版本的 h3，
> 显存只需要 8 个 g，速度越快越好尽量达到一分钟以内生成一短视频。然后开源。

Scope level: **L** (multi-deliverable, to be open-sourced, hard numeric targets).

A third party who has never seen the conversation must be able to judge pass/fail from
this table alone. Every ✅ below must carry evidence produced during the build
(command output, file path, `file:line`, or a rendered artifact).

## P0 — must pass

| #  | Acceptance criterion (verifiable) | Verification method |
|----|------------------------------------|---------------------|
| P0-1 | Research report identifies MiniMax-H3's architecture numbers, total parameter count, weight size, and the **lowest-resolution supported generation preset**, with a source URL per claim | `docs/RESEARCH-H3.md`, every number has a URL |
| P0-2 | A from-scratch, independent text-to-video model exists in this repo: causal 3D VAE + transformer DiT + text conditioning + flow-matching objective. No MiniMax weights or code are copied. | `nanoh3/models/*.py` + `tests/test_model.py` (shape/causality tests) |
| P0-3 | One documented command turns a text prompt into a playable `.mp4` on a machine with no GPU-grade hardware | `python -m nanoh3.cli generate ...`; actual `.mp4` produced during this build |
| P0-4 | The full training pipeline runs end-to-end and reduces loss: `train.py` finishes N steps on real data and writes a checkpoint | training log from this build, before/after loss printed |
| P0-5 | 8 GB VRAM target is a *computed, asserted* budget, not a claim: a tool computes peak VRAM for (model size, precision, resolution, frames) and fails if > 8 GB for the reference profile | `python -m nanoh3.engine.budget --profile 8gb` → asserts; unit test asserts the same |
| P0-6 | Quantization is implemented, not stubbed: fp8 (e4m3) weight storage and int8 per-channel activation quant with measured numerical error, selectable at load time | `nanoh3/engine/quant.py` + `tests/test_quant.py` reports max abs error |
| P0-7 | Block-wise CPU/GPU offload is implemented and is what makes the 8 GB profile possible; it is exercised at least once in a real run | `nanoh3/engine/offload.py` + log line showing offload active |
| P0-8 | Latency is modelled and measured: a harness measures real per-stage time on available hardware and computes a projection for a reference 8 GB NVIDIA GPU, with every assumption listed | `python -m nanoh3.engine.benchmark`; output file `docs/BENCHMARK.md` |
| P0-9 | <60 s target is stated as a **target vs. verified** distinction, never as a verified 8 GB result | `docs/BENCHMARK.md` has an explicit "VERIFIED / PROJECTED / UNKNOWN" table |
| P0-10 | Distillation from MiniMax-H3 is implementable: teacher-clip prep script (lowest-resolution preset), student distillation trainer, and a config that names the exact teacher repo + preset | `scripts/prepare_teacher_data.py`, `nanoh3/training/distill.py`, `configs/*.yaml` |
| P0-11 | Open-source hygiene: Apache-2.0 LICENSE, README with quickstart, no proprietary weights committed, `.gitignore` excludes checkpoints, and a one-command publish script | files exist; `git ls-files` shows no weights |
| P0-12 | Test suite passes | `pytest -q` all green, output pasted in the delivery report |

## P1 — quality bar

| #  | Criterion | Verification |
|----|-----------|--------------|
| P1-1 | Deterministic sampling given a seed | same seed → byte-identical latent hash, test asserts |
| P1-2 | The VAE is temporally causal (no future-frame leakage) | test asserts output at frame t is unchanged when frame t+1 is perturbed |
| P1-3 | Model config is a single declarative YAML, and every config used is committed | `configs/` + test that loads all of them |
| P1-4 | Code has type hints, no TODO/FIXME/placeholder text left behind | `grep -rn "TODO\|FIXME\|XXX\|placeholder"` → 0 hits |
| P1-5 | Honest limitations section exists and names what was *not* verified | `README.md` §Limitations |

## Explicit non-goals / declared limits

These are known before the build starts and are reported as-is, not silently dropped:

1. **No NVIDIA GPU is available in this environment** (Apple M5, 16 GB unified memory).
   Therefore: the 8 GB CUDA path and the <60 s wall-clock claim **cannot be measured here**.
   They are delivered as a tested code path + a memory budget that is enforced by assertion
   + a projection whose assumptions are listed. Anything claiming otherwise is a lie.
2. **Training a full-quality model needs GPU-days.** What is trained here is a small
   demonstration model on procedurally generated clips, proving the pipeline. The README
   states exactly which command trains the real model on rented hardware.
3. **MiniMax-H3 weights are not redistributed** (license review required first).
   The repo ships code + configs + a download script that fetches the teacher separately.

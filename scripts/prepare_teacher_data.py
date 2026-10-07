#!/usr/bin/env python
"""Build a reference-clip dataset from a teacher you have the right to use.

    # 1. dry run: just write the prompt manifest and check the licence guard
    python scripts/prepare_teacher_data.py --prompts-file prompts.txt --out data/teacher

    # 2. real run: invoke your teacher for each prompt
    python scripts/prepare_teacher_data.py \
        --backend command \
        --command "python my_teacher.py --prompt {prompt} --out {out}" \
        --repo my-org/my-teacher \
        --out data/teacher

MiniMax-H3 is blocked by default on purpose: its community licence section V.3
forbids using its outputs to improve another AI model. See
``nanoh3/data/teacher.py`` and ``docs/RESEARCH-H3.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from nanoh3.data.teacher import (  # noqa: E402
    DEFAULT_PROMPTS,
    TeacherConfig,
    TeacherRightsError,
    build_manifest,
    postprocess,
    run_backend,
    save_clip,
    write_manifest,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts-file", default=None, help="one prompt per line; defaults to the built-in set")
    ap.add_argument("--out", default="data/teacher")
    ap.add_argument("--repo", default="self/nanoh3", help="teacher identity, checked against the licence guard")
    ap.add_argument("--backend", default="dryrun", choices=["dryrun", "command"])
    ap.add_argument("--command", default="", help="template with {prompt} and {out} placeholders")
    ap.add_argument("--teacher-width", type=int, default=512)
    ap.add_argument("--teacher-height", type=int, default=512)
    ap.add_argument("--teacher-frames", type=int, default=33)
    ap.add_argument("--teacher-steps", type=int, default=20)
    ap.add_argument("--width", type=int, default=256, help="student clip width")
    ap.add_argument("--height", type=int, default=256, help="student clip height")
    ap.add_argument("--frames", type=int, default=16, help="student clip frame count")
    ap.add_argument("--assert-rights", action="store_true", help="override the licence guard (you are responsible)")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N prompts (0 = all)")
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = [line.strip() for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        prompts = prompts[: args.limit]

    cfg = TeacherConfig(
        backend=args.backend,
        command=args.command,
        repo=args.repo,
        width=args.teacher_width,
        height=args.teacher_height,
        frames=args.teacher_frames,
        steps=args.teacher_steps,
        target_width=args.width,
        target_height=args.height,
        target_frames=args.frames,
        out_dir=Path(args.out),
        assert_rights=args.assert_rights,
    )
    try:
        manifest = build_manifest(cfg, prompts)
    except TeacherRightsError as exc:
        print(f"licence guard: {exc}", file=sys.stderr)
        return 2

    manifest_path = write_manifest(manifest, cfg.out_dir / "manifest.json")
    print(f"wrote {len(manifest)} records to {manifest_path}")
    print(f"licence: {manifest[0]['licence_note']}")

    produced = 0
    for record in manifest:
        raw = run_backend(cfg, record, dry_run=args.backend == "dryrun")
        if raw is None:
            continue
        clip = postprocess(record, raw)
        save_clip(clip, cfg.out_dir / f"{record['id']}.mp4")
        torch.save(
            {
                "video": clip,
                "prompt": record["prompt"],
                "teacher": record["teacher"],
                "target": record["target"],
                "licence_note": record["licence_note"],
            },
            cfg.out_dir / f"{record['id']}.pt",
        )
        produced += 1
    print(f"clips produced: {produced}" + (" (dry run: no teacher was called)" if produced == 0 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

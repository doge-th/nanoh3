"""Data sources: procedurally generated clips, and reference clips from a teacher."""

from .procedural import ClipSpec, ProceduralClips, random_specs, render_clip
from .teacher import (
    DEFAULT_PROMPTS,
    RESTRICTED_TEACHERS,
    TeacherConfig,
    TeacherRightsError,
    build_manifest,
    check_teacher_rights,
    fit_clip,
    load_video_tensor,
    postprocess,
    run_backend,
    save_clip,
    write_manifest,
)

__all__ = [
    "ClipSpec",
    "ProceduralClips",
    "render_clip",
    "random_specs",
    "TeacherConfig",
    "TeacherRightsError",
    "RESTRICTED_TEACHERS",
    "check_teacher_rights",
    "build_manifest",
    "write_manifest",
    "run_backend",
    "load_video_tensor",
    "fit_clip",
    "save_clip",
    "postprocess",
    "DEFAULT_PROMPTS",
]

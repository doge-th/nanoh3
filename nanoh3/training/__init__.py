"""Training entry points: two-stage (VAE then denoiser) and few-step distillation."""

from .distill import DistillConfig, distill_step, teacher_trajectory
from .train import TrainConfig, build_models, load_bundle, save_bundle, train_dit, train_vae

__all__ = [
    "TrainConfig",
    "build_models",
    "load_bundle",
    "save_bundle",
    "train_vae",
    "train_dit",
    "DistillConfig",
    "distill_step",
    "teacher_trajectory",
]

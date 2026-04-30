"""
Configuration for Diffusion Refiner for 3C Model.

This module defines hyperparameters for training a conditional diffusion model
that refines the output of the 3C Attention UNet by learning to correct
artifacts and fill in details.
"""

import torch
from dataclasses import dataclass, field
from typing import Tuple


def _linear_beta_schedule(beta_start: float, beta_end: float, num_timesteps: int):
    """Linear beta schedule for DDPM."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def _cosine_beta_schedule(num_timesteps: int):
    """Cosine beta schedule for DDPM (smoother transition)."""
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


@dataclass
class DiffusionConfig:
    """DDPM/DDIM diffusion schedule configuration."""

    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "linear"

    sampling_steps: int = 50
    ddim_eta: float = 0.0

    def get_beta_schedule(self):
        if self.beta_schedule == "linear":
            return _linear_beta_schedule(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.beta_schedule == "cosine":
            return _cosine_beta_schedule(self.num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")


@dataclass
class ModelConfig:
    """Conditional U-Net denoiser architecture configuration."""

    base_filters: int = 32
    time_dim: int = 128

    in_channels: int = 1
    cond_channels: int = 5

    @property
    def total_in_channels(self):
        return self.in_channels + self.cond_channels


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 16
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4

    weight_physics: float = 0.05
    weight_residual: float = 0.1
    weight_noise: float = 1.0

    gradient_clip: float = 1.0

    val_frequency: int = 5
    save_frequency: int = 10

    n_train: int = 9500
    n_val: int = 400
    seed: int = 42


@dataclass
class InferenceConfig:
    """Inference configuration."""

    n_samples: int = 8
    sampling_steps: int = 50
    ddim_eta: float = 0.0
    device: str = "cuda"

    apply_constraints: bool = True

    save_samples: bool = False
    output_dir: str = "outputs/diffusion_refiner"


@dataclass
class RefinerConfig:
    """Complete configuration for Diffusion Refiner."""

    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    experiment_name: str = "diffusion_refiner_3c"
    checkpoint_dir: str = "checkpoints/diffusion_refiner"

    spatial_size: Tuple[int, int] = (64, 64)

    def to_dict(self):
        return {
            "diffusion": {
                "num_timesteps": self.diffusion.num_timesteps,
                "beta_start": self.diffusion.beta_start,
                "beta_end": self.diffusion.beta_end,
                "sampling_steps": self.diffusion.sampling_steps,
            },
            "model": {
                "base_filters": self.model.base_filters,
                "time_dim": self.model.time_dim,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "epochs": self.training.epochs,
                "lr": self.training.lr,
                "weight_physics": self.training.weight_physics,
            },
            "inference": {
                "n_samples": self.inference.n_samples,
                "sampling_steps": self.inference.sampling_steps,
            },
        }

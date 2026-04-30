"""
Loss functions for Diffusion Refiner training.

This module implements the combined loss function for training the conditional
diffusion refiner, including:
1. Noise prediction MSE (main diffusion loss)
2. Physics consistency loss (forward model constraint)
3. Residual consistency loss (ensure refinement is bounded)
4. Optional: KL divergence on output distribution
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


class RefinementLoss(nn.Module):
    """
    Combined loss for diffusion refinement training.

    The loss consists of:
    1. Noise prediction loss (main diffusion objective)
    2. Physics consistency loss (signal-space reconstruction)
    3. Residual consistency loss (bounded refinement)
    4. Distribution consistency loss (KL divergence)
    """

    def __init__(
        self,
        forward_model,
        weight_physics: float = 0.05,
        weight_residual: float = 0.1,
        weight_noise: float = 1.0,
        weight_kl: float = 0.05,
    ):
        """
        Args:
            forward_model: ForwardModel2D instance for physics constraint
            weight_physics: Weight for physics consistency loss
            weight_residual: Weight for residual consistency loss
            weight_noise: Weight for noise prediction loss
            weight_kl: Weight for KL divergence loss
        """
        super().__init__()
        kernel = torch.from_numpy(forward_model.kernel_matrix).float()
        self.register_buffer("kernel_matrix", kernel)
        self.n_b = forward_model.n_b

        self.weight_physics = weight_physics
        self.weight_residual = weight_residual
        self.weight_noise = weight_noise
        self.weight_kl = weight_kl

    def _normalize_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize output to valid probability distribution.

        Args:
            x: Input tensor [B, 1, H, W]

        Returns:
            Normalized tensor with sum-to-1 and non-negative
        """
        x_softplus = F.softplus(x)
        return x_softplus / (x_softplus.sum(dim=(2, 3), keepdim=True) + 1e-8)

    def _project_to_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project noisy input to valid distribution space for physics loss.

        Args:
            x: Input tensor [B, 1, H, W]

        Returns:
            Projected tensor
        """
        x_clamped = torch.clamp(x, min=0.0)
        return x_clamped / (x_clamped.sum(dim=(2, 3), keepdim=True) + 1e-8)

    def reconstruct_signal(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct signal from spectrum using forward model.

        Args:
            spectrum: Spectrum tensor [B, 1, 64, 64]

        Returns:
            Reconstructed signal [B, 1, 64, 64]
        """
        batch_size = spectrum.shape[0]
        pred_flat = spectrum.squeeze(1).reshape(batch_size, -1)
        signal_flat = pred_flat @ self.kernel_matrix.T
        return signal_flat.view(batch_size, 1, self.n_b, self.n_b)

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_true: torch.Tensor,
        f_refined: torch.Tensor,
        f_base: torch.Tensor,
        signal: torch.Tensor,
        x_t: torch.Tensor = None,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            noise_pred: Predicted noise [B, 1, H, W]
            noise_true: True noise [B, 1, H, W]
            f_refined: Refined spectrum [B, 1, H, W] (normalized)
            f_base: Baseline spectrum [B, 1, H, W]
            signal: Target signal [B, 1, H, W]
            x_t: Noisy input at timestep t (optional, for debugging)

        Returns:
            Dictionary with total loss and individual components
        """
        loss_noise = F.mse_loss(noise_pred, noise_true)

        f_refined_proj = self._project_to_distribution(f_refined)
        signal_recon = self.reconstruct_signal(f_refined_proj)
        loss_physics = F.mse_loss(signal_recon, signal)

        residual_pred = f_refined - f_base
        loss_residual = torch.mean(residual_pred ** 2)

        if self.weight_kl > 0:
            f_refined_safe = torch.clamp(f_refined, min=1e-8)
            f_base_safe = torch.clamp(f_base, min=1e-8)
            loss_kl = F.kl_div(
                torch.log(f_refined_safe),
                f_base_safe,
                reduction='batchmean'
            )
        else:
            loss_kl = torch.tensor(0.0, device=noise_pred.device)

        total_loss = (
            self.weight_noise * loss_noise
            + self.weight_physics * loss_physics
            + self.weight_residual * loss_residual
            + self.weight_kl * loss_kl
        )

        return {
            'loss': total_loss,
            'loss_noise': loss_noise,
            'loss_physics': loss_physics,
            'loss_residual': loss_residual,
            'loss_kl': loss_kl,
            'loss_total': total_loss,
        }


class RefinementLossSimple(nn.Module):
    """
    Simplified loss for diffusion refinement (noise prediction only).

    Use this for initial training, then add physics loss later.
    """

    def __init__(self, weight_mse: float = 1.0):
        super().__init__()
        self.weight_mse = weight_mse

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_true: torch.Tensor,
    ) -> dict:
        """Compute simple MSE loss on noise prediction."""
        loss_mse = F.mse_loss(noise_pred, noise_true)

        return {
            'loss': self.weight_mse * loss_mse,
            'loss_noise': loss_mse,
            'loss_physics': torch.tensor(0.0, device=noise_pred.device),
            'loss_residual': torch.tensor(0.0, device=noise_pred.device),
            'loss_kl': torch.tensor(0.0, device=noise_pred.device),
            'loss_total': loss_mse,
        }


class AdaptiveRefinementLoss(nn.Module):
    """
    Adaptive loss that adjusts weights based on training progress.

    Starts with simple MSE, gradually adds physics constraints.
    """

    def __init__(
        self,
        forward_model,
        initial_phase_steps: int = 5000,
        ramp_up_steps: int = 5000,
    ):
        super().__init__()
        self.base_loss = RefinementLoss(
            forward_model=forward_model,
            weight_physics=0.0,
            weight_residual=0.0,
            weight_kl=0.0,
        )
        self.target_weights = {
            'weight_physics': 0.05,
            'weight_residual': 0.1,
            'weight_kl': 0.05,
        }
        self.initial_phase_steps = initial_phase_steps
        self.ramp_up_steps = ramp_up_steps
        self.global_step = 0

    def update_step(self):
        """Increment global step counter."""
        self.global_step += 1

    def _get_adaptive_weights(self) -> dict:
        """Compute adaptive weights based on training progress."""
        if self.global_step < self.initial_phase_steps:
            return {
                'weight_physics': 0.0,
                'weight_residual': 0.0,
                'weight_kl': 0.0,
            }

        progress = min(
            1.0,
            (self.global_step - self.initial_phase_steps) / self.ramp_up_steps
        )

        return {
            'weight_physics': self.target_weights['weight_physics'] * progress,
            'weight_residual': self.target_weights['weight_residual'] * progress,
            'weight_kl': self.target_weights['weight_kl'] * progress,
        }

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_true: torch.Tensor,
        f_refined: torch.Tensor,
        f_base: torch.Tensor,
        signal: torch.Tensor,
    ) -> dict:
        """Compute adaptive combined loss."""
        weights = self._get_adaptive_weights()

        self.base_loss.weight_physics = weights['weight_physics']
        self.base_loss.weight_residual = weights['weight_residual']
        self.base_loss.weight_kl = weights['weight_kl']

        return self.base_loss(
            noise_pred=noise_pred,
            noise_true=noise_true,
            f_refined=f_refined,
            f_base=f_base,
            signal=signal,
        )


def create_loss(
    forward_model,
    loss_type: str = "combined",
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss module.

    Args:
        forward_model: ForwardModel2D instance
        loss_type: One of "combined", "simple", "adaptive"
        **kwargs: Additional arguments for loss constructor

    Returns:
        Loss module
    """
    if loss_type == "combined":
        return RefinementLoss(forward_model, **kwargs)
    elif loss_type == "simple":
        return RefinementLossSimple(**kwargs)
    elif loss_type == "adaptive":
        return AdaptiveRefinementLoss(forward_model, **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    import numpy as np
    from dexsy_core.forward_model import ForwardModel2D

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    forward_model = ForwardModel2D(n_d=64, n_b=64)
    loss_fn = RefinementLoss(forward_model).to(device)

    batch_size = 4
    noise_pred = torch.randn(batch_size, 1, 64, 64).to(device)
    noise_true = torch.randn(batch_size, 1, 64, 64).to(device)
    f_refined = torch.rand(batch_size, 1, 64, 64).to(device)
    f_refined = f_refined / f_refined.sum(dim=(2, 3), keepdim=True)
    f_base = torch.rand(batch_size, 1, 64, 64).to(device)
    f_base = f_base / f_base.sum(dim=(2, 3), keepdim=True)
    signal = torch.rand(batch_size, 1, 64, 64).to(device)

    losses = loss_fn(
        noise_pred=noise_pred,
        noise_true=noise_true,
        f_refined=f_refined,
        f_base=f_base,
        signal=signal,
    )

    print("\nLoss components:")
    for name, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {name}: {value.item():.6f}")

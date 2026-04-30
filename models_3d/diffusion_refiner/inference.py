"""
Inference module for Diffusion Refiner with uncertainty estimation.

This module provides the inference pipeline for the conditional diffusion
refiner, including:
- Single-sample refinement
- Multi-sample refinement with uncertainty estimation
- Confidence maps based on sample variance
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dexsy_core.preprocessing import build_model_inputs
from .config import DiffusionConfig, InferenceConfig
from .scheduler import DDIMScheduler, DDPMScheduler
from .model import ConditionalUNetDenoiser


@dataclass
class RefinementResult:
    """
    Result of diffusion refinement inference.

    Attributes:
        spectrum: Refined spectrum (mean of samples) [1, H, W]
        uncertainty: Uncertainty map (std of samples) [1, H, W]
        samples: Individual samples [N, 1, H, W]
        f_base: Baseline prediction [H, W] (2D)
        dei: DEI of refined spectrum
        dei_base: DEI of baseline
    """
    spectrum: np.ndarray
    uncertainty: np.ndarray
    samples: np.ndarray
    f_base: np.ndarray
    dei: float
    dei_base: float

    def __str__(self) -> str:
        return (
            f"RefinementResult(\n"
            f"  spectrum shape: {self.spectrum.shape},\n"
            f"  uncertainty range: [{self.uncertainty.min():.4f}, {self.uncertainty.max():.4f}],\n"
            f"  n_samples: {len(self.samples)},\n"
            f"  DEI: {self.dei:.4f} (baseline: {self.dei_base:.4f})\n"
            f")"
        )


def normalize_distribution(x: torch.Tensor) -> torch.Tensor:
    """Normalize to valid probability distribution (non-negative, sum-to-1)."""
    x_softplus = F.softplus(x)
    return x_softplus / (x_softplus.sum(dim=(2, 3), keepdim=True) + 1e-8)


class RefinementInference:
    """
    Inference pipeline for diffusion refinement.

    Provides methods for:
    - Single-sample refinement (fast, deterministic)
    - Multi-sample refinement with uncertainty (slow, stochastic)
    - Integration with baseline UNet model
    """

    def __init__(
        self,
        refiner_model: nn.Module,
        baseline_model: nn.Module,
        forward_model,
        diffusion_config: Optional[DiffusionConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        device: str = 'cuda',
    ):
        """
        Args:
            refiner_model: Trained diffusion refiner model
            baseline_model: Pre-trained baseline UNet model
            forward_model: ForwardModel2D instance
            diffusion_config: Diffusion configuration
            inference_config: Inference configuration
            device: Device to run inference on
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.refiner_model = refiner_model.to(self.device)
        self.baseline_model = baseline_model.to(self.device)
        self.forward_model = forward_model

        self.refiner_model.eval()
        self.baseline_model.eval()

        if diffusion_config is None:
            diffusion_config = DiffusionConfig()
        self.diffusion_config = diffusion_config

        if inference_config is None:
            inference_config = InferenceConfig()
        self.inference_config = inference_config

        self.scheduler = DDIMScheduler(diffusion_config, eta=diffusion_config.ddim_eta)
        self.scheduler.setup(self.device)

        from dexsy_core.metrics import compute_dei
        self.compute_dei = compute_dei

    def _build_condition(self, f_base: torch.Tensor, model_input: torch.Tensor) -> torch.Tensor:
        """Build conditioning tensor from baseline and model input."""
        signal_raw = model_input[:, 0:1, :, :]
        signal_log = model_input[:, 1:2, :, :]
        pos_ch = model_input[:, 2:3, :, :]

        condition = torch.cat([f_base, signal_raw, signal_log, pos_ch, f_base], dim=1)
        return condition

    def _get_timesteps_schedule(self, n_steps: int) -> torch.Tensor:
        """Get timesteps for DDIM sampling."""
        return self.scheduler.get_timesteps_schedule(n_steps, strategy="linear")

    @torch.no_grad()
    def _ddim_sample_single(
        self,
        x_T: torch.Tensor,
        condition: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """
        Generate single refined sample using DDIM.

        Args:
            x_T: Initial noise [B, 1, H, W]
            condition: Conditioning [B, 5, H, W]
            n_steps: Number of sampling steps

        Returns:
            Refined spectrum [B, 1, H, W]
        """
        timesteps = self._get_timesteps_schedule(n_steps)

        x_t = x_T

        for i, t in enumerate(timesteps):
            t_batch = torch.tensor([t] * x_T.shape[0], device=self.device)

            noise_pred = self.refiner_model(x_t, condition, t_batch)

            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            x_t = self.scheduler.ddim_step(
                noise_pred,
                t.item(),
                prev_t,
                x_t,
                clip_denoised=True,
            )

        return x_t

    @torch.no_grad()
    def refine(
        self,
        signal: Union[np.ndarray, torch.Tensor],
        f_base: Optional[torch.Tensor] = None,
        model_input: Optional[torch.Tensor] = None,
        n_samples: Optional[int] = None,
        sampling_steps: Optional[int] = None,
        apply_constraints: bool = True,
    ) -> RefinementResult:
        """
        Refine signal with uncertainty estimation.

        Args:
            signal: Input signal [1, 64, 64] or [64, 64]
            f_base: Pre-computed baseline (optional)
            model_input: Pre-computed model input (optional)
            n_samples: Number of samples for uncertainty (default: config.n_samples)
            sampling_steps: Number of DDIM steps (default: config.sampling_steps)
            apply_constraints: Whether to apply sum-to-1 normalization

        Returns:
            RefinementResult with refined spectrum and uncertainty
        """
        if n_samples is None:
            n_samples = self.inference_config.n_samples
        if sampling_steps is None:
            sampling_steps = self.inference_config.sampling_steps

        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal).float()
        signal = signal.to(self.device)

        if signal.dim() == 2:
            signal = signal.unsqueeze(0).unsqueeze(0)
        elif signal.dim() == 3:
            signal = signal.unsqueeze(0)

        if model_input is None:
            signal_np = signal.cpu().numpy()
            model_input_np = build_model_inputs(signal_np, self.forward_model)
            model_input = torch.from_numpy(model_input_np).float().to(self.device)

        if f_base is None:
            model_input_3ch = model_input[:, :3, :, :]
            f_base = self.baseline_model(model_input_3ch)

        condition = self._build_condition(f_base, model_input)

        samples = []

        for i in range(n_samples):
            x_T = torch.randn(1, 1, 64, 64, device=self.device)

            f_refined = self._ddim_sample_single(x_T, condition, sampling_steps)

            if apply_constraints:
                f_refined = normalize_distribution(f_refined)

            samples.append(f_refined.cpu())

        samples = torch.cat(samples, dim=0)

        mean_spectrum = samples.mean(dim=0).cpu().numpy()
        std_map = samples.std(dim=0).cpu().numpy()

        if apply_constraints:
            mean_spectrum = mean_spectrum / (mean_spectrum.sum() + 1e-8)

        mean_spectrum = np.clip(mean_spectrum, 0, None)

        dei_refined = self.compute_dei(mean_spectrum[0])
        dei_base = self.compute_dei(f_base[0, 0].cpu().numpy())

        return RefinementResult(
            spectrum=mean_spectrum,
            uncertainty=std_map,
            samples=samples.numpy(),
            f_base=f_base[0, 0].cpu().numpy(),
            dei=dei_refined,
            dei_base=dei_base,
        )

    @torch.no_grad()
    def refine_batch(
        self,
        signals: Union[np.ndarray, torch.Tensor],
        n_samples: int = 4,
        sampling_steps: int = 50,
    ) -> List[RefinementResult]:
        """
        Refine batch of signals (each with limited samples for speed).

        Args:
            signals: Batch of signals [B, 1, 64, 64]
            n_samples: Number of samples per signal
            sampling_steps: Number of DDIM steps

        Returns:
            List of RefinementResults
        """
        if isinstance(signals, np.ndarray):
            signals = torch.from_numpy(signals).float()
        signals = signals.to(self.device)

        batch_size = signals.shape[0]
        results = []

        for i in range(batch_size):
            signal = signals[i:i+1]
            result = self.refine(
                signal=signal,
                n_samples=n_samples,
                sampling_steps=sampling_steps,
            )
            results.append(result)

        return results

    @torch.no_grad()
    def refine_fast(
        self,
        signal: Union[np.ndarray, torch.Tensor],
        sampling_steps: int = 20,
    ) -> np.ndarray:
        """
        Fast single-sample refinement (for real-time use).

        Uses fewer sampling steps and no averaging.

        Args:
            signal: Input signal
            sampling_steps: Number of DDIM steps (20 for speed)

        Returns:
            Refined spectrum [1, 64, 64]
        """
        result = self.refine(
            signal=signal,
            n_samples=1,
            sampling_steps=sampling_steps,
        )
        return result.spectrum

    def save_samples(
        self,
        result: RefinementResult,
        output_path: str,
        save_individual: bool = False,
    ):
        """
        Save refinement results to files.

        Args:
            result: RefinementResult to save
            output_path: Output directory or file path
            save_individual: Whether to save individual samples
        """
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im0 = axes[0].imshow(result.spectrum[0], cmap='viridis')
        axes[0].set_title(f'Refined (DEI={result.dei:.3f})')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(result.f_base, cmap='viridis')
        axes[1].set_title(f'Baseline (DEI={result.dei_base:.3f})')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(result.uncertainty[0], cmap='hot')
        axes[2].set_title('Uncertainty (std)')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.savefig(output_path / 'refinement_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        if save_individual:
            for i, sample in enumerate(result.samples):
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                im = ax.imshow(sample[0], cmap='viridis')
                ax.set_title(f'Sample {i+1}')
                ax.axis('off')
                plt.colorbar(im, ax=ax)
                plt.savefig(output_path / f'sample_{i:02d}.png', dpi=150, bbox_inches='tight')
                plt.close()

        print(f"Saved results to {output_path}")


def load_refiner(
    refiner_checkpoint: str,
    baseline_checkpoint: str,
    forward_model,
    device: str = 'cuda',
    refiner_base_filters: int = 32,
    refiner_time_dim: int = 128,
    baseline_base_filters: int = 32,
) -> RefinementInference:
    """
    Load refiner model from checkpoints.

    Args:
        refiner_checkpoint: Path to refiner checkpoint
        baseline_checkpoint: Path to baseline UNet checkpoint
        forward_model: ForwardModel2D instance
        device: Device to load models on
        refiner_base_filters: Base filters for refiner model (if not in checkpoint)
        refiner_time_dim: Time embedding dim for refiner (if not in checkpoint)
        baseline_base_filters: Base filters for baseline UNet (if not in checkpoint)

    Returns:
        RefinementInference instance
    """
    from models_3d.attention_unet.model import AttentionUNet3C

    refiner_checkpoint_data = torch.load(refiner_checkpoint, map_location=device)
    baseline_checkpoint_data = torch.load(baseline_checkpoint, map_location=device)

    # Extract configuration from checkpoint if available
    refiner_config = refiner_checkpoint_data.get('config', {})
    baseline_config = baseline_checkpoint_data.get('config', {})

    # Use checkpoint config or fall back to provided values
    rf_base = refiner_config.get('base_filters', refiner_base_filters)
    rf_time = refiner_config.get('time_dim', refiner_time_dim)
    bl_base = baseline_config.get('base_filters', baseline_base_filters)

    refiner = ConditionalUNetDenoiser(base_filters=rf_base, time_dim=rf_time)
    baseline = AttentionUNet3C(in_channels=3, base_filters=bl_base)

    if 'model_state_dict' in refiner_checkpoint_data:
        refiner.load_state_dict(refiner_checkpoint_data['model_state_dict'])
    else:
        refiner.load_state_dict(refiner_checkpoint_data)

    if 'model_state_dict' in baseline_checkpoint_data:
        baseline.load_state_dict(baseline_checkpoint_data['model_state_dict'])
    else:
        baseline.load_state_dict(baseline_checkpoint_data)

    inference = RefinementInference(
        refiner_model=refiner,
        baseline_model=baseline,
        forward_model=forward_model,
        device=device,
    )

    return inference


if __name__ == "__main__":
    from dexsy_core.forward_model import ForwardModel2D
    from models_3d.attention_unet.model import AttentionUNet3C

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    forward_model = ForwardModel2D(n_d=64, n_b=64)

    print("\nGenerating test signal...")
    F, S, _, _ = forward_model.generate_batch(
        n_samples=1,
        noise_sigma=0.01,
        noise_model='rician',
        normalize=True,
        n_compartments=3,
    )
    S = S.reshape(1, 1, 64, 64).astype(np.float32)

    print(f"Signal shape: {S.shape}")
    print(f"Signal range: [{S.min():.4f}, {S.max():.4f}]")

    baseline = AttentionUNet3C(in_channels=3, base_filters=32).to(device)

    refiner = ConditionalUNetDenoiser(base_filters=32, time_dim=128).to(device)

    inference = RefinementInference(
        refiner_model=refiner,
        baseline_model=baseline,
        forward_model=forward_model,
        device=device,
    )

    print("\nRunning inference with 8 samples...")
    result = inference.refine(S, n_samples=8, sampling_steps=50)

    print(f"\n{result}")
    print(f"\nUncertainty statistics:")
    print(f"  Min: {result.uncertainty.min():.6f}")
    print(f"  Max: {result.uncertainty.max():.6f}")
    print(f"  Mean: {result.uncertainty.mean():.6f}")

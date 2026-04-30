"""
Inference module for Uncertainty Estimation via Diffusion Sampling.

This module provides inference pipeline where:
- Baseline UNet provides the primary (deterministic) output
- Diffusion model provides uncertainty estimation through multiple samples
- Uncertainty map helps identify high-risk samples/regions

Main class: UncertaintyEstimator
- predict(): Returns baseline + uncertainty
- predict_with_uncertainty(): Returns detailed result with samples
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

import sys
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dexsy_core.preprocessing import build_model_inputs
from .config import DiffusionConfig, InferenceConfig
from .scheduler import DDIMScheduler
from .model import ConditionalUNetDenoiser


@dataclass
class UncertaintyResult:
    """
    Result of uncertainty estimation.

    The baseline (UNet output) is the primary prediction.
    Diffusion samples are used only for uncertainty quantification.

    Attributes:
        baseline: Primary prediction from UNet [H, W]
        uncertainty: Uncertainty map from diffusion std [H, W]
        samples: Individual diffusion samples [N, 1, H, W]
        dei: DEI of baseline
        confidence: Overall confidence score (inverse of mean uncertainty)
        signal: Input signal [H, W]
    """
    baseline: np.ndarray
    uncertainty: np.ndarray
    samples: np.ndarray
    dei: float
    confidence: float
    signal: np.ndarray

    def __str__(self) -> str:
        return (
            f"UncertaintyResult(\n"
            f"  baseline shape: {self.baseline.shape},\n"
            f"  uncertainty range: [{self.uncertainty.min():.4f}, {self.uncertainty.max():.4f}],\n"
            f"  n_samples: {len(self.samples)},\n"
            f"  DEI: {self.dei:.4f},\n"
            f"  confidence: {self.confidence:.4f}\n"
            f")"
        )

    def get_high_risk_regions(self, threshold: float = 0.8) -> np.ndarray:
        """
        Get binary mask of high-risk regions (>threshold percentile of uncertainty).

        Args:
            threshold: Percentile threshold (0-1)

        Returns:
            Binary mask [H, W]
        """
        threshold_value = np.percentile(self.uncertainty, threshold * 100)
        return (self.uncertainty > threshold_value).astype(np.uint8)


@dataclass
class RefinementResult:
    """
    Backward-compatible result class for legacy RefinementInference API.

    This class wraps UncertaintyResult and provides the old field names
    as aliases to new field names. Deprecation warnings are issued when
    accessing old field names.

    Old fields -> New fields:
        spectrum -> baseline
        f_base -> baseline
        dei_base -> dei

    Usage:
        result = inference.refine(signal)
        # Old API (deprecated):
        print(result.spectrum)    # Same as result.baseline
        print(result.dei_base)    # Same as result.dei
        # New API:
        print(result.baseline)
        print(result.dei)
    """
    _inner: UncertaintyResult = field(default=None)

    def __init__(self, inner: UncertaintyResult):
        self._inner = inner
        self._warned_fields = set()

    @property
    def baseline(self) -> np.ndarray:
        """Primary prediction from baseline UNet."""
        return self._inner.baseline

    @property
    def spectrum(self) -> np.ndarray:
        """Deprecated: Use baseline instead. Returns same value."""
        if 'spectrum' not in self._warned_fields:
            warnings.warn(
                "result.spectrum is deprecated, use result.baseline instead",
                DeprecationWarning,
                stacklevel=2
            )
            self._warned_fields.add('spectrum')
        return self._inner.baseline

    @property
    def f_base(self) -> np.ndarray:
        """Deprecated: Use baseline instead. Returns same value."""
        if 'f_base' not in self._warned_fields:
            warnings.warn(
                "result.f_base is deprecated, use result.baseline instead",
                DeprecationWarning,
                stacklevel=2
            )
            self._warned_fields.add('f_base')
        return self._inner.baseline

    @property
    def uncertainty(self) -> np.ndarray:
        """Uncertainty map from diffusion sampling."""
        return self._inner.uncertainty

    @property
    def samples(self) -> np.ndarray:
        """Individual diffusion samples."""
        return self._inner.samples

    @property
    def dei(self) -> float:
        """DEI of baseline."""
        return self._inner.dei

    @property
    def dei_base(self) -> float:
        """Deprecated: DEI of baseline. Same as dei."""
        if 'dei_base' not in self._warned_fields:
            warnings.warn(
                "result.dei_base is deprecated, use result.dei instead",
                DeprecationWarning,
                stacklevel=2
            )
            self._warned_fields.add('dei_base')
        return self._inner.dei

    @property
    def confidence(self) -> float:
        """Confidence score."""
        return self._inner.confidence

    @property
    def signal(self) -> np.ndarray:
        """Input signal."""
        return self._inner.signal

    def get_high_risk_regions(self, threshold: float = 0.8) -> np.ndarray:
        """Get binary mask of high-risk regions."""
        return self._inner.get_high_risk_regions(threshold)

    def __getattr__(self, name: str):
        """Forward unknown attributes to underlying result."""
        return getattr(self._inner, name)

    def __str__(self) -> str:
        return (
            f"RefinementResult(\n"
            f"  baseline (spectrum, f_base): {self._inner.baseline.shape},\n"
            f"  uncertainty: {self._inner.uncertainty.shape},\n"
            f"  dei (dei_base): {self._inner.dei:.4f},\n"
            f"  confidence: {self._inner.confidence:.4f}\n"
            f")"
        )


@dataclass
class UncertaintyMetrics:
    """Metrics for uncertainty quality."""
    mean_uncertainty: float
    max_uncertainty: float
    uncertainty_std: float
    confidence: float
    coverage: float


def normalize_distribution(x: torch.Tensor) -> torch.Tensor:
    """Normalize to valid probability distribution (non-negative, sum-to-1)."""
    x_softplus = F.softplus(x)
    return x_softplus / (x_softplus.sum(dim=(2, 3), keepdims=True) + 1e-8)


class UncertaintyEstimator:
    """
    Uncertainty estimator using diffusion sampling.

    This class treats the diffusion model as an uncertainty module rather than
    a refinement module. The baseline UNet provides the primary output, and
    diffusion sampling quantifies uncertainty.

    Usage:
        estimator = UncertaintyEstimator(
            baseline_model=unet,
            diffusion_model=refiner,
            forward_model=forward_model,
        )

        result = estimator.predict(signal)
        print(f"DEI: {result.dei:.4f}, Confidence: {result.confidence:.4f}")
        high_risk = result.get_high_risk_regions(threshold=0.9)
    """

    def __init__(
        self,
        baseline_model: nn.Module,
        diffusion_model: nn.Module,
        forward_model,
        diffusion_config: Optional[DiffusionConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        device: str = 'cuda',
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.baseline_model = baseline_model.to(self.device)
        self.diffusion_model = diffusion_model.to(self.device)
        self.forward_model = forward_model

        self.baseline_model.eval()
        self.diffusion_model.eval()

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

        signal_mean = signal_raw.mean(dim=(2, 3), keepdim=True)
        signal_var = ((signal_raw - signal_mean) ** 2).mean(dim=(2, 3), keepdim=True)
        noise_indicator = signal_var.expand_as(signal_raw)

        condition = torch.cat([f_base, signal_raw, signal_log, pos_ch, noise_indicator], dim=1)
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
        Generate single sample using DDIM.

        Args:
            x_T: Initial noise [B, 1, H, W]
            condition: Conditioning [B, 5, H, W]
            n_steps: Number of sampling steps

        Returns:
            Sampled spectrum [B, 1, H, W]
        """
        timesteps = self._get_timesteps_schedule(n_steps)
        x_t = x_T

        for i, t in enumerate(timesteps):
            t_batch = torch.tensor([t] * x_T.shape[0], device=self.device)

            noise_pred = self.diffusion_model(x_t, condition, t_batch)

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
    def predict(
        self,
        signal: Union[np.ndarray, torch.Tensor],
        n_samples: Optional[int] = None,
        sampling_steps: Optional[int] = None,
        return_samples: bool = True,
    ) -> Union[UncertaintyResult, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict with uncertainty estimation.

        The baseline UNet provides the primary prediction.
        Diffusion samples quantify uncertainty (not used for final output).

        Args:
            signal: Input signal [1, 64, 64] or [64, 64]
            n_samples: Number of diffusion samples for uncertainty
            sampling_steps: Number of DDIM steps
            return_samples: If True, return UncertaintyResult; if False, return (baseline, uncertainty)

        Returns:
            Either UncertaintyResult (with full details) or (baseline, uncertainty) tuple
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

        signal_np = signal.cpu().numpy()[0, 0]

        model_input_np = build_model_inputs(signal.cpu().numpy(), self.forward_model)
        model_input = torch.from_numpy(model_input_np).float().to(self.device)

        model_input_3ch = model_input[:, :3, :, :]
        f_base = self.baseline_model(model_input_3ch)

        f_base_np = f_base[0, 0].cpu().numpy()

        condition = self._build_condition(f_base, model_input)

        samples = []
        for i in range(n_samples):
            x_T = torch.randn(1, 1, 64, 64, device=self.device)
            f_sample = self._ddim_sample_single(x_T, condition, sampling_steps)
            f_sample = normalize_distribution(f_sample)
            samples.append(f_sample.cpu())

        samples = torch.cat(samples, dim=0).numpy()
        uncertainty = samples.std(axis=0)[0]

        dei = self.compute_dei(f_base_np)
        confidence = 1.0 / (1.0 + uncertainty.mean())

        if return_samples:
            return UncertaintyResult(
                baseline=f_base_np,
                uncertainty=uncertainty,
                samples=samples,
                dei=dei,
                confidence=confidence,
                signal=signal_np,
            )
        else:
            return f_base_np, uncertainty

    @torch.no_grad()
    def predict_batch(
        self,
        signals: Union[np.ndarray, torch.Tensor],
        n_samples: int = 4,
        sampling_steps: int = 30,
    ) -> List[UncertaintyResult]:
        """
        Predict batch of signals with uncertainty.

        Args:
            signals: Batch of signals [B, 1, 64, 64]
            n_samples: Number of diffusion samples per signal
            sampling_steps: Number of DDIM steps

        Returns:
            List of UncertaintyResults
        """
        if isinstance(signals, np.ndarray):
            signals = torch.from_numpy(signals).float()
        signals = signals.to(self.device)

        batch_size = signals.shape[0]
        results = []

        for i in range(batch_size):
            signal = signals[i:i+1]
            result = self.predict(
                signal=signal,
                n_samples=n_samples,
                sampling_steps=sampling_steps,
                return_samples=True,
            )
            results.append(result)

        return results

    def evaluate_uncertainty(
        self,
        signals: np.ndarray,
        ground_truths: np.ndarray,
        n_samples: int = 8,
        sampling_steps: int = 50,
    ) -> dict:
        """
        Evaluate uncertainty quality.

        Checks if high uncertainty regions correlate with high error.

        Args:
            signals: Test signals [N, 1, 64, 64]
            ground_truths: Ground truth spectra [N, 1, 64, 64]
            n_samples: Number of diffusion samples
            sampling_steps: Number of DDIM steps

        Returns:
            Dictionary with calibration metrics
        """
        results = self.predict_batch(signals, n_samples, sampling_steps)

        errors = []
        uncertainties = []

        for i, result in enumerate(results):
            gt = ground_truths[i, 0]
            pred = result.baseline
            error = np.abs(pred - gt)
            errors.append(error)
            uncertainties.append(result.uncertainty)

        errors = np.stack(errors)
        uncertainties = np.stack(uncertainties)

        metrics = {}
        metrics['error_mean'] = errors.mean()
        metrics['error_std'] = errors.std()
        metrics['uncertainty_mean'] = uncertainties.mean()
        metrics['uncertainty_std'] = uncertainties.std()

        pixel_errors_flat = errors.reshape(len(errors), -1)
        pixel_uncertainties_flat = uncertainties.reshape(len(uncertainties), -1)

        correlation = np.corrcoef(
            pixel_errors_flat.mean(axis=0),
            pixel_uncertainties_flat.mean(axis=0)
        )[0, 1]
        metrics['error_uncertainty_correlation'] = correlation

        return metrics


class RefinementInference(UncertaintyEstimator):
    """
    Backward-compatible wrapper for RefinementInference.

    This class maintains the old API where:
    - result.spectrum returns baseline (UNet output)
    - result.f_base returns baseline
    - result.dei_base returns DEI of baseline
    - result.uncertainty returns uncertainty map
    - result.dei returns DEI of baseline

    Deprecation warnings are issued when using old field names.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        Backward-compatible refine method.

        Returns RefinementResult (wrapper around UncertaintyResult) with:
        - result.baseline = UNet prediction (same as old result.spectrum)
        - result.uncertainty = uncertainty from diffusion
        - result.dei = DEI of baseline

        Args:
            signal: Input signal
            f_base: Ignored (kept for backward compatibility)
            model_input: Ignored (kept for backward compatibility)
            n_samples: Number of samples
            sampling_steps: DDIM steps
            apply_constraints: Ignored

        Returns:
            RefinementResult with backward-compatible field access
        """
        uncertainty_result = self.predict(
            signal=signal,
            n_samples=n_samples,
            sampling_steps=sampling_steps,
            return_samples=True,
        )
        return RefinementResult(uncertainty_result)

    def refine_batch(
        self,
        signals: Union[np.ndarray, torch.Tensor],
        n_samples: int = 4,
        sampling_steps: int = 50,
    ) -> List[RefinementResult]:
        """Backward-compatible batch refine."""
        uncertainty_results = self.predict_batch(signals, n_samples, sampling_steps)
        return [RefinementResult(r) for r in uncertainty_results]


def load_estimator(
    baseline_checkpoint: str,
    diffusion_checkpoint: str,
    forward_model,
    device: str = 'cuda',
    baseline_base_filters: int = 32,
    diffusion_base_filters: int = 32,
    diffusion_time_dim: int = 128,
) -> UncertaintyEstimator:
    """
    Load uncertainty estimator from checkpoints.

    Args:
        baseline_checkpoint: Path to baseline UNet checkpoint
        diffusion_checkpoint: Path to diffusion model checkpoint
        forward_model: ForwardModel2D instance
        device: Device to load models on
        baseline_base_filters: Base filters for baseline UNet
        diffusion_base_filters: Base filters for diffusion model
        diffusion_time_dim: Time embedding dim for diffusion

    Returns:
        UncertaintyEstimator instance
    """
    from models_3d.attention_unet.model import AttentionUNet3C

    baseline_checkpoint_data = torch.load(baseline_checkpoint, map_location=device)
    diffusion_checkpoint_data = torch.load(diffusion_checkpoint, map_location=device)

    bl_config = baseline_checkpoint_data.get('config', {})
    df_config = diffusion_checkpoint_data.get('config', {})

    bl_base = bl_config.get('base_filters', baseline_base_filters)
    df_base = df_config.get('base_filters', diffusion_base_filters)
    df_time = df_config.get('time_dim', diffusion_time_dim)

    baseline = AttentionUNet3C(in_channels=3, base_filters=bl_base)
    diffusion = ConditionalUNetDenoiser(base_filters=df_base, time_dim=df_time)

    if 'model_state_dict' in baseline_checkpoint_data:
        baseline.load_state_dict(baseline_checkpoint_data['model_state_dict'])
    else:
        baseline.load_state_dict(baseline_checkpoint_data)

    if 'model_state_dict' in diffusion_checkpoint_data:
        diffusion.load_state_dict(diffusion_checkpoint_data['model_state_dict'])
    else:
        diffusion.load_state_dict(diffusion_checkpoint_data)

    return UncertaintyEstimator(
        baseline_model=baseline,
        diffusion_model=diffusion,
        forward_model=forward_model,
        device=device,
    )


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
    Load refiner inference pipeline from checkpoints.

    Returns RefinementInference which provides backward-compatible API.

    Args:
        refiner_checkpoint: Path to diffusion model checkpoint
        baseline_checkpoint: Path to baseline UNet checkpoint
        forward_model: ForwardModel2D instance
        device: Device to load models on
        refiner_base_filters: Base filters for diffusion model
        refiner_time_dim: Time embedding dim for diffusion
        baseline_base_filters: Base filters for baseline UNet

    Returns:
        RefinementInference instance with backward-compatible API
    """
    from models_3d.attention_unet.model import AttentionUNet3C

    baseline_checkpoint_data = torch.load(baseline_checkpoint, map_location=device)
    diffusion_checkpoint_data = torch.load(refiner_checkpoint, map_location=device)

    bl_config = baseline_checkpoint_data.get('config', {})
    df_config = diffusion_checkpoint_data.get('config', {})

    bl_base = bl_config.get('base_filters', baseline_base_filters)
    df_base = df_config.get('base_filters', refiner_base_filters)
    df_time = df_config.get('time_dim', refiner_time_dim)

    baseline = AttentionUNet3C(in_channels=3, base_filters=bl_base)
    diffusion = ConditionalUNetDenoiser(base_filters=df_base, time_dim=df_time)

    if 'model_state_dict' in baseline_checkpoint_data:
        baseline.load_state_dict(baseline_checkpoint_data['model_state_dict'])
    else:
        baseline.load_state_dict(baseline_checkpoint_data)

    if 'model_state_dict' in diffusion_checkpoint_data:
        diffusion.load_state_dict(diffusion_checkpoint_data['model_state_dict'])
    else:
        diffusion.load_state_dict(diffusion_checkpoint_data)

    return RefinementInference(
        baseline_model=baseline,
        diffusion_model=diffusion,
        forward_model=forward_model,
        device=device,
    )


if __name__ == "__main__":
    from dexsy_core.forward_model import ForwardModel2D
    from models_3d.attention_unet.model import AttentionUNet3C

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    forward_model = ForwardModel2D(n_d=64, n_b=64)

    print("\nGenerating test signal...")
    F, S, _ = forward_model.generate_batch(
        n_samples=1,
        noise_sigma=0.01,
        normalize=True,
        n_compartments=3,
    )
    S = S.reshape(1, 1, 64, 64).astype(np.float32)

    print(f"Signal shape: {S.shape}")

    baseline = AttentionUNet3C(in_channels=3, base_filters=32).to(device)
    from models_3d.diffusion_refiner.model import ConditionalUNetDenoiser
    diffusion = ConditionalUNetDenoiser(base_filters=32, time_dim=128).to(device)

    estimator = UncertaintyEstimator(
        baseline_model=baseline,
        diffusion_model=diffusion,
        forward_model=forward_model,
        device=device,
    )

    print("\nRunning prediction with uncertainty...")
    result = estimator.predict(S, n_samples=8, sampling_steps=30, return_samples=True)

    print(f"\n{result}")
    print(f"\nUncertainty statistics:")
    print(f"  Min: {result.uncertainty.min():.6f}")
    print(f"  Max: {result.uncertainty.max():.6f}")
    print(f"  Mean: {result.uncertainty.mean():.6f}")
    print(f"  Confidence: {result.confidence:.4f}")

    high_risk = result.get_high_risk_regions(threshold=0.9)
    print(f"  High-risk pixels: {high_risk.sum()} ({high_risk.mean()*100:.1f}%)")

"""
Evaluation module for Diffusion Refiner.

This module provides evaluation utilities for comparing the diffusion refiner
against the baseline UNet model.

Usage:
    from models_3d.diffusion_refiner.evaluate import evaluate_refiner

    results = evaluate_refiner(
        refiner_model=refiner,
        baseline_model=unet,
        test_signals=signals,
        ground_truth=spectra,
        forward_model=forward_model,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt

import sys
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dexsy_core.forward_model import ForwardModel2D
from dexsy_core.preprocessing import build_model_inputs
from dexsy_core.metrics import compute_dei

from .config import InferenceConfig
from .inference import RefinementInference, RefinementResult
from .model import ConditionalUNetDenoiser


def _normalize_spectrum(x: np.ndarray) -> np.ndarray:
    """
    Normalize spectrum array to valid probability distribution.

    Handles shapes [H,W], [1,H,W], or [N,1,H,W] by squeezing extra dimensions.

    Returns:
        Normalized array with shape [N,H,W] for batch or [H,W] for single
    """
    x = np.array(x)

    # Handle scalar case
    if x.ndim == 0:
        return x

    # Remove singleton dimensions
    while x.ndim > 2 and x.shape[-3] == 1:
        x = x.squeeze(axis=-3)

    # Ensure at least 3D for processing
    if x.ndim == 2:
        x = x[np.newaxis, ...]  # [1,H,W]

    return x


@dataclass
class ComparisonResult:
    """
    Result of comparing baseline vs refinement.

    Attributes:
        baseline_metrics: Dictionary of baseline metrics
        refinement_metrics: Dictionary of refinement metrics
        improvement: Improvement percentages
        uncertainty_stats: Statistics on uncertainty maps
    """
    baseline_metrics: dict
    refinement_metrics: dict
    improvement: dict
    uncertainty_stats: dict
    predictions: dict = field(default_factory=dict)


def compute_spectrum_metrics(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
) -> dict:
    """
    Compute metrics for spectrum predictions.

    Args:
        predictions: Predicted spectra [N, 1, H, W] or [N, H, W] (2D)
        ground_truths: Ground truth spectra [N, 1, H, W] or [N, H, W]

    Returns:
        Dictionary of metrics
    """
    # Handle 2D baseline predictions by adding channel dimension
    if predictions.ndim == 3:
        predictions = predictions[:, np.newaxis, :, :]

    predictions = np.clip(predictions, 0, None)
    predictions = predictions / (predictions.sum(axis=(1, 2, 3), keepdims=True) + 1e-8)

    ground_truths = np.clip(ground_truths, 0, None)
    ground_truths = ground_truths / (ground_truths.sum(axis=(1, 2, 3), keepdims=True) + 1e-8)

    mse = np.mean((predictions - ground_truths) ** 2, axis=(1, 2, 3))
    mae = np.mean(np.abs(predictions - ground_truths), axis=(1, 2, 3))

    dei_pred = np.array([compute_dei(p[0]) for p in predictions])
    dei_true = np.array([compute_dei(g[0]) for g in ground_truths])
    dei_error = np.abs(dei_pred - dei_true)

    return {
        'mse': mse,
        'mae': mae,
        'dei_predicted': dei_pred,
        'dei_true': dei_true,
        'dei_error': dei_error,
        'dei_error_mean': dei_error.mean(),
        'dei_error_std': dei_error.std(),
        'dei_error_p95': np.percentile(dei_error, 95),
        'dei_error_median': np.median(dei_error),
    }


def compute_improvement(baseline_metrics: dict, refinement_metrics: dict) -> dict:
    """
    Compute improvement of refinement over baseline.

    Args:
        baseline_metrics: Metrics for baseline predictions
        refinement_metrics: Metrics for refined predictions

    Returns:
        Dictionary of improvement percentages
    """
    improvement = {}

    for metric in ['mse', 'mae', 'dei_error']:
        base_mean = np.mean(baseline_metrics[metric])
        ref_mean = np.mean(refinement_metrics[metric])

        if base_mean > 0:
            improvement[f'{metric}_improvement_mean'] = (base_mean - ref_mean) / base_mean * 100
        else:
            improvement[f'{metric}_improvement_mean'] = 0.0

        base_p95 = np.percentile(baseline_metrics[metric], 95)
        ref_p95 = np.percentile(refinement_metrics[metric], 95)

        if base_p95 > 0:
            improvement[f'{metric}_improvement_p95'] = (base_p95 - ref_p95) / base_p95 * 100
        else:
            improvement[f'{metric}_improvement_p95'] = 0.0

    return improvement


def evaluate_refiner(
    refiner_model: ConditionalUNetDenoiser,
    baseline_model,
    forward_model: ForwardModel2D,
    test_signals: np.ndarray,
    ground_truths: np.ndarray,
    n_samples: int = 8,
    sampling_steps: int = 50,
    device: str = 'cuda',
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> ComparisonResult:
    """
    Evaluate the diffusion refiner against baseline.

    Args:
        refiner_model: Trained diffusion refiner
        baseline_model: Baseline UNet model
        forward_model: ForwardModel2D instance
        test_signals: Test signals [N, 1, 64, 64]
        ground_truths: Ground truth spectra [N, 1, 64, 64]
        n_samples: Number of samples for uncertainty
        sampling_steps: DDIM sampling steps
        device: Device to run on
        output_dir: Directory to save plots
        verbose: Whether to print progress

    Returns:
        ComparisonResult with metrics for both methods
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Evaluating Uncertainty Estimation")
        print(f"{'='*60}")
        print(f"Test samples: {len(test_signals)}")
        print(f"n_samples: {n_samples}, sampling_steps: {sampling_steps}")

    from .inference import UncertaintyEstimator
    inference = UncertaintyEstimator(
        baseline_model=baseline_model,
        diffusion_model=refiner_model,
        forward_model=forward_model,
        device=device,
    )

    baseline_predictions = []
    uncertainty_maps = []
    results = []

    for i in range(len(test_signals)):
        signal = test_signals[i:i+1]

        if verbose and (i + 1) % 50 == 0:
            print(f"  Processing sample {i+1}/{len(test_signals)}")

        result = inference.predict(
            signal=signal,
            n_samples=n_samples,
            sampling_steps=sampling_steps,
            return_samples=True,
        )

        baseline_predictions.append(result.baseline)
        uncertainty_maps.append(result.uncertainty)
        results.append(result)

    baseline_predictions = np.stack(baseline_predictions)
    uncertainty_maps = np.stack(uncertainty_maps)

    # Normalize ground_truths to [N,H,W] (handle [N,1,H,W] or [1,H,W])
    ground_truths_norm = _normalize_spectrum(ground_truths)

    if verbose:
        print("\nComputing metrics...")

    # Baseline metrics (UNet primary output)
    baseline_metrics = compute_spectrum_metrics(baseline_predictions, ground_truths)

    # Uncertainty statistics
    uncertainty_stats = {
        'mean': uncertainty_maps.mean(),
        'std': uncertainty_maps.std(),
        'min': uncertainty_maps.min(),
        'max': uncertainty_maps.max(),
    }

    # Uncertainty correlation with error (use normalized ground_truths)
    errors = np.abs(baseline_predictions - ground_truths_norm)
    correlation = np.corrcoef(
        errors.reshape(len(errors), -1).mean(axis=0),
        uncertainty_maps.reshape(len(uncertainty_maps), -1).mean(axis=0)
    )[0, 1]

    result = ComparisonResult(
        baseline_metrics=baseline_metrics,
        refinement_metrics={},  # No longer comparing refinement
        improvement={},  # No longer comparing refinement
        uncertainty_stats={**uncertainty_stats, 'error_correlation': correlation},
        predictions={
            'baseline': baseline_predictions,
            'uncertainty': uncertainty_maps,
            'ground_truth': ground_truths_norm,
        }
    )

    if verbose:
        print_summary_new(result)

    if output_dir:
        save_results_new(result, output_dir)

    return result


def print_summary(result: ComparisonResult):
    """Print a summary of the comparison results (legacy format)."""
    print_summary_new(result)


def print_summary_new(result: ComparisonResult):
    """Print a summary of uncertainty estimation results."""
    bm = result.baseline_metrics

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY - Uncertainty Estimation")
    print(f"{'='*60}")

    print("\nBaseline (UNet - Primary Output):")
    print(f"  MSE:     {bm['mse'].mean():.6f} +/- {bm['mse'].std():.6f}")
    print(f"  MAE:     {bm['mae'].mean():.6f} +/- {bm['mae'].std():.6f}")
    print(f"  DEI Err: {bm['dei_error'].mean():.6f} +/- {bm['dei_error'].std():.6f}")
    print(f"  DEI p95: {np.percentile(bm['dei_error'], 95):.6f}")

    print("\nUncertainty Statistics (from Diffusion Sampling):")
    print(f"  Mean: {result.uncertainty_stats['mean']:.6f}")
    print(f"  Std:  {result.uncertainty_stats['std']:.6f}")
    print(f"  Max:  {result.uncertainty_stats['max']:.6f}")

    if 'error_correlation' in result.uncertainty_stats:
        print(f"\nUncertainty Calibration:")
        print(f"  Error-Uncertainty Correlation: {result.uncertainty_stats['error_correlation']:.4f}")
        if result.uncertainty_stats['error_correlation'] > 0.3:
            print("  [Good] Positive correlation suggests uncertainty is informative")
        else:
            print("  [Warning] Low correlation - uncertainty may not reflect true error")


def save_results(result: ComparisonResult, output_dir: str):
    """Save comparison results to files (legacy format)."""
    save_results_new(result, output_dir)


def save_results_new(result: ComparisonResult, output_dir: str):
    """Save comparison results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = result.predictions
    np.savez(
        output_dir / "predictions.npz",
        baseline=predictions.get('baseline'),
        uncertainty=predictions.get('uncertainty'),
        ground_truth=predictions.get('ground_truth'),
    )

    import pandas as pd

    baseline_df = pd.DataFrame({
        'mse': result.baseline_metrics['mse'],
        'mae': result.baseline_metrics['mae'],
        'dei_error': result.baseline_metrics['dei_error'],
    })
    baseline_df.to_csv(output_dir / "baseline_metrics.csv", index=False)

    summary = {
        'metric': ['MSE_mean', 'MSE_std', 'DEI_error_mean', 'DEI_error_p95'],
        'baseline': [
            result.baseline_metrics['mse'].mean(),
            result.baseline_metrics['mse'].std(),
            result.baseline_metrics['dei_error'].mean(),
            np.percentile(result.baseline_metrics['dei_error'], 95),
        ],
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    print(f"\nSaved results to {output_dir}")


def plot_comparison_samples(
    result: ComparisonResult,
    n_samples: int = 5,
    output_dir: Optional[str] = None,
):
    """
    Plot comparison of baseline prediction with uncertainty.

    Shows: Ground Truth, Baseline, Error, Uncertainty, High-Risk Regions

    Args:
        result: ComparisonResult from evaluate_refiner
        n_samples: Number of samples to plot
        output_dir: Directory to save plots
    """
    predictions = result.predictions
    baseline = predictions['baseline']  # Shape: [N, H, W]
    uncertainty = predictions['uncertainty']  # Shape: [N, H, W]
    ground_truth = predictions['ground_truth']  # Shape: [N, H, W]

    n_show = min(n_samples, len(baseline))

    fig, axes = plt.subplots(n_show, 5, figsize=(20, 4 * n_show))

    if n_show == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_show):
        vmax = float(ground_truth[i].max())

        # Ground Truth
        axes[i, 0].imshow(ground_truth[i], cmap='magma', origin='lower', vmin=0, vmax=vmax)
        axes[i, 0].set_title(f'Ground Truth\nDEI={compute_dei(ground_truth[i]):.3f}')
        axes[i, 0].axis('off')

        # Baseline prediction
        axes[i, 1].imshow(baseline[i], cmap='magma', origin='lower', vmin=0, vmax=vmax)
        axes[i, 1].set_title(f'Baseline\nDEI={compute_dei(baseline[i]):.3f}')
        axes[i, 1].axis('off')

        # Error map
        error_map = np.abs(baseline[i] - ground_truth[i])
        axes[i, 2].imshow(error_map, cmap='hot', origin='lower')
        axes[i, 2].set_title(f'Error Map\nMAE={error_map.mean():.4f}')
        axes[i, 2].axis('off')

        # Uncertainty map
        axes[i, 3].imshow(uncertainty[i], cmap='hot', origin='lower')
        axes[i, 3].set_title(f'Uncertainty\nMean={uncertainty[i].mean():.4f}')
        axes[i, 3].axis('off')

        # High-risk regions (top 10% uncertainty)
        threshold_value = np.percentile(uncertainty[i], 90)
        high_risk = (uncertainty[i] > threshold_value).astype(np.float32)
        axes[i, 4].imshow(high_risk, cmap='Reds', origin='lower', vmin=0, vmax=1)
        axes[i, 4].set_title(f'High-Risk ({high_risk.mean()*100:.1f}%)')
        axes[i, 4].axis('off')

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "comparison_samples.png", dpi=150, bbox_inches='tight')
        print(f"Saved comparison plots to {output_dir}")

    plt.close()


def plot_error_distribution(
    result: ComparisonResult,
    output_dir: Optional[str] = None,
):
    """
    Plot error distributions for baseline predictions.

    Args:
        result: ComparisonResult from evaluate_refiner
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    bm = result.baseline_metrics

    # MSE distribution
    axes[0].hist(bm['mse'], bins=50, alpha=0.7, label='Baseline MSE', color='blue')
    axes[0].set_xlabel('MSE')
    axes[0].set_ylabel('Count')
    axes[0].set_title('MSE Distribution')
    axes[0].legend()

    # DEI Error distribution
    axes[1].hist(bm['dei_error'], bins=50, alpha=0.7, label='Baseline DEI Error', color='blue')
    axes[1].set_xlabel('DEI Error')
    axes[1].set_ylabel('Count')
    axes[1].set_title('DEI Error Distribution')
    axes[1].legend()

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "error_distribution.png", dpi=150, bbox_inches='tight')
        print(f"Saved error distribution to {output_dir}")

    plt.close()


def plot_uncertainty_calibration(
    result: ComparisonResult,
    output_dir: Optional[str] = None,
):
    """
    Plot uncertainty calibration.

    Checks if high uncertainty regions correlate with high error.

    Args:
        result: ComparisonResult from evaluate_refiner
        output_dir: Directory to save plots
    """
    predictions = result.predictions
    uncertainty = predictions['uncertainty']  # Shape: [N, H, W]
    baseline = predictions['baseline']  # Shape: [N, H, W]
    ground_truth = predictions['ground_truth']  # Shape: [N, H, W]

    # Error between baseline prediction and ground truth
    pixel_errors = np.abs(baseline - ground_truth).reshape(len(baseline), -1)
    pixel_uncertainty = uncertainty.reshape(len(uncertainty), -1)

    # Bin pixels by uncertainty and compute mean error per bin
    sorted_indices = np.argsort(pixel_uncertainty.mean(axis=0))
    n_bins = 10
    bin_size = len(sorted_indices) // n_bins

    mean_error = []
    mean_uncertainty = []

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(sorted_indices)
        indices = sorted_indices[start:end]

        mean_error.append(pixel_errors[:, indices].mean())
        mean_uncertainty.append(pixel_uncertainty[:, indices].mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(mean_uncertainty, mean_error)
    axes[0].set_xlabel('Mean Uncertainty')
    axes[0].set_ylabel('Mean Pixel Error')
    axes[0].set_title('Uncertainty vs Error')

    if mean_uncertainty[-1] > mean_uncertainty[0]:
        axes[0].set_xlim(0, None)
        axes[0].set_ylim(0, None)

    sorted_uncertainty = np.argsort(mean_uncertainty)
    axes[1].bar(range(n_bins), [mean_error[i] for i in sorted_uncertainty])
    axes[1].set_xticks(range(n_bins))
    axes[1].set_xticklabels([f'{mean_uncertainty[i]:.4f}' for i in sorted_uncertainty], rotation=45)
    axes[1].set_xlabel('Uncertainty Bin')
    axes[1].set_ylabel('Mean Pixel Error')
    axes[1].set_title('Error by Uncertainty Bin')

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "uncertainty_calibration.png", dpi=150, bbox_inches='tight')
        print(f"Saved uncertainty calibration to {output_dir}")

    plt.close()


if __name__ == "__main__":
    import torch
    from dexsy_core.forward_model import ForwardModel2D
    from models_3d.attention_unet.model import AttentionUNet3C
    from .model import ConditionalUNetDenoiser

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    forward_model = ForwardModel2D(n_d=64, n_b=64)

    print("\nGenerating test data...")
    n_test = 100
    F, S, _, _ = forward_model.generate_batch(
        n_samples=n_test,
        noise_sigma=0.01,
        noise_model='rician',
        normalize=True,
        n_compartments=3,
    )
    S = S.reshape(-1, 1, 64, 64).astype(np.float32)
    F = F.reshape(-1, 1, 64, 64).astype(np.float32)

    baseline = AttentionUNet3C(in_channels=3, base_filters=32).to(device)
    refiner = ConditionalUNetDenoiser(base_filters=32, time_dim=128).to(device)

    print("\nRunning evaluation (this will use random weights for demo)...")
    result = evaluate_refiner(
        refiner_model=refiner,
        baseline_model=baseline,
        forward_model=forward_model,
        test_signals=S[:50],
        ground_truths=F[:50],
        n_samples=4,
        sampling_steps=30,
        device=device,
        verbose=True,
    )

    print("\nDone!")

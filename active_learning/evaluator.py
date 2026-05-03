"""
Extended Evaluation for Active Learning.

This module provides extended evaluation metrics including:
- Standard metrics (MSE, MAE, DEI error)
- Worst-case metrics (95th percentile)
- Per-failure-type metrics
- Comparison tables across rounds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import sys
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dexsy_core.forward_model import ForwardModel2D, compute_dei
from dexsy_core.metrics import compute_mse, compute_mae
from .data_protocol import SplitData, SampleMetadata


@dataclass
class ExtendedMetrics:
    """Container for extended evaluation metrics."""
    # Standard metrics
    mse_mean: float
    mae_mean: float
    dei_error_mean: float

    # Worst-case metrics (95th percentile)
    mse_p95: float
    mae_p95: float
    dei_error_p95: float

    # Per-failure-type metrics
    failure_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    # Additional statistics
    mse_std: float = 0.0
    mae_std: float = 0.0
    dei_error_std: float = 0.0
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'mse_mean': self.mse_mean,
            'mae_mean': self.mae_mean,
            'dei_error_mean': self.dei_error_mean,
            'mse_p95': self.mse_p95,
            'mae_p95': self.mae_p95,
            'dei_error_p95': self.dei_error_p95,
            'mse_std': self.mse_std,
            'mae_std': self.mae_std,
            'dei_error_std': self.dei_error_std,
            'n_samples': self.n_samples,
        }
        for ftype, metrics in self.failure_metrics.items():
            for metric_name, value in metrics.items():
                result[f'{ftype}_{metric_name}'] = value
        return result


@dataclass
class RoundMetrics:
    """Metrics for a single AL round."""
    round_idx: int
    checkpoint_path: Path | None
    metrics: ExtendedMetrics
    training_time: float
    n_augmented: int


class ALEvaluator:
    """
    Extended evaluator for active learning rounds.

    Computes standard metrics, worst-case metrics, and per-failure-type metrics.
    """

    # Failure type thresholds for classification
    FAILURE_THRESHOLDS = {
        'high_exchange': {'threshold': 10.0, 'comparison': 'gt'},
        'small_separation': {'threshold': 8, 'comparison': 'lt'},
        'high_noise': {'threshold': 0.012, 'comparison': 'gt'},
        'low_noise': {'threshold': 0.007, 'comparison': 'lt'},
        'high_dei': {'threshold': 0.3, 'comparison': 'gt'},
        'low_dei': {'threshold': 0.05, 'comparison': 'lt'},
    }

    def __init__(
        self,
        forward_model: ForwardModel2D | None = None,
    ):
        """
        Initialize the evaluator.

        Args:
            forward_model: ForwardModel2D instance.
        """
        self.forward_model = forward_model or ForwardModel2D(n_d=64, n_b=64)

    def compute_metrics(
        self,
        predictions: np.ndarray,
        ground_truths: np.ndarray,
        metadata: list[SampleMetadata],
    ) -> ExtendedMetrics:
        """
        Compute extended metrics for predictions.

        Args:
            predictions: Predicted spectra (N, 64, 64).
            ground_truths: Ground truth spectra (N, 64, 64).
            metadata: List of SampleMetadata.

        Returns:
            ExtendedMetrics instance.
        """
        n_samples = len(predictions)

        # Compute per-sample metrics
        mse_scores = []
        mae_scores = []
        dei_errors = []

        for i in range(n_samples):
            pred = predictions[i]
            gt = ground_truths[i]

            mse_scores.append(compute_mse(pred, gt))
            mae_scores.append(compute_mae(pred, gt))

            dei_gt = metadata[i].theoretical_dei
            dei_pred = compute_dei(pred)
            dei_errors.append(abs(dei_gt - dei_pred))

        mse_scores = np.array(mse_scores)
        mae_scores = np.array(mae_scores)
        dei_errors = np.array(dei_errors)

        # Compute standard metrics
        metrics = ExtendedMetrics(
            mse_mean=float(np.mean(mse_scores)),
            mae_mean=float(np.mean(mae_scores)),
            dei_error_mean=float(np.mean(dei_errors)),
            mse_p95=float(np.percentile(mse_scores, 95)),
            mae_p95=float(np.percentile(mae_scores, 95)),
            dei_error_p95=float(np.percentile(dei_errors, 95)),
            mse_std=float(np.std(mse_scores)),
            mae_std=float(np.std(mae_scores)),
            dei_error_std=float(np.std(dei_errors)),
            n_samples=n_samples,
        )

        # Compute per-failure-type metrics
        metrics.failure_metrics = self._compute_failure_metrics(
            predictions, ground_truths, metadata, mse_scores, mae_scores, dei_errors
        )

        return metrics

    def _compute_failure_metrics(
        self,
        predictions: np.ndarray,
        ground_truths: np.ndarray,
        metadata: list[SampleMetadata],
        mse_scores: np.ndarray,
        mae_scores: np.ndarray,
        dei_errors: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """
        Compute metrics broken down by failure type.

        Args:
            predictions: Predicted spectra.
            ground_truths: Ground truth spectra.
            metadata: List of SampleMetadata.
            mse_scores: Pre-computed MSE scores.
            mae_scores: Pre-computed MAE scores.
            dei_errors: Pre-computed DEI errors.

        Returns:
            Dictionary mapping failure type to metrics.
        """
        failure_metrics = {}

        # Classify samples into failure types
        for ftype, config in self.FAILURE_THRESHOLDS.items():
            threshold = config['threshold']
            comparison = config['comparison']

            # Get indices for this failure type
            if ftype == 'high_exchange':
                mask = np.array([
                    m.get_total_exchange_rate() > threshold
                    for m in metadata
                ])
            elif ftype == 'small_separation':
                mask = np.array([
                    m.get_peak_separation() < threshold
                    for m in metadata
                ])
            elif ftype == 'high_noise':
                mask = np.array([m.noise_sigma > threshold for m in metadata])
            elif ftype == 'low_noise':
                mask = np.array([m.noise_sigma < threshold for m in metadata])
            elif ftype == 'high_dei':
                mask = np.array([m.theoretical_dei > threshold for m in metadata])
            elif ftype == 'low_dei':
                mask = np.array([m.theoretical_dei < threshold for m in metadata])
            else:
                continue

            if mask.sum() == 0:
                continue

            # Compute metrics for this failure type
            failure_metrics[ftype] = {
                'n_samples': int(mask.sum()),
                'mse_mean': float(np.mean(mse_scores[mask])),
                'mae_mean': float(np.mean(mae_scores[mask])),
                'dei_error_mean': float(np.mean(dei_errors[mask])),
                'mse_p95': float(np.percentile(mse_scores[mask], 95)),
                'mae_p95': float(np.percentile(mae_scores[mask], 95)),
            }

        return failure_metrics

    def evaluate_split(
        self,
        model: torch.nn.Module,
        split: SplitData,
        device: torch.device,
    ) -> tuple[np.ndarray, ExtendedMetrics]:
        """
        Evaluate a model on a data split.

        Args:
            model: PyTorch model.
            split: SplitData instance.
            device: Device to use.

        Returns:
            Tuple of (predictions, metrics).
        """
        model.eval()
        predictions = []

        # Use model_inputs (3-channel) for evaluation
        n_samples = split.model_inputs.shape[0]
        batch_size = 32

        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch = torch.from_numpy(split.model_inputs[start:end]).to(device)

                preds = model(batch).cpu().numpy()
                predictions.append(preds)

        predictions = np.concatenate(predictions, axis=0)[:, 0]  # Remove channel dim

        metrics = self.compute_metrics(
            predictions=predictions,
            ground_truths=split.labels[:, 0],
            metadata=split.metadata,
        )

        return predictions, metrics

    def compare_rounds(
        self,
        round_metrics: list[RoundMetrics],
    ) -> pd.DataFrame:
        """
        Create a comparison table across rounds.

        Args:
            round_metrics: List of RoundMetrics for each round.

        Returns:
            DataFrame with comparison table.
        """
        rows = []

        for rm in round_metrics:
            row = {
                'round': rm.round_idx,
                'n_augmented': rm.n_augmented,
                'training_time': f"{rm.training_time:.1f}s",
                'mse_mean': f"{rm.metrics.mse_mean:.6f}",
                'mae_mean': f"{rm.metrics.mae_mean:.6f}",
                'dei_error_mean': f"{rm.metrics.dei_error_mean:.6f}",
                'mse_p95': f"{rm.metrics.mse_p95:.6f}",
                'mae_p95': f"{rm.metrics.mae_p95:.6f}",
                'dei_error_p95': f"{rm.metrics.dei_error_p95:.6f}",
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def print_metrics(self, metrics: ExtendedMetrics, title: str = "METRICS") -> None:
        """
        Print metrics in a formatted way.

        Args:
            metrics: ExtendedMetrics instance.
            title: Title for the printout.
        """
        print("\n" + "=" * 70)
        print(f"{title}")
        print("=" * 70)

        print(f"\nSamples: {metrics.n_samples}")

        print("\nStandard Metrics:")
        print(f"  MSE:  {metrics.mse_mean:.6f} +/- {metrics.mse_std:.6f}")
        print(f"  MAE:  {metrics.mae_mean:.6f} +/- {metrics.mae_std:.6f}")
        print(f"  DEI:  {metrics.dei_error_mean:.6f} +/- {metrics.dei_error_std:.6f}")

        print("\nWorst-Case (95th Percentile):")
        print(f"  MSE:  {metrics.mse_p95:.6f}")
        print(f"  MAE:  {metrics.mae_p95:.6f}")
        print(f"  DEI:  {metrics.dei_error_p95:.6f}")

        if metrics.failure_metrics:
            print("\nPer-Failure-Type Metrics:")
            for ftype, fmetrics in metrics.failure_metrics.items():
                print(f"\n  {ftype.upper()} (n={fmetrics['n_samples']}):")
                print(f"    MSE:  {fmetrics['mse_mean']:.6f}")
                print(f"    MAE:  {fmetrics['mae_mean']:.6f}")
                print(f"    DEI:  {fmetrics['dei_error_mean']:.6f}")

        print("\n" + "=" * 70 + "\n")

    def save_metrics(
        self,
        metrics: ExtendedMetrics,
        output_path: Path,
    ) -> None:
        """
        Save metrics to disk.

        Args:
            metrics: ExtendedMetrics instance.
            output_path: Path to save metrics.
        """
        df = pd.DataFrame([metrics.to_dict()])
        df.to_csv(output_path, index=False)
        print(f"Metrics saved to {output_path}")


def evaluate_checkpoint(
    checkpoint_path: Path,
    test_split: SplitData,
    config,
    forward_model: ForwardModel2D | None = None,
) -> tuple[np.ndarray, ExtendedMetrics]:
    """
    Convenience function to evaluate a checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        test_split: Test data split.
        config: ALConfig instance.
        forward_model: ForwardModel2D instance.

    Returns:
        Tuple of (predictions, metrics).
    """
    import torch

    from models_3d.pinn.model import PINN3C

    # Create evaluator
    evaluator = ALEvaluator(forward_model=forward_model)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = PINN3C(
        signal_size=64,
        in_channels=3,
        base_filters=config.base_filters,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    predictions, metrics = evaluator.evaluate_split(model, test_split, device)

    return predictions, metrics


def create_ablation_table(
    baseline_metrics: ExtendedMetrics,
    round_metrics: list[RoundMetrics],
) -> pd.DataFrame:
    """
    Create an ablation table comparing baseline vs AL rounds.

    Args:
        baseline_metrics: Metrics from baseline model.
        round_metrics: Metrics from each AL round.

    Returns:
        DataFrame with ablation comparison.
    """
    rows = []

    # Baseline
    baseline_row = {
        'model': 'Baseline',
        'round': -1,
        'mse_mean': baseline_metrics.mse_mean,
        'mae_mean': baseline_metrics.mae_mean,
        'dei_error_mean': baseline_metrics.dei_error_mean,
        'mse_p95': baseline_metrics.mse_p95,
        'mae_p95': baseline_metrics.mae_p95,
        'dei_error_p95': baseline_metrics.dei_error_p95,
    }
    rows.append(baseline_row)

    # AL rounds
    for rm in round_metrics:
        # Calculate improvement relative to baseline
        mse_imp = (baseline_metrics.mse_mean - rm.metrics.mse_mean) / baseline_metrics.mse_mean * 100
        mae_imp = (baseline_metrics.mae_mean - rm.metrics.mae_mean) / baseline_metrics.mae_mean * 100
        dei_imp = (baseline_metrics.dei_error_mean - rm.metrics.dei_error_mean) / baseline_metrics.dei_error_mean * 100

        row = {
            'model': f'AL Round {rm.round_idx}',
            'round': rm.round_idx,
            'mse_mean': rm.metrics.mse_mean,
            'mae_mean': rm.metrics.mae_mean,
            'dei_error_mean': rm.metrics.dei_error_mean,
            'mse_p95': rm.metrics.mse_p95,
            'mae_p95': rm.metrics.mae_p95,
            'dei_error_p95': rm.metrics.dei_error_p95,
            'mse_improvement_%': mse_imp,
            'mae_improvement_%': mae_imp,
            'dei_improvement_%': dei_imp,
        }
        rows.append(row)

    return pd.DataFrame(rows)

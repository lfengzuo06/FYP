"""
Hard Case Scoring for Active Learning.

This module implements the scoring system for identifying hard cases in the
candidate pool. It combines multiple failure indicators into a composite score
using z-score normalization.

Scoring components:
1. Forward residual: ||S - K*f_hat||^2 (physics consistency)
2. DEI error: |DEI_gt - DEI_hat| (exchange quantification)
3. Off-diagonal MAE: MAE on off-diagonal regions (peak separation)
"""

from __future__ import annotations

from dataclasses import dataclass
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


def create_off_diagonal_mask(
    shape: tuple[int, int],
    diagonal_band_width: int = 5,
) -> np.ndarray:
    """
    Create a boolean mask for off-diagonal regions.

    Args:
        shape: Shape of the spectrum (typically 64x64).
        diagonal_band_width: Width of the diagonal band to exclude.

    Returns:
        Boolean mask where True indicates off-diagonal regions.
    """
    n = shape[0]
    ii, jj = np.indices((n, n))
    return np.abs(ii - jj) > diagonal_band_width


def compute_forward_residual(
    f_hat: np.ndarray,
    signal: np.ndarray,
    forward_model: ForwardModel2D,
) -> float:
    """
    Compute the forward residual: ||S - K*f_hat||^2.

    This measures physics consistency - how well the predicted spectrum
    reproduces the observed signal when passed through the forward model.

    Args:
        f_hat: Predicted spectrum (64x64).
        signal: Observed signal (64x64).
        forward_model: ForwardModel2D instance.

    Returns:
        Squared L2 norm of the residual.
    """
    # Compute forward prediction
    s_recon = forward_model.compute_signal(
        f_hat,
        noise_sigma=0,
        normalize=True,
        noise_model=None,
    )

    # Compute residual
    residual = signal - s_recon
    return float(np.sum(residual ** 2))


@dataclass
class SampleScores:
    """Scores for a single sample."""
    sample_idx: int
    seed: int

    # Individual metrics
    residual: float
    dei_error: float
    offdiag_mae: float
    total_mae: float
    total_mse: float

    # Normalized scores (z-scores)
    residual_z: float | None = None
    dei_error_z: float | None = None
    offdiag_mae_z: float | None = None

    # Composite score
    composite_score: float | None = None

    # Metadata reference
    metadata: SampleMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'sample_idx': self.sample_idx,
            'seed': self.seed,
            'residual': self.residual,
            'dei_error': self.dei_error,
            'offdiag_mae': self.offdiag_mae,
            'total_mae': self.total_mae,
            'total_mse': self.total_mse,
        }
        if self.residual_z is not None:
            result['residual_z'] = self.residual_z
        if self.dei_error_z is not None:
            result['dei_error_z'] = self.dei_error_z
        if self.offdiag_mae_z is not None:
            result['offdiag_mae_z'] = self.offdiag_mae_z
        if self.composite_score is not None:
            result['composite_score'] = self.composite_score
        return result


class HardCaseScorer:
    """
    Scores samples in the candidate pool to identify hard cases.

    Uses a combination of forward residual, DEI error, and off-diagonal MAE
    to identify samples where the model fails.
    """

    def __init__(
        self,
        forward_model: ForwardModel2D | None = None,
        diagonal_band_width: int = 5,
    ):
        """
        Initialize the scorer.

        Args:
            forward_model: ForwardModel2D instance.
            diagonal_band_width: Width for off-diagonal mask.
        """
        self.forward_model = forward_model or ForwardModel2D(n_d=64, n_b=64)
        self.diagonal_band_width = diagonal_band_width
        self.offdiag_mask = create_off_diagonal_mask(
            (self.forward_model.n_d, self.forward_model.n_d),
            diagonal_band_width,
        )

        # Cache for statistics
        self._stats: dict[str, float] = {}

    def compute_scores(
        self,
        predictions: np.ndarray,
        ground_truths: np.ndarray,
        signals: np.ndarray,
        metadata: list[SampleMetadata],
    ) -> list[SampleScores]:
        """
        Compute failure scores for all samples.

        Args:
            predictions: Predicted spectra (N, 64, 64).
            ground_truths: Ground truth spectra (N, 64, 64).
            signals: Input signals (N, 64, 64).
            metadata: List of SampleMetadata.

        Returns:
            List of SampleScores for each sample.
        """
        n_samples = len(predictions)
        scores_list = []

        for i in range(n_samples):
            f_hat = predictions[i]
            f_gt = ground_truths[i]
            signal = signals[i]

            # Compute forward residual
            residual = compute_forward_residual(
                f_hat, signal, self.forward_model
            )

            # Compute DEI error
            dei_gt = metadata[i].theoretical_dei
            dei_hat = compute_dei(f_hat, self.diagonal_band_width)
            dei_error = abs(dei_gt - dei_hat)

            # Compute off-diagonal MAE
            offdiag_mae = compute_mae(
                f_hat[self.offdiag_mask],
                f_gt[self.offdiag_mask],
            )

            # Compute total metrics
            total_mae = compute_mae(f_hat, f_gt)
            total_mse = compute_mse(f_hat, f_gt)

            scores = SampleScores(
                sample_idx=i,
                seed=metadata[i].seed,
                residual=residual,
                dei_error=dei_error,
                offdiag_mae=offdiag_mae,
                total_mae=total_mae,
                total_mse=total_mse,
                metadata=metadata[i],
            )
            scores_list.append(scores)

        return scores_list

    def normalize_scores(self, scores_list: list[SampleScores]) -> list[SampleScores]:
        """
        Normalize scores using z-score normalization.

        Computes statistics and adds normalized scores to each SampleScores.

        Args:
            scores_list: List of computed scores.

        Returns:
            Same list with normalized scores added.
        """
        # Extract raw scores
        residuals = np.array([s.residual for s in scores_list])
        dei_errors = np.array([s.dei_error for s in scores_list])
        offdiag_maes = np.array([s.offdiag_mae for s in scores_list])

        # Compute statistics
        stats = {
            'residual_mean': float(np.mean(residuals)),
            'residual_std': float(np.std(residuals)),
            'dei_error_mean': float(np.mean(dei_errors)),
            'dei_error_std': float(np.std(dei_errors)),
            'offdiag_mae_mean': float(np.mean(offdiag_maes)),
            'offdiag_mae_std': float(np.std(offdiag_maes)),
        }
        self._stats = stats

        # Normalize each score
        for scores in scores_list:
            scores.residual_z = (scores.residual - stats['residual_mean']) / (
                stats['residual_std'] + 1e-10
            )
            scores.dei_error_z = (scores.dei_error - stats['dei_error_mean']) / (
                stats['dei_error_std'] + 1e-10
            )
            scores.offdiag_mae_z = (scores.offdiag_mae - stats['offdiag_mae_mean']) / (
                stats['offdiag_mae_std'] + 1e-10
            )

            # Composite score (equal weight)
            scores.composite_score = (
                scores.residual_z +
                scores.dei_error_z +
                scores.offdiag_mae_z
            )

        return scores_list

    def to_dataframe(self, scores_list: list[SampleScores]) -> pd.DataFrame:
        """
        Convert scores to a pandas DataFrame.

        Args:
            scores_list: List of SampleScores.

        Returns:
            DataFrame with all scores.
        """
        data = [s.to_dict() for s in scores_list]
        return pd.DataFrame(data)

    def select_hard_seeds(
        self,
        scores_list: list[SampleScores],
        top_k_ratio: float = 0.15,
    ) -> list[SampleScores]:
        """
        Select hard seeds based on composite score.

        Args:
            scores_list: List of scores (should be normalized first).
            top_k_ratio: Fraction of samples to select as hard cases.

        Returns:
            List of hard seed scores.
        """
        # Sort by composite score
        sorted_scores = sorted(
            scores_list,
            key=lambda s: s.composite_score if s.composite_score is not None else 0,
            reverse=True,
        )

        # Select top K%
        n_select = max(1, int(len(sorted_scores) * top_k_ratio))
        return sorted_scores[:n_select]

    def compute_statistics(self, scores_list: list[SampleScores]) -> dict[str, Any]:
        """
        Compute summary statistics over all scores.

        Args:
            scores_list: List of SampleScores.

        Returns:
            Dictionary of statistics.
        """
        residuals = [s.residual for s in scores_list]
        dei_errors = [s.dei_error for s in scores_list]
        offdiag_maes = [s.offdiag_mae for s in scores_list]
        total_maes = [s.total_mae for s in scores_list]

        stats = {
            'n_samples': len(scores_list),
            'residual': {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'min': float(np.min(residuals)),
                'max': float(np.max(residuals)),
                'p50': float(np.percentile(residuals, 50)),
                'p95': float(np.percentile(residuals, 95)),
            },
            'dei_error': {
                'mean': float(np.mean(dei_errors)),
                'std': float(np.std(dei_errors)),
                'min': float(np.min(dei_errors)),
                'max': float(np.max(dei_errors)),
                'p50': float(np.percentile(dei_errors, 50)),
                'p95': float(np.percentile(dei_errors, 95)),
            },
            'offdiag_mae': {
                'mean': float(np.mean(offdiag_maes)),
                'std': float(np.std(offdiag_maes)),
                'min': float(np.min(offdiag_maes)),
                'max': float(np.max(offdiag_maes)),
                'p50': float(np.percentile(offdiag_maes, 50)),
                'p95': float(np.percentile(offdiag_maes, 95)),
            },
            'total_mae': {
                'mean': float(np.mean(total_maes)),
                'std': float(np.std(total_maes)),
                'p50': float(np.percentile(total_maes, 50)),
                'p95': float(np.percentile(total_maes, 95)),
            },
        }

        # Add composite score stats if available
        if scores_list[0].composite_score is not None:
            comp_scores = [s.composite_score for s in scores_list]
            stats['composite_score'] = {
                'mean': float(np.mean(comp_scores)),
                'std': float(np.std(comp_scores)),
                'min': float(np.min(comp_scores)),
                'max': float(np.max(comp_scores)),
            }

        return stats

    def print_summary(self, scores_list: list[SampleScores]) -> None:
        """
        Print a summary of the scoring results.

        Args:
            scores_list: List of SampleScores.
        """
        stats = self.compute_statistics(scores_list)

        print("\n" + "=" * 60)
        print("HARD CASE SCORING SUMMARY")
        print("=" * 60)

        print(f"\nTotal samples: {stats['n_samples']}")
        print(f"\nForward Residual:")
        print(f"  Mean: {stats['residual']['mean']:.6f}")
        print(f"  Std:  {stats['residual']['std']:.6f}")
        print(f"  P95:  {stats['residual']['p95']:.6f}")

        print(f"\nDEI Error:")
        print(f"  Mean: {stats['dei_error']['mean']:.6f}")
        print(f"  Std:  {stats['dei_error']['std']:.6f}")
        print(f"  P95:  {stats['dei_error']['p95']:.6f}")

        print(f"\nOff-Diagonal MAE:")
        print(f"  Mean: {stats['offdiag_mae']['mean']:.6f}")
        print(f"  Std:  {stats['offdiag_mae']['std']:.6f}")
        print(f"  P95:  {stats['offdiag_mae']['p95']:.6f}")

        print(f"\nTotal MAE:")
        print(f"  Mean: {stats['total_mae']['mean']:.6f}")
        print(f"  P95:  {stats['total_mae']['p95']:.6f}")

        if 'composite_score' in stats:
            print(f"\nComposite Score:")
            print(f"  Mean: {stats['composite_score']['mean']:.4f}")
            print(f"  Std:  {stats['composite_score']['std']:.4f}")
            print(f"  Max:  {stats['composite_score']['max']:.4f}")

        print("=" * 60 + "\n")


def score_candidate_pool(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    signals: np.ndarray,
    metadata: list[SampleMetadata],
    forward_model: ForwardModel2D | None = None,
    top_k_ratio: float = 0.15,
) -> tuple[list[SampleScores], pd.DataFrame, list[SampleScores]]:
    """
    Convenience function to score the entire candidate pool.

    Args:
        predictions: Model predictions (N, 64, 64).
        ground_truths: Ground truth spectra (N, 64, 64).
        signals: Input signals (N, 64, 64).
        metadata: List of SampleMetadata.
        forward_model: ForwardModel2D instance.
        top_k_ratio: Fraction to select as hard cases.

    Returns:
        Tuple of (all_scores, scores_df, hard_seeds).
    """
    scorer = HardCaseScorer(forward_model=forward_model)

    # Compute scores
    scores_list = scorer.compute_scores(
        predictions, ground_truths, signals, metadata
    )

    # Normalize
    scores_list = scorer.normalize_scores(scores_list)

    # Select hard seeds
    hard_seeds = scorer.select_hard_seeds(scores_list, top_k_ratio=top_k_ratio)

    # Convert to dataframe
    df = scorer.to_dataframe(scores_list)

    return scores_list, df, hard_seeds

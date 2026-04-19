"""
2D ILT Baseline for DEXSY Benchmark.

This module provides the traditional 2D Inverse Laplace Transform (ILT) baseline
using non-negative least squares (NNLS) with Tikhonov regularization.

Reference: Paper Section 2.3 - Python-based 2D ILT

Usage:
    from benchmarks_2d.ilt_baseline import ILTInferencePipeline, predict_ilt

    pipeline = ILTInferencePipeline()
    result = pipeline.predict(signal)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import numpy as np
import time

from dexsy_core.forward_model import ForwardModel2D, compute_dei
from dexsy_core.preprocessing import validate_signal_grid


@dataclass
class ILTPredictionResult:
    """Structured output for 2D ILT prediction."""

    signal: np.ndarray
    reconstructed_spectrum: np.ndarray
    dei: float
    inference_time: float
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    figure: Any | None = None


class ILTInferencePipeline:
    """
    2D ILT inference pipeline using NNLS with Tikhonov regularization.

    This implements the Python-based 2D ILT pipeline described in the paper,
    which provides a reproducible alternative to the MATLAB implementation.
    """

    MODEL_NAME = "2d_ilt"

    def __init__(
        self,
        alpha: float = 0.02,
        post_sharpen: bool = False,
        sharpen_sigma: float = 0.85,
        sharpen_strength: float = 0.38,
        forward_model: ForwardModel2D | None = None,
    ):
        """
        Initialize the ILT inference pipeline.

        Args:
            alpha: Tikhonov regularization parameter
            post_sharpen: Whether to apply sharpening after ILT
            sharpen_sigma: Gaussian sigma for sharpening
            sharpen_strength: Sharpening strength
            forward_model: ForwardModel2D instance
        """
        self.alpha = alpha
        self.post_sharpen = post_sharpen
        self.sharpen_sigma = sharpen_sigma
        self.sharpen_strength = sharpen_strength
        self.forward_model = forward_model or ForwardModel2D(n_d=64, n_b=64)

    def get_model_name(self) -> str:
        """Return the model name."""
        return self.MODEL_NAME

    def get_model_info(self) -> dict:
        """Return model information."""
        return {
            "model_name": self.MODEL_NAME,
            "model_type": "2D ILT (NNLS + Tikhonov)",
            "description": "Traditional 2D Inverse Laplace Transform baseline",
            "paper_reference": "Section 2.3 in steven_submission.pdf",
            "alpha": self.alpha,
            "post_sharpen": self.post_sharpen,
        }

    def predict(
        self,
        signal: np.ndarray,
        *,
        include_figure: bool = True,
    ) -> ILTPredictionResult:
        """
        Run 2D ILT on a single 64x64 DEXSY signal.

        Args:
            signal: Input signal array (64x64)
            include_figure: Whether to generate visualization

        Returns:
            ILTPredictionResult with spectrum, DEI, timing, and optional figure
        """
        validated = validate_signal_grid(signal, self.forward_model)
        signal_2d = validated[0, 0].astype(np.float64)

        start_time = time.time()
        reconstructed = self.forward_model.compute_ilt_nnls(
            signal_2d,
            alpha=self.alpha,
            post_sharpen=self.post_sharpen,
            sharpen_sigma=self.sharpen_sigma,
            sharpen_strength=self.sharpen_strength,
            renorm=True,
        )
        inference_time = time.time() - start_time

        dei = compute_dei(reconstructed.astype(np.float32))

        summary = {
            "signal_shape": tuple(signal_2d.shape),
            "spectrum_shape": tuple(reconstructed.shape),
            "prediction_mass": float(reconstructed.sum()),
            "prediction_peak": float(reconstructed.max()),
            "dei": float(dei),
            "inference_time_seconds": float(inference_time),
            "model_name": self.MODEL_NAME,
            "alpha": self.alpha,
        }

        figure = None
        if include_figure:
            figure = self._create_figure(signal_2d, reconstructed)

        return ILTPredictionResult(
            signal=signal_2d,
            reconstructed_spectrum=reconstructed.astype(np.float32),
            dei=float(dei),
            inference_time=float(inference_time),
            summary_metrics=summary,
            metadata=self.get_model_info(),
            figure=figure,
        )

    def predict_batch(
        self,
        signals: np.ndarray,
        *,
        include_figures: bool = False,
    ) -> list[ILTPredictionResult]:
        """
        Run 2D ILT on multiple 64x64 DEXSY signals.

        Args:
            signals: Input signals (N, 64, 64) or (N, 1, 64, 64)
            include_figures: Whether to generate visualizations

        Returns:
            List of ILTPredictionResult objects
        """
        validated = validate_signal_grid(signals, self.forward_model)

        results = []
        for idx in range(len(validated)):
            signal_2d = validated[idx, 0].astype(np.float64)

            start_time = time.time()
            reconstructed = self.forward_model.compute_ilt_nnls(
                signal_2d,
                alpha=self.alpha,
                post_sharpen=self.post_sharpen,
                sharpen_sigma=self.sharpen_sigma,
                sharpen_strength=self.sharpen_strength,
                renorm=True,
            )
            inference_time = time.time() - start_time

            dei = compute_dei(reconstructed.astype(np.float32))

            figure = None
            if include_figures:
                figure = self._create_figure(signal_2d, reconstructed)

            results.append(ILTPredictionResult(
                signal=signal_2d,
                reconstructed_spectrum=reconstructed.astype(np.float32),
                dei=float(dei),
                inference_time=float(inference_time),
                summary_metrics={
                    "inference_time_seconds": float(inference_time),
                    "dei": float(dei),
                },
                metadata=self.get_model_info(),
                figure=figure,
            ))

        return results

    def _create_figure(self, signal: np.ndarray, spectrum: np.ndarray):
        """Create a visual summary figure."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        im0 = axes[0].imshow(signal, cmap="viridis", origin="lower")
        axes[0].set_title("Input 64x64 DEXSY Signal")
        axes[0].set_xlabel("b2 index")
        axes[0].set_ylabel("b1 index")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(spectrum, cmap="magma", origin="lower")
        axes[1].set_title(f"ILT Reconstruction\nDEI={compute_dei(spectrum.astype(np.float32)):.3f}")
        axes[1].set_xlabel("D2 index")
        axes[1].set_ylabel("D1 index")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        fig.tight_layout()
        return fig


def predict_ilt(
    signal: np.ndarray,
    *,
    alpha: float = 0.02,
    forward_model: ForwardModel2D | None = None,
) -> np.ndarray:
    """
    Simple ILT prediction returning just the reconstructed spectrum.

    Args:
        signal: Input 64x64 signal
        alpha: Tikhonov regularization parameter
        forward_model: ForwardModel2D instance

    Returns:
        Reconstructed spectrum (64, 64)
    """
    if forward_model is None:
        forward_model = ForwardModel2D(n_d=64, n_b=64)

    validated = validate_signal_grid(signal, forward_model)
    signal_2d = validated[0, 0].astype(np.float64)
    return forward_model.compute_ilt_nnls(signal_2d, alpha=alpha, renorm=True).astype(np.float32)


if __name__ == "__main__":
    from dexsy_core.forward_model import ForwardModel2D

    print("Testing 2D ILT Baseline...")

    fm = ForwardModel2D()
    f, s, params = fm.generate_sample(n_compartments=2)

    pipeline = ILTInferencePipeline()
    result = pipeline.predict(s)

    print(f"  Model: {pipeline.get_model_name()}")
    print(f"  DEI: {result.dei:.4f}")
    print(f"  Inference time: {result.inference_time:.3f}s")
    print(f"  Prediction mass: {result.summary_metrics['prediction_mass']:.4f}")

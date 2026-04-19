"""
Inference pipeline for Plain U-Net on 2D DEXSY.

This module provides the standard inference interface for the Plain U-Net model,
compatible with the benchmark framework.

Usage:
    from models_2d.plain_unet import InferencePipeline

    pipeline = InferencePipeline()
    result = pipeline.predict(signal)

    # Or directly:
    from models_2d.plain_unet import predict

    result = predict(signal)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from dexsy_core.forward_model import ForwardModel2D, compute_dei
from dexsy_core.preprocessing import build_model_inputs, validate_signal_grid

from .model import PlainUNet2D


@dataclass
class PredictionResult:
    """Structured output for one reconstructed 2D DEXSY prediction."""

    signal: np.ndarray
    reconstructed_spectrum: np.ndarray
    dei: float
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth_spectrum: np.ndarray | None = None
    source_name: str | None = None
    figure: Any | None = None

    @property
    def spectrum(self) -> np.ndarray:
        """Backwards-friendly alias for the reconstructed spectrum."""
        return self.reconstructed_spectrum


class InferencePipeline:
    """
    Standard inference pipeline for Plain U-Net.

    This class implements the unified interface required by the benchmark framework:
    - __init__(checkpoint_path=None)
    - predict(signal) -> PredictionResult
    - get_model_name() -> str
    - get_model_info() -> dict
    """

    MODEL_NAME = "plain_unet"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: torch.device | str | None = None,
        forward_model: ForwardModel2D | None = None,
    ):
        """
        Initialize the inference pipeline.

        Args:
            checkpoint_path: Path to model checkpoint. If None, uses default.
            device: Device to use ('cuda', 'cpu', or None for auto)
            forward_model: ForwardModel2D instance (creates new if None)
        """
        self.forward_model = forward_model or ForwardModel2D(n_d=64, n_b=64)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        # Load model
        self.model, self.model_metadata = self._load_model()

    def _load_model(self) -> tuple:
        """Load the trained model from checkpoint."""
        if self.checkpoint_path is None:
            raise FileNotFoundError(
                "Checkpoint path is required for Plain U-Net. "
                "Please train the model first using models_2d.plain_unet.train_model()"
            )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}. "
                f"Please train the model first or provide a valid checkpoint path."
            )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Infer model config from state dict
        # Plain U-Net starts with enc1.conv1
        if 'enc1.conv1.weight' in state_dict:
            base_filters = state_dict['enc1.conv1.weight'].shape[0]
            in_channels = state_dict['enc1.conv1.weight'].shape[1]
        else:
            base_filters = 32
            in_channels = 3

        model = PlainUNet2D(
            in_channels=in_channels,
            base_filters=base_filters
        ).to(self.device)

        model.load_state_dict(state_dict)
        model.eval()

        metadata = {
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "base_filters": base_filters,
            "in_channels": in_channels,
            "model_name": self.MODEL_NAME,
            "epoch": checkpoint.get('epoch'),
            "val_loss": checkpoint.get('val_loss'),
        }

        return model, metadata

    def get_model_name(self) -> str:
        """Return the model name."""
        return self.MODEL_NAME

    def get_model_info(self) -> dict:
        """Return model information."""
        return {
            **self.model_metadata,
            "model_type": "Plain U-Net",
            "description": "CNN baseline U-Net with bilinear upsampling, no attention gates",
            "paper_reference": "Section 2.4.1 in steven_submission.pdf",
        }

    def _predict_distribution(
        self,
        inputs: np.ndarray,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Run batched inference."""
        array = np.asarray(inputs, dtype=np.float32)
        if array.ndim == 3:
            array = array[None, ...]

        outputs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(array), batch_size):
                batch = torch.from_numpy(array[start:start + batch_size]).to(self.device)
                preds = self.model(batch).cpu().numpy()
                outputs.append(preds)

        return np.concatenate(outputs, axis=0)

    def _summarize_prediction(
        self,
        signal: np.ndarray,
        reconstructed_spectrum: np.ndarray,
        true_spectrum: np.ndarray | None = None,
    ) -> dict:
        """Compute summary metrics for one prediction."""
        metrics = {
            "signal_shape": tuple(signal.shape),
            "spectrum_shape": tuple(reconstructed_spectrum.shape),
            "prediction_mass": float(reconstructed_spectrum.sum()),
            "prediction_peak": float(reconstructed_spectrum.max()),
            "dei": float(compute_dei(reconstructed_spectrum)),
            "model_name": self.MODEL_NAME,
        }

        if true_spectrum is not None:
            true_array = np.asarray(true_spectrum, dtype=np.float32)
            metrics["ground_truth_dei"] = float(compute_dei(true_array))
            metrics["mse"] = float(np.mean((true_array - reconstructed_spectrum) ** 2))
            metrics["mae"] = float(np.mean(np.abs(true_array - reconstructed_spectrum)))

        return metrics

    def _create_figure(
        self,
        signal: np.ndarray,
        reconstructed_spectrum: np.ndarray,
        true_spectrum: np.ndarray | None = None,
    ):
        """Create a visual summary figure."""
        import matplotlib.pyplot as plt

        if true_spectrum is None:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

            im0 = axes[0].imshow(signal, cmap="viridis", origin="lower")
            axes[0].set_title("Input 64x64 DEXSY Signal")
            axes[0].set_xlabel("b2 index")
            axes[0].set_ylabel("b1 index")
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(reconstructed_spectrum, cmap="magma", origin="lower")
            axes[1].set_title(f"Reconstructed Spectrum\nDEI={compute_dei(reconstructed_spectrum):.3f}")
            axes[1].set_xlabel("D2 index")
            axes[1].set_ylabel("D1 index")
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        else:
            true_array = np.asarray(true_spectrum, dtype=np.float32)
            vmax = max(float(true_array.max()), float(reconstructed_spectrum.max()), 1e-6)
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

            im0 = axes[0].imshow(signal, cmap="viridis", origin="lower")
            axes[0].set_title("Input 64x64 DEXSY Signal")
            axes[0].set_xlabel("b2 index")
            axes[0].set_ylabel("b1 index")
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(true_array, cmap="magma", origin="lower", vmin=0.0, vmax=vmax)
            axes[1].set_title(f"Ground Truth\nDEI={compute_dei(true_array):.3f}")
            axes[1].set_xlabel("D2 index")
            axes[1].set_ylabel("D1 index")
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(reconstructed_spectrum, cmap="magma", origin="lower", vmin=0.0, vmax=vmax)
            axes[2].set_title(f"Prediction\nDEI={compute_dei(reconstructed_spectrum):.3f}")
            axes[2].set_xlabel("D2 index")
            axes[2].set_ylabel("D1 index")
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        fig.tight_layout()
        return fig

    def predict(
        self,
        signal: np.ndarray,
        *,
        true_spectrum: np.ndarray | None = None,
        include_figure: bool = True,
        source_name: str | None = None,
    ) -> PredictionResult:
        """
        Run inference on a single 64x64 DEXSY signal.

        Args:
            signal: Input signal array (64x64)
            true_spectrum: Optional ground truth for metrics computation
            include_figure: Whether to generate a visualization
            source_name: Optional source name

        Returns:
            PredictionResult with spectrum, DEI, metrics, and optional figure
        """
        validated = validate_signal_grid(signal, self.forward_model)
        model_inputs = build_model_inputs(validated, self.forward_model)

        reconstructed = self._predict_distribution(model_inputs)[0, 0]
        signal_2d = validated[0, 0]

        summary = self._summarize_prediction(signal_2d, reconstructed, true_spectrum)

        figure = None
        if include_figure:
            figure = self._create_figure(signal_2d, reconstructed, true_spectrum)

        return PredictionResult(
            signal=signal_2d,
            reconstructed_spectrum=reconstructed,
            dei=float(summary["dei"]),
            summary_metrics=summary,
            metadata={**self.model_metadata},
            ground_truth_spectrum=np.asarray(true_spectrum, dtype=np.float32) if true_spectrum is not None else None,
            source_name=source_name,
            figure=figure,
        )

    def predict_batch(
        self,
        signals: np.ndarray,
        *,
        true_spectra: np.ndarray | None = None,
        source_names: list[str] | None = None,
        include_figures: bool = False,
        batch_size: int = 16,
    ) -> list[PredictionResult]:
        """
        Run batch inference on multiple 64x64 DEXSY signals.

        Args:
            signals: Input signals array (N, 64, 64) or (N, 1, 64, 64)
            true_spectra: Optional ground truths for metrics
            source_names: Optional source names for each signal
            include_figures: Whether to generate visualizations
            batch_size: Batch size for inference

        Returns:
            List of PredictionResult objects
        """
        validated = validate_signal_grid(signals, self.forward_model)
        model_inputs = build_model_inputs(validated, self.forward_model)
        predictions = self._predict_distribution(model_inputs, batch_size=batch_size)[:, 0]

        if true_spectra is not None:
            true_array = np.asarray(true_spectra, dtype=np.float32)
            if true_array.ndim == 4 and true_array.shape[1] == 1:
                true_array = true_array[:, 0]
        else:
            true_array = None

        if source_names is None:
            source_names = [f"sample_{idx:03d}" for idx in range(len(predictions))]

        results = []
        for idx, reconstructed in enumerate(predictions):
            signal_2d = validated[idx, 0]
            gt = true_array[idx] if true_array is not None else None
            summary = self._summarize_prediction(signal_2d, reconstructed, gt)

            figure = None
            if include_figures:
                figure = self._create_figure(signal_2d, reconstructed, gt)

            results.append(PredictionResult(
                signal=signal_2d,
                reconstructed_spectrum=reconstructed,
                dei=float(summary["dei"]),
                summary_metrics=summary,
                metadata={**self.model_metadata, "batch_index": idx},
                ground_truth_spectrum=gt,
                source_name=source_names[idx] if idx < len(source_names) else None,
                figure=figure,
            ))

        return results


def predict(
    signal: np.ndarray,
    *,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    forward_model: ForwardModel2D | None = None,
    true_spectrum: np.ndarray | None = None,
    include_figure: bool = True,
    source_name: str | None = None,
) -> PredictionResult:
    """
    High-level inference function for Plain U-Net.

    Args:
        signal: Input 64x64 DEXSY signal
        checkpoint_path: Optional checkpoint path
        device: Device to use
        forward_model: Optional ForwardModel2D instance
        true_spectrum: Optional ground truth for metrics
        include_figure: Whether to generate visualization
        source_name: Optional source name

    Returns:
        PredictionResult with prediction details
    """
    pipeline = InferencePipeline(
        checkpoint_path=checkpoint_path,
        device=device,
        forward_model=forward_model,
    )
    return pipeline.predict(
        signal,
        true_spectrum=true_spectrum,
        include_figure=include_figure,
        source_name=source_name,
    )


def predict_batch(
    signals: np.ndarray,
    *,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    true_spectra: np.ndarray | None = None,
    source_names: list[str] | None = None,
    include_figures: bool = False,
    batch_size: int = 16,
) -> list[PredictionResult]:
    """
    High-level batch inference function for Plain U-Net.

    Args:
        signals: Input signals array (N, 64, 64) or (N, 1, 64, 64)
        checkpoint_path: Optional checkpoint path
        device: Device to use
        true_spectra: Optional ground truths for metrics
        source_names: Optional source names for each signal
        include_figures: Whether to generate visualizations
        batch_size: Batch size for inference

    Returns:
        List of PredictionResult objects
    """
    pipeline = InferencePipeline(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return pipeline.predict_batch(
        signals,
        true_spectra=true_spectra,
        source_names=source_names,
        include_figures=include_figures,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    # Quick test
    from dexsy_core.forward_model import ForwardModel2D

    print("Testing Plain U-Net InferencePipeline...")

    fm = ForwardModel2D()
    f, s, params = fm.generate_sample(n_compartments=2)

    print(f"  Signal shape: {s.shape}")
    print(f"  Ground truth DEI: {compute_dei(f):.4f}")
    print("\n  Note: Plain U-Net requires a trained checkpoint.")
    print("  Train the model using: from models_2d.plain_unet import train_model")

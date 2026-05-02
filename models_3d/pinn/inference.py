"""
Inference pipeline for PINN on 3-Compartment DEXSY.

This module provides the standard inference interface for the PINN 3C model,
compatible with the benchmark framework.

Usage:
    from models_3d.pinn import PINNInferencePipeline3C
    
    pipeline = PINNInferencePipeline3C()
    result = pipeline.predict(signal)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import sys
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dexsy_core.forward_model import ForwardModel2D, compute_dei
from dexsy_core.preprocessing import build_model_inputs, validate_signal_grid

from .model import PINN3C


def _extract_state_dict(checkpoint_obj: Any) -> dict[str, torch.Tensor]:
    """Extract a state_dict from common checkpoint formats."""
    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["model_state_dict"]
    else:
        state_dict = checkpoint_obj

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict dictionary.")

    return state_dict


def _reference_state_keys() -> set[str]:
    """Expected PINN3C parameter keys for compatibility checks."""
    reference_model = PINN3C(signal_size=64, in_channels=3, base_filters=64)
    return set(reference_model.state_dict().keys())


def _checkpoint_compatibility_reason(checkpoint_path: Path) -> tuple[bool, str]:
    """
    Validate whether a checkpoint matches the current PINN3C architecture.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, f"cannot be loaded ({type(exc).__name__}: {exc})"

    try:
        state_dict = _extract_state_dict(checkpoint)
    except TypeError as exc:
        return False, str(exc)

    state_keys = set(state_dict.keys())
    expected_keys = _reference_state_keys()
    overlap = len(state_keys & expected_keys)
    overlap_ratio = overlap / max(len(expected_keys), 1)

    if overlap_ratio < 0.9:
        sample_keys = ", ".join(list(state_dict.keys())[:5])
        return (
            False,
            f"incompatible state_dict keys (matched {overlap}/{len(expected_keys)}). "
            f"Sample checkpoint keys: [{sample_keys}]",
        )

    return True, "compatible"


@dataclass
class PredictionResultPINN3C:
    """Structured output for one reconstructed 3C DEXSY prediction."""

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


class PINNInferencePipeline3C:
    """
    Standard inference pipeline for PINN 3C.

    This class implements the unified interface required by the benchmark framework:
    - __init__(checkpoint_path=None)
    - predict(signal) -> PredictionResultPINN3C
    - get_model_name() -> str
    - get_model_info() -> dict
    """

    MODEL_NAME = "pinn_3c"
    DEFAULT_CHECKPOINT = None  # Set after training

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
        self._incompatible_checkpoints: list[tuple[Path, str]] = []

        # Resolve checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self._find_default_checkpoint()

        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        # Load model
        self.model, self.model_metadata = self._load_model()

    def _find_default_checkpoint(self) -> Path | None:
        """Find the default compatible checkpoint from bundled locations."""
        root = Path(__file__).parent.parent.parent / "checkpoints_3d" / "pinn_3c"
        
        if not root.exists():
            return None
        
        # Find all best_model.pt files in timestamped subdirectories
        best_models = list(root.glob("*/best_model.pt"))
        if not best_models:
            return None
        
        # Check compatibility of each candidate
        for candidate in sorted(best_models, key=lambda p: p.stat().st_mtime, reverse=True):
            compatible, reason = _checkpoint_compatibility_reason(candidate)
            if compatible:
                return candidate
            self._incompatible_checkpoints.append((candidate, reason))
        
        return None

    def _load_model(self) -> tuple:
        """Load the trained model from checkpoint."""
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            incompat_summary = ""
            if self._incompatible_checkpoints:
                details = "\n".join(
                    f"  - {path}: {reason}"
                    for path, reason in self._incompatible_checkpoints
                )
                incompat_summary = f"\nFound PINN 3C checkpoints but they are incompatible:\n{details}\n"
            raise FileNotFoundError(
                f"PINN 3C checkpoint not found at: {self.checkpoint_path}. "
                f"Please train the model first using 'python -m models_3d.pinn.train' "
                f"or provide a valid checkpoint path via checkpoint_path argument."
                f"{incompat_summary}"
            )

        compatible, reason = _checkpoint_compatibility_reason(self.checkpoint_path)
        if not compatible:
            raise RuntimeError(
                f"PINN 3C checkpoint is incompatible with current PINN3C architecture: "
                f"{self.checkpoint_path}\nReason: {reason}"
            )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = _extract_state_dict(checkpoint)
        config = checkpoint.get('config', {})

        # Create model with config
        model = PINN3C(
            signal_size=config.get('signal_size', 64),
            in_channels=config.get('in_channels', 3),
            base_filters=config.get('base_filters', 64),
        ).to(self.device)

        model.load_state_dict(state_dict)
        model.eval()

        metadata = {
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "base_filters": config.get('base_filters', 64),
            "model_name": self.MODEL_NAME,
            "epoch": config.get('epoch'),
            "val_loss": config.get('val_loss'),
        }

        return model, metadata

    def get_model_name(self) -> str:
        """Return the model name."""
        return self.MODEL_NAME

    def get_model_info(self) -> dict:
        """Return model information."""
        return {
            **self.model_metadata,
            "model_type": "Physics-Informed Neural Network for 3C",
            "description": "PINN with simple encoder-decoder, physics-informed loss",
            "task": "3-compartment DEXSY inversion",
            "loss_components": [
                "Reconstruction loss (peak-weighted MSE)",
                "Forward consistency: ||S - K*f||^2",
                "Smoothness regularization (Laplacian)",
            ],
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
            metrics["dei_error"] = abs(metrics["ground_truth_dei"] - metrics["dei"])

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
            axes[0].set_title("Input 64x64 DEXSY Signal (3C)")
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
            axes[0].set_title("Input 64x64 DEXSY Signal (3C)")
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
    ) -> PredictionResultPINN3C:
        """
        Run inference on a single 64x64 DEXSY signal.

        Args:
            signal: Input signal array (64x64)
            true_spectrum: Optional ground truth for metrics computation
            include_figure: Whether to generate a visualization
            source_name: Optional source name

        Returns:
            PredictionResultPINN3C with spectrum, DEI, metrics, and optional figure
        """
        # Validate and preprocess
        validated = validate_signal_grid(signal, self.forward_model)
        model_inputs = build_model_inputs(validated, self.forward_model)

        # Run inference
        reconstructed = self._predict_distribution(model_inputs)[0, 0]
        signal_2d = validated[0, 0]

        # Compute metrics
        summary = self._summarize_prediction(signal_2d, reconstructed, true_spectrum)

        # Create figure
        figure = None
        if include_figure:
            figure = self._create_figure(signal_2d, reconstructed, true_spectrum)

        return PredictionResultPINN3C(
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
    ) -> list[PredictionResultPINN3C]:
        """
        Run batch inference on multiple 64x64 DEXSY signals.

        Args:
            signals: Input signals array (N, 64, 64) or (N, 1, 64, 64)
            true_spectra: Optional ground truths for metrics
            source_names: Optional source names for each signal
            include_figures: Whether to generate visualizations
            batch_size: Batch size for inference

        Returns:
            List of PredictionResultPINN3C objects
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

            results.append(PredictionResultPINN3C(
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
) -> PredictionResultPINN3C:
    """
    High-level inference function for PINN 3C.

    Args:
        signal: Input 64x64 DEXSY signal
        checkpoint_path: Optional checkpoint path
        device: Device to use
        forward_model: Optional ForwardModel2D instance
        true_spectrum: Optional ground truth for metrics
        include_figure: Whether to generate visualization
        source_name: Optional source name

    Returns:
        PredictionResultPINN3C with prediction details
    """
    pipeline = PINNInferencePipeline3C(
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
) -> list[PredictionResultPINN3C]:
    """
    High-level batch inference function for PINN 3C.

    Args:
        signals: Input signals array (N, 64, 64) or (N, 1, 64, 64)
        checkpoint_path: Optional checkpoint path
        device: Device to use
        true_spectra: Optional ground truths for metrics
        source_names: Optional source names for each signal
        include_figures: Whether to generate visualizations
        batch_size: Batch size for inference

    Returns:
        List of PredictionResultPINN3C objects
    """
    pipeline = PINNInferencePipeline3C(
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


def load_trained_model(
    checkpoint_path,
    device: str | None = None,
) -> tuple:
    """
    Load a trained PINN 3C model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model onto

    Returns:
        Tuple of (model, metadata)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint_path = Path(checkpoint_path)
    compatible, reason = _checkpoint_compatibility_reason(checkpoint_path)
    if not compatible:
        raise RuntimeError(
            f"Incompatible PINN 3C checkpoint: {checkpoint_path}\nReason: {reason}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    config = checkpoint.get('config', {})

    model = PINN3C(
        signal_size=config.get('signal_size', 64),
        in_channels=config.get('in_channels', 3),
        base_filters=config.get('base_filters', 64),
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "model_name": "pinn_3c",
        "epoch": config.get('epoch'),
        "val_loss": config.get('val_loss'),
    }

    return model, metadata


if __name__ == "__main__":
    from dexsy_core.forward_model import ForwardModel2D, compute_dei

    fm = ForwardModel2D()
    f, s, params = fm.generate_sample(n_compartments=3)

    print("Testing PINN InferencePipeline 3C...")
    try:
        pipeline = PINNInferencePipeline3C()
        result = pipeline.predict(s)
        print(f"  Model: {pipeline.get_model_name()}")
        print(f"  DEI: {result.dei:.4f}")
        print(f"  Prediction mass: {result.summary_metrics['prediction_mass']:.4f}")
    except FileNotFoundError as e:
        print(f"  Note: {e}")
        print("  (Model needs to be trained first)")

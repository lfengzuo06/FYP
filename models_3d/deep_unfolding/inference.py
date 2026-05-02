"""
Inference pipeline for Deep Unfolding (ISTA-Net style) on 3-Compartment DEXSY.

Reference: Paper Section 2.4.3 - Deep Unfolding

Usage:
    from models_3d.deep_unfolding import InferencePipeline3C

    pipeline = InferencePipeline3C(checkpoint_path="path/to/model.pt")
    result = pipeline.predict(signal)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dexsy_core.forward_model import ForwardModel2D, compute_dei
from dexsy_core.preprocessing import validate_signal_grid

from .model import DeepUnfolding3C


@dataclass
class PredictionResultDeep3C:
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


class InferencePipeline3C:
    """
    Inference pipeline for Deep Unfolding (ISTA-Net style) model on 3C.

    This class handles:
    - Loading trained model checkpoints
    - Running inference on single or batch of signals
    - Computing metrics and generating visualizations
    """

    MODEL_NAME = "deep_unfolding_3c"
    DEFAULT_CHECKPOINT = None  # Set after training

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device | None = None,
        forward_model: ForwardModel2D | None = None,
        n_layers: int = 12,
        hidden_dim: int = 256,
    ):
        """
        Initialize the Deep Unfolding 3C inference pipeline.

        Args:
            checkpoint_path: Path to model checkpoint. If None, searches default locations.
            device: Device to use ('cuda', 'cpu', or None for auto)
            forward_model: ForwardModel2D instance
            n_layers: Number of ISTA layers (must match checkpoint)
            hidden_dim: Hidden dimension (must match checkpoint)
        """
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.forward_model = forward_model or ForwardModel2D(n_d=64, n_b=64)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Model
        self.model: DeepUnfolding3C | None = None
        self.model_metadata: dict[str, Any] = {}

        # Load checkpoint
        if checkpoint_path is not None:
            self._load_model(checkpoint_path)
        else:
            checkpoint_path = self._find_default_checkpoint()
            if checkpoint_path is not None:
                self._load_model(checkpoint_path)
            else:
                raise FileNotFoundError(
                    f"No checkpoint found for deep_unfolding_3c. "
                    f"Train the model first using 'python -m models_3d.deep_unfolding.train'. "
                    f"Or provide a checkpoint_path explicitly."
                )

    def _find_default_checkpoint(self) -> Path | None:
        """Find the default checkpoint from bundled locations."""
        possible_paths = [
            Path(__file__).parent.parent.parent / "checkpoints" / "deep_unfolding_3c" / "best_model.pt",
            Path(__file__).parent.parent.parent / "training_output_3d" / "deep_unfolding_3c" / "checkpoints_latest" / "best_model.pt",
        ]

        for p in possible_paths:
            if p.exists():
                return p
        return None

    def _load_model(self, checkpoint_path: str | Path):
        """Load model from checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Get config from checkpoint or use defaults
        config = checkpoint.get('config', {})
        n_layers = config.get('n_layers', self.n_layers)
        hidden_dim = config.get('hidden_dim', self.hidden_dim)
        init_method = config.get('init_method', 'mlp')

        # Create model
        self.model = DeepUnfolding3C(
            n_layers=n_layers,
            n_d=self.forward_model.n_d,
            hidden_dim=hidden_dim,
            init_method=init_method,
        ).to(self.device)

        # Set kernel matrix
        K = self.forward_model.kernel_matrix.astype(np.float32)
        K_tensor = torch.from_numpy(K).float().to(self.device)
        self.model.set_kernel_matrix(K_tensor)

        # Load weights
        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=False,
        )
        self._validate_checkpoint_keys(missing_keys, unexpected_keys, path)
        self.model.eval()

        self.model_metadata = {
            'checkpoint_path': str(path),
            'n_layers': n_layers,
            'hidden_dim': hidden_dim,
            'init_method': init_method,
            'epoch': config.get('epoch', 'unknown'),
            'val_loss': config.get('val_loss', 'unknown'),
            'model_name': self.MODEL_NAME,
        }

    def _validate_checkpoint_keys(
        self,
        missing_keys: list[str],
        unexpected_keys: list[str],
        checkpoint_path: Path,
    ) -> None:
        """Guard against loading a checkpoint from the wrong model family."""
        if self.model is None:
            raise RuntimeError("Model must be initialized before validating checkpoint keys.")

        allowed_missing = {"_K", "_Kt"}
        if self.model.use_denoiser:
            allowed_missing.update(
                {f"ista_layers.{idx}.denoise_scale" for idx in range(self.model.n_layers)}
            )

        disallowed_missing = [key for key in missing_keys if key not in allowed_missing]
        if disallowed_missing or unexpected_keys:
            missing_preview = ", ".join(disallowed_missing[:6])
            unexpected_preview = ", ".join(unexpected_keys[:6])
            raise RuntimeError(
                "Checkpoint/model mismatch for deep_unfolding_3c. "
                f"Checkpoint: {checkpoint_path}. "
                f"Disallowed missing keys ({len(disallowed_missing)}): [{missing_preview}]. "
                f"Unexpected keys ({len(unexpected_keys)}): [{unexpected_preview}]. "
                "Please provide a deep_unfolding_3c checkpoint."
            )

    def get_model_name(self) -> str:
        """Return the model name."""
        return self.MODEL_NAME

    def get_model_info(self) -> dict:
        """Return model information."""
        return {
            "model_name": self.MODEL_NAME,
            "model_type": "Deep Unfolding (ISTA-Net style) for 3C",
            "description": "Deep Unfolding model with learnable ISTA iterations for 3-compartment DEXSY",
            "task": "3-compartment DEXSY inversion",
            **self.model_metadata,
        }

    def _predict_distribution(self, model_inputs: torch.Tensor | np.ndarray) -> np.ndarray:
        """Run model inference."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Provide checkpoint_path or call _load_model().")

        with torch.no_grad():
            if isinstance(model_inputs, np.ndarray):
                model_inputs = torch.from_numpy(model_inputs).float()

            # model_inputs is [B, 3, H, W], we need signal [B, 1, H, W]
            if model_inputs.shape[1] == 3:
                # Extract first channel (raw signal)
                signal = model_inputs[:, 0:1, :, :]
            else:
                signal = model_inputs

            predictions = self.model(signal.to(self.device, dtype=torch.float32))
            return predictions.cpu().numpy()

    def _summarize_prediction(
        self,
        signal: np.ndarray,
        reconstructed: np.ndarray,
        ground_truth: np.ndarray | None = None,
    ) -> dict:
        """Compute metrics for a prediction."""
        summary = {
            "signal_shape": tuple(signal.shape),
            "spectrum_shape": tuple(reconstructed.shape),
            "prediction_mass": float(reconstructed.sum()),
            "prediction_peak": float(reconstructed.max()),
            "prediction_mean": float(reconstructed.mean()),
            "dei": float(compute_dei(reconstructed)),
            "model_name": self.MODEL_NAME,
        }

        if ground_truth is not None:
            diff = reconstructed - ground_truth
            summary["mse"] = float(np.mean(diff ** 2))
            summary["mae"] = float(np.mean(np.abs(diff)))
            summary["ground_truth_dei"] = float(compute_dei(ground_truth))
            summary["ground_truth_mass"] = float(ground_truth.sum())
            summary["dei_error"] = abs(summary["ground_truth_dei"] - summary["dei"])

        return summary

    def _create_figure(
        self,
        signal: np.ndarray,
        reconstructed: np.ndarray,
        ground_truth: np.ndarray | None = None,
    ):
        """Create a visual summary figure."""
        import matplotlib.pyplot as plt

        if ground_truth is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        # Input signal
        im0 = axes[0].imshow(signal, cmap="viridis", origin="lower")
        axes[0].set_title("Input 64x64 DEXSY Signal (3C)")
        axes[0].set_xlabel("b2 index")
        axes[0].set_ylabel("b1 index")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Reconstructed spectrum
        dei = compute_dei(reconstructed)
        im1 = axes[1].imshow(reconstructed, cmap="magma", origin="lower")
        axes[1].set_title(f"Deep Unfolding Reconstruction\nDEI={dei:.3f}")
        axes[1].set_xlabel("D2 index")
        axes[1].set_ylabel("D1 index")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Ground truth (if available)
        if ground_truth is not None:
            gt_dei = compute_dei(ground_truth)
            im2 = axes[2].imshow(ground_truth, cmap="magma", origin="lower")
            axes[2].set_title(f"Ground Truth\nDEI={gt_dei:.3f}")
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
    ) -> PredictionResultDeep3C:
        """
        Run inference on a single 64x64 DEXSY signal.

        Args:
            signal: Input signal array (64x64)
            true_spectrum: Optional ground truth for metrics computation
            include_figure: Whether to generate a visualization
            source_name: Optional source name

        Returns:
            PredictionResultDeep3C with spectrum, DEI, metrics, and optional figure
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Provide checkpoint_path.")

        # Validate and preprocess
        validated = validate_signal_grid(signal, self.forward_model)

        # Predict
        predictions = self._predict_distribution(validated)
        reconstructed = predictions[0, 0]

        signal_2d = validated[0, 0]
        summary = self._summarize_prediction(signal_2d, reconstructed, true_spectrum)

        # Figure
        figure = None
        if include_figure:
            figure = self._create_figure(signal_2d, reconstructed, true_spectrum)

        return PredictionResultDeep3C(
            signal=signal_2d,
            reconstructed_spectrum=reconstructed,
            dei=float(summary["dei"]),
            summary_metrics=summary,
            metadata={**self.model_metadata, "source_name": source_name},
            ground_truth_spectrum=true_spectrum,
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
    ) -> list[PredictionResultDeep3C]:
        """
        Run batch inference on multiple 64x64 DEXSY signals.

        Args:
            signals: Input signals array (N, 64, 64) or (N, 1, 64, 64)
            true_spectra: Optional ground truths for metrics
            source_names: Optional source names for each signal
            include_figures: Whether to generate visualizations
            batch_size: Batch size for inference

        Returns:
            List of PredictionResultDeep3C objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Provide checkpoint_path.")

        validated = validate_signal_grid(signals, self.forward_model)

        # Predict in mini-batches to control memory usage.
        prediction_batches = []
        for start in range(0, len(validated), max(1, batch_size)):
            end = min(start + max(1, batch_size), len(validated))
            prediction_batches.append(self._predict_distribution(validated[start:end]))
        predictions = np.concatenate(prediction_batches, axis=0)

        if true_spectra is not None:
            true_array = np.asarray(true_spectra, dtype=np.float32)
            if true_array.ndim == 4 and true_array.shape[1] == 1:
                true_array = true_array[:, 0]
        else:
            true_array = None

        if source_names is None:
            source_names = [f"sample_{idx:03d}" for idx in range(len(predictions))]

        results = []
        for idx in range(len(predictions)):
            reconstructed = predictions[idx, 0]
            signal_2d = validated[idx, 0]
            gt = true_array[idx] if true_array is not None else None
            summary = self._summarize_prediction(signal_2d, reconstructed, gt)

            figure = None
            if include_figures:
                figure = self._create_figure(signal_2d, reconstructed, gt)

            results.append(PredictionResultDeep3C(
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
) -> PredictionResultDeep3C:
    """
    High-level inference function for Deep Unfolding 3C.

    Args:
        signal: Input 64x64 DEXSY signal
        checkpoint_path: Optional checkpoint path
        device: Device to use
        forward_model: Optional ForwardModel2D instance
        true_spectrum: Optional ground truth for metrics
        include_figure: Whether to generate visualization
        source_name: Optional source name

    Returns:
        PredictionResultDeep3C with prediction details
    """
    pipeline = InferencePipeline3C(
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
) -> list[PredictionResultDeep3C]:
    """
    High-level batch inference function for Deep Unfolding 3C.

    Args:
        signals: Input signals array (N, 64, 64) or (N, 1, 64, 64)
        checkpoint_path: Optional checkpoint path
        device: Device to use
        true_spectra: Optional ground truths for metrics
        source_names: Optional source names for each signal
        include_figures: Whether to generate visualizations
        batch_size: Batch size for inference

    Returns:
        List of PredictionResultDeep3C objects
    """
    pipeline = InferencePipeline3C(
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
    checkpoint_path: str | Path,
    device: str | None = None,
) -> DeepUnfolding3C:
    """
    Load a trained Deep Unfolding 3C model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model onto

    Returns:
        Loaded DeepUnfolding3C model
    """
    pipeline = InferencePipeline3C(checkpoint_path=checkpoint_path, device=device)
    return pipeline.model


if __name__ == "__main__":
    from dexsy_core.forward_model import ForwardModel2D, compute_dei

    print("Testing Deep Unfolding 3C InferencePipeline...")

    fm = ForwardModel2D()
    f, s, params = fm.generate_sample(n_compartments=3)

    print(f"  Model: {InferencePipeline3C.MODEL_NAME}")
    print(f"  Generated 3C sample: DEI={compute_dei(f):.4f}")
    print("  Note: Full inference requires trained checkpoint.")

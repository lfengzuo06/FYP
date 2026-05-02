"""
Inference pipeline for Neural Operator models (DeepONet, FNO).

Reference: Paper Section 2.4.4 - Neural Operators

Usage:
    from models_2d.neural_operators.inference import InferencePipeline

    pipeline = InferencePipeline(checkpoint_path="path/to/model.pt", model_type="deeponet")
    result = pipeline.predict(signal)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dexsy_core.forward_model import ForwardModel2D, compute_dei
from dexsy_core.preprocessing import validate_signal_grid, build_model_inputs

from .model import DeepONet2D, FNO2D


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


class InferencePipeline:
    """
    Inference pipeline for Neural Operator models.

    Supports both DeepONet and FNO architectures.
    """

    MODEL_NAME = "neural_operator"
    
    # Default checkpoint paths
    DEFAULT_PATHS = {
        "deeponet": "checkpoints_2d/deeponet/best_model.pt",
        "fno": "checkpoints_2d/fno/best_model.pt",
    }

    def __init__(
        self,
        model_type: str = "deeponet",
        checkpoint_path: str | Path | None = None,
        device: str | torch.device | None = None,
        forward_model: ForwardModel2D | None = None,
        **model_kwargs,
    ):
        """
        Initialize the Neural Operator inference pipeline.

        Args:
            model_type: Type of model ('deeponet' or 'fno')
            checkpoint_path: Path to model checkpoint. If None, searches default locations.
            device: Device to use ('cuda', 'cpu', or None for auto)
            forward_model: ForwardModel2D instance
            **model_kwargs: Additional arguments for the model
        """
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model_type = model_type
        self.forward_model = forward_model or ForwardModel2D(n_d=64, n_b=64)
        self.model_kwargs = model_kwargs

        # Model
        self.model: DeepONet2D | FNO2D | None = None
        self.model_metadata: dict[str, Any] = {}

        # Load checkpoint
        if checkpoint_path is not None:
            self._load_model(checkpoint_path)
        else:
            # Try to find default checkpoint
            checkpoint_path = self._find_default_checkpoint()
            if checkpoint_path is not None:
                self._load_model(checkpoint_path)
            else:
                raise FileNotFoundError(
                    f"No checkpoint found for {model_type}. "
                    f"Train the model first using 'python -m models_2d.neural_operators.train --model-type {model_type}'. "
                    f"Or provide a checkpoint_path explicitly."
                )

    def _find_default_checkpoint(self) -> Path | None:
        """Search for default checkpoint in common locations."""
        root = Path(__file__).parent.parent.parent
        
        # Check default path first
        default_path = self.DEFAULT_PATHS.get(self.model_type)
        if default_path:
            path = root / default_path
            if path.exists():
                return path
        
        # Search patterns for trained models
        search_patterns = {
            "deeponet": [
                "checkpoints_2d/deeponet/**/*.pt",
                "**/deeponet*best*.pt",
            ],
            "fno": [
                "checkpoints_2d/fno/**/*.pt",
                "**/fno*best*.pt",
            ],
        }
        
        for pattern in search_patterns.get(self.model_type, []):
            matches = list(root.glob(pattern))
            if matches:
                # Return the most recent checkpoint
                return sorted(matches, key=lambda p: p.stat().st_mtime)[-1]
        
        return None

    def _load_model(self, checkpoint_path: str | Path):
        """Load model from checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Get config from checkpoint or use defaults
        config = checkpoint.get('config', {})
        model_type = config.get('model_type', self.model_type)
        self.model_type = model_type

        # Create model
        if model_type == "deeponet":
            self.model = DeepONet2D(
                signal_dim=config.get('signal_dim', 64 * 64),
                grid_size=config.get('grid_size', 64),
                branch_dims=config.get('branch_dims', [512, 256, 128]),
                trunk_dims=config.get('trunk_dims', [128, 128, 128]),
                output_dim=config.get('output_dim', 128),
            )
        elif model_type == "fno":
            self.model = FNO2D(
                in_channels=config.get('in_channels', 3),
                hidden_channels=config.get('hidden_channels', 64),
                n_layers=config.get('n_layers', 4),
                modes=config.get('modes', 16),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)  # Move model to device
        self.model.eval()

        self.model_metadata = {
            'checkpoint_path': str(path),
            'model_type': model_type,
            'epoch': config.get('epoch', 'unknown'),
            'val_loss': config.get('val_loss', 'unknown'),
        }

    def get_model_name(self) -> str:
        """Return the model name."""
        return f"{self.model_type}"

    def get_model_info(self) -> dict:
        """Return model information."""
        return {
            "model_name": self.MODEL_NAME,
            "model_type": self.model_type,
            "model_type_full": "DeepONet" if self.model_type == "deeponet" else "FNO",
            "description": f"{self.model_type.upper()} Neural Operator for DEXSY",
            "paper_reference": "Section 2.4.4 in steven_submission.pdf",
            **self.model_metadata,
        }

    def _predict_distribution(self, model_inputs) -> np.ndarray:
        """Run model inference."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Provide checkpoint_path or call _load_model().")

        # Convert numpy to tensor if needed
        if isinstance(model_inputs, np.ndarray):
            model_inputs = torch.from_numpy(model_inputs).float()

        with torch.no_grad():
            predictions = self.model(model_inputs.to(self.device))
            return predictions.cpu().numpy()

    @staticmethod
    def _normalize_distribution(spectrum: np.ndarray) -> np.ndarray:
        """Normalize one 2D spectrum to unit mass."""
        total = float(np.sum(spectrum))
        if total <= 1e-12:
            return np.full_like(spectrum, 1.0 / spectrum.size)
        return spectrum / total

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
        axes[0].set_title(f"Input 64x64 DEXSY Signal")
        axes[0].set_xlabel("b2 index")
        axes[0].set_ylabel("b1 index")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Reconstructed spectrum
        dei = compute_dei(reconstructed)
        im1 = axes[1].imshow(reconstructed, cmap="magma", origin="lower")
        axes[1].set_title(f"{self.model_type.upper()} Reconstruction\nDEI={dei:.3f}")
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
        if self.model is None:
            raise RuntimeError("Model not loaded. Provide checkpoint_path.")

        # Validate signal
        validated = validate_signal_grid(signal, self.forward_model)
        signal_2d = validated[0, 0]

        # Select correct input based on model type
        if self.model_type == "deeponet":
            # DeepONet takes raw signal: [1, 1, H, W]
            model_inputs = validated
        else:
            # FNO takes 3-channel input
            model_inputs = build_model_inputs(validated, self.forward_model)

        # Predict
        predictions = self._predict_distribution(model_inputs)
        reconstructed = self._normalize_distribution(predictions[0, 0])

        summary = self._summarize_prediction(signal_2d, reconstructed, true_spectrum)

        # Figure
        figure = None
        if include_figure:
            figure = self._create_figure(signal_2d, reconstructed, true_spectrum)

        return PredictionResult(
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
        if self.model is None:
            raise RuntimeError("Model not loaded. Provide checkpoint_path.")

        validated = validate_signal_grid(signals, self.forward_model)

        # Select correct input based on model type
        if self.model_type == "deeponet":
            # DeepONet takes raw signal: [N, 1, H, W]
            model_inputs = validated
        else:
            # FNO takes 3-channel input
            model_inputs = build_model_inputs(validated, self.forward_model)

        # Predict
        predictions = self._predict_distribution(model_inputs)
        # predictions shape: [N, 1, H, W]

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
            # reconstructed shape: [1, H, W] -> take [0] to get [H, W]
            if reconstructed.ndim == 3:
                reconstructed_2d = reconstructed[0]  # [H, W]
            else:
                reconstructed_2d = reconstructed
            reconstructed_2d = self._normalize_distribution(reconstructed_2d)

            signal_2d = validated[idx, 0]
            gt = true_array[idx] if true_array is not None else None
            summary = self._summarize_prediction(signal_2d, reconstructed_2d, gt)

            figure = None
            if include_figures:
                figure = self._create_figure(signal_2d, reconstructed_2d, gt)

            results.append(PredictionResult(
                signal=signal_2d,
                reconstructed_spectrum=reconstructed_2d,
                dei=float(summary["dei"]),
                summary_metrics=summary,
                metadata={**self.model_metadata, "batch_index": idx},
                ground_truth_spectrum=gt,
                source_name=source_names[idx] if idx < len(source_names) else None,
                figure=figure,
            ))

        return results


if __name__ == "__main__":
    print("Testing Neural Operator InferencePipeline...")
    print("\nNote: Neural Operators require trained checkpoints.")
    print("Supported model types: 'deeponet', 'fno'")

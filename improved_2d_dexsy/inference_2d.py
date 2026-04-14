"""Inference helpers and high-level prediction API for the 2D DEXSY model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .forward_model_2d import ForwardModel2D, compute_dei
from .model_2d import get_model
from .preprocessing_2d import build_model_inputs, build_position_channel, validate_signal_grid

DEFAULT_MODEL_NAME = "attention_unet"
DEFAULT_CHECKPOINTS = {
    "attention_unet": "attention_unet_best_model.pt",
}


def available_models() -> list[str]:
    """Return the currently supported high-level inference model names."""
    return sorted(DEFAULT_CHECKPOINTS.keys())


def resolve_repo_root() -> Path:
    """Resolve the repository root from the package location."""
    return Path(__file__).resolve().parent.parent


def resolve_checkpoint_path(
    checkpoint_path: str | Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Path:
    """
    Resolve a checkpoint path for the selected model.

    If ``checkpoint_path`` is omitted, the bundled checkpoint in ``checkpoints/``
    is used. This keeps ``predict_from_signal(signal)`` simple for Colab and
    local usage while leaving room to swap models later.
    """
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.is_absolute():
            path = (resolve_repo_root() / path).resolve()
        return path

    try:
        filename = DEFAULT_CHECKPOINTS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available_models()}"
        ) from exc
    return resolve_repo_root() / "checkpoints" / filename


@dataclass
class PredictionResult:
    """Structured output for one reconstructed 2D DEXSY prediction."""

    signal: np.ndarray
    model_inputs: np.ndarray
    reconstructed_spectrum: np.ndarray
    dei: float
    summary_metrics: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    figure: Any | None = None

    @property
    def spectrum(self) -> np.ndarray:
        """Backwards-friendly alias for the reconstructed spectrum."""
        return self.reconstructed_spectrum


class DEXSYInferencePipeline:
    """
    Stable inference entrypoint for the current 2D workflow.

    This class wraps:
    - forward-model-aware preprocessing
    - checkpoint resolution and model loading
    - spectrum reconstruction
    - DEI computation
    - summary metric and figure generation
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        checkpoint_path: str | Path | None = None,
        device: torch.device | str | None = None,
        forward_model: ForwardModel2D | None = None,
    ):
        self.model_name = model_name
        self.checkpoint_path = resolve_checkpoint_path(checkpoint_path, model_name=model_name)
        self.forward_model = forward_model or ForwardModel2D(n_d=64, n_b=64)
        self.model, self.model_metadata = load_trained_model(
            self.checkpoint_path,
            device=device,
            model_name=model_name,
        )
        self.device = next(self.model.parameters()).device

    def summarize_prediction(
        self,
        signal: np.ndarray,
        reconstructed_spectrum: np.ndarray,
        true_spectrum: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute compact summary metrics for one prediction."""
        metrics: dict[str, Any] = {
            "signal_shape": tuple(signal.shape),
            "spectrum_shape": tuple(reconstructed_spectrum.shape),
            "prediction_mass": float(reconstructed_spectrum.sum()),
            "prediction_peak": float(reconstructed_spectrum.max()),
            "prediction_min": float(reconstructed_spectrum.min()),
            "signal_min": float(signal.min()),
            "signal_max": float(signal.max()),
            "signal_mean": float(signal.mean()),
            "dei": float(compute_dei(reconstructed_spectrum)),
            "model_name": self.model_name,
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
        }
        if true_spectrum is not None:
            true_array = np.asarray(true_spectrum, dtype=np.float32)
            if true_array.shape != reconstructed_spectrum.shape:
                raise ValueError(
                    "true_spectrum must match the reconstructed spectrum shape. "
                    f"Expected {reconstructed_spectrum.shape}, got {true_array.shape}."
                )
            metrics["ground_truth_dei"] = float(compute_dei(true_array))
            metrics["mse"] = float(np.mean((true_array - reconstructed_spectrum) ** 2))
            metrics["mae"] = float(np.mean(np.abs(true_array - reconstructed_spectrum)))
        return metrics

    def create_figure(
        self,
        signal: np.ndarray,
        reconstructed_spectrum: np.ndarray,
        true_spectrum: np.ndarray | None = None,
    ):
        """Create a compact visual summary for one prediction."""
        import matplotlib.pyplot as plt

        if true_spectrum is None:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            axes = np.atleast_1d(axes)

            im0 = axes[0].imshow(signal, cmap="viridis", origin="lower")
            axes[0].set_title("Input 64x64 DEXSY Signal")
            axes[0].set_xlabel("b2 index")
            axes[0].set_ylabel("b1 index")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(reconstructed_spectrum, cmap="magma", origin="lower")
            axes[1].set_title(f"Reconstructed Spectrum\nDEI={compute_dei(reconstructed_spectrum):.3f}")
            axes[1].set_xlabel("D2 index")
            axes[1].set_ylabel("D1 index")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        else:
            true_array = np.asarray(true_spectrum, dtype=np.float32)
            vmax = max(float(true_array.max()), float(reconstructed_spectrum.max()), 1e-6)
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

            im0 = axes[0].imshow(signal, cmap="viridis", origin="lower")
            axes[0].set_title("Input 64x64 DEXSY Signal")
            axes[0].set_xlabel("b2 index")
            axes[0].set_ylabel("b1 index")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(true_array, cmap="magma", origin="lower", vmin=0.0, vmax=vmax)
            axes[1].set_title(f"Ground Truth Spectrum\nDEI={compute_dei(true_array):.3f}")
            axes[1].set_xlabel("D2 index")
            axes[1].set_ylabel("D1 index")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(reconstructed_spectrum, cmap="magma", origin="lower", vmin=0.0, vmax=vmax)
            axes[2].set_title(f"Reconstructed Spectrum\nDEI={compute_dei(reconstructed_spectrum):.3f}")
            axes[2].set_xlabel("D2 index")
            axes[2].set_ylabel("D1 index")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        fig.tight_layout()
        return fig

    def predict_from_signal(
        self,
        signal: np.ndarray,
        *,
        true_spectrum: np.ndarray | None = None,
        include_figure: bool = True,
    ) -> PredictionResult:
        """Run the full inference pipeline for one 64x64 DEXSY signal matrix."""
        validated = validate_signal_grid(signal, self.forward_model)
        model_inputs = build_model_inputs(validated, self.forward_model)
        reconstructed = predict_distribution(self.model, model_inputs, device=self.device)[0, 0]
        signal_2d = validated[0, 0]
        summary = self.summarize_prediction(signal_2d, reconstructed, true_spectrum=true_spectrum)
        figure = None
        if include_figure:
            figure = self.create_figure(signal_2d, reconstructed, true_spectrum=true_spectrum)
        metadata = {
            **self.model_metadata,
            "model_name": self.model_name,
            "forward_model_grid": {
                "n_b": int(self.forward_model.n_b),
                "n_d": int(self.forward_model.n_d),
            },
        }
        return PredictionResult(
            signal=signal_2d,
            model_inputs=model_inputs[0],
            reconstructed_spectrum=reconstructed,
            dei=float(summary["dei"]),
            summary_metrics=summary,
            metadata=metadata,
            figure=figure,
        )


def load_trained_model(
    checkpoint_path: str | Path,
    device: torch.device | str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
):
    """Load a trained model from a bundled checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    base_filters = state_dict["enc1.conv1.weight"].shape[0]
    in_channels = state_dict["enc1.conv1.weight"].shape[1]

    model = get_model(
        model_name=model_name,
        base_filters=base_filters,
        in_channels=in_channels,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "base_filters": int(base_filters),
        "in_channels": int(in_channels),
        "epoch": checkpoint.get("epoch"),
        "val_loss": checkpoint.get("val_loss"),
    }
    return model, metadata


def predict_distribution(
    model: torch.nn.Module,
    inputs: np.ndarray,
    device: torch.device | str | None = None,
    batch_size: int = 16,
) -> np.ndarray:
    """Run batched inference and return numpy predictions."""
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    array = np.asarray(inputs, dtype=np.float32)
    if array.ndim == 3:
        array = array[None, :, :, :]
    if array.ndim != 4:
        raise ValueError(f"Unsupported input shape: {array.shape}")

    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(array), batch_size):
            batch = torch.from_numpy(array[start:start + batch_size]).to(device)
            preds = model(batch).cpu().numpy()
            outputs.append(preds)
    return np.concatenate(outputs, axis=0)


def predict_from_signal(
    signal: np.ndarray,
    *,
    checkpoint_path: str | Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: torch.device | str | None = None,
    forward_model: ForwardModel2D | None = None,
    true_spectrum: np.ndarray | None = None,
    include_figure: bool = True,
) -> PredictionResult:
    """
    High-level inference entrypoint for one 64x64 DEXSY signal matrix.

    This is the main stable API for the interface layer:

    - input: ``64x64`` DEXSY signal matrix
    - outputs: reconstructed spectrum, DEI, figure, summary metrics
    """
    pipeline = DEXSYInferencePipeline(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device,
        forward_model=forward_model,
    )
    return pipeline.predict_from_signal(
        signal,
        true_spectrum=true_spectrum,
        include_figure=include_figure,
    )

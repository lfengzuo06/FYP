"""
Inference utilities for non-Gaussian 3C inverse CNN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import (
    NonGaussian3CInverseNet,
    PATHWAY_ORDER_3C,
)


@dataclass
class NonGaussian3CInferenceResult:
    """Structured output for one prediction."""

    signal: np.ndarray
    pathway_weights: np.ndarray
    pathway_weight_matrix: np.ndarray
    dei: float
    pathway_dict: dict[str, float]
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class InferencePipeline:
    """
    Inference pipeline for non-Gaussian 3C inverse CNN.
    """

    MODEL_NAME = "non_gaussian_3c_cnn"

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device | str | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        cfg = checkpoint.get("config", {})

        self.model = NonGaussian3CInverseNet(
            base_channels=int(cfg.get("base_channels", 32)),
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            dropout=float(cfg.get("dropout", 0.15)),
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.config = cfg

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.MODEL_NAME,
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "config": self.config,
        }

    def _to_input_tensor(self, signal: np.ndarray) -> torch.Tensor:
        s = np.asarray(signal, dtype=np.float32)
        if s.ndim != 2:
            raise ValueError(f"Expected 2D signal matrix, got shape {s.shape}")
        return torch.from_numpy(s[None, None, :, :]).to(self.device)

    @staticmethod
    def _summary(weights: np.ndarray, dei: float) -> dict[str, float]:
        return {
            "dei": float(dei),
            "w_sum": float(weights.sum()),
            "w_max": float(weights.max()),
            "w_min": float(weights.min()),
        }

    def predict(self, signal: np.ndarray) -> NonGaussian3CInferenceResult:
        x = self._to_input_tensor(signal)

        with torch.no_grad():
            pred = self.model(x)
            w = pred.pathway_weights[0].detach().cpu().numpy().astype(np.float32)
            wm = pred.pathway_weight_matrix[0].detach().cpu().numpy().astype(np.float32)
            dei = float(pred.dei[0].item())

        pathway_dict = {k: float(v) for k, v in zip(PATHWAY_ORDER_3C, w)}

        return NonGaussian3CInferenceResult(
            signal=np.asarray(signal, dtype=np.float32),
            pathway_weights=w,
            pathway_weight_matrix=wm,
            dei=dei,
            pathway_dict=pathway_dict,
            summary_metrics=self._summary(w, dei),
            metadata={
                "model_name": self.MODEL_NAME,
                "checkpoint_path": str(self.checkpoint_path),
                "device": str(self.device),
            },
        )

    def predict_batch(self, signals: np.ndarray) -> list[NonGaussian3CInferenceResult]:
        arr = np.asarray(signals, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected shape (N,H,W), got {arr.shape}")

        out: list[NonGaussian3CInferenceResult] = []
        for i in range(arr.shape[0]):
            out.append(self.predict(arr[i]))
        return out


def predict(signal: np.ndarray, checkpoint_path: str | Path, device: str | None = None) -> NonGaussian3CInferenceResult:
    """Convenience API for one-shot inference."""
    return InferencePipeline(checkpoint_path=checkpoint_path, device=device).predict(signal)


__all__ = [
    "NonGaussian3CInferenceResult",
    "InferencePipeline",
    "predict",
]

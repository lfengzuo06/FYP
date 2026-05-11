"""
Inference utilities for Unified Attention U-Net (N-compartment).

This module provides a lightweight inference pipeline that predicts:
1) Reconstructed spectrum
2) Number of compartments (N)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dexsy_core.forward_model_nc import ForwardModelNC, create_forward_model_nc
from dexsy_core.preprocessing import build_model_inputs

from .model import AttentionUNetUnified


@dataclass
class UnifiedPredictionResult:
    """Structured prediction output for one sample."""

    signal: np.ndarray
    reconstructed_spectrum: np.ndarray
    predicted_n: int
    n_logits: np.ndarray
    n_probabilities: np.ndarray
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth_spectrum: np.ndarray | None = None
    ground_truth_n: int | None = None


class UnifiedInferencePipeline:
    """Inference pipeline for unified N-compartment Attention U-Net."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: torch.device | str | None = None,
        forward_model: ForwardModelNC | None = None,
        n_d: int | None = None,
        n_b: int | None = None,
    ):
        if checkpoint_path is None:
            checkpoint_path = self._find_default_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError(
                "No unified checkpoint found under checkpoints_nd. "
                "Please pass checkpoint_path explicitly."
            )

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        cfg = checkpoint.get("config", {})
        # best_model.pt in training loop may not include config; recover from
        # sibling final_model.pt when available.
        if not cfg:
            final_ckpt = self.checkpoint_path.parent / "final_model.pt"
            if final_ckpt.exists():
                try:
                    final_obj = torch.load(final_ckpt, map_location="cpu")
                    cfg = final_obj.get("config", {}) or cfg
                except Exception:
                    pass

        self.n_d = int(n_d if n_d is not None else cfg.get("n_d", 64))
        self.n_b = int(n_b if n_b is not None else cfg.get("n_b", self.n_d))
        self.n_min = int(cfg.get("n_min", 2))
        if "n_max" in cfg:
            self.n_max = int(cfg["n_max"])
        else:
            # Infer number of classes from classifier output layer.
            n_classes = None
            if "n_classifier.fusion.4.bias" in state_dict:
                n_classes = int(state_dict["n_classifier.fusion.4.bias"].shape[0])
            elif "n_classifier.fusion.4.weight" in state_dict:
                n_classes = int(state_dict["n_classifier.fusion.4.weight"].shape[0])
            if n_classes is None:
                n_classes = 6  # fallback for default n_min=2,n_max=7
            self.n_max = self.n_min + n_classes - 1
        in_channels = int(cfg.get("in_channels", 3))
        base_filters = int(cfg.get("base_filters", 32))

        self.forward_model = (
            forward_model
            if forward_model is not None
            else create_forward_model_nc(n_d=self.n_d, n_b=self.n_b)
        )

        self.model = AttentionUNetUnified(
            in_channels=in_channels,
            base_filters=base_filters,
            n_min=self.n_min,
            n_max=self.n_max,
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _find_default_checkpoint(self) -> Path | None:
        """Find newest unified best-model checkpoint under checkpoints_nd."""
        root = Path(__file__).resolve().parent.parent.parent / "checkpoints_nd"
        if not root.exists():
            return None
        candidates = sorted(root.glob("unified_n*_g*/checkpoints_*/best_model.pt"))
        if not candidates:
            candidates = sorted(root.glob("unified_n*/checkpoints_*/best_model.pt"))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _to_input_tensor(self, signal: np.ndarray) -> torch.Tensor:
        sig = np.asarray(signal, dtype=np.float32)
        if sig.shape != (self.n_b, self.n_b):
            raise ValueError(
                f"Expected signal shape ({self.n_b}, {self.n_b}), got {sig.shape}"
            )
        sig_4d = sig.reshape(1, 1, self.n_b, self.n_b)
        inp = build_model_inputs(sig_4d, self.forward_model)
        return torch.from_numpy(inp).to(self.device)

    @staticmethod
    def _spectrum_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
        pred_f = pred.astype(np.float64)
        true_f = true.astype(np.float64)
        mse = float(np.mean((pred_f - true_f) ** 2))
        mae = float(np.mean(np.abs(pred_f - true_f)))
        return {"mse": mse, "mae": mae}

    def predict(
        self,
        signal: np.ndarray,
        true_spectrum: np.ndarray | None = None,
        true_n: int | None = None,
    ) -> UnifiedPredictionResult:
        """Run inference on one signal matrix."""
        inp = self._to_input_tensor(signal)

        with torch.no_grad():
            spectrum_pred, n_logits = self.model(inp)
            spectrum_np = spectrum_pred[0, 0].cpu().numpy().astype(np.float32)
            logits_np = n_logits[0].cpu().numpy().astype(np.float32)
            probs_np = torch.softmax(n_logits[0], dim=0).cpu().numpy().astype(np.float32)
            n_pred = int(torch.argmax(n_logits[0]).item() + self.n_min)

        metrics: dict[str, Any] = {}
        if true_spectrum is not None:
            metrics.update(self._spectrum_metrics(spectrum_np, np.asarray(true_spectrum)))
        if true_n is not None:
            metrics["n_accuracy"] = float(int(n_pred == int(true_n)))

        return UnifiedPredictionResult(
            signal=np.asarray(signal, dtype=np.float32),
            reconstructed_spectrum=spectrum_np,
            predicted_n=n_pred,
            n_logits=logits_np,
            n_probabilities=probs_np,
            summary_metrics=metrics,
            metadata={
                "checkpoint_path": str(self.checkpoint_path),
                "n_range": [self.n_min, self.n_max],
                "grid_size": self.n_d,
                "device": str(self.device),
            },
            ground_truth_spectrum=None if true_spectrum is None else np.asarray(true_spectrum, dtype=np.float32),
            ground_truth_n=None if true_n is None else int(true_n),
        )

    def predict_batch(
        self,
        signals: np.ndarray,
        true_spectra: np.ndarray | None = None,
        true_n_values: np.ndarray | None = None,
    ) -> list[UnifiedPredictionResult]:
        """Run inference on a batch of signal matrices."""
        arr = np.asarray(signals, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1:] != (self.n_b, self.n_b):
            raise ValueError(
                f"Expected signals shape (N, {self.n_b}, {self.n_b}), got {arr.shape}"
            )

        results: list[UnifiedPredictionResult] = []
        for i in range(arr.shape[0]):
            gt_s = None if true_spectra is None else true_spectra[i]
            gt_n = None if true_n_values is None else int(true_n_values[i])
            results.append(self.predict(arr[i], true_spectrum=gt_s, true_n=gt_n))
        return results

"""Helpers for loading the trained 2D model and running inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .forward_model_2d import ForwardModel2D
from .model_2d import get_model


def build_position_channel(forward_model: ForwardModel2D) -> np.ndarray:
    """Build the same positional channel used during training."""
    b1 = forward_model.b1.astype(np.float32)
    b2 = forward_model.b2.astype(np.float32)
    positive = np.concatenate([b1[b1 > 0], b2[b2 > 0]])
    floor = float(np.min(positive)) if positive.size else 1.0
    log_b1 = np.log10(np.maximum(b1, floor))
    log_b2 = np.log10(np.maximum(b2, floor))
    log_b1 = (log_b1 - log_b1.min()) / (log_b1.max() - log_b1.min() + 1e-8)
    log_b2 = (log_b2 - log_b2.min()) / (log_b2.max() - log_b2.min() + 1e-8)
    return (0.5 * (log_b1[:, None] + log_b2[None, :])).astype(np.float32)


def build_model_inputs(noisy_signals: np.ndarray, forward_model: ForwardModel2D) -> np.ndarray:
    """
    Convert one or more DEXSY signals into the 3-channel model input.

    Accepted shapes:
    - (n_b, n_b)
    - (N, n_b, n_b)
    - (N, 1, n_b, n_b)
    """
    signals = np.asarray(noisy_signals, dtype=np.float32)
    if signals.ndim == 2:
        signals = signals[None, None, :, :]
    elif signals.ndim == 3:
        signals = signals[:, None, :, :]
    elif signals.ndim != 4:
        raise ValueError(f"Unsupported signal shape: {signals.shape}")

    raw = signals[:, 0]
    log_signal = np.log(raw + 1e-6)
    log_min = log_signal.min(axis=(1, 2), keepdims=True)
    log_max = log_signal.max(axis=(1, 2), keepdims=True)
    log_signal = (log_signal - log_min) / (log_max - log_min + 1e-8)
    pos = np.broadcast_to(build_position_channel(forward_model), raw.shape)
    stacked = np.stack([raw, log_signal.astype(np.float32), pos.astype(np.float32)], axis=1)
    return stacked.astype(np.float32)


def load_trained_model(
    checkpoint_path: str | Path,
    device: torch.device | str | None = None,
    model_name: str = "attention_unet",
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

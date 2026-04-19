"""
Shared preprocessing helpers for 2D DEXSY training and inference.
"""

from __future__ import annotations

import numpy as np


def ensure_signal_batch(signals: np.ndarray) -> np.ndarray:
    """
    Normalise supported signal shapes to ``(N, 1, n_b, n_b)``.

    Accepted inputs:
    - ``(n_b, n_b)``
    - ``(N, n_b, n_b)``
    - ``(N, 1, n_b, n_b)``
    """
    array = np.asarray(signals, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, None, :, :]
    elif array.ndim == 3:
        array = array[:, None, :, :]
    elif array.ndim != 4:
        raise ValueError(f"Unsupported signal shape: {array.shape}")

    if array.shape[1] != 1:
        raise ValueError(
            "Expected one signal channel after normalisation, "
            f"but received shape {array.shape}."
        )
    return array


def validate_signal_grid(signals: np.ndarray, forward_model) -> np.ndarray:
    """Validate that a signal batch matches the forward model signal grid."""
    array = ensure_signal_batch(signals)
    expected_shape = (forward_model.n_b, forward_model.n_b)
    if tuple(array.shape[-2:]) != expected_shape:
        raise ValueError(
            "Signal matrix shape does not match the configured forward model. "
            f"Expected {expected_shape}, got {tuple(array.shape[-2:])}."
        )
    return array


def build_position_channel(forward_model) -> np.ndarray:
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


def build_model_inputs(noisy_signals: np.ndarray, forward_model) -> np.ndarray:
    """
    Convert one or more DEXSY signals into the 3-channel model input.

    Channel 1: Raw signal
    Channel 2: Log-transformed signal (normalized)
    Channel 3: Positional encoding from D-grid

    Args:
        noisy_signals: Input signal array
        forward_model: ForwardModel2D instance

    Returns:
        3-channel input array with shape (N, 3, n_b, n_b)
    """
    signals = validate_signal_grid(noisy_signals, forward_model)
    raw = signals[:, 0]
    log_signal = np.log(raw + 1e-6)
    log_min = log_signal.min(axis=(1, 2), keepdims=True)
    log_max = log_signal.max(axis=(1, 2), keepdims=True)
    log_signal = (log_signal - log_min) / (log_max - log_min + 1e-8)
    pos = np.broadcast_to(build_position_channel(forward_model), raw.shape)
    stacked = np.stack([raw, log_signal.astype(np.float32), pos.astype(np.float32)], axis=1)
    return stacked.astype(np.float32)


def normalize_spectrum(spectrum: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize spectrum to sum to 1 (L1 normalization).

    Args:
        spectrum: Input spectrum array
        eps: Small value to avoid division by zero

    Returns:
        Normalized spectrum
    """
    array = np.asarray(spectrum, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    normalized = array / (array.sum(axis=(1, 2), keepdims=True) + eps)
    return normalized[0] if spectrum.ndim == 2 else normalized


if __name__ == "__main__":
    from dexsy_core.forward_model import ForwardModel2D

    fm = ForwardModel2D()
    signals = np.random.rand(64, 64).astype(np.float32)
    inputs = build_model_inputs(signals, fm)
    print(f"Input shape: {inputs.shape}")  # Should be (1, 3, 64, 64)
    print(f"Channels: {inputs[0, 0].min():.4f} - {inputs[0, 0].max():.4f}")
    print(f"Log channel: {inputs[0, 1].min():.4f} - {inputs[0, 1].max():.4f}")
    print(f"Position channel: {inputs[0, 2].min():.4f} - {inputs[0, 2].max():.4f}")

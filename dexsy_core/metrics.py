"""
Shared evaluation metrics for DEXSY experiments.

Provides:
- DEI (Diffusion Exchange Index) computation
- MSE, MAE, RMSE metrics
- SSIM (Structural Similarity Index)
- Utility functions for metrics computation
"""

from __future__ import annotations

import numpy as np


def compute_dei(f: np.ndarray, diagonal_band_width: int = 5) -> float:
    """
    Compute Diffusion Exchange Index for broadened spectra.

    Mass with |i - j| <= ``diagonal_band_width`` is treated as lying in the
    D1 \\approx D2 (self-diffusion) band on the discrete grid; the rest counts
    toward exchange. A width of ~5 matches the spatial spread of NNLS-based
    2D ILT on this 64x64 grid.

    Args:
        f: 2D diffusion spectrum array
        diagonal_band_width: Width of diagonal band for DEI computation

    Returns:
        DEI value (ratio of off-diagonal to diagonal mass)
    """
    n = f.shape[0]
    ii, jj = np.indices((n, n))
    diagonal_mask = np.abs(ii - jj) <= diagonal_band_width
    diag_sum = float(f[diagonal_mask].sum())
    off_diag_sum = float(f[~diagonal_mask].sum())
    return off_diag_sum / (diag_sum + 1e-10)


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error.

    Args:
        y_true: Ground truth array
        y_pred: Predicted array

    Returns:
        MSE value
    """
    return float(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        y_true: Ground truth array
        y_pred: Predicted array

    Returns:
        MAE value
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        y_true: Ground truth array
        y_pred: Predicted array

    Returns:
        RMSE value
    """
    return np.sqrt(compute_mse(y_true, y_pred))


def compute_ssim(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_size: int = 8,
    k1: float = 0.01,
    k2: float = 0.03,
    L: float = None,
) -> float:
    """
    Compute Structural Similarity Index (SSIM) for 2D arrays.

    Simplified implementation for DEXSY spectra.

    Args:
        y_true: Ground truth 2D array
        y_pred: Predicted 2D array
        window_size: Size of the moving window
        k1, k2: Stability constants
        L: Dynamic range (if None, uses max value)

    Returns:
        SSIM value between -1 and 1
    """
    if L is None:
        L = max(y_true.max(), y_pred.max()) - min(y_true.min(), y_pred.min())
        L = max(L, 1e-8)

    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    mu_true = _uniform_filter(y_true, window_size)
    mu_pred = _uniform_filter(y_pred, window_size)

    mu_true_sq = mu_true ** 2
    mu_pred_sq = mu_pred ** 2
    mu_true_pred = mu_true * mu_pred

    sigma_true_sq = _uniform_filter(y_true ** 2, window_size) - mu_true_sq
    sigma_pred_sq = _uniform_filter(y_pred ** 2, window_size) - mu_pred_sq
    sigma_true_pred = _uniform_filter(y_true * y_pred, window_size) - mu_true_pred

    numerator = (2 * mu_true_pred + c1) * (2 * sigma_true_pred + c2)
    denominator = (mu_true_sq + mu_pred_sq + c1) * (sigma_true_sq + sigma_pred_sq + c2)
    ssim_map = numerator / (denominator + 1e-8)

    return float(np.mean(ssim_map))


def compute_dssim(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """
    Compute Dissimilarity Structural Similarity Index.

    DSSIM = 1 - SSIM/2, where 0 indicates perfect similarity.

    Args:
        y_true: Ground truth array
        y_pred: Predicted array
        **kwargs: Passed to compute_ssim

    Returns:
        DSSIM value (0 = perfect, higher = more dissimilar)
    """
    ssim = compute_ssim(y_true, y_pred, **kwargs)
    return 1.0 - ssim / 2.0


def _uniform_filter(arr: np.ndarray, size: int) -> np.ndarray:
    """Simple uniform (box) filter implementation."""
    from scipy.ndimage import uniform_filter as _scipy_uniform_filter
    if _scipy_uniform_filter is not None:
        return _scipy_uniform_filter(arr, size=size)
    kernel = np.ones((size, size), dtype=np.float64) / (size * size)
    from scipy.ndimage import convolve
    return convolve(arr.astype(np.float64), kernel, mode='nearest')


def compute_metrics_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dei_true: float = None,
    dei_pred: float = None,
    diagonal_band_width: int = 5,
) -> dict[str, float]:
    """
    Compute all standard metrics as a dictionary.

    Args:
        y_true: Ground truth spectrum
        y_pred: Predicted spectrum
        dei_true: Pre-computed DEI for ground truth (if None, computed)
        dei_pred: Pre-computed DEI for prediction (if None, computed)
        diagonal_band_width: DEI diagonal band width

    Returns:
        Dictionary with all metrics
    """
    if dei_true is None:
        dei_true = compute_dei(y_true, diagonal_band_width)
    if dei_pred is None:
        dei_pred = compute_dei(y_pred, diagonal_band_width)

    return {
        "mse": compute_mse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "rmse": np.sqrt(compute_mse(y_true, y_pred)),
        "ssim": compute_ssim(y_true, y_pred),
        "dssim": compute_dssim(y_true, y_pred),
        "dei_true": dei_true,
        "dei_pred": dei_pred,
        "dei_error": abs(dei_true - dei_pred),
    }


def compute_batch_metrics(
    y_true_batch: np.ndarray,
    y_pred_batch: np.ndarray,
    diagonal_band_width: int = 5,
) -> dict[str, dict]:
    """
    Compute per-sample and aggregate metrics for a batch.

    Args:
        y_true_batch: Batch of ground truth spectra (N, H, W) or (N, 1, H, W)
        y_pred_batch: Batch of predicted spectra (N, H, W) or (N, 1, H, W)
        diagonal_band_width: DEI diagonal band width

    Returns:
        Dictionary with 'per_sample' and 'aggregate' metrics
    """
    if y_true_batch.ndim == 4 and y_true_batch.shape[1] == 1:
        y_true_batch = y_true_batch[:, 0]
    if y_pred_batch.ndim == 4 and y_pred_batch.shape[1] == 1:
        y_pred_batch = y_pred_batch[:, 0]

    n_samples = len(y_true_batch)
    per_sample = {
        "mse": np.zeros(n_samples),
        "mae": np.zeros(n_samples),
        "dei_true": np.zeros(n_samples),
        "dei_pred": np.zeros(n_samples),
        "dei_error": np.zeros(n_samples),
    }

    for i in range(n_samples):
        per_sample["mse"][i] = compute_mse(y_true_batch[i], y_pred_batch[i])
        per_sample["mae"][i] = compute_mae(y_true_batch[i], y_pred_batch[i])
        per_sample["dei_true"][i] = compute_dei(y_true_batch[i], diagonal_band_width)
        per_sample["dei_pred"][i] = compute_dei(y_pred_batch[i], diagonal_band_width)
        per_sample["dei_error"][i] = abs(per_sample["dei_true"][i] - per_sample["dei_pred"][i])

    aggregate = {
        "mse_mean": float(np.mean(per_sample["mse"])),
        "mse_std": float(np.std(per_sample["mse"])),
        "mae_mean": float(np.mean(per_sample["mae"])),
        "mae_std": float(np.std(per_sample["mae"])),
        "dei_error_mean": float(np.mean(per_sample["dei_error"])),
        "dei_error_std": float(np.std(per_sample["dei_error"])),
    }

    return {"per_sample": per_sample, "aggregate": aggregate}


if __name__ == "__main__":
    y_true = np.random.rand(64, 64).astype(np.float32)
    y_pred = y_true + np.random.randn(64, 64).astype(np.float32) * 0.01

    print("Metrics test:")
    print(f"  MSE: {compute_mse(y_true, y_pred):.6f}")
    print(f"  MAE: {compute_mae(y_true, y_pred):.6f}")
    print(f"  RMSE: {compute_rmse(y_true, y_pred):.6f}")
    print(f"  SSIM: {compute_ssim(y_true, y_pred):.4f}")
    print(f"  DSSIM: {compute_dssim(y_true, y_pred):.4f}")
    print(f"  DEI: {compute_dei(y_true):.4f}")

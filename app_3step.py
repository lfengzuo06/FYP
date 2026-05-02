#!/usr/bin/env python3
"""
3-Step DEXSY Inference Interface

Step 1: Data Input (parametric generation / random generation / image upload)
Step 2: Model Selection & Inference
Step 3: Results Display (visualizations + metrics + export)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
import matplotlib.pyplot as plt

from improved_2d_dexsy import (
    CHECKPOINTS_DIR,
    CHECKPOINTS_DIR_3D,
    DEXSYInferencePipeline,
    ForwardModel2D,
    available_models,
    create_output_archive,
    create_run_output_dir,
    is_3c_model,
    list_available_checkpoints,
    list_available_checkpoints_3d,
    load_matrix,
    resolve_checkpoint_path,
    save_prediction_result,
    to_serializable,
)
from dexsy_core.metrics import compute_metrics_dict

# 2C and 3C checkpoint choices
CHECKPOINT_CHOICES_2D = [
    str(path.relative_to(CHECKPOINTS_DIR))
    for path in list_available_checkpoints()
]
CHECKPOINT_CHOICES_3D = [
    str(path.relative_to(CHECKPOINTS_DIR_3D))
    for path in list_available_checkpoints_3d()
]
CHECKPOINT_CHOICES = CHECKPOINT_CHOICES_2D + CHECKPOINT_CHOICES_3D


def _get_model_filtered_checkpoints(model_name: str) -> list[str]:
    """Get checkpoint choices filtered to a specific model folder."""
    if model_name in ("2d_ilt", "3d_ilt"):
        return []  # ILT doesn't need checkpoints

    # Determine which checkpoint list to use
    is_3c = is_3c_model(model_name)
    choices = CHECKPOINT_CHOICES_3D if is_3c else CHECKPOINT_CHOICES_2D

    # Build prefix based on model type
    model_prefix_map = {
        "attention_unet": "attention_unet/",
        "plain_unet": "plain_unet/",
        "pinn": "pinn" if is_3c else "pinn/",
        "deep_unfolding": "deep_unfolding" if is_3c else "deep_unfolding/",
        "deeponet": "deeponet/",
        "fno": "fno/",
        "attention_unet_3c": "attention_unet_3c/",
        "plain_unet_3c": "plain_unet_3c/",
        "pinn_3c": "pinn_3c/",
        "deep_unfolding_3c": "deep_unfolding_3c/",
        "diffusion_refiner": "diffusion_refiner/",
    }
    model_prefix = model_prefix_map.get(model_name, f"{model_name}/")

    filtered = [c for c in choices if c.startswith(model_prefix)]
    if not filtered:
        # Fallback: try without folder prefix
        filtered = [c for c in choices if model_name.replace("_3c", "") in c.lower()]
    return filtered


@dataclass
class SignalInputResult:
    """Result from Step 1 data input."""
    signal: np.ndarray
    ground_truth: np.ndarray | None
    params: dict
    input_method: str  # "params" | "random" | "image"
    n_compartments: int = 2  # 2 or 3


@dataclass
class InferenceResult:
    """Result from Step 2 inference."""
    signal: np.ndarray
    prediction: np.ndarray
    ground_truth: np.ndarray | None
    metrics: dict[str, Any]
    inference_time: float
    model_name: str
    figure: plt.Figure | None = None
    metadata: dict = field(default_factory=dict)


def default_checkpoint_choice(model_name: str) -> str | None:
    """Pick the preferred checkpoint shown in the dropdown for a model."""
    default_path = resolve_checkpoint_path(model_name=model_name)
    if default_path is None:
        return None
    try:
        # Try relative to 2D checkpoint dir first
        candidate = str(default_path.relative_to(CHECKPOINTS_DIR))
    except ValueError:
        try:
            # Try relative to 3D checkpoint dir
            candidate = str(default_path.relative_to(CHECKPOINTS_DIR_3D))
        except ValueError:
            candidate = str(default_path)
    if candidate in CHECKPOINT_CHOICES:
        return candidate
    return CHECKPOINT_CHOICES[0] if CHECKPOINT_CHOICES else None


def _plot_heatmap(data: np.ndarray, title: str, cmap: str = "plasma", figsize=(4, 3.5)) -> plt.Figure:
    """Create a heatmap figure."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, origin="lower")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("D2 / b2 index")
    ax.set_ylabel("D1 / b1 index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _plot_comparison(signal, ground_truth, prediction, dei_gt=None, dei_pred=None) -> plt.Figure:
    """Create a 4-panel comparison figure."""
    n_plots = 4 if ground_truth is not None else 3

    if n_plots == 4:
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    # Signal
    im0 = axes[0].imshow(signal, cmap="viridis", origin="lower")
    axes[0].set_title("Input Signal", fontsize=10)
    axes[0].set_xlabel("b2 index")
    axes[0].set_ylabel("b1 index")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    if ground_truth is not None:
        vmax = max(float(ground_truth.max()), float(prediction.max()), 1e-6)
        # Ground Truth
        im1 = axes[1].imshow(ground_truth, cmap="magma", origin="lower", vmin=0, vmax=vmax)
        gt_title = "Ground Truth"
        if dei_gt is not None:
            gt_title += f"\nDEI={dei_gt:.4f}"
        axes[1].set_title(gt_title, fontsize=10)
        axes[1].set_xlabel("D2 index")
        axes[1].set_ylabel("D1 index")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Prediction
        im2 = axes[2].imshow(prediction, cmap="magma", origin="lower", vmin=0, vmax=vmax)
        pred_title = "Model Output"
        if dei_pred is not None:
            pred_title += f"\nDEI={dei_pred:.4f}"
        axes[2].set_title(pred_title, fontsize=10)
        axes[2].set_xlabel("D2 index")
        axes[2].set_ylabel("D1 index")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # Difference
        diff = prediction - ground_truth
        im3 = axes[3].imshow(diff, cmap="RdBu_r", origin="lower", vmin=-vmax/4, vmax=vmax/4)
        axes[3].set_title("Difference\n(Pred - GT)", fontsize=10)
        axes[3].set_xlabel("D2 index")
        axes[3].set_ylabel("D1 index")
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    else:
        # No ground truth - just signal and prediction
        im1 = axes[1].imshow(prediction, cmap="magma", origin="lower")
        pred_title = "Model Output"
        if dei_pred is not None:
            pred_title += f"\nDEI={dei_pred:.4f}"
        axes[1].set_title(pred_title, fontsize=10)
        axes[1].set_xlabel("D2 index")
        axes[1].set_ylabel("D1 index")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Log signal
        im2 = axes[2].imshow(np.log(signal + 1e-6), cmap="viridis", origin="lower")
        axes[2].set_title("Log(Signal)", fontsize=10)
        axes[2].set_xlabel("b2 index")
        axes[2].set_ylabel("b1 index")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


def _generate_parametric(
    n_compartments: int,
    diffusion_1: float,
    diffusion_2: float,
    volume_fraction: float,
    mixing_time: float,
    noise_sigma: float,
    exchange_rate: float | None = None,
    diffusion_3: float | None = None,
    volume_fraction_3: float | None = None,
    exchange_rate_01: float | None = None,
    exchange_rate_02: float | None = None,
    exchange_rate_12: float | None = None,
) -> SignalInputResult:
    """Generate signal from parametric input."""
    fm = ForwardModel2D()

    if n_compartments == 2:
        if exchange_rate is None:
            raise ValueError("2-compartment generation requires exchange_rate.")
        diffusions = np.array([diffusion_1, diffusion_2], dtype=np.float64)
        volume_fractions = np.array([volume_fraction, 1 - volume_fraction], dtype=np.float64)

        # Use validation spectrum generator for exact parameters
        spectrum, _clean_signal, params = fm.generate_2c_validation_spectrum(
            diffusions=diffusions,
            volume_fractions=volume_fractions,
            exchange_rate=exchange_rate,
            mixing_time=mixing_time,
            jitter_pixels=0,
            smoothing_sigma=0.8,
        )

        # Add noise
        signal = fm.compute_signal(
            spectrum,
            noise_sigma=noise_sigma,
            normalize=True,
        ).astype(np.float32)
        spectrum = spectrum.astype(np.float32)

        params["mixing_time"] = mixing_time
        params["noise_sigma"] = noise_sigma
        params["diffusions"] = diffusions.tolist()
        params["volume_fractions"] = volume_fractions.tolist()
        params["exchange_rates"] = {"0-1": float(exchange_rate)}
        params["n_compartments"] = 2

        return SignalInputResult(
            signal=signal.astype(np.float32),
            ground_truth=spectrum.astype(np.float32),
            params=params,
            input_method="params",
            n_compartments=2,
        )

    else:
        # 3-compartment: build spectrum manually to match user-specified parameters
        vf1 = volume_fraction
        vf3 = volume_fraction_3
        vf2 = max(0.01, 1 - vf1 - vf3)  # Ensure positive
        diffusions = np.array([diffusion_1, diffusion_2, diffusion_3], dtype=np.float64)
        volume_fractions = np.array([vf1, vf2, vf3], dtype=np.float64)
        volume_fractions /= volume_fractions.sum()  # Normalize

        exchange_rates_mat = np.zeros((3, 3), dtype=np.float64)
        exchange_rates_mat[0, 1] = exchange_rates_mat[1, 0] = exchange_rate_01
        exchange_rates_mat[0, 2] = exchange_rates_mat[2, 0] = exchange_rate_02
        exchange_rates_mat[1, 2] = exchange_rates_mat[2, 1] = exchange_rate_12

        # Build weight matrix (same logic as _build_weight_matrix in forward model)
        exchange_probs = np.zeros_like(exchange_rates_mat)
        for i in range(3):
            for j in range(i + 1, 3):
                prob = 1.0 - np.exp(-exchange_rates_mat[i, j] * mixing_time)
                exchange_probs[i, j] = prob
                exchange_probs[j, i] = prob

        offdiag = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(i + 1, 3):
                offdiag_mass = exchange_probs[i, j] * volume_fractions[i] * volume_fractions[j]
                offdiag[i, j] = offdiag_mass
                offdiag[j, i] = offdiag_mass

        row_offdiag = offdiag.sum(axis=1)
        allowed = 0.49 * volume_fractions
        scale = 1.0
        if np.any(row_offdiag > allowed):
            scale = float(np.min(allowed[row_offdiag > 0] / row_offdiag[row_offdiag > 0]))
            scale = min(scale, 1.0)
            offdiag *= scale

        diag = volume_fractions - offdiag.sum(axis=1)
        diag = np.clip(diag, 1e-8, None)
        weight_matrix = offdiag.copy()
        np.fill_diagonal(weight_matrix, diag)
        weight_matrix = np.clip(weight_matrix, 0.0, None)
        weight_matrix /= weight_matrix.sum() + 1e-12

        # Project to grid
        base_indices = [fm._nearest_diffusion_index(d) for d in diffusions]
        smoothing_sigma = 0.8
        spectrum = np.zeros((64, 64), dtype=np.float64)

        for i in range(3):
            for j in range(3):
                weight = float(weight_matrix[i, j])
                if weight <= 0:
                    continue
                ii = base_indices[i]
                jj = base_indices[j]
                if ii == jj:
                    # Diagonal - spread along diagonal
                    radius = 3
                    for k in range(-radius, radius + 1):
                        if 0 <= ii + k < 64:
                            spectrum[ii + k, ii + k] += weight * np.exp(-0.5 * (k / smoothing_sigma) ** 2)
                else:
                    # Off-diagonal - 2D Gaussian
                    radius = int(np.ceil(3 * smoothing_sigma))
                    for di in range(-radius, radius + 1):
                        for dj in range(-radius, radius + 1):
                            ii2 = ii + di
                            jj2 = jj + dj
                            if 0 <= ii2 < 64 and 0 <= jj2 < 64:
                                spectrum[ii2, jj2] += weight * np.exp(-0.5 * ((di ** 2 + dj ** 2) / (smoothing_sigma ** 2)))

        spectrum = np.clip(spectrum, 0.0, None)
        spectrum /= spectrum.sum() + 1e-12
        spectrum = spectrum.astype(np.float32)

        # Generate signal with noise
        signal = fm.compute_signal(
            spectrum,
            noise_sigma=noise_sigma,
            normalize=True,
        ).astype(np.float32)

        params = {
            "n_compartments": 3,
            "mixing_time": mixing_time,
            "noise_sigma": noise_sigma,
            "diffusions": diffusions.tolist(),
            "volume_fractions": volume_fractions.tolist(),
            "exchange_rates": {
                "0-1": float(exchange_rate_01),
                "0-2": float(exchange_rate_02),
                "1-2": float(exchange_rate_12),
            },
            "weight_matrix": weight_matrix,
            "smoothing_sigma": smoothing_sigma,
        }

    return SignalInputResult(
        signal=signal.astype(np.float32),
        ground_truth=spectrum.astype(np.float32),
        params=params,
        input_method="params",
        n_compartments=n_compartments,
    )


def _generate_random(n_compartments: int) -> SignalInputResult:
    """Generate random signal."""
    fm = ForwardModel2D()

    if n_compartments == 2:
        spectrum, signal, params = fm.generate_2compartment_paper()
    else:
        spectrum, signal, params = fm.generate_3compartment_paper()

    return SignalInputResult(
        signal=signal.astype(np.float32),
        ground_truth=spectrum.astype(np.float32),
        params=params,
        input_method="random",
        n_compartments=n_compartments,
    )


def _generate_preview_plot(result: SignalInputResult) -> tuple[plt.Figure, str]:
    """Generate preview plot and parameter summary."""
    signal = result.signal
    ground_truth = result.ground_truth
    params = result.params

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    # Signal
    im0 = axes[0].imshow(signal, cmap="viridis", origin="lower")
    axes[0].set_title("Input Signal", fontsize=11)
    axes[0].set_xlabel("b2 index")
    axes[0].set_ylabel("b1 index")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Log Signal
    log_signal = np.log(signal + 1e-6)
    im1 = axes[1].imshow(log_signal, cmap="viridis", origin="lower")
    axes[1].set_title("Log(Signal)", fontsize=11)
    axes[1].set_xlabel("b2 index")
    axes[1].set_ylabel("b1 index")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Ground Truth
    if ground_truth is not None:
        im2 = axes[2].imshow(ground_truth, cmap="magma", origin="lower")
        gt_title = "Ground Truth Spectrum"
        if "mixing_time" in params:
            gt_title += f"\nTm={params['mixing_time']:.3f}s"
        axes[2].set_title(gt_title, fontsize=11)
        axes[2].set_xlabel("D2 index")
        axes[2].set_ylabel("D1 index")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        axes[2].axis("off")
        axes[2].set_title("No Ground Truth", fontsize=11)

    fig.tight_layout()

    # Build parameter summary
    summary_parts = []
    summary_parts.append(f"Compartments: {params.get('n_compartments', 'N/A')}")
    if "mixing_time" in params:
        summary_parts.append(f"Mixing Time: {params['mixing_time']:.4f}s")
    if "noise_sigma" in params:
        summary_parts.append(f"Noise Sigma: {params['noise_sigma']:.4f}")
    if "diffusions" in params:
        summary_parts.append(f"Diffusions: {params['diffusions']}")
    if "volume_fractions" in params:
        summary_parts.append(f"VF: {[f'{v:.3f}' for v in params['volume_fractions']]}")
    if "exchange_rates" in params:
        summary_parts.append(f"Exchange Rates: {params['exchange_rates']}")

    summary = "\n".join(summary_parts)
    return fig, summary


def _run_inference(
    signal: np.ndarray,
    model_name: str,
    checkpoint_name: str | None,
    device_name: str,
    ground_truth: np.ndarray | None,
) -> InferenceResult:
    """Run inference with the selected model."""
    is_3c = is_3c_model(model_name)

    if checkpoint_name:
        if is_3c:
            checkpoint_path = (CHECKPOINTS_DIR_3D / checkpoint_name).resolve()
        else:
            checkpoint_path = (CHECKPOINTS_DIR / checkpoint_name).resolve()
    else:
        checkpoint_path = resolve_checkpoint_path(model_name=model_name)

    device = None if device_name == "auto" else device_name
    fm = ForwardModel2D()

    start_time = time.time()

    pipeline = DEXSYInferencePipeline(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device,
        forward_model=fm,
    )

    result = pipeline.predict(
        signal,
        true_spectrum=ground_truth,
        include_figure=False,
    )

    inference_time = time.time() - start_time

    # Compute detailed metrics
    metrics = {}
    prediction = result.reconstructed_spectrum

    if ground_truth is not None:
        metrics = compute_metrics_dict(
            ground_truth.astype(np.float32),
            prediction.astype(np.float32),
        )
        metrics["inference_time"] = inference_time
        metrics["model_dei"] = result.dei
        metrics["prediction_mass"] = float(prediction.sum())
        metrics["prediction_peak"] = float(prediction.max())
    else:
        from dexsy_core.forward_model import compute_dei as fm_compute_dei
        metrics = {
            "inference_time": inference_time,
            "model_dei": result.dei,
            "prediction_mass": float(prediction.sum()),
            "prediction_peak": float(prediction.max()),
        }

    return InferenceResult(
        signal=signal,
        prediction=prediction,
        ground_truth=ground_truth,
        metrics=metrics,
        inference_time=inference_time,
        model_name=model_name,
        metadata=result.metadata,
    )


def _create_metrics_html(metrics: dict, ground_truth_available: bool) -> str:
    """Create HTML table for metrics display."""
    if ground_truth_available:
        rows = [
            ("MSE", f"{metrics.get('mse', 0):.6f}"),
            ("MAE", f"{metrics.get('mae', 0):.6f}"),
            ("RMSE", f"{metrics.get('rmse', 0):.6f}"),
            ("SSIM", f"{metrics.get('ssim', 0):.4f}"),
            ("DEI (Ground Truth)", f"{metrics.get('dei_true', 0):.4f}"),
            ("DEI (Prediction)", f"{metrics.get('dei_pred', metrics.get('model_dei', 0)):.4f}"),
            ("DEI Error", f"{metrics.get('dei_error', 0):.4f}"),
            ("Inference Time", f"{metrics.get('inference_time', 0):.4f}s"),
            ("Prediction Mass", f"{metrics.get('prediction_mass', 0):.4f}"),
            ("Prediction Peak", f"{metrics.get('prediction_peak', 0):.6f}"),
        ]
    else:
        rows = [
            ("DEI (Prediction)", f"{metrics.get('model_dei', 0):.4f}"),
            ("Inference Time", f"{metrics.get('inference_time', 0):.4f}s"),
            ("Prediction Mass", f"{metrics.get('prediction_mass', 0):.4f}"),
            ("Prediction Peak", f"{metrics.get('prediction_peak', 0):.6f}"),
        ]

    html = "<table style='width:100%; border-collapse: collapse;'>"
    for i, (name, value) in enumerate(rows):
        bg = "#f8f9fa" if i % 2 == 0 else "#ffffff"
        html += f"<tr style='background-color: {bg};'>"
        html += f"<td style='padding: 8px; border: 1px solid #dee2e6; font-weight: bold;'>{name}</td>"
        html += f"<td style='padding: 8px; border: 1px solid #dee2e6; text-align: right;'>{value}</td>"
        html += "</tr>"
    html += "</table>"
    return html


def _save_bundle(result: InferenceResult, output_dir: Path, stem: str = "result") -> Path:
    """Save results bundle to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{stem}_signal.npy", result.signal.astype(np.float32))
    np.save(output_dir / f"{stem}_prediction.npy", result.prediction.astype(np.float32))

    if result.ground_truth is not None:
        np.save(output_dir / f"{stem}_ground_truth.npy", result.ground_truth.astype(np.float32))

    # Metrics JSON
    with open(output_dir / f"{stem}_metrics.json", "w") as f:
        json.dump(to_serializable(result.metrics), f, indent=2)

    # Metadata JSON
    with open(output_dir / f"{stem}_metadata.json", "w") as f:
        json.dump(to_serializable(result.metadata), f, indent=2)

    # Comparison figure
    dei_gt = result.metrics.get("dei_true") if result.ground_truth is not None else None
    dei_pred = result.metrics.get("dei_pred") or result.metrics.get("model_dei")

    fig = _plot_comparison(
        result.signal,
        result.ground_truth,
        result.prediction,
        dei_gt=dei_gt,
        dei_pred=dei_pred,
    )
    fig.savefig(output_dir / f"{stem}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Create archive
    archive_path = create_output_archive(output_dir, output_dir / f"{stem}_bundle")
    return archive_path


def build_app():
    with gr.Blocks(title="2D DEXSY Inference Interface", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 2D DEXSY Reconstruction Interface

        A three-step interface for DEXSY signal reconstruction:
        1. **Data Input** - Generate or upload signals
        2. **Inference** - Run selected model
        3. **Results** - View outputs and metrics
        """)

        # Global state
        input_state = gr.State(value=None)
        result_state = gr.State(value=None)
        output_dir_state = gr.State(value=None)

        with gr.Tabs() as tabs:
            # ================================================================
            # TAB 1: DATA INPUT
            # ================================================================
            with gr.TabItem("Step 1: Data Input") as tab1:
                gr.Markdown("### Select Input Method")

                with gr.Tabs():
                    # --- Tab 1A: Parametric Generation ---
                    with gr.TabItem("Parametric Generation"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("**2-Compartment Parameters**")

                                n_compartments_2c = gr.Radio(
                                    choices=[2, 3],
                                    value=2,
                                    label="Number of Compartments",
                                )

                                with gr.Group(visible=True) as params_2c:
                                    diffusion_1 = gr.Slider(
                                        minimum=5e-12, maximum=3e-11,
                                        value=1e-11, step=1e-12,
                                        label="D1 (Intracellular, m²/s)",
                                    )
                                    diffusion_2 = gr.Slider(
                                        minimum=3e-11, maximum=5e-9,
                                        value=1e-9, step=1e-10,
                                        label="D2 (Extracellular, m²/s)",
                                    )
                                    volume_fraction = gr.Slider(
                                        minimum=0.1, maximum=0.9,
                                        value=0.5, step=0.05,
                                        label="Volume Fraction (D1)",
                                    )
                                    exchange_rate = gr.Slider(
                                        minimum=0.1, maximum=30.0,
                                        value=5.0, step=0.5,
                                        label="Exchange Rate (s⁻¹)",
                                    )
                                    mixing_time = gr.Slider(
                                        minimum=0.015, maximum=0.300,
                                        value=0.100, step=0.005,
                                        label="Mixing Time (s)",
                                    )
                                    noise_sigma = gr.Slider(
                                        minimum=0.001, maximum=0.050,
                                        value=0.010, step=0.001,
                                        label="Noise Sigma",
                                    )

                                with gr.Group(visible=False) as params_3c:
                                    gr.Markdown("**3-Compartment Parameters**")
                                    d1_3c = gr.Slider(5e-12, 3e-11, value=1e-11, step=1e-12,
                                                      label="D1 (Intracellular)")
                                    d2_3c = gr.Slider(3e-11, 5e-9, value=5e-10, step=1e-10,
                                                      label="D2 (Extracellular)")
                                    d3_3c = gr.Slider(5e-9, 5e-8, value=1e-8, step=1e-9,
                                                      label="D3 (Fast)")
                                    vf1_3c = gr.Slider(0.1, 0.8, value=0.3, step=0.05,
                                                       label="Volume Fraction 1")
                                    vf3_3c = gr.Slider(0.05, 0.5, value=0.2, step=0.05,
                                                       label="Volume Fraction 3")
                                    rate_01 = gr.Slider(0.1, 30.0, value=5.0, step=0.5,
                                                        label="Exchange Rate 0-1")
                                    rate_02 = gr.Slider(0.1, 30.0, value=3.0, step=0.5,
                                                        label="Exchange Rate 0-2")
                                    rate_12 = gr.Slider(0.1, 30.0, value=4.0, step=0.5,
                                                        label="Exchange Rate 1-2")
                                    tm_3c = gr.Slider(0.015, 0.300, value=0.100, step=0.005,
                                                      label="Mixing Time (s)")
                                    ns_3c = gr.Slider(0.001, 0.050, value=0.010, step=0.001,
                                                      label="Noise Sigma")

                                btn_gen_2c = gr.Button("Generate from Parameters", variant="primary")

                            with gr.Column(scale=1):
                                preview_plot_2c = gr.Plot(label="Signal Preview")
                                preview_info_2c = gr.JSON(label="Parameters")

                    # --- Tab 1B: Random Generation ---
                    with gr.TabItem("Random Generation"):
                        gr.Markdown("Generate a random 2-compartment or 3-compartment sample.")

                        with gr.Row():
                            with gr.Column(scale=1):
                                n_compartments_random = gr.Radio(
                                    choices=[2, 3],
                                    value=2,
                                    label="Number of Compartments",
                                )
                                btn_random = gr.Button("Generate Random Sample", variant="primary")

                            with gr.Column(scale=1):
                                preview_plot_random = gr.Plot(label="Random Sample Preview")
                                preview_info_random = gr.JSON(label="Sample Parameters")

                    # --- Tab 1C: Image Upload ---
                    with gr.TabItem("Upload Signal Image"):
                        gr.Markdown("Upload a pre-existing 64x64 signal matrix (no ground truth available).")

                        with gr.Row():
                            with gr.Column(scale=1):
                                uploaded_file = gr.File(
                                    label="Upload Signal (.npy, .npz, .csv, .txt)",
                                    file_count="single",
                                    file_types=[".npy", ".npz", ".csv", ".txt"],
                                )
                                btn_upload = gr.Button("Load Signal", variant="primary")

                            with gr.Column(scale=1):
                                preview_plot_upload = gr.Plot(label="Uploaded Signal Preview")
                                preview_info_upload = gr.JSON(label="Signal Info")

                # --- Common: Confirm & Continue ---
                gr.Markdown("---")
                with gr.Row():
                    btn_confirm = gr.Button("Confirm Input & Continue to Step 2", variant="primary", scale=2)
                    btn_clear = gr.Button("Clear / Reset", variant="secondary", scale=1)

                current_input_display = gr.JSON(label="Current Input Summary", visible=True)

            # ================================================================
            # TAB 2: MODEL SELECTION & INFERENCE
            # ================================================================
            with gr.TabItem("Step 2: Model Selection & Inference") as tab2:
                gr.Markdown("### Select Model and Run Inference")

                with gr.Row():
                    with gr.Column(scale=1):
                        model_name = gr.Dropdown(
                            choices=available_models(),
                            value="attention_unet",
                            label="Model",
                            info="Choose the reconstruction model",
                        )
                        checkpoint_name = gr.Dropdown(
                            choices=CHECKPOINT_CHOICES,
                            value=default_checkpoint_choice("attention_unet"),
                            label="Checkpoint",
                            info="Model checkpoint file",
                        )
                        device_name = gr.Dropdown(
                            choices=["auto", "cpu", "cuda"],
                            value="auto",
                            label="Device",
                        )
                        btn_run = gr.Button("Run Inference", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        inference_status = gr.HTML(
                            "<p style='color: gray;'>Ready to run inference. "
                            "Confirm input in Step 1 first.</p>"
                        )
                        inference_progress = gr.HTML("")

            # ================================================================
            # TAB 3: RESULTS
            # ================================================================
            with gr.TabItem("Step 3: Results") as tab3:
                gr.Markdown("### Reconstruction Results")

                with gr.Row():
                    with gr.Column(scale=2):
                        result_plot = gr.Plot(label="Comparison Visualization")
                    with gr.Column(scale=1):
                        metrics_html = gr.HTML("<p>No results yet.</p>")
                        gr.Markdown("---")
                        gr.Markdown("**Export Results**")
                        btn_download = gr.DownloadButton(
                            label="Download Result Bundle (.zip)",
                            value=None,
                            interactive=False,
                        )
                        metrics_json = gr.Code(label="Full Metrics (JSON)", language="json", lines=10)

        # ================================================================
        # EVENT HANDLERS
        # ================================================================

        # Toggle 2C vs 3C parameters
        def toggle_compartment_params(n_comp):
            return (
                gr.update(visible=(n_comp == 2)),
                gr.update(visible=(n_comp == 3)),
            )

        n_compartments_2c.change(
            fn=toggle_compartment_params,
            inputs=[n_compartments_2c],
            outputs=[params_2c, params_3c],
        )

        # --- Parametric Generation (2C) ---
        def gen_2c_params(d1, d2, vf, rate, tm, ns):
            result = _generate_parametric(
                n_compartments=2,
                diffusion_1=d1,
                diffusion_2=d2,
                volume_fraction=vf,
                exchange_rate=rate,
                mixing_time=tm,
                noise_sigma=ns,
            )
            fig, summary = _generate_preview_plot(result)
            return fig, result.params, asdict(result)

        # --- Parametric Generation (3C) ---
        def gen_3c_params(d1, d2, d3, vf1, vf3, r01, r02, r12, tm, ns):
            result = _generate_parametric(
                n_compartments=3,
                diffusion_1=d1,
                diffusion_2=d2,
                diffusion_3=d3,
                volume_fraction=vf1,
                volume_fraction_3=vf3,
                exchange_rate_01=r01,
                exchange_rate_02=r02,
                exchange_rate_12=r12,
                mixing_time=tm,
                noise_sigma=ns,
            )
            fig, summary = _generate_preview_plot(result)
            return fig, result.params, asdict(result)

        # --- Unified wrapper for parametric generation ---
        def gen_params_wrapper(
            n_comp,
            d1, d2, vf, rate, tm, ns,
            d1_3c, d2_3c, d3_3c, vf1_3c, vf3_3c, r01, r02, r12, tm_3c, ns_3c,
        ):
            if n_comp == 2:
                return gen_2c_params(d1, d2, vf, rate, tm, ns)
            else:
                return gen_3c_params(d1_3c, d2_3c, d3_3c, vf1_3c, vf3_3c, r01, r02, r12, tm_3c, ns_3c)

        # Override the 2C button to use the wrapper
        btn_gen_2c.click(
            fn=gen_params_wrapper,
            inputs=[
                n_compartments_2c,
                diffusion_1, diffusion_2, volume_fraction, exchange_rate, mixing_time, noise_sigma,
                d1_3c, d2_3c, d3_3c, vf1_3c, vf3_3c, rate_01, rate_02, rate_12, tm_3c, ns_3c,
            ],
            outputs=[preview_plot_2c, preview_info_2c, input_state],
        )

        # --- Random Generation ---
        def gen_random(n_comp):
            result = _generate_random(n_comp)
            fig, summary = _generate_preview_plot(result)
            return fig, result.params, asdict(result)

        btn_random.click(
            fn=gen_random,
            inputs=[n_compartments_random],
            outputs=[preview_plot_random, preview_info_random, input_state],
        )

        # --- Image Upload ---
        def load_uploaded(file_path):
            if file_path is None:
                return None, None, None

            # Handle Gradio file object (dict with 'name' key) or string path
            if isinstance(file_path, dict):
                file_path = file_path.get("name", file_path)

            signal = load_matrix(file_path)

            # Ensure it's a numpy array with .shape
            if hasattr(signal, 'shape'):
                shape_val = list(signal.shape)
            else:
                shape_val = [0]

            result = SignalInputResult(
                signal=signal,
                ground_truth=None,
                params={"shape": shape_val, "source": str(file_path)},
                input_method="image",
                n_compartments=2,  # Default to 2C for uploaded images
            )
            fig, summary = _generate_preview_plot(result)
            info = {
                "shape": signal.shape,
                "min": float(signal.min()),
                "max": float(signal.max()),
                "mean": float(signal.mean()),
                "std": float(signal.std()),
            }
            return fig, info, asdict(result)

        btn_upload.click(
            fn=load_uploaded,
            inputs=[uploaded_file],
            outputs=[preview_plot_upload, preview_info_upload, input_state],
        )

        # --- Confirm Input ---
        def confirm_input(state_dict):
            if state_dict is None:
                return gr.update(visible=True), "<p style='color: red;'>No input confirmed. Please generate or upload a signal first.</p>"
            result = SignalInputResult(**state_dict)
            summary = {
                "Method": result.input_method,
                "Signal Shape": result.signal.shape,
                "Has Ground Truth": result.ground_truth is not None,
                "Compartments": result.params.get("n_compartments", "N/A"),
                "Mixing Time": f"{result.params.get('mixing_time', 'N/A'):.4f}" if isinstance(result.params.get('mixing_time'), float) else str(result.params.get('mixing_time', 'N/A')),
                "Noise Sigma": f"{result.params.get('noise_sigma', 'N/A'):.4f}" if isinstance(result.params.get('noise_sigma'), float) else str(result.params.get('noise_sigma', 'N/A')),
            }
            html = "<table style='width:100%; border-collapse: collapse;'>"
            for k, v in summary.items():
                html += f"<tr><td style='padding: 6px; border: 1px solid #dee2e6; font-weight: bold;'>{k}</td>"
                html += f"<td style='padding: 6px; border: 1px solid #dee2e6;'>{v}</td></tr>"
            html += "</table>"
            return summary, html

        btn_confirm.click(
            fn=confirm_input,
            inputs=[input_state],
            outputs=[current_input_display, inference_status],
        )

        # --- Update Model Dropdown based on n_compartments ---
        def update_model_choices(state_dict):
            if state_dict is None:
                return gr.update(choices=available_models())

            result = SignalInputResult(**state_dict)
            n_comp = result.n_compartments

            if n_comp == 3:
                # 3C mode: only show 3C models
                models_3c = [
                    "attention_unet_3c", "plain_unet_3c", "pinn_3c",
                    "deep_unfolding_3c", "diffusion_refiner", "3d_ilt"
                ]
                return gr.update(choices=models_3c)
            else:
                # 2C mode: only show 2C models
                models_2c = [
                    "attention_unet", "plain_unet", "pinn",
                    "deep_unfolding", "deeponet", "fno", "2d_ilt"
                ]
                return gr.update(choices=models_2c)

        input_state.change(
            fn=update_model_choices,
            inputs=[input_state],
            outputs=[model_name],
        )

        # --- Clear ---
        def clear_all():
            return None, None, None, "<p>Cleared. Ready for new input.</p>", None, None, None, None

        btn_clear.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                input_state, preview_plot_2c, preview_info_2c, inference_status,
                preview_plot_random, preview_info_random,
                preview_plot_upload, preview_info_upload,
            ],
        )

        # --- Update Checkpoint Dropdown ---
        def update_checkpoint(model):
            filtered_choices = _get_model_filtered_checkpoints(model)
            default = default_checkpoint_choice(model) if filtered_choices else None
            return gr.update(choices=filtered_choices if filtered_choices else CHECKPOINT_CHOICES, value=default)

        model_name.change(
            fn=update_checkpoint,
            inputs=[model_name],
            outputs=[checkpoint_name],
        )

        # --- Run Inference ---
        def run_inference_wrapper(input_state_dict, model, checkpoint, device):
            if input_state_dict is None:
                return None, "<p style='color: red;'>No input confirmed. Please complete Step 1 first.</p>", "", None, None

            result = SignalInputResult(**input_state_dict)
            try:
                inference_result = _run_inference(
                    signal=result.signal,
                    model_name=model,
                    checkpoint_name=checkpoint,
                    device_name=device,
                    ground_truth=result.ground_truth,
                )

                # Create comparison figure
                dei_gt = inference_result.metrics.get("dei_true")
                dei_pred = inference_result.metrics.get("dei_pred") or inference_result.metrics.get("model_dei")
                fig = _plot_comparison(
                    inference_result.signal,
                    inference_result.ground_truth,
                    inference_result.prediction,
                    dei_gt=dei_gt,
                    dei_pred=dei_pred,
                )

                # Create metrics HTML
                metrics_html = _create_metrics_html(
                    inference_result.metrics,
                    inference_result.ground_truth is not None,
                )

                # Save bundle
                output_dir = create_run_output_dir(prefix="interface")
                archive_path = _save_bundle(inference_result, output_dir)

                # JSON for code display
                metrics_json = json.dumps(to_serializable(inference_result.metrics), indent=2)

                return (
                    fig,
                    metrics_html,
                    f"<p style='color: green;'>Inference completed in {inference_result.inference_time:.4f}s using {model}</p>",
                    archive_path,
                    metrics_json,
                    asdict(inference_result),
                    str(output_dir),
                )

            except Exception as e:
                import traceback
                error_msg = f"<p style='color: red;'>Error: {str(e)}</p><pre>{traceback.format_exc()}</pre>"
                return None, error_msg, "", None, "", None, None

        btn_run.click(
            fn=run_inference_wrapper,
            inputs=[input_state, model_name, checkpoint_name, device_name],
            outputs=[
                result_plot, metrics_html, inference_progress,
                btn_download, metrics_json,
                result_state, output_dir_state,
            ],
        )

        # Tab switching helpers
        tab1.select(fn=None, inputs=None, outputs=None)
        tab2.select(fn=None, inputs=None, outputs=None)
        tab3.select(fn=None, inputs=None, outputs=None)

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(share=True)

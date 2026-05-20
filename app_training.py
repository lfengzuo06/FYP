#!/usr/bin/env python3
"""
DEXSY training interface focused on reproducibility and fair comparison.

Tabs:
1) Dataset Lab       - create/list/verify/preview/extend immutable datasets
2) Training Runs     - train on selected dataset with fixed seeds
3) Fair Compare      - run model x seed grid on the same dataset
4) Experiment History - inspect run manifests and checkpoints
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dexsy_core.metrics import compute_batch_metrics
from dexsy_datasets import create_dataset, extend_dataset, load_dataset

DATASETS_DIR_DEFAULT = ROOT / "datasets"
RUNS_DIR_DEFAULT = ROOT / "outputs" / "research_runs"
RUNS_HISTORY_JSONL = RUNS_DIR_DEFAULT / "history.jsonl"

DISPLAY_CMAP = "viridis"
DISPLAY_SPECTRUM_CMAP = "magma"


MODEL_SPECS: dict[str, dict[str, str]] = {
    # 2D reconstruction
    "2d_attention_unet": {"label": "2D Attention U-Net", "family": "reconstruction"},
    "2d_plain_unet": {"label": "2D Plain U-Net", "family": "reconstruction"},
    "2d_pinn": {"label": "2D PINN", "family": "reconstruction"},
    "2d_deep_unfolding": {"label": "2D Deep Unfolding", "family": "reconstruction"},
    "2d_fno": {"label": "2D FNO", "family": "reconstruction"},
    "2d_deeponet": {"label": "2D DeepONet", "family": "reconstruction"},
    # 3D/3C reconstruction
    "3d_attention_unet": {"label": "3D Attention U-Net (3C)", "family": "reconstruction"},
    "3d_plain_unet": {"label": "3D Plain U-Net (3C)", "family": "reconstruction"},
    "3d_pinn": {"label": "3D PINN (3C)", "family": "reconstruction"},
    "3d_deep_unfolding": {"label": "3D Deep Unfolding (3C)", "family": "reconstruction"},
    # N-compartment reconstruction
    "nd_attention_unet": {"label": "ND Attention U-Net", "family": "reconstruction"},
    # Non-Gaussian pathway regression
    "nonGaussian_cnn": {"label": "NonGaussian 3C CNN", "family": "pathway_regression"},
}


def _ensure_run_storage() -> None:
    RUNS_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
    if not RUNS_HISTORY_JSONL.exists():
        RUNS_HISTORY_JSONL.touch()


def _now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def _dataset_dir(base_path: str, dataset_id: str) -> Path:
    return Path(base_path) / dataset_id


def _dataset_meta_row(base_path: str, dataset_id: str) -> dict[str, Any]:
    ds_dir = _dataset_dir(base_path, dataset_id)
    config = _read_yaml(ds_dir / "config.yaml")
    metadata = _read_json(ds_dir / "metadata.json")
    splits = _read_json(ds_dir / "splits.json")

    n_comp = metadata.get("n_compartments")
    if n_comp is None:
        model_type = config.get("model_type")
        if model_type == "gaussian_2c":
            n_comp = 2
        elif model_type in ("gaussian_3c", "nongaussian_3c"):
            n_comp = 3
        elif model_type == "gaussian_nc":
            n_comp = int(config.get("params", {}).get("n_compartments", 3))
        else:
            n_comp = "?"

    return {
        "dataset_id": dataset_id,
        "task_type": config.get("task_type", "unknown"),
        "model_type": config.get("model_type", "unknown"),
        "n_b": int(config.get("n_b", -1)),
        "n_compartments": n_comp,
        "n_train": len(splits.get("train", [])),
        "n_val": len(splits.get("val", [])),
        "n_test": len(splits.get("test", [])),
        "seed": config.get("seed", None),
        "created_at": metadata.get("created_at", "unknown"),
    }


def _list_dataset_ids(base_path: str) -> list[str]:
    base = Path(base_path)
    if not base.exists():
        return []
    dataset_ids = []
    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        if (item / "config.yaml").exists() and (item / "metadata.json").exists():
            dataset_ids.append(item.name)
    return dataset_ids


def _build_dataset_table(base_path: str) -> tuple[list[list[Any]], list[str]]:
    dataset_ids = _list_dataset_ids(base_path)
    rows = []
    for ds_id in dataset_ids:
        try:
            meta = _dataset_meta_row(base_path, ds_id)
            rows.append(
                [
                    meta["dataset_id"],
                    meta["task_type"],
                    meta["model_type"],
                    meta["n_b"],
                    meta["n_compartments"],
                    meta["n_train"],
                    meta["n_val"],
                    meta["n_test"],
                    meta["seed"],
                    meta["created_at"],
                ]
            )
        except Exception as exc:
            rows.append([ds_id, "ERROR", str(exc), "", "", "", "", "", "", ""])
    return rows, dataset_ids


def _refresh_dataset_registry(base_path: str):
    rows, dataset_ids = _build_dataset_table(base_path)
    status = f"Loaded {len(dataset_ids)} dataset(s) from `{base_path}`."
    first = dataset_ids[0] if dataset_ids else None

    dd_update = gr.update(choices=dataset_ids, value=first)
    return (
        rows,
        status,
        dd_update,  # lab select
        dd_update,  # extend select
        dd_update,  # training select
        dd_update,  # compare select
    )


def _dataset_detail_json(base_path: str, dataset_id: str) -> dict[str, Any]:
    if not dataset_id:
        return {}
    ds_dir = _dataset_dir(base_path, dataset_id)
    config = _read_yaml(ds_dir / "config.yaml")
    metadata = _read_json(ds_dir / "metadata.json")
    splits = _read_json(ds_dir / "splits.json")
    return {
        "dataset_id": dataset_id,
        "path": str(ds_dir),
        "config": config,
        "metadata": metadata,
        "split_sizes": {k: len(v) for k, v in splits.items()},
    }


def _verify_dataset(base_path: str, dataset_id: str) -> tuple[str, dict[str, Any]]:
    if not dataset_id:
        return "Please select a dataset first.", {}
    try:
        ds = load_dataset(dataset_id, base_path=base_path, verify=True)
        ds.validate()
        return f"Verification passed for `{dataset_id}`.", _dataset_detail_json(base_path, dataset_id)
    except Exception as exc:
        return f"Verification failed for `{dataset_id}`: {exc}", {}


def _plot_dataset_preview(dataset, split_name: str, sample_index: int) -> tuple[plt.Figure, dict[str, Any]]:
    split = dataset.get_split(split_name)
    n = len(split["signals"])
    if n == 0:
        raise ValueError(f"Split `{split_name}` is empty.")

    idx = int(np.clip(sample_index, 0, n - 1))
    sig = np.asarray(split["signals"][idx], dtype=np.float32)
    if sig.ndim == 3:
        sig = sig[0]

    fig_cols = 2
    has_spectrum = "spectra" in split
    has_pathway = "pathway_weights" in split
    if has_spectrum or has_pathway:
        fig_cols = 3

    fig, axes = plt.subplots(1, fig_cols, figsize=(4.5 * fig_cols, 3.8))
    if fig_cols == 1:
        axes = [axes]

    ax0 = axes[0]
    im0 = ax0.imshow(sig, cmap=DISPLAY_CMAP, origin="lower")
    ax0.set_title(f"{split_name} sample #{idx} signal")
    ax0.set_xlabel("b2 index")
    ax0.set_ylabel("b1 index")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = axes[1]
    log_sig = np.log(np.clip(sig, 1e-8, None))
    im1 = ax1.imshow(log_sig, cmap=DISPLAY_CMAP, origin="lower")
    ax1.set_title("log(signal)")
    ax1.set_xlabel("b2 index")
    ax1.set_ylabel("b1 index")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    sample_info = {
        "split": split_name,
        "index": idx,
        "signal_shape": list(sig.shape),
        "signal_min": float(sig.min()),
        "signal_max": float(sig.max()),
        "signal_mean": float(sig.mean()),
    }

    if has_spectrum:
        spec = np.asarray(split["spectra"][idx], dtype=np.float32)
        if spec.ndim == 3:
            spec = spec[0]
        ax2 = axes[2]
        vmax = float(np.max(spec)) if np.max(spec) > 0 else 1.0
        im2 = ax2.imshow(spec, cmap=DISPLAY_SPECTRUM_CMAP, origin="lower", vmin=0.0, vmax=vmax)
        ax2.set_title("ground truth spectrum")
        ax2.set_xlabel("D2 index")
        ax2.set_ylabel("D1 index")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        sample_info["label_type"] = "spectrum"
        sample_info["label_shape"] = list(spec.shape)
    elif has_pathway:
        pathway = np.asarray(split["pathway_weights"][idx], dtype=np.float32)
        if pathway.ndim == 1 and pathway.size == 9:
            pathway = pathway.reshape(3, 3)
        ax2 = axes[2]
        im2 = ax2.imshow(pathway, cmap=DISPLAY_SPECTRUM_CMAP, origin="lower")
        ax2.set_title("pathway weights (3x3)")
        ax2.set_xlabel("to")
        ax2.set_ylabel("from")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        sample_info["label_type"] = "pathway_weights"
        sample_info["label_shape"] = list(pathway.shape)
        if "dei" in split:
            sample_info["dei"] = float(np.asarray(split["dei"])[idx])

    fig.tight_layout()
    return fig, sample_info


def _preview_dataset(base_path: str, dataset_id: str, split_name: str, sample_index: int):
    if not dataset_id:
        return None, {"error": "Please select a dataset."}
    try:
        ds = load_dataset(dataset_id, base_path=base_path, verify=False)
        fig, info = _plot_dataset_preview(ds, split_name, int(sample_index))
        info["dataset_id"] = dataset_id
        return fig, info
    except Exception as exc:
        return None, {"error": str(exc)}


def _export_dataset_config(base_path: str, dataset_id: str):
    if not dataset_id:
        return "Please select a dataset first.", None
    config_path = _dataset_dir(base_path, dataset_id) / "config.yaml"
    if not config_path.exists():
        return f"Config not found for `{dataset_id}`.", None
    return f"Config ready: `{config_path}`", str(config_path)


def _default_params_for_model(model_type: str, n_compartments_nc: int) -> tuple[str, dict[str, Any]]:
    if model_type == "nongaussian_3c":
        task_type = "pathway_regression"
        params = {
            "extracellular_diffusivity_range": [1.0e-9, 2.5e-9],
            "intracellular_diffusivity_range": [0.4e-9, 1.2e-9],
            "axon_restricted_length_range": [0.5e-6, 2.0e-6],
            "sphere_radius_range": [1.0e-6, 6.0e-6],
            "mixing_time_range": [0.015, 0.300],
            "extracellular_fraction_range": [0.3, 0.7],
            "axon_fraction_range": [0.1, 0.4],
            "sphere_fraction_range": [0.1, 0.4],
            "noise_sigma": None,
        }
        return task_type, params

    task_type = "reconstruction"
    params = {
        "d_min": 5e-12,
        "d_max": 5e-9,
        "g_max": 0.3,
        "delta": 0.010,
        "DELTA": 0.040,
        "noise_sigma_range": [0.005, 0.015],
    }
    if model_type == "gaussian_nc":
        params["n_compartments"] = max(2, int(n_compartments_nc))
    return task_type, params


def _create_dataset_from_form(
    base_path: str,
    model_type: str,
    n_b: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    n_compartments_nc: int,
):
    try:
        task_type, params = _default_params_for_model(model_type, n_compartments_nc)
        config = {
            "version": "1.0",
            "protocol_version": "1",
            "generator_version": "1.0.0",
            "task_type": task_type,
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "n_b": int(n_b),
            "model_type": model_type,
            "params": params,
            "sampling_strategy": "log_uniform",
            "min_index_separation": 0,
            "seed": int(seed),
        }
        ds = create_dataset(config=config, output_dir=base_path, generator_version="1.0.0")
        status = f"Created dataset `{ds.dataset_id}` ({model_type}, task={task_type})."
    except Exception as exc:
        status = f"Failed to create dataset: {exc}"
    rows, refresh_status, lab_dd, ext_dd, train_dd, compare_dd = _refresh_dataset_registry(base_path)
    return rows, f"{refresh_status}\n{status}", lab_dd, ext_dd, train_dd, compare_dd


def _extend_dataset_from_form(base_path: str, dataset_id: str, add_train: int):
    if not dataset_id:
        rows, refresh_status, lab_dd, ext_dd, train_dd, compare_dd = _refresh_dataset_registry(base_path)
        return rows, f"{refresh_status}\nPlease select a base dataset.", lab_dd, ext_dd, train_dd, compare_dd
    try:
        ds = extend_dataset(
            base_dataset_id=dataset_id,
            n_add_train=int(add_train),
            output_dir=base_path,
            generator_version="1.0.0",
        )
        status = (
            f"Extended `{dataset_id}` by +{int(add_train)} train samples.\n"
            f"New dataset: `{ds.dataset_id}` (train={ds.n_train}, val={ds.n_val}, test={ds.n_test})."
        )
    except Exception as exc:
        status = f"Failed to extend dataset: {exc}"

    rows, refresh_status, lab_dd, ext_dd, train_dd, compare_dd = _refresh_dataset_registry(base_path)
    return rows, f"{refresh_status}\n{status}", lab_dd, ext_dd, train_dd, compare_dd


def _infer_dataset_n_compartments(dataset) -> int:
    meta_n = dataset.metadata.get("n_compartments")
    if meta_n is not None:
        return int(meta_n)
    model_type = dataset.config.get("model_type", "")
    if model_type == "gaussian_2c":
        return 2
    if model_type in ("gaussian_3c", "nongaussian_3c"):
        return 3
    if model_type == "gaussian_nc":
        return int(dataset.config.get("params", {}).get("n_compartments", 3))
    return 2


def _models_for_dataset(dataset) -> list[str]:
    task_type = dataset.config.get("task_type")
    model_type = dataset.config.get("model_type")
    n_comp = _infer_dataset_n_compartments(dataset)

    if task_type == "pathway_regression":
        return ["nonGaussian_cnn"]

    if model_type == "gaussian_2c":
        return [
            "2d_attention_unet",
            "2d_plain_unet",
            "2d_pinn",
            "2d_deep_unfolding",
            "2d_fno",
            "2d_deeponet",
        ]
    if model_type == "gaussian_3c":
        return [
            "3d_attention_unet",
            "3d_plain_unet",
            "3d_pinn",
            "3d_deep_unfolding",
        ]
    if model_type == "gaussian_nc":
        return ["nd_attention_unet"]

    # Fallback by inferred compartment count.
    if n_comp == 2:
        return ["2d_attention_unet", "2d_plain_unet", "2d_pinn", "2d_deep_unfolding", "2d_fno", "2d_deeponet"]
    if n_comp == 3:
        return ["3d_attention_unet", "3d_plain_unet", "3d_pinn", "3d_deep_unfolding"]
    return ["nd_attention_unet"]


def _model_choices_from_keys(model_keys: list[str]) -> list[tuple[str, str]]:
    return [(MODEL_SPECS.get(k, {}).get("label", k), k) for k in model_keys]


def _on_training_dataset_change(base_path: str, dataset_id: str):
    if not dataset_id:
        return gr.update(choices=[], value=None), {}
    try:
        ds = load_dataset(dataset_id, base_path=base_path, verify=False)
        model_keys = _models_for_dataset(ds)
        detail = _dataset_detail_json(base_path, dataset_id)
        return gr.update(choices=_model_choices_from_keys(model_keys), value=model_keys[0]), detail
    except Exception as exc:
        return gr.update(choices=[], value=None), {"error": str(exc)}


def _on_compare_dataset_change(base_path: str, dataset_id: str):
    if not dataset_id:
        return gr.update(choices=[], value=[]), {}
    try:
        ds = load_dataset(dataset_id, base_path=base_path, verify=False)
        model_keys = _models_for_dataset(ds)
        detail = _dataset_detail_json(base_path, dataset_id)
        return gr.update(choices=_model_choices_from_keys(model_keys), value=model_keys[:1]), detail
    except Exception as exc:
        return gr.update(choices=[], value=[]), {"error": str(exc)}


def _run_id(prefix: str = "run") -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rand = hex(int(time.time() * 1e6))[-6:]
    return f"{prefix}_{ts}_{rand}"


def _append_history(entry: dict[str, Any]) -> None:
    _ensure_run_storage()
    with open(RUNS_HISTORY_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _extract_model_output_tensor(model_out: Any) -> torch.Tensor:
    if torch.is_tensor(model_out):
        return model_out
    if hasattr(model_out, "pathway_weights"):
        return model_out.pathway_weights
    raise TypeError(f"Unsupported model output type: {type(model_out)}")


def _evaluate_reconstruction_model(model, datasets: dict[str, Any], batch_size: int = 64) -> dict[str, float]:
    test_inputs = np.asarray(datasets["test"]["inputs"], dtype=np.float32)
    test_labels = np.asarray(datasets["test"]["labels"], dtype=np.float32)
    device = next(model.parameters()).device

    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_inputs), batch_size):
            x = torch.from_numpy(test_inputs[i:i + batch_size]).to(device)
            out = _extract_model_output_tensor(model(x))
            preds.append(out.detach().cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    metrics = compute_batch_metrics(test_labels, y_pred)["aggregate"]
    return {k: float(v) for k, v in metrics.items()}


def _dei_from_pathway_vector(weights: np.ndarray) -> np.ndarray:
    # Weights shape: (N, 9) in EE ET ES / TE TT TS / SE ST SS order.
    diag = weights[:, [0, 4, 8]].sum(axis=1)
    off = weights.sum(axis=1) - diag
    return off / (diag + 1e-10)


def _evaluate_nongaussian_model(model, datasets: dict[str, Any], batch_size: int = 128) -> dict[str, float]:
    test_split = datasets["test"]
    signals = np.asarray(test_split["signals_noisy"], dtype=np.float32)
    targets = np.asarray(test_split["pathway_weights"], dtype=np.float32)
    dei_true = np.asarray(test_split["dei"], dtype=np.float32)

    device = next(model.parameters()).device
    model.eval()
    preds = []
    dei_preds = []
    with torch.no_grad():
        for i in range(0, len(signals), batch_size):
            x = torch.from_numpy(signals[i:i + batch_size]).to(device)
            out = model(x)
            weights = np.asarray(out.pathway_weights.detach().cpu().numpy(), dtype=np.float32)
            preds.append(weights)
            dei_preds.append(np.asarray(out.dei.detach().cpu().numpy(), dtype=np.float32))

    pred_weights = np.concatenate(preds, axis=0)
    pred_dei = np.concatenate(dei_preds, axis=0)
    if pred_dei.ndim == 0:
        pred_dei = np.full((pred_weights.shape[0],), float(pred_dei))

    pathway_mse = float(np.mean((pred_weights - targets) ** 2))
    dei_error = float(np.mean(np.abs(pred_dei - dei_true)))
    # Robust fallback if model output DEI shape differs.
    if not np.isfinite(dei_error):
        dei_error = float(np.mean(np.abs(_dei_from_pathway_vector(pred_weights) - dei_true)))

    return {
        "pathway_mse": pathway_mse,
        "dei_error": dei_error,
    }


def _plot_training_curves(history: dict[str, list[float]]) -> plt.Figure:
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if train_loss:
        axes[0].plot(epochs, train_loss, label="train", linewidth=2)
        axes[1].plot(epochs, train_loss, label="train", linewidth=2)
    if val_loss:
        axes[0].plot(epochs, val_loss, label="val", linewidth=2)
        axes[1].plot(epochs, val_loss, label="val", linewidth=2)

    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Loss Curves (log scale)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    return fig


def _train_single_run(
    *,
    dataset_id: str,
    base_path: str,
    model_key: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    seed: int,
    init_seed: int,
    dataloader_seed: int,
    base_filters: int,
    n_layers: int,
    hidden_dim: int,
    use_denoiser: bool,
    fno_hidden_channels: int,
    fno_n_layers: int,
    fno_modes: int,
    lambda_dei: float,
    n_restrict_terms: int,
):
    ds = load_dataset(dataset_id, base_path=base_path, verify=False)
    n_comp = _infer_dataset_n_compartments(ds)
    n_b = int(ds.config.get("n_b", 16))

    run_id = _run_id("train")
    run_dir = RUNS_DIR_DEFAULT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    common = {
        "output_dir": str(run_dir),
        "dataset_id": dataset_id,
        "datasets_dir": base_path,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "early_stopping_patience": int(early_stopping_patience),
        "seed": int(seed),
        "init_seed": int(init_seed),
        "dataloader_seed": int(dataloader_seed),
        "n_d": int(n_b),
        "n_b": int(n_b),
    }

    if model_key == "2d_attention_unet":
        from models_2d.attention_unet.train import train_model as trainer

        model, history, datasets, _fm = trainer(
            **common,
            n_compartments=int(n_comp),
            base_filters=int(base_filters),
        )
        checkpoint_path = run_dir / "best_model.pt"
    elif model_key == "2d_plain_unet":
        from models_2d.plain_unet.train import train_model as trainer

        model, history, datasets, _fm = trainer(
            **common,
            n_compartments=int(n_comp),
            base_filters=int(base_filters),
        )
        checkpoint_path = run_dir / "best_model.pt"
    elif model_key == "2d_pinn":
        from models_2d.pinn.train import train_model as trainer

        model, history, datasets, _fm = trainer(
            **common,
            n_compartments=int(n_comp),
        )
        checkpoint_path = run_dir / "best_model.pt"
    elif model_key == "2d_deep_unfolding":
        from models_2d.deep_unfolding.train import train_model as trainer

        model, history, datasets, _fm = trainer(
            **common,
            n_compartments=int(n_comp),
            n_layers=int(n_layers),
            hidden_dim=int(hidden_dim),
            use_denoiser=bool(use_denoiser),
        )
        checkpoint_path = run_dir / "best_model.pt"
    elif model_key in ("2d_fno", "2d_deeponet"):
        from models_2d.neural_operators.train import train_model as trainer

        model_type = "fno" if model_key == "2d_fno" else "deeponet"
        model, history, datasets, _fm = trainer(
            **common,
            model_type=model_type,
            n_compartments=int(n_comp),
            fno_hidden_channels=int(fno_hidden_channels),
            fno_n_layers=int(fno_n_layers),
            fno_modes=int(fno_modes),
        )
        checkpoint_path = run_dir / "best_model.pt"
    elif model_key == "3d_attention_unet":
        from models_3d.attention_unet.train import train_model as trainer

        model, history, datasets, _fm = trainer(
            **common,
            n_compartments=3,
            base_filters=int(base_filters),
        )
        checkpoint_path = run_dir / "best_model.pt"
    elif model_key == "3d_plain_unet":
        from models_3d.plain_unet.train import train_model as trainer

        model, history, datasets, _fm = trainer(
            **common,
            n_compartments=3,
            base_filters=int(base_filters),
        )
        checkpoint_path = run_dir / "best_model.pt"
    elif model_key == "3d_pinn":
        from models_3d.pinn.train import train_model as trainer

        model, history, datasets, _fm = trainer(
            **common,
            n_compartments=3,
        )
        checkpoint_path = run_dir / "best_model.pt"
    elif model_key == "3d_deep_unfolding":
        from models_3d.deep_unfolding.train import train_model as trainer

        model, history, datasets, _fm = trainer(
            **common,
            n_compartments=3,
            n_layers=int(n_layers),
            hidden_dim=int(hidden_dim),
            use_denoiser=bool(use_denoiser),
        )
        checkpoint_path = run_dir / "best_model.pt"
    elif model_key == "nd_attention_unet":
        from models_nd.attention_unet.train import train_model as trainer

        model, history, datasets, _fm = trainer(
            **common,
            n_compartments=int(n_comp),
            base_filters=int(base_filters),
            grid_size=int(n_b),
        )
        checkpoint_path = run_dir / "best_model.pt"
    elif model_key == "nonGaussian_cnn":
        from models_nonGaussian.cnn.train import train_nongaussian_inverse_model as trainer

        model, history, datasets, _fm, actual_run_dir = trainer(
            output_dir=str(run_dir),
            n_train=int(ds.n_train),
            n_val=int(ds.n_val),
            n_test=int(ds.n_test),
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            early_stopping_patience=int(early_stopping_patience),
            lambda_dei=float(lambda_dei),
            hidden_dim=int(hidden_dim),
            n_b=int(n_b),
            n_restrict_terms=int(n_restrict_terms),
            seed=int(seed),
            init_seed=int(init_seed),
            dataloader_seed=int(dataloader_seed),
            dataset_id=dataset_id,
            datasets_dir=base_path,
        )
        checkpoint_path = Path(actual_run_dir) / "best_model.pt"
    else:
        raise ValueError(f"Unsupported model key: {model_key}")

    if ds.task_type == "pathway_regression":
        test_metrics = _evaluate_nongaussian_model(model, datasets)
    else:
        test_metrics = _evaluate_reconstruction_model(model, datasets)

    duration_seconds = float(time.time() - started)
    best_val_loss = float(min(history.get("val_loss", [float("nan")])))
    history_plot = _plot_training_curves(history)

    manifest = {
        "run_id": run_id,
        "timestamp": _now_utc_iso(),
        "status": "completed",
        "dataset_id": dataset_id,
        "dataset_task_type": ds.task_type,
        "dataset_model_type": ds.config.get("model_type"),
        "model_key": model_key,
        "model_label": MODEL_SPECS.get(model_key, {}).get("label", model_key),
        "output_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "duration_seconds": duration_seconds,
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
        "config": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "early_stopping_patience": int(early_stopping_patience),
            "seed": int(seed),
            "init_seed": int(init_seed),
            "dataloader_seed": int(dataloader_seed),
            "base_filters": int(base_filters),
            "n_layers": int(n_layers),
            "hidden_dim": int(hidden_dim),
            "use_denoiser": bool(use_denoiser),
            "fno_hidden_channels": int(fno_hidden_channels),
            "fno_n_layers": int(fno_n_layers),
            "fno_modes": int(fno_modes),
            "lambda_dei": float(lambda_dei),
            "n_restrict_terms": int(n_restrict_terms),
        },
    }

    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _append_history(manifest)
    return manifest, history_plot


def _run_training_ui(
    base_path: str,
    dataset_id: str,
    model_key: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    seed: int,
    init_seed: int,
    dataloader_seed: int,
    base_filters: int,
    n_layers: int,
    hidden_dim: int,
    use_denoiser: bool,
    fno_hidden_channels: int,
    fno_n_layers: int,
    fno_modes: int,
    lambda_dei: float,
    n_restrict_terms: int,
):
    if not dataset_id:
        return "Please select a dataset first.", None, {}, {}
    if not model_key:
        return "Please select a model first.", None, {}, {}
    try:
        manifest, history_plot = _train_single_run(
            dataset_id=dataset_id,
            base_path=base_path,
            model_key=model_key,
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            early_stopping_patience=int(early_stopping_patience),
            seed=int(seed),
            init_seed=int(init_seed),
            dataloader_seed=int(dataloader_seed),
            base_filters=int(base_filters),
            n_layers=int(n_layers),
            hidden_dim=int(hidden_dim),
            use_denoiser=bool(use_denoiser),
            fno_hidden_channels=int(fno_hidden_channels),
            fno_n_layers=int(fno_n_layers),
            fno_modes=int(fno_modes),
            lambda_dei=float(lambda_dei),
            n_restrict_terms=int(n_restrict_terms),
        )
        status = (
            f"Training completed: `{manifest['run_id']}`\n"
            f"checkpoint: `{manifest['checkpoint_path']}`\n"
            f"duration: {manifest['duration_seconds']:.1f}s"
        )
        return status, history_plot, manifest.get("test_metrics", {}), manifest
    except Exception as exc:
        return f"Training failed: {exc}", None, {}, {}


def _parse_seed_list(seed_text: str) -> list[int]:
    if not seed_text.strip():
        return [42]
    seeds = []
    for part in seed_text.replace(" ", "").split(","):
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        seeds = [42]
    return seeds


def _plot_compare_summary(summary_rows: list[dict[str, Any]], metric_key: str) -> plt.Figure | None:
    if not summary_rows:
        return None
    names = [r["model_key"] for r in summary_rows]
    means = [float(r["metric_mean"]) for r in summary_rows]
    stds = [float(r["metric_std"]) for r in summary_rows]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 4.8))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel(metric_key)
    ax.set_title(f"Fair Compare Summary ({metric_key}, mean ± std)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _run_fair_compare_ui(
    base_path: str,
    dataset_id: str,
    model_keys: list[str],
    seeds_text: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    base_filters: int,
    n_layers: int,
    hidden_dim: int,
    use_denoiser: bool,
    fno_hidden_channels: int,
    fno_n_layers: int,
    fno_modes: int,
    lambda_dei: float,
    n_restrict_terms: int,
):
    if not dataset_id:
        return "Please select a dataset first.", [], [], None
    if not model_keys:
        return "Please select at least one model.", [], [], None

    try:
        ds = load_dataset(dataset_id, base_path=base_path, verify=False)
        seeds = _parse_seed_list(seeds_text)
        metric_name = "pathway_mse" if ds.task_type == "pathway_regression" else "mse_mean"

        run_rows = []
        for model_key in model_keys:
            for seed in seeds:
                manifest, _plot = _train_single_run(
                    dataset_id=dataset_id,
                    base_path=base_path,
                    model_key=model_key,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    learning_rate=float(learning_rate),
                    weight_decay=float(weight_decay),
                    early_stopping_patience=int(early_stopping_patience),
                    seed=int(seed),
                    init_seed=int(seed),
                    dataloader_seed=int(seed),
                    base_filters=int(base_filters),
                    n_layers=int(n_layers),
                    hidden_dim=int(hidden_dim),
                    use_denoiser=bool(use_denoiser),
                    fno_hidden_channels=int(fno_hidden_channels),
                    fno_n_layers=int(fno_n_layers),
                    fno_modes=int(fno_modes),
                    lambda_dei=float(lambda_dei),
                    n_restrict_terms=int(n_restrict_terms),
                )
                test_metrics = manifest.get("test_metrics", {})
                run_rows.append(
                    {
                        "run_id": manifest["run_id"],
                        "dataset_id": dataset_id,
                        "model_key": model_key,
                        "seed": seed,
                        "best_val_loss": manifest.get("best_val_loss"),
                        "metric_name": metric_name,
                        "metric_value": float(test_metrics.get(metric_name, np.nan)),
                        "duration_s": float(manifest.get("duration_seconds", 0.0)),
                    }
                )

        summary_rows = []
        by_model: dict[str, list[float]] = {}
        for row in run_rows:
            by_model.setdefault(row["model_key"], []).append(float(row["metric_value"]))
        for model_key, vals in by_model.items():
            arr = np.asarray(vals, dtype=np.float64)
            summary_rows.append(
                {
                    "model_key": model_key,
                    "metric_name": metric_name,
                    "metric_mean": float(np.nanmean(arr)),
                    "metric_std": float(np.nanstd(arr)),
                    "n_runs": int(arr.size),
                }
            )
        summary_rows.sort(key=lambda r: r["metric_mean"])
        fig = _plot_compare_summary(summary_rows, metric_name)

        summary_table = [
            [
                r["model_key"],
                r["metric_name"],
                r["metric_mean"],
                r["metric_std"],
                r["n_runs"],
            ]
            for r in summary_rows
        ]
        run_table = [
            [
                r["run_id"],
                r["dataset_id"],
                r["model_key"],
                r["seed"],
                r["best_val_loss"],
                r["metric_name"],
                r["metric_value"],
                r["duration_s"],
            ]
            for r in run_rows
        ]

        status = (
            f"Fair compare completed on dataset `{dataset_id}`. "
            f"Runs: {len(run_rows)} ({len(model_keys)} model(s) x {len(seeds)} seed(s))."
        )
        return status, summary_table, run_table, fig
    except Exception as exc:
        return f"Fair compare failed: {exc}", [], [], None


def _load_history_rows() -> list[dict[str, Any]]:
    _ensure_run_storage()
    rows = []
    with open(RUNS_HISTORY_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return rows


def _refresh_history_ui():
    rows = _load_history_rows()
    table_rows = []
    run_ids = []
    for r in rows:
        metric_preview = ""
        tm = r.get("test_metrics", {})
        if "mse_mean" in tm:
            metric_preview = f"mse={tm['mse_mean']:.4g}"
        elif "pathway_mse" in tm:
            metric_preview = f"pathway_mse={tm['pathway_mse']:.4g}"
        table_rows.append(
            [
                r.get("run_id"),
                r.get("timestamp"),
                r.get("dataset_id"),
                r.get("model_key"),
                r.get("status"),
                metric_preview,
                r.get("duration_seconds"),
                r.get("checkpoint_path"),
            ]
        )
        rid = r.get("run_id")
        if rid:
            run_ids.append(rid)
    first = run_ids[0] if run_ids else None
    status = f"Loaded {len(rows)} run record(s)."
    return table_rows, status, gr.update(choices=run_ids, value=first)


def _history_detail(run_id: str) -> dict[str, Any]:
    if not run_id:
        return {}
    rows = _load_history_rows()
    for r in rows:
        if r.get("run_id") == run_id:
            return r
    return {"error": f"Run `{run_id}` not found."}


def build_app():
    _ensure_run_storage()
    DATASETS_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)

    dataset_headers = [
        "dataset_id",
        "task_type",
        "model_type",
        "n_b",
        "n_compartments",
        "n_train",
        "n_val",
        "n_test",
        "seed",
        "created_at",
    ]
    history_headers = [
        "run_id",
        "timestamp",
        "dataset_id",
        "model_key",
        "status",
        "metric",
        "duration_s",
        "checkpoint_path",
    ]
    compare_summary_headers = ["model_key", "metric_name", "metric_mean", "metric_std", "n_runs"]
    compare_run_headers = [
        "run_id",
        "dataset_id",
        "model_key",
        "seed",
        "best_val_loss",
        "metric_name",
        "metric_value",
        "duration_s",
    ]

    with gr.Blocks(title="DEXSY Research Interface", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # DEXSY Reproducible Research Interface

            This interface is built for:
            - immutable dataset management
            - reproducible training with explicit seeds
            - fair model comparison on the same dataset
            - persistent experiment history
            """
        )

        with gr.Row():
            datasets_base_path = gr.Textbox(
                label="Datasets Base Path",
                value=str(DATASETS_DIR_DEFAULT),
                scale=3,
            )
            refresh_datasets_btn = gr.Button("Refresh Datasets", variant="secondary", scale=1)

        with gr.Tabs():
            # ==========================================================
            # TAB 1: DATASET LAB
            # ==========================================================
            with gr.TabItem("Dataset Lab"):
                dataset_registry_status = gr.Textbox(label="Registry Status", lines=2)
                dataset_table = gr.Dataframe(
                    headers=dataset_headers,
                    datatype=["str"] * len(dataset_headers),
                    label="Dataset Registry",
                    interactive=False,
                    wrap=True,
                )

                with gr.Row():
                    dataset_select_lab = gr.Dropdown(label="Select Dataset", choices=[])
                    verify_btn = gr.Button("Verify Dataset", variant="secondary")
                verify_status = gr.Textbox(label="Verification Status", lines=2)
                dataset_detail = gr.JSON(label="Dataset Details")
                with gr.Row():
                    export_config_btn = gr.Button("Export Config YAML", variant="secondary")
                    exported_config_file = gr.File(label="Dataset Config File")
                export_config_status = gr.Textbox(label="Config Export Status", lines=1)

                with gr.Row():
                    preview_split = gr.Dropdown(choices=["train", "val", "test"], value="train", label="Preview Split")
                    preview_index = gr.Number(value=0, precision=0, label="Sample Index")
                    preview_btn = gr.Button("Preview Sample")
                preview_plot = gr.Plot(label="Dataset Sample Preview")
                preview_info = gr.JSON(label="Sample Info")

                gr.Markdown("---")
                gr.Markdown("### Create Dataset")
                with gr.Row():
                    create_model_type = gr.Dropdown(
                        label="Model Type",
                        choices=["gaussian_2c", "gaussian_3c", "gaussian_nc", "nongaussian_3c"],
                        value="gaussian_2c",
                    )
                    create_n_b = gr.Radio(choices=[16, 64], value=16, label="Grid Size (n_b)")
                    create_seed = gr.Number(value=42, precision=0, label="Seed")
                    create_n_comp_nc = gr.Number(value=4, precision=0, label="n_comp (for gaussian_nc)")
                with gr.Row():
                    create_n_train = gr.Number(value=9500, precision=0, label="n_train")
                    create_n_val = gr.Number(value=400, precision=0, label="n_val")
                    create_n_test = gr.Number(value=400, precision=0, label="n_test")
                    create_btn = gr.Button("Create Dataset", variant="primary")
                create_status = gr.Textbox(label="Create Status", lines=3)

                gr.Markdown("---")
                gr.Markdown("### Extend Dataset (add train samples)")
                with gr.Row():
                    dataset_select_extend = gr.Dropdown(label="Base Dataset", choices=[])
                    extend_add_train = gr.Number(value=1000, precision=0, label="Add Train Samples")
                    extend_btn = gr.Button("Extend Dataset", variant="secondary")
                extend_status = gr.Textbox(label="Extend Status", lines=3)

            # ==========================================================
            # TAB 2: TRAINING RUNS
            # ==========================================================
            with gr.TabItem("Training Runs"):
                gr.Markdown("### Reproducible training (must select dataset_id)")
                with gr.Row():
                    train_dataset_id = gr.Dropdown(label="Dataset ID", choices=[])
                    train_model_key = gr.Dropdown(label="Model", choices=[])
                train_dataset_info = gr.JSON(label="Selected Dataset Snapshot")

                with gr.Row():
                    train_epochs = gr.Number(value=60, precision=0, label="Epochs")
                    train_batch_size = gr.Number(value=8, precision=0, label="Batch Size")
                    train_learning_rate = gr.Number(value=5e-4, label="Learning Rate")
                    train_weight_decay = gr.Number(value=1e-4, label="Weight Decay")
                    train_early_stop = gr.Number(value=12, precision=0, label="Early Stop Patience")

                with gr.Row():
                    train_seed = gr.Number(value=42, precision=0, label="Data Seed (logged)")
                    train_init_seed = gr.Number(value=42, precision=0, label="Init Seed")
                    train_dataloader_seed = gr.Number(value=42, precision=0, label="Dataloader Seed")

                with gr.Accordion("Advanced Model Args", open=False):
                    with gr.Row():
                        train_base_filters = gr.Number(value=32, precision=0, label="base_filters")
                        train_n_layers = gr.Number(value=12, precision=0, label="n_layers")
                        train_hidden_dim = gr.Number(value=256, precision=0, label="hidden_dim")
                        train_use_denoiser = gr.Checkbox(value=True, label="use_denoiser")
                    with gr.Row():
                        train_fno_hidden_channels = gr.Number(value=64, precision=0, label="fno_hidden_channels")
                        train_fno_n_layers = gr.Number(value=4, precision=0, label="fno_n_layers")
                        train_fno_modes = gr.Number(value=16, precision=0, label="fno_modes")
                    with gr.Row():
                        train_lambda_dei = gr.Number(value=1.0, label="lambda_dei (nonGaussian)")
                        train_n_restrict_terms = gr.Number(value=500, precision=0, label="n_restrict_terms (nonGaussian)")

                run_training_btn = gr.Button("Run Training", variant="primary")
                train_status = gr.Textbox(label="Training Status", lines=3)
                training_curve_plot = gr.Plot(label="Training Curves")
                train_test_metrics = gr.JSON(label="Test Metrics")
                train_manifest = gr.JSON(label="Run Manifest")

            # ==========================================================
            # TAB 3: FAIR COMPARE
            # ==========================================================
            with gr.TabItem("Fair Compare"):
                gr.Markdown("### Compare models on the SAME dataset_id")
                with gr.Row():
                    compare_dataset_id = gr.Dropdown(label="Dataset ID", choices=[])
                    compare_models = gr.CheckboxGroup(label="Models", choices=[])
                compare_dataset_info = gr.JSON(label="Selected Dataset Snapshot")

                with gr.Row():
                    compare_seeds = gr.Textbox(value="42,43,44", label="Seeds (comma-separated)")
                    compare_epochs = gr.Number(value=40, precision=0, label="Epochs")
                    compare_batch_size = gr.Number(value=8, precision=0, label="Batch Size")
                    compare_learning_rate = gr.Number(value=5e-4, label="Learning Rate")
                with gr.Row():
                    compare_weight_decay = gr.Number(value=1e-4, label="Weight Decay")
                    compare_early_stop = gr.Number(value=10, precision=0, label="Early Stop Patience")

                with gr.Accordion("Advanced Model Args", open=False):
                    with gr.Row():
                        compare_base_filters = gr.Number(value=32, precision=0, label="base_filters")
                        compare_n_layers = gr.Number(value=12, precision=0, label="n_layers")
                        compare_hidden_dim = gr.Number(value=256, precision=0, label="hidden_dim")
                        compare_use_denoiser = gr.Checkbox(value=True, label="use_denoiser")
                    with gr.Row():
                        compare_fno_hidden_channels = gr.Number(value=64, precision=0, label="fno_hidden_channels")
                        compare_fno_n_layers = gr.Number(value=4, precision=0, label="fno_n_layers")
                        compare_fno_modes = gr.Number(value=16, precision=0, label="fno_modes")
                    with gr.Row():
                        compare_lambda_dei = gr.Number(value=1.0, label="lambda_dei (nonGaussian)")
                        compare_n_restrict_terms = gr.Number(value=500, precision=0, label="n_restrict_terms (nonGaussian)")

                run_compare_btn = gr.Button("Run Fair Compare", variant="primary")
                compare_status = gr.Textbox(label="Compare Status", lines=3)
                compare_summary = gr.Dataframe(
                    headers=compare_summary_headers,
                    datatype=["str", "str", "number", "number", "number"],
                    label="Summary (mean ± std)",
                    interactive=False,
                    wrap=True,
                )
                compare_runs = gr.Dataframe(
                    headers=compare_run_headers,
                    datatype=["str", "str", "str", "number", "number", "str", "number", "number"],
                    label="Per-Run Results",
                    interactive=False,
                    wrap=True,
                )
                compare_plot = gr.Plot(label="Compare Summary Plot")

            # ==========================================================
            # TAB 4: HISTORY
            # ==========================================================
            with gr.TabItem("Experiment History"):
                with gr.Row():
                    refresh_history_btn = gr.Button("Refresh History", variant="secondary")
                    history_status = gr.Textbox(label="History Status", lines=1)
                history_table = gr.Dataframe(
                    headers=history_headers,
                    datatype=["str"] * len(history_headers),
                    label="Run History",
                    interactive=False,
                    wrap=True,
                )
                history_run_select = gr.Dropdown(label="Select Run ID", choices=[])
                history_detail = gr.JSON(label="Run Detail")

        # ==========================================================
        # EVENTS
        # ==========================================================

        refresh_datasets_btn.click(
            fn=_refresh_dataset_registry,
            inputs=[datasets_base_path],
            outputs=[
                dataset_table,
                dataset_registry_status,
                dataset_select_lab,
                dataset_select_extend,
                train_dataset_id,
                compare_dataset_id,
            ],
        )

        dataset_select_lab.change(
            fn=_dataset_detail_json,
            inputs=[datasets_base_path, dataset_select_lab],
            outputs=[dataset_detail],
        )
        verify_btn.click(
            fn=_verify_dataset,
            inputs=[datasets_base_path, dataset_select_lab],
            outputs=[verify_status, dataset_detail],
        )
        export_config_btn.click(
            fn=_export_dataset_config,
            inputs=[datasets_base_path, dataset_select_lab],
            outputs=[export_config_status, exported_config_file],
        )
        preview_btn.click(
            fn=_preview_dataset,
            inputs=[datasets_base_path, dataset_select_lab, preview_split, preview_index],
            outputs=[preview_plot, preview_info],
        )

        create_btn.click(
            fn=_create_dataset_from_form,
            inputs=[
                datasets_base_path,
                create_model_type,
                create_n_b,
                create_n_train,
                create_n_val,
                create_n_test,
                create_seed,
                create_n_comp_nc,
            ],
            outputs=[
                dataset_table,
                create_status,
                dataset_select_lab,
                dataset_select_extend,
                train_dataset_id,
                compare_dataset_id,
            ],
        )

        extend_btn.click(
            fn=_extend_dataset_from_form,
            inputs=[datasets_base_path, dataset_select_extend, extend_add_train],
            outputs=[
                dataset_table,
                extend_status,
                dataset_select_lab,
                dataset_select_extend,
                train_dataset_id,
                compare_dataset_id,
            ],
        )

        train_dataset_id.change(
            fn=_on_training_dataset_change,
            inputs=[datasets_base_path, train_dataset_id],
            outputs=[train_model_key, train_dataset_info],
        )
        run_training_btn.click(
            fn=_run_training_ui,
            inputs=[
                datasets_base_path,
                train_dataset_id,
                train_model_key,
                train_epochs,
                train_batch_size,
                train_learning_rate,
                train_weight_decay,
                train_early_stop,
                train_seed,
                train_init_seed,
                train_dataloader_seed,
                train_base_filters,
                train_n_layers,
                train_hidden_dim,
                train_use_denoiser,
                train_fno_hidden_channels,
                train_fno_n_layers,
                train_fno_modes,
                train_lambda_dei,
                train_n_restrict_terms,
            ],
            outputs=[train_status, training_curve_plot, train_test_metrics, train_manifest],
        )

        compare_dataset_id.change(
            fn=_on_compare_dataset_change,
            inputs=[datasets_base_path, compare_dataset_id],
            outputs=[compare_models, compare_dataset_info],
        )
        run_compare_btn.click(
            fn=_run_fair_compare_ui,
            inputs=[
                datasets_base_path,
                compare_dataset_id,
                compare_models,
                compare_seeds,
                compare_epochs,
                compare_batch_size,
                compare_learning_rate,
                compare_weight_decay,
                compare_early_stop,
                compare_base_filters,
                compare_n_layers,
                compare_hidden_dim,
                compare_use_denoiser,
                compare_fno_hidden_channels,
                compare_fno_n_layers,
                compare_fno_modes,
                compare_lambda_dei,
                compare_n_restrict_terms,
            ],
            outputs=[compare_status, compare_summary, compare_runs, compare_plot],
        )

        refresh_history_btn.click(
            fn=_refresh_history_ui,
            inputs=[],
            outputs=[history_table, history_status, history_run_select],
        )
        history_run_select.change(
            fn=_history_detail,
            inputs=[history_run_select],
            outputs=[history_detail],
        )

        # Auto-load datasets on startup.
        demo.load(
            fn=_refresh_dataset_registry,
            inputs=[datasets_base_path],
            outputs=[
                dataset_table,
                dataset_registry_status,
                dataset_select_lab,
                dataset_select_extend,
                train_dataset_id,
                compare_dataset_id,
            ],
        )
        # Auto-load history on startup.
        demo.load(
            fn=_refresh_history_ui,
            inputs=[],
            outputs=[history_table, history_status, history_run_select],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7861)

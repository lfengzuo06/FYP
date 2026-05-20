#!/usr/bin/env python3
"""
DEXSY training interface focused on reproducibility and fair comparison.

Tabs:
1) Dataset Lab       - create/list/verify/preview/extend immutable datasets
2) Training Runs     - train on selected dataset with fixed seeds
3) Fair Compare      - evaluate already-trained runs on a selected test dataset
4) Experiment History - inspect run manifests and checkpoints
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from copy import deepcopy
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
from dexsy_datasets.core import ImmutableDataset, compute_dataset_id
from dexsy_datasets.storage import save_dataset

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


COMPARE_DATASET_PURPOSES = {
    "train_val_test": "Training Dataset (train/val/test)",
    "compare_test_only": "Compare/Test Dataset (test only)",
}


def _sanitize_name(text: str, fallback: str = "run") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text or "").strip())
    cleaned = cleaned.strip("_").lower()
    return cleaned[:48] if cleaned else fallback


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_python_scalar(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_python_scalar(v) for v in value]
    return value


def _default_params_yaml(model_type: str, n_compartments_nc: int) -> str:
    _, params = _default_params_for_model(model_type, n_compartments_nc)
    return yaml.safe_dump(params, sort_keys=False, default_flow_style=False)


def _parse_params_override_text(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    loaded = yaml.safe_load(raw)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Parameter override must be a YAML/JSON dictionary.")
    if isinstance(loaded.get("params"), dict):
        return dict(loaded["params"])
    return dict(loaded)


def _hash_payload(payload: dict[str, Any]) -> str:
    normalized = _to_python_scalar(payload)
    data = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _coerce_float_list(value: Any, expected_len: int | None = None, name: str = "value") -> list[float]:
    if value is None:
        raise ValueError(f"`{name}` is required.")
    if isinstance(value, str):
        parts = [p for p in value.replace(" ", "").split(",") if p]
        arr = [float(p) for p in parts]
    elif isinstance(value, (list, tuple, np.ndarray)):
        arr = [float(v) for v in value]
    else:
        raise ValueError(f"`{name}` must be a list/tuple or comma-separated string.")
    if expected_len is not None and len(arr) != expected_len:
        raise ValueError(f"`{name}` must have length {expected_len}, got {len(arr)}.")
    return arr


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


def _on_create_model_change(model_type: str, n_compartments_nc: int) -> str:
    return _default_params_yaml(model_type, int(n_compartments_nc))


def _on_create_purpose_change(dataset_purpose: str):
    if dataset_purpose == "compare_test_only":
        return (
            gr.update(value=0),
            gr.update(value=0),
            gr.update(value=400),
        )
    return (
        gr.update(),
        gr.update(),
        gr.update(),
    )


def _create_dataset_from_form(
    base_path: str,
    model_type: str,
    n_b: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    n_compartments_nc: int,
    sampling_strategy: str,
    dataset_purpose: str,
    params_override_text: str,
):
    try:
        task_type, default_params = _default_params_for_model(model_type, n_compartments_nc)
        params_override = _parse_params_override_text(params_override_text)
        params = {**default_params, **params_override}

        n_train_i = int(n_train)
        n_val_i = int(n_val)
        n_test_i = int(n_test)
        if dataset_purpose == "compare_test_only":
            n_train_i = 0
            n_val_i = 0

        config = {
            "version": "1.0",
            "protocol_version": "1",
            "generator_version": "1.0.0",
            "task_type": task_type,
            "n_train": n_train_i,
            "n_val": n_val_i,
            "n_test": n_test_i,
            "n_b": int(n_b),
            "model_type": model_type,
            "params": params,
            "sampling_strategy": sampling_strategy,
            "min_index_separation": 0,
            "seed": int(seed),
        }
        if dataset_purpose == "compare_test_only":
            config["dataset_role"] = "compare_test_only"
        if params_override:
            config["params_override_hash"] = _hash_payload(params_override)

        ds = create_dataset(config=config, output_dir=base_path, generator_version="1.0.0")
        status = (
            f"Created dataset `{ds.dataset_id}` ({model_type}, task={task_type}, purpose={dataset_purpose}).\n"
            f"split sizes: train={ds.n_train}, val={ds.n_val}, test={ds.n_test}"
        )
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


def _manual_sample_template() -> str:
    template = {
        "diffusions": [1.0e-9, 2.0e-9],
        "volume_fractions": [0.6, 0.4],
        "exchange_rates": [2.0],
        "mixing_time": 0.08,
        "noise_sigma": 0.01,
    }
    return yaml.safe_dump(template, sort_keys=False, default_flow_style=False)


def _build_manual_sample(base_dataset: ImmutableDataset, params: dict[str, Any]) -> dict[str, np.ndarray]:
    model_type = base_dataset.config.get("model_type")
    n_b = int(base_dataset.config.get("n_b", 16))

    if model_type == "gaussian_2c":
        from dexsy_core.forward_model import create_forward_model

        fm = create_forward_model(n_d=n_b, n_b=n_b)
        diffusions = np.asarray(
            _coerce_float_list(params.get("diffusions", [1.0e-9, 2.0e-9]), expected_len=2, name="diffusions"),
            dtype=np.float64,
        )
        fractions = np.asarray(
            _coerce_float_list(params.get("volume_fractions", [0.6, 0.4]), expected_len=2, name="volume_fractions"),
            dtype=np.float64,
        )
        fractions /= max(float(fractions.sum()), 1e-12)
        exchange_rate = float(params.get("exchange_rate", params.get("exchange_rate_01", 2.0)))
        mixing_time = float(params.get("mixing_time", 0.08))
        noise_sigma = float(params.get("noise_sigma", 0.01))
        jitter_pixels = int(params.get("jitter_pixels", 0))
        smoothing_sigma = float(params.get("smoothing_sigma", 0.8))
        spectrum, _, _ = fm.generate_2c_validation_spectrum(
            diffusions=diffusions,
            volume_fractions=fractions,
            exchange_rate=exchange_rate,
            mixing_time=mixing_time,
            jitter_pixels=jitter_pixels,
            smoothing_sigma=smoothing_sigma,
        )
        signal = fm.compute_signal(spectrum, noise_sigma=noise_sigma, normalize=True, noise_model="rician")
        return {"signal": np.asarray(signal, dtype=np.float32), "spectrum": np.asarray(spectrum, dtype=np.float32)}

    if model_type == "gaussian_3c":
        from dexsy_core.forward_model import create_forward_model

        fm = create_forward_model(n_d=n_b, n_b=n_b)
        diffusions = np.asarray(
            _coerce_float_list(params.get("diffusions", [0.8e-9, 1.7e-9, 2.8e-9]), expected_len=3, name="diffusions"),
            dtype=np.float64,
        )
        fractions = np.asarray(
            _coerce_float_list(params.get("volume_fractions", [0.5, 0.3, 0.2]), expected_len=3, name="volume_fractions"),
            dtype=np.float64,
        )
        fractions /= max(float(fractions.sum()), 1e-12)
        exchange_rates = _coerce_float_list(
            params.get("exchange_rates", [2.0, 1.2, 0.8]),
            expected_len=3,
            name="exchange_rates",
        )
        mixing_time = float(params.get("mixing_time", 0.08))
        noise_sigma = float(params.get("noise_sigma", 0.01))
        jitter_pixels = int(params.get("jitter_pixels", 0))
        smoothing_sigma = float(params.get("smoothing_sigma", 0.8))
        spectrum, _, _ = fm.generate_3c_validation_spectrum(
            diffusions=diffusions,
            volume_fractions=fractions,
            exchange_rates=(exchange_rates[0], exchange_rates[1], exchange_rates[2]),
            mixing_time=mixing_time,
            jitter_pixels=jitter_pixels,
            smoothing_sigma=smoothing_sigma,
        )
        signal = fm.compute_signal(spectrum, noise_sigma=noise_sigma, normalize=True, noise_model="rician")
        return {"signal": np.asarray(signal, dtype=np.float32), "spectrum": np.asarray(spectrum, dtype=np.float32)}

    if model_type == "gaussian_nc":
        from dexsy_core.forward_model_nc import create_forward_model_nc

        n_comp = int(base_dataset.config.get("params", {}).get("n_compartments", 3))
        fm = create_forward_model_nc(n_d=n_b, n_b=n_b)
        diffusions = np.asarray(
            _coerce_float_list(params.get("diffusions", np.geomspace(6e-10, 3e-9, n_comp).tolist()), expected_len=n_comp, name="diffusions"),
            dtype=np.float64,
        )
        fractions = np.asarray(
            _coerce_float_list(params.get("volume_fractions", [1.0 / n_comp] * n_comp), expected_len=n_comp, name="volume_fractions"),
            dtype=np.float64,
        )
        fractions /= max(float(fractions.sum()), 1e-12)
        exchange_rate = float(params.get("exchange_rate", 1.0))
        kappa = np.full((n_comp, n_comp), 0.0, dtype=np.float64)
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                kappa[i, j] = exchange_rate
                kappa[j, i] = exchange_rate
        mixing_time = float(params.get("mixing_time", 0.08))
        noise_sigma = float(params.get("noise_sigma", 0.01))
        spectrum, signal, _ = fm.generate_ncompartment_sample(
            N=n_comp,
            phi=fractions,
            D=diffusions,
            kappa=kappa,
            mixing_time=mixing_time,
            noise_sigma=noise_sigma,
            normalize=True,
        )
        return {"signal": np.asarray(signal, dtype=np.float32), "spectrum": np.asarray(spectrum, dtype=np.float32)}

    if model_type == "nongaussian_3c":
        from dexsy_core.forward_model_3c_nongaussian import ForwardModel3CNonGaussian

        cfg = base_dataset.config.get("params", {})
        fm = ForwardModel3CNonGaussian(
            n_b=n_b,
            mixing_time_range=tuple(cfg.get("mixing_time_range", [0.015, 0.300])),
        )
        phi = np.asarray(_coerce_float_list(params.get("phi", [0.5, 0.3, 0.2]), expected_len=3, name="phi"), dtype=np.float64)
        phi /= max(float(phi.sum()), 1e-12)
        rates = {
            "k_et": float(params.get("k_et", 1.0)),
            "k_te": float(params.get("k_te", 1.0)),
            "k_es": float(params.get("k_es", 1.0)),
            "k_se": float(params.get("k_se", 1.0)),
            "k_ts": float(params.get("k_ts", 0.0)),
            "k_st": float(params.get("k_st", 0.0)),
        }
        signal_clean, details = fm.compute_signal(
            phi=phi,
            mixing_time=float(params.get("mixing_time", 0.08)),
            extracellular_diffusivity=float(params.get("D_E", 1.8e-9)),
            intracellular_diffusivity=float(params.get("D_I", 0.8e-9)),
            axon_restricted_length=float(params.get("l_T", 1.2e-6)),
            sphere_radius=float(params.get("R_S", 3.0e-6)),
            rates=rates,
            normalize=False,
        )
        signal = fm.add_rician_noise(
            signal=signal_clean,
            noise_sigma=float(params.get("noise_sigma", 0.01)),
            normalize=True,
        )
        pathway = np.asarray(details["weight_matrix"], dtype=np.float32)
        dei = np.float32(fm.compute_dei_from_weight_matrix(pathway))
        return {
            "signal": np.asarray(signal, dtype=np.float32),
            "pathway_weights": pathway,
            "dei": np.asarray(dei, dtype=np.float32),
        }

    raise ValueError(f"Manual append is not supported for model_type={model_type}")


def _append_manual_sample_from_form(
    base_path: str,
    dataset_id: str,
    target_split: str,
    params_text: str,
):
    if not dataset_id:
        rows, refresh_status, lab_dd, ext_dd, train_dd, compare_dd = _refresh_dataset_registry(base_path)
        return rows, f"{refresh_status}\nPlease select a base dataset.", lab_dd, ext_dd, train_dd, compare_dd
    try:
        base = load_dataset(dataset_id, base_path=base_path, verify=False)
        params = _parse_params_override_text(params_text)
        sample = _build_manual_sample(base, params)

        new_signals = np.concatenate([base.signals, sample["signal"][None, ...]], axis=0).astype(np.float32)
        new_spectra = None
        new_pathway = None
        new_dei = None
        if base.spectra is not None:
            if "spectrum" not in sample:
                raise ValueError("Manual sample must provide `spectrum` for reconstruction datasets.")
            new_spectra = np.concatenate([base.spectra, sample["spectrum"][None, ...]], axis=0).astype(np.float32)
        if base.pathway_weights is not None:
            if "pathway_weights" not in sample:
                raise ValueError("Manual sample must provide `pathway_weights` for pathway datasets.")
            pw = sample["pathway_weights"]
            if pw.ndim == 2:
                pw = pw.reshape(1, 3, 3)
            new_pathway = np.concatenate([base.pathway_weights, pw.astype(np.float32)], axis=0).astype(np.float32)
            if base.dei is not None:
                dei_val = float(np.asarray(sample.get("dei", 0.0)).reshape(-1)[0])
                new_dei = np.concatenate([base.dei, np.asarray([dei_val], dtype=np.float32)], axis=0).astype(np.float32)

        splits = {k: list(v) for k, v in base.splits.items()}
        split_name = target_split if target_split in {"train", "val", "test"} else "train"
        new_idx = int(new_signals.shape[0] - 1)
        splits.setdefault(split_name, [])
        splits[split_name].append(new_idx)

        new_config = deepcopy(base.config)
        if split_name == "train":
            new_config["n_train"] = int(new_config.get("n_train", 0)) + 1
        elif split_name == "val":
            new_config["n_val"] = int(new_config.get("n_val", 0)) + 1
        else:
            new_config["n_test"] = int(new_config.get("n_test", 0)) + 1
        new_config["manual_extension"] = {
            "source_dataset_id": dataset_id,
            "target_split": split_name,
            "sample_hash": _hash_payload(_to_python_scalar(sample)),
            "params_hash": _hash_payload(_to_python_scalar(params)),
        }

        new_dataset = ImmutableDataset(
            dataset_id=compute_dataset_id(new_config),
            signals=new_signals,
            spectra=new_spectra,
            pathway_weights=new_pathway,
            dei=new_dei,
            splits=splits,
            config=_to_python_scalar(new_config),
            metadata=dict(base.metadata),
        )
        save_dataset(new_dataset, base_path=base_path, generator_version="1.0.0")
        status = (
            f"Manual sample appended to `{split_name}`.\n"
            f"New dataset: `{new_dataset.dataset_id}` (train={new_dataset.n_train}, val={new_dataset.n_val}, test={new_dataset.n_test})."
        )
    except Exception as exc:
        status = f"Failed to append manual sample: {exc}"

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
        return gr.update(choices=[], value=[]), {}, "Select a test dataset to load compatible runs."
    try:
        ds = load_dataset(dataset_id, base_path=base_path, verify=False)
        detail = _dataset_detail_json(base_path, dataset_id)
        rows = _load_history_rows()
        choices = []
        for row in rows:
            if not _is_run_compatible_with_dataset(ds, row):
                continue
            ckpt = row.get("checkpoint_path")
            if not ckpt or not Path(ckpt).exists():
                continue
            run_id = row.get("run_id")
            if not run_id:
                continue
            label = (
                f"{run_id} | {row.get('model_key', '?')} | "
                f"name={row.get('run_name') or '-'} | train_ds={row.get('dataset_id', '?')}"
            )
            choices.append((label, run_id))
        status = f"Loaded {len(choices)} compatible trained run(s) for dataset `{dataset_id}`."
        values = [c[1] for c in choices[: min(4, len(choices))]]
        return gr.update(choices=choices, value=values), detail, status
    except Exception as exc:
        return gr.update(choices=[], value=[]), {"error": str(exc)}, f"Failed to load compare runs: {exc}"


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
    run_name: str,
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

    run_prefix = f"train_{_sanitize_name(run_name)}" if str(run_name or "").strip() else "train"
    run_id = _run_id(run_prefix)
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
        "run_name": str(run_name or "").strip(),
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
    run_name: str,
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
            run_name=run_name,
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


def _is_run_compatible_with_dataset(dataset, run_row: dict[str, Any]) -> bool:
    model_key = str(run_row.get("model_key", ""))
    if not model_key:
        return False
    if dataset.task_type == "pathway_regression":
        return model_key == "nonGaussian_cnn"

    model_type = dataset.config.get("model_type")
    if model_type == "gaussian_2c":
        return model_key.startswith("2d_")
    if model_type == "gaussian_3c":
        return model_key.startswith("3d_")
    if model_type == "gaussian_nc":
        return model_key == "nd_attention_unet"
    return False


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _load_model_state(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    state_dict = _normalize_state_dict_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"_K", "_Kt"}
    for key in missing:
        if key.endswith("denoise_scale"):
            allowed_missing.add(key)
    disallowed = [m for m in missing if m not in allowed_missing]
    if disallowed:
        preview = ", ".join(disallowed[:8])
        raise RuntimeError(f"Checkpoint missing keys not allowed: {preview}")
    if unexpected:
        preview = ", ".join(unexpected[:8])
        raise RuntimeError(f"Checkpoint has unexpected keys: {preview}")


def _evaluate_reconstruction_checkpoint(
    dataset,
    model_key: str,
    checkpoint_path: Path,
    run_row: dict[str, Any],
    batch_size_eval: int,
) -> dict[str, float]:
    from dexsy_core.forward_model import create_forward_model
    from dexsy_core.forward_model_nc import create_forward_model_nc
    from dexsy_core.preprocessing import build_model_inputs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(run_row.get("config", {}))
    n_b = int(dataset.config.get("n_b", 16))
    n_comp = _infer_dataset_n_compartments(dataset)
    split = dataset.get_split("test")
    labels = np.asarray(split["spectra"], dtype=np.float32).reshape(-1, 1, n_b, n_b)
    signals = np.asarray(split["signals"], dtype=np.float32).reshape(-1, 1, n_b, n_b)
    if model_key == "nd_attention_unet":
        forward_model = create_forward_model_nc(n_d=n_b, n_b=n_b)
    else:
        forward_model = create_forward_model(n_d=n_b, n_b=n_b)
    prepared_inputs = build_model_inputs(signals, forward_model).astype(np.float32)

    if model_key == "2d_attention_unet":
        from models_2d.attention_unet.model import AttentionUNet2D

        model = AttentionUNet2D(in_channels=3, base_filters=int(cfg.get("base_filters", 32)))
        model_inputs = prepared_inputs
    elif model_key == "2d_plain_unet":
        from models_2d.plain_unet.model import PlainUNet2D

        model = PlainUNet2D(in_channels=3, base_filters=int(cfg.get("base_filters", 32)))
        model_inputs = prepared_inputs
    elif model_key == "2d_pinn":
        from models_2d.pinn.model import PINN2D

        model = PINN2D(signal_size=n_b, in_channels=3)
        model_inputs = prepared_inputs
    elif model_key == "2d_deep_unfolding":
        from models_2d.deep_unfolding.model import DeepUnfolding2D

        model = DeepUnfolding2D(
            n_layers=int(cfg.get("n_layers", 12)),
            n_d=n_b,
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            use_denoiser=bool(cfg.get("use_denoiser", True)),
        )
        model.set_kernel_matrix(torch.from_numpy(forward_model.kernel_matrix).float().to(device))
        model_inputs = signals
    elif model_key == "2d_fno":
        from models_2d.neural_operators.fno import FNO2D

        model = FNO2D(
            in_channels=3,
            hidden_channels=int(cfg.get("fno_hidden_channels", 64)),
            n_layers=int(cfg.get("fno_n_layers", 4)),
            modes=int(cfg.get("fno_modes", 16)),
        )
        model_inputs = prepared_inputs
    elif model_key == "2d_deeponet":
        from models_2d.neural_operators.deeponet import DeepONet2D

        model = DeepONet2D(signal_dim=n_b * n_b, grid_size=n_b)
        model_inputs = prepared_inputs[:, 0:1]
    elif model_key == "3d_attention_unet":
        from models_3d.attention_unet.model import AttentionUNet3C

        model = AttentionUNet3C(in_channels=3, base_filters=int(cfg.get("base_filters", 32)))
        model_inputs = prepared_inputs
    elif model_key == "3d_plain_unet":
        from models_3d.plain_unet.model import PlainUNet3C

        model = PlainUNet3C(in_channels=3, base_filters=int(cfg.get("base_filters", 32)))
        model_inputs = prepared_inputs
    elif model_key == "3d_pinn":
        from models_3d.pinn.model import PINN3C

        model = PINN3C(signal_size=n_b, in_channels=3)
        model_inputs = prepared_inputs
    elif model_key == "3d_deep_unfolding":
        from models_3d.deep_unfolding.model import DeepUnfolding3C

        model = DeepUnfolding3C(
            n_layers=int(cfg.get("n_layers", 12)),
            n_d=n_b,
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            use_denoiser=bool(cfg.get("use_denoiser", True)),
        )
        model.set_kernel_matrix(torch.from_numpy(forward_model.kernel_matrix).float().to(device))
        model_inputs = signals
    elif model_key == "nd_attention_unet":
        from models_nd.attention_unet.model import AttentionUNetND

        model = AttentionUNetND(in_channels=3, base_filters=int(cfg.get("base_filters", 32)))
        model_inputs = prepared_inputs
    else:
        raise ValueError(f"Unsupported reconstruction model key: {model_key}")

    model = model.to(device)
    _load_model_state(model, checkpoint_path=checkpoint_path, device=device)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(model_inputs), int(batch_size_eval)):
            batch = torch.from_numpy(model_inputs[i:i + int(batch_size_eval)]).float().to(device)
            out = _extract_model_output_tensor(model(batch))
            preds.append(out.detach().cpu().numpy().astype(np.float32))
    y_pred = np.concatenate(preds, axis=0)
    return {k: float(v) for k, v in compute_batch_metrics(labels, y_pred)["aggregate"].items()}


def _evaluate_pathway_checkpoint(
    dataset,
    checkpoint_path: Path,
    run_row: dict[str, Any],
    batch_size_eval: int,
) -> dict[str, float]:
    from models_nonGaussian.cnn.model import NonGaussian3CInverseNet

    split = dataset.get_split("test")
    signals = np.asarray(split["signals"], dtype=np.float32)
    if signals.ndim == 3:
        signals = signals[:, None, :, :]
    true_pathway = np.asarray(split["pathway_weights"], dtype=np.float32)
    if true_pathway.ndim == 3:
        true_pathway = true_pathway.reshape(true_pathway.shape[0], -1)
    true_dei = np.asarray(split["dei"], dtype=np.float32).reshape(-1)
    cfg = dict(run_row.get("config", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NonGaussian3CInverseNet(
        hidden_dim=int(cfg.get("hidden_dim", 256)),
    ).to(device)
    _load_model_state(model, checkpoint_path=checkpoint_path, device=device)
    model.eval()

    pred_pathway = []
    pred_dei = []
    with torch.no_grad():
        for i in range(0, len(signals), int(batch_size_eval)):
            x = torch.from_numpy(signals[i:i + int(batch_size_eval)]).float().to(device)
            out = model(x)
            pred_pathway.append(out.pathway_weights.detach().cpu().numpy().astype(np.float32))
            pred_dei.append(out.dei.detach().cpu().numpy().astype(np.float32).reshape(-1))
    pathway_arr = np.concatenate(pred_pathway, axis=0)
    dei_arr = np.concatenate(pred_dei, axis=0)
    return {
        "pathway_mse": float(np.mean((pathway_arr - true_pathway) ** 2)),
        "dei_error": float(np.mean(np.abs(dei_arr - true_dei))),
    }


def _plot_compare_summary(compare_rows: list[dict[str, Any]], metric_name: str) -> plt.Figure | None:
    if not compare_rows:
        return None
    labels = [f"{r['run_id']}\n({r['model_key']})" for r in compare_rows]
    values = [float(r["metric_value"]) for r in compare_rows]
    fig, ax = plt.subplots(figsize=(max(8, len(compare_rows) * 1.4), 4.8))
    x = np.arange(len(compare_rows))
    ax.bar(x, values, color="steelblue", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Fair Compare on Selected Test Dataset ({metric_name})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _run_fair_compare_ui(
    base_path: str,
    dataset_id: str,
    run_ids: list[str],
    batch_size_eval: int,
):
    if not dataset_id:
        return "Please select a test dataset first.", [], [], None
    if not run_ids:
        return "Please select trained runs to compare.", [], [], None

    try:
        dataset = load_dataset(dataset_id, base_path=base_path, verify=False)
        if dataset.n_test <= 0:
            return f"Dataset `{dataset_id}` has no test samples.", [], [], None

        history_rows = _load_history_rows()
        by_id = {str(r.get("run_id")): r for r in history_rows}

        compare_rows: list[dict[str, Any]] = []
        for run_id in run_ids:
            run = by_id.get(str(run_id))
            if run is None:
                continue
            if not _is_run_compatible_with_dataset(dataset, run):
                continue
            ckpt_raw = run.get("checkpoint_path")
            ckpt_path = Path(str(ckpt_raw))
            if not ckpt_path.is_absolute():
                ckpt_path = (ROOT / ckpt_path).resolve()
            if not ckpt_path.exists():
                continue

            t0 = time.time()
            if dataset.task_type == "pathway_regression":
                metrics = _evaluate_pathway_checkpoint(
                    dataset=dataset,
                    checkpoint_path=ckpt_path,
                    run_row=run,
                    batch_size_eval=int(batch_size_eval),
                )
                metric_name = "pathway_mse"
            else:
                metrics = _evaluate_reconstruction_checkpoint(
                    dataset=dataset,
                    model_key=str(run.get("model_key")),
                    checkpoint_path=ckpt_path,
                    run_row=run,
                    batch_size_eval=int(batch_size_eval),
                )
                metric_name = "mse_mean"
            compare_rows.append(
                {
                    "run_id": run.get("run_id"),
                    "run_name": run.get("run_name") or "",
                    "train_dataset_id": run.get("dataset_id"),
                    "test_dataset_id": dataset_id,
                    "model_key": run.get("model_key"),
                    "metric_name": metric_name,
                    "metric_value": float(metrics.get(metric_name, np.nan)),
                    "metrics": metrics,
                    "eval_seconds": float(time.time() - t0),
                    "checkpoint_path": str(ckpt_path),
                }
            )

        if not compare_rows:
            return "No compatible/evaluable runs were found for this dataset.", [], [], None

        compare_rows.sort(key=lambda r: float(r["metric_value"]))
        metric_name = compare_rows[0]["metric_name"]

        summary_by_model: dict[str, list[float]] = {}
        for row in compare_rows:
            summary_by_model.setdefault(str(row["model_key"]), []).append(float(row["metric_value"]))
        summary_rows = []
        for model_key, vals in summary_by_model.items():
            arr = np.asarray(vals, dtype=np.float64)
            summary_rows.append(
                [
                    model_key,
                    metric_name,
                    float(np.nanmean(arr)),
                    float(np.nanstd(arr)),
                    int(arr.size),
                ]
            )
        summary_rows.sort(key=lambda r: float(r[2]))

        run_rows = [
            [
                r["run_id"],
                r["run_name"],
                r["train_dataset_id"],
                r["test_dataset_id"],
                r["model_key"],
                r["metric_name"],
                r["metric_value"],
                r["eval_seconds"],
                r["checkpoint_path"],
            ]
            for r in compare_rows
        ]
        fig = _plot_compare_summary(compare_rows, metric_name)
        status = (
            f"Fair compare completed on test dataset `{dataset_id}`.\n"
            f"Evaluated {len(compare_rows)} trained run(s). Lower `{metric_name}` is better."
        )
        return status, summary_rows, run_rows, fig
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
        "run_name",
        "train_dataset_id",
        "test_dataset_id",
        "model_key",
        "metric_name",
        "metric_value",
        "eval_seconds",
        "checkpoint_path",
    ]

    with gr.Blocks(title="DEXSY Research Interface", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # DEXSY Reproducible Research Interface

            This interface is built for:
            - immutable dataset management
            - reproducible training with explicit seeds
            - fair model comparison across trained runs on one test dataset
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
                    create_dataset_purpose = gr.Dropdown(
                        label="Dataset Purpose",
                        choices=list(COMPARE_DATASET_PURPOSES.keys()),
                        value="train_val_test",
                        info="`compare_test_only` will create test-only dataset (train/val forced to 0).",
                    )
                    create_sampling_strategy = gr.Dropdown(
                        label="Sampling Strategy",
                        choices=["log_uniform", "uniform"],
                        value="log_uniform",
                    )
                with gr.Row():
                    create_n_train = gr.Number(value=9500, precision=0, label="n_train")
                    create_n_val = gr.Number(value=400, precision=0, label="n_val")
                    create_n_test = gr.Number(value=400, precision=0, label="n_test")
                    create_btn = gr.Button("Create Dataset", variant="primary")
                create_params_override = gr.Textbox(
                    label="Parameter Ranges (YAML/JSON Override)",
                    lines=12,
                    value=_default_params_yaml("gaussian_2c", 4),
                )
                create_status = gr.Textbox(label="Create Status", lines=3)

                gr.Markdown("---")
                gr.Markdown("### Extend Dataset (add train samples)")
                with gr.Row():
                    dataset_select_extend = gr.Dropdown(label="Base Dataset", choices=[])
                    extend_add_train = gr.Number(value=1000, precision=0, label="Add Train Samples")
                    extend_btn = gr.Button("Extend Dataset", variant="secondary")
                extend_status = gr.Textbox(label="Extend Status", lines=3)

                gr.Markdown("---")
                gr.Markdown("### Append Manual Sample (one-by-one, via parameters)")
                with gr.Row():
                    append_target_split = gr.Dropdown(
                        label="Append to Split",
                        choices=["train", "val", "test"],
                        value="train",
                    )
                    append_btn = gr.Button("Append Manual Sample", variant="secondary")
                append_params_text = gr.Textbox(
                    label="Manual Sample Parameters (YAML/JSON)",
                    lines=10,
                    value=_manual_sample_template(),
                )
                append_status = gr.Textbox(label="Append Status", lines=3)

            # ==========================================================
            # TAB 2: TRAINING RUNS
            # ==========================================================
            with gr.TabItem("Training Runs"):
                gr.Markdown("### Reproducible training (must select dataset_id)")
                with gr.Row():
                    train_dataset_id = gr.Dropdown(label="Dataset ID", choices=[])
                    train_model_key = gr.Dropdown(label="Model", choices=[])
                    train_run_name = gr.Textbox(label="Run Name (optional)", placeholder="e.g. ablation_lr_low")
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
                gr.Markdown("### Compare already-trained runs on one test dataset")
                with gr.Row():
                    compare_dataset_id = gr.Dropdown(label="Dataset ID", choices=[])
                    compare_batch_size_eval = gr.Number(value=64, precision=0, label="Eval Batch Size")
                compare_dataset_info = gr.JSON(label="Selected Dataset Snapshot")
                compare_run_choices = gr.CheckboxGroup(
                    label="Trained Runs (from history, compatible only)",
                    choices=[],
                )
                compare_candidates_status = gr.Textbox(label="Run Selection Status", lines=2)

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
                    datatype=["str", "str", "str", "str", "str", "str", "number", "number", "str"],
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

        create_model_type.change(
            fn=_on_create_model_change,
            inputs=[create_model_type, create_n_comp_nc],
            outputs=[create_params_override],
        )
        create_n_comp_nc.change(
            fn=_on_create_model_change,
            inputs=[create_model_type, create_n_comp_nc],
            outputs=[create_params_override],
        )
        create_dataset_purpose.change(
            fn=_on_create_purpose_change,
            inputs=[create_dataset_purpose],
            outputs=[create_n_train, create_n_val, create_n_test],
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
                create_sampling_strategy,
                create_dataset_purpose,
                create_params_override,
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
        append_btn.click(
            fn=_append_manual_sample_from_form,
            inputs=[datasets_base_path, dataset_select_extend, append_target_split, append_params_text],
            outputs=[
                dataset_table,
                append_status,
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
                train_run_name,
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
            outputs=[compare_run_choices, compare_dataset_info, compare_candidates_status],
        )
        run_compare_btn.click(
            fn=_run_fair_compare_ui,
            inputs=[
                datasets_base_path,
                compare_dataset_id,
                compare_run_choices,
                compare_batch_size_eval,
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
        demo.load(
            fn=_on_compare_dataset_change,
            inputs=[datasets_base_path, compare_dataset_id],
            outputs=[compare_run_choices, compare_dataset_info, compare_candidates_status],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7861, share=True)

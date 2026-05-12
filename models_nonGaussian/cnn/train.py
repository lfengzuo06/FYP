"""
Training pipeline for non-Gaussian 3C inverse CNN.

Usage:
    python -m models_nonGaussian
or:
    python -m models_nonGaussian.cnn.train
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dexsy_core.forward_model_3c_nongaussian import (
    ForwardModel3CNonGaussian,
)

from .model import (
    PATHWAY_ORDER_3C,
    NonGaussian3CInverseNet,
    NonGaussian3CLoss,
    compute_dei_from_pathway_weights,
)


class NonGaussian3CDataset(Dataset):
    """Dataset for signal -> pathway weights + DEI."""

    def __init__(
        self,
        signals: np.ndarray,
        pathway_weights: np.ndarray,
        dei: np.ndarray | None = None,
    ):
        s = np.asarray(signals, dtype=np.float32)
        w = np.asarray(pathway_weights, dtype=np.float32)

        if s.ndim == 3:
            s = s[:, None, :, :]
        if s.ndim != 4:
            raise ValueError(f"Expected signals with ndim 3 or 4, got shape={s.shape}")

        if w.ndim != 2 or w.shape[1] != 9:
            raise ValueError(f"Expected pathway_weights shape (N,9), got {w.shape}")

        if s.shape[0] != w.shape[0]:
            raise ValueError(
                f"signals N ({s.shape[0]}) must match pathway_weights N ({w.shape[0]})."
            )

        if dei is None:
            dei_t = compute_dei_from_pathway_weights(torch.from_numpy(w)).numpy().astype(np.float32)
        else:
            dei_t = np.asarray(dei, dtype=np.float32).reshape(-1)

        if dei_t.shape[0] != s.shape[0]:
            raise ValueError(f"dei length ({dei_t.shape[0]}) must match N ({s.shape[0]}).")

        self.signals = torch.from_numpy(s)
        self.pathway_weights = torch.from_numpy(w)
        self.dei = torch.from_numpy(dei_t)

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.signals[idx], self.pathway_weights[idx], self.dei[idx]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pathway_weights_from_params_list(
    params_list: Iterable[dict],
    pathway_order: tuple[str, ...] = PATHWAY_ORDER_3C,
) -> np.ndarray:
    """
    Extract (N,9) pathway weights from forward-model metadata params_list.
    """
    rows: list[list[float]] = []
    for p in params_list:
        if "pathway_weights" in p:
            pw = p["pathway_weights"]
            rows.append([float(pw[k]) for k in pathway_order])
        elif "weight_matrix" in p:
            wm = np.asarray(p["weight_matrix"], dtype=np.float64)
            if wm.shape != (3, 3):
                raise ValueError(
                    f"weight_matrix must be 3x3 when present, got {wm.shape}."
                )
            rows.append(wm.reshape(-1).astype(np.float64).tolist())
        else:
            raise KeyError("Each params item must contain 'pathway_weights' or 'weight_matrix'.")

    return np.asarray(rows, dtype=np.float32)


def sample_3c_nongaussian_inverse_dataset(
    forward_model: ForwardModel3CNonGaussian,
    n_samples: int,
    *,
    phi_alpha: tuple[float, float, float] = (4.0, 3.0, 2.0),
    d_e_range: tuple[float, float] = (1.2e-9, 2.4e-9),
    d_i_range: tuple[float, float] = (0.5e-9, 1.5e-9),
    l_t_range: tuple[float, float] = (0.3e-6, 2.0e-6),
    r_s_range: tuple[float, float] = (1.0e-6, 6.0e-6),
    k_te_range: tuple[float, float] = (0.1, 5.0),
    k_se_range: tuple[float, float] = (0.1, 5.0),
    mixing_time_range: tuple[float, float] = (0.015, 0.300),
    snr_choices: tuple[float, ...] = (20.0, 40.0, 80.0, 120.0),
    k_ts: float = 0.0,
    k_st: float = 0.0,
    seed: int | None = None,
) -> dict[str, np.ndarray | list[dict]]:
    """
    Sample synthetic training data using the non-Gaussian 3C forward model.

    detailed balance:
        k_ET = (phi_T / phi_E) * k_TE
        k_ES = (phi_S / phi_E) * k_SE
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")

    rng = np.random.default_rng(seed)
    alpha = np.asarray(phi_alpha, dtype=np.float64)
    if alpha.shape != (3,) or np.any(alpha <= 0):
        raise ValueError("phi_alpha must be a length-3 positive tuple.")

    def _sample_uniform(r: tuple[float, float]) -> float:
        lo, hi = float(r[0]), float(r[1])
        return float(rng.uniform(lo, hi))

    signals_noisy = np.zeros((n_samples, forward_model.n_b, forward_model.n_b), dtype=np.float32)
    signals_clean = np.zeros_like(signals_noisy)
    pathway_weights = np.zeros((n_samples, 9), dtype=np.float32)
    dei = np.zeros((n_samples,), dtype=np.float32)
    params_list: list[dict] = []

    for i in range(n_samples):
        phi = rng.dirichlet(alpha)
        phi_e, phi_t, phi_s = float(phi[0]), float(phi[1]), float(phi[2])

        d_e = _sample_uniform(d_e_range)
        d_i = _sample_uniform(d_i_range)
        l_t = _sample_uniform(l_t_range)
        r_s = _sample_uniform(r_s_range)
        k_te = _sample_uniform(k_te_range)
        k_se = _sample_uniform(k_se_range)
        tm = _sample_uniform(mixing_time_range)

        if phi_e <= 0.0:
            phi_e = 1e-12

        k_et = (phi_t / phi_e) * k_te
        k_es = (phi_s / phi_e) * k_se

        q = forward_model.build_generator(
            k_et=k_et,
            k_te=k_te,
            k_es=k_es,
            k_se=k_se,
            k_ts=k_ts,
            k_st=k_st,
        )

        clean_un, details = forward_model.compute_signal(
            phi=phi,
            q=q,
            mixing_time=tm,
            extracellular_diffusivity=d_e,
            intracellular_diffusivity=d_i,
            axon_restricted_length=l_t,
            sphere_radius=r_s,
            normalize=False,
        )

        snr = float(rng.choice(np.asarray(snr_choices, dtype=np.float64)))
        noise_sigma = 1.0 / max(snr, 1e-8)

        noisy = forward_model.add_rician_noise(
            signal=clean_un,
            noise_sigma=noise_sigma,
            normalize=True,
            rng=rng,
        )

        s0 = max(float(clean_un[0, 0]), 1e-12)
        clean = clean_un / s0

        pw_dict = details["pathway_weights"]
        pw = np.array([float(pw_dict[k]) for k in PATHWAY_ORDER_3C], dtype=np.float32)
        dei_i = float(forward_model.compute_dei_from_weight_matrix(details["weight_matrix"]))

        signals_noisy[i] = noisy.astype(np.float32)
        signals_clean[i] = clean.astype(np.float32)
        pathway_weights[i] = pw
        dei[i] = np.float32(dei_i)

        params_list.append(
            {
                "phi": phi.astype(np.float64).tolist(),
                "D_E": float(d_e),
                "D_I": float(d_i),
                "l_T": float(l_t),
                "R_S": float(r_s),
                "k_TE": float(k_te),
                "k_SE": float(k_se),
                "k_ET": float(k_et),
                "k_ES": float(k_es),
                "k_TS": float(k_ts),
                "k_ST": float(k_st),
                "mixing_time": float(tm),
                "snr": float(snr),
                "noise_sigma": float(noise_sigma),
                "pathway_weights": {k: float(v) for k, v in zip(PATHWAY_ORDER_3C, pw)},
                "weight_matrix": np.asarray(details["weight_matrix"], dtype=np.float64).tolist(),
                "dei": float(dei_i),
            }
        )

    return {
        "signals_noisy": signals_noisy,
        "signals_clean": signals_clean,
        "pathway_weights": pathway_weights,
        "dei": dei,
        "params_list": params_list,
    }


def generate_nongaussian_inverse_splits(
    forward_model: ForwardModel3CNonGaussian,
    *,
    n_train: int = 12000,
    n_val: int = 1200,
    n_test: int = 1200,
    seed: int = 42,
    sample_kwargs: dict | None = None,
) -> dict[str, dict]:
    """Generate train/val/test splits for inverse training."""
    sample_kwargs = {} if sample_kwargs is None else dict(sample_kwargs)

    train = sample_3c_nongaussian_inverse_dataset(
        forward_model=forward_model,
        n_samples=int(n_train),
        seed=int(seed),
        **sample_kwargs,
    )
    val = sample_3c_nongaussian_inverse_dataset(
        forward_model=forward_model,
        n_samples=int(n_val),
        seed=int(seed) + 1,
        **sample_kwargs,
    )
    test = sample_3c_nongaussian_inverse_dataset(
        forward_model=forward_model,
        n_samples=int(n_test),
        seed=int(seed) + 2,
        **sample_kwargs,
    )

    return {"train": train, "val": val, "test": test}


def _make_loader(split: dict, batch_size: int, shuffle: bool) -> DataLoader:
    ds = NonGaussian3CDataset(
        signals=split["signals_noisy"],
        pathway_weights=split["pathway_weights"],
        dei=split["dei"],
    )
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def _evaluate_loader(
    model: NonGaussian3CInverseNet,
    loader: DataLoader,
    criterion: NonGaussian3CLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    n_batches = 0
    n_samples = 0
    total_loss = 0.0
    total_loss_w = 0.0
    total_loss_dei = 0.0
    total_w_mse = 0.0
    total_dei_mae = 0.0

    for signals, w_true, dei_true in loader:
        signals = signals.to(device)
        w_true = w_true.to(device)
        dei_true = dei_true.to(device)

        pred = model(signals)
        loss, metrics = criterion(pred, w_true, dei_true)

        bsz = int(signals.shape[0])
        n_batches += 1
        n_samples += bsz
        total_loss += float(loss.item())
        total_loss_w += float(metrics["loss_w"].item())
        total_loss_dei += float(metrics["loss_dei"].item())
        total_w_mse += float(F.mse_loss(pred.pathway_weights, w_true, reduction="sum").item())
        total_dei_mae += float(torch.abs(pred.dei - dei_true).sum().item())

    if n_batches == 0 or n_samples == 0:
        return {
            "loss": float("nan"),
            "loss_w": float("nan"),
            "loss_dei": float("nan"),
            "w_mse_per_sample": float("nan"),
            "dei_mae": float("nan"),
        }

    return {
        "loss": total_loss / n_batches,
        "loss_w": total_loss_w / n_batches,
        "loss_dei": total_loss_dei / n_batches,
        "w_mse_per_sample": total_w_mse / n_samples,
        "dei_mae": total_dei_mae / n_samples,
    }


def train_nongaussian_inverse_model(
    *,
    output_dir: str | Path | None = None,
    n_train: int = 12000,
    n_val: int = 1200,
    n_test: int = 1200,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    reduce_lr_patience: int = 6,
    reduce_lr_factor: float = 0.5,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 1e-5,
    lambda_dei: float = 1.0,
    base_channels: int = 32,
    hidden_dim: int = 256,
    dropout: float = 0.15,
    n_b: int = 16,
    n_restrict_terms: int = 500,
    seed: int = 42,
    device: str | None = None,
    checkpoint_path: str | None = None,
    sample_kwargs: dict | None = None,
) -> tuple[NonGaussian3CInverseNet, dict, dict, ForwardModel3CNonGaussian, Path]:
    """Train inverse model directly on sampled 3C non-Gaussian data."""
    set_seed(int(seed))

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "checkpoints_nonGaussian" / f"inverse_3c_g{int(n_b)}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    forward_model = ForwardModel3CNonGaussian(
        n_b=int(n_b),
        n_restrict_terms=int(n_restrict_terms),
    )

    datasets = generate_nongaussian_inverse_splits(
        forward_model=forward_model,
        n_train=int(n_train),
        n_val=int(n_val),
        n_test=int(n_test),
        seed=int(seed),
        sample_kwargs=sample_kwargs,
    )

    train_loader = _make_loader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(datasets["val"], batch_size=batch_size, shuffle=False)
    test_loader = _make_loader(datasets["test"], batch_size=batch_size, shuffle=False)

    model = NonGaussian3CInverseNet(
        base_channels=int(base_channels),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    ).to(device_t)

    criterion = NonGaussian3CLoss(lambda_dei=float(lambda_dei)).to(device_t)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(reduce_lr_factor),
        patience=int(reduce_lr_patience),
        min_lr=1e-6,
    )

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device_t)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)

    history: dict[str, list[float] | int | None] = {
        "train_loss": [],
        "train_loss_w": [],
        "train_loss_dei": [],
        "val_loss": [],
        "val_loss_w": [],
        "val_loss_dei": [],
        "val_dei_mae": [],
        "lr": [],
        "best_epoch": None,
    }

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    print(f"[NonGaussian-3C] device={device_t}")
    print(f"[NonGaussian-3C] run_dir={run_dir}")
    print(
        f"[NonGaussian-3C] samples train/val/test="
        f"{len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}"
    )
    print(
        f"[NonGaussian-3C] params: epochs={epochs}, batch_size={batch_size}, "
        f"lambda_dei={lambda_dei}, n_b={n_b}"
    )

    t0 = time.time()

    for epoch in range(int(epochs)):
        model.train()
        n_batches = 0
        train_loss = 0.0
        train_loss_w = 0.0
        train_loss_dei = 0.0

        for signals, w_true, dei_true in train_loader:
            signals = signals.to(device_t)
            w_true = w_true.to(device_t)
            dei_true = dei_true.to(device_t)

            optimizer.zero_grad(set_to_none=True)
            pred = model(signals)
            loss, metrics = criterion(pred, w_true, dei_true)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            n_batches += 1
            train_loss += float(loss.item())
            train_loss_w += float(metrics["loss_w"].item())
            train_loss_dei += float(metrics["loss_dei"].item())

        train_loss /= max(1, n_batches)
        train_loss_w /= max(1, n_batches)
        train_loss_dei /= max(1, n_batches)

        val_metrics = _evaluate_loader(model, val_loader, criterion, device_t)
        val_loss = float(val_metrics["loss"])
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_loss_w"].append(train_loss_w)
        history["train_loss_dei"].append(train_loss_dei)
        history["val_loss"].append(val_loss)
        history["val_loss_w"].append(float(val_metrics["loss_w"]))
        history["val_loss_dei"].append(float(val_metrics["loss_dei"]))
        history["val_dei_mae"].append(float(val_metrics["dei_mae"]))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"train={train_loss:.6f} (W={train_loss_w:.6f}, DEI={train_loss_dei:.6f}) | "
            f"val={val_loss:.6f} (W={val_metrics['loss_w']:.6f}, DEI={val_metrics['loss_dei']:.6f}, "
            f"DEI_MAE={val_metrics['dei_mae']:.6f}) | lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        improved = (best_val - val_loss) > float(early_stopping_min_delta)
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch + 1
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": best_state,
                    "history": history,
                    "config": {
                        "n_b": int(n_b),
                        "base_channels": int(base_channels),
                        "hidden_dim": int(hidden_dim),
                        "dropout": float(dropout),
                        "lambda_dei": float(lambda_dei),
                    },
                },
                run_dir / "best_model.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= int(early_stopping_patience):
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(patience={early_stopping_patience})."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _evaluate_loader(model, test_loader, criterion, device_t)
    elapsed = time.time() - t0

    print(
        f"[NonGaussian-3C] done in {elapsed:.1f}s | "
        f"test loss={test_metrics['loss']:.6f}, "
        f"test W-MSE/sample={test_metrics['w_mse_per_sample']:.6f}, "
        f"test DEI-MAE={test_metrics['dei_mae']:.6f}"
    )

    final_ckpt = {
        "model_state_dict": model.state_dict(),
        "history": history,
        "test_metrics": test_metrics,
        "config": {
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "lambda_dei": float(lambda_dei),
            "base_channels": int(base_channels),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "n_b": int(n_b),
            "n_restrict_terms": int(n_restrict_terms),
            "seed": int(seed),
            "device": str(device_t),
        },
    }
    torch.save(final_ckpt, run_dir / "final_model.pt")

    history_payload = {
        "train_loss": history["train_loss"],
        "train_loss_w": history["train_loss_w"],
        "train_loss_dei": history["train_loss_dei"],
        "val_loss": history["val_loss"],
        "val_loss_w": history["val_loss_w"],
        "val_loss_dei": history["val_loss_dei"],
        "val_dei_mae": history["val_dei_mae"],
        "lr": history["lr"],
        "best_epoch": history["best_epoch"],
        "test_metrics": test_metrics,
    }
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    return model, history, datasets, forward_model, run_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train non-Gaussian 3C inverse model (16x16 signal -> 9 pathway weights)."
    )
    parser.add_argument("--n-train", type=int, default=12000)
    parser.add_argument("--n-val", type=int, default=1200)
    parser.add_argument("--n-test", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-dei", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--n-b", type=int, default=16)
    parser.add_argument("--n-restrict-terms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    train_nongaussian_inverse_model(
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lambda_dei=args.lambda_dei,
        base_channels=args.base_channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        n_b=args.n_b,
        n_restrict_terms=args.n_restrict_terms,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "NonGaussian3CDataset",
    "set_seed",
    "pathway_weights_from_params_list",
    "sample_3c_nongaussian_inverse_dataset",
    "generate_nongaussian_inverse_splits",
    "train_nongaussian_inverse_model",
    "main",
]

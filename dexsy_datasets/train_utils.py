"""
Training utilities and fair comparison framework.

Provides unified training interface and model comparison utilities.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from dexsy_datasets import ImmutableDataset, TrainingRunConfig, fix_all_seeds


@dataclass
class TrainingResult:
    """Container for training results."""

    dataset_id: str
    model: str
    init_seed: int
    dataloader_seed: int
    test_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    history: Dict[str, List[float]] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "model": self.model,
            "init_seed": self.init_seed,
            "dataloader_seed": self.dataloader_seed,
            "test_metrics": self.test_metrics,
            "val_metrics": self.val_metrics,
            "history": self.history,
            "checkpoint_path": self.checkpoint_path,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
        }


class DatasetWrapper(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for ImmutableDataset."""

    def __init__(self, dataset: ImmutableDataset, split: str = "train"):
        self.dataset = dataset
        self.split = split
        self.indices = dataset.splits.get(split, [])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple:
        real_idx = self.indices[idx]
        signals = np.asarray(self.dataset.signals[real_idx], dtype=np.float32)
        if signals.ndim == 2:
            signals = signals[None, :, :]

        if self.dataset.task_type == "pathway_regression":
            targets = np.asarray(self.dataset.pathway_weights[real_idx], dtype=np.float32)
            if targets.ndim == 2:
                targets = targets.reshape(-1)
            if targets.shape[-1] != 9:
                targets = targets.reshape(-1)[:9]
            dei = float(self.dataset.dei[real_idx])
            return (
                torch.from_numpy(signals).float(),
                torch.from_numpy(targets).float(),
                torch.tensor(dei).float(),
            )
        else:
            spectra = np.asarray(self.dataset.spectra[real_idx], dtype=np.float32)
            if spectra.ndim == 2:
                spectra = spectra[None, :, :]
            return (
                torch.from_numpy(signals).float(),
                torch.from_numpy(spectra).float(),
            )


def create_model(model_name: str, task_type: str, **kwargs) -> torch.nn.Module:
    """Create a model by name."""
    if task_type == "pathway_regression":
        if model_name == "cnn_nongaussian":
            return create_cnn_nongaussian(**kwargs)
        else:
            raise ValueError(f"Unknown model for pathway_regression: {model_name}")

    else:  # reconstruction
        if model_name == "attention_unet":
            return create_attention_unet(**kwargs)
        elif model_name == "plain_unet":
            return create_plain_unet(**kwargs)
        else:
            raise ValueError(f"Unknown model for reconstruction: {model_name}")


def create_cnn_nongaussian(
    in_channels: int = 1,
    hidden_dim: int = 256,
    n_pathways: int = 9,
    base_channels: int = 32,
    dropout: float = 0.15,
    architecture: str = "hybrid_transformer",
    transformer_depth: int = 4,
    transformer_heads: int = 8,
    transformer_mlp_ratio: float = 3.0,
) -> torch.nn.Module:
    """Create CNN for pathway regression."""
    try:
        from models_nonGaussian.cnn.model import NonGaussian3CInverseNet
        return NonGaussian3CInverseNet(
            base_channels=base_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            architecture=architecture,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
            transformer_mlp_ratio=transformer_mlp_ratio,
        )
    except ImportError:
        return _SimplePathwayCNN(in_channels, hidden_dim, n_pathways, base_channels, dropout)


def create_attention_unet(
    in_channels: int = 1,
    out_channels: int = 1,
    base_filters: int = 32,
    **kwargs,
) -> torch.nn.Module:
    """Create Attention U-Net."""
    try:
        from models_2d.attention_unet.model import AttentionUNet2D
        return AttentionUNet2D(
            in_channels=in_channels,
            base_filters=base_filters,
        )
    except ImportError:
        return _SimpleUNet(in_channels, out_channels, base_filters)


def create_plain_unet(
    in_channels: int = 1,
    out_channels: int = 1,
    base_filters: int = 32,
    **kwargs,
) -> torch.nn.Module:
    """Create plain U-Net."""
    try:
        from models_2d.plain_unet.model import PlainUNet2D
        return PlainUNet2D(
            in_channels=in_channels,
            base_filters=base_filters,
        )
    except ImportError:
        return _SimpleUNet(in_channels, out_channels, base_filters)


class _SimplePathwayCNN(torch.nn.Module):
    """Fallback CNN for pathway regression."""

    def __init__(self, in_channels, hidden_dim, n_outputs, base_channels, dropout):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, base_channels, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(base_channels * 4, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, n_outputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.nn.functional.softmax(x, dim=1)


class _SimpleUNet(torch.nn.Module):
    """Fallback U-Net for reconstruction."""

    def __init__(self, in_channels, out_channels, base_filters):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, base_filters, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(base_filters, base_filters, 3, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(base_filters, base_filters, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(base_filters, out_channels, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_and_eval(
    dataset: ImmutableDataset,
    run_config: Union[TrainingRunConfig, Dict[str, Any]],
    device: Optional[str] = None,
) -> TrainingResult:
    """Train a model on the dataset and evaluate."""
    if isinstance(run_config, dict):
        run_config = TrainingRunConfig.from_dict(run_config)

    fix_all_seeds(run_config.init_seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    task_type = dataset.task_type

    train_ds = DatasetWrapper(dataset, "train")
    val_ds = DatasetWrapper(dataset, "val")
    test_ds = DatasetWrapper(dataset, "test")

    g = torch.Generator()
    g.manual_seed(run_config.dataloader_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=run_config.batch_size,
        shuffle=True,
        generator=g,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=run_config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=run_config.batch_size, shuffle=False)

    model = create_model(
        run_config.model,
        task_type,
        **run_config.model_kwargs,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=run_config.reduce_lr_factor, patience=run_config.reduce_lr_patience
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    start_time = time.time()

    for epoch in range(run_config.epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            signals = batch[0].to(device)
            targets = batch[1].to(device)
            dei_target = batch[2] if len(batch) > 2 else None

            optimizer.zero_grad()

            model_out = model(signals)
            if task_type == "pathway_regression":
                dei_target = dei_target.to(device) if dei_target is not None else None
                loss = compute_pathway_loss(model_out, targets, dei_target)
            else:
                loss = torch.nn.functional.mse_loss(model_out, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                signals = batch[0].to(device)
                targets = batch[1].to(device)
                dei_target = batch[2] if len(batch) > 2 else None

                model_out = model(signals)
                if task_type == "pathway_regression":
                    dei_target = dei_target.to(device) if dei_target is not None else None
                    loss = compute_pathway_loss(model_out, targets, dei_target)
                else:
                    loss = torch.nn.functional.mse_loss(model_out, targets)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= run_config.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    duration = time.time() - start_time

    test_metrics = evaluate_model(model, test_loader, device, task_type)
    val_metrics = {"val_loss": best_val_loss}

    output_dir = Path(run_config.output_dir or "checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{run_config.model}_{dataset.dataset_id}.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "config": run_config.to_dict(),
            "dataset_id": dataset.dataset_id,
            "task_type": task_type,
        },
        checkpoint_path,
    )

    return TrainingResult(
        dataset_id=dataset.dataset_id,
        model=run_config.model,
        init_seed=run_config.init_seed,
        dataloader_seed=run_config.dataloader_seed,
        test_metrics=test_metrics,
        val_metrics=val_metrics,
        history=history,
        checkpoint_path=str(checkpoint_path),
        duration_seconds=duration,
    )


def _extract_pathway_output(model_output: Any) -> torch.Tensor:
    """Extract (B,9) pathway weights from model output."""
    if torch.is_tensor(model_output):
        out = model_output
    elif hasattr(model_output, "pathway_weights"):
        out = model_output.pathway_weights
    else:
        raise TypeError("Unsupported model output type for pathway regression.")

    if out.ndim == 3 and out.shape[-2:] == (3, 3):
        out = out.reshape(out.shape[0], -1)
    if out.ndim != 2 or out.shape[1] != 9:
        raise ValueError(f"Expected pathway output shape (B,9), got {tuple(out.shape)}")
    return out


def compute_pathway_loss(
    model_output: Any,
    targets: torch.Tensor,
    dei_target: Optional[torch.Tensor] = None,
    lambda_dei: float = 1.0,
) -> torch.Tensor:
    """Compute loss for pathway regression."""
    output = _extract_pathway_output(model_output)
    if targets.ndim == 3 and targets.shape[-2:] == (3, 3):
        targets = targets.reshape(targets.shape[0], -1)

    mse_loss = torch.nn.functional.mse_loss(output, targets, reduction="mean")

    if dei_target is not None:
        pred_dei = compute_dei_from_weights(output)
        dei_loss = torch.nn.functional.mse_loss(pred_dei, dei_target)
        return mse_loss + lambda_dei * dei_loss

    return mse_loss


def compute_dei_from_weights(weights: torch.Tensor) -> torch.Tensor:
    """Compute DEI from pathway weights."""
    if weights.ndim == 3 and weights.shape[-2:] == (3, 3):
        diag = torch.diagonal(weights, dim1=1, dim2=2).sum(dim=1)
        off_diag = weights.sum(dim=(1, 2)) - diag
    else:
        if weights.ndim != 2 or weights.shape[1] != 9:
            raise ValueError(f"Expected pathway weights shape (B,9) or (B,3,3), got {tuple(weights.shape)}")
        diag = weights[:, [0, 4, 8]].sum(dim=1)
        off_diag = weights.sum(dim=1) - diag
    dei = off_diag / (diag + 1e-10)
    return dei


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    task_type: str,
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            signals = batch[0].to(device)
            targets = batch[1].to(device)
            dei_target = batch[2] if len(batch) > 2 else None

            model_out = model(signals)
            if task_type == "pathway_regression":
                dei_target = dei_target.to(device) if dei_target is not None else None
                loss = compute_pathway_loss(model_out, targets, dei_target)
            else:
                loss = torch.nn.functional.mse_loss(model_out, targets)

            total_loss += loss.item() * len(signals)
            total_samples += len(signals)

            if task_type == "pathway_regression":
                pred_w = _extract_pathway_output(model_out)
                all_preds.append(pred_w.cpu().numpy())
                if targets.ndim == 3 and targets.shape[-2:] == (3, 3):
                    all_targets.append(targets.reshape(targets.shape[0], -1).cpu().numpy())
                else:
                    all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / total_samples
    metrics = {"loss": avg_loss}

    if task_type == "pathway_regression" and all_preds:
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        pred_dei = compute_dei_batch(preds)
        target_dei = compute_dei_batch(targets)
        dei_error = np.mean(np.abs(pred_dei - target_dei))
        metrics["dei_error"] = float(dei_error)

        pathway_mse = np.mean((preds - targets) ** 2)
        metrics["pathway_mse"] = float(pathway_mse)

    return metrics


def compute_dei_batch(weights: np.ndarray) -> np.ndarray:
    """Compute DEI from pathway weight matrix batch."""
    w = np.asarray(weights)
    if w.ndim == 3 and w.shape[-2:] == (3, 3):
        diag = np.diagonal(w, axis1=1, axis2=2).sum(axis=1)
        off_diag = w.sum(axis=(1, 2)) - diag
    else:
        if w.ndim != 2 or w.shape[1] != 9:
            raise ValueError(f"Expected pathway weights shape (B,9) or (B,3,3), got {w.shape}")
        diag = w[:, [0, 4, 8]].sum(axis=1)
        off_diag = w.sum(axis=1) - diag
    dei = off_diag / (diag + 1e-10)
    return dei


def compare_models(
    dataset: ImmutableDataset,
    models: List[str],
    seeds: List[int],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run fair model comparison on the same dataset.
    Returns:
        Dict with results DataFrame and summary statistics
    """
    results = []

    for model_name in models:
        for seed in seeds:
            print(f"Training {model_name} with seed {seed}...")

            run_config = TrainingRunConfig(
                dataset_id=dataset.dataset_id,
                model=model_name,
                init_seed=seed,
                dataloader_seed=seed,
            )

            result = train_and_eval(dataset, run_config)
            results.append(result.to_dict())

    rows = []
    for r in results:
        row = {
            "model": r["model"],
            "seed": r["init_seed"],
            "test_loss": r["test_metrics"].get("loss", 0),
            "test_dei_error": r["test_metrics"].get("dei_error", None),
            "test_pathway_mse": r["test_metrics"].get("pathway_mse", None),
            "val_loss": r["val_metrics"].get("val_loss", 0),
            "duration_s": r["duration_seconds"],
        }
        rows.append(row)

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        summary = df.groupby("model").agg(["mean", "std"])
    except ImportError:
        df = None
        summary = None

    output = {
        "dataset_id": dataset.dataset_id,
        "models": models,
        "seeds": seeds,
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
        "summary_df": df,
        "summary_stats": summary,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump({k: v for k, v in output.items() if k not in ["summary_df"]}, f, indent=2, default=str)

    return output

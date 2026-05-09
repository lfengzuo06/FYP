"""
Training script for PINN on 3-Compartment DEXSY.

Usage:
    python -m models_3d.pinn.train
    
Or import and use programmatically:
    from models_3d.pinn import train_model
    
    model, history = train_model(
        n_train=9500,
        n_val=400,
        epochs=60,
        n_compartments=3,
    )
"""

from __future__ import annotations

import os
import time
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dexsy_core.forward_model import ForwardModel2D
from dexsy_core.preprocessing import build_model_inputs

from .model import PINN3C, PINNLoss3C


class DEXSYDataset3C(Dataset):
    """PyTorch Dataset for 3C DEXSY data."""

    def __init__(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        clean_signals: np.ndarray,
        augment: bool = False,
    ):
        self.inputs = torch.from_numpy(inputs).float()
        self.labels = torch.from_numpy(labels).float()
        self.clean_signals = torch.from_numpy(clean_signals).float()
        self.augment = augment

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.labels[idx]
        clean = self.clean_signals[idx]

        if self.augment and random.random() < 0.5:
            x = x.transpose(-1, -2)
            y = y.transpose(-1, -2)
            clean = clean.transpose(-1, -2)

        return x, y, clean


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_dataset(
    forward_model: ForwardModel2D,
    n_train: int = 9500,
    n_val: int = 400,
    n_test: int = 100,
    noise_sigma_range: tuple = (0.005, 0.015),
    n_compartments: int = 3,
    seed: int = 42,
) -> dict:
    """Generate train/val/test datasets for 3C training."""
    set_seed(seed)

    datasets = {}
    generation_kwargs = {
        'n_compartments': n_compartments,
        'normalize': True,
        'noise_model': 'rician',
    }

    for split_name, n_samples in [("train", n_train), ("val", n_val), ("test", n_test)]:
        F, S, _, S_clean = forward_model.generate_batch(
            n_samples=n_samples,
            noise_sigma=None,
            noise_sigma_range=noise_sigma_range,
            return_reference_signal=True,
            **generation_kwargs
        )

        S = S.reshape(-1, 1, forward_model.n_b, forward_model.n_b).astype(np.float32)
        S_clean = S_clean.reshape(-1, 1, forward_model.n_b, forward_model.n_b).astype(np.float32)
        F = F.reshape(-1, 1, forward_model.n_d, forward_model.n_d).astype(np.float32)
        inputs = build_model_inputs(S, forward_model)

        datasets[split_name] = {
            'inputs': inputs,
            'signals': S,
            'clean_signals': S_clean,
            'labels': F
        }

        print(f"  {split_name}: {S.shape[0]} samples, "
              f"noisy signal range [{S.min():.4f}, {S.max():.4f}], "
              f"label range [{F.min():.4f}, {F.max():.4f}]")

    return datasets


def train_model(
    output_dir: str = None,
    n_train: int = 9500,
    n_val: int = 400,
    n_test: int = 100,
    epochs: int = 60,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 12,
    early_stopping_min_delta: float = 1e-4,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    noise_sigma_range: tuple = (0.005, 0.015),
    n_compartments: int = 3,
    seed: int = 42,
    device: str = None,
    checkpoint_path: str = None,
    n_d: int = 64,
    n_b: int = 64,
) -> tuple:
    """
    Train a PINN model on 3C DEXSY.
    
    Args:
        output_dir: Directory for saving outputs
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        epochs: Maximum epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        weight_decay: Weight decay
        early_stopping_patience: Early stopping patience
        early_stopping_min_delta: Minimum improvement for early stopping
        reduce_lr_patience: Reduce LR patience
        reduce_lr_factor: Reduce LR factor
        noise_sigma_range: Noise sigma range
        n_compartments: Number of compartments (3)
        seed: Random seed
        device: Device to use ('cuda', 'cpu', or None for auto)
        checkpoint_path: Optional path to load existing weights
        n_d: Grid size for diffusion dimension
        n_b: Grid size for b-value dimension
        
    Returns:
        (model, history, datasets, forward_model)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "checkpoints_3d" / f"pinn_3c_g{n_d}"
    else:
        output_dir = Path(output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    print(f"\n{'='*60}")
    print(f"PINN Training on 3C Data")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    print(f"Compartments: {n_compartments}")

    # Forward model
    forward_model = ForwardModel2D(n_d=n_d, n_b=n_b)

    # Generate datasets
    print(f"\nGenerating 3C datasets (n_compartments={n_compartments})...")
    datasets = generate_dataset(
        forward_model=forward_model,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        noise_sigma_range=noise_sigma_range,
        n_compartments=n_compartments,
        seed=seed,
    )

    # Create datasets
    train_dataset = DEXSYDataset3C(
        inputs=datasets['train']['inputs'],
        labels=datasets['train']['labels'],
        clean_signals=datasets['train']['clean_signals'],
        augment=True,
    )
    val_dataset = DEXSYDataset3C(
        inputs=datasets['val']['inputs'],
        labels=datasets['val']['labels'],
        clean_signals=datasets['val']['clean_signals'],
        augment=False,
    )
    test_dataset = DEXSYDataset3C(
        inputs=datasets['test']['inputs'],
        labels=datasets['test']['labels'],
        clean_signals=datasets['test']['clean_signals'],
        augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = PINN3C(
        signal_size=forward_model.n_b,
        in_channels=3,
        base_filters=64,
    ).to(device)

    model_config = {
        'signal_size': forward_model.n_b,
        'in_channels': 3,
        'base_filters': 64,
        'model_name': 'pinn_3c',
        'architecture': 'simple_encoder_decoder',
        'n_compartments': n_compartments,
        'checkpoint_format_version': 1,
        'n_d': n_d,
        'n_b': n_b,
    }

    # Print model info
    print(f"\nModel: PINN3C")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', checkpoint.get('config', {}).get('epoch', 0))
        print(f"Loaded checkpoint from epoch {start_epoch}")

    # Loss
    criterion = PINNLoss3C(
        forward_model=forward_model,
        alpha_recon=1.0,
        alpha_forward=1.0,
        alpha_smooth=0.1,
        peak_weight=10.0,
    ).to(device)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=1e-6,
    )

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'best_epoch': None}

    print(f"\nStarting training from epoch {start_epoch}...")
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0

        for inputs, labels, clean_signals in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            clean_signals = clean_signals.to(device)

            optimizer.zero_grad()
            predictions = model(inputs)
            losses = criterion(predictions, labels, clean_signals)
            loss = losses['total']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels, clean_signals in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                clean_signals = clean_signals.to(device)

                predictions = model(inputs)
                losses = criterion(predictions, labels, clean_signals)
                val_loss += losses['total'].item()

        val_loss /= len(val_loader)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        # Early stopping check
        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
            history['best_epoch'] = epoch + 1
            improved = " (improved)"
        else:
            patience_counter += 1
            improved = ""

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {current_lr:.2e} | {epoch_time:.1f}s{improved}")

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = output_dir / f"checkpoints_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save best model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
        'val_loss': best_val_loss,
        'config': {
            **model_config,
            'epoch': epochs,
            'val_loss': best_val_loss,
        }
    }, model_dir / "best_model.pt")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': {
            **model_config,
            'epoch': epochs,
            'val_loss': best_val_loss,
        }
    }, model_dir / "final_model.pt")

    # Save training log
    import csv
    log_path = model_dir / "training_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])
        for i in range(len(history['train_loss'])):
            writer.writerow([i+1, history['train_loss'][i], history['val_loss'][i], history['lr'][i]])

    # Save config
    config_path = model_dir / "config.txt"
    with open(config_path, 'w') as f:
        f.write(f"n_compartments: {n_compartments}\n")
        f.write(f"n_train: {n_train}\n")
        f.write(f"n_val: {n_val}\n")
        f.write(f"n_test: {n_test}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"noise_sigma_range: {noise_sigma_range}\n")
        f.write(f"best_val_loss: {best_val_loss:.6f}\n")
        f.write(f"best_epoch: {history['best_epoch']}\n")

    print(f"\nBest model saved to {model_dir}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return model, history, datasets, forward_model


def main():
    """Main training script."""
    import argparse

    parser = argparse.ArgumentParser(description="Train PINN on 3C DEXSY")
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--n_train', type=int, default=9500, help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=400, help='Number of validation samples')
    parser.add_argument('--n_test', type=int, default=100, help='Number of test samples')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_d', type=int, default=64, help='Grid size for diffusion dimension')
    parser.add_argument('--n_b', type=int, default=64, help='Grid size for b-value dimension')
    args = parser.parse_args()

    train_model(
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_compartments=3,
        seed=args.seed,
        n_d=args.n_d,
        n_b=args.n_b,
    )


if __name__ == "__main__":
    main()

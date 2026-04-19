"""
Training script for Attention U-Net on 2D DEXSY.

Usage:
    python -m models_2d.attention_unet.train

Or import and use programmatically:
    from models_2d.attention_unet import train_model

    model, history = train_model(
        n_train=9500,
        n_val=400,
        epochs=60,
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

from dexsy_core.forward_model import ForwardModel2D
from dexsy_core.preprocessing import build_model_inputs

from .model import AttentionUNet2D, PhysicsInformedLoss


class DEXSYDataset(Dataset):
    """PyTorch Dataset for DEXSY data."""

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
    n_compartments: int = 2,
    seed: int = 42,
) -> dict:
    """Generate train/val/test datasets."""
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
    base_filters: int = 32,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 12,
    early_stopping_min_delta: float = 1e-4,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    noise_sigma_range: tuple = (0.005, 0.015),
    n_compartments: int = 2,
    seed: int = 42,
    device: str = None,
    checkpoint_path: str = None,
) -> tuple:
    """
    Train the Attention U-Net model.

    Args:
        output_dir: Directory for saving outputs
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        epochs: Maximum epochs
        batch_size: Batch size
        base_filters: Base filters
        learning_rate: Initial learning rate
        weight_decay: Weight decay
        early_stopping_patience: Early stopping patience
        early_stopping_min_delta: Minimum improvement for early stopping
        reduce_lr_patience: Reduce LR patience
        reduce_lr_factor: Reduce LR factor
        noise_sigma_range: Noise sigma range
        n_compartments: Number of compartments (2 or 3)
        seed: Random seed
        device: Device to use ('cuda', 'cpu', or None for auto)
        checkpoint_path: Optional path to load existing weights

    Returns:
        (model, history, datasets, forward_model)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "training_output_2d" / "attention_unet"
    else:
        output_dir = Path(output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    set_seed(seed)

    # Initialize forward model
    forward_model = ForwardModel2D(n_d=64, n_b=64)

    # Generate datasets
    print("Generating datasets...")
    datasets = generate_dataset(
        forward_model=forward_model,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        noise_sigma_range=noise_sigma_range,
        n_compartments=n_compartments,
        seed=seed,
    )

    # Create model
    model = AttentionUNet2D(
        in_channels=datasets['train']['inputs'].shape[1],
        base_filters=base_filters
    ).to(device)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)

    # Loss and optimizer
    criterion = PhysicsInformedLoss(
        forward_model=forward_model,
        alpha_kl=1.0,
        alpha_rec=0.2,
        alpha_signal=0.1,
        alpha_sum=0.05,
        peak_weight=6.0,
        alpha_smooth=2e-2,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=1e-6
    )

    # Data loaders
    train_dataset = DEXSYDataset(
        datasets['train']['inputs'],
        datasets['train']['labels'],
        datasets['train']['clean_signals'],
        augment=True,
    )
    val_dataset = DEXSYDataset(
        datasets['val']['inputs'],
        datasets['val']['labels'],
        datasets['val']['clean_signals'],
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )

    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = output_dir / f"checkpoints_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'best_epoch': None}

    print(f"\nTraining Attention U-Net...")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0

        for inputs, labels, clean_signals in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            clean_signals = clean_signals.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels, clean_signals)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches

        # Validate
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for inputs, labels, clean_signals in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                clean_signals = clean_signals.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels, clean_signals)

                total_val_loss += loss.item()
                n_val_batches += 1

        val_loss = total_val_loss / n_val_batches

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: {train_loss:.6f} | "
              f"Val: {val_loss:.6f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
            history['best_epoch'] = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_dir / "best_model.pt")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': {
            'base_filters': base_filters,
            'in_channels': datasets['train']['inputs'].shape[1],
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

    print(f"Saved checkpoints to {model_dir}")

    return model, history, datasets, forward_model


def main():
    """Main training script."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Attention U-Net on 2D DEXSY")
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--n_train', type=int, default=9500, help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=400, help='Number of validation samples')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--base_filters', type=int, default=32, help='Base filters')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    train_model(
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_filters=args.base_filters,
        learning_rate=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

"""
Training script for Unified Attention U-Net on N-Compartment DEXSY.

This script supports:
- Mixed N training: Randomly sample N from a range
- Unified model: Single model handles all N values
- N classification: Automatically predicts number of compartments

Usage:
    python -m models_nd.attention_unet.train_unified

    # Mixed N training (N from 2 to 7)
    python -m models_nd.attention_unet.train_unified --n_min 2 --n_max 7 --epochs 100

    # Single N training (for comparison)
    python -m models_nd.attention_unet.train_unified --n_min 3 --n_max 3 --epochs 60
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

from dexsy_core.forward_model import ForwardModel2D, create_forward_model
from dexsy_core.preprocessing import build_model_inputs

from .model import AttentionUNetUnified, AttentionUNetND, UnifiedLoss, PhysicsInformedLossND


class DEXSYDatasetMixedN(Dataset):
    """PyTorch Dataset for Mixed N-Compartment DEXSY data."""

    def __init__(
        self,
        forward_model: ForwardModel2D,
        n_samples: int,
        n_min: int = 2,
        n_max: int = 7,
        noise_sigma_range: tuple = (0.005, 0.015),
        augment: bool = False,
        seed: int = None,
    ):
        """
        Initialize dataset with on-the-fly data generation.

        Args:
            forward_model: ForwardModelNC instance
            n_samples: Number of samples to generate
            n_min: Minimum N value
            n_max: Maximum N value
            noise_sigma_range: Range for noise sampling
            augment: Whether to apply data augmentation
            seed: Random seed (if None, uses random seed)
        """
        self.forward_model = forward_model
        self.n_samples = n_samples
        self.n_min = n_min
        self.n_max = n_max
        self.noise_sigma_range = noise_sigma_range
        self.augment = augment

        if seed is not None:
            np.random.seed(seed)

        print(f"  Generating {n_samples} samples with N in [{n_min}, {n_max}]...")
        self._generate_data()

    def _generate_data(self):
        """Generate all data samples."""
        # Pre-generate N values for each sample
        self.N_values = np.random.randint(self.n_min, self.n_max + 1, size=self.n_samples)

        # Pre-generate noise levels
        self.noise_levels = np.random.uniform(
            self.noise_sigma_range[0],
            self.noise_sigma_range[1],
            size=self.n_samples
        )

        # Storage for data
        self.inputs = []
        self.labels = []
        self.clean_signals = []
        self.noisy_signals = []

        for i in range(self.n_samples):
            N = int(self.N_values[i])
            noise_sigma = self.noise_levels[i]

            # Generate sample
            f, S, params, S_clean = self.forward_model.generate_ncompartment_sample(
                N=N,
                noise_sigma=noise_sigma,
                return_reference_signal=True,
            )

            # Build inputs
            S_reshaped = S.reshape(1, 1, self.forward_model.n_b, self.forward_model.n_b).astype(np.float32)
            inp = build_model_inputs(S_reshaped, self.forward_model)
            self.inputs.append(inp[0])
            self.labels.append(f.astype(np.float32))
            self.clean_signals.append(S_clean.astype(np.float32))
            self.noisy_signals.append(S.astype(np.float32))

            if (i + 1) % 1000 == 0:
                print(f"    Generated {i + 1}/{self.n_samples} samples")

        self.inputs = np.stack(self.inputs)
        self.labels = np.stack(self.labels)
        self.clean_signals = np.stack(self.clean_signals)
        self.noisy_signals = np.stack(self.noisy_signals)

        print(f"    Generated {self.n_samples} samples")
        print(f"    N distribution: {dict(zip(*np.unique(self.N_values, return_counts=True)))}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).float()
        y = torch.from_numpy(self.labels[idx]).float().unsqueeze(0)
        clean = torch.from_numpy(self.clean_signals[idx]).float().unsqueeze(0)
        noisy = torch.from_numpy(self.noisy_signals[idx]).float()
        N = torch.tensor(self.N_values[idx], dtype=torch.long)

        if self.augment and random.random() < 0.5:
            x = x.transpose(-1, -2)
            y = y.transpose(-1, -2)
            clean = clean.transpose(-1, -2)
            noisy = noisy.transpose(-1, -2)

        return x, y, clean, N


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_unified_model(
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
    n_min: int = 2,
    n_max: int = 7,
    alpha_n_class: float = 0.5,
    seed: int = 42,
    device: str = None,
    checkpoint_path: str = None,
    n_d: int = 64,
    n_b: int = 64,
) -> tuple:
    """
    Train the unified Attention U-Net model on Mixed N-Compartment DEXSY data.

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
        n_min: Minimum number of compartments
        n_max: Maximum number of compartments
        alpha_n_class: Weight for N classification loss
        seed: Random seed
        device: Device to use ('cuda', 'cpu', or None for auto)
        checkpoint_path: Optional path to load existing weights
        n_d: Grid size for diffusion dimension
        n_b: Grid size for b-value dimension

    Returns:
        (model, history, datasets, forward_model)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "checkpoints_nd" / f"unified_n{n_min}_{n_max}_g{n_d}"
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

    # Initialize forward model using factory function
    forward_model = create_forward_model(n_d=n_d, n_b=n_b)

    # Generate datasets
    print(f"\nGenerating Mixed N datasets (N in [{n_min}, {n_max}])...")
    print("Training set:")
    train_dataset = DEXSYDatasetMixedN(
        forward_model=forward_model,
        n_samples=n_train,
        n_min=n_min,
        n_max=n_max,
        noise_sigma_range=noise_sigma_range,
        augment=True,
        seed=seed,
    )

    print("Validation set:")
    val_dataset = DEXSYDatasetMixedN(
        forward_model=forward_model,
        n_samples=n_val,
        n_min=n_min,
        n_max=n_max,
        noise_sigma_range=noise_sigma_range,
        augment=False,
        seed=seed + 1,
    )

    # Create model
    model = AttentionUNetUnified(
        in_channels=3,
        base_filters=base_filters,
        n_min=n_min,
        n_max=n_max,
    ).to(device)

    print(f"\nModel: AttentionUNetUnified (N in [{n_min}, {n_max}])")
    print(f"  Input channels: 3")
    print(f"  Base filters: {base_filters}")
    print(f"  N classes: {model.n_classes}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)

    # Loss and optimizer
    criterion = UnifiedLoss(
        forward_model=forward_model,
        n_min=n_min,
        n_max=n_max,
        alpha_kl=1.0,
        alpha_rec=0.2,
        alpha_signal=0.1,
        alpha_sum=0.05,
        peak_weight=6.0,
        alpha_smooth=2e-2,
        alpha_n_class=alpha_n_class,
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
    history = {
        'train_loss': [],
        'train_physics_loss': [],
        'train_class_loss': [],
        'train_n_accuracy': [],
        'val_loss': [],
        'val_physics_loss': [],
        'val_class_loss': [],
        'val_n_accuracy': [],
        'lr': [],
        'best_epoch': None
    }

    print(f"\nTraining Unified Attention U-Net (N in [{n_min}, {n_max}])...")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}")
    print(f"  Classification weight: {alpha_n_class}")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        model.train()
        total_loss = 0.0
        total_physics_loss = 0.0
        total_class_loss = 0.0
        correct_n = 0
        total_n = 0
        n_batches = 0

        for inputs, labels, clean_signals, N_values in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            clean_signals = clean_signals.to(device)
            N_values = N_values.to(device)

            optimizer.zero_grad()
            spectrum_pred, n_logits = model(inputs)

            loss, loss_dict = criterion(
                spectrum_pred, n_logits,
                labels, N_values, clean_signals
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss_dict['total_loss']
            total_physics_loss += loss_dict['physics_loss']
            total_class_loss += loss_dict['class_loss']

            # Compute N accuracy
            n_pred = torch.argmax(n_logits, dim=1) + n_min
            correct_n += (n_pred == N_values).sum().item()
            total_n += N_values.shape[0]

            n_batches += 1

        train_loss = total_loss / n_batches
        train_physics = total_physics_loss / n_batches
        train_class = total_class_loss / n_batches
        train_n_acc = correct_n / total_n

        # Validate
        model.eval()
        total_val_loss = 0.0
        total_val_physics = 0.0
        total_val_class = 0.0
        correct_val_n = 0
        total_val_n = 0
        n_val_batches = 0

        with torch.no_grad():
            for inputs, labels, clean_signals, N_values in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                clean_signals = clean_signals.to(device)
                N_values = N_values.to(device)

                spectrum_pred, n_logits = model(inputs)

                loss, loss_dict = criterion(
                    spectrum_pred, n_logits,
                    labels, N_values, clean_signals
                )

                total_val_loss += loss_dict['total_loss']
                total_val_physics += loss_dict['physics_loss']
                total_val_class += loss_dict['class_loss']

                n_pred = torch.argmax(n_logits, dim=1) + n_min
                correct_val_n += (n_pred == N_values).sum().item()
                total_val_n += N_values.shape[0]

                n_val_batches += 1

        val_loss = total_val_loss / n_val_batches
        val_physics = total_val_physics / n_val_batches
        val_class = total_val_class / n_val_batches
        val_n_acc = correct_val_n / total_val_n

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['train_physics_loss'].append(train_physics)
        history['train_class_loss'].append(train_class)
        history['train_n_accuracy'].append(train_n_acc)
        history['val_loss'].append(val_loss)
        history['val_physics_loss'].append(val_physics)
        history['val_class_loss'].append(val_class)
        history['val_n_accuracy'].append(val_n_acc)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: {train_loss:.4f} (phys:{train_physics:.4f}, cls:{train_class:.4f}, N-acc:{train_n_acc:.2%}) | "
              f"Val: {val_loss:.4f} (phys:{val_physics:.4f}, cls:{val_class:.4f}, N-acc:{val_n_acc:.2%}) | "
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
                'val_n_accuracy': val_n_acc,
            }, model_dir / "best_model.pt")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {history['best_epoch']}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': {
            'base_filters': base_filters,
            'in_channels': 3,
            'n_d': n_d,
            'n_b': n_b,
            'n_min': n_min,
            'n_max': n_max,
            'n_classes': model.n_classes,
            'alpha_n_class': alpha_n_class,
        }
    }, model_dir / "final_model.pt")

    # Save training log
    import csv
    log_path = model_dir / "training_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'train_physics_loss', 'train_class_loss', 'train_n_accuracy',
            'val_loss', 'val_physics_loss', 'val_class_loss', 'val_n_accuracy', 'lr'
        ])
        for i in range(len(history['train_loss'])):
            writer.writerow([
                i+1,
                history['train_loss'][i],
                history['train_physics_loss'][i],
                history['train_class_loss'][i],
                history['train_n_accuracy'][i],
                history['val_loss'][i],
                history['val_physics_loss'][i],
                history['val_class_loss'][i],
                history['val_n_accuracy'][i],
                history['lr'][i],
            ])

    # Save config
    config_path = model_dir / "config.txt"
    with open(config_path, 'w') as f:
        f.write(f"n_d: {n_d}\n")
        f.write(f"n_b: {n_b}\n")
        f.write(f"n_min: {n_min}\n")
        f.write(f"n_max: {n_max}\n")
        f.write(f"n_train: {n_train}\n")
        f.write(f"n_val: {n_val}\n")
        f.write(f"n_test: {n_test}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"base_filters: {base_filters}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"noise_sigma_range: {noise_sigma_range}\n")
        f.write(f"alpha_n_class: {alpha_n_class}\n")
        f.write(f"best_val_loss: {best_val_loss:.6f}\n")
        f.write(f"best_epoch: {history['best_epoch']}\n")
        f.write(f"train_time: {train_time:.2f}s\n")

    print(f"Saved checkpoints to {model_dir}")

    return model, history, {'train': train_dataset, 'val': val_dataset}, forward_model


def main():
    """Main training script."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Unified Attention U-Net on N-Compartment DEXSY")
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--n_train', type=int, default=9500, help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=400, help='Number of validation samples')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--base_filters', type=int, default=32, help='Base filters')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_min', type=int, default=2, help='Minimum number of compartments')
    parser.add_argument('--n_max', type=int, default=7, help='Maximum number of compartments')
    parser.add_argument('--alpha_n_class', type=float, default=0.5, help='Classification loss weight')
    parser.add_argument('--n_test', type=int, default=100, help='Number of test samples')
    parser.add_argument('--n_d', type=int, default=64, help='Grid size for diffusion dimension')
    parser.add_argument('--n_b', type=int, default=64, help='Grid size for b-value dimension')
    parser.add_argument('--grid_size', type=int, default=64, help='Shorthand: set both n_d and n_b')
    args = parser.parse_args()

    # Use grid_size shorthand if provided
    n_d = args.grid_size if args.grid_size != 64 else args.n_d
    n_b = args.grid_size if args.grid_size != 64 else args.n_b

    train_unified_model(
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_filters=args.base_filters,
        learning_rate=args.lr,
        n_min=args.n_min,
        n_max=args.n_max,
        alpha_n_class=args.alpha_n_class,
        seed=args.seed,
        n_d=n_d,
        n_b=n_b,
    )


if __name__ == "__main__":
    main()

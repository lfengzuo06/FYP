"""
Training script for Deep Unfolding (ISTA-Net style) on 2D DEXSY.

Usage:
    python -m models_2d.deep_unfolding.train

Or import and use programmatically:
    from models_2d.deep_unfolding import train_model

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

from dexsy_core.forward_model import ForwardModel2D, compute_dei
from dexsy_core.preprocessing import build_model_inputs

from .model import DeepUnfolding2D


class DEXSYDataset(Dataset):
    """PyTorch Dataset for DEXSY data for Deep Unfolding."""

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        forward_kernel: np.ndarray,
        augment: bool = False,
    ):
        self.signals = torch.from_numpy(signals).float()
        self.labels = torch.from_numpy(labels).float()
        self.forward_kernel = torch.from_numpy(forward_kernel).float()
        self.augment = augment

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        K = self.forward_kernel

        if self.augment and random.random() < 0.5:
            # Augmentation: transpose the signal and labels
            signal = signal.transpose(-1, -2)
            label = label.transpose(-1, -2)

        return signal, label, K


class DeepUnfoldingLoss(nn.Module):
    """
    Loss function for Deep Unfolding.

    Combines:
    - Forward consistency: ||K @ f - s||^2
    - Reconstruction loss: MSE between predicted and ground truth
    - L1 sparsity: encourages sparse solutions
    """

    def __init__(
        self,
        forward_kernel: torch.Tensor,
        alpha_consistency: float = 1.0,
        alpha_recon: float = 1.0,
        alpha_sparsity: float = 0.01,
    ):
        super().__init__()
        self.register_buffer('_K', forward_kernel)
        self.alpha_consistency = alpha_consistency
        self.alpha_recon = alpha_recon
        self.alpha_sparsity = alpha_sparsity

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        signals: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            predictions: [B, 1, H, W] predicted spectra
            targets: [B, 1, H, W] ground truth spectra
            signals: [B, 1, H, W] input signals

        Returns:
            dict with total_loss and individual components
        """
        batch_size = predictions.shape[0]

        # Ensure non-negative predictions
        predictions_pos = torch.clamp(predictions, min=0)

        # Flatten for matrix operations
        pred_flat = predictions_pos.view(batch_size, -1)
        target_flat = targets.view(batch_size, -1)
        signal_flat = signals.view(batch_size, -1)

        # Forward consistency loss: ||K @ f - s||^2
        Kf = torch.matmul(pred_flat, self._K.T)
        consistency_loss = torch.mean((Kf - signal_flat) ** 2)

        # Reconstruction loss: MSE
        recon_loss = torch.mean((pred_flat - target_flat) ** 2)

        # Sparsity loss: L1 on predictions (encourages sparse solutions)
        sparsity_loss = torch.mean(torch.abs(pred_flat))

        # Total loss
        total_loss = (
            self.alpha_consistency * consistency_loss
            + self.alpha_recon * recon_loss
            + self.alpha_sparsity * sparsity_loss
        )

        return {
            'total': total_loss,
            'consistency': consistency_loss,
            'reconstruction': recon_loss,
            'sparsity': sparsity_loss,
        }


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

        datasets[split_name] = {
            'signals': S,
            'clean_signals': S_clean,
            'labels': F,
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
    n_layers: int = 12,
    hidden_dim: int = 256,
    use_denoiser: bool = True,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    alpha_consistency: float = 1.0,
    alpha_recon: float = 1.0,
    alpha_sparsity: float = 0.01,
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
    Train the Deep Unfolding model.

    Args:
        output_dir: Directory for saving outputs
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        epochs: Maximum epochs
        batch_size: Batch size
        n_layers: Number of ISTA layers
        hidden_dim: Hidden dimension for denoiser
        use_denoiser: Whether to use learnable denoiser
        learning_rate: Initial learning rate
        weight_decay: Weight decay
        alpha_consistency: Weight for forward consistency loss
        alpha_recon: Weight for reconstruction loss
        alpha_sparsity: Weight for sparsity loss
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
        output_dir = Path(__file__).parent.parent.parent / "training_output_2d" / "deep_unfolding"
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
    print(f"Deep Unfolding Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"ISTA layers: {n_layers}, Hidden dim: {hidden_dim}")
    print(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    print(f"Loss weights: consistency={alpha_consistency}, recon={alpha_recon}, sparsity={alpha_sparsity}")

    # Forward model and kernel matrix
    forward_model = ForwardModel2D(n_d=64, n_b=64)
    K = forward_model.kernel_matrix.astype(np.float32)
    K_tensor = torch.from_numpy(K).float().to(device)

    # Generate datasets
    print(f"\nGenerating datasets...")
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
    train_dataset = DEXSYDataset(
        signals=datasets['train']['signals'],
        labels=datasets['train']['labels'],
        forward_kernel=K,
        augment=True,
    )
    val_dataset = DEXSYDataset(
        signals=datasets['val']['signals'],
        labels=datasets['val']['labels'],
        forward_kernel=K,
        augment=False,
    )
    test_dataset = DEXSYDataset(
        signals=datasets['test']['signals'],
        labels=datasets['test']['labels'],
        forward_kernel=K,
        augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = DeepUnfolding2D(
        n_layers=n_layers,
        n_d=64,
        hidden_dim=hidden_dim,
        use_denoiser=use_denoiser,
    ).to(device)

    # Set kernel matrix
    model.set_kernel_matrix(torch.from_numpy(K).float().to(device))

    # Loss
    criterion = DeepUnfoldingLoss(
        forward_kernel=K_tensor,
        alpha_consistency=alpha_consistency,
        alpha_recon=alpha_recon,
        alpha_sparsity=alpha_sparsity,
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Schedulers
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=1e-6,
    )

    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint from epoch {start_epoch}")

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    print(f"\nStarting training from epoch {start_epoch}...")
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_consistency = 0.0
        train_recon = 0.0

        for signals, labels, K_batch in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward
            predictions = model(signals)

            # Loss
            losses = criterion(predictions, labels, signals)
            loss = losses['total']

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_consistency += losses['consistency'].item()
            train_recon += losses['reconstruction'].item()

        n_batches = len(train_loader)
        train_loss /= n_batches
        train_consistency /= n_batches
        train_recon /= n_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_consistency = 0.0
        val_recon = 0.0

        with torch.no_grad():
            for signals, labels, K_batch in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)

                predictions = model(signals)
                losses = criterion(predictions, labels, signals)

                val_loss += losses['total'].item()
                val_consistency += losses['consistency'].item()
                val_recon += losses['reconstruction'].item()

        val_loss /= len(val_loader)
        val_consistency /= len(val_loader)
        val_recon /= len(val_loader)

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
            improved = " (improved)"
        else:
            patience_counter += 1
            improved = ""

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: {train_loss:.4f} (cons={train_consistency:.4f}, recon={train_recon:.4f}) | "
              f"Val: {val_loss:.4f} (cons={val_consistency:.4f}, recon={val_recon:.4f}) | "
              f"LR: {current_lr:.2e} | {epoch_time:.1f}s{improved}")

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'n_layers': n_layers,
                    'hidden_dim': hidden_dim,
                    'use_denoiser': use_denoiser,
                }
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save best model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'n_layers': n_layers,
            'hidden_dim': hidden_dim,
            'use_denoiser': use_denoiser,
            'epoch': epochs,
            'val_loss': best_val_loss,
        }
    }, output_dir / "best_model.pt")
    print(f"\nBest model saved to {output_dir / 'best_model.pt'}")

    # Test evaluation
    model.eval()
    test_loss = 0.0
    test_dei = []

    with torch.no_grad():
        for signals, labels, K_batch in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            predictions = model(signals)
            losses = criterion(predictions, labels, signals)

            test_loss += losses['total'].item()

            # Compute DEI on predictions
            preds_np = predictions.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(preds_np)):
                pred_spectrum = preds_np[i, 0]
                true_spectrum = labels_np[i, 0]
                dei = compute_dei(pred_spectrum)
                test_dei.append(dei)

    test_loss /= len(test_loader)
    test_dei = np.array(test_dei)

    print(f"\n{'='*60}")
    print(f"Test Results")
    print(f"{'='*60}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test DEI: mean={test_dei.mean():.4f}, std={test_dei.std():.4f}")
    print(f"Test DEI range: [{test_dei.min():.4f}, {test_dei.max():.4f}]")

    return model, history, datasets, forward_model


if __name__ == "__main__":
    import sys

    print("Training Deep Unfolding model...")
    print(f"Output directory: {Path(__file__).parent.parent.parent / 'training_output_2d' / 'deep_unfolding'}")

    try:
        model, history, datasets, forward_model = train_model(
            n_train=500,
            n_val=100,
            n_test=50,
            epochs=10,
            batch_size=4,
            n_layers=6,
            hidden_dim=128,
        )
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)

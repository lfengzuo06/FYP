"""
Training script for Deep Unfolding (ISTA-Net style) on 3-Compartment DEXSY.

Usage:
    python -m models_3d.deep_unfolding.train

Or import and use programmatically:
    from models_3d.deep_unfolding import train_model

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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dexsy_core.forward_model import ForwardModel2D, compute_dei

from .model import DeepUnfolding3C


class DEXSYDataset3C(Dataset):
    """PyTorch Dataset for 3C DEXSY data for Deep Unfolding."""

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        clean_signals: np.ndarray,
        augment: bool = False,
    ):
        self.signals = torch.from_numpy(signals).float()
        self.labels = torch.from_numpy(labels).float()
        self.clean_signals = torch.from_numpy(clean_signals).float()
        self.augment = augment

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        clean_signal = self.clean_signals[idx]

        if self.augment and random.random() < 0.5:
            signal = signal.transpose(-1, -2)
            label = label.transpose(-1, -2)
            clean_signal = clean_signal.transpose(-1, -2)

        return signal, label, clean_signal


class DeepUnfoldingLoss3C(nn.Module):
    """
    Loss function for Deep Unfolding on 3C data.

    Combines:
    - KL divergence in spectrum space
    - Weighted reconstruction loss in spectrum space
    - Forward consistency: ||K @ f - s_clean||^2
    - Mass and smoothness regularization
    """

    def __init__(
        self,
        forward_kernel: torch.Tensor,
        n_b: int = 64,
        alpha_kl: float = 1.0,
        alpha_recon: float = 0.2,
        alpha_consistency: float = 0.1,
        alpha_sum: float = 0.05,
        alpha_smooth: float = 2e-2,
        peak_weight: float = 6.0,
    ):
        super().__init__()
        self.register_buffer('_K', forward_kernel)
        self.n_b = n_b
        self.alpha_kl = alpha_kl
        self.alpha_consistency = alpha_consistency
        self.alpha_recon = alpha_recon
        self.alpha_sum = alpha_sum
        self.alpha_smooth = alpha_smooth
        self.peak_weight = peak_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        clean_signals: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            predictions: [B, 1, H, W] predicted spectra
            targets: [B, 1, H, W] ground truth spectra
            clean_signals: [B, 1, H, W] clean target signals

        Returns:
            dict with total_loss and individual components
        """
        batch_size = predictions.shape[0]

        predictions_norm = predictions / (predictions.sum(dim=(2, 3), keepdim=True) + 1e-8)
        targets_norm = targets / (targets.sum(dim=(2, 3), keepdim=True) + 1e-8)

        # Weighted reconstruction + KL
        relative_peak = targets_norm / (targets_norm.amax(dim=(2, 3), keepdim=True) + 1e-8)
        weights = 1.0 + self.peak_weight * torch.sqrt(relative_peak + 1e-8)
        recon_loss = torch.mean(weights * (predictions_norm - targets_norm) ** 2)
        kl_loss = F.kl_div(torch.log(predictions_norm + 1e-8), targets_norm, reduction="batchmean")

        # Forward consistency: ||K @ f - s_clean||^2
        pred_flat = predictions_norm.view(batch_size, -1)
        signal_flat = clean_signals.view(batch_size, -1)
        Kf = torch.matmul(pred_flat, self._K.T)
        consistency_loss = torch.mean((Kf - signal_flat) ** 2)

        total_mass = predictions_norm.sum(dim=(2, 3), keepdim=True)
        sum_penalty = torch.mean((total_mass - 1.0) ** 2)
        smooth_x = torch.mean(torch.abs(predictions_norm[:, :, 1:, :] - predictions_norm[:, :, :-1, :]))
        smooth_y = torch.mean(torch.abs(predictions_norm[:, :, :, 1:] - predictions_norm[:, :, :, :-1]))
        smoothness = smooth_x + smooth_y

        total_loss = (
            self.alpha_kl * kl_loss
            + self.alpha_recon * recon_loss
            + self.alpha_consistency * consistency_loss
            + self.alpha_sum * sum_penalty
            + self.alpha_smooth * smoothness
        )

        return {
            'total': total_loss,
            'kl': kl_loss,
            'consistency': consistency_loss,
            'reconstruction': recon_loss,
            'sum_penalty': sum_penalty,
            'smoothness': smoothness,
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
    init_method: str = "mlp",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    alpha_kl: float = 1.0,
    alpha_consistency: float = 0.1,
    alpha_recon: float = 0.2,
    alpha_sum: float = 0.05,
    alpha_smooth: float = 2e-2,
    peak_weight: float = 6.0,
    early_stopping_patience: int = 12,
    early_stopping_min_delta: float = 1e-4,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    noise_sigma_range: tuple = (0.005, 0.015),
    n_compartments: int = 3,
    seed: int = 42,
    device: str = None,
    checkpoint_path: str = None,
) -> tuple:
    """
    Train the Deep Unfolding model on 3C data.

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
        init_method: Initialization method ('zero', 'constant', 'mlp')
        learning_rate: Initial learning rate
        weight_decay: Weight decay
        alpha_kl: Weight for KL divergence term
        alpha_consistency: Weight for forward consistency loss
        alpha_recon: Weight for reconstruction loss
        alpha_sum: Weight for sum-to-one penalty
        alpha_smooth: Weight for smoothness regularization
        peak_weight: Peak weighting factor
        early_stopping_patience: Early stopping patience
        early_stopping_min_delta: Minimum improvement for early stopping
        reduce_lr_patience: Reduce LR patience
        reduce_lr_factor: Reduce LR factor
        noise_sigma_range: Noise sigma range
        n_compartments: Number of compartments (3)
        seed: Random seed
        device: Device to use ('cuda', 'cpu', or None for auto)
        checkpoint_path: Optional path to load existing weights

    Returns:
        (model, history, datasets, forward_model)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "checkpoints_3d" / "deep_unfolding_3c"
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
    print(f"Deep Unfolding Training on 3C Data")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"ISTA layers: {n_layers}, Hidden dim: {hidden_dim}, Init: {init_method}")
    print(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    print(f"Compartments: {n_compartments}")
    print(
        "Loss weights: "
        f"kl={alpha_kl}, recon={alpha_recon}, consistency={alpha_consistency}, "
        f"sum={alpha_sum}, smooth={alpha_smooth}"
    )

    # Forward model and kernel matrix
    forward_model = ForwardModel2D(n_d=64, n_b=64)
    K = forward_model.kernel_matrix.astype(np.float32)
    K_tensor = torch.from_numpy(K).float().to(device)

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
        signals=datasets['train']['signals'],
        labels=datasets['train']['labels'],
        clean_signals=datasets['train']['clean_signals'],
        augment=True,
    )
    val_dataset = DEXSYDataset3C(
        signals=datasets['val']['signals'],
        labels=datasets['val']['labels'],
        clean_signals=datasets['val']['clean_signals'],
        augment=False,
    )
    test_dataset = DEXSYDataset3C(
        signals=datasets['test']['signals'],
        labels=datasets['test']['labels'],
        clean_signals=datasets['test']['clean_signals'],
        augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = DeepUnfolding3C(
        n_layers=n_layers,
        n_d=64,
        hidden_dim=hidden_dim,
        use_denoiser=use_denoiser,
        init_method=init_method,
    ).to(device)

    print(f"Model: DeepUnfolding3C")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Set kernel matrix
    model.set_kernel_matrix(torch.from_numpy(K).float().to(device))

    # Loss
    criterion = DeepUnfoldingLoss3C(
        forward_kernel=K_tensor,
        n_b=forward_model.n_b,
        alpha_kl=alpha_kl,
        alpha_consistency=alpha_consistency,
        alpha_recon=alpha_recon,
        alpha_sum=alpha_sum,
        alpha_smooth=alpha_smooth,
        peak_weight=peak_weight,
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
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=False,
        )
        allowed_missing = {"_K", "_Kt"}
        if use_denoiser:
            allowed_missing.update({f"ista_layers.{idx}.denoise_scale" for idx in range(n_layers)})
        disallowed_missing = [key for key in missing_keys if key not in allowed_missing]
        if disallowed_missing or unexpected_keys:
            missing_preview = ", ".join(disallowed_missing[:6])
            unexpected_preview = ", ".join(unexpected_keys[:6])
            raise RuntimeError(
                "Checkpoint/model mismatch for deep_unfolding_3c training resume. "
                f"Checkpoint: {checkpoint_path}. "
                f"Disallowed missing keys ({len(disallowed_missing)}): [{missing_preview}]. "
                f"Unexpected keys ({len(unexpected_keys)}): [{unexpected_preview}]."
            )
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint from epoch {start_epoch}")

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
        train_consistency = 0.0
        train_recon = 0.0

        for signals, labels, clean_signals in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            clean_signals = clean_signals.to(device)

            optimizer.zero_grad()

            # Forward
            predictions = model(signals)

            # Loss
            losses = criterion(predictions, labels, clean_signals)
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
            for signals, labels, clean_signals in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                clean_signals = clean_signals.to(device)

                predictions = model(signals)
                losses = criterion(predictions, labels, clean_signals)

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
            history['best_epoch'] = epoch + 1
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
                    'init_method': init_method,
                    'n_compartments': n_compartments,
                }
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")

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
        'config': {
            'n_layers': n_layers,
            'hidden_dim': hidden_dim,
            'use_denoiser': use_denoiser,
            'init_method': init_method,
            'n_compartments': n_compartments,
            'epoch': epochs,
            'val_loss': best_val_loss,
        }
    }, model_dir / "best_model.pt")
    print(f"\nBest model saved to {model_dir / 'best_model.pt'}")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': {
            'n_layers': n_layers,
            'hidden_dim': hidden_dim,
            'use_denoiser': use_denoiser,
            'init_method': init_method,
            'n_compartments': n_compartments,
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
        f.write(f"n_layers: {n_layers}\n")
        f.write(f"hidden_dim: {hidden_dim}\n")
        f.write(f"use_denoiser: {use_denoiser}\n")
        f.write(f"init_method: {init_method}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"noise_sigma_range: {noise_sigma_range}\n")
        f.write(f"best_val_loss: {best_val_loss:.6f}\n")
        f.write(f"best_epoch: {history['best_epoch']}\n")

    # Test evaluation
    model.eval()
    test_loss = 0.0
    test_dei = []

    with torch.no_grad():
        for signals, labels, clean_signals in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            clean_signals = clean_signals.to(device)

            predictions = model(signals)
            losses = criterion(predictions, labels, clean_signals)

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

    print(f"\nSaved checkpoints to {model_dir}")

    return model, history, datasets, forward_model


def main():
    """Main training script."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Deep Unfolding on 3C DEXSY")
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--n_train', type=int, default=9500, help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=400, help='Number of validation samples')
    parser.add_argument('--n_test', type=int, default=100, help='Number of test samples')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--n_layers', type=int, default=12, help='Number of ISTA layers')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    train_model(
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        n_compartments=3,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

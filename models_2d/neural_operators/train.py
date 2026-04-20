"""
Training script for Neural Operator models (DeepONet, FNO) on 2D DEXSY.

Usage:
    python -m models_2d.neural_operators.train

Or import and use programmatically:
    from models_2d.neural_operators.train import train_model

    model, history = train_model(
        model_type='deeponet',
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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dexsy_core.forward_model import ForwardModel2D, compute_dei
from dexsy_core.preprocessing import build_model_inputs

from .model import DeepONet2D, FNO2D


class DEXSYDataset(Dataset):
    """PyTorch Dataset for DEXSY data."""

    def __init__(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ):
        self.inputs = torch.from_numpy(inputs).float()
        self.labels = torch.from_numpy(labels).float()
        self.augment = augment

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.labels[idx]

        if self.augment and random.random() < 0.5:
            x = x.transpose(-1, -2)
            y = y.transpose(-1, -2)

        return x, y


class NeuralOperatorLoss(nn.Module):
    """
    Loss function for Neural Operators.

    Combines:
    - MSE reconstruction loss
    - Forward consistency loss (optional)
    - Smoothness regularization (optional)
    """

    def __init__(
        self,
        alpha_recon: float = 1.0,
        alpha_smooth: float = 0.01,
    ):
        super().__init__()
        self.alpha_recon = alpha_recon
        self.alpha_smooth = alpha_smooth

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            predictions: [B, 1, H, W] predicted spectra
            targets: [B, 1, H, W] ground truth spectra

        Returns:
            dict with total_loss and individual components
        """
        # Reconstruction loss
        recon_loss = torch.mean((predictions - targets) ** 2)

        # Smoothness regularization (Laplacian)
        if self.alpha_smooth > 0:
            laplacian = predictions[:, :, 1:-1, 1:-1] * 4 \
                - predictions[:, :, :-2, 1:-1] \
                - predictions[:, :, 2:, 1:-1] \
                - predictions[:, :, 1:-1, :-2] \
                - predictions[:, :, 1:-1, 2:]
            smooth_loss = torch.mean(laplacian ** 2)
        else:
            # Use same device as predictions to avoid device mismatch
            smooth_loss = torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)

        # Total loss
        total_loss = self.alpha_recon * recon_loss + self.alpha_smooth * smooth_loss

        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'smoothness': smooth_loss,
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
        F = F.reshape(-1, 1, forward_model.n_d, forward_model.n_d).astype(np.float32)
        inputs = build_model_inputs(S, forward_model)

        datasets[split_name] = {
            'inputs': inputs,
            'signals': S,
            'labels': F
        }

        print(f"  {split_name}: {S.shape[0]} samples, "
              f"signal range [{S.min():.4f}, {S.max():.4f}], "
              f"label range [{F.min():.4f}, {F.max():.4f}]")

    return datasets


def train_model(
    model_type: str = "deeponet",
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
    n_compartments: int = 2,
    seed: int = 42,
    device: str = None,
    checkpoint_path: str = None,
    # Model-specific args
    deeponet_branch_dims: list = None,
    deeponet_trunk_dims: list = None,
    deeponet_output_dim: int = 128,
    fno_hidden_channels: int = 64,
    fno_n_layers: int = 4,
    fno_modes: int = 16,
) -> tuple:
    """
    Train a Neural Operator model.

    Args:
        model_type: Type of model ('deeponet' or 'fno')
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
        n_compartments: Number of compartments (2 or 3)
        seed: Random seed
        device: Device to use ('cuda', 'cpu', or None for auto)
        checkpoint_path: Optional path to load existing weights
        deeponet_*: DeepONet-specific arguments
        fno_*: FNO-specific arguments

    Returns:
        (model, history, datasets, forward_model)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "training_output_2d" / f"neural_operators_{model_type}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    print(f"\n{'='*60}")
    print(f"Neural Operator Training ({model_type.upper()})")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")

    # Forward model
    forward_model = ForwardModel2D(n_d=64, n_b=64)

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
        inputs=datasets['train']['inputs'],
        labels=datasets['train']['labels'],
        augment=True,
    )
    val_dataset = DEXSYDataset(
        inputs=datasets['val']['inputs'],
        labels=datasets['val']['labels'],
        augment=False,
    )
    test_dataset = DEXSYDataset(
        inputs=datasets['test']['inputs'],
        labels=datasets['test']['labels'],
        augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    # DeepONet uses raw signal (1-channel), FNO uses 3-channel input (build_model_inputs)
    if model_type == "deeponet":
        if deeponet_branch_dims is None:
            deeponet_branch_dims = [512, 256, 128]
        if deeponet_trunk_dims is None:
            deeponet_trunk_dims = [128, 128, 128]

        # DeepONet takes raw signal: signal_dim = n_b * n_b (4096 for 64x64)
        model = DeepONet2D(
            signal_dim=forward_model.n_b * forward_model.n_b,  # 64 * 64 = 4096
            grid_size=forward_model.n_d,  # 64
            branch_dims=deeponet_branch_dims,
            trunk_dims=deeponet_trunk_dims,
            output_dim=deeponet_output_dim,
        )
        model_config = {
            'model_type': 'deeponet',
            'signal_dim': forward_model.n_b * forward_model.n_b,
            'grid_size': forward_model.n_d,
            'branch_dims': deeponet_branch_dims,
            'trunk_dims': deeponet_trunk_dims,
            'output_dim': deeponet_output_dim,
        }
    elif model_type == "fno":
        # FNO takes 3-channel input: signal + log(signal) + position
        model = FNO2D(
            in_channels=3,
            hidden_channels=fno_hidden_channels,
            n_layers=fno_n_layers,
            modes=fno_modes,
        )
        model_config = {
            'model_type': 'fno',
            'in_channels': 3,
            'hidden_channels': fno_hidden_channels,
            'n_layers': fno_n_layers,
            'modes': fno_modes,
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # Loss
    criterion = NeuralOperatorLoss(alpha_recon=1.0, alpha_smooth=0.01).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Scheduler
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

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Select correct input based on model type
            if model_type == "deeponet":
                # DeepONet takes raw signal: inputs[:, 0:1] is [B, 1, H, W]
                model_input = inputs[:, 0:1]
            else:
                # FNO takes 3-channel input
                model_input = inputs

            optimizer.zero_grad()
            predictions = model(model_input)
            losses = criterion(predictions, labels)
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
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Select correct input based on model type
                if model_type == "deeponet":
                    model_input = inputs[:, 0:1]
                else:
                    model_input = inputs

                predictions = model(model_input)
                losses = criterion(predictions, labels)
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

    # Save best model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            **model_config,
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
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Select correct input based on model type
            if model_type == "deeponet":
                model_input = inputs[:, 0:1]
            else:
                model_input = inputs

            predictions = model(model_input)
            losses = criterion(predictions, labels)
            test_loss += losses['total'].item()

            # Compute DEI on predictions
            preds_np = predictions.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(preds_np)):
                pred_spectrum = preds_np[i, 0]
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

    print("Training Neural Operator model...")
    print(f"Output directory: {Path(__file__).parent.parent.parent / 'training_output_2d'}")

    try:
        model, history, datasets, forward_model = train_model(
            model_type='deeponet',
            n_train=500,
            n_val=100,
            n_test=50,
            epochs=10,
            batch_size=4,
        )
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)

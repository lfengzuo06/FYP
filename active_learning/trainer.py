"""
Fine-tuning Trainer for Active Learning.

This module handles the fine-tuning of PINN models with reduced learning rate
and early stopping on the fixed validation set.
"""

from __future__ import annotations

import csv
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dexsy_core.forward_model import ForwardModel2D
from dexsy_core.preprocessing import build_model_inputs
from models_3d.pinn.model import PINN3C, PINNLoss3C
from models_3d.pinn.train import DEXSYDataset3C
from .config import ALConfig
from .data_protocol import SplitData


@dataclass
class TrainingResult:
    """Container for training results."""
    checkpoint_path: Path
    best_epoch: int
    best_val_loss: float
    final_epoch: int
    history: dict[str, list]
    training_time: float
    n_augmented: int = 0  # Number of augmented samples used in training


class ALDataset(Dataset):
    """
    Custom dataset for active learning training.

    Handles both base training data and augmented data.
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        clean_signals: np.ndarray,
        augment: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            signals: Input signals (N, 1, 64, 64) or (N, 64, 64).
            labels: Ground truth spectra (N, 1, 64, 64) or (N, 64, 64).
            clean_signals: Clean signals (N, 1, 64, 64) or (N, 64, 64).
            augment: Whether to apply augmentation.
        """
        # Handle 3D inputs (N, 64, 64)
        if signals.ndim == 3:
            signals = signals[:, None, :, :]
        if labels.ndim == 3:
            labels = labels[:, None, :, :]
        if clean_signals.ndim == 3:
            clean_signals = clean_signals[:, None, :, :]

        self.signals = torch.from_numpy(signals).float()
        self.labels = torch.from_numpy(labels).float()
        self.clean_signals = torch.from_numpy(clean_signals).float()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int):
        x = self.signals[idx]
        y = self.labels[idx]
        clean = self.clean_signals[idx]

        if self.augment:
            # Simple augmentation: horizontal flip
            if torch.rand(1).item() < 0.5:
                x = x.flip(-1)
                y = y.flip(-1)
                clean = clean.flip(-1)

        return x, y, clean


class FineTuningTrainer:
    """
    Fine-tuning trainer for active learning rounds.

    Fine-tunes from a baseline checkpoint with:
    - Reduced learning rate
    - Early stopping on val_fixed
    - Checkpoint saving per round
    """

    def __init__(
        self,
        config: ALConfig,
        forward_model: ForwardModel2D | None = None,
        device: str | torch.device | None = None,
    ):
        """
        Initialize the trainer.

        Args:
            config: ALConfig with training parameters.
            forward_model: ForwardModel2D instance.
            device: Device to use ('cuda', 'cpu', or None for auto).
        """
        self.config = config
        self.forward_model = forward_model or ForwardModel2D(n_d=64, n_b=64)

        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.output_dir = config.output_dir / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def finetune(
        self,
        base_checkpoint: Path,
        train_data: SplitData,
        val_data: SplitData,
        round_idx: int,
        augmented_data: list | None = None,
        hard_ratio: float = 0.3,
    ) -> TrainingResult:
        """
        Fine-tune from a baseline checkpoint.

        Args:
            base_checkpoint: Path to the baseline checkpoint.
            train_data: Base training data.
            val_data: Validation data (val_fixed).
            round_idx: Current AL round index.
            augmented_data: Optional augmented data to add.

        Returns:
            TrainingResult with checkpoint path and history.
        """
        print(f"\n{'='*60}")
        print(f"Fine-tuning Round {round_idx + 1}")
        print(f"{'='*60}")

        # Create model
        model = self._create_model()

        # Load checkpoint
        checkpoint = torch.load(base_checkpoint, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded baseline from {base_checkpoint}")

        # Create datasets
        train_dataset = self._create_train_dataset(
            train_data, augmented_data, hard_ratio=hard_ratio
        )
        val_dataset = self._create_val_dataset(val_data)

        # Count augmented samples actually used
        n_aug = 0
        if augmented_data is not None:
            n_base = len(train_data.model_inputs)
            n_aug_total = len(augmented_data)
            n_hard = min(int(n_base * hard_ratio / (1 - hard_ratio)), n_aug_total)
            n_aug = n_hard

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Create optimizer with reduced learning rate
        lr = self.config.finetune_lr
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        # Loss function
        criterion = PINNLoss3C(
            forward_model=self.forward_model,
            alpha_recon=1.0,
            alpha_forward=1.0,
            alpha_smooth=0.1,
            peak_weight=10.0,
        ).to(self.device)

        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.reduce_lr_factor,
            patience=self.config.reduce_lr_patience,
            min_lr=1e-6,
        )

        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'lr': [], 'best_epoch': None}

        start_time = time.time()

        for epoch in range(self.config.finetune_epochs):
            epoch_start = time.time()

            # Training
            model.train()
            train_loss = 0.0

            for inputs, labels, clean_signals in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                clean_signals = clean_signals.to(self.device)

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
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    clean_signals = clean_signals.to(self.device)

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
            if val_loss < best_val_loss - self.config.early_stopping_min_delta:
                best_val_loss = val_loss
                best_model_state = deepcopy(model.state_dict())
                patience_counter = 0
                history['best_epoch'] = epoch + 1
                improved = " (improved)"
            else:
                patience_counter += 1
                improved = ""

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}/{self.config.finetune_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"LR: {current_lr:.2e} | {epoch_time:.1f}s{improved}")

            if patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        training_time = time.time() - start_time

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save checkpoint
        checkpoint_path = self.output_dir / f"best_model_round_{round_idx}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': len(history['train_loss']),
            'val_loss': best_val_loss,
            'round': round_idx,
            'config': {
                'signal_size': 64,
                'in_channels': 3,
                'base_filters': self.config.base_filters,
                'model_name': f'pinn_3c_al_round_{round_idx}',
                'n_compartments': 3,
            }
        }, checkpoint_path)

        # Save training log
        log_path = self.output_dir / f"training_log_round_{round_idx}.csv"
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])
            for i in range(len(history['train_loss'])):
                writer.writerow([i+1, history['train_loss'][i], history['val_loss'][i], history['lr'][i]])

        print(f"\nRound {round_idx + 1} checkpoint saved to {checkpoint_path}")
        print(f"Best validation loss: {best_val_loss:.6f} at epoch {history['best_epoch']}")
        print(f"Training time: {training_time:.1f}s")

        return TrainingResult(
            checkpoint_path=checkpoint_path,
            best_epoch=history['best_epoch'] or len(history['train_loss']),
            best_val_loss=best_val_loss,
            final_epoch=len(history['train_loss']),
            history=history,
            training_time=training_time,
            n_augmented=n_aug,
        )

    def _create_model(self) -> PINN3C:
        """Create a PINN3C model."""
        model = PINN3C(
            signal_size=64,
            in_channels=self.config.in_channels,
            base_filters=self.config.base_filters,
        ).to(self.device)
        return model

    def _create_train_dataset(
        self,
        train_data: SplitData,
        augmented_data: list | None,
        hard_ratio: float = 0.3,
    ) -> Dataset:
        """
        Create training dataset from base data and augmented data.

        Args:
            train_data: Base training data (SplitData with model_inputs).
            augmented_data: List of augmented (signal, label, clean, metadata) tuples.
            hard_ratio: Fraction of hard cases in the mixed set.

        Returns:
            ALDataset with 3-channel model inputs.
        """
        # Use model_inputs for training (3-channel input for PINN)
        base_inputs = train_data.model_inputs  # (N, 3, 64, 64)
        base_labels = train_data.labels  # (N, 1, 64, 64)
        base_clean = train_data.clean_signals  # (N, 1, 64, 64)

        if augmented_data is None or len(augmented_data) == 0:
            # Use base training data only
            return ALDataset(
                signals=base_inputs,
                labels=base_labels,
                clean_signals=base_clean,
                augment=True,
            )

        # Apply hard_ratio to determine how many augmented samples to use
        # If we want 30% hard cases: n_hard / (n_base + n_hard) = 0.3
        n_base = len(base_inputs)
        n_aug = len(augmented_data)
        n_hard = int(n_base * hard_ratio / (1 - hard_ratio))
        n_hard = min(n_hard, n_aug)

        if n_hard == 0:
            return ALDataset(
                signals=base_inputs,
                labels=base_labels,
                clean_signals=base_clean,
                augment=True,
            )

        # Sample n_hard augmented samples
        import random
        selected_indices = random.sample(range(n_aug), n_hard)

        # Process augmented samples and compute model_inputs
        aug_inputs_list = [base_inputs]
        aug_labels_list = [base_labels]
        aug_clean_list = [base_clean]

        for idx in selected_indices:
            signal, label, clean, _ = augmented_data[idx]
            # signal: (64, 64), label: (64, 64), clean: (64, 64)
            # Reshape to 4D
            signal_4d = signal.reshape(1, 1, 64, 64).astype(np.float32)
            label_4d = label.reshape(1, 1, 64, 64).astype(np.float32)
            clean_4d = clean.reshape(1, 1, 64, 64).astype(np.float32)
            # Compute model inputs (3-channel)
            aug_input = build_model_inputs(signal_4d, self.forward_model)
            aug_inputs_list.append(aug_input)
            aug_labels_list.append(label_4d)
            aug_clean_list.append(clean_4d)

        # Concatenate
        final_inputs = np.concatenate(aug_inputs_list, axis=0)
        final_labels = np.concatenate(aug_labels_list, axis=0)
        final_clean = np.concatenate(aug_clean_list, axis=0)

        print(f"  Training set: {n_base} base + {n_hard} augmented = {len(final_inputs)} total")

        return ALDataset(
            signals=final_inputs,
            labels=final_labels,
            clean_signals=final_clean,
            augment=True,
        )

    def _create_val_dataset(self, val_data: SplitData) -> Dataset:
        """Create validation dataset."""
        # Use model_inputs for validation (3-channel input for PINN)
        return ALDataset(
            signals=val_data.model_inputs,  # (N, 3, 64, 64)
            labels=val_data.labels,
            clean_signals=val_data.clean_signals,
            augment=False,
        )


def finetune_model(
    base_checkpoint: Path,
    train_data: SplitData,
    val_data: SplitData,
    config: ALConfig,
    round_idx: int,
    augmented_data: list | None = None,
    forward_model: ForwardModel2D | None = None,
) -> TrainingResult:
    """
    Convenience function for fine-tuning.

    Args:
        base_checkpoint: Path to baseline checkpoint.
        train_data: Training data.
        val_data: Validation data.
        config: ALConfig instance.
        round_idx: Current AL round.
        augmented_data: Augmented samples.
        forward_model: ForwardModel2D instance.

    Returns:
        TrainingResult instance.
    """
    trainer = FineTuningTrainer(
        config=config,
        forward_model=forward_model,
    )

    return trainer.finetune(
        base_checkpoint=base_checkpoint,
        train_data=train_data,
        val_data=val_data,
        round_idx=round_idx,
        augmented_data=augmented_data,
    )

"""
Training pipeline for 2D DEXSY Inversion Models (PyTorch).

This script trains the improved 2D DEXSY inversion models and compares
them with the baseline (original ResNet-Dense-SE) approach.
"""

import os
import time
import random
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

try:
    from .forward_model_2d import ForwardModel2D, compute_dei
    from .preprocessing_2d import build_model_inputs, build_position_channel
    from .model_2d import (
        get_model, AttentionUNet2D, MultiscaleUNet2D,
        PhysicsInformedLoss
    )
except ImportError:  # pragma: no cover - allows running as a script.
    from forward_model_2d import ForwardModel2D, compute_dei
    from preprocessing_2d import build_model_inputs, build_position_channel
    from model_2d import (
        get_model, AttentionUNet2D, MultiscaleUNet2D,
        PhysicsInformedLoss
    )


class DEXSYDataset(Dataset):
    """PyTorch Dataset for DEXSY data."""

    def __init__(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        clean_signals: np.ndarray,
        augment: bool = False,
    ):
        """
        Args:
            inputs: Model inputs, shape (N, C, n_b, n_b)
            labels: Label arrays, shape (N, 1, n_d, n_d)
            clean_signals: Noise-free signal targets, shape (N, 1, n_b, n_b)
            augment: If True, apply symmetry-preserving augmentation.
        """
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

        # Swapping the two encoding axes is physically valid for symmetric DEXSY.
        if self.augment and random.random() < 0.5:
            x = x.transpose(-1, -2)
            y = y.transpose(-1, -2)
            clean = clean.transpose(-1, -2)

        return x, y, clean


class TrainingPipeline2D:
    """
    Training pipeline for 2D DEXSY inversion (PyTorch).

    Features:
    - Automatic data generation
    - Multiple model support
    - Metrics tracking
    - Checkpoint saving
    """

    def __init__(
        self,
        output_dir: str = "training_output_2d",
        n_d: int = 64,
        n_b: int = 64,
        seed: int = 42,
        device: str = None
    ):
        """
        Initialize training pipeline.

        Args:
            output_dir: Directory for saving outputs
            n_d: Diffusion coefficient grid size
            n_b: b-value grid size
            seed: Random seed for reproducibility
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.output_dir = output_dir
        self.n_d = n_d
        self.n_b = n_b
        self.seed = seed

        # Set seeds
        self._set_seed(seed)

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

        # Initialize forward model
        self.forward_model = ForwardModel2D(n_d=n_d, n_b=n_b)
        self.position_channel = self._build_position_channel()

        # Trackers
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}

    def _build_position_channel(self) -> np.ndarray:
        """Build one positional channel from the log-b sampling coordinates."""
        return build_position_channel(self.forward_model)

    def _build_model_inputs(self, noisy_signals: np.ndarray) -> np.ndarray:
        """
        Build the paper-style three-channel input:
        raw signal, log signal, and one positional encoding channel.
        """
        return build_model_inputs(noisy_signals, self.forward_model)

    def _generate_split(
        self,
        n_samples: int,
        noise_levels: list = None,
        balance_noise: bool = False,
        noise_sigma_range: tuple = (0.005, 0.015),
        **kwargs
    ) -> tuple:
        """Generate one dataset split using paper-style continuous noise or legacy discrete levels."""
        if noise_levels is None:
            return self.forward_model.generate_batch(
                n_samples=n_samples,
                noise_sigma=None,
                noise_sigma_range=noise_sigma_range,
                return_reference_signal=True,
                **kwargs
            )

        if balance_noise:
            counts = np.full(len(noise_levels), n_samples // len(noise_levels), dtype=int)
            counts[:n_samples % len(noise_levels)] += 1
            assigned_noise = np.concatenate([
                np.full(count, noise, dtype=float)
                for noise, count in zip(noise_levels, counts)
                if count > 0
            ])
        else:
            assigned_noise = np.random.choice(noise_levels, size=n_samples).astype(float)

        F_parts = []
        S_parts = []
        S_clean_parts = []
        params = []

        for noise in np.unique(assigned_noise):
            count = int(np.sum(assigned_noise == noise))
            if count == 0:
                continue
            F, S, split_params, S_clean = self.forward_model.generate_batch(
                n_samples=count,
                noise_sigma=float(noise),
                return_reference_signal=True,
                **kwargs
            )
            F_parts.append(F)
            S_parts.append(S)
            S_clean_parts.append(S_clean)
            params.extend(split_params)

        F = np.concatenate(F_parts, axis=0)
        S = np.concatenate(S_parts, axis=0)
        S_clean = np.concatenate(S_clean_parts, axis=0)
        perm = np.random.permutation(F.shape[0])
        F = F[perm]
        S = S[perm]
        S_clean = S_clean[perm]
        params = [params[i] for i in perm]
        return F, S, params, S_clean

    def _set_seed(self, seed: int):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate_dataset(
        self,
        n_train: int = 5000,
        n_val: int = 500,
        n_test: int = 100,
        noise_levels: list = None,
        noise_sigma_range: tuple = (0.005, 0.015),
        balance_noise: bool = False,
        n_compartments: int = 2,
    ) -> dict:
        """
        Generate training, validation, and test datasets.

        Args:
            n_train: Number of training samples
            n_val: Number of validation samples
            n_test: Number of test samples
            noise_levels: Optional discrete noise levels. If None, sample sigma continuously.
            noise_sigma_range: Continuous paper-style noise range.
            balance_noise: If True, balance samples across discrete noise levels.
            n_compartments: 2 or 3.

        Returns:
            Dictionary with train, val, test data
        """
        print("Generating datasets...")

        datasets = {}
        generation_kwargs = {
            'n_compartments': n_compartments,
            'normalize': True,
            'noise_model': 'rician',
        }

        for split_name, n_samples in [("train", n_train), ("val", n_val), ("test", n_test)]:
            F, S, _, S_clean = self._generate_split(
                n_samples=n_samples,
                noise_levels=noise_levels,
                balance_noise=(balance_noise and split_name == "train"),
                noise_sigma_range=noise_sigma_range,
                **generation_kwargs
            )

            # Reshape for model input: PyTorch format (batch, 1, n_b, n_b)
            S = S.reshape(-1, 1, self.n_b, self.n_b).astype(np.float32)
            S_clean = S_clean.reshape(-1, 1, self.n_b, self.n_b).astype(np.float32)
            F = F.reshape(-1, 1, self.n_d, self.n_d).astype(np.float32)
            inputs = self._build_model_inputs(S)

            datasets[split_name] = {
                'inputs': inputs,
                'signals': S,
                'clean_signals': S_clean,
                'labels': F
            }

            print(f"  {split_name}: {S.shape[0]} samples, "
                  f"noisy signal range [{S.min():.4f}, {S.max():.4f}], "
                  f"clean signal range [{S_clean.min():.4f}, {S_clean.max():.4f}], "
                  f"label range [{F.min():.4f}, {F.max():.4f}]")

        return datasets

    def _train_epoch(self, model, dataloader, criterion, optimizer, device):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for inputs, labels, clean_signals in dataloader:
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

        return total_loss / n_batches

    def _validate(self, model, dataloader, criterion, device):
        """Validate the model."""
        model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for inputs, labels, clean_signals in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                clean_signals = clean_signals.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels, clean_signals)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def _build_dataloaders(self, datasets, batch_size):
        """Build PyTorch DataLoaders."""
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
            pin_memory=(self.device.type == 'cuda')
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == 'cuda')
        )

        return train_loader, val_loader

    def train_model(
        self,
        model_name: str = "attention_unet",
        datasets: dict = None,
        epochs: int = 100,
        batch_size: int = 8,
        base_filters: int = 32,
        learning_rate: float = 5e-4,
        early_stopping_patience: int = 12,
        early_stopping_min_delta: float = 1e-4,
        reduce_lr_patience: int = 5,
        reduce_lr_factor: float = 0.5,
        model_weights_path: str = None
    ) -> tuple:
        """
        Train a model.

        Args:
            model_name: Name of model to train
            datasets: Dictionary with train/val data
            epochs: Maximum epochs
            batch_size: Batch size
            base_filters: Base filters
            learning_rate: Initial learning rate
            early_stopping_patience: Early stopping patience
            reduce_lr_patience: Reduce LR patience
            reduce_lr_factor: Reduce LR factor
            model_weights_path: Path to load existing weights (for fine-tuning)

        Returns:
            (model, history)
        """
        print(f"\nTraining {model_name}...")
        print("=" * 60)

        # Create model
        model = get_model(
            model_name=model_name,
            base_filters=base_filters,
            in_channels=datasets['train']['inputs'].shape[1],
        ).to(self.device)

        # Load weights if provided
        if model_weights_path and os.path.exists(model_weights_path):
            print(f"Loading weights from {model_weights_path}")
            checkpoint = torch.load(model_weights_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)

        # Loss and optimizer
        criterion = PhysicsInformedLoss(
            forward_model=self.forward_model,
            alpha_kl=1.0,
            alpha_rec=0.2,
            alpha_signal=0.1,
            alpha_sum=0.05,
            peak_weight=6.0,
            alpha_smooth=2e-2,
        ).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=1e-6
        )

        # Data loaders
        train_loader, val_loader = self._build_dataloaders(datasets, batch_size)

        # Create checkpoint directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.output_dir, "checkpoints", f"{model_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        history = {'train_loss': [], 'val_loss': [], 'lr': [], 'best_epoch': None}

        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer, self.device)
            val_loss = self._validate(model, val_loader, criterion, self.device)

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
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")

            # Save best model
            if val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = val_loss
                best_model_state = deepcopy(model.state_dict())
                patience_counter = 0
                history['best_epoch'] = epoch + 1
                # Save checkpoint
                best_path = os.path.join(model_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, best_path)
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
        final_path = os.path.join(model_dir, "final_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
        }, final_path)
        print(f"Saved final model to {final_path}")

        # Save training log
        log_path = os.path.join(model_dir, "training_log.csv")
        import csv
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])
            for i in range(len(history['train_loss'])):
                writer.writerow([i+1, history['train_loss'][i], history['val_loss'][i], history['lr'][i]])

        return model, history

    def evaluate_model(
        self,
        model,
        datasets: dict,
        forward_model: ForwardModel2D = None
    ) -> dict:
        """
        Evaluate a trained model.

        Args:
            model: Trained model
            datasets: Dictionary with test data
            forward_model: Forward model for DEI computation

        Returns:
            Dictionary with metrics
        """
        print("\nEvaluating model...")

        model.eval()

        X_test = torch.from_numpy(datasets['test']['inputs']).float()
        y_test = datasets['test']['labels']
        clean_test = datasets['test']['clean_signals']
        if forward_model is None:
            forward_model = self.forward_model

        # Predict in batches
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X_test), 64):
                batch = X_test[i:i+64].to(self.device)
                pred = model(batch)
                predictions.append(pred.cpu().numpy())

        y_pred = np.concatenate(predictions, axis=0)
        pred_signals = np.einsum(
            'bij,mnij->bmn',
            y_pred[:, 0],
            forward_model._kernel
        )[:, None, :, :]

        # Compute metrics (pure numpy, no sklearn needed)
        y_test_flat = y_test.reshape(y_test.shape[0], -1)
        y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
        signal_test_flat = clean_test.reshape(clean_test.shape[0], -1)
        pred_signal_flat = pred_signals.reshape(pred_signals.shape[0], -1)

        mse = np.mean((y_test_flat - y_pred_flat) ** 2)
        mae = np.mean(np.abs(y_test_flat - y_pred_flat))
        rmse = np.sqrt(mse)
        signal_mse = np.mean((signal_test_flat - pred_signal_flat) ** 2)

        # Compute per-sample metrics (pure numpy, no sklearn needed)
        mse_per_sample = np.array([
            np.mean((y_test[i].flatten() - y_pred[i].flatten()) ** 2)
            for i in range(len(y_test))
        ])
        mae_per_sample = np.array([
            np.mean(np.abs(y_test[i].flatten() - y_pred[i].flatten()))
            for i in range(len(y_test))
        ])
        true_dei = np.array([compute_dei(y_test[i, 0]) for i in range(len(y_test))])
        pred_dei = np.array([compute_dei(y_pred[i, 0]) for i in range(len(y_pred))])
        dei_mae = np.mean(np.abs(true_dei - pred_dei))

        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'signal_mse': signal_mse,
            'signal_mse_noisy': np.mean((datasets['test']['signals'].reshape(datasets['test']['signals'].shape[0], -1) - pred_signal_flat) ** 2),
            'dei_mae': dei_mae,
            'true_dei': true_dei,
            'pred_dei': pred_dei,
            'mse_std': mse_per_sample.std(),
            'mae_std': mae_per_sample.std(),
            'mse_per_sample': mse_per_sample,
            'mae_per_sample': mae_per_sample,
            'predictions': y_pred,
            'predicted_signals': pred_signals,
            'test_labels': y_test,
            'test_signals': datasets['test']['signals'],
            'test_clean_signals': clean_test,
        }

        print(f"  MSE: {mse:.6f} (+/- {mse_per_sample.std():.6f})")
        print(f"  MAE: {mae:.6f} (+/- {mae_per_sample.std():.6f})")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  Signal MSE (clean): {signal_mse:.6f}")
        print(f"  Signal MSE (noisy): {metrics['signal_mse_noisy']:.6f}")
        print(f"  DEI MAE: {dei_mae:.6f}")

        return metrics

    def plot_comparison(
        self,
        model,
        datasets: dict,
        n_examples: int = 5,
        save_path: str = None
    ):
        """
        Plot comparison between true and predicted distributions.

        Args:
            model: Trained model
            datasets: Test dataset
            n_examples: Number of examples to show
            save_path: Path to save figure
        """
        X_test = datasets['test']['signals']
        X_inputs = datasets['test']['inputs']
        y_test = datasets['test']['labels']

        # Random examples
        indices = random.sample(range(len(X_test)), min(n_examples, len(X_test)))

        fig, axes = plt.subplots(n_examples, 3, figsize=(12, 4 * n_examples))

        if n_examples == 1:
            axes = axes.reshape(1, -1)

        model.eval()
        with torch.no_grad():
            for i, idx in enumerate(indices):
                # Input signal
                axes[i, 0].imshow(X_test[idx, 0], cmap='viridis', aspect='auto')
                axes[i, 0].set_title(f'Signal (Test {idx})')
                axes[i, 0].set_xlabel('b1 index')
                axes[i, 0].set_ylabel('b2 index')

                # True distribution
                x = torch.from_numpy(X_inputs[idx:idx+1]).float().to(self.device)
                y_pred = model(x).cpu().numpy()[0, 0]
                vmax = max(float(y_test[idx, 0].max()), float(y_pred.max()))
                vmax = max(vmax, 1e-6)

                axes[i, 1].imshow(
                    y_test[idx, 0],
                    cmap='viridis',
                    aspect='auto',
                    vmin=0.0,
                    vmax=vmax
                )
                axes[i, 1].set_title('True Distribution')
                axes[i, 1].set_xlabel('D1 index')
                axes[i, 1].set_ylabel('D2 index')

                # Predicted distribution
                im = axes[i, 2].imshow(
                    y_pred,
                    cmap='viridis',
                    aspect='auto',
                    vmin=0.0,
                    vmax=vmax
                )
                pred_mass = float(y_pred.sum())
                axes[i, 2].set_title(f'Predicted Distribution\n(sum={pred_mass:.3f})')
                axes[i, 2].set_xlabel('D1 index')
                axes[i, 2].set_ylabel('D2 index')
                plt.colorbar(im, ax=axes[i, 2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison plot to {save_path}")

        plt.close()

    def plot_training_history(self, history: dict, save_path: str = None):
        """Plot training and validation loss curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        def ema(values, alpha=0.25):
            smoothed = []
            running = None
            for value in values:
                running = value if running is None else alpha * value + (1 - alpha) * running
                smoothed.append(running)
            return smoothed

        # Loss
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].plot(ema(history['val_loss']), label='Val Loss (EMA)', linewidth=2, linestyle='--')
        best_epoch = history.get('best_epoch')
        if best_epoch is not None:
            axes[0].axvline(best_epoch - 1, color='grey', linestyle=':', linewidth=1.5, label=f'Best Epoch ({best_epoch})')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Learning rate
        axes[1].plot(history['lr'], label='Learning Rate', linewidth=2, color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('LR')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved training history plot to {save_path}")

        plt.close()

    def compare_models(
        self,
        datasets: dict,
        model_configs: list = None
    ) -> dict:
        """
        Compare multiple models.

        Args:
            datasets: Dictionary with train/val/test data
            model_configs: List of model config dicts

        Returns:
            Dictionary with comparison results
        """
        if model_configs is None:
            model_configs = [
                {'name': 'attention_unet', 'epochs': 60, 'batch_size': 8, 'base_filters': 32, 'learning_rate': 5e-4, 'early_stopping_patience': 12},
                {'name': 'multiscale_unet', 'epochs': 60, 'batch_size': 8, 'base_filters': 32, 'learning_rate': 5e-4, 'early_stopping_patience': 12},
            ]

        results = {}

        for config in model_configs:
            model_name = config['name']
            epochs = config.get('epochs', 50)

            print(f"\n{'='*60}")
            print(f"Training and evaluating: {model_name}")
            print(f"{'='*60}")

            # Train
            model, history = self.train_model(
                model_name=model_name,
                datasets=datasets,
                epochs=epochs,
                batch_size=config.get('batch_size', 8),
                base_filters=config.get('base_filters', 32),
                learning_rate=config.get('learning_rate', 5e-4),
                early_stopping_patience=config.get('early_stopping_patience', 12),
                early_stopping_min_delta=config.get('early_stopping_min_delta', 1e-4),
            )

            # Evaluate
            metrics = self.evaluate_model(model, datasets)

            # Plot training history
            fig_path = os.path.join(
                self.output_dir, "figures",
                f"history_{model_name}.png"
            )
            self.plot_training_history(history, save_path=fig_path)

            results[model_name] = {
                'model': model,
                'history': history,
                'metrics': metrics
            }

            # Save comparison plot
            fig_path = os.path.join(
                self.output_dir, "figures",
                f"comparison_{model_name}.png"
            )
            self.plot_comparison(model, datasets, save_path=fig_path)

        # Summary comparison
        self._print_comparison_summary(results)

        return results

    def _print_comparison_summary(self, results: dict):
        """Print summary comparison of models."""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Model':<25} {'MSE':<15} {'MAE':<15} {'RMSE':<15}")
        print("-" * 60)

        for name, data in results.items():
            m = data['metrics']
            print(f"{name:<25} {m['mse']:<15.6f} {m['mae']:<15.6f} {m['rmse']:<15.6f}")

        print("=" * 60)


def main():
    """Main training script."""
    print("=" * 60)
    print("2D DEXSY Inversion - Training Pipeline (PyTorch)")
    print("=" * 60)

    # Initialize pipeline
    pipeline = TrainingPipeline2D(
        output_dir="training_output_2d",
        n_d=64,
        n_b=64,
        seed=42
    )

    # Generate data
    datasets = pipeline.generate_dataset(
        n_train=9500,
        n_val=400,
        n_test=100,
        noise_levels=None,
        noise_sigma_range=(0.005, 0.015),
        n_compartments=2,
    )

    # Define models to compare
    model_configs = [
        {'name': 'attention_unet', 'epochs': 60, 'batch_size': 8, 'base_filters': 32, 'learning_rate': 5e-4, 'early_stopping_patience': 12},
    ]

    # Train and compare
    results = pipeline.compare_models(datasets, model_configs)

    # Evaluate best model
    best_model_name = min(results.keys(), key=lambda k: results[k]['metrics']['mse'])
    best_model = results[best_model_name]['model']

    print(f"\nBest model: {best_model_name}")

    # Final evaluation with plots
    final_metrics = pipeline.evaluate_model(best_model, datasets)
    pipeline.plot_comparison(
        best_model, datasets,
        n_examples=5,
        save_path=os.path.join(pipeline.output_dir, "figures", "final_comparison.png")
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

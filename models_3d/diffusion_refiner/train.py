"""
Training script for Diffusion Refiner on 3C Model.

This module implements the training pipeline for the conditional diffusion
refiner that learns to refine UNet baseline predictions.

Usage:
    python -m models_3d.diffusion_refiner.train

Or import and use programmatically:
    from models_3d.diffusion_refiner import train_refiner

    model, history = train_refiner(
        baseline_model=unet_model,
        n_train=9500,
        n_val=400,
        epochs=100,
    )
"""

from __future__ import annotations

import os
import time
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dexsy_core.forward_model import ForwardModel2D
from dexsy_core.metrics import compute_dei

from .config import RefinerConfig, DiffusionConfig, ModelConfig, TrainingConfig
from .scheduler import DDIMScheduler
from .model import ConditionalUNetDenoiser, RefinerWithBaseline
from .dataset import RefinementDataset, generate_refinement_dataset
from .loss import RefinementLoss, create_loss


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_refined_spectrum(
    model: nn.Module,
    scheduler: DDIMScheduler,
    x_t: torch.Tensor,
    condition: torch.Tensor,
    num_steps: int = 50,
) -> torch.Tensor:
    """
    Compute refined spectrum by denoising x_t to x_0.

    Args:
        model: Denoiser model
        scheduler: Diffusion scheduler
        x_t: Starting noisy input [B, 1, H, W]
        condition: Conditioning information [B, 5, H, W]
        num_steps: Number of DDIM steps

    Returns:
        Refined spectrum [B, 1, H, W]
    """
    timesteps = torch.linspace(
        scheduler.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=x_t.device
    )

    x_current = x_t
    for i, t in enumerate(timesteps):
        t_batch = torch.tensor([t] * x_t.shape[0], device=x_t.device)

        with torch.no_grad():
            noise_pred = model(x_current, condition, t_batch)

        prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else 0
        x_current = scheduler.ddim_step(
            noise_pred, t.item(), prev_t, x_current, clip_denoised=True
        )

    return x_current


def normalize_distribution(x: torch.Tensor) -> torch.Tensor:
    """Normalize to valid probability distribution."""
    x_softplus = torch.nn.functional.softplus(x)
    return x_softplus / (x_softplus.sum(dim=(2, 3), keepdim=True) + 1e-8)


def evaluate_on_batch(
    model: nn.Module,
    scheduler: DDIMScheduler,
    batch: dict,
    device: str,
    num_eval_steps: int = 20,
) -> dict:
    """
    Evaluate model on a single batch.

    Args:
        model: Refiner model
        scheduler: Diffusion scheduler
        batch: Batch dictionary
        device: Device
        num_eval_steps: Number of denoising steps for evaluation

    Returns:
        Dictionary of metrics
    """
    model.eval()

    f_base = batch['f_base'].to(device)
    model_input = batch['model_input'].to(device)
    f_gt = batch['f_gt'].cpu().numpy()
    signal = batch['signal'].cpu().numpy()

    condition = torch.cat([
        f_base, model_input[:, 0:1], model_input[:, 1:2], model_input[:, 2:3], f_base
    ], dim=1)

    x_t = torch.randn_like(f_base)

    with torch.no_grad():
        f_refined = compute_refined_spectrum(
            model, scheduler, x_t, condition, num_steps=num_eval_steps
        )

    f_refined_np = f_refined.cpu().numpy()
    f_base_np = f_base.cpu().numpy()

    metrics = {}

    for name, spectra in [('refined', f_refined_np), ('baseline', f_base_np)]:
        spectra_clipped = np.clip(spectra, 0, None)
        spectra_clipped = spectra_clipped / (spectra_clipped.sum(axis=(2, 3), keepdims=True) + 1e-8)

        mse = np.mean((spectra_clipped - f_gt) ** 2)
        mae = np.mean(np.abs(spectra_clipped - f_gt))

        dei_pred = [compute_dei(s[0]) for s in spectra_clipped]
        dei_true = [compute_dei(s[0]) for s in f_gt]
        dei_error = np.abs(np.array(dei_pred) - np.array(dei_true))

        metrics[name] = {
            'mse': mse,
            'mae': mae,
            'dei_error_mean': dei_error.mean(),
            'dei_error_p95': np.percentile(dei_error, 95),
        }

    return metrics


def train_refiner(
    baseline_model: nn.Module,
    forward_model: ForwardModel2D,
    output_dir: str = None,
    n_train: int = 9500,
    n_val: int = 400,
    n_test: int = 100,
    epochs: int = 100,
    batch_size: int = 16,
    base_filters: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    num_timesteps: int = 1000,
    sampling_steps: int = 50,
    weight_physics: float = 0.05,
    weight_residual: float = 0.1,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 1e-5,
    reduce_lr_patience: int = 7,
    reduce_lr_factor: float = 0.5,
    noise_sigma_range: Tuple[float, float] = (0.005, 0.015),
    n_compartments: int = 3,
    seed: int = 42,
    device: str = None,
    baseline_checkpoint_path: str = None,
    resume_checkpoint_path: str = None,
    grad_clip: float = 1.0,
    val_frequency: int = 5,
    save_frequency: int = 10,
) -> Tuple[nn.Module, dict]:
    """
    Train the diffusion refiner.

    Args:
        baseline_model: Pre-trained UNet model (frozen)
        forward_model: ForwardModel2D instance
        output_dir: Directory for saving outputs
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        epochs: Maximum epochs
        batch_size: Batch size
        base_filters: Base filters for refiner
        learning_rate: Initial learning rate
        weight_decay: Weight decay
        num_timesteps: Diffusion timesteps
        sampling_steps: DDIM sampling steps
        weight_physics: Weight for physics loss
        weight_residual: Weight for residual loss
        early_stopping_patience: Early stopping patience
        early_stopping_min_delta: Minimum improvement for early stopping
        reduce_lr_patience: Reduce LR patience
        reduce_lr_factor: Reduce LR factor
        noise_sigma_range: Noise sigma range
        n_compartments: Number of compartments
        seed: Random seed
        device: Device ('cuda', 'cpu', or None for auto)
        baseline_checkpoint_path: Path to load baseline weights
        resume_checkpoint_path: Path to resume training
        grad_clip: Gradient clipping value
        val_frequency: Validation frequency (every N epochs)
        save_frequency: Checkpoint save frequency

    Returns:
        (model, history)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "training_output_3d" / "diffusion_refiner"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print(f"Using device: {device}")
    set_seed(seed)

    print("\n" + "=" * 60)
    print("Initializing Diffusion Refiner Training")
    print("=" * 60)

    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    for param in baseline_model.parameters():
        param.requires_grad = False

    if baseline_checkpoint_path and os.path.exists(baseline_checkpoint_path):
        print(f"Loading baseline from {baseline_checkpoint_path}")
        checkpoint = torch.load(baseline_checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        baseline_model.load_state_dict(state_dict)

    print("\nGenerating refinement datasets...")
    datasets = generate_refinement_dataset(
        forward_model=forward_model,
        baseline_model=baseline_model,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        noise_sigma_range=noise_sigma_range,
        n_compartments=n_compartments,
        device=device,
        batch_size=64,
        seed=seed,
    )

    train_dataset = datasets['train']
    val_dataset = datasets['val']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    diffusion_config = DiffusionConfig(
        num_timesteps=num_timesteps,
        sampling_steps=sampling_steps,
    )
    scheduler = DDIMScheduler(diffusion_config)
    scheduler.setup(device)

    model = ConditionalUNetDenoiser(
        base_filters=base_filters,
        time_dim=128,
        in_channels=1,
        cond_channels=5,
    ).to(device)

    print(f"\nRefiner Model: ConditionalUNetDenoiser")
    print(f"  Base filters: {base_filters}")
    print(f"  Time embedding dim: 128")
    print(f"  Conditioning channels: 5")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        print(f"Loading refiner from {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    criterion = RefinementLoss(
        forward_model=forward_model,
        weight_physics=weight_physics,
        weight_residual=weight_residual,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=1e-6,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = output_dir / f"checkpoints_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    history = {
        'train_loss': [],
        'train_loss_noise': [],
        'train_loss_physics': [],
        'train_loss_residual': [],
        'val_loss': [],
        'val_dei_error_p95': [],
        'lr': [],
        'best_epoch': None,
    }

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight physics: {weight_physics}")
    print(f"  Weight residual: {weight_residual}")
    print(f"  Gradient clip: {grad_clip}")
    print(f"  Diffusion timesteps: {num_timesteps}")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        total_loss = 0.0
        total_loss_noise = 0.0
        total_loss_physics = 0.0
        total_loss_residual = 0.0
        n_batches = 0

        for batch in train_loader:
            signal = batch['signal'].to(device)
            f_base = batch['f_base'].to(device)
            f_gt = batch['f_gt'].to(device)
            model_input = batch['model_input'].to(device)

            t = torch.randint(
                0, num_timesteps, (signal.shape[0],), device=device
            )

            noise = torch.randn_like(f_gt)

            sqrt_alpha_bar_t = scheduler.sqrt_alphas_bar[t][:, None, None, None]
            sqrt_one_minus_alpha_bar_t = scheduler.sqrt_one_minus_alphas_bar[t][:, None, None, None]

            x_t = sqrt_alpha_bar_t * f_gt + sqrt_one_minus_alpha_bar_t * noise

            condition = torch.cat([
                f_base,
                model_input[:, 0:1],
                model_input[:, 1:2],
                model_input[:, 2:3],
                f_base,
            ], dim=1)

            noise_pred = model(x_t, condition, t)

            with torch.no_grad():
                x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / (sqrt_alpha_bar_t + 1e-8)
                x_0_pred = normalize_distribution(x_0_pred)

            loss_dict = criterion(
                noise_pred=noise_pred,
                noise_true=noise,
                f_refined=x_0_pred,
                f_base=f_base,
                signal=signal,
            )

            loss = loss_dict['loss']

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_loss_noise += loss_dict['loss_noise'].item()
            total_loss_physics += loss_dict['loss_physics'].item()
            total_loss_residual += loss_dict['loss_residual'].item()
            n_batches += 1

        train_loss = total_loss / n_batches
        train_loss_noise = total_loss_noise / n_batches
        train_loss_physics = total_loss_physics / n_batches
        train_loss_residual = total_loss_residual / n_batches

        history['train_loss'].append(train_loss)
        history['train_loss_noise'].append(train_loss_noise)
        history['train_loss_physics'].append(train_loss_physics)
        history['train_loss_residual'].append(train_loss_residual)

        val_loss = 0.0
        val_dei_p95 = 0.0

        if (epoch + 1) % val_frequency == 0 or epoch == 0:
            model.eval()
            total_val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    signal = batch['signal'].to(device)
                    f_base = batch['f_base'].to(device)
                    f_gt = batch['f_gt'].to(device)
                    model_input = batch['model_input'].to(device)

                    t = torch.randint(
                        0, num_timesteps, (signal.shape[0],), device=device
                    )

                    noise = torch.randn_like(f_gt)

                    sqrt_alpha_bar_t = scheduler.sqrt_alphas_bar[t][:, None, None, None]
                    sqrt_one_minus_alpha_bar_t = scheduler.sqrt_one_minus_alphas_bar[t][:, None, None, None]

                    x_t = sqrt_alpha_bar_t * f_gt + sqrt_one_minus_alpha_bar_t * noise

                    condition = torch.cat([
                        f_base,
                        model_input[:, 0:1],
                        model_input[:, 1:2],
                        model_input[:, 2:3],
                        f_base,
                    ], dim=1)

                    noise_pred = model(x_t, condition, t)

                    x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / (sqrt_alpha_bar_t + 1e-8)
                    x_0_pred = normalize_distribution(x_0_pred)

                    loss_dict = criterion(
                        noise_pred=noise_pred,
                        noise_true=noise,
                        f_refined=x_0_pred,
                        f_base=f_base,
                        signal=signal,
                    )

                    total_val_loss += loss_dict['loss'].item()
                    n_val_batches += 1

            val_loss = total_val_loss / n_val_batches
            history['val_loss'].append(val_loss)

            eval_metrics = evaluate_on_batch(
                model, scheduler, next(iter(val_loader)), device, num_eval_steps=20
            )
            val_dei_p95 = eval_metrics['refined']['dei_error_p95']
            history['val_dei_error_p95'].append(val_dei_p95)

            scheduler_lr.step(val_loss)

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
                    'val_dei_p95': val_dei_p95,
                    'config': {
                        'base_filters': base_filters,
                        'time_dim': 128,
                    },
                }, model_dir / "best_model.pt")
            else:
                patience_counter += 1

        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} (n:{train_loss_noise:.4f}, p:{train_loss_physics:.4f}, r:{train_loss_residual:.4f}) | "
                  f"Val: {val_loss:.4f} | "
                  f"DEI p95: {val_dei_p95:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")

        if (epoch + 1) % save_frequency == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, model_dir / f"checkpoint_epoch_{epoch+1}.pt")

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds ({train_time/60:.1f} minutes)")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {history['best_epoch']}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': {
            'base_filters': base_filters,
            'time_dim': 128,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_physics': weight_physics,
            'weight_residual': weight_residual,
        }
    }, model_dir / "final_model.pt")

    import csv
    log_path = model_dir / "training_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['epoch', 'train_loss', 'train_loss_noise', 'train_loss_physics',
                   'train_loss_residual', 'val_loss', 'val_dei_p95', 'lr']
        writer.writerow(headers)
        for i in range(len(history['train_loss'])):
            dei_p95 = history['val_dei_error_p95'][i] if i < len(history['val_dei_error_p95']) else ''
            writer.writerow([
                i+1,
                history['train_loss'][i],
                history['train_loss_noise'][i],
                history['train_loss_physics'][i],
                history['train_loss_residual'][i],
                history['val_loss'][i] if i < len(history['val_loss']) else '',
                dei_p95,
                history['lr'][i],
            ])

    print(f"Saved checkpoints to {model_dir}")

    return model, history, forward_model


def main():
    """Main training script."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Diffusion Refiner on 3C DEXSY")
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--baseline_checkpoint', type=str, default=None, help='Baseline UNet checkpoint')
    parser.add_argument('--n_train', type=int, default=9500, help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=400, help='Number of validation samples')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--base_filters', type=int, default=32, help='Base filters')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--compartments', type=int, default=3, help='Number of compartments')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    forward_model = ForwardModel2D(n_d=64, n_b=64)

    from models_3d.attention_unet.model import AttentionUNet3C
    baseline_model = AttentionUNet3C(in_channels=3, base_filters=32)

    if args.baseline_checkpoint:
        checkpoint = torch.load(args.baseline_checkpoint, map_location=device)
        baseline_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))

    train_refiner(
        baseline_model=baseline_model,
        forward_model=forward_model,
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_filters=args.base_filters,
        learning_rate=args.lr,
        n_compartments=args.compartments,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

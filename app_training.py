#!/usr/bin/env python3
"""
DEXSY Training Interface

A Gradio-based interface for training DEXSY models with:
- Model selection (Attention U-Net, Plain U-Net, Deep Unfolding, FNO)
- Grid size configuration (16x16 or 64x64)
- Hyperparameter tuning
- Real-time training progress
- Post-training visualization (loss curves, sample predictions)
- Benchmark comparison with other models
"""

from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import gradio as gr
import matplotlib.pyplot as plt
import torch

from dexsy_core.forward_model import create_forward_model, ForwardModel2D
from dexsy_core.preprocessing import build_model_inputs
from dexsy_core.metrics import compute_batch_metrics
from improved_2d_dexsy import (
    OTHER_MODELS_DIR,
    CHECKPOINTS_DIR,
    CHECKPOINTS_DIR_3D,
    DEXSYInferencePipeline,
    list_available_checkpoints,
    list_available_checkpoints_3d,
    to_serializable,
)

# Visualization constants
DISPLAY_CMAP = "viridis"
DISPLAY_SPECTRUM_CMAP = "magma"

# Model grid support mapping
MODEL_GRID_SUPPORT = {
    "attention_unet": [16, 64],
    "plain_unet": [16, 64],
    "deep_unfolding": [16, 64],
    "fno": [16, 64],
}


def _plot_training_curves(history: dict) -> plt.Figure:
    """Plot training and validation loss curves."""
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Linear scale
    if history.get('train_loss'):
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if history.get('val_loss'):
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Right: Log scale
    if history.get('train_loss'):
        axes[1].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
    if history.get('val_loss'):
        axes[1].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='s', markersize=3)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss (log scale)', fontsize=12)
    axes[1].set_title('Loss Curves (Log Scale)', fontsize=14)
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def _plot_sample_predictions(
    signals: list,
    ground_truths: list,
    predictions: list | None = None,
    n_samples: int = 3,
) -> plt.Figure:
    """Plot sample predictions: signal input, ground truth, and model prediction."""
    n_samples = min(n_samples, len(signals))
    
    if predictions is None:
        fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
    else:
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        sig = signals[i]
        gt = ground_truths[i]
        
        # Signal Input
        axes[i, 0].imshow(sig, cmap=DISPLAY_CMAP, origin='lower')
        axes[i, 0].set_title(f'Sample {i+1}: Signal Input', fontsize=11)
        axes[i, 0].set_xlabel('b2 index')
        axes[i, 0].set_ylabel('b1 index')
        
        # Ground Truth
        vmax = float(np.max(gt))
        axes[i, 1].imshow(gt, cmap=DISPLAY_SPECTRUM_CMAP, origin='lower', vmin=0, vmax=vmax)
        axes[i, 1].set_title(f'Sample {i+1}: Ground Truth', fontsize=11)
        axes[i, 1].set_xlabel('D2 index')
        axes[i, 1].set_ylabel('D1 index')
        
        # Prediction (if available)
        if predictions is not None:
            pred = predictions[i]
            axes[i, 2].imshow(pred, cmap=DISPLAY_SPECTRUM_CMAP, origin='lower', vmin=0, vmax=vmax)
            axes[i, 2].set_title(f'Sample {i+1}: Prediction', fontsize=11)
            axes[i, 2].set_xlabel('D2 index')
            axes[i, 2].set_ylabel('D1 index')
        else:
            axes[i, 2].text(0.5, 0.5, 'No prediction available', 
                           ha='center', va='center', transform=axes[i, 2].transAxes)
            axes[i, 2].set_title(f'Sample {i+1}: Prediction', fontsize=11)
    
    fig.tight_layout()
    return fig


def _plot_benchmark_results(results: dict) -> plt.Figure:
    """Plot benchmark comparison results."""
    model_names = list(results.keys())
    metrics = ['MSE', 'MAE', 'DEI_Error']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        values = [results[m].get(metric.lower(), 0) for m in model_names]
        colors = ['green' if 'Trained' in m else 'steelblue' for m in model_names]
        
        axes[idx].barh(model_names, values, color=colors)
        axes[idx].set_xlabel(metric, fontsize=12)
        axes[idx].set_title(f'{metric} Comparison', fontsize=14)
        axes[idx].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for j, v in enumerate(values):
            axes[idx].text(v + max(values) * 0.01, j, f'{v:.4f}', va='center', fontsize=9)
    
    fig.tight_layout()
    return fig


def generate_samples(
    grid_size: int,
    n_compartments: int,
    n_samples: int = 3,
    seed: int = 42,
) -> tuple[list, list]:
    """Generate sample signals and ground truths."""
    np.random.seed(seed)
    fm = create_forward_model(n_d=grid_size, n_b=grid_size)
    
    signals, ground_truths = [], []
    
    for _ in range(n_samples):
        if n_compartments == 2:
            gt, sig, _ = fm.generate_2compartment_paper()
        else:
            gt, sig, _ = fm.generate_3compartment_paper()
        signals.append(sig.astype(np.float32))
        ground_truths.append(gt.astype(np.float32))
    
    return signals, ground_truths


def run_training(
    model_type: str,
    grid_size: int,
    n_compartments: int,
    custom_name: str,
    n_train: int,
    n_val: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    early_stopping_patience: int,
    progress_callback=None,
) -> dict:
    """
    Run model training and return results.
    
    Args:
        model_type: Model architecture ('attention_unet', 'plain_unet', etc.)
        grid_size: Grid size (16 or 64)
        n_compartments: Number of compartments (2 or 3)
        custom_name: User-defined name for the trained model
        n_train: Number of training samples
        n_val: Number of validation samples
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        early_stopping_patience: Early stopping patience
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with training results
    """
    import random
    
    # Create output directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = custom_name.replace(' ', '_').replace('/', '_')
    output_dir = OTHER_MODELS_DIR / f"{safe_name}_{grid_size}x{grid_size}_{n_compartments}c_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize forward model
    fm = create_forward_model(n_d=grid_size, n_b=grid_size)
    
    # Generate datasets
    if progress_callback:
        progress_callback(0.05, "Generating training data...")
    
    datasets = {}
    for split_name, n_samples in [("train", n_train), ("val", n_val), ("test", 100)]:
        F, S, _, S_clean = fm.generate_batch(
            n_samples=n_samples,
            noise_sigma=None,
            noise_sigma_range=(0.005, 0.015),
            n_compartments=n_compartments,
            normalize=True,
            noise_model='rician',
            return_reference_signal=True,
        )
        
        S = S.reshape(-1, 1, grid_size, grid_size).astype(np.float32)
        S_clean = S_clean.reshape(-1, 1, grid_size, grid_size).astype(np.float32)
        F = F.reshape(-1, 1, grid_size, grid_size).astype(np.float32)
        inputs = build_model_inputs(S, fm)
        
        datasets[split_name] = {
            'inputs': inputs,
            'signals': S,
            'clean_signals': S_clean,
            'labels': F,
        }
    
    if progress_callback:
        progress_callback(0.15, "Data generation complete. Initializing model...")
    
    # Create model based on type
    if model_type == 'attention_unet':
        from models_2d.attention_unet.model import AttentionUNet2D, PhysicsInformedLoss
        
        model = AttentionUNet2D(
            in_channels=datasets['train']['inputs'].shape[1],
            base_filters=32,
        ).to(device)
        
        criterion = PhysicsInformedLoss(
            forward_model=fm,
            alpha_kl=1.0,
            alpha_rec=0.2,
            alpha_signal=0.1,
            alpha_sum=0.05,
            peak_weight=6.0,
            alpha_smooth=2e-2,
        ).to(device)
        
    elif model_type == 'plain_unet':
        from models_2d.plain_unet.model import PlainUNet2D, PlainUNetLoss
        
        model = PlainUNet2D(
            in_channels=datasets['train']['inputs'].shape[1],
            base_filters=32,
        ).to(device)
        
        criterion = PlainUNetLoss(
            alpha_smooth=0.01,
            peak_weight=4.0,
        ).to(device)
        
    elif model_type == 'deep_unfolding':
        from models_2d.deep_unfolding.model import DeepUnfolding2D
        from models_2d.deep_unfolding.train import DeepUnfoldingLoss
        
        # Deep unfolding uses signal-only input (1 channel)
        # Extract signal channel from multi-channel input
        signal_channel = datasets['train']['inputs'][:, 0:1, :, :]  # Take first channel (signal)
        
        model = DeepUnfolding2D(
            n_d=grid_size,
            use_denoiser=True,
        ).to(device)
        
        # Set kernel matrix for the model
        K_tensor = torch.from_numpy(fm.kernel_matrix.astype(np.float32)).to(device)
        model.set_kernel_matrix(K_tensor)
        
        criterion = DeepUnfoldingLoss(
            forward_kernel=K_tensor,
            n_b=grid_size,
            alpha_kl=1.0,
            alpha_recon=0.2,
            alpha_consistency=0.1,
            alpha_sum=0.05,
            alpha_smooth=2e-2,
            peak_weight=6.0,
        ).to(device)
        
        # Override datasets to use signal-only for deep unfolding
        datasets['train']['inputs'] = signal_channel
        datasets['val']['inputs'] = datasets['val']['inputs'][:, 0:1, :, :]
        datasets['test']['inputs'] = datasets['test']['inputs'][:, 0:1, :, :]
        
    elif model_type == 'fno':
        from models_2d.neural_operators.fno import FNO2D
        from models_2d.neural_operators.train import PhysicsInformedNeuralOperatorLoss
        
        model = FNO2D(
            in_channels=datasets['train']['inputs'].shape[1],
            hidden_channels=64,
            n_layers=4,
            modes=16,
        ).to(device)
        
        criterion = PhysicsInformedNeuralOperatorLoss(
            forward_model=fm,
            alpha_kl=1.0,
            alpha_rec=0.2,
            alpha_signal=0.1,
            alpha_smooth=2e-2,
            alpha_sum=0.05,
            peak_weight=6.0,
        ).to(device)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Data loaders
    from torch.utils.data import DataLoader, TensorDataset
    
    train_dataset = TensorDataset(
        torch.from_numpy(datasets['train']['inputs']),
        torch.from_numpy(datasets['train']['labels']),
        torch.from_numpy(datasets['train']['clean_signals']),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(datasets['val']['inputs']),
        torch.from_numpy(datasets['val']['labels']),
        torch.from_numpy(datasets['val']['clean_signals']),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    if progress_callback:
        progress_callback(0.2, "Starting training...")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for inputs, labels, clean_signals in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            clean_signals = clean_signals.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Call loss with appropriate arguments based on model type
            if model_type == 'plain_unet':
                loss_output = criterion(outputs, labels)
            elif model_type in ('attention_unet', 'fno'):
                loss_output = criterion(outputs, labels, clean_signals)
            elif model_type == 'deep_unfolding':
                loss_output = criterion(outputs, labels, clean_signals)
            
            # Handle both tensor and dict return types
            # Keys can be 'total' or 'total_loss' depending on the loss function
            if isinstance(loss_output, dict):
                loss = loss_output.get('total_loss') or loss_output.get('total')
            else:
                loss = loss_output
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= n_batches
        
        # Validate
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for inputs, labels, clean_signals in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                clean_signals = clean_signals.to(device)
                
                outputs = model(inputs)
                
                # Call loss with appropriate arguments based on model type
                if model_type == 'plain_unet':
                    loss_output = criterion(outputs, labels)
                elif model_type in ('attention_unet', 'fno'):
                    loss_output = criterion(outputs, labels, clean_signals)
                elif model_type == 'deep_unfolding':
                    loss_output = criterion(outputs, labels, clean_signals)
                
                # Handle both tensor and dict return types
                # Keys can be 'total' or 'total_loss' depending on the loss function
                if isinstance(loss_output, dict):
                    val_loss_item = loss_output.get('total_loss') or loss_output.get('total')
                else:
                    val_loss_item = loss_output
                
                val_loss += val_loss_item.item()
                n_val_batches += 1
        
        val_loss /= n_val_batches
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Update best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Progress update
        progress_pct = 0.2 + (epoch + 1) / epochs * 0.6
        if progress_callback:
            progress_callback(
                progress_pct,
                f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}"
            )
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            if progress_callback:
                progress_callback(progress_pct, f"Early stopping at epoch {epoch+1}")
            break
    
    # Save best model
    if progress_callback:
        progress_callback(0.85, "Saving model checkpoint...")
    
    model.load_state_dict(best_model_state)
    
    checkpoint_path = output_dir / "best_model.pt"
    torch.save({
        'model_state_dict': best_model_state,
        'config': {
            'model_type': model_type,
            'grid_size': grid_size,
            'n_compartments': n_compartments,
            'custom_name': custom_name,
            'n_d': grid_size,
            'n_b': grid_size,
            'in_channels': datasets['train']['inputs'].shape[1],
        },
        'history': history,
        'best_val_loss': best_val_loss,
    }, checkpoint_path)
    
    # Save training config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'model_type': model_type,
            'grid_size': grid_size,
            'n_compartments': n_compartments,
            'custom_name': custom_name,
            'n_train': n_train,
            'n_val': n_val,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'early_stopping_patience': early_stopping_patience,
            'best_val_loss': float(best_val_loss),
        }, f, indent=2)
    
    # Generate sample predictions
    if progress_callback:
        progress_callback(0.90, "Generating sample predictions...")
    
    model.eval()
    test_signals = [datasets['test']['signals'][i, 0] for i in range(3)]
    test_gts = [datasets['test']['labels'][i, 0] for i in range(3)]
    test_inputs = datasets['test']['inputs'][:3]
    
    predictions = []
    with torch.no_grad():
        for i in range(3):
            inp = torch.from_numpy(test_inputs[i:i+1]).to(device)
            pred = model(inp)
            predictions.append(pred.cpu().numpy()[0, 0])
    
    # Save training history plot
    if progress_callback:
        progress_callback(0.95, "Saving visualizations...")
    
    fig_curves = _plot_training_curves(history)
    fig_curves.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close(fig_curves)
    
    fig_samples = _plot_sample_predictions(test_signals, test_gts, predictions)
    fig_samples.savefig(output_dir / "sample_predictions.png", dpi=150, bbox_inches='tight')
    plt.close(fig_samples)
    
    if progress_callback:
        progress_callback(1.0, "Training complete!")
    
    return {
        'output_dir': str(output_dir),
        'checkpoint_path': str(checkpoint_path),
        'history': history,
        'best_val_loss': float(best_val_loss),
        'test_signals': [s.tolist() for s in test_signals],
        'test_ground_truths': [gt.tolist() for gt in test_gts],
        'predictions': [p.tolist() for p in predictions],
        'config': {
            'model_type': model_type,
            'grid_size': grid_size,
            'n_compartments': n_compartments,
            'custom_name': custom_name,
        },
    }


def run_benchmark(
    trained_checkpoint: str,
    model_type: str,
    grid_size: int,
    n_compartments: int,
    n_test_samples: int = 100,
) -> dict:
    """Run benchmark comparison between trained model and baseline models."""
    
    # Generate test dataset
    fm = create_forward_model(n_d=grid_size, n_b=grid_size)
    
    signals, ground_truths = [], []
    for _ in range(n_test_samples):
        if n_compartments == 2:
            gt, sig, _ = fm.generate_2compartment_paper()
        else:
            gt, sig, _ = fm.generate_3compartment_paper()
        signals.append(sig.astype(np.float32))
        ground_truths.append(gt.astype(np.float32))
    
    # Load trained model
    checkpoint = torch.load(trained_checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})
    
    # Create inference pipeline for trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine model class based on model_type
    if model_type == 'attention_unet':
        from models_2d.attention_unet.model import AttentionUNet2D
        in_channels = config.get('in_channels', 3)
        trained_model = AttentionUNet2D(in_channels=in_channels, base_filters=32)
        use_multi_channel = True
    elif model_type == 'plain_unet':
        from models_2d.plain_unet.model import PlainUNet2D
        in_channels = config.get('in_channels', 3)
        trained_model = PlainUNet2D(in_channels=in_channels, base_filters=32)
        use_multi_channel = True
    elif model_type == 'deep_unfolding':
        from models_2d.deep_unfolding.model import DeepUnfolding2D
        trained_model = DeepUnfolding2D(n_d=grid_size, use_denoiser=True)
        use_multi_channel = False  # Uses signal-only input
    elif model_type == 'fno':
        from models_2d.neural_operators.fno import FNO2D
        in_channels = config.get('in_channels', 3)
        trained_model = FNO2D(
            in_channels=in_channels,
            hidden_channels=64,
            n_layers=4,
            modes=16,
        )
        use_multi_channel = True
    else:
        raise ValueError(f"Unsupported model type for benchmark: {model_type}")
    
    trained_model.load_state_dict(checkpoint['model_state_dict'])
    trained_model = trained_model.to(device)
    
    # Set kernel matrix for deep unfolding model
    if model_type == 'deep_unfolding':
        K_tensor = torch.from_numpy(fm.kernel_matrix.astype(np.float32)).to(device)
        trained_model.set_kernel_matrix(K_tensor)
    
    trained_model.eval()
    
    # Run inference on trained model
    trained_preds = []
    with torch.no_grad():
        for sig in signals:
            if use_multi_channel:
                inp = build_model_inputs(sig.reshape(1, 1, grid_size, grid_size).astype(np.float32), fm)
            else:
                # Deep unfolding uses signal-only input
                inp = sig.reshape(1, 1, grid_size, grid_size).astype(np.float32)
            inp_tensor = torch.from_numpy(inp).to(device)
            pred = trained_model(inp_tensor)
            trained_preds.append(pred.cpu().numpy()[0, 0].astype(np.float32))
    
    # Compute aggregate metrics over the benchmark batch
    trained_batch_metrics = compute_batch_metrics(
        np.array(ground_truths),
        np.array(trained_preds),
    )
    trained_metrics = {
        "mse": trained_batch_metrics["aggregate"]["mse_mean"],
        "mae": trained_batch_metrics["aggregate"]["mae_mean"],
        "dei_error": trained_batch_metrics["aggregate"]["dei_error_mean"],
        "mse_std": trained_batch_metrics["aggregate"]["mse_std"],
        "mae_std": trained_batch_metrics["aggregate"]["mae_std"],
        "dei_error_std": trained_batch_metrics["aggregate"]["dei_error_std"],
    }
    
    results = {f"Trained ({model_type})": trained_metrics}
    
    # Run inference on baseline models (same grid size)
    baseline_models = _get_baseline_models_for_grid(model_type, grid_size, n_compartments)
    
    for name, pipeline in baseline_models.items():
        preds = []
        try:
            with torch.no_grad():
                for sig in signals:
                    result = pipeline.predict(sig)
                    # Extract spectrum from result object
                    pred = getattr(result, 'reconstructed_spectrum', result)
                    if hasattr(pred, 'numpy'):
                        pred = pred.cpu().numpy()
                    preds.append(pred)

            batch_metrics = compute_batch_metrics(np.array(ground_truths), np.array(preds))
            results[name] = {
                "mse": batch_metrics["aggregate"]["mse_mean"],
                "mae": batch_metrics["aggregate"]["mae_mean"],
                "dei_error": batch_metrics["aggregate"]["dei_error_mean"],
                "mse_std": batch_metrics["aggregate"]["mse_std"],
                "mae_std": batch_metrics["aggregate"]["mae_std"],
                "dei_error_std": batch_metrics["aggregate"]["dei_error_std"],
            }
        except Exception as exc:
            print(f"[benchmark] Skip baseline '{name}': {exc}")
            continue
    
    return results


def _get_baseline_models_for_grid(
    trained_model_type: str,
    grid_size: int,
    n_compartments: int,
) -> dict[str, DEXSYInferencePipeline]:
    """Get baseline models for comparison (ALL models with matching grid/compartments)."""
    models = {}
    
    # Map to 2C/3C model names
    if n_compartments == 3:
        model_suffix = "_3c"
    else:
        model_suffix = ""
    
    # Base model names for each architecture type
    base_model_names = ["attention_unet", "plain_unet", "deep_unfolding", "fno"]

    # Include PINN baseline only for 64x64
    if grid_size == 64:
        base_model_names.append("pinn")

    checkpoint_pool = (
        list_available_checkpoints_3d() if n_compartments == 3 else list_available_checkpoints()
    )

    def _checkpoint_grid(path: Path) -> int | None:
        try:
            ckpt = torch.load(path, map_location="cpu")
            cfg = ckpt.get("config", {})
            n_d = cfg.get("n_d") or cfg.get("grid_size")
            if n_d is not None:
                return int(n_d)
        except Exception:
            pass
        return None

    def _pick_checkpoint(model_name: str) -> Path | None:
        target = model_name.lower()
        candidates = []
        for path in checkpoint_pool:
            s = str(path).lower()
            if target not in s:
                continue
            g = _checkpoint_grid(path)
            if g is not None and g != grid_size:
                continue
            # For legacy checkpoints without config, use filename hint as fallback.
            if g is None and grid_size == 16 and "g16" not in s:
                continue
            candidates.append(path)
        if not candidates:
            return None
        return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

    # Always pass an explicit forward model to avoid accidental 64x64 defaults
    # in environments where pipeline grid auto-detection may differ.
    fm = create_forward_model(n_d=grid_size, n_b=grid_size)

    for base_name in base_model_names:
        canonical_name = base_name + model_suffix
        # Prefer explicit g16 registry names for 16x16 checkpoints.
        if grid_size == 16:
            full_name = f"{canonical_name}_g16"
        else:
            full_name = canonical_name
        try:
            checkpoint_path = _pick_checkpoint(canonical_name)
            if checkpoint_path is None:
                continue
            pipeline = DEXSYInferencePipeline(
                model_name=full_name,
                checkpoint_path=str(checkpoint_path),
                forward_model=fm,
                grid_size=grid_size,
            )
            models[full_name] = pipeline
        except Exception:
            # Fallback to canonical name if _g16 alias is unavailable in registry.
            try:
                pipeline = DEXSYInferencePipeline(
                    model_name=canonical_name,
                    checkpoint_path=str(checkpoint_path),
                    forward_model=fm,
                    grid_size=grid_size,
                )
                models[canonical_name] = pipeline
            except Exception:
                pass
    
    return models


def build_app():
    with gr.Blocks(title="DEXSY Training Interface", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # DEXSY Model Training Interface
        
        Train custom DEXSY reconstruction models with your preferred parameters.
        After training, you can compare your model with existing baselines.
        """)
        
        # Training state
        training_state = gr.State(value=None)
        last_checkpoint = gr.State(value=None)
        
        with gr.Tabs():
            # ================================================================
            # TAB 1: TRAINING CONFIGURATION
            # ================================================================
            with gr.TabItem("Training Configuration"):
                gr.Markdown("### Configure Your Training")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Model Settings")
                        
                        model_type = gr.Dropdown(
                            choices=["attention_unet", "plain_unet", "deep_unfolding", "fno"],
                            value="attention_unet",
                            label="Model Architecture",
                            info="Choose the model to train",
                        )
                        
                        n_compartments = gr.Radio(
                            choices=[2, 3],
                            value=2,
                            label="Number of Compartments",
                            info="2 for 2-compartment, 3 for 3-compartment data",
                        )
                        
                        grid_size = gr.Radio(
                            choices=[16, 64],
                            value=64,
                            label="Grid Size",
                            info="16x16 for faster training, 64x64 for higher resolution",
                        )
                        
                        custom_name = gr.Textbox(
                            label="Model Name",
                            placeholder="my_custom_model",
                            value="custom_model",
                            info="Name for your trained model (used in inference)",
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### Data Configuration")
                        
                        n_train = gr.Slider(
                            minimum=1000,
                            maximum=20000,
                            value=9500,
                            step=500,
                            label="Training Samples",
                            info="Number of training samples to generate",
                        )
                        
                        n_val = gr.Slider(
                            minimum=100,
                            maximum=2000,
                            value=400,
                            step=100,
                            label="Validation Samples",
                            info="Number of validation samples",
                        )
                
                gr.Markdown("---")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Training Hyperparameters")
                        
                        epochs = gr.Slider(
                            minimum=10,
                            maximum=200,
                            value=60,
                            step=10,
                            label="Epochs",
                            info="Maximum number of training epochs",
                        )
                        
                        batch_size = gr.Dropdown(
                            choices=[4, 8, 16, 32],
                            value=8,
                            label="Batch Size",
                        )
                        
                        learning_rate = gr.Slider(
                            minimum=1e-5,
                            maximum=1e-2,
                            value=5e-4,
                            step=1e-5,
                            label="Learning Rate",
                            info="Initial learning rate",
                        )
                        
                        early_stopping_patience = gr.Slider(
                            minimum=5,
                            maximum=30,
                            value=12,
                            step=1,
                            label="Early Stopping Patience",
                            info="Stop if no improvement for this many epochs",
                        )
                
                gr.Markdown("---")
                
                with gr.Row():
                    start_training_btn = gr.Button("Start Training", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)
        
            # ================================================================
            # TAB 2: TRAINING PROGRESS
            # ================================================================
            with gr.TabItem("Training Progress"):
                gr.Markdown("### Training Progress")
                
                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Progress (%)",
                    interactive=False,
                )
                
                progress_text = gr.Textbox(
                    label="Status",
                    placeholder="Training not started...",
                    lines=2,
                )
                
                gr.Markdown("---")
                
                with gr.Row():
                    training_curves_plot = gr.Plot(label="Training Curves")
                    sample_predictions_plot = gr.Plot(label="Sample Predictions")
                
                training_metrics = gr.JSON(label="Training Metrics")
            
            # ================================================================
            # TAB 3: BENCHMARK
            # ================================================================
            with gr.TabItem("Benchmark"):
                gr.Markdown("### Compare with Other Models")
                
                gr.Markdown("*Run benchmark to compare your trained model with existing baselines.*")
                
                with gr.Row():
                    benchmark_btn = gr.Button("Run Benchmark", variant="primary")
                    benchmark_status = gr.Textbox(label="Status", lines=2)
                
                with gr.Row():
                    benchmark_plot = gr.Plot(label="Benchmark Results")
                    benchmark_table = gr.JSON(label="Detailed Metrics")
        
        # ================================================================
        # EVENT HANDLERS
        # ================================================================
        
        def start_training(
            model_type,
            grid_size,
            n_compartments,
            custom_name,
            n_train,
            n_val,
            epochs,
            batch_size,
            learning_rate,
            early_stopping_patience,
        ):
            """Start the training process."""
            if not custom_name.strip():
                return (
                    0, "Error: Please enter a model name",
                    None, None, None, None, None
                )
            
            try:
                # Run training with progress callback
                result = {}
                def progress_callback(pct, message):
                    result['progress'] = int(pct * 100)
                    result['message'] = message
                
                training_result = run_training(
                    model_type=model_type,
                    grid_size=grid_size,
                    n_compartments=n_compartments,
                    custom_name=custom_name,
                    n_train=n_train,
                    n_val=n_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    early_stopping_patience=early_stopping_patience,
                    progress_callback=progress_callback,
                )
                
                # Generate plots
                fig_curves = _plot_training_curves(training_result['history'])
                
                fig_samples = _plot_sample_predictions(
                    [np.array(s) for s in training_result['test_signals']],
                    [np.array(gt) for gt in training_result['test_ground_truths']],
                    [np.array(p) for p in training_result['predictions']],
                )
                
                metrics_info = {
                    'model_type': training_result['config']['model_type'],
                    'grid_size': training_result['config']['grid_size'],
                    'n_compartments': training_result['config']['n_compartments'],
                    'best_val_loss': training_result['best_val_loss'],
                    'total_epochs': len(training_result['history']['train_loss']),
                    'checkpoint_path': training_result['checkpoint_path'],
                }
                
                return (
                    100,
                    f"Training complete! Best val loss: {training_result['best_val_loss']:.6f}",
                    fig_curves,
                    fig_samples,
                    metrics_info,
                    training_result['checkpoint_path'],
                    training_result,
                )
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return (
                    0,
                    f"Training failed: {str(e)}",
                    None, None, None, None, None
                )
        
        start_training_btn.click(
            fn=start_training,
            inputs=[
                model_type, grid_size, n_compartments, custom_name,
                n_train, n_val, epochs, batch_size, learning_rate, early_stopping_patience,
            ],
            outputs=[
                progress_bar, progress_text,
                training_curves_plot, sample_predictions_plot,
                training_metrics, last_checkpoint, training_state,
            ],
        )
        
        def run_benchmark_wrapper(state_dict, last_ckpt, n_samples):
            """Run benchmark comparison."""
            if state_dict is None and last_ckpt is None:
                return None, None, "No trained model available. Train a model first."
            
            # Use state_dict config if available, otherwise use last_checkpoint path
            if state_dict is not None:
                config = state_dict.get('config', {})
                checkpoint_path = state_dict.get('checkpoint_path')
                model_type = config.get('model_type', 'attention_unet')
                grid_size = config.get('grid_size', 64)
                n_compartments = config.get('n_compartments', 2)
            else:
                checkpoint_path = last_ckpt
                model_type = 'attention_unet'
                grid_size = 64
                n_compartments = 2
            
            try:
                results = run_benchmark(
                    trained_checkpoint=checkpoint_path,
                    model_type=model_type,
                    grid_size=grid_size,
                    n_compartments=n_compartments,
                    n_test_samples=int(n_samples),
                )
                
                fig = _plot_benchmark_results(results)
                
                return fig, results, "Benchmark complete!"
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, None, f"Benchmark failed: {str(e)}"
        
        benchmark_btn.click(
            fn=run_benchmark_wrapper,
            inputs=[training_state, last_checkpoint, gr.Number(value=100, label="Test Samples")],
            outputs=[benchmark_plot, benchmark_table, benchmark_status],
        )
        
        def clear_training():
            """Clear training state and outputs."""
            return (
                0, "Training not started...",
                None, None, None, None, None, None
            )
        
        clear_btn.click(
            fn=clear_training,
            inputs=[],
            outputs=[
                progress_bar, progress_text,
                training_curves_plot, sample_predictions_plot,
                training_metrics, last_checkpoint, training_state,
            ],
        )
    
    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7861)

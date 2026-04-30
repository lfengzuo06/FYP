"""
Dataset for Diffusion Refiner training.

This module provides PyTorch datasets for training the conditional diffusion
refiner. The dataset includes pre-computed baseline predictions from the
frozen UNet model.

Usage:
    # Generate baselines using frozen UNet
    dataset = RefinementDataset.from_existing_data(
        signals=signals,
        spectra=ground_truth,
        baseline_model=unet_model,
        forward_model=forward_model,
    )

    # Or load from saved baselines
    dataset = RefinementDataset.load(baselines_path)
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dexsy_core.preprocessing import build_model_inputs
from .model import ConditionalUNetDenoiser


class RefinementDataset(Dataset):
    """
    Dataset for diffusion refinement training.

    Each sample contains:
    - signal: Raw noisy signal [1, 64, 64]
    - f_base: UNet baseline prediction [1, 64, 64]
    - f_gt: Ground truth spectrum [1, 64, 64]
    - model_input: 3-channel model input [3, 64, 64]
    - residual: Log-domain residual r_gt = log(f_gt) - log(f_base)
    """

    def __init__(
        self,
        signals: np.ndarray,
        f_base: np.ndarray,
        f_gt: np.ndarray,
        forward_model,
        augment: bool = False,
    ):
        """
        Args:
            signals: Noisy signals [N, 1, 64, 64]
            f_base: Baseline predictions from UNet [N, 1, 64, 64]
            f_gt: Ground truth spectra [N, 1, 64, 64]
            forward_model: ForwardModel2D instance for preprocessing
            augment: Whether to apply data augmentation
        """
        self.signals = torch.from_numpy(signals).float()
        self.f_base = torch.from_numpy(f_base).float()
        self.f_gt = torch.from_numpy(f_gt).float()
        self.augment = augment
        self.forward_model = forward_model

        self._precompute_model_inputs()
        self._precompute_residuals()

    def _precompute_model_inputs(self):
        """Precompute 3-channel model inputs for efficiency."""
        signals_np = self.signals.numpy()
        model_inputs_np = build_model_inputs(signals_np, self.forward_model)
        self.model_inputs = torch.from_numpy(model_inputs_np).float()

    def _precompute_residuals(self):
        """Precompute log-domain residuals."""
        f_base_safe = torch.clamp(self.f_base, min=1e-8)
        f_gt_safe = torch.clamp(self.f_gt, min=1e-8)
        self.residual = torch.log(f_gt_safe) - torch.log(f_base_safe)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> dict:
        signal = self.signals[idx]
        f_base = self.f_base[idx]
        f_gt = self.f_gt[idx]
        model_input = self.model_inputs[idx]
        residual = self.residual[idx]

        if self.augment and random.random() < 0.5:
            signal = signal.transpose(-1, -2)
            f_base = f_base.transpose(-1, -2)
            f_gt = f_gt.transpose(-1, -2)
            model_input = model_input.transpose(-1, -2)
            residual = residual.transpose(-1, -2)

        return {
            'signal': signal,
            'f_base': f_base,
            'f_gt': f_gt,
            'residual': residual,
            'model_input': model_input,
        }

    @classmethod
    def from_existing_data(
        cls,
        signals: np.ndarray,
        spectra: np.ndarray,
        baseline_model: torch.nn.Module,
        forward_model,
        device: str = 'cuda',
        batch_size: int = 64,
        augment: bool = False,
    ) -> 'RefinementDataset':
        """
        Create dataset by generating baselines using the frozen UNet.

        Args:
            signals: Noisy signals [N, 1, 64, 64]
            spectra: Ground truth spectra [N, 1, 64, 64]
            baseline_model: Trained UNet model
            forward_model: ForwardModel2D instance
            device: Device to run inference on
            batch_size: Batch size for baseline generation
            augment: Whether to apply data augmentation

        Returns:
            RefinementDataset instance
        """
        baseline_model.eval()
        baseline_model.to(device)

        n_samples = len(signals)
        f_base = np.zeros((n_samples, 1, 64, 64), dtype=np.float32)

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_signals = signals[i:end_idx]

                signals_tensor = torch.from_numpy(batch_signals).float().to(device)
                model_inputs = build_model_inputs(batch_signals, forward_model)
                model_inputs_tensor = torch.from_numpy(model_inputs).float().to(device)

                batch_f_base = baseline_model(model_inputs_tensor)
                f_base[i:end_idx] = batch_f_base.cpu().numpy()

                if (i // batch_size) % 50 == 0:
                    print(f"  Baseline generation: {i}/{n_samples}")

        baseline_model.cpu()

        return cls(
            signals=signals,
            f_base=f_base,
            f_gt=spectra,
            forward_model=forward_model,
            augment=augment,
        )

    @classmethod
    def load(
        cls,
        path: str,
        forward_model,
    ) -> 'RefinementDataset':
        """Load dataset from saved numpy files."""
        path = Path(path)
        data = np.load(path)
        return cls(
            signals=data['signals'],
            f_base=data['f_base'],
            f_gt=data['f_gt'],
            forward_model=forward_model,
        )

    def save(self, path: str):
        """Save dataset to numpy file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            signals=self.signals.numpy(),
            f_base=self.f_base.numpy(),
            f_gt=self.f_gt.numpy(),
        )
        print(f"Saved refinement dataset to {path}")


class RefinementDataLoader:
    """
    DataLoader wrapper with additional utilities for refinement training.
    """

    def __init__(
        self,
        train_dataset: RefinementDataset,
        val_dataset: Optional[RefinementDataset] = None,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.val_loader = (
            DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            if val_dataset is not None
            else None
        )

    @property
    def train_size(self) -> int:
        return len(self.train_loader.dataset)

    @property
    def val_size(self) -> int:
        return len(self.val_loader.dataset) if self.val_loader else 0


def generate_refinement_dataset(
    forward_model,
    baseline_model: torch.nn.Module,
    n_train: int = 9500,
    n_val: int = 400,
    n_test: int = 100,
    noise_sigma_range: Tuple[float, float] = (0.005, 0.015),
    n_compartments: int = 3,
    device: str = 'cuda',
    batch_size: int = 64,
    seed: int = 42,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Generate complete refinement dataset with baselines.

    Args:
        forward_model: ForwardModel2D instance
        baseline_model: Trained UNet model
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        noise_sigma_range: Noise sigma range
        n_compartments: Number of compartments
        device: Device for baseline generation
        batch_size: Batch size for inference
        seed: Random seed
        save_dir: Optional directory to save generated data

    Returns:
        Dictionary with train/val/test RefinementDatasets
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    generation_kwargs = {
        'n_compartments': n_compartments,
        'normalize': True,
        'noise_model': 'rician',
    }

    datasets = {}

    for split_name, n_samples in [("train", n_train), ("val", n_val), ("test", n_test)]:
        print(f"\nGenerating {split_name} set ({n_samples} samples)...")

        F, S, _, S_clean = forward_model.generate_batch(
            n_samples=n_samples,
            noise_sigma=None,
            noise_sigma_range=noise_sigma_range,
            return_reference_signal=True,
            **generation_kwargs
        )

        S = S.reshape(-1, 1, forward_model.n_b, forward_model.n_b).astype(np.float32)
        F = F.reshape(-1, 1, forward_model.n_d, forward_model.n_d).astype(np.float32)

        print(f"  Generating baselines with UNet...")
        dataset = RefinementDataset.from_existing_data(
            signals=S,
            spectra=F,
            baseline_model=baseline_model,
            forward_model=forward_model,
            device=device,
            batch_size=batch_size,
        )

        if save_dir:
            save_path = Path(save_dir) / f"refinement_{split_name}.npz"
            dataset.save(save_path)

        datasets[split_name] = dataset
        print(f"  {split_name}: {len(dataset)} samples")

    return datasets


if __name__ == "__main__":
    import torch

    from dexsy_core.forward_model import ForwardModel2D
    from models_3d.attention_unet.model import AttentionUNet3C

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Initializing...")
    forward_model = ForwardModel2D(n_d=64, n_b=64)
    unet = AttentionUNet3C(in_channels=3, base_filters=32).to(device)

    print("\nGenerating small test dataset...")
    datasets = generate_refinement_dataset(
        forward_model=forward_model,
        baseline_model=unet,
        n_train=100,
        n_val=20,
        n_test=10,
        device=device,
        batch_size=32,
    )

    sample = datasets['train'][0]
    print("\nSample structure:")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, range=[{val.min():.4f}, {val.max():.4f}]")

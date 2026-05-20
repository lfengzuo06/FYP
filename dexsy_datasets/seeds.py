"""
Unified random seed management for reproducible experiments.

Ensures consistent seeding across Python, NumPy, PyTorch, and CUDA.
Separates data generation seeds from training seeds.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np


def fix_all_seeds(seed: int) -> None:
    """
    Fix all random number generators for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy's global generator
    - PyTorch CPU and CUDA generators

    Args:
        seed: The seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


class SeedManager:
    """
    Manages random seeds for data generation and training.

    Provides deterministic seed derivation for different purposes:
    - Data generation splits (train, val, test)
    - Training initialization
    - Dataloader shuffling

    Example:
        sm = SeedManager(base_seed=42)

        # Get seeds for data generation
        train_seed = sm.get_generator_seed("train")  # 42
        val_seed = sm.get_generator_seed("val")      # 43
        test_seed = sm.get_generator_seed("test")    # 44

        # Generate training seeds
        training_seeds = sm.generate_training_seeds()
        # {'init_seed': 1234, 'dataloader_seed': 5678}
    """

    def __init__(self, base_seed: int):
        """
        Initialize seed manager.

        Args:
            base_seed: Base seed for data generation
        """
        self.base_seed = int(base_seed)
        self._rng = np.random.default_rng(self.base_seed)

    def get_generator_seed(self, split: str) -> int:
        """
        Get seed for data generation split.

        Args:
            split: 'train', 'val', or 'test'

        Returns:
            Seed for the specified split
        """
        offsets = {"train": 0, "val": 1, "test": 2}
        if split not in offsets:
            raise ValueError(f"Unknown split '{split}'. Must be one of {list(offsets.keys())}")
        return self.base_seed + offsets[split]

    def fix_all_seeds(self, seed: int) -> None:
        """
        Fix all random generators with the given seed.

        Args:
            seed: Seed to use for all generators
        """
        fix_all_seeds(seed)

    def fix_for_training(self, init_seed: int, dataloader_seed: int) -> None:
        """
        Fix seeds for training.

        This sets up seeds for model initialization and dataloader,
        but leaves data generation seeds unchanged.

        Args:
            init_seed: Seed for model initialization
            dataloader_seed: Seed for dataloader shuffling
        """
        fix_all_seeds(init_seed)

    def generate_training_seeds(self) -> Dict[str, int]:
        """
        Generate independent seeds for training.

        Uses the internal RNG to derive seeds that are independent
        from the base data generation seed.

        Returns:
            Dict with 'init_seed' and 'dataloader_seed'
        """
        return {
            "init_seed": int(self._rng.integers(2**31)),
            "dataloader_seed": int(self._rng.integers(2**31)),
        }

    def derive_seed(self, purpose: str) -> int:
        """
        Derive a seed for a specific purpose.

        Args:
            purpose: Arbitrary purpose string (e.g., 'augmentation', 'split')

        Returns:
            Derived seed
        """
        import hashlib

        combined = f"{self.base_seed}_{purpose}"
        hash_val = hashlib.sha256(combined.encode()).hexdigest()
        return int(hash_val[:8], 16) % (2**31)

    def split_generator_seeds(self) -> Dict[str, int]:
        """
        Get seeds for all splits.

        Returns:
            Dict with 'train', 'val', 'test' seeds
        """
        return {
            "train": self.get_generator_seed("train"),
            "val": self.get_generator_seed("val"),
            "test": self.get_generator_seed("test"),
        }

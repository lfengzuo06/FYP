"""
Data Protocol for Active Learning.

This module handles dataset splitting, metadata management, and sample tracking
for the active learning framework. It ensures reproducible splits and preserves
full metadata for failure analysis.

Key principles:
- val_fixed and test_fixed are locked once generated (never participate in AL)
- All samples have complete metadata for failure attribution
- Seeds are tracked for reproducibility
"""

from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dexsy_core.forward_model import ForwardModel2D, compute_dei
from dexsy_core.preprocessing import build_model_inputs
from models_3d.pinn.train import DEXSYDataset3C


@dataclass
class SampleMetadata:
    """
    Complete metadata for a DEXSY sample.

    This metadata is essential for:
    - Failure type attribution
    - Hard case regeneration
    - Augmentation targeting
    """
    seed: int
    compartment_params: dict  # {diffusions, volume_fractions}
    exchange_rates: dict  # {rate_01, rate_02, rate_12}
    vf: list  # Volume fractions [v0, v1, v2]
    noise_sigma: float
    mixing_time: float
    theoretical_dei: float  # Ground truth DEI from weight matrix
    compartment_indices: tuple  # Grid indices for peak positions
    pair_blob_dei: dict  # Per-pair DEI {dei_01_blob, dei_02_blob, dei_12_blob}
    exchange_probs: dict  # Exchange probabilities {0-1, 0-2, 1-2}
    diffusions: list  # Raw diffusion coefficients

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SampleMetadata:
        """Create from dictionary."""
        return cls(**d)

    def get_peak_separation(self) -> int:
        """Compute minimum separation between compartment peaks."""
        indices = list(self.compartment_indices)
        if len(indices) < 2:
            return 0
        min_sep = min(abs(indices[i] - indices[j])
                       for i in range(len(indices))
                       for j in range(i + 1, len(indices)))
        return min_sep

    def get_vf_imbalance(self) -> float:
        """Compute volume fraction imbalance (max - min)."""
        return max(self.vf) - min(self.vf)

    def get_rate_asymmetry(self) -> float:
        """Compute rate asymmetry (max_rate / min_rate)."""
        rates = list(self.exchange_rates.values())
        rates = [r for r in rates if r > 0]
        if len(rates) < 2:
            return 0.0
        return max(rates) / (min(rates) + 1e-10)

    def get_total_exchange_rate(self) -> float:
        """Compute mean exchange rate."""
        return sum(self.exchange_rates.values()) / 3


@dataclass
class SplitData:
    """Container for a data split with signals, labels, and metadata."""
    signals: np.ndarray  # Shape: (N, 64, 64) - raw noisy signals
    model_inputs: np.ndarray  # Shape: (N, 3, 64, 64) - 3-channel model inputs
    labels: np.ndarray  # Shape: (N, 1, 64, 64)
    clean_signals: np.ndarray  # Shape: (N, 1, 64, 64)
    metadata: list[SampleMetadata]
    seeds: list[int]  # RNG seeds for reproducibility

    @property
    def n_samples(self) -> int:
        return len(self.signals)

    def to_torch_dataset(self, augment: bool = False) -> Dataset:
        """Convert to PyTorch Dataset using 3-channel model inputs."""
        return DEXSYDataset3C(
            inputs=self.model_inputs,  # (N, 3, 64, 64) for PINN
            labels=self.labels,
            clean_signals=self.clean_signals,
            augment=augment,
        )

    @property
    def inputs(self) -> np.ndarray:
        """Alias for model_inputs for backwards compatibility."""
        return self.model_inputs  # Now returns 3-channel inputs


class DataProtocol:
    """
    Manages dataset splitting and metadata for active learning.

    Ensures reproducible splits with complete metadata preservation.
    """

    def __init__(
        self,
        config,
        forward_model: ForwardModel2D | None = None,
    ):
        """
        Initialize the data protocol.

        Args:
            config: ALConfig instance with dataset sizes and paths.
            forward_model: ForwardModel2D instance (creates new if None).
        """
        self.config = config
        self.forward_model = forward_model or ForwardModel2D(n_d=64, n_b=64)
        self.output_dir = config.output_dir / "data_splits"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._splits: dict[str, SplitData] = {}
        self._split_generated = False

    def setup(self, force_regenerate: bool = False) -> None:
        """
        Generate or load fixed data splits.

        Args:
            force_regenerate: If True, regenerate splits even if they exist.
        """
        split_file = self.output_dir / "splits_metadata.json"

        if not force_regenerate and split_file.exists():
            self._load_splits()
            print(f"Loaded existing splits from {split_file}")
        else:
            self._generate_splits()
            self._save_splits()
            print(f"Generated new splits and saved to {split_file}")

    def _generate_splits(self) -> None:
        """Generate all data splits with metadata."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        generation_order = [
            ('test_fixed', self.config.n_test_fixed, 0),
            ('val_fixed', self.config.n_val_fixed, self.config.n_test_fixed),
            ('candidate_pool', self.config.n_candidate_pool,
             self.config.n_test_fixed + self.config.n_val_fixed),
            ('train_base', self.config.n_train_base,
             self.config.n_test_fixed + self.config.n_val_fixed + self.config.n_candidate_pool),
        ]

        for split_name, n_samples, seed_offset in generation_order:
            print(f"Generating {split_name} ({n_samples} samples)...")
            split_data = self._generate_split_data(
                n_samples=n_samples,
                base_seed=self.config.seed + seed_offset,
            )
            self._splits[split_name] = split_data

        self._split_generated = True

    def _generate_split_data(
        self,
        n_samples: int,
        base_seed: int,
    ) -> SplitData:
        """
        Generate a single data split with full metadata.

        Args:
            n_samples: Number of samples to generate.
            base_seed: Base RNG seed for reproducibility.

        Returns:
            SplitData with signals, labels, clean_signals, and metadata.
        """
        signals = []
        labels = []
        clean_signals = []
        metadata_list = []

        for i in range(n_samples):
            seed = base_seed + i

            # Set seed for this sample
            rng = random.Random(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)

            # Generate 3C sample using generate_3compartment_paper (handles None params)
            generated = self.forward_model.generate_3compartment_paper(
                normalize=True,
                return_reference_signal=True,
            )

            # generated can be 3 or 4 values depending on return_reference_signal
            if len(generated) == 4:
                f, s_noisy, params, s_clean = generated
            else:
                f, s_noisy, params = generated
                s_clean = self.forward_model.compute_signal(f, noise_sigma=0.0, normalize=True, noise_model=None)

            # Extract metadata
            metadata = SampleMetadata(
                seed=seed,
                compartment_params={
                    'diffusions': params.get('diffusions', []),
                    'volume_fractions': params.get('volume_fractions', []),
                },
                exchange_rates={
                    'rate_01': params['exchange_rates'].get('0-1', 0.0),
                    'rate_02': params['exchange_rates'].get('0-2', 0.0),
                    'rate_12': params['exchange_rates'].get('1-2', 0.0),
                },
                vf=params.get('volume_fractions', []),
                noise_sigma=params.get('noise_sigma', 0.01),
                mixing_time=params.get('mixing_time', 0.1),
                theoretical_dei=params.get('theoretical_dei', 0.0),
                compartment_indices=params.get('compartment_indices', (0, 0, 0)),
                pair_blob_dei={},
                exchange_probs={
                    '0-1': params.get('exchange_probabilities', {}).get('0-1', 0.0),
                    '0-2': params.get('exchange_probabilities', {}).get('0-2', 0.0),
                    '1-2': params.get('exchange_probabilities', {}).get('1-2', 0.0),
                },
                diffusions=params.get('diffusions', []),
            )

            # Reshape to (1, 64, 64) for storage
            s_noisy_4d = s_noisy.reshape(1, 1, self.forward_model.n_b, self.forward_model.n_b)
            s_clean_4d = s_clean.reshape(1, 1, self.forward_model.n_b, self.forward_model.n_b)
            f_4d = f.reshape(1, 1, self.forward_model.n_d, self.forward_model.n_d)

            # Store 3D signals for residual computation (shape: (64, 64))
            signals.append(s_noisy[0] if s_noisy.ndim == 3 else s_noisy)
            clean_signals.append(s_clean[0] if s_clean.ndim == 3 else s_clean)
            labels.append(f[0] if f.ndim == 3 else f)
            metadata_list.append(metadata)

        # Stack to (N, 64, 64)
        signals = np.stack(signals, axis=0).astype(np.float32)
        clean_signals = np.stack(clean_signals, axis=0).astype(np.float32)
        labels = np.stack(labels, axis=0).astype(np.float32)

        # Compute 3-channel model inputs for PINN
        signals_4d = signals.reshape(-1, 1, self.forward_model.n_b, self.forward_model.n_b)
        model_inputs = build_model_inputs(signals_4d, self.forward_model)

        return SplitData(
            signals=signals,  # (N, 64, 64) - for residual computation
            model_inputs=model_inputs,  # (N, 3, 64, 64) - for PINN input
            labels=labels.reshape(-1, 1, self.forward_model.n_d, self.forward_model.n_d),  # (N, 1, 64, 64)
            clean_signals=clean_signals.reshape(-1, 1, self.forward_model.n_b, self.forward_model.n_b),  # (N, 1, 64, 64)
            metadata=metadata_list,
            seeds=[m.seed for m in metadata_list],
        )

    def _save_splits(self) -> None:
        """Save splits to disk."""
        # Save metadata as JSON
        splits_info = {}
        for name, split in self._splits.items():
            splits_info[name] = {
                'n_samples': split.n_samples,
                'seeds': split.seeds,
            }

        with open(self.output_dir / "splits_metadata.json", 'w') as f:
            json.dump(splits_info, f, indent=2)

        # Save each split's data
        for name, split in self._splits.items():
            split_dir = self.output_dir / name
            split_dir.mkdir(exist_ok=True)

            np.save(split_dir / "signals.npy", split.signals)
            np.save(split_dir / "model_inputs.npy", split.model_inputs)
            np.save(split_dir / "labels.npy", split.labels)
            np.save(split_dir / "clean_signals.npy", split.clean_signals)

            with open(split_dir / "metadata.pkl", 'wb') as f:
                pickle.dump(split.metadata, f)

    def _load_splits(self) -> None:
        """Load splits from disk."""
        splits_info_path = self.output_dir / "splits_metadata.json"
        with open(splits_info_path, 'r') as f:
            splits_info = json.load(f)

        for name, info in splits_info.items():
            split_dir = self.output_dir / name

            signals = np.load(split_dir / "signals.npy")
            model_inputs = np.load(split_dir / "model_inputs.npy")
            labels = np.load(split_dir / "labels.npy")
            clean_signals = np.load(split_dir / "clean_signals.npy")

            with open(split_dir / "metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)

            self._splits[name] = SplitData(
                signals=signals,
                model_inputs=model_inputs,
                labels=labels,
                clean_signals=clean_signals,
                metadata=metadata,
                seeds=[m.seed for m in metadata],
            )

        self._split_generated = True

    def get_split(self, name: str) -> SplitData:
        """
        Get a data split by name.

        Args:
            name: One of 'train_base', 'candidate_pool', 'val_fixed', 'test_fixed'.

        Returns:
            SplitData instance.
        """
        if not self._split_generated:
            self.setup()
        return self._splits[name]

    def get_candidate_pool(self) -> SplitData:
        """Get the candidate pool for AL sampling."""
        return self.get_split('candidate_pool')

    def get_val_fixed(self) -> SplitData:
        """Get the fixed validation set."""
        return self.get_split('val_fixed')

    def get_test_fixed(self) -> SplitData:
        """Get the fixed test set."""
        return self.get_split('test_fixed')

    def get_train_base(self) -> SplitData:
        """Get the base training set."""
        return self.get_split('train_base')

    def create_mixed_training_set(
        self,
        augmented_data: list[tuple],
        hard_ratio: float = 0.3,
    ) -> SplitData:
        """
        Create a mixed training set from base + augmented data.

        Args:
            augmented_data: List of (signal, label, clean_signal, metadata) tuples.
            hard_ratio: Fraction of hard cases in the mixed set.

        Returns:
            SplitData with mixed training data.
        """
        train_base = self.get_train_base()

        # Calculate how many augmented samples to add
        n_base = train_base.n_samples
        # If we want 30% hard cases, then: n_hard / (n_base + n_hard) = 0.3
        # => n_hard = 0.3 * n_base / 0.7
        n_hard = int(n_base * hard_ratio / (1 - hard_ratio))
        n_hard = min(n_hard, len(augmented_data))

        # Sample augmented data
        if n_hard > 0 and len(augmented_data) > 0:
            indices = random.sample(range(len(augmented_data)), min(n_hard, len(augmented_data)))
            aug_signals = []
            aug_labels = []
            aug_clean = []
            aug_metadata = []
            aug_model_inputs = []

            for idx in indices:
                sig, lab, cln, meta = augmented_data[idx]
                
                # Normalize all shapes to match train_base for concatenation:
                # - train_base.signals: (N, 64, 64) - 3D
                # - train_base.labels: (N, 1, 64, 64) - 4D
                # - train_base.clean_signals: (N, 1, 64, 64) - 4D
                # - train_base.model_inputs: (N, 3, 64, 64) - 4D
                
                # Normalize signals to 3D (1, 64, 64) for concatenation
                if sig.ndim == 2:
                    sig = sig.reshape(1, 64, 64)
                elif sig.ndim == 3:
                    if sig.shape[0] == 1:
                        pass  # Already (1, 64, 64), keep as is
                    else:
                        sig = sig.reshape(1, 64, 64)
                elif sig.ndim == 4:
                    sig = sig[:, 0] if sig.shape[1] == 1 else sig.reshape(-1, 64, 64)
                
                # Normalize labels to 4D (1, 1, 64, 64)
                if lab.ndim == 2:
                    lab = lab.reshape(1, 1, 64, 64)
                elif lab.ndim == 3:
                    if lab.shape[0] == 1:
                        lab = lab.reshape(1, 1, 64, 64)
                    else:
                        lab = lab.reshape(1, 1, 64, 64)
                elif lab.ndim == 4:
                    pass  # Already 4D
                
                # Normalize clean signals to 4D (1, 1, 64, 64)
                if cln.ndim == 2:
                    cln = cln.reshape(1, 1, 64, 64)
                elif cln.ndim == 3:
                    if cln.shape[0] == 1:
                        cln = cln.reshape(1, 1, 64, 64)
                    else:
                        cln = cln.reshape(1, 1, 64, 64)
                elif cln.ndim == 4:
                    pass  # Already 4D

                aug_signals.append(sig)  # (1, 64, 64)
                aug_labels.append(lab)  # (1, 1, 64, 64)
                aug_clean.append(cln)  # (1, 1, 64, 64)
                aug_metadata.append(meta)

                # Compute model_inputs for augmented sample (3-channel)
                # build_model_inputs expects (N, 1, 64, 64)
                sig_for_model = sig.reshape(1, 1, 64, 64).astype(np.float32)
                model_inp = build_model_inputs(sig_for_model, self.forward_model)  # (1, 3, 64, 64)
                aug_model_inputs.append(model_inp[0])  # (3, 64, 64)

            # Combine with base training data
            combined_signals = np.concatenate([train_base.signals] + aug_signals, axis=0)  # 3D
            combined_labels = np.concatenate([train_base.labels] + aug_labels, axis=0)  # 4D
            combined_clean = np.concatenate([train_base.clean_signals] + aug_clean, axis=0)  # 4D
            # model_inputs: augment each to 4D (1, 3, 64, 64) before concat
            aug_model_inputs_4d = [m.reshape(1, 3, 64, 64) for m in aug_model_inputs]
            combined_model_inputs = np.concatenate([train_base.model_inputs] + aug_model_inputs_4d, axis=0)  # 4D
            combined_metadata = train_base.metadata + aug_metadata
            combined_seeds = [m.seed for m in combined_metadata]
        else:
            combined_signals = train_base.signals
            combined_labels = train_base.labels
            combined_clean = train_base.clean_signals
            combined_model_inputs = train_base.model_inputs
            combined_metadata = train_base.metadata
            combined_seeds = train_base.seeds

        return SplitData(
            signals=combined_signals,
            model_inputs=combined_model_inputs,
            labels=combined_labels,
            clean_signals=combined_clean,
            metadata=combined_metadata,
            seeds=combined_seeds,
        )

    def regenerate_from_metadata(
        self,
        metadata: SampleMetadata,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Regenerate a sample from its metadata.

        This is useful for augmentation with controlled perturbations.

        Args:
            metadata: SampleMetadata with original parameters.

        Returns:
            Tuple of (spectrum, noisy_signal, clean_signal, params).
        """
        # Set seed for reproducibility
        random.seed(metadata.seed)
        np.random.seed(metadata.seed)
        torch.manual_seed(metadata.seed)

        # Regenerate with specific parameters
        f, s_clean, params = self.forward_model.generate_3c_validation_spectrum(
            diffusions=np.array(metadata.diffusions),
            volume_fractions=np.array(metadata.vf),
            exchange_rates=(
                metadata.exchange_rates['rate_01'],
                metadata.exchange_rates['rate_02'],
                metadata.exchange_rates['rate_12'],
            ),
            mixing_time=metadata.mixing_time,
            jitter_pixels=1,
            normalize=True,
        )

        # Add noise
        s_noisy = self.forward_model.compute_signal(
            f,
            noise_sigma=metadata.noise_sigma,
            noise_model='rician',
            normalize=True,
        )

        return f, s_noisy, s_clean, params

    def get_dataloader(
        self,
        split_name: str,
        augment: bool = False,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Get a DataLoader for a split.

        Args:
            split_name: Name of the split.
            augment: Whether to apply augmentation.
            shuffle: Whether to shuffle the data.

        Returns:
            DataLoader instance.
        """
        split = self.get_split(split_name)

        dataset = DEXSYDataset3C(
            inputs=split.model_inputs,  # (N, 3, 64, 64) for PINN
            labels=split.labels,
            clean_signals=split.clean_signals,
            augment=augment,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    def summary(self) -> dict[str, Any]:
        """Get summary of all splits."""
        summary = {}
        for name in ['train_base', 'candidate_pool', 'val_fixed', 'test_fixed']:
            split = self.get_split(name)
            summary[name] = {
                'n_samples': split.n_samples,
                'seeds': f"{min(split.seeds)}-{max(split.seeds)}",
            }
        return summary

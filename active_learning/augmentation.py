"""
Hard Case Augmentation for Active Learning.

This module generates augmented samples around hard seed cases by perturbing
parameters within controlled bounds. This enables the model to focus on
specific failure modes.

Perturbation strategies:
- Exchange rates: multiplicative (20% variation)
- Volume fractions: additive (0.05 variation)
- Peak positions: additive (2-4 bins)
- Noise levels: categorical (multiple levels)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import sys
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dexsy_core.forward_model import ForwardModel2D
from .data_protocol import SampleMetadata
from .config import ALConfig


@dataclass
class PerturbationBounds:
    """Configuration for perturbation bounds."""
    rate_factor: float = 0.2  # 20% multiplicative factor
    vf_delta: float = 0.05  # Additive delta
    peak_delta: int = 4  # Peak position perturbation
    noise_levels: tuple = (0.005, 0.01, 0.015, 0.02)  # Categorical


@dataclass
class AugmentedSample:
    """Container for an augmented sample with its metadata."""
    spectrum: np.ndarray  # Ground truth spectrum (64x64)
    signal: np.ndarray  # Noisy signal (64x64)
    clean_signal: np.ndarray  # Clean signal (64x64)
    metadata: SampleMetadata  # Updated metadata
    parent_seed: int  # Seed of the parent hard case
    perturbations_applied: dict  # Record of perturbations

    def to_tuple(self):
        """Convert to tuple format."""
        return (
            self.signal,
            self.spectrum,
            self.clean_signal,
            self.metadata,
        )


class HardCaseAugmenter:
    """
    Generates augmented samples around hard seed cases.

    Takes hard seeds and creates variations by perturbing parameters
    within controlled bounds to expand the training set.
    """

    def __init__(
        self,
        forward_model: ForwardModel2D | None = None,
        perturbation_bounds: PerturbationBounds | None = None,
    ):
        """
        Initialize the augmenter.

        Args:
            forward_model: ForwardModel2D instance.
            perturbation_bounds: Configuration for perturbations.
        """
        self.forward_model = forward_model or ForwardModel2D(n_d=64, n_b=64)
        self.bounds = perturbation_bounds or PerturbationBounds()

    def augment_single(
        self,
        seed_metadata: SampleMetadata,
        n_augmentations: int = 4,
        seed_offset: int = 0,
    ) -> list[AugmentedSample]:
        """
        Generate augmented samples for a single hard seed.

        Args:
            seed_metadata: Metadata of the hard seed.
            n_augmentations: Number of variations to generate.
            seed_offset: Offset for RNG seed.

        Returns:
            List of AugmentedSample instances.
        """
        augmented = []

        for i in range(n_augmentations):
            # Generate unique seed for this augmentation
            aug_seed = seed_metadata.seed * 1000 + i + seed_offset

            # Set RNG seeds
            random.seed(aug_seed)
            np.random.seed(aug_seed)

            # Perturb parameters
            perturbed = self._perturb_params(seed_metadata)

            # Generate new sample
            sample = self._generate_sample(
                perturbed=perturbed,
                parent_seed=seed_metadata.seed,
                perturbations=perturbed,
            )
            augmented.append(sample)

        return augmented

    def augment_batch(
        self,
        seeds: list[SampleMetadata],
        n_augmentations: int = 4,
    ) -> list[AugmentedSample]:
        """
        Generate augmented samples for multiple hard seeds.

        Args:
            seeds: List of hard seed metadata.
            n_augmentations: Number of variations per seed.

        Returns:
            List of all AugmentedSample instances.
        """
        all_augmented = []

        for idx, seed_meta in enumerate(seeds):
            augmented = self.augment_single(
                seed_metadata=seed_meta,
                n_augmentations=n_augmentations,
                seed_offset=idx * 10000,
            )
            all_augmented.extend(augmented)

        return all_augmented

    def _perturb_params(
        self,
        metadata: SampleMetadata,
    ) -> dict[str, Any]:
        """
        Perturb the parameters of a sample.

        Args:
            metadata: Original sample metadata.

        Returns:
            Dictionary of perturbed parameters.
        """
        bounds = self.bounds

        # Perturb exchange rates (multiplicative)
        perturbed_rates = {}
        for key, value in metadata.exchange_rates.items():
            factor = 1.0 + random.uniform(-bounds.rate_factor, bounds.rate_factor)
            perturbed_rates[key] = max(0.01, value * factor)  # Clamp to positive

        # Perturb volume fractions (additive with renormalization)
        perturbed_vf = [
            max(0.05, min(0.9, vf + random.uniform(-bounds.vf_delta, bounds.vf_delta)))
            for vf in metadata.vf
        ]
        # Renormalize to sum to 1
        vf_sum = sum(perturbed_vf)
        perturbed_vf = [vf / vf_sum for vf in perturbed_vf]

        # Perturb diffusion coefficients (shift peak positions)
        perturbed_diffusions = []
        for i, d in enumerate(metadata.diffusions):
            # Get index and perturb
            idx = self.forward_model._nearest_diffusion_index(d)
            delta = random.randint(-bounds.peak_delta, bounds.peak_delta)
            new_idx = max(0, min(self.forward_model.n_d - 1, idx + delta))
            new_d = self.forward_model.D1[new_idx]
            perturbed_diffusions.append(new_d)

        # Perturb noise level (categorical)
        perturbed_noise = random.choice(bounds.noise_levels)

        # Perturb mixing time slightly
        perturbed_mixing_time = metadata.mixing_time + random.uniform(-0.005, 0.005)
        perturbed_mixing_time = max(
            self.forward_model.mixing_time_range[0],
            min(self.forward_model.mixing_time_range[1], perturbed_mixing_time)
        )

        return {
            'diffusions': perturbed_diffusions,
            'volume_fractions': perturbed_vf,
            'exchange_rates': perturbed_rates,
            'noise_sigma': perturbed_noise,
            'mixing_time': perturbed_mixing_time,
        }

    def _generate_sample(
        self,
        perturbed: dict[str, Any],
        parent_seed: int,
        perturbations: dict[str, Any],
    ) -> AugmentedSample:
        """
        Generate a new sample with perturbed parameters.

        Args:
            perturbed: Dictionary of perturbed parameters.
            parent_seed: Seed of the parent hard case.
            perturbations: Record of perturbations applied.

        Returns:
            AugmentedSample instance.
        """
        # Generate spectrum and signals
        f, s_clean, params = self.forward_model.generate_3c_validation_spectrum(
            diffusions=np.array(perturbed['diffusions']),
            volume_fractions=np.array(perturbed['volume_fractions']),
            exchange_rates=(
                perturbed['exchange_rates']['rate_01'],
                perturbed['exchange_rates']['rate_02'],
                perturbed['exchange_rates']['rate_12'],
            ),
            mixing_time=perturbed['mixing_time'],
            jitter_pixels=1,
            normalize=True,
        )

        # Add noise
        s_noisy = self.forward_model.compute_signal(
            f,
            noise_sigma=perturbed['noise_sigma'],
            noise_model='rician',
            normalize=True,
        )

        # Create new metadata
        new_metadata = SampleMetadata(
            seed=parent_seed * 1000 + hash(str(perturbed)) % 1000000,
            compartment_params={
                'diffusions': perturbed['diffusions'],
                'volume_fractions': perturbed['volume_fractions'],
            },
            exchange_rates=perturbed['exchange_rates'],
            vf=perturbed['volume_fractions'],
            noise_sigma=perturbed['noise_sigma'],
            mixing_time=perturbed['mixing_time'],
            theoretical_dei=params.get('theoretical_dei', 0.0),
            compartment_indices=params.get('compartment_indices', (0, 0, 0)),
            pair_blob_dei={},
            exchange_probs={
                '0-1': params.get('exchange_probabilities', {}).get('0-1', 0.0),
                '0-2': params.get('exchange_probabilities', {}).get('0-2', 0.0),
                '1-2': params.get('exchange_probabilities', {}).get('1-2', 0.0),
            },
            diffusions=perturbed['diffusions'],
        )

        return AugmentedSample(
            spectrum=f,
            signal=s_noisy,
            clean_signal=s_clean,
            metadata=new_metadata,
            parent_seed=parent_seed,
            perturbations_applied=perturbations,
        )

    def convert_to_training_format(
        self,
        augmented_samples: list[AugmentedSample],
    ) -> list[tuple]:
        """
        Convert augmented samples to training format.

        Args:
            augmented_samples: List of AugmentedSample instances.

        Returns:
            List of (signal, label, clean_signal, metadata) tuples.
        """
        result = []
        for sample in augmented_samples:
            # Reshape for model
            signal = sample.signal.reshape(1, 1, 64, 64).astype(np.float32)
            label = sample.spectrum.reshape(1, 1, 64, 64).astype(np.float32)
            clean = sample.clean_signal.reshape(1, 1, 64, 64).astype(np.float32)
            result.append((signal, label, clean, sample.metadata))
        return result


class BalancedAugmenter:
    """
    Augmenter that ensures balanced augmentation across failure types.

    This prevents over-representation of a single failure type in the
    augmented training set.
    """

    def __init__(
        self,
        forward_model: ForwardModel2D | None = None,
        perturbation_bounds: PerturbationBounds | None = None,
        max_per_type: int = 3,
    ):
        """
        Initialize the balanced augmenter.

        Args:
            forward_model: ForwardModel2D instance.
            perturbation_bounds: Perturbation configuration.
            max_per_type: Maximum augmentations per failure type.
        """
        self.base_augmenter = HardCaseAugmenter(
            forward_model=forward_model,
            perturbation_bounds=perturbation_bounds,
        )
        self.max_per_type = max_per_type

    def augment_balanced(
        self,
        failure_groups: dict,
        n_augmentations: int = 4,
    ) -> list[AugmentedSample]:
        """
        Augment with balanced sampling across failure types.

        Args:
            failure_groups: Dictionary of failure type groups.
            n_augmentations: Number of augmentations per seed.

        Returns:
            List of AugmentedSample instances.
        """
        all_samples = []

        for ftype, group in failure_groups.items():
            # Limit samples per failure type
            seeds_to_augment = group.metadata_list[:self.max_per_type]

            # Augment each seed
            for metadata in seeds_to_augment:
                augmented = self.base_augmenter.augment_single(
                    seed_metadata=metadata,
                    n_augmentations=n_augmentations,
                )
                all_samples.extend(augmented)

        return all_samples


def augment_hard_cases(
    seeds: list[SampleMetadata],
    forward_model: ForwardModel2D | None = None,
    n_augmentations: int = 4,
    rate_factor: float = 0.2,
    vf_delta: float = 0.05,
    peak_delta: int = 4,
    noise_levels: tuple = (0.005, 0.01, 0.015, 0.02),
) -> list[AugmentedSample]:
    """
    Convenience function to augment hard cases.

    Args:
        seeds: List of hard seed metadata.
        forward_model: ForwardModel2D instance.
        n_augmentations: Number of augmentations per seed.
        rate_factor: Exchange rate perturbation factor.
        vf_delta: Volume fraction perturbation delta.
        peak_delta: Peak position perturbation.
        noise_levels: Noise levels for categorical perturbation.

    Returns:
        List of AugmentedSample instances.
    """
    bounds = PerturbationBounds(
        rate_factor=rate_factor,
        vf_delta=vf_delta,
        peak_delta=peak_delta,
        noise_levels=noise_levels,
    )

    augmenter = HardCaseAugmenter(
        forward_model=forward_model,
        perturbation_bounds=bounds,
    )

    return augmenter.augment_batch(seeds, n_augmentations=n_augmentations)

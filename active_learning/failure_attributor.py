"""
Failure Type Attribution for Active Learning.

This module classifies hard cases into failure types based on their metadata
and error patterns. This enables targeted augmentation of specific failure modes.

Failure types:
- high_exchange: Samples with high exchange rates
- small_separation: Samples with closely spaced compartment peaks
- edge_position: Samples with peaks near grid edges
- high_noise: Samples with high noise levels
- low_noise: Samples with low noise levels
- vf_imbalanced: Samples with imbalanced volume fractions
- asymmetric_rates: Samples with asymmetric exchange rates
- high_dei: Samples with high DEI (strong exchange)
- low_dei: Samples with low DEI (weak exchange)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

import sys
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from .data_protocol import SampleMetadata
from .scorer import SampleScores
from .config import FailureTypeConfig, DEFAULT_FAILURE_CONFIG


# Type alias for failure predicates
FailurePredicate = Callable[[SampleMetadata], bool]


@dataclass
class FailureGroup:
    """Container for samples belonging to a failure type."""
    failure_type: str
    samples: list[tuple[SampleMetadata, SampleScores]]
    description: str = ""

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def metadata_list(self) -> list[SampleMetadata]:
        return [m for m, _ in self.samples]

    @property
    def scores_list(self) -> list[SampleScores]:
        return [s for _, s in self.samples]

    def is_empty(self) -> bool:
        return len(self.samples) == 0


class FailureAttributor:
    """
    Classifies hard cases into failure types based on metadata.

    Uses configurable thresholds to determine failure categories and ensures
    balanced sampling across failure types to prevent model bias.
    """

    # Failure type definitions with predicates
    FAILURE_TYPES: dict[str, FailurePredicate] = {
        'high_exchange': lambda m: m.get_total_exchange_rate() > 10.0,
        'small_separation': lambda m: m.get_peak_separation() < 8,
        'edge_position': lambda m: any(i < 10 or i > 54 for i in m.compartment_indices),
        'high_noise': lambda m: m.noise_sigma > 0.012,
        'low_noise': lambda m: m.noise_sigma < 0.007,
        'vf_imbalanced': lambda m: m.get_vf_imbalance() > 0.4,
        'asymmetric_rates': lambda m: m.get_rate_asymmetry() > 3.0,
        'high_dei': lambda m: m.theoretical_dei > 0.3,
        'low_dei': lambda m: m.theoretical_dei < 0.05,
    }

    # Descriptions for each failure type
    DESCRIPTIONS: dict[str, str] = {
        'high_exchange': 'High exchange rates (>10 s^-1 mean)',
        'small_separation': 'Small peak separation (<8 bins)',
        'edge_position': 'Peaks near grid edges',
        'high_noise': 'High noise level (>0.012)',
        'low_noise': 'Low noise level (<0.007)',
        'vf_imbalanced': 'Imbalanced volume fractions',
        'asymmetric_rates': 'Asymmetric exchange rates',
        'high_dei': 'High DEI (>0.3, strong exchange)',
        'low_dei': 'Low DEI (<0.05, weak exchange)',
    }

    def __init__(
        self,
        config: FailureTypeConfig | None = None,
        max_per_group: int = 3,
    ):
        """
        Initialize the failure attributor.

        Args:
            config: FailureTypeConfig with threshold values.
            max_per_group: Maximum samples per failure type group.
        """
        self.config = config or DEFAULT_FAILURE_CONFIG
        self.max_per_group = max_per_group

        # Build failure type predicates with config thresholds
        self._failure_predicates = self._build_predicates()

    def _build_predicates(self) -> dict[str, FailurePredicate]:
        """Build failure predicates using config thresholds."""
        cfg = self.config

        return {
            'high_exchange': lambda m: m.get_total_exchange_rate() > cfg.high_exchange_threshold,
            'small_separation': lambda m: m.get_peak_separation() < cfg.small_separation_threshold,
            'edge_position': lambda m: any(
                i < cfg.edge_margin or i > 64 - cfg.edge_margin
                for i in m.compartment_indices
            ),
            'high_noise': lambda m: m.noise_sigma > cfg.high_noise_threshold,
            'low_noise': lambda m: m.noise_sigma < cfg.low_noise_threshold,
            'vf_imbalanced': lambda m: m.get_vf_imbalance() > cfg.vf_imbalance_threshold,
            'asymmetric_rates': lambda m: m.get_rate_asymmetry() > cfg.rate_asymmetry_threshold,
            'high_dei': lambda m: m.theoretical_dei > cfg.high_dei_threshold,
            'low_dei': lambda m: m.theoretical_dei < cfg.low_dei_threshold,
        }

    def attribute(
        self,
        hard_seeds: list[tuple[SampleMetadata, SampleScores]],
    ) -> dict[str, FailureGroup]:
        """
        Attribute hard seeds to failure types.

        Args:
            hard_seeds: List of (metadata, scores) tuples.

        Returns:
            Dictionary mapping failure type to FailureGroup.
        """
        # Initialize groups
        groups = {}
        for ftype in self._failure_predicates.keys():
            groups[ftype] = FailureGroup(
                failure_type=ftype,
                samples=[],
                description=self.DESCRIPTIONS.get(ftype, ""),
            )

        # Attribute each sample
        for metadata, scores in hard_seeds:
            for ftype, predicate in self._failure_predicates.items():
                if predicate(metadata):
                    groups[ftype].samples.append((metadata, scores))

        # Cap each group
        for ftype, group in groups.items():
            if len(group.samples) > self.max_per_group:
                # Sort by composite score and take top
                sorted_samples = sorted(
                    group.samples,
                    key=lambda x: x[1].composite_score or 0,
                    reverse=True,
                )
                groups[ftype].samples = sorted_samples[:self.max_per_group]

        return groups

    def get_capped_groups(
        self,
        groups: dict[str, FailureGroup],
        max_total: int | None = None,
    ) -> list[FailureGroup]:
        """
        Get a balanced list of groups with caps applied.

        Args:
            groups: Dictionary of FailureGroups.
            max_total: Maximum total samples across all groups.

        Returns:
            List of non-empty FailureGroups.
        """
        result = []

        for ftype, group in groups.items():
            if not group.is_empty():
                result.append(group)

        # Sort by number of samples (ascending) for balanced selection
        result.sort(key=lambda g: g.n_samples)

        if max_total is not None:
            # Limit total samples
            total = 0
            capped = []
            for group in result:
                remaining = max_total - total
                if remaining <= 0:
                    break
                samples_to_add = min(group.n_samples, remaining)
                if samples_to_add > 0:
                    capped.append(group)
                    total += samples_to_add
            result = capped

        return result

    def print_attribution_summary(
        self,
        groups: dict[str, FailureGroup],
    ) -> None:
        """
        Print a summary of failure type attribution.

        Args:
            groups: Dictionary of FailureGroups.
        """
        print("\n" + "=" * 70)
        print("FAILURE TYPE ATTRIBUTION SUMMARY")
        print("=" * 70)

        total_samples = sum(g.n_samples for g in groups.values())
        print(f"\nTotal hard seeds: {total_samples}")
        print(f"Failure types detected: {sum(1 for g in groups.values() if not g.is_empty())}")

        print("\nBy Failure Type:")
        print("-" * 70)

        # Sort by number of samples (descending)
        sorted_groups = sorted(
            groups.items(),
            key=lambda x: x[1].n_samples,
            reverse=True,
        )

        for ftype, group in sorted_groups:
            if group.is_empty():
                continue

            print(f"\n{ftype.upper()} ({group.n_samples} samples)")
            print(f"  Description: {group.description}")

            # Print sample details
            for metadata, scores in group.samples[:3]:  # Show top 3
                print(f"  - Seed {metadata.seed}: ")
                print(f"      DEI={metadata.theoretical_dei:.3f}, ")
                print(f"      noise={metadata.noise_sigma:.4f}, ")
                print(f"      separation={metadata.get_peak_separation()}")
                if scores.composite_score is not None:
                    print(f"      score={scores.composite_score:.3f}")

        print("\n" + "=" * 70 + "\n")

    def get_augmentation_targets(
        self,
        groups: dict[str, FailureGroup],
    ) -> list[SampleMetadata]:
        """
        Get unique metadata targets for augmentation.

        Ensures no duplicate seeds across failure types.

        Args:
            groups: Dictionary of FailureGroups.

        Returns:
            List of unique SampleMetadata for augmentation.
        """
        seen_seeds = set()
        targets = []

        for group in groups.values():
            for metadata, _ in group.samples:
                if metadata.seed not in seen_seeds:
                    seen_seeds.add(metadata.seed)
                    targets.append(metadata)

        return targets


def create_failure_attributor(
    config: FailureTypeConfig | None = None,
    max_per_group: int = 3,
) -> FailureAttributor:
    """
    Factory function to create a FailureAttributor.

    Args:
        config: FailureTypeConfig with thresholds.
        max_per_group: Maximum samples per failure type.

    Returns:
        Configured FailureAttributor instance.
    """
    return FailureAttributor(config=config, max_per_group=max_per_group)


def quick_attribute(
    hard_seeds: list[tuple[SampleMetadata, SampleScores]],
    max_per_group: int = 3,
) -> dict[str, FailureGroup]:
    """
    Quick failure attribution without creating an instance.

    Args:
        hard_seeds: List of (metadata, scores) tuples.
        max_per_group: Maximum samples per group.

    Returns:
        Dictionary of FailureGroups.
    """
    attributor = FailureAttributor(max_per_group=max_per_group)
    return attributor.attribute(hard_seeds)

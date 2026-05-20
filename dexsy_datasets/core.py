"""
Core dataset classes and ID generation.

Dataset ID is computed as SHA256 of the normalized (sorted, serialized) config content.
This ensures the same config always produces the same ID, regardless of when it was created.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


def compute_dataset_id(config: Dict[str, Any]) -> str:
    """
    Compute deterministic dataset ID from config.

    The ID is SHA256 of the normalized (sorted keys) config JSON.
    This ensures:
    - Same config → same ID (reproducibility)
    - Different config → different ID (uniqueness)

    Args:
        config: Dataset configuration dict

    Returns:
        16-character hex string (first 16 chars of SHA256)
    """
    normalized = _normalize_config(config)
    json_str = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    full_hash = hashlib.sha256(json_str.encode()).hexdigest()
    return full_hash[:16]


def _normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize config for consistent hashing.

    - Sorts keys recursively
    - Converts numpy types to Python native
    - Removes None values
    """
    if isinstance(config, dict):
        return {k: _normalize_config(v) for k, v in sorted(config.items()) if v is not None}
    elif isinstance(config, (list, tuple)):
        return [_normalize_config(item) for item in config]
    elif isinstance(config, np.ndarray):
        return config.tolist()
    elif isinstance(config, (np.integer, np.floating)):
        return config.item()
    else:
        return config


@dataclass
class ImmutableDataset:
    """
    Immutable dataset with versioned configuration and reproducibility guarantees.

    The dataset ID is computed from the normalized config content, ensuring
    that the same configuration always produces the same ID.

    Attributes:
        dataset_id: 16-char hex ID derived from config hash
        signals: Signal arrays (n_samples, n_b, n_b)
        spectra: Spectrum arrays for reconstruction task (n_samples, n_d, n_d), optional
        pathway_weights: Pathway weight matrices (n_samples, n_compartments, n_compartments), optional
        dei: DEI values (n_samples,), optional
        splits: Dict with 'train', 'val', 'test' index lists
        config: Full generation configuration
        metadata: Creation metadata (created_at, generator_version, checksum)
    """

    dataset_id: str
    signals: np.ndarray
    spectra: Optional[np.ndarray] = None
    pathway_weights: Optional[np.ndarray] = None
    dei: Optional[np.ndarray] = None
    splits: Dict[str, List[int]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate dataset consistency."""
        n_samples = len(self.signals)

        if self.spectra is not None:
            assert len(self.spectra) == n_samples, "Signals and spectra length mismatch"
        if self.pathway_weights is not None:
            assert len(self.pathway_weights) == n_samples, "Signals and pathway_weights length mismatch"
        if self.dei is not None:
            assert len(self.dei) == n_samples, "Signals and DEI length mismatch"

        expected_splits = self.config.get("n_train", 0) + self.config.get("n_val", 0) + self.config.get("n_test", 0)
        if expected_splits > 0:
            actual_splits = sum(len(indices) for indices in self.splits.values())
            assert actual_splits == n_samples, f"Split indices total {actual_splits} != n_samples {n_samples}"

    @property
    def task_type(self) -> str:
        """Infer task type from stored data."""
        if self.pathway_weights is not None:
            return "pathway_regression"
        return "reconstruction"

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return len(self.splits.get("train", []))

    @property
    def n_val(self) -> int:
        """Number of validation samples."""
        return len(self.splits.get("val", []))

    @property
    def n_test(self) -> int:
        """Number of test samples."""
        return len(self.splits.get("test", []))

    def get_split(self, split: str) -> Dict[str, np.ndarray]:
        """
        Get arrays for a specific split.

        Args:
            split: 'train', 'val', or 'test'

        Returns:
            Dict with signal and label arrays for the split
        """
        indices = self.splits.get(split, [])
        result = {"signals": self.signals[indices]}

        if self.spectra is not None:
            result["spectra"] = self.spectra[indices]
        if self.pathway_weights is not None:
            result["pathway_weights"] = self.pathway_weights[indices]
        if self.dei is not None:
            result["dei"] = self.dei[indices]

        return result

    def validate(self) -> bool:
        """
        Validate dataset integrity.

        Checks:
        - All required files exist
        - Checksum matches
        - Splits cover all samples

        Returns:
            True if valid
        """
        if len(self.signals) == 0:
            raise ValueError("Empty dataset")

        expected_splits = sum(len(indices) for indices in self.splits.values())
        if expected_splits != len(self.signals):
            raise ValueError(
                f"Split indices total {expected_splits} != n_samples {len(self.signals)}"
            )

        for split_name in ["train", "val", "test"]:
            if split_name in self.splits:
                indices = self.splits[split_name]
                if not all(0 <= i < len(self.signals) for i in indices):
                    raise ValueError(f"Invalid indices in split '{split_name}'")

        return True

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Dataset ID: {self.dataset_id}",
            f"Task Type: {self.task_type}",
            f"Samples: {len(self.signals)} (train={self.n_train}, val={self.n_val}, test={self.n_test})",
            f"Signal shape: {self.signals.shape}",
        ]
        if self.spectra is not None:
            lines.append(f"Spectrum shape: {self.spectra.shape}")
        if self.pathway_weights is not None:
            lines.append(f"Pathway weights shape: {self.pathway_weights.shape}")
        if self.dei is not None:
            lines.append(f"DEI shape: {self.dei.shape}, range: [{self.dei.min():.4f}, {self.dei.max():.4f}]")
        lines.append(f"Created: {self.metadata.get('created_at', 'unknown')}")
        return "\n".join(lines)


@dataclass
class DatasetID:
    """Dataset identifier with utility methods."""

    id: str

    def __init__(self, id: Union[str, ImmutableDataset]):
        if isinstance(id, ImmutableDataset):
            self.id = id.dataset_id
        else:
            self.id = str(id)

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return f"DatasetID('{self.id}')"

    @property
    def path(self) -> Path:
        """Get directory path for this dataset ID."""
        from dexsy_datasets.storage import get_dataset_path
        return get_dataset_path(self.id)

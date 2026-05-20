"""
Storage layer for immutable datasets.

Handles:
- Directory structure and file I/O
- Checksum computation and verification
- Dataset listing and discovery
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml


def get_dataset_path(dataset_id: str, base_path: Union[str, Path] = "datasets") -> Path:
    """Get directory path for a dataset."""
    return Path(base_path) / dataset_id


def compute_checksum(data_dir: Union[str, Path]) -> str:
    """
    Compute SHA256 checksum of core dataset payload files.

    Intentionally excludes metadata/checksum bookkeeping files to avoid
    self-referential checksum updates.

    Args:
        data_dir: Path to dataset directory

    Returns:
        SHA256 hex digest
    """
    data_dir = Path(data_dir)
    hasher = hashlib.sha256()

    payload_files = [
        "signals.npz",
        "spectra.npz",
        "pathway_weights.npz",
        "dei.npz",
        "splits.json",
        "config.yaml",
    ]
    for name in payload_files:
        file_path = data_dir / name
        if file_path.exists():
            # Include file name and bytes to keep ordering unambiguous.
            hasher.update(name.encode("utf-8"))
            hasher.update(file_path.read_bytes())

    return hasher.hexdigest()


def verify_checksum(data_dir: Union[str, Path], expected_checksum: str) -> bool:
    """
    Verify dataset integrity by comparing checksum.

    Args:
        data_dir: Path to dataset directory
        expected_checksum: Expected SHA256 hex digest

    Returns:
        True if checksum matches
    """
    actual = compute_checksum(data_dir)
    return actual == expected_checksum


def save_dataset(
    dataset,
    base_path: Union[str, Path] = "datasets",
    generator_version: str = "1.0.0",
) -> Path:
    """
    Save immutable dataset to disk.

    Creates directory structure:
        datasets/<dataset_id>/
        ├── signals.npz
        ├── spectra.npz (optional)
        ├── pathway_weights.npz (optional)
        ├── dei.npz (optional)
        ├── splits.json
        ├── config.yaml
        ├── metadata.json
        └── checksum.sha256

    Args:
        dataset: ImmutableDataset instance
        base_path: Base directory for datasets
        generator_version: Version of the generator code

    Returns:
        Path to saved dataset directory
    """
    data_dir = get_dataset_path(dataset.dataset_id, base_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(data_dir / "signals.npz", signals=dataset.signals)

    if dataset.spectra is not None:
        np.savez_compressed(data_dir / "spectra.npz", spectra=dataset.spectra)

    if dataset.pathway_weights is not None:
        np.savez_compressed(
            data_dir / "pathway_weights.npz", pathway_weights=dataset.pathway_weights
        )

    if dataset.dei is not None:
        np.savez_compressed(data_dir / "dei.npz", dei=dataset.dei)

    with open(data_dir / "splits.json", "w") as f:
        json.dump(dataset.splits, f)

    with open(data_dir / "config.yaml", "w") as f:
        yaml.dump(dataset.config, f, default_flow_style=False, sort_keys=False)

    # Compute checksum from payload files only (no self-reference).
    checksum = compute_checksum(data_dir)

    metadata = {
        **dataset.metadata,
        "dataset_id": dataset.dataset_id,
        "created_at": dataset.metadata.get("created_at", datetime.utcnow().isoformat()),
        "generator_version": generator_version,
        "checksum": checksum,
    }
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(data_dir / "checksum.sha256", "w") as f:
        f.write(checksum)

    return data_dir


def load_dataset(
    dataset_id: str,
    base_path: Union[str, Path] = "datasets",
    verify: bool = True,
) -> "ImmutableDataset":
    """
    Load immutable dataset from disk.

    Args:
        dataset_id: Dataset identifier
        base_path: Base directory for datasets
        verify: Whether to verify checksum

    Returns:
        ImmutableDataset instance

    Raises:
        FileNotFoundError: Dataset directory not found
        ValueError: Checksum verification failed
    """
    from dexsy_datasets.core import ImmutableDataset

    data_dir = get_dataset_path(dataset_id, base_path)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    signals = np.load(data_dir / "signals.npz")["signals"]

    spectra = None
    if (data_dir / "spectra.npz").exists():
        spectra = np.load(data_dir / "spectra.npz")["spectra"]

    pathway_weights = None
    if (data_dir / "pathway_weights.npz").exists():
        pathway_weights = np.load(data_dir / "pathway_weights.npz")["pathway_weights"]

    dei = None
    if (data_dir / "dei.npz").exists():
        dei = np.load(data_dir / "dei.npz")["dei"]

    with open(data_dir / "splits.json", "r") as f:
        splits = json.load(f)
    splits = {k: list(v) for k, v in splits.items()}

    with open(data_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    if verify:
        expected_checksum = metadata.get("checksum")
        actual = compute_checksum(data_dir)
        with open(data_dir / "checksum.sha256", "r") as f:
            stored = f.read().strip()

        if expected_checksum and expected_checksum != actual:
            raise ValueError(f"Checksum verification failed for dataset {dataset_id}")
        if stored != actual:
            raise ValueError(f"Checksum file mismatch for dataset {dataset_id}")

    return ImmutableDataset(
        dataset_id=dataset_id,
        signals=signals,
        spectra=spectra,
        pathway_weights=pathway_weights,
        dei=dei,
        splits=splits,
        config=config,
        metadata=metadata,
    )


def list_datasets(
    base_path: Union[str, Path] = "datasets",
    include_metadata: bool = False,
) -> List[Union[str, Dict[str, Any]]]:
    """
    List all datasets in the base directory.

    Args:
        base_path: Base directory for datasets
        include_metadata: Whether to include metadata for each dataset

    Returns:
        List of dataset IDs or dicts with metadata
    """
    base_path = Path(base_path)

    if not base_path.exists():
        return []

    datasets = []
    for dataset_dir in sorted(base_path.iterdir()):
        if not dataset_dir.is_dir():
            continue

        metadata_path = dataset_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        if include_metadata:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            datasets.append(metadata)
        else:
            datasets.append(dataset_dir.name)

    return datasets


def dataset_exists(dataset_id: str, base_path: Union[str, Path] = "datasets") -> bool:
    """Check if a dataset exists."""
    return get_dataset_path(dataset_id, base_path).exists()


def get_dataset_info(
    dataset_id: str,
    base_path: Union[str, Path] = "datasets",
) -> Dict[str, Any]:
    """
    Get summary info for a dataset.

    Args:
        dataset_id: Dataset identifier
        base_path: Base directory for datasets

    Returns:
        Dict with dataset info
    """
    data_dir = get_dataset_path(dataset_id, base_path)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    with open(data_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    signals = np.load(data_dir / "signals.npz")["signals"]

    return {
        "dataset_id": dataset_id,
        "path": str(data_dir),
        "task_type": config.get("task_type"),
        "model_type": config.get("model_type"),
        "n_samples": len(signals),
        "n_b": config.get("n_b"),
        "created_at": metadata.get("created_at"),
        "generator_version": metadata.get("generator_version"),
        "checksum": metadata.get("checksum"),
    }

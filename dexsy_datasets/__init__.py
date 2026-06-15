"""
dexsy_datasets - Immutable dataset management for DEXSY experiments.

Provides:
    - ImmutableDataset: Versioned, reproducible datasets
    - DatasetConfig / TrainingRunConfig: Separated configuration schemas
    - SeedManager: Unified random seed control
    - create_dataset, load_dataset, extend_dataset: Main API functions
"""

from dexsy_datasets.core import ImmutableDataset, DatasetID, compute_dataset_id
from dexsy_datasets.config import DatasetConfig, load_config, validate_config, DEFAULT_CONFIG
from dexsy_datasets.run_config import TrainingRunConfig, load_run_config
from dexsy_datasets.seeds import SeedManager, fix_all_seeds
from dexsy_datasets.storage import (
    save_dataset,
    load_dataset,
    list_datasets,
    compute_checksum,
    verify_checksum,
    dataset_exists,
    get_dataset_info,
)
from dexsy_datasets.generators import generate_nongaussian_dataset, generate_gaussian_dataset

__version__ = "1.0.0"

__all__ = [
    # Core
    "ImmutableDataset",
    "DatasetID",
    "compute_dataset_id",
    # Config
    "DatasetConfig",
    "load_config",
    "validate_config",
    "DEFAULT_CONFIG",
    "TrainingRunConfig",
    "load_run_config",
    # Seeds
    "SeedManager",
    "fix_all_seeds",
    # Storage
    "save_dataset",
    "load_dataset",
    "list_datasets",
    "compute_checksum",
    "verify_checksum",
    "dataset_exists",
    "get_dataset_info",
    # Generators
    "generate_nongaussian_dataset",
    "generate_gaussian_dataset",
]


def create_dataset(
    config: "DatasetConfig | dict | str",
    output_dir: str = "datasets",
    generator_version: str = "1.0.0",
) -> ImmutableDataset:
    """Create and save a new dataset."""
    if isinstance(config, str):
        config = load_config(config)

    errors = validate_config(config)
    if errors:
        raise ValueError(f"Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    if isinstance(config, dict):
        config = DatasetConfig.from_dict(config)

    model_type = config.model_type
    if model_type == "nongaussian_3c":
        dataset = generate_nongaussian_dataset(config, output_dir, generator_version)
    elif model_type in ["gaussian_2c", "gaussian_3c", "gaussian_nc"]:
        dataset = generate_gaussian_dataset(config, output_dir, generator_version)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return dataset


def extend_dataset(
    base_dataset_id: str,
    n_add_train: int,
    output_dir: str = "datasets",
    generator_version: str = "1.0.0",
) -> ImmutableDataset:
    """Extend an existing dataset by adding training samples."""
    base = load_dataset(base_dataset_id, output_dir)

    config = DatasetConfig.from_dict(base.config)
    config.n_train = config.n_train + n_add_train

    return create_dataset(config, output_dir, generator_version)

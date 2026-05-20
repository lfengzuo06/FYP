"""
Dataset configuration schema and validation.

Separates dataset generation config from training run config.
DatasetConfig contains only data generation parameters.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# Schema version for config format
PROTOCOL_VERSION = "1"
SUPPORTED_TASK_TYPES = ["reconstruction", "pathway_regression"]
SUPPORTED_MODEL_TYPES = ["gaussian_2c", "gaussian_3c", "gaussian_nc", "nongaussian_3c"]


DEFAULT_CONFIG = {
    "version": "1.0",
    "protocol_version": PROTOCOL_VERSION,
    "task_type": "pathway_regression",
    "n_train": 9500,
    "n_val": 400,
    "n_test": 400,
    "n_b": 16,
    "model_type": "nongaussian_3c",
    "params": {
        "extracellular_diffusivity_range": [1.0e-9, 2.5e-9],
        "intracellular_diffusivity_range": [0.4e-9, 1.2e-9],
        "axon_restricted_length_range": [0.5e-6, 2.0e-6],
        "sphere_radius_range": [1.0e-6, 6.0e-6],
        "mixing_time_range": [0.015, 0.300],
        "extracellular_fraction_range": [0.3, 0.7],
        "axon_fraction_range": [0.1, 0.4],
        "sphere_fraction_range": [0.1, 0.4],
        "noise_sigma": None,
    },
    "sampling_strategy": "log_uniform",
    "min_index_separation": 0,
    "seed": 42,
}


class DatasetConfig:
    """
    Dataset generation configuration.

    Contains only data generation parameters (not training parameters).
    Separated from TrainingRunConfig to ensure dataset reproducibility
    independent of training choices.

    Attributes:
        task_type: 'reconstruction' (signal->spectrum) or 'pathway_regression' (signal->pathway_weights)
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        n_b: Grid size
        model_type: Forward model type
        params: Physical parameters for the forward model
        sampling_strategy: 'log_uniform' or 'uniform'
        seed: Base random seed for data generation
    """

    def __init__(
        self,
        task_type: str = "pathway_regression",
        n_train: int = 9500,
        n_val: int = 400,
        n_test: int = 400,
        n_b: int = 16,
        model_type: str = "nongaussian_3c",
        params: Optional[Dict[str, Any]] = None,
        sampling_strategy: str = "log_uniform",
        min_index_separation: int = 0,
        seed: int = 42,
        version: str = "1.0",
        protocol_version: str = PROTOCOL_VERSION,
        generator_version: str = "1.0.0",
    ):
        self.version = version
        self.protocol_version = protocol_version
        self.generator_version = generator_version
        self.task_type = task_type
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.n_b = n_b
        self.model_type = model_type
        self.params = params or {}
        self.sampling_strategy = sampling_strategy
        self.min_index_separation = min_index_separation
        self.seed = seed

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DatasetConfig":
        """Create config from dictionary."""
        return cls(
            version=config.get("version", "1.0"),
            protocol_version=config.get("protocol_version", PROTOCOL_VERSION),
            generator_version=config.get("generator_version", "1.0.0"),
            task_type=config.get("task_type", "pathway_regression"),
            n_train=config.get("n_train", 9500),
            n_val=config.get("n_val", 400),
            n_test=config.get("n_test", 400),
            n_b=config.get("n_b", 16),
            model_type=config.get("model_type", "nongaussian_3c"),
            params=config.get("params", {}),
            sampling_strategy=config.get("sampling_strategy", "log_uniform"),
            min_index_separation=config.get("min_index_separation", 0),
            seed=config.get("seed", 42),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DatasetConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "protocol_version": self.protocol_version,
            "generator_version": self.generator_version,
            "task_type": self.task_type,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "n_b": self.n_b,
            "model_type": self.model_type,
            "params": self.params,
            "sampling_strategy": self.sampling_strategy,
            "min_index_separation": self.min_index_separation,
            "seed": self.seed,
        }

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def get_generator_seeds(self) -> Dict[str, int]:
        """Get seeds for each split (train=seed, val=seed+1, test=seed+2)."""
        return {
            "train": self.seed,
            "val": self.seed + 1,
            "test": self.seed + 2,
        }

    def __repr__(self) -> str:
        return (
            f"DatasetConfig(task_type={self.task_type}, "
            f"n_train={self.n_train}, n_val={self.n_val}, n_test={self.n_test}, "
            f"model_type={self.model_type}, seed={self.seed})"
        )


def validate_config(config: Union[Dict[str, Any], DatasetConfig]) -> List[str]:
    """
    Validate dataset configuration.

    Args:
        config: Config dict or DatasetConfig object

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if isinstance(config, DatasetConfig):
        config = config.to_dict()

    task_type = config.get("task_type")
    if task_type not in SUPPORTED_TASK_TYPES:
        errors.append(f"task_type must be one of {SUPPORTED_TASK_TYPES}, got '{task_type}'")

    model_type = config.get("model_type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        errors.append(f"model_type must be one of {SUPPORTED_MODEL_TYPES}, got '{model_type}'")

    n_train = config.get("n_train", 0)
    n_val = config.get("n_val", 0)
    n_test = config.get("n_test", 0)

    if n_train < 0:
        errors.append(f"n_train must be non-negative, got {n_train}")
    if n_val < 0:
        errors.append(f"n_val must be non-negative, got {n_val}")
    if n_test < 0:
        errors.append(f"n_test must be non-negative, got {n_test}")
    if (n_train + n_val + n_test) <= 0:
        errors.append("At least one of n_train/n_val/n_test must be > 0.")

    n_b = config.get("n_b")
    if n_b not in [16, 64]:
        errors.append(f"n_b must be 16 or 64, got {n_b}")

    seed = config.get("seed")
    if seed is None or not isinstance(seed, int):
        errors.append(f"seed must be an integer, got {seed}")

    params = config.get("params", {})

    if model_type == "nongaussian_3c":
        required_params = [
            "extracellular_diffusivity_range",
            "intracellular_diffusivity_range",
            "axon_restricted_length_range",
            "sphere_radius_range",
            "mixing_time_range",
        ]
        for param in required_params:
            if param not in params:
                errors.append(f"model_type '{model_type}' requires params.{param}")

    elif model_type in ["gaussian_2c", "gaussian_3c"]:
        required_params = [
            "d_min",
            "d_max",
            "g_max",
            "delta",
            "DELTA",
        ]
        for param in required_params:
            if param not in params:
                errors.append(f"model_type '{model_type}' requires params.{param}")
    elif model_type == "gaussian_nc":
        n_compartments = int(params.get("n_compartments", 3))
        if n_compartments < 2:
            errors.append(
                f"model_type '{model_type}' requires params.n_compartments >= 2, got {n_compartments}"
            )

    protocol_version = config.get("protocol_version")
    if protocol_version != PROTOCOL_VERSION:
        errors.append(
            f"protocol_version mismatch: expected {PROTOCOL_VERSION}, got {protocol_version}"
        )

    return errors


def load_config(path: Union[str, Path]) -> DatasetConfig:
    """
    Load and validate dataset config from YAML file.

    Args:
        path: Path to config YAML file

    Returns:
        Validated DatasetConfig

    Raises:
        FileNotFoundError: Config file not found
        ValueError: Config validation failed
    """
    config = DatasetConfig.from_yaml(path)
    errors = validate_config(config)
    if errors:
        raise ValueError(f"Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    return config

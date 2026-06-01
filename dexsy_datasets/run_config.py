"""
Training run configuration schema.

Separated from DatasetConfig to ensure training parameters (init_seed, dataloader_seed)
are not mixed with data generation parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


PROTOCOL_VERSION = "1"


@dataclass
class TrainingRunConfig:
    """
    Training run configuration.

    Contains only training-specific parameters, not data generation.
    References a dataset by ID and specifies training hyperparameters.

    Attributes:
        dataset_id: ID of the immutable dataset to train on
        model: Model name (e.g., 'attention_unet', 'plain_unet', 'cnn_nongaussian')
        init_seed: Random seed for model initialization
        dataloader_seed: Random seed for dataloader shuffling
        model_kwargs: Model-specific hyperparameters
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        early_stopping_patience: Epochs to wait before early stopping
        reduce_lr_patience: Epochs before reducing learning rate
        reduce_lr_factor: Learning rate reduction factor
    """

    version: str = "1.0"
    dataset_id: str = ""
    model: str = "attention_unet"
    init_seed: int = 42
    dataloader_seed: int = 42
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 12
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    output_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TrainingRunConfig":
        """Create config from dictionary."""
        return cls(
            version=config.get("version", "1.0"),
            dataset_id=config.get("dataset_id", ""),
            model=config.get("model", "attention_unet"),
            init_seed=config.get("init_seed", 42),
            dataloader_seed=config.get("dataloader_seed", 42),
            model_kwargs=config.get("model_kwargs", {}),
            epochs=config.get("epochs", 80),
            batch_size=config.get("batch_size", 128),
            learning_rate=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
            early_stopping_patience=config.get("early_stopping_patience", 12),
            reduce_lr_patience=config.get("reduce_lr_patience", 5),
            reduce_lr_factor=config.get("reduce_lr_factor", 0.5),
            output_dir=config.get("output_dir"),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingRunConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "dataset_id": self.dataset_id,
            "model": self.model,
            "init_seed": self.init_seed,
            "dataloader_seed": self.dataloader_seed,
            "model_kwargs": self.model_kwargs,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "early_stopping_patience": self.early_stopping_patience,
            "reduce_lr_patience": self.reduce_lr_patience,
            "reduce_lr_factor": self.reduce_lr_factor,
            "output_dir": self.output_dir,
        }

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        return (
            f"TrainingRunConfig(dataset_id={self.dataset_id[:8]}..., "
            f"model={self.model}, init_seed={self.init_seed})"
        )


def load_run_config(path: Union[str, Path]) -> TrainingRunConfig:
    """Load training run config from YAML file."""
    return TrainingRunConfig.from_yaml(path)

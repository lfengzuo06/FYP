"""
Configuration for Active Learning Agent.

This module defines the configuration dataclass for the AL agent,
including dataset sizes, AL parameters, and training settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ALConfig:
    """
    Configuration for the Active Learning Agent.

    Attributes:
        baseline_checkpoint: Path to the baseline PINN checkpoint.
        output_dir: Directory for saving AL run outputs.
        experiment_name: Name of the experiment (used in subdirectories).
        seed: Random seed for reproducibility.

        # Dataset sizes
        n_train_base: Size of initial training set.
        n_candidate_pool: Size of AL sampling pool.
        n_val_fixed: Size of fixed validation set (never participates in AL).
        n_test_fixed: Size of fixed test set (never participates in AL).

        # Data generation
        noise_sigma_range: Range for noise sigma sampling.
        batch_size: Batch size for training.

        # AL parameters
        hard_ratio: Fraction of hard cases in mixed training set (0.3 = 30%).
        top_k_ratio: Fraction of top candidates selected as hard seeds (0.15 = 15%).
        n_augmentations_per_seed: Number of augmented samples per hard seed.
        max_rounds: Maximum number of AL rounds.
        improvement_threshold: Minimum improvement threshold for stopping (as fraction).
        min_stopping_rounds: Minimum consecutive rounds with < threshold improvement to stop.

        # Fine-tuning parameters
        lr_ratio: Learning rate multiplier for fine-tuning (0.3 = 30% of original).
        finetune_epochs: Maximum epochs for fine-tuning.
        early_stopping_patience: Patience for early stopping during fine-tuning.
        reduce_lr_patience: Patience for learning rate reduction.
        reduce_lr_factor: Factor for learning rate reduction.

        # Failure grouping
        max_per_failure_type: Maximum samples per failure type group.
        min_group_size: Minimum samples required in a group to be included.

        # Model parameters (inherited from training)
        base_filters: Base number of filters for the model.
        in_channels: Number of input channels.

        # Paths
        forward_model_config: Additional forward model configuration.
    """

    # Paths
    baseline_checkpoint: str | Path | None = None
    output_dir: str | Path = "active_learning_runs"
    experiment_name: str = "al_pinn_3c"
    seed: int = 42

    # Dataset sizes
    n_train_base: int = 5000
    n_candidate_pool: int = 10000
    n_val_fixed: int = 500
    n_test_fixed: int = 500

    # Data generation
    noise_sigma_range: tuple[float, float] = (0.005, 0.015)
    batch_size: int = 8

    # AL parameters
    hard_ratio: float = 0.3
    top_k_ratio: float = 0.15
    n_augmentations_per_seed: int = 4
    max_rounds: int = 4
    improvement_threshold: float = 0.01
    min_stopping_rounds: int = 2

    # Fine-tuning parameters
    lr_ratio: float = 0.3
    finetune_epochs: int = 30
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5

    # Failure grouping
    max_per_failure_type: int = 3
    min_group_size: int = 2

    # Model parameters
    base_filters: int = 64
    in_channels: int = 3
    original_lr: float = 1e-3

    # Perturbation bounds for augmentation
    rate_perturbation_factor: float = 0.2
    vf_perturbation_delta: float = 0.05
    peak_position_perturbation: int = 4
    noise_levels_for_augmentation: tuple[float, ...] = (0.005, 0.01, 0.015, 0.02)

    def __post_init__(self):
        """Process paths and set defaults."""
        if self.baseline_checkpoint is None:
            # Try to find the most recent baseline checkpoint
            default_path = Path(__file__).parent.parent / "checkpoints_3d" / "pinn_3c"
            if default_path.exists():
                best_models = sorted(default_path.glob("*/best_model.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                if best_models:
                    self.baseline_checkpoint = str(best_models[0])

        if self.baseline_checkpoint is not None:
            self.baseline_checkpoint = Path(self.baseline_checkpoint)

        self.output_dir = Path(self.output_dir) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def finetune_lr(self) -> float:
        """Compute fine-tuning learning rate."""
        return self.original_lr * self.lr_ratio

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, tuple):
                result[key] = list(value)
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ALConfig:
        """Create config from dictionary."""
        # Filter out computed properties
        d = {k: v for k, v in d.items() if not k.startswith('_') and k != 'finetune_lr'}
        return cls(**d)


@dataclass
class FailureTypeConfig:
    """
    Configuration for failure type detection thresholds.

    These thresholds determine how samples are classified into failure types.
    """

    # Exchange rate thresholds
    high_exchange_threshold: float = 10.0
    low_exchange_threshold: float = 1.0

    # Peak separation thresholds
    small_separation_threshold: int = 8
    large_separation_threshold: int = 20

    # Grid position thresholds
    edge_margin: int = 10

    # Noise thresholds
    high_noise_threshold: float = 0.012
    low_noise_threshold: float = 0.007

    # Volume fraction thresholds
    vf_imbalance_threshold: float = 0.4

    # Rate asymmetry threshold
    rate_asymmetry_threshold: float = 3.0

    # DEI thresholds
    high_dei_threshold: float = 0.3
    low_dei_threshold: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# Default failure type configuration
DEFAULT_FAILURE_CONFIG = FailureTypeConfig()

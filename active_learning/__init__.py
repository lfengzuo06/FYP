"""
Active Learning Agent for 3C PINN Model Optimization.

This package provides an active learning framework for iterative refinement
of the 3-compartment DEXSY PINN model through targeted hard case mining,
failure attribution, and curriculum-based retraining.

Example usage:
    from active_learning import ActiveLearningAgent, ALConfig

    config = ALConfig(
        baseline_checkpoint="checkpoints_3d/pinn_3c/...",
        experiment_name="run_001",
    )
    agent = ActiveLearningAgent(config)
    agent.setup()
    agent.run()

Or via CLI:
    python -m active_learning.scripts.run_al \
        --baseline_checkpoint checkpoints_3d/pinn_3c/... \
        --experiment_name my_experiment \
        --max_rounds 4
"""

from .config import ALConfig, FailureTypeConfig, DEFAULT_FAILURE_CONFIG
from .data_protocol import DataProtocol, SampleMetadata, SplitData
from .scorer import HardCaseScorer, SampleScores
from .failure_attributor import FailureAttributor, FailureGroup
from .augmentation import HardCaseAugmenter, AugmentedSample, BalancedAugmenter
from .trainer import FineTuningTrainer, TrainingResult
from .evaluator import ALEvaluator, ExtendedMetrics, RoundMetrics
from .agent import ActiveLearningAgent, ALState, run_active_learning

__all__ = [
    # Config
    "ALConfig",
    "FailureTypeConfig",
    "DEFAULT_FAILURE_CONFIG",
    # Data Protocol
    "DataProtocol",
    "SampleMetadata",
    "SplitData",
    # Scorer
    "HardCaseScorer",
    "SampleScores",
    # Failure Attributor
    "FailureAttributor",
    "FailureGroup",
    # Augmentation
    "HardCaseAugmenter",
    "AugmentedSample",
    "BalancedAugmenter",
    # Trainer
    "FineTuningTrainer",
    "TrainingResult",
    # Evaluator
    "ALEvaluator",
    "ExtendedMetrics",
    "RoundMetrics",
    # Agent
    "ActiveLearningAgent",
    "ALState",
    "run_active_learning",
]

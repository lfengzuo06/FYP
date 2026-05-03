"""
CLI entry point for Active Learning Agent.

Usage:
    python -m active_learning.scripts.run_al --config config.json

Or with arguments:
    python -m active_learning.scripts.run_al \
        --baseline_checkpoint checkpoints_3d/pinn_3c/... \
        --experiment_name my_experiment \
        --n_train_base 5000 \
        --n_candidate_pool 10000 \
        --max_rounds 4 \
        --hard_ratio 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dexsy_core.forward_model import ForwardModel2D
from active_learning import ALConfig, ActiveLearningAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Active Learning Agent for 3C PINN optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument(
        '--baseline_checkpoint',
        type=str,
        default=None,
        help='Path to baseline PINN checkpoint',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='active_learning_runs',
        help='Output directory for results',
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='al_pinn_3c',
        help='Name of the experiment',
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default=None,
        help='Path to JSON config file (overrides other args)',
    )

    # Dataset sizes
    parser.add_argument(
        '--n_train_base',
        type=int,
        default=5000,
        help='Size of base training set',
    )
    parser.add_argument(
        '--n_candidate_pool',
        type=int,
        default=10000,
        help='Size of candidate pool',
    )
    parser.add_argument(
        '--n_val_fixed',
        type=int,
        default=500,
        help='Size of fixed validation set',
    )
    parser.add_argument(
        '--n_test_fixed',
        type=int,
        default=500,
        help='Size of fixed test set',
    )

    # AL parameters
    parser.add_argument(
        '--hard_ratio',
        type=float,
        default=0.3,
        help='Fraction of hard cases in training (0.0-1.0)',
    )
    parser.add_argument(
        '--top_k_ratio',
        type=float,
        default=0.15,
        help='Fraction of top candidates selected as hard seeds',
    )
    parser.add_argument(
        '--n_augmentations_per_seed',
        type=int,
        default=4,
        help='Number of augmented samples per hard seed',
    )
    parser.add_argument(
        '--max_rounds',
        type=int,
        default=4,
        help='Maximum number of AL rounds',
    )
    parser.add_argument(
        '--improvement_threshold',
        type=float,
        default=0.01,
        help='Minimum improvement threshold for stopping (as fraction)',
    )

    # Fine-tuning parameters
    parser.add_argument(
        '--lr_ratio',
        type=float,
        default=0.3,
        help='Learning rate multiplier for fine-tuning',
    )
    parser.add_argument(
        '--finetune_epochs',
        type=int,
        default=30,
        help='Maximum epochs for fine-tuning',
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=8,
        help='Early stopping patience',
    )

    # Other
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    parser.add_argument(
        '--force_regenerate_splits',
        action='store_true',
        help='Force regenerate data splits even if they exist',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)',
    )

    return parser.parse_args()


def create_config_from_args(args) -> ALConfig:
    """Create ALConfig from parsed arguments."""
    # Load from config file if provided
    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        return ALConfig.from_dict(config_dict)

    # Create from arguments
    return ALConfig(
        baseline_checkpoint=args.baseline_checkpoint,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        seed=args.seed,
        n_train_base=args.n_train_base,
        n_candidate_pool=args.n_candidate_pool,
        n_val_fixed=args.n_val_fixed,
        n_test_fixed=args.n_test_fixed,
        hard_ratio=args.hard_ratio,
        top_k_ratio=args.top_k_ratio,
        n_augmentations_per_seed=args.n_augmentations_per_seed,
        max_rounds=args.max_rounds,
        improvement_threshold=args.improvement_threshold,
        lr_ratio=args.lr_ratio,
        finetune_epochs=args.finetune_epochs,
        early_stopping_patience=args.early_stopping_patience,
    )


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 70)
    print("Active Learning Agent for 3C PINN Optimization")
    print("=" * 70)

    # Create config
    config = create_config_from_args(args)

    print("\nConfiguration:")
    print(f"  Baseline checkpoint: {config.baseline_checkpoint}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Experiment name: {config.experiment_name}")
    print(f"  Seed: {config.seed}")
    print(f"\nDataset sizes:")
    print(f"  Train base: {config.n_train_base}")
    print(f"  Candidate pool: {config.n_candidate_pool}")
    print(f"  Val fixed: {config.n_val_fixed}")
    print(f"  Test fixed: {config.n_test_fixed}")
    print(f"\nAL parameters:")
    print(f"  Hard ratio: {config.hard_ratio}")
    print(f"  Top-K ratio: {config.top_k_ratio}")
    print(f"  Augmentations per seed: {config.n_augmentations_per_seed}")
    print(f"  Max rounds: {config.max_rounds}")
    print(f"  Improvement threshold: {config.improvement_threshold}")
    print(f"\nFine-tuning:")
    print(f"  LR ratio: {config.lr_ratio}")
    print(f"  Finetune epochs: {config.finetune_epochs}")
    print(f"  Early stopping patience: {config.early_stopping_patience}")

    # Validate baseline checkpoint
    if config.baseline_checkpoint is None:
        print("\n[ERROR] No baseline checkpoint specified!")
        print("Please provide --baseline_checkpoint argument.")
        sys.exit(1)

    if not config.baseline_checkpoint.exists():
        print(f"\n[ERROR] Baseline checkpoint not found: {config.baseline_checkpoint}")
        sys.exit(1)

    # Create forward model
    forward_model = ForwardModel2D(n_d=64, n_b=64)

    # Create and run agent
    agent = ActiveLearningAgent(
        config=config,
        forward_model=forward_model,
        device=args.device,
    )

    # Setup
    agent.setup(force_regenerate_splits=args.force_regenerate_splits)

    # Run
    agent.run()

    # Print summary
    agent.print_summary()

    print("\nDone!")


if __name__ == "__main__":
    main()

"""
Active Learning Agent for 3C PINN Model Optimization.

This module provides the main orchestrator for the active learning loop,
coordinating all components: data protocol, scoring, failure attribution,
augmentation, training, and evaluation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import sys
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dexsy_core.forward_model import ForwardModel2D
from .config import ALConfig
from .data_protocol import DataProtocol, SplitData, SampleMetadata
from .scorer import HardCaseScorer, SampleScores
from .failure_attributor import FailureAttributor, FailureGroup
from .augmentation import HardCaseAugmenter, AugmentedSample
from .trainer import FineTuningTrainer, TrainingResult
from .evaluator import ALEvaluator, ExtendedMetrics, RoundMetrics


@dataclass
class ALState:
    """State of the active learning agent."""
    current_round: int = 0
    current_checkpoint: Path | None = None
    round_history: list = field(default_factory=list)
    baseline_metrics: dict | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'current_round': self.current_round,
            'current_checkpoint': str(self.current_checkpoint) if self.current_checkpoint else None,
            'round_history': self.round_history,
            'baseline_metrics': self.baseline_metrics,
        }


class ActiveLearningAgent:
    """
    Main orchestrator for the active learning loop.

    Coordinates all components:
    1. Data protocol management
    2. Candidate pool scoring
    3. Failure type attribution
    4. Hard case augmentation
    5. Fine-tuning
    6. Evaluation
    """

    def __init__(
        self,
        config: ALConfig,
        forward_model: ForwardModel2D | None = None,
        device: str | torch.device | None = None,
    ):
        """
        Initialize the Active Learning Agent.

        Args:
            config: ALConfig with all parameters.
            forward_model: ForwardModel2D instance.
            device: Device to use ('cuda', 'cpu', or None for auto).
        """
        self.config = config
        self.forward_model = forward_model or ForwardModel2D(n_d=64, n_b=64)

        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Initialize components
        self.data_protocol = DataProtocol(
            config=config,
            forward_model=self.forward_model,
        )
        self.scorer = HardCaseScorer(forward_model=self.forward_model)
        self.failure_attributor = FailureAttributor(max_per_group=config.max_per_failure_type)
        self.augmenter = HardCaseAugmenter(forward_model=self.forward_model)
        self.trainer = FineTuningTrainer(
            config=config,
            forward_model=self.forward_model,
            device=device,
        )
        self.evaluator = ALEvaluator(forward_model=self.forward_model)

        # State
        self.state = ALState()
        self.state.current_checkpoint = config.baseline_checkpoint

        # Results
        self.baseline_metrics: ExtendedMetrics | None = None
        self.round_results: list[RoundMetrics] = []

    def setup(self, force_regenerate_splits: bool = False) -> None:
        """
        Setup the data splits.

        Args:
            force_regenerate_splits: If True, regenerate splits even if they exist.
        """
        print("\n" + "=" * 70)
        print("ACTIVE LEARNING AGENT SETUP")
        print("=" * 70)

        # Generate or load data splits
        self.data_protocol.setup(force_regenerate=force_regenerate_splits)

        # Print summary
        summary = self.data_protocol.summary()
        print("\nData Splits:")
        for name, info in summary.items():
            print(f"  {name}: {info['n_samples']} samples (seeds {info['seeds']})")

        # Evaluate baseline
        print("\nEvaluating baseline model...")
        self.baseline_metrics = self._evaluate_baseline()
        self.state.baseline_metrics = self.baseline_metrics.to_dict()
        self.evaluator.print_metrics(self.baseline_metrics, "BASELINE METRICS")

        print("Setup complete!")

    def run(self) -> None:
        """
        Run the active learning loop.

        Executes multiple rounds of:
        1. Score candidate pool
        2. Select hard seeds
        3. Attribute failures
        4. Augment hard cases
        5. Fine-tune model
        6. Evaluate
        """
        print("\n" + "=" * 70)
        print("STARTING ACTIVE LEARNING LOOP")
        print("=" * 70)
        print(f"Max rounds: {self.config.max_rounds}")
        print(f"Improvement threshold: {self.config.improvement_threshold * 100:.1f}%")
        print(f"Hard ratio: {self.config.hard_ratio * 100:.0f}%")
        print(f"Top-K ratio: {self.config.top_k_ratio * 100:.0f}%")
        print("=" * 70)

        total_start = time.time()

        for round_idx in range(self.config.max_rounds):
            print(f"\n{'#'*70}")
            print(f"# ROUND {round_idx + 1} / {self.config.max_rounds}")
            print(f"{'#'*70}")

            round_start = time.time()

            try:
                # Step 1: Score candidate pool
                scores_df, hard_seeds = self._score_candidate_pool()

                # Step 2: Attribute failures
                failure_groups = self._attribute_failures(hard_seeds)

                # Step 3: Augment hard cases
                augmented_samples = self._augment_hard_cases(failure_groups)

                # Step 4: Fine-tune
                training_result = self._finetune(round_idx, augmented_samples)

                # Step 5: Evaluate
                round_metrics = self._evaluate_round(training_result.checkpoint_path, training_result)

                # Store results
                self.round_results.append(round_metrics)

                # Check stopping condition
                if self._check_stopping():
                    print("\nStopping condition met. Terminating AL loop.")
                    break

                self.state.current_round = round_idx + 1

            except Exception as e:
                print(f"\nError in round {round_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue

            round_time = time.time() - round_start
            print(f"\nRound {round_idx + 1} completed in {round_time:.1f}s")

        total_time = time.time() - total_start
        print(f"\n{'='*70}")
        print("ACTIVE LEARNING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Rounds completed: {len(self.round_results)}")

        # Save final results
        self._save_results()

    def _score_candidate_pool(self) -> tuple[pd.DataFrame, list[SampleScores]]:
        """
        Score the candidate pool and select hard seeds.

        Returns:
            Tuple of (scores DataFrame, hard seeds list).
        """
        print("\n[Step 1] Scoring candidate pool...")

        # Get candidate pool
        candidate_pool = self.data_protocol.get_candidate_pool()

        # Run inference on candidate pool using 3-channel model inputs
        predictions = self._predict_batch(candidate_pool.model_inputs)

        # signals are now (N, 64, 64) for residual computation
        # Ground truths are (N, 64, 64)
        scores_list = self.scorer.compute_scores(
            predictions=predictions,
            ground_truths=candidate_pool.labels[:, 0],  # (N, 64, 64)
            signals=candidate_pool.signals,  # (N, 64, 64)
            metadata=candidate_pool.metadata,
        )

        # Normalize
        scores_list = self.scorer.normalize_scores(scores_list)

        # Print summary
        self.scorer.print_summary(scores_list)

        # Select hard seeds
        hard_seeds = self.scorer.select_hard_seeds(
            scores_list,
            top_k_ratio=self.config.top_k_ratio,
        )

        print(f"Selected {len(hard_seeds)} hard seeds (top {self.config.top_k_ratio * 100:.0f}%)")

        # Convert to dataframe
        df = self.scorer.to_dataframe(scores_list)

        return df, hard_seeds

    def _attribute_failures(self, hard_seeds: list[SampleScores]) -> dict[str, FailureGroup]:
        """
        Attribute hard seeds to failure types.

        Args:
            hard_seeds: List of hard seed scores.

        Returns:
            Dictionary of failure groups.
        """
        print("\n[Step 2] Attributing failures...")

        # Create (metadata, scores) pairs
        hard_pairs = [
            (seed.metadata, seed)
            for seed in hard_seeds
            if seed.metadata is not None
        ]

        # Attribute
        failure_groups = self.failure_attributor.attribute(hard_pairs)

        # Print summary
        self.failure_attributor.print_attribution_summary(failure_groups)

        return failure_groups

    def _augment_hard_cases(
        self,
        failure_groups: dict[str, FailureGroup],
    ) -> list[AugmentedSample]:
        """
        Augment hard cases from each failure type.

        Args:
            failure_groups: Dictionary of failure groups.

        Returns:
            List of augmented samples.
        """
        print("\n[Step 3] Augmenting hard cases...")

        # Get unique augmentation targets
        targets = self.failure_attributor.get_augmentation_targets(failure_groups)

        print(f"Augmentation targets: {len(targets)} seeds")

        # Augment
        augmented = self.augmenter.augment_batch(
            seeds=targets,
            n_augmentations=self.config.n_augmentations_per_seed,
        )

        print(f"Generated {len(augmented)} augmented samples")

        return augmented

    def _finetune(
        self,
        round_idx: int,
        augmented_samples: list[AugmentedSample],
    ) -> TrainingResult:
        """
        Fine-tune the model.

        Args:
            round_idx: Current round index.
            augmented_samples: List of augmented samples.

        Returns:
            TrainingResult instance.
        """
        print("\n[Step 4] Fine-tuning model...")

        # Get training data
        train_base = self.data_protocol.get_train_base()
        val_fixed = self.data_protocol.get_val_fixed()

        # Convert augmented samples to training format
        aug_training_format = self.augmenter.convert_to_training_format(augmented_samples)

        # Fine-tune
        result = self.trainer.finetune(
            base_checkpoint=self.state.current_checkpoint,
            train_data=train_base,
            val_data=val_fixed,
            round_idx=round_idx,
            augmented_data=aug_training_format,
            hard_ratio=self.config.hard_ratio,
        )

        # Update current checkpoint
        self.state.current_checkpoint = result.checkpoint_path

        return result

    def _evaluate_baseline(self) -> ExtendedMetrics:
        """
        Evaluate the baseline model.

        Returns:
            ExtendedMetrics for baseline.
        """
        if self.state.current_checkpoint is None:
            raise ValueError("No baseline checkpoint specified")

        print(f"\nEvaluating baseline: {self.state.current_checkpoint}")

        test_fixed = self.data_protocol.get_test_fixed()

        # Load model
        checkpoint = torch.load(
            self.state.current_checkpoint,
            map_location=self.device,
            weights_only=False,
        )

        from models_3d.pinn.model import PINN3C
        model = PINN3C(
            signal_size=64,
            in_channels=3,
            base_filters=self.config.base_filters,
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate
        predictions, metrics = self.evaluator.evaluate_split(model, test_fixed, self.device)

        return metrics

    def _evaluate_round(self, checkpoint_path: Path, training_result: TrainingResult | None = None) -> RoundMetrics:
        """
        Evaluate a round.

        Args:
            checkpoint_path: Path to the round checkpoint.
            training_result: Optional training result for time/n_augmented info.

        Returns:
            RoundMetrics instance.
        """
        print("\n[Step 5] Evaluating round...")

        test_fixed = self.data_protocol.get_test_fixed()

        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        from models_3d.pinn.model import PINN3C
        model = PINN3C(
            signal_size=64,
            in_channels=3,
            base_filters=self.config.base_filters,
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate
        predictions, metrics = self.evaluator.evaluate_split(model, test_fixed, self.device)

        # Print comparison with baseline
        if self.baseline_metrics is not None:
            self._print_comparison(self.baseline_metrics, metrics)

        # Get training info if available
        training_time = training_result.training_time if training_result else 0
        n_augmented = training_result.n_augmented if training_result else 0

        return RoundMetrics(
            round_idx=len(self.round_results),
            checkpoint_path=checkpoint_path,
            metrics=metrics,
            training_time=training_time,
            n_augmented=n_augmented,
        )

    def _predict_batch(self, model_inputs: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch of signals.

        Args:
            model_inputs: Input model inputs (N, 3, 64, 64) - 3-channel for PINN.

        Returns:
            Predicted spectra (N, 64, 64).
        """
        from models_3d.pinn.model import PINN3C

        # Load model
        checkpoint = torch.load(
            self.state.current_checkpoint,
            map_location=self.device,
            weights_only=False,
        )

        model = PINN3C(
            signal_size=64,
            in_channels=3,
            base_filters=self.config.base_filters,
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Run inference
        predictions = []
        batch_size = 32

        with torch.no_grad():
            for start in range(0, len(model_inputs), batch_size):
                end = min(start + batch_size, len(model_inputs))
                batch = torch.from_numpy(model_inputs[start:end]).to(self.device)

                preds = model(batch).cpu().numpy()
                predictions.append(preds)

        predictions = np.concatenate(predictions, axis=0)[:, 0]

        return predictions

    def _check_stopping(self) -> bool:
        """
        Check if the stopping condition is met.

        Returns:
            True if should stop.
        """
        if len(self.round_results) < self.config.min_stopping_rounds:
            return False

        # Check consecutive rounds with improvement < threshold
        recent_results = self.round_results[-self.config.min_stopping_rounds:]

        improvements = []
        prev_metrics = self.baseline_metrics

        for result in recent_results:
            # Compute improvement in DEI error
            if prev_metrics is not None:
                improvement = (
                    (prev_metrics.dei_error_mean - result.metrics.dei_error_mean) /
                    prev_metrics.dei_error_mean
                )
                improvements.append(improvement)
                prev_metrics = result.metrics

        # Stop if all recent improvements are below threshold
        if len(improvements) >= self.config.min_stopping_rounds:
            should_stop = all(imp < self.config.improvement_threshold for imp in improvements)
            if should_stop:
                print(f"\n[STOPPING] Improvement below {self.config.improvement_threshold * 100:.1f}% for "
                      f"{len(improvements)} consecutive rounds")
            return should_stop

        return False

    def _print_comparison(
        self,
        baseline: ExtendedMetrics,
        current: ExtendedMetrics,
    ) -> None:
        """
        Print comparison between baseline and current metrics.

        Args:
            baseline: Baseline metrics.
            current: Current metrics.
        """
        print("\n" + "-" * 50)
        print("METRIC COMPARISON (vs Baseline)")
        print("-" * 50)

        metrics = [
            ('MSE Mean', baseline.mse_mean, current.mse_mean, True),
            ('MAE Mean', baseline.mae_mean, current.mae_mean, True),
            ('DEI Error Mean', baseline.dei_error_mean, current.dei_error_mean, True),
            ('MSE P95', baseline.mse_p95, current.mse_p95, True),
            ('MAE P95', baseline.mae_p95, current.mae_p95, True),
            ('DEI Error P95', baseline.dei_error_p95, current.dei_error_p95, True),
        ]

        for name, base_val, curr_val, lower_is_better in metrics:
            if lower_is_better:
                diff = base_val - curr_val
                pct = diff / base_val * 100 if base_val != 0 else 0
                direction = "improved" if diff > 0 else "worsened"
            else:
                diff = curr_val - base_val
                pct = diff / base_val * 100 if base_val != 0 else 0
                direction = "improved" if diff > 0 else "worsened"

            print(f"  {name:20s}: {curr_val:.6f} ({direction} {abs(pct):.2f}%)")

        print("-" * 50)

    def _save_results(self) -> None:
        """Save all results to disk."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save state
        with open(output_dir / "state.json", 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)

        # Save baseline metrics
        if self.baseline_metrics is not None:
            baseline_df = pd.DataFrame([self.baseline_metrics.to_dict()])
            baseline_df.to_csv(output_dir / "baseline_metrics.csv", index=False)

        # Save round results
        if self.round_results:
            rows = []
            for rm in self.round_results:
                row = rm.metrics.to_dict()
                row['round'] = rm.round_idx
                row['checkpoint'] = str(rm.checkpoint_path)
                rows.append(row)

            rounds_df = pd.DataFrame(rows)
            rounds_df.to_csv(output_dir / "round_metrics.csv", index=False)

        # Create ablation table
        if self.baseline_metrics is not None and self.round_results:
            from .evaluator import create_ablation_table
            ablation_df = create_ablation_table(self.baseline_metrics, self.round_results)
            ablation_df.to_csv(output_dir / "ablation_table.csv", index=False)
            print(f"\nAblation table saved to {output_dir / 'ablation_table.csv'}")

        print(f"\nResults saved to {output_dir}")

    def print_summary(self) -> None:
        """Print a final summary of the AL run."""
        print("\n" + "=" * 70)
        print("ACTIVE LEARNING SUMMARY")
        print("=" * 70)

        if self.baseline_metrics is not None:
            print("\nBaseline Metrics:")
            print(f"  MSE:  {self.baseline_metrics.mse_mean:.6f}")
            print(f"  MAE:  {self.baseline_metrics.mae_mean:.6f}")
            print(f"  DEI:  {self.baseline_metrics.dei_error_mean:.6f}")

        if self.round_results:
            print(f"\nRounds completed: {len(self.round_results)}")

            for i, rm in enumerate(self.round_results):
                print(f"\nRound {i + 1}:")
                print(f"  MSE:  {rm.metrics.mse_mean:.6f} "
                      f"({(self.baseline_metrics.mse_mean - rm.metrics.mse_mean) / self.baseline_metrics.mse_mean * 100:+.2f}%)")
                print(f"  MAE:  {rm.metrics.mae_mean:.6f} "
                      f"({(self.baseline_metrics.mae_mean - rm.metrics.mae_mean) / self.baseline_metrics.mae_mean * 100:+.2f}%)")
                print(f"  DEI:  {rm.metrics.dei_error_mean:.6f} "
                      f"({(self.baseline_metrics.dei_error_mean - rm.metrics.dei_error_mean) / self.baseline_metrics.dei_error_mean * 100:+.2f}%)")

        print("\n" + "=" * 70)


def run_active_learning(
    config: ALConfig | None = None,
    forward_model: ForwardModel2D | None = None,
) -> ActiveLearningAgent:
    """
    Convenience function to run active learning.

    Args:
        config: ALConfig instance.
        forward_model: ForwardModel2D instance.

    Returns:
        ActiveLearningAgent with completed run.
    """
    agent = ActiveLearningAgent(
        config=config,
        forward_model=forward_model,
    )
    agent.setup()
    agent.run()
    agent.print_summary()
    return agent

"""
Unified Benchmark Evaluator for 3-Compartment DEXSY Models.

This module provides a standardized framework for evaluating multiple model
architectures on 3C test data with consistent metrics.

Usage:
    from benchmarks_3d.evaluator import BenchmarkEvaluator3C

    evaluator = BenchmarkEvaluator3C(n_test=500, seed=42)
    results = evaluator.evaluate_model(model_pipeline, test_dataset)
    table = evaluator.generate_comparison_table()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dexsy_core.forward_model import ForwardModel2D, compute_dei
from dexsy_core.preprocessing import validate_signal_grid


@dataclass
class MetricsResult3C:
    """Container for computed metrics."""
    mse: float
    mae: float
    rmse: float
    dei_error: float
    dei_predicted: float
    dei_ground_truth: float
    inference_time: float
    prediction_mass: float
    prediction_peak: float


@dataclass
class ModelBenchmarkResult3C:
    """Benchmark results for one model."""
    model_name: str
    model_info: dict[str, Any]
    metrics: list[MetricsResult3C]
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    predictions: list[np.ndarray] = field(default_factory=list)
    execution_time: float = 0.0


class TestDataset3C:
    """Standardized test dataset for 3C benchmarks."""

    def __init__(
        self,
        n_test: int = 500,
        seed: int = 42,
        noise_sigma_range: tuple[float, float] = (0.005, 0.015),
        forward_model: ForwardModel2D | None = None,
    ):
        """
        Generate a standardized test dataset for 3C.

        Args:
            n_test: Number of test samples
            seed: Random seed for reproducibility
            noise_sigma_range: Range of noise levels
            forward_model: ForwardModel2D instance
        """
        self.n_test = n_test
        self.seed = seed
        self.noise_sigma_range = noise_sigma_range
        self.forward_model = forward_model or ForwardModel2D(n_d=64, n_b=64)

        # Generate dataset
        self.signals, self.ground_truths, self.params = self._generate()

    def _generate(self) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Generate the test dataset."""
        np.random.seed(self.seed)

        signals = []
        ground_truths = []
        params = []

        for i in range(self.n_test):
            # Generate 3C sample
            generated = self.forward_model.generate_sample(
                n_compartments=3,
                noise_sigma_range=self.noise_sigma_range,
                return_reference_signal=True,
            )

            # Handle both return formats
            if len(generated) == 4:
                F, S, params_i, S_clean = generated
            else:
                F, S, params_i = generated

            signals.append(S[0, 0] if S.ndim == 3 else S)
            ground_truths.append(F[0, 0] if F.ndim == 3 else F)
            params.append(params_i)

        signals = np.stack(signals, axis=0).astype(np.float32)
        ground_truths = np.stack(ground_truths, axis=0).astype(np.float32)

        return signals, ground_truths, params

    def __len__(self) -> int:
        return self.n_test

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, dict]:
        return self.signals[idx], self.ground_truths[idx], self.params[idx]


class BenchmarkEvaluator3C:
    """
    Unified benchmark evaluator for 3C DEXSY models.

    Evaluates model performance on a standardized test dataset with
    consistent metrics and reporting.
    """

    def __init__(
        self,
        n_test: int = 500,
        seed: int = 42,
        noise_sigma_range: tuple[float, float] = (0.005, 0.015),
        output_dir: Path | str | None = None,
    ):
        """
        Initialize the benchmark evaluator.

        Args:
            n_test: Number of test samples
            seed: Random seed for reproducibility
            noise_sigma_range: Noise range for test samples
            output_dir: Directory for saving results
        """
        self.n_test = n_test
        self.seed = seed
        self.noise_sigma_range = noise_sigma_range
        self.output_dir = Path(output_dir) if output_dir else None

        # Generate test dataset (3C)
        self.test_dataset = TestDataset3C(
            n_test=n_test,
            seed=seed,
            noise_sigma_range=noise_sigma_range,
        )

        # Store results
        self.model_results: dict[str, ModelBenchmarkResult3C] = {}

    def evaluate_model(
        self,
        pipeline,
        model_name: str = None,
        model_info: dict[str, Any] = None,
    ) -> ModelBenchmarkResult3C:
        """
        Evaluate a model on the test dataset.

        Args:
            pipeline: Model inference pipeline with predict_batch method
            model_name: Name of the model
            model_info: Additional model information

        Returns:
            ModelBenchmarkResult3C with metrics
        """
        if model_name is None:
            model_name = pipeline.get_model_name()

        if model_info is None:
            model_info = pipeline.get_model_info()

        print(f"\nEvaluating {model_name} on {self.n_test} 3C samples...")

        metrics_list = []
        predictions = []
        execution_times = []

        start_time = time.time()

        # Process in batches for efficiency
        batch_size = 32
        signals = self.test_dataset.signals
        ground_truths = self.test_dataset.ground_truths

        for batch_start in range(0, len(signals), batch_size):
            batch_end = min(batch_start + batch_size, len(signals))
            batch_signals = signals[batch_start:batch_end]
            batch_truths = ground_truths[batch_start:batch_end]

            batch_start_time = time.time()

            # Run batch prediction
            try:
                batch_results = pipeline.predict_batch(
                    batch_signals,
                    true_spectra=batch_truths,
                    include_figures=False,
                    batch_size=batch_size,
                )

                batch_time = time.time() - batch_start_time

                for idx, result in enumerate(batch_results):
                    metrics = MetricsResult3C(
                        mse=result.summary_metrics.get('mse', 0.0),
                        mae=result.summary_metrics.get('mae', 0.0),
                        rmse=np.sqrt(result.summary_metrics.get('mse', 0.0)),
                        dei_error=result.summary_metrics.get('dei_error', 0.0),
                        dei_predicted=result.summary_metrics.get('dei', 0.0),
                        dei_ground_truth=result.summary_metrics.get('ground_truth_dei', 0.0),
                        inference_time=batch_time / len(batch_signals),
                        prediction_mass=result.summary_metrics.get('prediction_mass', 0.0),
                        prediction_peak=result.summary_metrics.get('prediction_peak', 0.0),
                    )
                    metrics_list.append(metrics)
                    predictions.append(result.reconstructed_spectrum)

            except FileNotFoundError as e:
                print(f"  Error: {e}")
                print("  Please train the model first or provide a valid checkpoint path.")
                raise

        total_time = time.time() - start_time

        # Aggregate metrics
        aggregate = self._aggregate_metrics(metrics_list)

        result = ModelBenchmarkResult3C(
            model_name=model_name,
            model_info=model_info,
            metrics=metrics_list,
            aggregate_metrics=aggregate,
            predictions=predictions,
            execution_time=total_time,
        )

        self.model_results[model_name] = result

        # Print summary
        print(f"\n  Results for {model_name}:")
        print(f"    MSE:      {aggregate['mse_mean']:.6f} +/- {aggregate['mse_std']:.6f}")
        print(f"    MAE:      {aggregate['mae_mean']:.6f} +/- {aggregate['mae_std']:.6f}")
        print(f"    DEI Err:  {aggregate['dei_error_mean']:.6f} +/- {aggregate['dei_error_std']:.6f}")
        print(f"    Time:     {total_time:.2f}s ({self.n_test/total_time:.1f} samples/s)")

        return result

    def _aggregate_metrics(self, metrics_list: list[MetricsResult3C]) -> dict[str, float]:
        """Compute aggregate statistics over all metrics."""
        mse = [m.mse for m in metrics_list]
        mae = [m.mae for m in metrics_list]
        dei_error = [m.dei_error for m in metrics_list]
        dei_predicted = [m.dei_predicted for m in metrics_list]
        dei_ground_truth = [m.dei_ground_truth for m in metrics_list]
        inference_time = [m.inference_time for m in metrics_list]

        return {
            'mse_mean': np.mean(mse),
            'mse_std': np.std(mse),
            'mse_median': np.median(mse),
            'mse_p95': np.percentile(mse, 95),
            'mae_mean': np.mean(mae),
            'mae_std': np.std(mae),
            'mae_median': np.median(mae),
            'mae_p95': np.percentile(mae, 95),
            'dei_error_mean': np.mean(dei_error),
            'dei_error_std': np.std(dei_error),
            'dei_error_median': np.median(dei_error),
            'dei_error_p95': np.percentile(dei_error, 95),
            'dei_predicted_mean': np.mean(dei_predicted),
            'dei_predicted_std': np.std(dei_predicted),
            'dei_ground_truth_mean': np.mean(dei_ground_truth),
            'dei_ground_truth_std': np.std(dei_ground_truth),
            'inference_time_mean': np.mean(inference_time),
            'inference_time_total': np.sum(inference_time),
            'throughput': self.n_test / sum(inference_time),
        }

    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate a comparison table for all evaluated models."""
        rows = []
        for model_name, result in self.model_results.items():
            agg = result.aggregate_metrics
            rows.append({
                'Model': model_name,
                'MSE': f"{agg['mse_mean']:.6f} +/- {agg['mse_std']:.6f}",
                'MAE': f"{agg['mae_mean']:.6f} +/- {agg['mae_std']:.6f}",
                'DEI Error': f"{agg['dei_error_mean']:.6f} +/- {agg['dei_error_std']:.6f}",
                'Throughput (samples/s)': f"{agg['throughput']:.1f}",
                'Val Loss': result.model_info.get('val_loss', 'N/A'),
            })

        df = pd.DataFrame(rows)
        return df

    def plot_sample_comparison(
        self,
        model_pipelines: dict[str, Any],
        n_samples: int = 5,
        output_dir: Path | str | None = None,
    ) -> None:
        """
        Plot comparison of multiple models on sample predictions.

        Args:
            model_pipelines: Dict of {model_name: pipeline}
            n_samples: Number of samples to visualize
            output_dir: Directory to save plots
        """
        if output_dir is None:
            output_dir = Path("outputs/benchmark_3c/samples")
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        signals = self.test_dataset.signals[:n_samples]
        ground_truths = self.test_dataset.ground_truths[:n_samples]

        for sample_idx in range(n_samples):
            signal = signals[sample_idx]
            gt = ground_truths[sample_idx]

            n_models = len(model_pipelines)
            fig, axes = plt.subplots(2, n_models + 1, figsize=(4 * (n_models + 1), 8))

            # Ground truth
            vmax = float(gt.max())
            axes[0, 0].imshow(gt, cmap="magma", origin="lower", vmin=0, vmax=vmax)
            axes[0, 0].set_title(f"Ground Truth\nDEI={compute_dei(gt):.3f}")
            axes[0, 0].set_xlabel("D2 index")
            axes[0, 0].set_ylabel("D1 index")

            axes[1, 0].imshow(signal, cmap="viridis", origin="lower")
            axes[1, 0].set_title("Input Signal")
            axes[1, 0].set_xlabel("b2 index")
            axes[1, 0].set_ylabel("b1 index")

            for col, (model_name, pipeline) in enumerate(model_pipelines.items(), start=1):
                try:
                    result = pipeline.predict(
                        signal,
                        true_spectrum=gt,
                        include_figure=False,
                    )
                    pred = result.reconstructed_spectrum

                    axes[0, col].imshow(pred, cmap="magma", origin="lower", vmin=0, vmax=vmax)
                    axes[0, col].set_title(f"{model_name}\nDEI={compute_dei(pred):.3f}")
                    axes[0, col].set_xlabel("D2 index")

                    error = np.abs(pred - gt)
                    axes[1, col].imshow(error, cmap="hot", origin="lower")
                    axes[1, col].set_title(f"Abs Error\nMSE={result.summary_metrics.get('mse', 0):.6f}")
                    axes[1, col].set_xlabel("b2 index")

                except Exception as e:
                    axes[0, col].text(0.5, 0.5, f"Error:\n{str(e)}",
                                     ha='center', va='center', transform=axes[0, col].transAxes)
                    axes[0, col].set_title(model_name)
                    axes[1, col].text(0.5, 0.5, "Prediction failed",
                                     ha='center', va='center', transform=axes[1, col].transAxes)

            plt.tight_layout()
            plt.savefig(output_dir / f"sample_{sample_idx:03d}.png", dpi=150, bbox_inches='tight')
            plt.close()

        print(f"Saved sample comparisons to {output_dir}")

    def save_results(self, output_dir: Path | str | None = None) -> None:
        """Save all results to disk."""
        if output_dir is None:
            output_dir = Path("outputs/benchmark_3c")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save comparison table
        table = self.generate_comparison_table()
        table.to_csv(output_dir / "comparison_table.csv", index=False)

        # Save detailed metrics for each model
        for model_name, result in self.model_results.items():
            model_dir = output_dir / model_name.replace(" ", "_")
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save metrics as CSV
            metrics_data = []
            for m in result.metrics:
                metrics_data.append({
                    'mse': m.mse,
                    'mae': m.mae,
                    'rmse': m.rmse,
                    'dei_error': m.dei_error,
                    'dei_predicted': m.dei_predicted,
                    'dei_ground_truth': m.dei_ground_truth,
                    'inference_time': m.inference_time,
                    'prediction_mass': m.prediction_mass,
                    'prediction_peak': m.prediction_peak,
                })

            df = pd.DataFrame(metrics_data)
            df.to_csv(model_dir / "metrics.csv", index=False)

            # Save aggregate metrics
            agg_df = pd.DataFrame([result.aggregate_metrics])
            agg_df.to_csv(model_dir / "aggregate_metrics.csv", index=False)

            # Save predictions
            np.savez_compressed(
                model_dir / "predictions.npz",
                predictions=np.stack(result.predictions),
                ground_truths=self.test_dataset.ground_truths,
            )

        print(f"Saved results to {output_dir}")


def run_benchmark(
    model_pipelines: dict[str, Any],
    n_test: int = 500,
    seed: int = 42,
    output_dir: str = None,
) -> BenchmarkEvaluator3C:
    """
    Run a complete benchmark on multiple models.

    Args:
        model_pipelines: Dict of {model_name: pipeline}
        n_test: Number of test samples
        seed: Random seed
        output_dir: Output directory for results

    Returns:
        BenchmarkEvaluator3C with results
    """
    evaluator = BenchmarkEvaluator3C(
        n_test=n_test,
        seed=seed,
        output_dir=output_dir,
    )

    for model_name, pipeline in model_pipelines.items():
        try:
            evaluator.evaluate_model(pipeline, model_name)
        except FileNotFoundError as e:
            print(f"Skipping {model_name}: {e}")
            continue

    # Save results
    if output_dir:
        evaluator.save_results(output_dir)

    return evaluator


if __name__ == "__main__":
    print("Benchmark Evaluator for 3-Compartment DEXSY Models")
    print("=" * 60)
    print("Note: This module provides evaluation utilities.")
    print("Please train models first using models_3d.attention_unet.train")

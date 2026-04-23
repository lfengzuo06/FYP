"""
Unified Benchmark Evaluator for 2D DEXSY Models.

This module provides a standardized framework for evaluating multiple model
architectures on the same test dataset with consistent metrics.

Reference: Paper Section 2.4 - Model Benchmarks

Usage:
    from benchmarks_2d.evaluator import BenchmarkEvaluator

    evaluator = BenchmarkEvaluator(n_test=500, seed=42)
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

from dexsy_core.forward_model import ForwardModel2D, compute_dei
from dexsy_core.preprocessing import validate_signal_grid


@dataclass
class MetricsResult:
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
    sssim: float | None = None


@dataclass
class ModelBenchmarkResult:
    """Benchmark results for one model."""
    model_name: str
    model_info: dict[str, Any]
    metrics: list[MetricsResult]
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    predictions: list[np.ndarray] = field(default_factory=list)
    execution_time: float = 0.0


def compute_ssim(img1: np.ndarray, img2: np.ndarray, window_size: int = 8) -> float:
    """Compute Structural Similarity Index (SSIM) between two images."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim)


class TestDataset:
    """Standardized test dataset for benchmarks."""
    
    def __init__(
        self,
        n_test: int = 500,
        seed: int = 42,
        noise_sigma_range: tuple[float, float] = (0.005, 0.015),
        n_compartments: int = 2,
        forward_model: ForwardModel2D | None = None,
    ):
        """
        Generate a standardized test dataset.
        
        Args:
            n_test: Number of test samples
            seed: Random seed for reproducibility
            noise_sigma_range: Range of noise levels
            n_compartments: Number of compartments (2 or 3)
            forward_model: ForwardModel2D instance
        """
        self.n_test = n_test
        self.seed = seed
        self.noise_sigma_range = noise_sigma_range
        self.n_compartments = n_compartments
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
            # Generate sample with specified noise
            # generate_sample returns (F, S, params) or (F, S, params, S_clean)
            generated = self.forward_model.generate_sample(
                n_compartments=self.n_compartments,
                noise_sigma_range=self.noise_sigma_range,
                return_reference_signal=True,
            )
            
            # Handle both return formats
            if len(generated) == 4:
                F, S, params_i, S_clean = generated
            else:
                F, S, params_i = generated
            
            signals.append(S[0, 0] if S.ndim == 3 else S)  # Shape: (64, 64)
            ground_truths.append(F[0, 0] if F.ndim == 3 else F)  # Shape: (64, 64)
            params.append(params_i)
        
        signals = np.stack(signals, axis=0).astype(np.float32)
        ground_truths = np.stack(ground_truths, axis=0).astype(np.float32)
        
        return signals, ground_truths, params
    
    def __len__(self) -> int:
        return self.n_test
    
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, dict]:
        return self.signals[idx], self.ground_truths[idx], self.params[idx]


class BenchmarkEvaluator:
    """
    Unified benchmark evaluator for 2D DEXSY models.
    
    Evaluates model performance on a standardized test dataset with
    consistent metrics and reporting.
    """
    
    def __init__(
        self,
        n_test: int = 500,
        seed: int = 42,
        noise_sigma_range: tuple[float, float] = (0.005, 0.015),
        n_compartments: int = 2,
        output_dir: Path | str | None = None,
    ):
        """
        Initialize the benchmark evaluator.
        
        Args:
            n_test: Number of test samples (default: 500 for statistical significance)
            seed: Random seed for reproducibility
            noise_sigma_range: Noise range for test samples
            n_compartments: Number of compartments in test data
            output_dir: Directory for saving results
        """
        self.n_test = n_test
        self.seed = seed
        self.noise_sigma_range = noise_sigma_range
        self.n_compartments = n_compartments
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Generate test dataset
        self.test_dataset = TestDataset(
            n_test=n_test,
            seed=seed,
            noise_sigma_range=noise_sigma_range,
            n_compartments=n_compartments,
        )
        
        # Store results
        self.model_results: dict[str, ModelBenchmarkResult] = {}
    
    def evaluate_model(
        self,
        model_pipeline: Any,
        model_name: str | None = None,
    ) -> ModelBenchmarkResult:
        """
        Evaluate a single model on the test dataset.
        
        Args:
            model_pipeline: Model inference pipeline with predict() method
            model_name: Optional model name override
            
        Returns:
            ModelBenchmarkResult with all metrics
        """
        if model_name is None:
            model_name = getattr(model_pipeline, 'get_model_name', lambda: 'unknown')()
        
        model_info = getattr(model_pipeline, 'get_model_info', lambda: {})()
        if callable(model_info):
            model_info = model_info()
        
        metrics_list: list[MetricsResult] = []
        predictions: list[np.ndarray] = []
        
        start_time = time.time()
        
        for i in range(len(self.test_dataset)):
            signal, ground_truth, _ = self.test_dataset[i]
            
            # Run inference
            infer_start = time.time()
            result = model_pipeline.predict(signal)
            infer_time = time.time() - infer_start
            
            # Get prediction
            pred = result.reconstructed_spectrum
            predictions.append(pred)
            
            # Normalize ground truth
            gt_normalized = ground_truth / (ground_truth.sum() + 1e-10)
            
            # Compute metrics
            mse = float(np.mean((pred - gt_normalized) ** 2))
            mae = float(np.mean(np.abs(pred - gt_normalized)))
            rmse = float(np.sqrt(mse))
            
            dei_pred = compute_dei(pred)
            dei_gt = compute_dei(gt_normalized)
            dei_error = abs(dei_pred - dei_gt)
            
            ssim = compute_ssim(pred, gt_normalized)
            
            metrics = MetricsResult(
                mse=mse,
                mae=mae,
                rmse=rmse,
                dei_error=dei_error,
                dei_predicted=dei_pred,
                dei_ground_truth=dei_gt,
                inference_time=infer_time,
                prediction_mass=float(pred.sum()),
                prediction_peak=float(pred.max()),
                sssim=ssim,
            )
            metrics_list.append(metrics)
        
        total_time = time.time() - start_time
        
        # Compute aggregate metrics
        agg = self._aggregate_metrics(metrics_list)
        
        result = ModelBenchmarkResult(
            model_name=model_name,
            model_info=model_info,
            metrics=metrics_list,
            aggregate_metrics=agg,
            predictions=predictions,
            execution_time=total_time,
        )
        
        self.model_results[model_name] = result
        return result
    
    def _aggregate_metrics(self, metrics_list: list[MetricsResult]) -> dict[str, float]:
        """Compute aggregate statistics over all samples."""
        n = len(metrics_list)
        
        return {
            # Central tendencies
            "mean_mse": np.mean([m.mse for m in metrics_list]),
            "mean_mae": np.mean([m.mae for m in metrics_list]),
            "mean_rmse": np.mean([m.rmse for m in metrics_list]),
            "mean_dei_error": np.mean([m.dei_error for m in metrics_list]),
            "mean_dei_predicted": np.mean([m.dei_predicted for m in metrics_list]),
            "mean_sssim": np.mean([m.sssim for m in metrics_list]),
            
            # Spread
            "std_mse": np.std([m.mse for m in metrics_list]),
            "std_mae": np.std([m.mae for m in metrics_list]),
            "std_rmse": np.std([m.rmse for m in metrics_list]),
            "std_dei_error": np.std([m.dei_error for m in metrics_list]),
            
            # Timing
            "mean_inference_time": np.mean([m.inference_time for m in metrics_list]),
            "total_inference_time": np.sum([m.inference_time for m in metrics_list]),
            "samples_per_second": n / np.sum([m.inference_time for m in metrics_list]),
            
            # Other
            "median_mse": np.median([m.mse for m in metrics_list]),
            "p95_mse": np.percentile([m.mse for m in metrics_list], 95),
        }
    
    def compute_all_metrics(
        self,
        predictions: list[np.ndarray],
        ground_truths: np.ndarray,
    ) -> dict[str, Any]:
        """
        Compute comprehensive metrics for a set of predictions.
        
        Args:
            predictions: List of predicted spectra
            ground_truths: Array of ground truth spectra
            
        Returns:
            Dictionary of computed metrics
        """
        mse_values = []
        mae_values = []
        dei_errors = []
        
        for pred, gt in zip(predictions, ground_truths):
            gt_norm = gt / (gt.sum() + 1e-10)
            
            mse_values.append(np.mean((pred - gt_norm) ** 2))
            mae_values.append(np.mean(np.abs(pred - gt_norm)))
            
            dei_pred = compute_dei(pred)
            dei_gt = compute_dei(gt_norm)
            dei_errors.append(abs(dei_pred - dei_gt))
        
        return {
            "mean_mse": np.mean(mse_values),
            "std_mse": np.std(mse_values),
            "median_mse": np.median(mse_values),
            "p95_mse": np.percentile(mse_values, 95),
            "mean_mae": np.mean(mae_values),
            "std_mae": np.std(mae_values),
            "mean_dei_error": np.mean(dei_errors),
            "std_dei_error": np.std(dei_errors),
        }
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Generate a comparison table across all evaluated models.
        
        Returns:
            DataFrame with model comparison
        """
        rows = []
        for model_name, result in self.model_results.items():
            agg = result.aggregate_metrics
            row = {
                "Model": model_name,
                "Mean MSE": f"{agg['mean_mse']:.6f}",
                "Median MSE": f"{agg['median_mse']:.6f}",
                "P95 MSE": f"{agg['p95_mse']:.6f}",
                "Mean MAE": f"{agg['mean_mae']:.6f}",
                "Mean DEI Error": f"{agg['mean_dei_error']:.4f}",
                "Mean SSIM": f"{agg['mean_sssim']:.4f}",
                "Mean Inference (ms)": f"{agg['mean_inference_time']*1000:.2f}",
                "Samples/sec": f"{agg['samples_per_second']:.1f}",
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values("Mean MSE")
        return df
    
    def plot_comparison_figures(self) -> list[plt.Figure]:
        """
        Generate comparison figures across all models.
        
        Returns:
            List of matplotlib Figures
        """
        figures = []
        
        # Figure 1: MSE comparison boxplot
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        mse_data = []
        labels = []
        for model_name, result in self.model_results.items():
            mse_values = [m.mse for m in result.metrics]
            mse_data.append(mse_values)
            labels.append(model_name)
        
        bp = ax1.boxplot(mse_data, labels=labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax1.set_ylabel("MSE")
        ax1.set_title("MSE Distribution by Model")
        ax1.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        figures.append(fig1)
        
        # Figure 2: DEI Error comparison
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        dei_data = []
        for model_name, result in self.model_results.items():
            dei_values = [m.dei_error for m in result.metrics]
            dei_data.append(dei_values)
        
        bp2 = ax2.boxplot(dei_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_ylabel("DEI Error (absolute)")
        ax2.set_title("DEI Error Distribution by Model")
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        figures.append(fig2)
        
        # Figure 3: Inference time comparison
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        times = [result.aggregate_metrics["mean_inference_time"]*1000 
                 for result in self.model_results.values()]
        bars = ax3.bar(labels, times, color=colors[:len(labels)])
        ax3.set_ylabel("Mean Inference Time (ms)")
        ax3.set_title("Inference Time by Model")
        ax3.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        figures.append(fig3)
        
        return figures
    
    def save_results(self, output_dir: Path | str | None = None) -> Path:
        """Save all results to disk."""
        if output_dir is None:
            output_dir = self.output_dir or Path("benchmark_results")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison table
        table = self.generate_comparison_table()
        table.to_csv(output_dir / "comparison_table.csv", index=False)
        
        # Save detailed results
        import json
        for model_name, result in self.model_results.items():
            model_dir = output_dir / model_name.replace(" ", "_")
            model_dir.mkdir(exist_ok=True)
            
            # Metrics summary
            summary = {
                "model_name": result.model_name,
                "model_info": result.model_info,
                "aggregate_metrics": result.aggregate_metrics,
                "execution_time": result.execution_time,
                "n_samples": len(result.metrics),
            }
            with open(model_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
        
        # Save figures
        figures = self.plot_comparison_figures()
        figure_names = ["mse_comparison", "dei_error_comparison", "inference_time"]
        for fig, name in zip(figures, figure_names):
            fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        return output_dir


def evaluate_all_models(
    n_test: int = 500,
    seed: int = 42,
    output_dir: Path | str | None = None,
    skip_models: list[str] | None = None,
) -> dict[str, ModelBenchmarkResult]:
    """
    Evaluate all available models with their default checkpoints.
    
    Args:
        n_test: Number of test samples
        seed: Random seed
        output_dir: Output directory for results
        skip_models: Model names to skip
        
    Returns:
        Dictionary of model results
    """
    skip_models = skip_models or []
    
    # Import model pipelines
    from models_2d.attention_unet import InferencePipeline as AttentionPipeline
    from models_2d.plain_unet import InferencePipeline as PlainPipeline
    from models_2d.deep_unfolding import InferencePipeline as DeepPipeline
    from models_2d.neural_operators.inference import InferencePipeline as NeuralPipeline
    from benchmarks_2d.ilt_baseline import ILTInferencePipeline
    
    # Create evaluator
    evaluator = BenchmarkEvaluator(
        n_test=n_test,
        seed=seed,
        output_dir=output_dir,
    )
    
    # Define models to evaluate
    models_to_evaluate = [
        ("2d_ilt", lambda: ILTInferencePipeline()),
    ]
    
    # Add neural operators (FNO and DeepONet)
    try:
        fno_pipeline = NeuralPipeline(model_type="fno")
        models_to_evaluate.append(("fno", lambda: fno_pipeline))
    except FileNotFoundError:
        print("Warning: FNO checkpoint not found, skipping...")
    
    try:
        deeponet_pipeline = NeuralPipeline(model_type="deeponet")
        models_to_evaluate.append(("deeponet", lambda: deeponet_pipeline))
    except FileNotFoundError:
        print("Warning: DeepONet checkpoint not found, skipping...")
    
    # Add CNN-based models
    try:
        attention_pipeline = AttentionPipeline()
        models_to_evaluate.append(("attention_unet", lambda: attention_pipeline))
    except FileNotFoundError:
        print("Warning: Attention U-Net checkpoint not found, skipping...")
    
    try:
        plain_pipeline = PlainPipeline()
        models_to_evaluate.append(("plain_unet", lambda: plain_pipeline))
    except FileNotFoundError:
        print("Warning: Plain U-Net checkpoint not found, skipping...")
    
    try:
        deep_pipeline = DeepPipeline()
        models_to_evaluate.append(("deep_unfolding", lambda: deep_pipeline))
    except FileNotFoundError:
        print("Warning: Deep Unfolding checkpoint not found, skipping...")
    
    # PINN needs retraining - skip by default
    # from models_2d.pinn import PINNInferencePipeline
    # try:
    #     pinn_pipeline = PINNInferencePipeline()
    #     models_to_evaluate.append(("pinn", lambda: pinn_pipeline))
    # except FileNotFoundError:
    #     print("Warning: PINN checkpoint not found, skipping...")
    
    # Evaluate each model
    print(f"Evaluating {len(models_to_evaluate)} models on {n_test} test samples...")
    print("-" * 60)
    
    for model_name, pipeline_fn in models_to_evaluate:
        if model_name in skip_models:
            print(f"Skipping {model_name}...")
            continue
        
        print(f"Evaluating {model_name}...", end=" ", flush=True)
        try:
            pipeline = pipeline_fn()
            result = evaluator.evaluate_model(pipeline, model_name)
            
            agg = result.aggregate_metrics
            print(f"MSE={agg['mean_mse']:.6f}, DEI Error={agg['mean_dei_error']:.4f}, "
                  f"Time={agg['mean_inference_time']*1000:.2f}ms")
        except Exception as e:
            print(f"Error: {e}")
    
    print("-" * 60)
    
    # Generate summary
    print("\nComparison Table:")
    print(evaluator.generate_comparison_table().to_string(index=False))
    
    return evaluator.model_results


if __name__ == "__main__":
    from pathlib import Path
    
    print("Running DEXSY Benchmark Evaluation...")
    print()
    
    # Run with sufficient samples for statistical significance
    results = evaluate_all_models(
        n_test=500,
        seed=42,
        output_dir=Path("outputs/benchmark"),
    )
    
    print("\nBenchmark complete!")

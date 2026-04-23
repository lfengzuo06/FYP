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

Visualization:
    from benchmarks_2d.evaluator import plot_training_curves, plot_sample_comparison

    # Plot training curves for all models
    plot_training_curves(output_dir="outputs/benchmark/training_curves")

    # Plot sample comparison
    plot_sample_comparison(n_samples=10, output_dir="outputs/benchmark/samples")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import time
import re

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


def parse_training_log(log_path: Path) -> dict[str, list]:
    """Parse training log file and extract train/val losses."""
    if not log_path.exists():
        return {"epochs": [], "train_loss": [], "val_loss": [], "model_name": log_path.stem}

    epochs = []
    train_losses = []
    val_losses = []
    model_name = log_path.stem

    # Try to extract model name from log content
    content = log_path.read_text()

    # Parse different log formats
    # Format 1: Epoch   1/60 | Train: 2.3105 | Val: 0.9969
    pattern1 = re.compile(r'Epoch\s+(\d+)/\d+\s+\|\s+Train:\s+([\d.]+)\s+\|\s+Val:\s+([\d.]+)')
    # Format 2: Epoch   1/60 | Train Loss: 1.394540 | Val Loss: 0.743333
    pattern2 = re.compile(r'Epoch\s+(\d+)/\d+\s+\|\s+Train Loss:\s+([\d.]+)\s+\|\s+Val Loss:\s+([\d.]+)')

    for match in pattern1.finditer(content):
        epochs.append(int(match.group(1)))
        train_losses.append(float(match.group(2)))
        val_losses.append(float(match.group(3)))

    for match in pattern2.finditer(content):
        epochs.append(int(match.group(1)))
        train_losses.append(float(match.group(2)))
        val_losses.append(float(match.group(3)))

    return {
        "epochs": epochs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "model_name": model_name,
    }


def get_available_training_logs() -> dict[str, Path]:
    """Find all available training log files."""
    repo_root = Path(__file__).parent.parent
    log_files = {}

    # Known log file locations
    candidates = [
        ("attention_unet", repo_root / "improved_2d_dexsy" / "train_log.txt"),
        ("fno", repo_root / "train_fno.log"),
        ("deeponet", repo_root / "train_deeponet.log"),
        ("pinn", repo_root / "train_pinn.log"),
        # Also check training_output directories
        ("fno", repo_root / "training_output_2d" / "neural_operators_fno" / "training.log"),
        ("deeponet", repo_root / "training_output_2d" / "neural_operators_deeponet" / "training.log"),
        ("pinn", repo_root / "training_output_2d" / "pinn" / "training.log"),
    ]

    for item in candidates:
        model_name, path = item if isinstance(item, tuple) else (None, item)
        if path.exists():
            if model_name is None:
                model_name = path.stem.replace("_log", "").replace("train_", "")
            log_files[model_name] = path

    return log_files


def plot_training_curves(output_dir: Path | str | None = None, save: bool = True) -> list[plt.Figure]:
    """
    Plot training and validation loss curves for all available models.

    Args:
        output_dir: Directory to save figures
        save: Whether to save figures to disk

    Returns:
        List of matplotlib Figures
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)

    log_files = get_available_training_logs()
    figures = []

    if not log_files:
        print("Warning: No training logs found")
        return figures

    # Create individual plots for each model
    for model_name, log_path in log_files.items():
        data = parse_training_log(log_path)

        if not data["epochs"]:
            print(f"Warning: No data found in {log_path}")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data["epochs"], data["train_loss"], label="Train Loss", linewidth=2)
        ax.plot(data["epochs"], data["val_loss"], label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Curves - {model_name.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save and output_dir is not None:
            fig.savefig(output_dir / f"training_curve_{model_name}.png", dpi=150, bbox_inches='tight')

        figures.append(fig)

    # Create combined comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (model_name, log_path) in enumerate(log_files.items()):
        if idx >= len(axes):
            break

        data = parse_training_log(log_path)

        if not data["epochs"]:
            continue

        ax = axes[idx]
        ax.plot(data["epochs"], data["train_loss"], label="Train", linewidth=2)
        ax.plot(data["epochs"], data["val_loss"], label="Val", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(model_name.upper())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(log_files), len(axes)):
        axes[idx].axis('off')

    fig.suptitle("Training Curves Comparison", fontsize=14)
    fig.tight_layout()

    if save and output_dir is not None:
        fig.savefig(output_dir / "training_curves_comparison.png", dpi=150, bbox_inches='tight')

    figures.append(fig)

    if save:
        plt.close('all')

    print(f"Training curves saved to: {output_dir}")
    return figures


def plot_sample_comparison(
    n_samples: int = 10,
    seed: int = 42,
    output_dir: Path | str | None = None,
    save: bool = True,
) -> list[plt.Figure]:
    """
    Generate comparison figures for sample predictions across all models.

    Each figure shows:
    - Input signal
    - Ground truth spectrum
    - Reconstructed spectrum from each model with MSE, DEI, and inference time

    Args:
        n_samples: Number of samples to visualize
        seed: Random seed for reproducibility
        output_dir: Directory to save figures
        save: Whether to save figures to disk

    Returns:
        List of matplotlib Figures
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)

    # Generate test dataset
    np.random.seed(seed)
    fm = ForwardModel2D(n_d=64, n_b=64)

    # Generate samples
    signals = []
    ground_truths = []
    for _ in range(n_samples):
        generated = fm.generate_sample(n_compartments=2, return_reference_signal=True)
        # Handle both batched and unbatched return formats
        if len(generated) == 4:
            F, S, _, S_clean = generated
        else:
            F, S, _ = generated
        
        # Handle both batched (N, H, W) and unbatched (H, W) shapes
        if F.ndim == 3:
            F = F[0]
        if S.ndim == 3:
            S = S[0]
        
        signals.append(S)
        ground_truths.append(F)

    signals = np.array(signals)
    ground_truths = np.array(ground_truths)

    print(f"Generated {n_samples} samples: signals shape {signals.shape}, ground truths shape {ground_truths.shape}")

    # Collect all model pipelines
    from models_2d.attention_unet import InferencePipeline as AttentionPipeline
    from models_2d.plain_unet import InferencePipeline as PlainPipeline
    from models_2d.deep_unfolding import InferencePipeline as DeepPipeline
    from models_2d.neural_operators.inference import InferencePipeline as NeuralPipeline
    from benchmarks_2d.ilt_baseline import ILTInferencePipeline

    models = []
    model_names = []

    # Add ILT first
    try:
        models.append(("2d_ilt", ILTInferencePipeline()))
        model_names.append("2d_ilt")
    except Exception:
        pass

    # Add neural operators
    for model_type, display_name in [("fno", "fno"), ("deeponet", "deeponet")]:
        try:
            pipeline = NeuralPipeline(model_type=model_type)
            models.append((display_name, pipeline))
            model_names.append(display_name)
        except FileNotFoundError:
            pass

    # Add CNN-based models
    for model_cls, name in [(AttentionPipeline, "attention_unet"), (PlainPipeline, "plain_unet")]:
        try:
            pipeline = model_cls()
            models.append((name, pipeline))
            model_names.append(name)
        except FileNotFoundError:
            pass

    try:
        pipeline = DeepPipeline()
        models.append(("deep_unfolding", pipeline))
        model_names.append("deep_unfolding")
    except FileNotFoundError:
        pass

    figures = []

    # Create one figure per sample
    for i in range(n_samples):
        signal = signals[i]
        gt = ground_truths[i]

        # Number of columns: signal, gt, + one per model
        n_cols = 2 + len(models)
        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

        # Row 0: Signal and Ground Truth
        ax = axes[0, 0]
        im = ax.imshow(signal, cmap="viridis", origin="lower")
        ax.set_title("Input Signal")
        ax.set_xlabel("b2")
        ax.set_ylabel("b1")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[0, 1]
        im = ax.imshow(gt, cmap="magma", origin="lower")
        ax.set_title("Ground Truth")
        ax.set_xlabel("D2")
        ax.set_ylabel("D1")
        gt_dei = compute_dei(gt)
        ax.text(0.5, -0.15, f"DEI={gt_dei:.4f}", transform=ax.transAxes, ha='center')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row 1: Model predictions (each model gets its own column)
        for col_idx, (model_name, pipeline) in enumerate(models):
            ax = axes[1, col_idx]

            start_time = time.time()
            result = pipeline.predict(signal)
            infer_time = time.time() - start_time

            pred = result.reconstructed_spectrum

            # Normalize for comparison
            gt_norm = gt / (gt.sum() + 1e-10)

            im = ax.imshow(pred, cmap="magma", origin="lower")
            ax.set_title(model_name.upper())

            # Compute metrics
            mse = float(np.mean((pred - gt_norm) ** 2))
            pred_dei = compute_dei(pred)
            dei_error = abs(pred_dei - gt_dei)

            # Add metrics text
            metrics_text = f"MSE={mse:.2e}\nDEI={pred_dei:.4f}\nDEI Err={dei_error:.4f}\nTime={infer_time*1000:.1f}ms"
            ax.text(0.5, -0.25, metrics_text, transform=ax.transAxes, ha='center', fontsize=9)

            ax.set_xlabel("D2")
            ax.set_ylabel("D1")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide extra columns if any
        for col_idx in range(len(models), axes.shape[1]):
            axes[0, col_idx].axis('off')
            if len(axes) > 1:
                axes[1, col_idx].axis('off')

        fig.suptitle(f"Sample {i+1}/{n_samples}", fontsize=14)
        fig.tight_layout()

        if save and output_dir is not None:
            fig.savefig(output_dir / f"sample_{i:03d}.png", dpi=150, bbox_inches='tight')

        figures.append(fig)

    if save:
        plt.close('all')

    print(f"Sample comparison figures saved to: {output_dir}")
    return figures


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

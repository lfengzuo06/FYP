"""
2D Benchmark Module.

This module provides benchmark infrastructure for comparing different
2D DEXSY inversion methods.

Available benchmarks:
- 2D ILT baseline (NNLS + Tikhonov)
- Unified evaluator for all models

Usage:
    from benchmarks_2d import ILTInferencePipeline

    pipeline = ILTInferencePipeline()
    result = pipeline.predict(signal)

    # Or run full benchmark
    from benchmarks_2d.evaluator import evaluate_all_models
    results = evaluate_all_models(n_test=500)
"""

from .ilt_baseline import (
    ILTInferencePipeline,
    ILTPredictionResult,
    predict_ilt,
)

from .evaluator import (
    BenchmarkEvaluator,
    TestDataset,
    MetricsResult,
    ModelBenchmarkResult,
    evaluate_all_models,
)

__all__ = [
    # ILT baseline
    "ILTInferencePipeline",
    "ILTPredictionResult",
    "predict_ilt",
    # Evaluator
    "BenchmarkEvaluator",
    "TestDataset",
    "MetricsResult",
    "ModelBenchmarkResult",
    "evaluate_all_models",
]

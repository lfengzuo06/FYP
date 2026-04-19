"""
2D Benchmark Module.

This module provides benchmark infrastructure for comparing different
2D DEXSY inversion methods.

Available benchmarks:
- 2D ILT baseline (NNLS + Tikhonov)

Usage:
    from benchmarks_2d import ILTInferencePipeline

    pipeline = ILTInferencePipeline()
    result = pipeline.predict(signal)
"""

from .ilt_baseline import (
    ILTInferencePipeline,
    ILTPredictionResult,
    predict_ilt,
)

__all__ = [
    "ILTInferencePipeline",
    "ILTPredictionResult",
    "predict_ilt",
]

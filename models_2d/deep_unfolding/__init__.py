"""
Deep Unfolding (ISTA-Net style) model for 2D DEXSY Inversion.

Reference: Paper Section 2.4.3 - Deep Unfolding

Usage:
    from models_2d.deep_unfolding import DeepUnfolding2D, get_model, InferencePipeline

    model = get_model(n_layers=12, n_d=64)
    pipeline = InferencePipeline(checkpoint_path="path/to/model.pt")
    result = pipeline.predict(signal)
"""

from .model import (
    DeepUnfolding2D,
    get_model,
    soft_threshold,
    ISTAProximalStep,
)
from .inference import InferencePipeline, PredictionResult
from . import train
from . import inference

__all__ = [
    "DeepUnfolding2D",
    "get_model",
    "soft_threshold",
    "ISTAProximalStep",
    "InferencePipeline",
    "PredictionResult",
    "train",
    "inference",
]

"""Slim 2D DEXSY package for training and Colab inference."""

from .forward_model_2d import ForwardModel2D, compute_dei
from .inference_2d import (
    DEXSYInferencePipeline,
    PredictionResult,
    available_models,
    build_model_inputs,
    build_position_channel,
    load_trained_model,
    predict_from_signal,
    predict_distribution,
    resolve_checkpoint_path,
)
from .model_2d import (
    AttentionUNet2D,
    MultiscaleUNet2D,
    PhysicsInformedLoss,
    get_model,
)
from .preprocessing_2d import validate_signal_grid

__all__ = [
    "AttentionUNet2D",
    "DEXSYInferencePipeline",
    "MultiscaleUNet2D",
    "PhysicsInformedLoss",
    "ForwardModel2D",
    "PredictionResult",
    "available_models",
    "build_model_inputs",
    "build_position_channel",
    "compute_dei",
    "get_model",
    "load_trained_model",
    "predict_from_signal",
    "predict_distribution",
    "resolve_checkpoint_path",
    "validate_signal_grid",
]

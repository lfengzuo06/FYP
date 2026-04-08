"""Slim 2D DEXSY package for training and Colab inference."""

from .forward_model_2d import ForwardModel2D, compute_dei
from .inference_2d import (
    build_model_inputs,
    build_position_channel,
    load_trained_model,
    predict_distribution,
)
from .model_2d import (
    AttentionUNet2D,
    MultiscaleUNet2D,
    PhysicsInformedLoss,
    get_model,
)

__all__ = [
    "AttentionUNet2D",
    "MultiscaleUNet2D",
    "PhysicsInformedLoss",
    "ForwardModel2D",
    "build_model_inputs",
    "build_position_channel",
    "compute_dei",
    "get_model",
    "load_trained_model",
    "predict_distribution",
]

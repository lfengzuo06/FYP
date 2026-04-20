"""
Neural Operators for 2D DEXSY Inversion.

Available models:
- DeepONet: Branch + Trunk network architecture
- FNO: Fourier Neural Operator

Reference: Paper Section 2.4.4 - Neural Operators

Usage:
    from models_2d.neural_operators import DeepONet2D, FNO2D

    # DeepONet
    model = DeepONet2D(signal_dim=64*64, grid_size=64)
    spectrum = model(signal)

    # FNO
    model = FNO2D(in_channels=3, hidden_channels=64, n_layers=4)
    spectrum = model(model_inputs)
"""

from .model import (
    DeepONet2D,
    FNO2D,
    get_deeponet_model,
    get_fno_model,
)
from .inference import InferencePipeline, PredictionResult
from . import train
from . import inference

__all__ = [
    "DeepONet2D",
    "FNO2D",
    "get_deeponet_model",
    "get_fno_model",
    "InferencePipeline",
    "PredictionResult",
    "train",
    "inference",
]

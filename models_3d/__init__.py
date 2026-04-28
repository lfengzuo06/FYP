"""
Attention U-Net model for 3-Compartment DEXSY Inversion.

This module provides the Attention U-Net architecture adapted for 3C data,
with physics-informed training support.
"""

from .attention_unet.model import AttentionUNet3C, PhysicsInformedLoss3C, get_model
from .attention_unet.train import train_model
from .attention_unet.inference import InferencePipeline3C, predict, predict_batch

__all__ = [
    "AttentionUNet3C",
    "PhysicsInformedLoss3C",
    "get_model",
    "train_model",
    "InferencePipeline3C",
    "predict",
    "predict_batch",
]

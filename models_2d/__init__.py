"""
2D DEXSY Models Module.

This module contains all 2D model architectures for DEXSY inversion.

Available models:
- attention_unet: Attention U-Net with physics-informed loss (Section 2.4.2)
- plain_unet: Plain U-Net CNN baseline (Section 2.4.1)

Each model implements the standard interface:
- model.py: Model architecture and loss function
- train.py: Training pipeline
- inference.py: Inference pipeline with unified interface
"""

from .attention_unet import (
    AttentionUNet2D,
    InferencePipeline as AttentionInferencePipeline,
    train_model as train_attention_unet,
    predict as predict_attention_unet,
)

from .plain_unet import (
    PlainUNet2D,
    InferencePipeline as PlainInferencePipeline,
    train_model as train_plain_unet,
    predict as predict_plain_unet,
)

__all__ = [
    # Attention U-Net
    "AttentionUNet2D",
    "AttentionInferencePipeline",
    "train_attention_unet",
    "predict_attention_unet",
    # Plain U-Net
    "PlainUNet2D",
    "PlainInferencePipeline",
    "train_plain_unet",
    "predict_plain_unet",
]

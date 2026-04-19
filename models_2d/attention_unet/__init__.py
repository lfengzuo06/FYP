"""
Attention U-Net model for 2D DEXSY Inversion.

Standard interface module that exports:
- AttentionUNet2D: Model architecture
- InferencePipeline: Inference pipeline with standard interface
- PhysicsInformedLoss: Loss function for training
- predict: High-level inference function
- train_model: Training function

Usage:
    from models_2d.attention_unet import train_model, predict, InferencePipeline

    # Training
    model, history, datasets, fm = train_model(n_train=9500, epochs=60)

    # Inference
    result = predict(signal)
    pipeline = InferencePipeline()
    result = pipeline.predict(signal)
"""

from .model import (
    AttentionUNet2D,
    PhysicsInformedLoss,
    get_model,
    AttentionGate,
    ResidualDenseBlock,
)

from .inference import (
    InferencePipeline,
    PredictionResult,
    predict,
    predict_batch,
    predict_distribution,
    load_trained_model,
)

from .train import (
    train_model,
    generate_dataset,
    DEXSYDataset,
)

__all__ = [
    # Model
    "AttentionUNet2D",
    "PhysicsInformedLoss",
    "get_model",
    "AttentionGate",
    "ResidualDenseBlock",
    # Inference
    "InferencePipeline",
    "PredictionResult",
    "predict",
    "predict_batch",
    "load_trained_model",
    # Training
    "train_model",
    "generate_dataset",
    "DEXSYDataset",
]

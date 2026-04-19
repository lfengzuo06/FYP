"""
Plain U-Net model for 2D DEXSY Inversion.

CNN baseline implementation as described in paper Section 2.4.1.

Standard interface module that exports:
- PlainUNet2D: Model architecture
- PlainUNetLoss: Loss function for training
- InferencePipeline: Inference pipeline with standard interface
- predict: High-level inference function
- train_model: Training function

Usage:
    from models_2d.plain_unet import train_model, predict, InferencePipeline

    # Training
    model, history, datasets, fm = train_model(n_train=9500, epochs=60)

    # Inference
    pipeline = InferencePipeline(checkpoint_path="path/to/checkpoint.pt")
    result = pipeline.predict(signal)
"""

from .model import (
    PlainUNet2D,
    PlainUNetLoss,
    get_model,
)

from .inference import (
    InferencePipeline,
    PredictionResult,
    predict,
    predict_batch,
)

from .train import (
    train_model,
    generate_dataset,
    DEXSYDataset,
)

__all__ = [
    # Model
    "PlainUNet2D",
    "PlainUNetLoss",
    "get_model",
    # Inference
    "InferencePipeline",
    "PredictionResult",
    "predict",
    "predict_batch",
    # Training
    "train_model",
    "generate_dataset",
    "DEXSYDataset",
]

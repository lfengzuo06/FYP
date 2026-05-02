"""
Attention U-Net model for 3-Compartment DEXSY Inversion.

This module provides the Attention U-Net architecture adapted for 3C data,
with physics-informed training support.
"""

from .attention_unet.model import AttentionUNet3C, PhysicsInformedLoss3C, get_model as get_attention_unet
from .attention_unet.train import train_model as train_attention_unet
from .attention_unet.inference import InferencePipeline3C, predict as predict_attention, predict_batch as predict_batch_attention

from .plain_unet.model import PlainUNet3C, PlainUNetLoss3C, get_model as get_plain_unet
from .plain_unet.train import train_model as train_plain_unet
from .plain_unet.inference import InferencePipelinePlain3C, predict as predict_plain, predict_batch as predict_batch_plain

from .deep_unfolding.model import DeepUnfolding3C, get_model as get_deep_unfolding
from .deep_unfolding.train import train_model as train_deep_unfolding
from .deep_unfolding.inference import InferencePipeline3C as InferencePipelineDeep3C, predict as predict_deep, predict_batch as predict_batch_deep, load_trained_model as load_deep_unfolding

from .pinn.model import PINN3C, PINNLoss3C, get_model as get_pinn
from .pinn.train import train_model as train_pinn
from .pinn.inference import PINNInferencePipeline3C, predict as predict_pinn, predict_batch as predict_batch_pinn, load_trained_model as load_pinn_model

__all__ = [
    # Attention U-Net
    "AttentionUNet3C",
    "PhysicsInformedLoss3C",
    "get_attention_unet",
    "train_attention_unet",
    "InferencePipeline3C",
    "predict_attention",
    "predict_batch_attention",
    # Plain U-Net
    "PlainUNet3C",
    "PlainUNetLoss3C",
    "get_plain_unet",
    "train_plain_unet",
    "InferencePipelinePlain3C",
    "predict_plain",
    "predict_batch_plain",
    # Deep Unfolding
    "DeepUnfolding3C",
    "get_deep_unfolding",
    "train_deep_unfolding",
    "InferencePipelineDeep3C",
    "predict_deep",
    "predict_batch_deep",
    "load_deep_unfolding",
    # PINN
    "PINN3C",
    "PINNLoss3C",
    "get_pinn",
    "train_pinn",
    "PINNInferencePipeline3C",
    "predict_pinn",
    "predict_batch_pinn",
    "load_pinn_model",
]

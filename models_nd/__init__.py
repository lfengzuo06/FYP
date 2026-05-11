"""
Models for N-Compartment DEXSY Inversion.

This package contains neural network models for N-compartment DEXSY data.
"""

from .attention_unet import (
    # Single-N model
    AttentionUNetND,
    PhysicsInformedLossND,
    get_model,
    get_loss,
    train_model,
    DEXSYDatasetND,
    # Unified model
    AttentionUNetUnified,
    UnifiedLoss,
    get_unified_model,
    get_unified_loss,
    train_unified_model,
    DEXSYDatasetMixedN,
    UnifiedInferencePipeline,
    UnifiedPredictionResult,
)

__all__ = [
    # Single-N model
    'AttentionUNetND',
    'PhysicsInformedLossND',
    'get_model',
    'get_loss',
    'train_model',
    'DEXSYDatasetND',
    # Unified model
    'AttentionUNetUnified',
    'UnifiedLoss',
    'get_unified_model',
    'get_unified_loss',
    'train_unified_model',
    'DEXSYDatasetMixedN',
    'UnifiedInferencePipeline',
    'UnifiedPredictionResult',
]

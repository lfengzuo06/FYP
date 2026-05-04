"""
Models for N-Compartment DEXSY Inversion.

This package contains neural network models for N-compartment DEXSY data.
Supports both single-N and unified (mixed-N) training.
"""

from .model import (
    AttentionUNetND,
    AttentionUNetUnified,
    PhysicsInformedLossND,
    UnifiedLoss,
    get_model,
    get_unified_model,
    get_loss,
    get_unified_loss,
)

from .train import train_model, DEXSYDatasetND
from .train_unified import train_unified_model, DEXSYDatasetMixedN

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
]

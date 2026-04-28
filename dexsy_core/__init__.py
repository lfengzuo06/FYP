"""
Shared scientific core module for DEXSY 2D/3D experiments.

This module provides:
- ForwardModel2D: Physics-based forward model for synthetic data generation
- compute_dei: Diffusion Exchange Index computation
- Utility functions for metrics and preprocessing
"""

from .forward_model import (
    ForwardModel2D,
    compute_dei,
    compute_weight_matrix_dei,
    compute_pair_blob_masses,
    compute_pair_blob_dei,
    compute_pairwise_3c_dei,
    local_square_mask,
)

__all__ = [
    "ForwardModel2D",
    "compute_dei",
    "compute_weight_matrix_dei",
    "compute_pair_blob_masses",
    "compute_pair_blob_dei",
    "compute_pairwise_3c_dei",
    "local_square_mask",
]

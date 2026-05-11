"""
Shared scientific core module for DEXSY 2D/3D experiments.

This module provides:
- ForwardModel2D: Physics-based forward model for synthetic data generation
- create_forward_model: Factory function for creating models with grid size profiles
- GRID_PROFILES: Predefined configurations for different grid sizes (64, 16)
- compute_dei: Diffusion Exchange Index computation
- Utility functions for metrics and preprocessing
"""

from .forward_model import (
    ForwardModel2D,
    create_forward_model,
    GRID_PROFILES,
    compute_dei,
    compute_weight_matrix_dei,
    compute_pair_blob_masses,
    compute_pair_blob_dei,
    compute_pairwise_3c_dei,
    local_square_mask,
)
from .forward_model_nc import (
    ForwardModelNC,
    GRID_PROFILES_NC,
    create_forward_model_nc,
    compute_nc_weight_matrix_dei,
)

__all__ = [
    "ForwardModel2D",
    "create_forward_model",
    "GRID_PROFILES",
    "compute_dei",
    "compute_weight_matrix_dei",
    "compute_pair_blob_masses",
    "compute_pair_blob_dei",
    "compute_pairwise_3c_dei",
    "local_square_mask",
    "ForwardModelNC",
    "GRID_PROFILES_NC",
    "create_forward_model_nc",
    "compute_nc_weight_matrix_dei",
]

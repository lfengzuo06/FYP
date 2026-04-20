"""
Neural Operators for 2D DEXSY Inversion.

Available models:
- DeepONet: Branch + Trunk network architecture
- FNO: Fourier Neural Operator

Reference: Paper Section 2.4.4 - Neural Operators
"""

from __future__ import annotations

from .deeponet import DeepONet2D, get_model as get_deeponet_model
from .fno import FNO2D, get_model as get_fno_model

__all__ = [
    "DeepONet2D",
    "FNO2D",
    "get_deeponet_model",
    "get_fno_model",
]

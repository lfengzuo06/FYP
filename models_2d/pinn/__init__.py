"""
Physics-Informed Neural Network (PINN) for 2D DEXSY Inversion.

PINNs incorporate physical laws (forward model) into the loss function:
- Forward equation constraint: S = K * f
- Non-negativity constraint for spectrum
- Energy/boundary conditions

Reference: Paper Section 2.4.4 - PINNs
"""

from .model import PINN2D, PINNLoss
from .train import train_model
from .inference import PINNInferencePipeline

__all__ = [
    "PINN2D",
    "PINNLoss",
    "train_model",
    "PINNInferencePipeline",
]

"""
DeepONet (Deep Operator Network) for 2D DEXSY Inversion.

DeepONet learns to map functions (operators) to functions. In our context:
- Branch Net: Takes the input signal as a fixed-dimensional vector
- Trunk Net: Takes the spatial coordinates (D1, D2 grid) as input
- Output: Predicted spectrum value at each grid point

Reference: Paper Section 2.4.4 - Neural Operators
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchNet(nn.Module):
    """
    Branch network of DeepONet.

    Takes the input signal as a fixed-dimensional vector.
    """

    def __init__(
        self,
        input_dim: int = 64 * 64,  # Flattened signal size
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            prev_dim = dim

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier for better training."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input signal [B, input_dim] or [B, 1, H, W]

        Returns:
            Branch features [B, output_dim]
        """
        if x.dim() == 4:
            x = x.view(x.shape[0], -1)
        return self.network(x)


class TrunkNet(nn.Module):
    """
    Trunk network of DeepONet.

    Takes the spatial coordinates (D1, D2) as input.
    """

    def __init__(
        self,
        input_dim: int = 2,  # (D1, D2) coordinates
        grid_size: int = 64,
        hidden_dims: list[int] | None = None,
        output_dim: int = 128,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 128]

        self.grid_size = grid_size
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier for better training."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Grid coordinates [B, grid_size * grid_size, 2] or [B * grid_size * grid_size, 2]

        Returns:
            Trunk features [B, num_points, output_dim] or [num_points, output_dim]
        """
        return self.network(coords)


class DeepONet2D(nn.Module):
    """
    DeepONet for 2D DEXSY Inversion.

    Architecture:
    - Branch Net: Processes the input signal into a feature vector
    - Trunk Net: Processes spatial coordinates into feature vectors
    - Dot product: Combines branch and trunk features to predict spectrum values

    The output is a 64x64 spectrum predicted from the 64x64 input signal.
    """

    def __init__(
        self,
        signal_dim: int = 64 * 64,
        grid_size: int = 64,
        branch_dims: list[int] | None = None,
        trunk_dims: list[int] | None = None,
        output_dim: int = 128,
    ):
        super().__init__()
        if branch_dims is None:
            branch_dims = [512, 256, 128]
        if trunk_dims is None:
            trunk_dims = [128, 128, 128]

        self.grid_size = grid_size
        self.signal_dim = signal_dim
        self.output_dim = output_dim

        self.branch = BranchNet(
            input_dim=signal_dim,
            hidden_dims=branch_dims,
        )

        self.trunk = TrunkNet(
            input_dim=2,  # (D1, D2) coordinates
            grid_size=grid_size,
            hidden_dims=trunk_dims,
            output_dim=output_dim,
        )

        # Output scaling factor - log parameterization for positivity and stability
        self.log_scale = nn.Parameter(torch.tensor(-4.0))  # exp(-4) ≈ 0.018

        # Output bias - initialize to reasonable values for normalized spectra
        # Since scale is small initially, set bias to produce output ~ 0.1-0.5
        self.bias = nn.Parameter(torch.zeros(1, grid_size * grid_size, 1))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepONet.

        Args:
            signal: Input signal [B, 1, H, W] or [B, H*W]

        Returns:
            Reconstructed spectrum [B, 1, H, W]
        """
        batch_size = signal.shape[0]

        # Branch network: process signal
        branch_out = self.branch(signal)  # [B, output_dim]
        # Expand to match number of grid points
        branch_out = branch_out.unsqueeze(1)  # [B, 1, output_dim]

        # Generate grid coordinates
        coords = self._generate_coordinates(batch_size, signal.device)  # [B, H*W, 2]

        # Trunk network: process coordinates
        trunk_out = self.trunk(coords)  # [B, H*W, output_dim]

        # Dot product: exp(log_scale) * (branch · trunk) + bias
        # Scale is log-parameterized for stability
        scale = torch.exp(self.log_scale)
        output = scale * torch.sum(branch_out * trunk_out, dim=-1, keepdim=True)  # [B, H*W, 1]
        output = output + self.bias  # [B, H*W, 1]

        # Apply non-negative activation
        output = F.softplus(output)  # [B, H*W, 1]

        # Reshape to 2D grid
        output = output.view(batch_size, 1, self.grid_size, self.grid_size)

        return output

    def _generate_coordinates(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate normalized grid coordinates [0, 1].

        Args:
            batch_size: Batch size
            device: Device

        Returns:
            Coordinates [batch_size, grid_size * grid_size, 2]
        """
        # Create meshgrid
        x = torch.linspace(0, 1, self.grid_size, device=device)
        y = torch.linspace(0, 1, self.grid_size, device=device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        # Stack and flatten
        coords = torch.stack([xx, yy], dim=-1).view(-1, 2)  # [H*W, 2]

        # Expand for batch
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W, 2]

        return coords

    def get_model_name(self) -> str:
        """Return the model name."""
        return "deeponet"


def get_model(
    signal_dim: int = 64 * 64,
    grid_size: int = 64,
    branch_dims: list[int] | None = None,
    trunk_dims: list[int] | None = None,
    output_dim: int = 128,
    **kwargs,
) -> DeepONet2D:
    """
    Factory function to create a DeepONet model.

    Args:
        signal_dim: Input signal dimension
        grid_size: Grid size for output
        branch_dims: Branch network hidden dimensions
        trunk_dims: Trunk network hidden dimensions
        output_dim: Output dimension for branch/trunk combination

    Returns:
        DeepONet2D model
    """
    return DeepONet2D(
        signal_dim=signal_dim,
        grid_size=grid_size,
        branch_dims=branch_dims,
        trunk_dims=trunk_dims,
        output_dim=output_dim,
        **kwargs,
    )

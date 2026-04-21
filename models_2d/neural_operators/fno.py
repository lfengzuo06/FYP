"""
FNO (Fourier Neural Operator) for 2D DEXSY Inversion.

FNO uses Fourier transforms to handle global dependencies efficiently:
- Transform to Fourier domain
- Apply learnable spectral filters
- Transform back to spatial domain

Reference: Paper Section 2.4.4 - Neural Operators
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2D(nn.Module):
    """
    Spectral convolution layer for FNO.

    Applies a 2D Fourier transform, multiplies by learnable weights,
    and applies inverse Fourier transform.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Learnable spectral weights (complex)
        self.weight_real = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, modes) * 0.02
        )
        self.weight_imag = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, modes) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C_in, H, W]

        Returns:
            Output tensor [B, C_out, H, W]
        """
        batch_size, channels, height, width = x.shape

        # FFT
        x_fft = torch.fft.rfft2(x, dim=(-2, -1))

        # Get frequency dimensions: [B, C_in, H, W//2+1]
        freq_h = x_fft.shape[2]
        freq_w = x_fft.shape[3]

        # Truncate to modes
        modes_h = min(self.modes, freq_h)
        modes_w = min(self.modes, freq_w)

        # Get the spectral weights
        weight = torch.complex(self.weight_real, self.weight_imag)

        # Pad weight to match frequency size [C_in, C_out, H, W//2+1]
        weight_full = torch.zeros(
            self.in_channels, self.out_channels, freq_h, freq_w,
            device=x.device, dtype=torch.complex64
        )
        weight_full[:, :, :modes_h, :modes_w] = weight[:, :, :modes_h, :modes_w]

        # Transpose to [C_in, B, H, W//2+1] for matrix multiplication
        x_fft = x_fft.permute(1, 0, 2, 3)  # [C_in, B, H, W//2+1]

        # Multiply in Fourier space (contract over input channels)
        # x_fft [C_in, B, H, W//2+1] @ weight_full [C_in, C_out, H, W//2+1]
        # Result: [C_out, B, H, W//2+1]
        x_fft = torch.einsum('c b h w, c o h w -> o b h w', x_fft, weight_full)

        # Transpose back to [B, C_out, H, W//2+1]
        x_fft = x_fft.permute(1, 0, 2, 3)

        # Inverse FFT
        x = torch.fft.irfft2(x_fft, s=(height, width), dim=(-2, -1))

        return x


class FNOBlock2D(nn.Module):
    """
    Fourier Neural Operator block.

    Contains:
    - Spectral convolution
    - Pointwise linear transformation
    - Activation
    - Residual connection
    """

    def __init__(
        self,
        channels: int,
        modes: int = 16,
        activation: str = "gelu",
    ):
        super().__init__()
        self.channels = channels

        self.spectral_conv = SpectralConv2D(channels, channels, modes=modes)
        self.linear = nn.Conv2d(channels, channels, kernel_size=1)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, C, H, W]
        """
        residual = x
        x = self.spectral_conv(x)
        x = self.linear(x)
        x = self.activation(x)
        x = x + residual
        return x


class FNO2D(nn.Module):
    """
    Fourier Neural Operator for 2D DEXSY Inversion.

    Architecture:
    - Input projection: 3-channel input (signal + log + positional) -> hidden channels
    - N FNO blocks with spectral convolutions
    - Output projection: hidden channels -> 1-channel spectrum

    Uses Fourier domain convolutions for efficient global dependencies.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        n_layers: int = 4,
        modes: int = 16,
        activation: str = "gelu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.modes = modes

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
        )

        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock2D(hidden_channels, modes=modes, activation=activation)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
            nn.Softplus(),  # Non-negative output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 3, H, W] (signal + log + positional)

        Returns:
            Reconstructed spectrum [B, 1, H, W]
        """
        # Input projection
        x = self.input_proj(x)

        # FNO blocks
        for block in self.fno_blocks:
            x = block(x)

        # Output projection
        x = self.output_proj(x)
        x = x / (x.sum(dim=(2, 3), keepdim=True) + 1e-8)

        return x

    def get_model_name(self) -> str:
        """Return the model name."""
        return "fno"


def get_model(
    in_channels: int = 3,
    hidden_channels: int = 64,
    n_layers: int = 4,
    modes: int = 16,
    **kwargs,
) -> FNO2D:
    """
    Factory function to create an FNO model.

    Args:
        in_channels: Input channels
        hidden_channels: Hidden channels for FNO blocks
        n_layers: Number of FNO layers
        modes: Number of Fourier modes

    Returns:
        FNO2D model
    """
    return FNO2D(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        modes=modes,
        **kwargs,
    )

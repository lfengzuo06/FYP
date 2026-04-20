"""
Deep Unfolding model for 2D DEXSY Inversion (ISTA-Net style).

This module implements the Deep Unfolding architecture based on the paper's
Section 2.4.3, which unrolls the ISTA (Iterative Shrinkage-Thresholding Algorithm)
optimization into a learnable neural network.

Key Mechanism:
- Each layer = one ISTA iteration
- x_{k+1} = soft_threshold(x_k - η * K^T(Kx_k - s), θ)
- Learnable step size (η) and threshold (θ) per layer
- Optional learnable denoiser per layer
- Forward kernel K is embedded into the architecture

Reference: Paper Section 2.4.3 - Deep Unfolding (ISTA-Net style)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_threshold(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Soft-thresholding operator (proximal operator for L1 norm).

    S_theta(x) = sign(x) * max(|x| - theta, 0)

    Args:
        x: Input tensor (flattened) [B, n_pixels]
        theta: Threshold parameter (scalar)

    Returns:
        Shrunk tensor [B, n_pixels]
    """
    return torch.sign(x) * F.relu(torch.abs(x) - theta)


class ISTAProximalStep(nn.Module):
    """
    Single ISTA (Iterative Shrinkage-Thresholding Algorithm) layer.

    Implements: x_{k+1} = soft_threshold(x_k - η * (K^T @ (Kx_k - s)), θ)

    The gradient is passed in already normalized, and this layer applies
    the proximal operator (soft-thresholding) with optional denoising.
    """

    def __init__(
        self,
        n_d: int = 64,
        use_denoiser: bool = True,
    ):
        super().__init__()
        self.n_d = n_d
        self.n_pixels = n_d * n_d
        self.use_denoiser = use_denoiser

        # Learnable threshold (θ) - initialized to very small value
        # Using log parameterization for stability
        self.log_threshold = nn.Parameter(torch.tensor(-8.0))  # ~0.0003

        # Optional learnable denoiser: CNN operating on 2D [B, 1, 64, 64]
        if use_denoiser:
            self.denoise_scale = nn.Parameter(torch.tensor(0.10))
            self.denoiser = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=3, padding=1),
            )

    def forward(
        self,
        x_thresh: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply denoiser to thresholded output.

        Args:
            x_thresh: Already thresholded and gradient-updated estimate [B, 1, H, W]

        Returns:
            Denoised estimate [B, 1, H, W]
        """
        if self.use_denoiser:
            # Bounded residual keeps unfolded iterations numerically stable.
            residual = torch.tanh(self.denoiser(x_thresh))
            x_denoised = x_thresh + self.denoise_scale * residual
        else:
            x_denoised = x_thresh

        return x_denoised


class DeepUnfolding2D(nn.Module):
    """
    Deep Unfolding (ISTA-Net style) for 2D DEXSY Inversion.

    Architecture:
    - Initial estimate module: estimate x0 from signal s
    - N ISTA layers: learned iterative refinement
    - Each layer has learnable step size and threshold
    - Optional denoiser per layer (operates on 2D)
    - Final activation ensures non-negativity

    The forward kernel K is embedded as fixed linear operations.
    """

    def __init__(
        self,
        n_layers: int = 12,
        n_d: int = 64,
        hidden_dim: int = 256,
        use_denoiser: bool = True,
        init_method: str = "mlp",
    ):
        """
        Initialize Deep Unfolding model.

        Args:
            n_layers: Number of ISTA layers (iterations)
            n_d: Grid dimension (n_d x n_d = 4096)
            hidden_dim: Hidden dimension for denoiser CNN
            use_denoiser: Whether to use learnable denoiser per layer
            init_method: How to initialize x0 ('mlp', 'zero', 'constant')
        """
        super().__init__()
        self.n_layers = n_layers
        self.n_d = n_d
        self.n_pixels = n_d * n_d
        self.use_denoiser = use_denoiser
        if init_method not in {"mlp", "zero", "constant"}:
            raise ValueError("init_method must be one of: 'mlp', 'zero', 'constant'")
        self.init_method = init_method

        # Learnable initial estimate module
        self.init_estimator = nn.Sequential(
            nn.Linear(self.n_pixels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_pixels),
        )

        # ISTA layers (deep unfolding)
        # Each layer has learnable step size and threshold
        self.ista_layers = nn.ModuleList([
            ISTAProximalStep(
                n_d=n_d,
                use_denoiser=use_denoiser,
            )
            for _ in range(n_layers)
        ])

        # Learnable step sizes per layer (log parameterization)
        self.log_eta = nn.Parameter(torch.zeros(n_layers))

        # Output activation: softplus for non-negativity
        self.output_activation = nn.Softplus()

        # Register buffer for kernel matrix
        self.register_buffer('_K', None)
        self.register_buffer('_Kt', None)

    def set_kernel_matrix(self, K: torch.Tensor):
        """
        Set the forward kernel matrix from ForwardModel2D.

        Args:
            K: Kernel matrix of shape [n_pixels, n_pixels]
        """
        self._K = K
        self._Kt = K.T

    def _compute_K(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Compute K @ x."""
        if self._K is None:
            raise RuntimeError(
                "Kernel matrix not set. Call set_kernel_matrix() first."
            )
        return torch.matmul(x_flat, self._K.T)

    def _compute_Kt(self, y_flat: torch.Tensor) -> torch.Tensor:
        """Compute K^T @ y."""
        if self._Kt is None:
            raise RuntimeError(
                "Kernel matrix not set. Call set_kernel_matrix() first."
            )
        return torch.matmul(y_flat, self._Kt.T)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Deep Unfolding model.

        Args:
            signal: Input signal [B, 1, H, W] or [B, H*W]

        Returns:
            Reconstructed spectrum [B, 1, H, W]
        """
        batch_size = signal.shape[0]

        # Flatten signal if needed
        if signal.dim() == 4:
            s_flat = signal.view(batch_size, -1)
        else:
            s_flat = signal

        # Initial estimate x0
        if self.init_method == "zero":
            x_flat = torch.zeros(
                batch_size, self.n_pixels,
                device=signal.device, dtype=signal.dtype
            )
        elif self.init_method == "constant":
            x_flat = torch.ones(
                batch_size, self.n_pixels,
                device=signal.device, dtype=signal.dtype
            ) * 0.1
        else:  # mlp
            s_centered = s_flat - s_flat.mean(dim=1, keepdim=True)
            s_norm = s_centered / (s_flat.std(dim=1, keepdim=True) + 1e-6)
            # Bounded initialization prevents numerical blow-up at early layers.
            x_flat = 0.1 * torch.tanh(self.init_estimator(s_norm))

        # Reshape to 2D for iterative refinement.
        x_2d = x_flat.view(batch_size, 1, self.n_d, self.n_d)

        # ISTA iterations
        for layer_idx, layer in enumerate(self.ista_layers):
            # Compute gradient: K^T @ (K @ x - s)
            x_flat_prev = x_2d.view(batch_size, -1)
            Kx = self._compute_K(x_flat_prev)
            residual = Kx - s_flat
            gradient_flat = self._compute_Kt(residual)

            # Normalize gradient to have unit norm (for stability)
            grad_norm = torch.norm(gradient_flat, p=2, dim=1, keepdim=True) + 1e-8
            gradient_normalized = gradient_flat / grad_norm

            # Step size (learnable, log parameterization for positivity)
            eta = torch.exp(self.log_eta[layer_idx])

            # Gradient step: x = x - eta * gradient
            x_flat = x_flat_prev - eta * gradient_normalized

            # Soft-thresholding (learnable threshold, log parameterization)
            theta = torch.exp(layer.log_threshold)
            x_flat = soft_threshold(x_flat, theta)

            # Reshape to 2D
            x_2d = x_flat.view(batch_size, 1, self.n_d, self.n_d)

            # Denoiser step
            x_2d = layer(x_2d)
            x_2d = torch.clamp(x_2d, min=-20.0, max=20.0)

            # Reshape back for next iteration
            x_flat = x_2d.view(batch_size, -1)

        # Final projection to a stable, non-negative normalized spectrum.
        return self._normalize_distribution(x_2d)

    def _normalize_distribution(self, x_2d: torch.Tensor) -> torch.Tensor:
        """
        Normalize spectrum mass to 1 per sample with numeric guard.

        Adding epsilon avoids an all-zero collapse from producing NaNs.
        """
        x_2d = self.output_activation(x_2d) + 1e-8
        return x_2d / (x_2d.sum(dim=(2, 3), keepdim=True) + 1e-8)

    def get_model_name(self) -> str:
        """Return the model name."""
        return "deep_unfolding"

    def get_layer_info(self) -> dict:
        """Return information about the model layers."""
        return {
            "n_layers": self.n_layers,
            "use_denoiser": self.use_denoiser,
            "init_method": self.init_method,
            "step_sizes": [torch.exp(self.log_eta[i]).item() for i in range(self.n_layers)],
            "thresholds": [torch.exp(l.log_threshold).item() for l in self.ista_layers],
        }


def get_model(
    n_layers: int = 12,
    n_d: int = 64,
    hidden_dim: int = 256,
    use_denoiser: bool = True,
    **kwargs,
) -> DeepUnfolding2D:
    """
    Factory function to create a Deep Unfolding model.

    Args:
        n_layers: Number of ISTA layers
        n_d: Grid dimension
        hidden_dim: Hidden dimension for denoiser
        use_denoiser: Whether to use learnable denoiser

    Returns:
        DeepUnfolding2D model
    """
    return DeepUnfolding2D(
        n_layers=n_layers,
        n_d=n_d,
        hidden_dim=hidden_dim,
        use_denoiser=use_denoiser,
        **kwargs,
    )

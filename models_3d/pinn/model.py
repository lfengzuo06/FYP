"""
Physics-Informed Neural Network (PINN) model for 3-Compartment DEXSY.

PINNs embed the forward model equation into the loss:
    S = K * f
where K is the kernel matrix and f is the spectrum.

Key features:
- Learn spectrum distribution f from signal S
- Forward consistency constraint in loss
- Non-negative output via softplus
- Energy-preserving architecture

Architecture: Simple encoder-decoder with physics-informed loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEncoderDecoder(nn.Module):
    """
    Simple encoder-decoder architecture for spectrum prediction.
    
    Input: 64x64, Output: 64x64
    Architecture: 64 -> 32 -> 16 -> 16 -> 32 -> 64
    """
    
    def __init__(self, in_channels: int = 3, base_filters: int = 64):
        super().__init__()
        
        # Encoder: 64 -> 32 -> 16
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck: 16 -> 16
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, 3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
        )
        
        # Decoder: 16 -> 32 -> 64
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_filters * 4, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_filters * 2, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        
        # Output layer: 64 -> 64
        self.output = nn.Sequential(
            nn.Conv2d(base_filters, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)      # 64 -> 64
        e2 = self.enc2(self.pool1(e1))  # 64 -> 32 -> 32
        
        # Bottleneck
        b = self.bottleneck(self.pool2(e2))  # 32 -> 16 -> 16
        
        # Decoder
        d2 = self.up2(b)  # 16 -> 32
        d1 = self.up1(d2)  # 32 -> 64
        
        # Output
        out = self.output(d1)  # 64 -> 64
        
        return out


class PINN3C(nn.Module):
    """
    Physics-Informed Neural Network for 3-Compartment DEXSY Inversion.
    
    Uses a simple encoder-decoder architecture with physics-informed loss.
    The forward physics (S = K*f) is enforced in the loss function.
    
    Input: (batch, 3, 64, 64) - 3-channel input
    Output: (batch, 1, 64, 64) - normalized distribution
    """
    
    def __init__(
        self,
        signal_size: int = 64,
        in_channels: int = 3,
        base_filters: int = 64,
    ):
        super().__init__()
        self.signal_size = signal_size
        self.base_filters = base_filters
        
        # Encoder-decoder for spectrum prediction
        self.model = SimpleEncoderDecoder(in_channels=in_channels, base_filters=base_filters)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PINN.
        
        Args:
            x: Input signal [B, C, H, W] where C is typically 3
               (raw signal, log signal, position encoding)
               
        Returns:
            Reconstructed spectrum [B, 1, H, W]
        """
        # Predict raw spectrum
        raw_spectrum = self.model(x)
        
        # Apply softplus for non-negativity
        spectrum = F.softplus(raw_spectrum)
        
        # Normalize to unit sum (energy preserving)
        spectrum_sum = spectrum.sum(dim=(2, 3), keepdim=True) + 1e-8
        spectrum = spectrum / spectrum_sum
        
        return spectrum
    
    def get_model_name(self) -> str:
        """Return model name."""
        return "pinn_3c"


class PINNLoss3C(nn.Module):
    """
    Physics-Informed Loss for PINN on 3C DEXSY.
    
    Components:
    1. Reconstruction loss: MSE between predicted and true spectrum
    2. Forward consistency: ||S - K^T*f_pred||^2 (physics constraint)
    3. Smoothness regularization: Laplacian on spectrum
    4. Non-negativity enforcement via softplus (built into forward pass)
    """
    
    def __init__(
        self,
        forward_model,
        alpha_recon: float = 1.0,
        alpha_forward: float = 1.0,
        alpha_smooth: float = 0.1,
        peak_weight: float = 10.0,
    ):
        super().__init__()
        # Register kernel matrix as buffer
        kernel = torch.from_numpy(forward_model.kernel_matrix).float()
        self.register_buffer("kernel_matrix", kernel)
        self.n_b = forward_model.n_b
        self.n_d = forward_model.n_d
        self.alpha_recon = alpha_recon
        self.alpha_forward = alpha_forward
        self.alpha_smooth = alpha_smooth
        self.peak_weight = peak_weight
    
    def forward_consistency(
        self, 
        spectrum_pred: torch.Tensor, 
        signal_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute forward consistency loss: ||S - K^T*f||^2
        
        Args:
            spectrum_pred: [B, 1, H, W] predicted spectrum (normalized)
            signal_input: [B, C, H, W] input signal (C channels)
        """
        batch_size = spectrum_pred.shape[0]
        
        # Use first channel (raw signal) for forward consistency
        if signal_input.dim() == 4 and signal_input.shape[1] > 1:
            signal = signal_input[:, 0:1]  # Take raw signal channel
        else:
            signal = signal_input
        
        # Flatten spectrum: [B, 1, H, W] -> [B, H*W]
        f_flat = spectrum_pred.squeeze(1).reshape(batch_size, -1)
        
        # Forward model: S_pred = K^T @ f (transpose to match signal space)
        S_pred = f_flat @ self.kernel_matrix.T  # [B, H*W]
        
        # Target signal: [B, 1, H, W] -> [B, H*W]
        S_target = signal.squeeze(1).reshape(batch_size, -1)
        
        # Normalize both signals to [0, 1] range for fair comparison
        S_pred_min = S_pred.min(dim=1, keepdim=True)[0]
        S_pred_max = S_pred.max(dim=1, keepdim=True)[0]
        S_pred_norm = (S_pred - S_pred_min) / (S_pred_max - S_pred_min + 1e-8)
        
        S_target_min = S_target.min(dim=1, keepdim=True)[0]
        S_target_max = S_target.max(dim=1, keepdim=True)[0]
        S_target_norm = (S_target - S_target_min) / (S_target_max - S_target_min + 1e-8)
        
        # MSE between normalized signals
        return torch.mean((S_pred_norm - S_target_norm) ** 2)
    
    def reconstruction_loss(
        self,
        spectrum_pred: torch.Tensor,
        spectrum_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Peak-weighted reconstruction loss.
        
        Normalizes both prediction and target to ensure scale consistency.
        """
        # Normalize target
        target_sum = spectrum_target.sum(dim=(2, 3), keepdim=True)
        target_norm = spectrum_target / (target_sum + 1e-8)
        
        # Peak weighting
        relative_peak = target_norm / (target_norm.amax(dim=(2, 3), keepdim=True) + 1e-8)
        weights = 1.0 + self.peak_weight * torch.sqrt(relative_peak + 1e-8)
        
        # Prediction is already normalized (unit sum), but we need to align it with target
        # First normalize prediction to match target's sum
        pred_sum = spectrum_pred.sum(dim=(2, 3), keepdim=True)
        pred_scaled = spectrum_pred * (target_sum / (pred_sum + 1e-8))
        
        return torch.mean(weights * (pred_scaled - target_norm) ** 2)
    
    def smoothness_loss(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        Laplacian smoothness regularization.
        """
        laplacian = spectrum[:, :, 1:-1, 1:-1] * 4 \
            - spectrum[:, :, :-2, 1:-1] \
            - spectrum[:, :, 2:, 1:-1] \
            - spectrum[:, :, 1:-1, :-2] \
            - spectrum[:, :, 1:-1, 2:]
        return torch.mean(laplacian ** 2)
    
    def forward(
        self,
        spectrum_pred: torch.Tensor,
        spectrum_target: torch.Tensor,
        signal_input: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute total physics-informed loss.
        
        Args:
            spectrum_pred: [B, 1, H, W] predicted spectrum
            spectrum_target: [B, 1, H, W] ground truth spectrum
            signal_input: [B, C, H, W] input signal
        """
        recon_loss = self.reconstruction_loss(spectrum_pred, spectrum_target)
        forward_loss = self.forward_consistency(spectrum_pred, signal_input)
        smooth_loss = self.smoothness_loss(spectrum_pred)
        
        total = (self.alpha_recon * recon_loss + 
                  self.alpha_forward * forward_loss + 
                  self.alpha_smooth * smooth_loss)
        
        return {
            'total': total,
            'reconstruction': recon_loss,
            'forward': forward_loss,
            'smoothness': smooth_loss,
        }


def get_model(
    signal_size: int = 64,
    in_channels: int = 3,
    base_filters: int = 64,
    **kwargs,
) -> PINN3C:
    """Factory function to create a PINN model for 3C."""
    return PINN3C(
        signal_size=signal_size,
        in_channels=in_channels,
        base_filters=base_filters,
        **kwargs,
    )


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PINN3C(signal_size=64, in_channels=3, base_filters=64).to(device)
    print(f"\nPINN3C (Physics-Informed Neural Network for 3C):")
    print(f"  Input: (batch, 3, 64, 64)")
    print(f"  Output: (batch, 1, 64, 64) normalized distribution")

    # Test forward pass
    x = torch.randn(2, 3, 64, 64).to(device)
    out = model(x)
    print(f"  Output shape: {out.shape}")
    print(f"  Output sum: {out.sum(dim=(2,3)).mean().item():.4f} (should be ~1.0)")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

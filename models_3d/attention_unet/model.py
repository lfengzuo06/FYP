"""
Attention U-Net model for 3-Compartment DEXSY Inversion.

This module provides the Attention U-Net architecture adapted for 3C data,
with physics-informed training support.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(num_channels: int) -> nn.GroupNorm:
    """Stable normalisation for small-batch training."""
    num_groups = min(8, num_channels)
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class AttentionGate(nn.Module):
    """
    Attention gate to focus on relevant features.

    The gate helps the decoder focus on relevant encoder features
    by learning attention weights.
    """

    def __init__(self, x_channels: int, g_channels: int, out_channels: int):
        super().__init__()
        self.W_x = nn.Conv2d(x_channels, out_channels, kernel_size=1, padding=0)
        self.W_g = nn.Conv2d(g_channels, out_channels, kernel_size=1, padding=0)
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.res_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        theta_x = self.W_x(x)
        phi_g = self.W_g(g)
        add = theta_x + phi_g
        act = F.leaky_relu(add, negative_slope=0.2)
        psi = self.psi(act)
        out = x * psi
        out = out + self.res_conv(out)
        return out


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block with improved gradient flow.

    Each layer receives the feature maps of all preceding layers,
    which helps feature reuse and gradient flow.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = _group_norm(out_channels)
        self.conv3 = nn.Conv2d(in_channels + 2 * out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = _group_norm(out_channels)
        if in_channels != out_channels:
            self.input_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.input_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        concat1 = torch.cat([x, c1], dim=1)
        c2 = F.leaky_relu(self.bn2(self.conv2(concat1)), negative_slope=0.2)
        concat2 = torch.cat([x, c1, c2], dim=1)
        c3 = self.bn3(self.conv3(concat2))
        identity = self.input_proj(x) if self.input_proj is not None else x
        res = identity + c3
        res = F.leaky_relu(res, negative_slope=0.2)
        return res


class AttentionUNet3C(nn.Module):
    """
    Attention U-Net for 3-Compartment DEXSY Inversion.

    Architecture:
    - Encoder: Conv + RDB + MaxPool
    - Bottleneck: Conv + RDB
    - Decoder: UpSample + Attention Gate + Conv + RDB
    - Skip connections carry multi-scale features

    Input: (batch, 3, 64, 64) for 3-channel input (signal + auxiliary features)
    Output: (batch, 1, 64, 64) normalized distribution
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.output_activation = nn.Softplus()

        self.input_norm = nn.InstanceNorm2d(in_channels, affine=False)

        # Encoder Path
        self.enc1 = ResidualDenseBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ResidualDenseBlock(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ResidualDenseBlock(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ResidualDenseBlock(base_filters * 4, base_filters * 8)

        # Decoder Path
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att3 = AttentionGate(base_filters * 4, base_filters * 8, base_filters * 4)
        self.dec3 = ResidualDenseBlock(base_filters * 4 + base_filters * 8, base_filters * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att2 = AttentionGate(base_filters * 2, base_filters * 4, base_filters * 2)
        self.dec2 = ResidualDenseBlock(base_filters * 2 + base_filters * 4, base_filters * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att1 = AttentionGate(base_filters, base_filters * 2, base_filters)
        self.dec1 = ResidualDenseBlock(base_filters + base_filters * 2, base_filters)

        # Output layer
        self.output_conv = nn.Conv2d(base_filters, 1, kernel_size=1, padding=0)

    def _normalize_distribution(self, x: torch.Tensor) -> torch.Tensor:
        x = self.output_activation(x)
        return x / (x.sum(dim=(2, 3), keepdim=True) + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.up3(b)
        a3 = self.att3(e3, d3)
        d3 = self.dec3(torch.cat([a3, d3], dim=1))

        d2 = self.up2(d3)
        a2 = self.att2(e2, d2)
        d2 = self.dec2(torch.cat([a2, d2], dim=1))

        d1 = self.up1(d2)
        a1 = self.att1(e1, d1)
        d1 = self.dec1(torch.cat([a1, d1], dim=1))

        out = self._normalize_distribution(self.output_conv(d1))
        return out


class PhysicsInformedLoss3C(nn.Module):
    """
    Physics-informed loss for 3C DEXSY training.

    The loss consists of:
    1. KL divergence for distribution matching
    2. Peak-weighted reconstruction loss
    3. Forward consistency loss in signal space
    4. Sum-to-one penalty
    5. Smoothness regularization
    """

    def __init__(
        self,
        forward_model,
        alpha_kl: float = 1.0,
        alpha_rec: float = 0.2,
        alpha_signal: float = 0.1,
        alpha_sum: float = 0.05,
        peak_weight: float = 6.0,
        alpha_smooth: float = 2e-2,
    ):
        super().__init__()
        kernel = torch.from_numpy(forward_model.kernel_matrix).float()
        self.register_buffer("kernel_matrix", kernel)
        self.n_b = forward_model.n_b
        self.alpha_kl = alpha_kl
        self.alpha_rec = alpha_rec
        self.alpha_signal = alpha_signal
        self.alpha_sum = alpha_sum
        self.peak_weight = peak_weight
        self.alpha_smooth = alpha_smooth

    def reconstruct_signal(self, y_pred: torch.Tensor) -> torch.Tensor:
        batch_size = y_pred.shape[0]
        pred_flat = y_pred.squeeze(1).reshape(batch_size, -1)
        signal_flat = pred_flat @ self.kernel_matrix.T
        return signal_flat.view(batch_size, 1, self.n_b, self.n_b)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        signal_targets: torch.Tensor,
    ) -> torch.Tensor:
        y_pred = y_pred / (y_pred.sum(dim=(2, 3), keepdim=True) + 1e-8)
        y_true = y_true / (y_true.sum(dim=(2, 3), keepdim=True) + 1e-8)

        relative_peak_map = y_true / (y_true.amax(dim=(2, 3), keepdim=True) + 1e-8)
        weights = 1.0 + self.peak_weight * torch.sqrt(relative_peak_map + 1e-8)
        rec_loss = torch.mean(weights * (y_pred - y_true) ** 2)
        kl_loss = F.kl_div(torch.log(y_pred + 1e-8), y_true, reduction="batchmean")

        pred_signals = self.reconstruct_signal(y_pred)
        signal_loss = F.mse_loss(pred_signals, signal_targets)

        total_mass = y_pred.sum(dim=(2, 3), keepdim=True)
        sum_penalty = torch.mean((total_mass - 1.0) ** 2)
        smooth_x = torch.mean(torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]))
        smooth_y = torch.mean(torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]))
        smoothness = smooth_x + smooth_y

        total_loss = (
            self.alpha_kl * kl_loss +
            self.alpha_rec * rec_loss +
            self.alpha_signal * signal_loss +
            self.alpha_sum * sum_penalty +
            self.alpha_smooth * smoothness
        )
        return total_loss


def get_model(base_filters: int = 32, in_channels: int = 3) -> nn.Module:
    """
    Factory function to create the Attention U-Net model for 3C data.

    Args:
        base_filters: Base number of filters
        in_channels: Number of input channels

    Returns:
        AttentionUNet3C model
    """
    return AttentionUNet3C(in_channels=in_channels, base_filters=base_filters)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = AttentionUNet3C(in_channels=3, base_filters=32).to(device)
    print(f"\nAttentionUNet3C for 3-Compartment DEXSY:")
    print(f"  Input: (batch, 3, 64, 64)")

    x = torch.randn(2, 3, 64, 64).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"  Output shape: {out.shape}")
    print(f"  Output sum: {out.sum(dim=(2, 3)).mean().item():.4f} (should be ~1.0)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

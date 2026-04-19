"""
Plain U-Net model for 2D DEXSY Inversion.

This implements the CNN baseline from paper Section 2.4.1:
- 4 encoder blocks (2 conv each) + 4 decoder blocks
- Max pooling downsampling
- Bilinear upsampling + conv refinement (NOT transposed conv)
- Skip connections
- Channel progression: 32 -> 64 -> 128 -> 256
- Single-channel output
- NO attention gates, NO auxiliary mask branch
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


class EncoderBlock(nn.Module):
    """
    Plain encoder block: 2x(Conv3x3 + BatchNorm + ReLU) + MaxPool.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = _group_norm(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        skip = x
        x = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    """
    Plain decoder block: BilinearUpsample + Conv3x3 refinement + skip connection.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # After upsampling: in_channels channels, then concatenate with skip_channels
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = _group_norm(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch between upsampled feature and skip connection
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class PlainUNet2D(nn.Module):
    """
    Plain U-Net for 2D DEXSY Inversion.

    Architecture (from paper Section 2.4.1):
    - 4 encoder blocks: 2x(conv 3x3 + BatchNorm + ReLU) + MaxPool
    - 4 decoder blocks: BilinearUpsample + 2x(conv 3x3 + BatchNorm + ReLU)
    - Skip connections
    - Channel progression: 32 -> 64 -> 128 -> 256
    - Output: 1x64x64 (distribution)
    - NO attention gates, NO auxiliary mask branch
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.output_activation = nn.Softplus()

        # Input normalization
        self.input_norm = nn.InstanceNorm2d(in_channels, affine=False)

        # Encoder Path
        # Level 1: 64x64 -> 32x32, channels: in -> base
        self.enc1 = EncoderBlock(in_channels, base_filters)
        # Level 2: 32x32 -> 16x16, channels: base -> base*2
        self.enc2 = EncoderBlock(base_filters, base_filters * 2)
        # Level 3: 16x16 -> 8x8, channels: base*2 -> base*4
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4)
        # Level 4: 8x8 -> 4x4, channels: base*4 -> base*8
        self.enc4 = EncoderBlock(base_filters * 4, base_filters * 8)

        # Bottleneck: 4x4 -> 4x4, channels: base*8 -> base*8
        self.bottleneck_conv1 = nn.Conv2d(base_filters * 8, base_filters * 8, kernel_size=3, padding=1)
        self.bottleneck_bn1 = _group_norm(base_filters * 8)
        self.bottleneck_conv2 = nn.Conv2d(base_filters * 8, base_filters * 8, kernel_size=3, padding=1)
        self.bottleneck_bn2 = _group_norm(base_filters * 8)

        # Decoder Path
        # Level 4: 4x4 -> 8x8, channels: base*8 + base*8(skip) -> base*4
        self.dec4 = DecoderBlock(base_filters * 8, base_filters * 8, base_filters * 4)
        # Level 3: 8x8 -> 16x16, channels: base*4 + base*4(skip) -> base*2
        self.dec3 = DecoderBlock(base_filters * 4, base_filters * 4, base_filters * 2)
        # Level 2: 16x16 -> 32x32, channels: base*2 + base*2(skip) -> base
        self.dec2 = DecoderBlock(base_filters * 2, base_filters * 2, base_filters)
        # Level 1: 32x32 -> 64x64, channels: base + base(skip) -> base
        self.dec1 = DecoderBlock(base_filters, base_filters, base_filters)

        # Output layer
        self.output_conv = nn.Conv2d(base_filters, 1, kernel_size=1, padding=0)

    def _normalize_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize output to sum to 1 (probability distribution)."""
        x = self.output_activation(x)
        return x / (x.sum(dim=(2, 3), keepdim=True) + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)

        # Encoder
        e1, s1 = self.enc1(x)    # e1: 32x32, s1: 64x64
        e2, s2 = self.enc2(e1)   # e2: 16x16, s2: 32x32
        e3, s3 = self.enc3(e2)   # e3: 8x8, s3: 16x16
        e4, s4 = self.enc4(e3)   # e4: 4x4, s4: 8x8

        # Bottleneck
        b = F.relu(self.bottleneck_bn1(self.bottleneck_conv1(e4)))
        b = F.relu(self.bottleneck_bn2(self.bottleneck_conv2(b)))

        # Decoder with skip connections
        d4 = self.dec4(b, s4)    # 8x8
        d3 = self.dec3(d4, s3)    # 16x16
        d2 = self.dec2(d3, s2)    # 32x32
        d1 = self.dec1(d2, s1)    # 64x64

        out = self._normalize_distribution(self.output_conv(d1))
        return out


class PlainUNetLoss(nn.Module):
    """
    Loss function for Plain U-Net.

    Uses MSE + Smoothness (simpler than physics-informed loss used by Attention U-Net).
    This is appropriate for the CNN baseline as described in the paper.
    """

    def __init__(
        self,
        alpha_smooth: float = 0.01,
        peak_weight: float = 4.0,
    ):
        super().__init__()
        self.alpha_smooth = alpha_smooth
        self.peak_weight = peak_weight

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss.

        Args:
            y_pred: Predicted distribution (batch, 1, H, W)
            y_true: Ground truth distribution (batch, 1, H, W)

        Returns:
            Loss value
        """
        # Normalize predictions
        y_pred = y_pred / (y_pred.sum(dim=(2, 3), keepdim=True) + 1e-8)
        y_true = y_true / (y_true.sum(dim=(2, 3), keepdim=True) + 1e-8)

        # Peak-weighted MSE
        relative_peak_map = y_true / (y_true.amax(dim=(2, 3), keepdim=True) + 1e-8)
        weights = 1.0 + self.peak_weight * torch.sqrt(relative_peak_map + 1e-8)
        mse_loss = torch.mean(weights * (y_pred - y_true) ** 2)

        # Smoothness regularization
        smooth_x = torch.mean(torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]))
        smooth_y = torch.mean(torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]))
        smoothness = smooth_x + smooth_y

        total_loss = mse_loss + self.alpha_smooth * smoothness
        return total_loss


def get_model(base_filters: int = 32, in_channels: int = 3) -> nn.Module:
    """
    Factory function to create the Plain U-Net model.

    Args:
        base_filters: Base number of filters
        in_channels: Number of input channels

    Returns:
        PlainUNet2D model
    """
    return PlainUNet2D(in_channels=in_channels, base_filters=base_filters)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PlainUNet2D(in_channels=3, base_filters=32).to(device)
    print(f"\nPlainUNet2D (CNN Baseline):")
    print(f"  Input: (batch, 3, 64, 64)")

    # Test forward pass
    x = torch.randn(2, 3, 64, 64).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"  Output shape: {out.shape}")
    print(f"  Output sum: {out.sum(dim=(2, 3))}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Compare with Attention U-Net
    from models_2d.attention_unet import AttentionUNet2D
    attn_model = AttentionUNet2D(in_channels=3, base_filters=32).to(device)
    n_params_attn = sum(p.numel() for p in attn_model.parameters())
    print(f"\n  Attention U-Net parameters: {n_params_attn:,}")
    print(f"  Ratio: {n_params/n_params_attn:.2f}x")

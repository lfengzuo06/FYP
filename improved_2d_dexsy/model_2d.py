"""
Physics-Informed U-Net for 2D DEXSY Inversion (PyTorch).

Key improvements over baseline (ResNet-Dense-SE):
1. U-Net architecture with skip connections preserves spatial information
2. Residual Dense blocks for better feature extraction
3. Attention gates focus on relevant diffusion coefficient regions
4. Multi-scale output heads for different compartment scales
5. Physics-informed loss: encourages output to satisfy forward model consistency
"""

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
        # x: skip connection (encoder), g: gating signal (decoder)
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
        # Project input channels to match output channels for residual
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
        # Project input if needed
        identity = self.input_proj(x) if self.input_proj is not None else x
        res = identity + c3
        res = F.leaky_relu(res, negative_slope=0.2)
        return res


class AttentionUNet2D(nn.Module):
    """
    Attention U-Net for 2D DEXSY Inversion.

    Architecture:
    - Encoder: Conv + RDB + MaxPool
    - Bottleneck: Conv + RDB
    - Decoder: UpSample + Attention Gate + Conv + RDB
    - Skip connections carry multi-scale features
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.output_activation = nn.Softplus()

        # Normalize per channel without hard-coding the spatial grid size.
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
        # Level 3: cat(att_gate=256, up=512) = 768 -> 256
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att3 = AttentionGate(base_filters * 4, base_filters * 8, base_filters * 4)
        self.dec3 = ResidualDenseBlock(base_filters * 4 + base_filters * 8, base_filters * 4)

        # Level 2: cat(att_gate=128, up=256) = 384 -> 128
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att2 = AttentionGate(base_filters * 2, base_filters * 4, base_filters * 2)
        self.dec2 = ResidualDenseBlock(base_filters * 2 + base_filters * 4, base_filters * 2)

        # Level 1: cat(att_gate=64, up=128) = 192 -> 64
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

        # Decoder: cat(up, skip) -> RDB -> out_channels.
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


class MultiscaleUNet2D(nn.Module):
    """
    Multi-scale U-Net that predicts at multiple resolutions.

    This helps capture both fine-grained peaks and global structure.
    The final output is the combination of multi-scale predictions.
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 32):
        super().__init__()
        self.base_filters = base_filters
        self.output_activation = nn.Softplus()

        self.input_norm = nn.InstanceNorm2d(in_channels, affine=False)

        # Encoder: outputs are [bf, bf*2, bf*4, bf*8]
        self.enc1 = ResidualDenseBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ResidualDenseBlock(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = ResidualDenseBlock(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ResidualDenseBlock(base_filters * 4, base_filters * 8)

        # ---- Scale 1 decoder (full resolution) ----
        # Level 3: 2->4, cat with e3(4), att -> bf*8+bf*4=bf*12 -> bf*4
        self.up3_s1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att3 = AttentionGate(base_filters * 4, base_filters * 8, base_filters * 4)
        self.dec3 = ResidualDenseBlock(base_filters * 4 + base_filters * 8, base_filters * 4)

        # Level 2: 4->8, cat with e2(8), att -> bf*4+bf*2=bf*6 -> bf*2
        self.up2_s1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att2 = AttentionGate(base_filters * 2, base_filters * 4, base_filters * 2)
        self.dec2 = ResidualDenseBlock(base_filters * 2 + base_filters * 4, base_filters * 2)

        # Level 1: 8->16, cat with e1(16), att -> bf*2+bf=bf*3 -> bf
        self.up1_s1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att1 = AttentionGate(base_filters, base_filters * 2, base_filters)
        self.dec1 = ResidualDenseBlock(base_filters + base_filters * 2, base_filters)
        self.out1 = nn.Conv2d(base_filters, 1, kernel_size=1)

        # ---- Scale 2 decoder (coarse path) ----
        # Level 2: start from d3(4,4), e3 interp(8,8), b interp(8,8) -> 128+128+256=512 -> bf*2
        self.up2_s2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2_s2 = ResidualDenseBlock(
            base_filters * 4 + base_filters * 4 + base_filters * 8,
            base_filters * 2
        )

        # Level 1: 8->16, interp e1(16), interp d2_s2(16) -> 64+32+64=160 -> bf
        self.up1_s2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1_s2 = ResidualDenseBlock(
            base_filters * 2 + base_filters + base_filters * 2,
            base_filters
        )
        self.out2 = nn.Conv2d(base_filters, 1, kernel_size=1)

        # Final combine
        self.combine_conv = nn.Conv2d(2, 1, kernel_size=1)

    def _interp(self, x, target_size):
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

    def _normalize_distribution(self, x: torch.Tensor) -> torch.Tensor:
        x = self.output_activation(x)
        return x / (x.sum(dim=(2, 3), keepdim=True) + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)

        # Encoder
        e1 = self.enc1(x)         # bf=32, 16
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)        # bf*2=64, 8
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)        # bf*4=128, 4
        p3 = self.pool3(e3)
        b  = self.bottleneck(p3)  # bf*8=256, 2

        # ---- Scale 1 path ----
        d3 = self.up3_s1(b)                          # 256, 4
        a3 = self.att3(e3, d3)                       # 128, 4
        d3 = self.dec3(torch.cat([a3, d3], 1))       # 128, 4

        d2 = self.up2_s1(d3)                          # 128, 8
        a2 = self.att2(e2, d2)                        # 64, 8
        d2 = self.dec2(torch.cat([a2, d2], 1))       # 64, 8

        d1 = self.up1_s1(d2)                          # 64, 16
        a1 = self.att1(e1, d1)                        # 32, 16
        d1 = self.dec1(torch.cat([a1, d1], 1))        # 32, 16
        out1 = self._normalize_distribution(self.out1(d1))  # 1, 16

        # ---- Scale 2 path (coarse) ----
        d2_s2 = self.up2_s2(d3)                                                    # 128, 8
        e3_up  = self._interp(e3, d2_s2.shape[2:])                                 # 128, 8
        b_up   = self._interp(b,  d2_s2.shape[2:])                                 # 256, 8
        d2_s2  = self.dec2_s2(torch.cat([d2_s2, e3_up, b_up], 1))                 # 64, 8

        d1_s2 = self.up1_s2(d2_s2)                                                # 64, 16
        e1_up  = self._interp(e1, d1_s2.shape[2:])                                 # 32, 16
        d2_up  = self._interp(d2_s2, d1_s2.shape[2:])                             # 64, 16
        d1_s2  = self.dec1_s2(torch.cat([d1_s2, e1_up, d2_up], 1))                # 32, 16
        out2 = self._normalize_distribution(self.out2(d1_s2))                      # 1, 16

        # Combine multi-scale outputs
        combined = torch.cat([out1, out2], dim=1)   # 2, 16
        out_final = self._normalize_distribution(self.combine_conv(combined))
        return out_final


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss combining MSE with forward model consistency.

    The loss consists of:
    1. Weighted reconstruction loss on the target distribution
    2. Forward consistency loss in signal space
    3. Sum-to-one penalty for the predicted distribution
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


def get_model(model_name: str = "attention_unet", base_filters: int = 32,
              in_channels: int = 3,
              input_shape: tuple = None, learning_rate: float = 1e-3) -> nn.Module:
    """
    Factory function to create a model by name (PyTorch).

    Args:
        model_name: Name of the model ('attention_unet', 'multiscale_unet', 'physics_unet')
        base_filters: Base number of filters
        input_shape: Ignored (for Keras compatibility); the model is spatially flexible.
        learning_rate: Ignored (for Keras compatibility) - set in optimizer instead

    Returns:
        PyTorch model
    """
    """
    Factory function to create a model by name.

    Args:
        model_name: Name of the model ('attention_unet', 'multiscale_unet', 'physics_unet')
        base_filters: Base number of filters

    Returns:
        PyTorch model
    """
    if model_name == "attention_unet":
        model = AttentionUNet2D(in_channels=in_channels, base_filters=base_filters)
    elif model_name == "multiscale_unet":
        model = MultiscaleUNet2D(in_channels=in_channels, base_filters=base_filters)
    elif model_name == "physics_unet":
        model = AttentionUNet2D(in_channels=in_channels, base_filters=base_filters)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for model_name in ["attention_unet", "multiscale_unet", "physics_unet"]:
        model = get_model(model_name).to(device)
        print(f"\n{model_name}:")
        print(f"  Input: (batch, 3, 64, 64)")

        # Test forward pass
        x = torch.randn(2, 3, 64, 64).to(device)
        with torch.no_grad():
            out = model(x)
        print(f"  Output shape: {out.shape}")

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

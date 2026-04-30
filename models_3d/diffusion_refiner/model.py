"""
Conditional U-Net Denoiser for Diffusion Refinement.

This module implements a conditional denoising U-Net that learns to refine
the output of the 3C Attention UNet by predicting noise residuals.

Key features:
- Multi-scale conditioning from baseline prediction (f_base)
- Time embedding injection via FiLM (Feature-wise Linear Modulation)
- Shared encoder/decoder architecture with skip connections
- Outputs predicted noise epsilon for the diffusion process
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig
from .scheduler import SinusoidalPosEmb


def _group_norm(num_channels: int, num_groups: int = None) -> nn.GroupNorm:
    """Stable normalisation for small-batch training."""
    if num_groups is None:
        num_groups = min(8, num_channels)
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class TimeEmbedding(nn.Module):
    """
    Time embedding module that converts timestep to feature vector.

    Uses sinusoidal positional encoding followed by MLP.
    """

    def __init__(self, time_dim: int = 128):
        super().__init__()
        self.time_dim = time_dim
        self.pos_emb = SinusoidalPosEmb(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps [B] (values in [0, num_timesteps-1])
        Returns:
            Time embeddings [B, time_dim]
        """
        t_emb = self.pos_emb(t)
        return self.mlp(t_emb)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer for time conditioning.

    Applies affine transformation based on time embedding:
    y = gamma * x + beta
    where gamma, beta are derived from time embedding.
    """

    def __init__(self, num_channels: int, time_dim: int):
        super().__init__()
        self.gamma = nn.Linear(time_dim, num_channels)
        self.beta = nn.Linear(time_dim, num_channels)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]
            time_emb: Time embedding [B, time_dim]
        Returns:
            Modulated features [B, C, H, W]
        """
        gamma = self.gamma(time_emb)[:, :, None, None]
        beta = self.beta(time_emb)[:, :, None, None]
        return gamma * x + beta


class ConditionalResidualDenseBlock(nn.Module):
    """
    Residual Dense Block with time conditioning via FiLM.

    Each layer receives the feature maps of all preceding layers,
    plus time embedding for modulation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int = 128,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = _group_norm(out_channels)
        self.film1 = FiLMLayer(out_channels, time_dim)

        self.conv2 = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = _group_norm(out_channels)
        self.film2 = FiLMLayer(out_channels, time_dim)

        self.conv3 = nn.Conv2d(in_channels + 2 * out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = _group_norm(out_channels)
        self.film3 = FiLMLayer(out_channels, time_dim)

        if in_channels != out_channels:
            self.input_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.input_proj = None

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        identity = self.input_proj(x) if self.input_proj is not None else x

        c1 = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        c1 = self.film1(c1, time_emb)
        concat1 = torch.cat([x, c1], dim=1)

        c2 = F.leaky_relu(self.bn2(self.conv2(concat1)), negative_slope=0.2)
        c2 = self.film2(c2, time_emb)
        concat2 = torch.cat([x, c1, c2], dim=1)

        c3 = self.bn3(self.conv3(concat2))
        c3 = self.film3(c3, time_emb)

        return identity + c3


class ConditionalAttentionGate(nn.Module):
    """
    Attention gate for conditional U-Net.

    Combines skip connection features with decoder features,
    gated by attention mechanism.
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


class ConditionalUNetDenoiser(nn.Module):
    """
    Conditional U-Net Denoiser for Diffusion Refinement.

    This model takes:
    - Noisy input x_t at diffusion timestep t
    - Conditioning information c = [f_base, signal, signal_log, pos_ch]

    And predicts the noise epsilon that was added to the clean data.

    Architecture:
    - Time embedding via sinusoidal + MLP
    - FiLM conditioning at each layer
    - Multi-scale conditioning from f_base (concatenated at each resolution)
    - Encoder-decoder with attention gates

    Input:
        x_t: [B, 1, 64, 64] - noisy spectrum at timestep t
        condition: [B, 5, 64, 64] - [f_base, signal, signal_log, pos_ch, ...]
        t: [B] - timesteps

    Output:
        noise_pred: [B, 1, 64, 64] - predicted noise
    """

    def __init__(
        self,
        base_filters: int = 32,
        time_dim: int = 128,
        in_channels: int = 1,
        cond_channels: int = 5,
    ):
        super().__init__()
        self.base_filters = base_filters
        self.time_dim = time_dim
        self.in_channels = in_channels
        self.cond_channels = cond_channels

        self.time_emb = TimeEmbedding(time_dim)

        self.input_conv = nn.Conv2d(
            in_channels + cond_channels, base_filters, kernel_size=3, padding=1
        )
        self.input_norm = nn.InstanceNorm2d(base_filters, affine=False)

        channels = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8]

        self.enc1 = ConditionalResidualDenseBlock(
            channels[0], channels[0], time_dim
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.proj1 = nn.Conv2d(channels[0], channels[1], kernel_size=1)

        self.enc2 = ConditionalResidualDenseBlock(
            channels[1], channels[1], time_dim
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.proj2 = nn.Conv2d(channels[1], channels[2], kernel_size=1)

        self.enc3 = ConditionalResidualDenseBlock(
            channels[2], channels[2], time_dim
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.proj3 = nn.Conv2d(channels[2], channels[3], kernel_size=1)

        self.bottleneck = ConditionalResidualDenseBlock(
            channels[3], channels[3], time_dim
        )

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att3 = ConditionalAttentionGate(channels[2], channels[3], channels[2])
        self.dec3 = ConditionalResidualDenseBlock(
            channels[2] + channels[3] + cond_channels, channels[2], time_dim
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att2 = ConditionalAttentionGate(channels[1], channels[2], channels[1])
        self.dec2 = ConditionalResidualDenseBlock(
            channels[1] + channels[2] + cond_channels, channels[1], time_dim
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att1 = ConditionalAttentionGate(channels[0], channels[1], channels[0])
        self.dec1 = ConditionalResidualDenseBlock(
            channels[0] + channels[1] + cond_channels, channels[0], time_dim
        )

        self.output_conv = nn.Conv2d(channels[0], in_channels, kernel_size=1, padding=0)

    def forward(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the conditional denoiser.

        Args:
            x_t: Noisy input at timestep t [B, 1, H, W]
            condition: Conditioning [B, 5, H, W] (f_base, signal, signal_log, pos_ch)
            t: Timesteps [B]

        Returns:
            Predicted noise [B, 1, H, W]
        """
        time_emb = self.time_emb(t)

        x = torch.cat([x_t, condition], dim=1)
        x = self.input_conv(x)
        x = self.input_norm(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        e1 = self.enc1(x, time_emb)
        p1 = self.pool1(e1)
        p1 = self.proj1(p1)

        e2 = self.enc2(p1, time_emb)
        p2 = self.pool2(e2)
        p2 = self.proj2(p2)

        e3 = self.enc3(p2, time_emb)
        p3 = self.pool3(e3)
        p3 = self.proj3(p3)

        b = self.bottleneck(p3, time_emb)

        d3 = self.up3(b)
        a3 = self.att3(e3, d3)
        cond3 = F.interpolate(condition, size=a3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([a3, d3, cond3], dim=1), time_emb)

        d2 = self.up2(d3)
        a2 = self.att2(e2, d2)
        cond2 = F.interpolate(condition, size=a2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([a2, d2, cond2], dim=1), time_emb)

        d1 = self.up1(d2)
        a1 = self.att1(e1, d1)
        cond1 = F.interpolate(condition, size=a1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([a1, d1, cond1], dim=1), time_emb)

        noise_pred = self.output_conv(d1)

        return noise_pred


class RefinerWithBaseline(nn.Module):
    """
    Wrapper that combines baseline UNet with diffusion refiner.

    This module is used during training to:
    1. Generate baseline predictions from frozen UNet
    2. Pass baseline to diffusion refiner for refinement

    During inference, only the refiner is used with pre-computed baselines.
    """

    def __init__(
        self,
        baseline_unet: nn.Module,
        refiner: ConditionalUNetDenoiser,
        freeze_baseline: bool = True,
    ):
        super().__init__()
        self.baseline_unet = baseline_unet
        self.refiner = refiner
        self.freeze_baseline = freeze_baseline

        if freeze_baseline:
            for param in self.baseline_unet.parameters():
                param.requires_grad = False

    def forward(
        self,
        x_t: torch.Tensor,
        model_input: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate baseline and refine it.

        Args:
            x_t: Noisy input at timestep t [B, 1, H, W]
            model_input: 3-channel model input [B, 3, H, W]
            t: Timesteps [B]

        Returns:
            Predicted noise [B, 1, H, W]
        """
        with torch.no_grad() if self.freeze_baseline else torch.enable_grad():
            f_base = self.baseline_unet(model_input)

        condition = self._build_condition(f_base, model_input)

        noise_pred = self.refiner(x_t, condition, t)

        return noise_pred

    def _build_condition(
        self,
        f_base: torch.Tensor,
        model_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build conditioning tensor from baseline and model input.

        Conditioning channels:
        1. f_base (1 channel)
        2. signal_raw (1 channel)
        3. signal_log (1 channel)
        4. pos_channel (1 channel)
        5. extra channel (repetition of f_base or noise level)

        Total: 5 channels
        """
        signal_raw = model_input[:, 0:1, :, :]
        signal_log = model_input[:, 1:2, :, :]
        pos_ch = model_input[:, 2:3, :, :]

        condition = torch.cat([f_base, signal_raw, signal_log, pos_ch, f_base], dim=1)

        return condition


def get_denoiser(
    base_filters: int = 32,
    time_dim: int = 128,
    in_channels: int = 1,
    cond_channels: int = 5,
) -> ConditionalUNetDenoiser:
    """
    Factory function to create the conditional denoiser.

    Args:
        base_filters: Base number of filters
        time_dim: Dimension of time embedding
        in_channels: Input channels (1 for spectrum)
        cond_channels: Conditioning channels (5 for baseline + signal info)

    Returns:
        ConditionalUNetDenoiser model
    """
    return ConditionalUNetDenoiser(
        base_filters=base_filters,
        time_dim=time_dim,
        in_channels=in_channels,
        cond_channels=cond_channels,
    )


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = ConditionalUNetDenoiser(base_filters=32, time_dim=128).to(device)
    print(f"\nConditionalUNetDenoiser for Diffusion Refinement:")
    print(f"  Input: x_t [B, 1, 64, 64], condition [B, 5, 64, 64], t [B]")

    batch_size = 2
    x_t = torch.randn(batch_size, 1, 64, 64).to(device)
    condition = torch.randn(batch_size, 5, 64, 64).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)

    with torch.no_grad():
        noise_pred = model(x_t, condition, t)
    print(f"  Output shape: {noise_pred.shape}")
    print(f"  Output range: [{noise_pred.min().item():.3f}, {noise_pred.max().item():.3f}]")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

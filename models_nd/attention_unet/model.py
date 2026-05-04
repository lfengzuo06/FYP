"""
Attention U-Net model for N-Compartment DEXSY Inversion.

This module provides the Attention U-Net architecture adapted for N-compartment data,
with physics-informed training support and unified training across multiple N values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class EncoderDecoder(nn.Module):
    """
    Shared encoder-decoder backbone for unified N-compartment model.

    Takes feature extraction from multi-scale encoder and reconstructs
    the spectrum at the output resolution.
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters

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

        # Feature extraction at multiple scales for classification
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> tuple:
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

        # Return decoder features for spectrum prediction and encoder features for classification
        return d1, e1, e2, e3, b


class NClassificationHead(nn.Module):
    """
    Classification head to predict the number of compartments.

    Uses multi-scale features from the encoder to classify N.
    """

    def __init__(self, base_filters: int = 32, n_classes: int = 10, n_min: int = 2):
        super().__init__()
        self.n_classes = n_classes
        self.n_min = n_min

        # Process each encoder scale
        self.proc1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proc2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proc3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proc_bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(base_filters * 8, base_filters * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )

        # Fusion layer
        total_features = base_filters * 2 + base_filters * 2 + base_filters * 4 + base_filters * 4
        self.fusion = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_features, base_filters * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(base_filters * 8, n_classes),
        )

    def forward(self, e1: torch.Tensor, e2: torch.Tensor, e3: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        f1 = self.proc1(e1).flatten(1)
        f2 = self.proc2(e2).flatten(1)
        f3 = self.proc3(e3).flatten(1)
        fb = self.proc_bottleneck(b).flatten(1)

        fused = torch.cat([f1, f2, f3, fb], dim=1)
        logits = self.fusion(fused)

        # Shift logits so class 0 corresponds to n_min
        # Final prediction is argmax(logits) + n_min
        return logits


class SpectrumDecoder(nn.Module):
    """
    Decoder head for spectrum prediction.

    Takes decoder features and produces the normalized spectrum.
    """

    def __init__(self, base_filters: int = 32):
        super().__init__()
        self.output_activation = nn.Softplus()
        self.output_conv = nn.Conv2d(base_filters, 1, kernel_size=1, padding=0)

    def _normalize_distribution(self, x: torch.Tensor) -> torch.Tensor:
        x = self.output_activation(x)
        return x / (x.sum(dim=(2, 3), keepdim=True) + 1e-8)

    def forward(self, decoder_features: torch.Tensor) -> torch.Tensor:
        out = self._normalize_distribution(self.output_conv(decoder_features))
        return out


class AttentionUNetUnified(nn.Module):
    """
    Unified Attention U-Net for N-Compartment DEXSY Inversion.

    This model handles mixed N training by:
    1. Sharing encoder/decoder weights across all N values
    2. Predicting the number of compartments via a classification head
    3. Outputting a normalized spectrum (works for any N)

    Architecture:
    - Encoder-Decoder: Shared backbone for feature extraction
    - Classification Head: Predicts N from multi-scale encoder features
    - Spectrum Decoder: Produces normalized distribution

    Input: (batch, in_channels, 64, 64)
    Output:
        - spectrum: (batch, 1, 64, 64) normalized distribution
        - n_logits: (batch, n_classes) for N classification
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int = 32,
        n_min: int = 2,
        n_max: int = 10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.n_min = n_min
        self.n_max = n_max
        self.n_classes = n_max - n_min + 1

        self.encoder_decoder = EncoderDecoder(in_channels, base_filters)
        self.n_classifier = NClassificationHead(base_filters, self.n_classes, n_min)
        self.spectrum_decoder = SpectrumDecoder(base_filters)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, in_channels, 64, 64)

        Returns:
            spectrum: Normalized spectrum (batch, 1, 64, 64)
            n_logits: N classification logits (batch, n_classes)
        """
        d1, e1, e2, e3, b = self.encoder_decoder(x)
        spectrum = self.spectrum_decoder(d1)
        n_logits = self.n_classifier(e1, e2, e3, b)
        return spectrum, n_logits

    def predict_n(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the number of compartments."""
        _, n_logits = self.forward(x)
        n_pred = torch.argmax(n_logits, dim=1) + self.n_min
        return n_pred


class AttentionUNetND(nn.Module):
    """
    Attention U-Net for N-Compartment DEXSY Inversion (fixed N version).

    This is the original single-N model kept for compatibility.
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


class PhysicsInformedLossND(nn.Module):
    """
    Physics-informed loss for N-Compartment DEXSY training.

    The loss consists of:
    1. KL divergence for distribution matching
    2. Peak-weighted reconstruction loss
    3. Forward consistency loss in signal space
    4. Sum-to-one penalty
    5. Smoothness regularization
    6. N-compartment specific: compartment-aware regularization
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
        alpha_dei: float = 0.0,
    ):
        super().__init__()
        kernel = torch.from_numpy(forward_model.kernel_matrix).float()
        self.register_buffer("kernel_matrix", kernel)
        self.n_b = forward_model.n_b
        self.n_d = forward_model.n_d
        self.alpha_kl = alpha_kl
        self.alpha_rec = alpha_rec
        self.alpha_signal = alpha_signal
        self.alpha_sum = alpha_sum
        self.peak_weight = peak_weight
        self.alpha_smooth = alpha_smooth
        self.alpha_dei = alpha_dei

    def reconstruct_signal(self, y_pred: torch.Tensor) -> torch.Tensor:
        batch_size = y_pred.shape[0]
        pred_flat = y_pred.squeeze(1).reshape(batch_size, -1)
        signal_flat = pred_flat @ self.kernel_matrix.T
        return signal_flat.view(batch_size, 1, self.n_b, self.n_b)

    def _compute_dei_loss(self, y_pred: torch.Tensor) -> torch.Tensor:
        batch_size = y_pred.shape[0]
        total_loss = torch.zeros(batch_size, device=y_pred.device)

        for b in range(batch_size):
            spectrum = y_pred[b, 0]
            n_d = spectrum.shape[0]

            i_idx, j_idx = torch.meshgrid(
                torch.arange(n_d, device=y_pred.device),
                torch.arange(n_d, device=y_pred.device),
                indexing='ij'
            )
            diag_mask = torch.abs(i_idx - j_idx) <= 5
            off_diag_mask = ~diag_mask

            diag_sum = spectrum[diag_mask].sum()
            off_diag_sum = spectrum[off_diag_mask].sum()

            dei = off_diag_sum / (diag_sum + off_diag_sum + 1e-8)
            total_loss[b] = torch.relu(0.1 - dei)

        return total_loss.mean()

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

        dei_loss = self._compute_dei_loss(y_pred) if self.alpha_dei > 0 else torch.tensor(0.0, device=y_pred.device)

        total_loss = (
            self.alpha_kl * kl_loss +
            self.alpha_rec * rec_loss +
            self.alpha_signal * signal_loss +
            self.alpha_sum * sum_penalty +
            self.alpha_smooth * smoothness +
            self.alpha_dei * dei_loss
        )

        return total_loss

    def get_loss_components(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        signal_targets: torch.Tensor,
    ) -> dict:
        with torch.no_grad():
            y_pred_norm = y_pred / (y_pred.sum(dim=(2, 3), keepdim=True) + 1e-8)
            y_true_norm = y_true / (y_true.sum(dim=(2, 3), keepdim=True) + 1e-8)

            relative_peak_map = y_true_norm / (y_true_norm.amax(dim=(2, 3), keepdim=True) + 1e-8)
            weights = 1.0 + self.peak_weight * torch.sqrt(relative_peak_map + 1e-8)
            rec_loss = torch.mean(weights * (y_pred_norm - y_true_norm) ** 2).item()

            kl_loss = F.kl_div(torch.log(y_pred_norm + 1e-8), y_true_norm, reduction="batchmean").item()

            pred_signals = self.reconstruct_signal(y_pred_norm)
            signal_loss = F.mse_loss(pred_signals, signal_targets).item()

            total_mass = y_pred_norm.sum(dim=(2, 3), keepdim=True)
            sum_penalty = torch.mean((total_mass - 1.0) ** 2).item()

            smooth_x = torch.mean(torch.abs(y_pred_norm[:, :, 1:, :] - y_pred_norm[:, :, :-1, :])).item()
            smooth_y = torch.mean(torch.abs(y_pred_norm[:, :, :, 1:] - y_pred_norm[:, :, :, :-1])).item()
            smoothness = smooth_x + smooth_y

            dei_loss = self._compute_dei_loss(y_pred_norm).item() if self.alpha_dei > 0 else 0.0

            return {
                'kl_loss': kl_loss,
                'rec_loss': rec_loss,
                'signal_loss': signal_loss,
                'sum_penalty': sum_penalty,
                'smoothness': smoothness,
                'dei_loss': dei_loss,
                'total_loss': (
                    self.alpha_kl * kl_loss +
                    self.alpha_rec * rec_loss +
                    self.alpha_signal * signal_loss +
                    self.alpha_sum * sum_penalty +
                    self.alpha_smooth * smoothness +
                    self.alpha_dei * dei_loss
                )
            }


class UnifiedLoss(nn.Module):
    """
    Combined loss for unified N-compartment training.

    Combines:
    1. Physics-informed loss for spectrum prediction
    2. Classification loss for N prediction
    """

    def __init__(
        self,
        forward_model,
        n_min: int = 2,
        n_max: int = 10,
        alpha_kl: float = 1.0,
        alpha_rec: float = 0.2,
        alpha_signal: float = 0.1,
        alpha_sum: float = 0.05,
        peak_weight: float = 6.0,
        alpha_smooth: float = 2e-2,
        alpha_n_class: float = 0.5,
        alpha_dei: float = 0.0,
    ):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.n_classes = n_max - n_min + 1
        self.alpha_n_class = alpha_n_class

        self.physics_loss = PhysicsInformedLossND(
            forward_model=forward_model,
            alpha_kl=alpha_kl,
            alpha_rec=alpha_rec,
            alpha_signal=alpha_signal,
            alpha_sum=alpha_sum,
            peak_weight=peak_weight,
            alpha_smooth=alpha_smooth,
            alpha_dei=alpha_dei,
        )
        self.class_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        spectrum_pred: torch.Tensor,
        n_pred: torch.Tensor,
        spectrum_true: torch.Tensor,
        n_true: torch.Tensor,
        signal_targets: torch.Tensor,
    ) -> tuple:
        """
        Compute combined loss.

        Args:
            spectrum_pred: Predicted spectrum (batch, 1, 64, 64)
            n_pred: Predicted N logits (batch, n_classes)
            spectrum_true: Ground truth spectrum (batch, 1, 64, 64)
            n_true: Ground truth N values (batch,) - actual N values
            signal_targets: Target signals (batch, 1, 64, 64)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Physics loss
        physics_loss = self.physics_loss(spectrum_pred, spectrum_true, signal_targets)

        # Classification loss (convert N to class indices)
        n_class_indices = n_true - self.n_min  # Convert N -> class index
        class_loss = self.class_criterion(n_pred, n_class_indices)

        # Combined loss
        total_loss = physics_loss + self.alpha_n_class * class_loss

        loss_dict = {
            'physics_loss': physics_loss.item(),
            'class_loss': class_loss.item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_dict

    def get_n_predictions(self, n_pred: torch.Tensor) -> torch.Tensor:
        """Convert logits to N predictions."""
        n_class = torch.argmax(n_pred, dim=1)
        return n_class + self.n_min


def get_model(base_filters: int = 32, in_channels: int = 3) -> nn.Module:
    """Factory function for single-N model."""
    return AttentionUNetND(in_channels=in_channels, base_filters=base_filters)


def get_unified_model(
    base_filters: int = 32,
    in_channels: int = 3,
    n_min: int = 2,
    n_max: int = 10,
) -> nn.Module:
    """Factory function for unified N model."""
    return AttentionUNetUnified(
        in_channels=in_channels,
        base_filters=base_filters,
        n_min=n_min,
        n_max=n_max,
    )


def get_loss(
    forward_model,
    alpha_kl: float = 1.0,
    alpha_rec: float = 0.2,
    alpha_signal: float = 0.1,
    alpha_sum: float = 0.05,
    peak_weight: float = 6.0,
    alpha_smooth: float = 2e-2,
    alpha_dei: float = 0.0,
) -> nn.Module:
    """Factory function for physics-informed loss."""
    return PhysicsInformedLossND(
        forward_model=forward_model,
        alpha_kl=alpha_kl,
        alpha_rec=alpha_rec,
        alpha_signal=alpha_signal,
        alpha_sum=alpha_sum,
        peak_weight=peak_weight,
        alpha_smooth=alpha_smooth,
        alpha_dei=alpha_dei,
    )


def get_unified_loss(
    forward_model,
    n_min: int = 2,
    n_max: int = 10,
    alpha_kl: float = 1.0,
    alpha_rec: float = 0.2,
    alpha_signal: float = 0.1,
    alpha_sum: float = 0.05,
    peak_weight: float = 6.0,
    alpha_smooth: float = 2e-2,
    alpha_n_class: float = 0.5,
    alpha_dei: float = 0.0,
) -> nn.Module:
    """Factory function for unified loss with classification."""
    return UnifiedLoss(
        forward_model=forward_model,
        n_min=n_min,
        n_max=n_max,
        alpha_kl=alpha_kl,
        alpha_rec=alpha_rec,
        alpha_signal=alpha_signal,
        alpha_sum=alpha_sum,
        peak_weight=peak_weight,
        alpha_smooth=alpha_smooth,
        alpha_n_class=alpha_n_class,
        alpha_dei=alpha_dei,
    )


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test unified model
    print("\n=== Testing Unified Model ===")
    model = AttentionUNetUnified(in_channels=3, base_filters=32, n_min=2, n_max=7).to(device)
    print(f"Model: AttentionUNetUnified (N={model.n_min}-{model.n_max})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(4, 3, 64, 64).to(device)
    with torch.no_grad():
        spectrum, n_logits = model(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Spectrum shape: {spectrum.shape}")
    print(f"  N logits shape: {n_logits.shape}")
    print(f"  Spectrum sum: {spectrum.sum(dim=(2, 3)).mean().item():.4f} (should be ~1.0)")

    # Test N prediction
    n_pred = torch.argmax(n_logits, dim=1) + model.n_min
    print(f"  N predictions: {n_pred.tolist()}")

    # Test single-N model
    print("\n=== Testing Single-N Model ===")
    model_single = AttentionUNetND(in_channels=3, base_filters=32).to(device)
    print(f"Model: AttentionUNetND")
    print(f"  Parameters: {sum(p.numel() for p in model_single.parameters()):,}")

    with torch.no_grad():
        out_single = model_single(x)

    print(f"  Output shape: {out_single.shape}")
    print(f"  Output sum: {out_single.sum(dim=(2, 3)).mean().item():.4f} (should be ~1.0)")

    print("\nAll tests passed!")

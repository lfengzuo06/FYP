"""
CNN model definition for 3C non-Gaussian inverse DEXSY.

Task:
- Input: 16x16 (or n_b x n_b) DEXSY signal matrix
- Output: 9 pathway weights over EE, ET, ES, TE, TT, TS, SE, ST, SS
- Constraint: softmax output so sum(W)=1
- DEI is derived from W
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PATHWAY_ORDER_3C = ("EE", "ET", "ES", "TE", "TT", "TS", "SE", "ST", "SS")
DIAGONAL_PATHWAYS_3C = ("EE", "TT", "SS")


def _pathway_index(pathway: str) -> int:
    return PATHWAY_ORDER_3C.index(pathway)


DIAGONAL_INDICES_3C = [_pathway_index(p) for p in DIAGONAL_PATHWAYS_3C]
OFF_DIAGONAL_INDICES_3C = [
    i for i in range(len(PATHWAY_ORDER_3C)) if i not in DIAGONAL_INDICES_3C
]


@dataclass
class NonGaussian3CPrediction:
    """Structured model output."""

    logits: torch.Tensor
    pathway_weights: torch.Tensor
    pathway_weight_matrix: torch.Tensor
    dei: torch.Tensor


def reshape_pathway_vector_to_matrix(pathway_weights: torch.Tensor) -> torch.Tensor:
    """Convert (...,9) pathway vector to (...,3,3) matrix in E/T/S order."""
    if pathway_weights.shape[-1] != 9:
        raise ValueError(
            f"Expected last dimension=9 for pathway vector, got {pathway_weights.shape}."
        )
    new_shape = pathway_weights.shape[:-1] + (3, 3)
    return pathway_weights.reshape(new_shape)


def flatten_pathway_matrix_to_vector(weight_matrix: torch.Tensor) -> torch.Tensor:
    """Convert (...,3,3) matrix to (...,9) vector."""
    if weight_matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected trailing shape (3,3), got {weight_matrix.shape}."
        )
    new_shape = weight_matrix.shape[:-2] + (9,)
    return weight_matrix.reshape(new_shape)


def compute_dei_from_pathway_weights(
    pathway_weights: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute DEI from pathway vector (...,9):
        DEI = sum(off-diagonal) / sum(diagonal)
    """
    if pathway_weights.shape[-1] != 9:
        raise ValueError(
            f"Expected last dimension=9 for pathway weights, got {pathway_weights.shape}."
        )

    diag = pathway_weights[..., DIAGONAL_INDICES_3C].sum(dim=-1)
    off = pathway_weights[..., OFF_DIAGONAL_INDICES_3C].sum(dim=-1)
    return off / (diag + eps)


def compute_dei_from_weight_matrix(
    weight_matrix: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute DEI from matrix form (...,3,3)."""
    return compute_dei_from_pathway_weights(
        flatten_pathway_matrix_to_vector(weight_matrix),
        eps=eps,
    )


class _ResidualConvBlock(nn.Module):
    """Compact residual block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=max(1, min(8, out_ch)), num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=max(1, min(8, out_ch)), num_channels=out_ch)

        if in_ch != out_ch or stride != 1:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + identity
        return F.gelu(out)


class NonGaussian3CInverseNet(nn.Module):
    """
    CNN + MLP inverse model for non-Gaussian 3C signals.

    Input:
        - (B,H,W), or
        - (B,C,H,W) where channel 0 is raw signal.

    Output:
        NonGaussian3CPrediction with W_hat and DEI_hat.
    """

    def __init__(
        self,
        base_channels: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.15,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.eps = float(eps)

        in_ch = 4  # raw, log(raw), symmetric, antisymmetric

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=max(1, min(8, base_channels)), num_channels=base_channels),
            nn.GELU(),
        )

        self.encoder = nn.Sequential(
            _ResidualConvBlock(base_channels, base_channels, stride=1),
            _ResidualConvBlock(base_channels, base_channels * 2, stride=2),  # 16->8
            _ResidualConvBlock(base_channels * 2, base_channels * 2, stride=1),
            _ResidualConvBlock(base_channels * 2, base_channels * 4, stride=2),  # 8->4
            _ResidualConvBlock(base_channels * 4, base_channels * 4, stride=1),
        )

        pooled_dim = base_channels * 4 * 2  # avg + max
        summary_dim = 8
        fused_dim = pooled_dim + summary_dim

        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 9),
        )

    def _extract_raw_signal(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            raw = x.unsqueeze(1)
        elif x.ndim == 4:
            raw = x[:, :1, :, :]
        else:
            raise ValueError(
                f"Expected (B,H,W) or (B,C,H,W), got {tuple(x.shape)}"
            )
        return raw

    def _engineer_channels(self, raw: torch.Tensor) -> torch.Tensor:
        safe = raw.clamp_min(1e-8)
        log_raw = torch.log(safe)
        sym = 0.5 * (raw + raw.transpose(-1, -2))
        asym = 0.5 * (raw - raw.transpose(-1, -2))

        channels = torch.cat([raw, log_raw, sym, asym], dim=1)

        mean = channels.mean(dim=(-2, -1), keepdim=True)
        std = channels.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (channels - mean) / std

    def _summary_features(self, raw: torch.Tensor) -> torch.Tensor:
        s = raw[:, 0]  # (B,H,W)
        b, h, w = s.shape

        diag = torch.diagonal(s, dim1=-2, dim2=-1)
        anti_diag = torch.diagonal(torch.flip(s, dims=[-1]), dim1=-2, dim2=-1)

        eye = torch.eye(h, device=s.device, dtype=torch.bool).unsqueeze(0)
        off_vals = s.masked_select(~eye).reshape(b, h * w - h)

        upper = torch.triu(s, diagonal=1)
        lower = torch.tril(s, diagonal=-1)

        return torch.stack(
            [
                s[:, 0, 0],
                s.mean(dim=(-2, -1)),
                s.std(dim=(-2, -1)),
                diag.mean(dim=-1),
                anti_diag.mean(dim=-1),
                off_vals.mean(dim=-1),
                upper.sum(dim=(-2, -1)) / max(1, h * (h - 1) // 2),
                lower.sum(dim=(-2, -1)) / max(1, h * (h - 1) // 2),
            ],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> NonGaussian3CPrediction:
        raw = self._extract_raw_signal(x)
        engineered = self._engineer_channels(raw)

        feat = self.stem(engineered)
        feat = self.encoder(feat)

        avg_pool = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
        max_pool = F.adaptive_max_pool2d(feat, output_size=1).flatten(1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        summary = self._summary_features(raw)
        fused = torch.cat([pooled, summary], dim=1)

        logits = self.head(fused)
        pathway_weights = torch.softmax(logits, dim=-1)
        pathway_weight_matrix = reshape_pathway_vector_to_matrix(pathway_weights)
        dei = compute_dei_from_pathway_weights(pathway_weights, eps=self.eps)

        return NonGaussian3CPrediction(
            logits=logits,
            pathway_weights=pathway_weights,
            pathway_weight_matrix=pathway_weight_matrix,
            dei=dei,
        )


class NonGaussian3CLoss(nn.Module):
    """
    Loss:
        L = MSE(W, W_hat) + lambda_dei * MSE(DEI, DEI_hat)
    """

    def __init__(self, lambda_dei: float = 1.0, eps: float = 1e-12):
        super().__init__()
        self.lambda_dei = float(lambda_dei)
        self.eps = float(eps)

    def forward(
        self,
        pred: NonGaussian3CPrediction,
        target_pathway_weights: torch.Tensor,
        target_dei: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if target_pathway_weights.shape[-1] != 9:
            raise ValueError(
                f"Expected target_pathway_weights last dim=9, got {target_pathway_weights.shape}"
            )

        w_loss = F.mse_loss(pred.pathway_weights, target_pathway_weights)

        if target_dei is None:
            target_dei = compute_dei_from_pathway_weights(target_pathway_weights, eps=self.eps)

        dei_loss = F.mse_loss(pred.dei, target_dei)
        total = w_loss + self.lambda_dei * dei_loss

        metrics = {
            "loss_total": total.detach(),
            "loss_w": w_loss.detach(),
            "loss_dei": dei_loss.detach(),
        }
        return total, metrics


__all__ = [
    "PATHWAY_ORDER_3C",
    "DIAGONAL_PATHWAYS_3C",
    "NonGaussian3CPrediction",
    "reshape_pathway_vector_to_matrix",
    "flatten_pathway_matrix_to_vector",
    "compute_dei_from_pathway_weights",
    "compute_dei_from_weight_matrix",
    "NonGaussian3CInverseNet",
    "NonGaussian3CLoss",
]

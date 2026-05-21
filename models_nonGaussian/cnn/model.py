"""
Model definition for 3C non-Gaussian inverse DEXSY.

Task:
- Input: 16x16 (or n_b x n_b) DEXSY signal matrix
- Output: 9 pathway weights over EE, ET, ES, TE, TT, TS, SE, ST, SS
- Constraint: softmax output so sum(W)=1
- DEI is derived from W
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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


class _TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden = max(embed_dim, int(embed_dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


def _canonicalize_architecture(name: str) -> str:
    n = str(name).strip().lower()
    if n in {"rescnn", "cnn", "legacy_cnn"}:
        return "rescnn"
    if n in {"hybrid_transformer", "signal_transformer", "transformer", "vit"}:
        return "hybrid_transformer"
    raise ValueError(
        f"Unknown architecture '{name}'. Use one of: "
        f"rescnn, hybrid_transformer."
    )


def _resolve_num_heads(embed_dim: int, requested_heads: int) -> int:
    req = max(1, int(requested_heads))
    if embed_dim % req == 0:
        return req
    for h in range(req, 0, -1):
        if embed_dim % h == 0:
            return h
    return 1


def infer_architecture_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> str:
    """
    Infer architecture name from checkpoint keys.

    This keeps legacy checkpoints loadable when explicit config is missing.
    """
    keys = list(state_dict.keys())
    if any(
        k.startswith("token_embed")
        or k.startswith("transformer_blocks")
        or k.startswith("row_embed")
        or k.startswith("col_embed")
        for k in keys
    ):
        return "hybrid_transformer"
    if any(k.startswith("stem") or k.startswith("encoder") for k in keys):
        return "rescnn"
    # Default to new model for unknown key layouts.
    return "hybrid_transformer"


class NonGaussian3CInverseNet(nn.Module):
    """
    Inverse model for non-Gaussian 3C signals.

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
        architecture: str = "hybrid_transformer",
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        transformer_mlp_ratio: float = 3.0,
        max_grid_size: int = 64,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.eps = float(eps)
        self.architecture = _canonicalize_architecture(architecture)

        in_ch = 4  # raw, log(raw), symmetric, antisymmetric

        if self.architecture == "rescnn":
            # Legacy CNN path retained for checkpoint compatibility.
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
            fused_dim = pooled_dim + 8
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
        else:
            embed_dim = max(32, int(base_channels) * 4)
            self.max_grid_size = int(max_grid_size)
            n_heads = _resolve_num_heads(embed_dim, int(transformer_heads))
            summary_dim = max(32, embed_dim // 2)

            self.token_embed = nn.Sequential(
                nn.Conv2d(in_ch, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups=max(1, min(8, embed_dim)), num_channels=embed_dim),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0),
                nn.GELU(),
            )
            self.row_embed = nn.Parameter(torch.randn(self.max_grid_size, embed_dim) * 0.02)
            self.col_embed = nn.Parameter(torch.randn(self.max_grid_size, embed_dim) * 0.02)
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

            self.transformer_blocks = nn.ModuleList(
                [
                    _TransformerEncoderBlock(
                        embed_dim=embed_dim,
                        num_heads=n_heads,
                        mlp_ratio=float(transformer_mlp_ratio),
                        dropout=float(dropout),
                    )
                    for _ in range(max(1, int(transformer_depth)))
                ]
            )
            self.transformer_norm = nn.LayerNorm(embed_dim)

            self.summary_proj = nn.Sequential(
                nn.LayerNorm(8),
                nn.Linear(8, summary_dim),
                nn.GELU(),
            )
            fused_dim = embed_dim * 2 + summary_dim
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
        summary = self._summary_features(raw)

        if self.architecture == "rescnn":
            feat = self.stem(engineered)
            feat = self.encoder(feat)
            avg_pool = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
            max_pool = F.adaptive_max_pool2d(feat, output_size=1).flatten(1)
            pooled = torch.cat([avg_pool, max_pool], dim=1)
            fused = torch.cat([pooled, summary], dim=1)
            logits = self.head(fused)
        else:
            feat = self.token_embed(engineered)
            b, c, h, w = feat.shape
            if h > self.max_grid_size or w > self.max_grid_size:
                raise ValueError(
                    f"Input grid ({h},{w}) exceeds max_grid_size={self.max_grid_size} "
                    f"for hybrid_transformer architecture."
                )

            tokens = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
            pos = (self.row_embed[:h, None, :] + self.col_embed[None, :w, :]).reshape(1, h * w, c)
            tokens = tokens + pos
            cls = self.cls_token.expand(b, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

            for block in self.transformer_blocks:
                tokens = block(tokens)
            tokens = self.transformer_norm(tokens)

            cls_feat = tokens[:, 0, :]
            mean_feat = tokens[:, 1:, :].mean(dim=1)
            summary_feat = self.summary_proj(summary)
            fused = torch.cat([cls_feat, mean_feat, summary_feat], dim=1)
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

    def __init__(self, lambda_dei: float = 1.0, lambda_kl: float = 0.20, eps: float = 1e-12):
        super().__init__()
        self.lambda_dei = float(lambda_dei)
        self.lambda_kl = float(lambda_kl)
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

        w_mse = F.mse_loss(pred.pathway_weights, target_pathway_weights)
        if self.lambda_kl > 0.0:
            target_safe = target_pathway_weights.clamp_min(self.eps)
            pred_safe = pred.pathway_weights.clamp_min(self.eps)
            w_kl = torch.sum(target_safe * (torch.log(target_safe) - torch.log(pred_safe)), dim=-1).mean()
        else:
            w_kl = pred.pathway_weights.new_zeros(())

        w_loss = w_mse + self.lambda_kl * w_kl

        if target_dei is None:
            target_dei = compute_dei_from_pathway_weights(target_pathway_weights, eps=self.eps)

        dei_loss = F.mse_loss(pred.dei, target_dei)
        total = w_loss + self.lambda_dei * dei_loss

        metrics = {
            "loss_total": total.detach(),
            "loss_w": w_loss.detach(),
            "loss_w_mse": w_mse.detach(),
            "loss_w_kl": w_kl.detach(),
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
    "infer_architecture_from_state_dict",
    "NonGaussian3CInverseNet",
    "NonGaussian3CLoss",
]

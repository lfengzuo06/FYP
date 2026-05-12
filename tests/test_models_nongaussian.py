"""Tests for the non-Gaussian 3C inverse model module."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dexsy_core.forward_model_3c_nongaussian import ForwardModel3CNonGaussian
from models_nonGaussian import (
    NonGaussian3CInverseNet,
    NonGaussian3CLoss,
    compute_dei_from_pathway_weights,
    sample_3c_nongaussian_inverse_dataset,
)


def test_inverse_model_output_shapes_and_normalization():
    model = NonGaussian3CInverseNet(base_channels=16, hidden_dim=128).eval()

    x = torch.rand(4, 16, 16)
    pred = model(x)

    assert pred.logits.shape == (4, 9)
    assert pred.pathway_weights.shape == (4, 9)
    assert pred.pathway_weight_matrix.shape == (4, 3, 3)
    assert pred.dei.shape == (4,)

    sums = pred.pathway_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_dei_formula_matches_manual_computation():
    # pathway order: EE,ET,ES,TE,TT,TS,SE,ST,SS
    w = torch.tensor([[0.40, 0.10, 0.05, 0.07, 0.20, 0.06, 0.04, 0.03, 0.05]], dtype=torch.float32)
    dei = compute_dei_from_pathway_weights(w)

    diag = 0.40 + 0.20 + 0.05
    off = 0.10 + 0.05 + 0.07 + 0.06 + 0.04 + 0.03
    manual = off / diag

    assert torch.allclose(dei, torch.tensor([manual], dtype=torch.float32), atol=1e-7)


def test_loss_zero_when_prediction_matches_target():
    model = NonGaussian3CInverseNet(base_channels=16, hidden_dim=128).eval()
    criterion = NonGaussian3CLoss(lambda_dei=5.0)

    x = torch.rand(3, 16, 16)
    pred = model(x)

    target_w = pred.pathway_weights.detach().clone()
    target_dei = pred.dei.detach().clone()

    total, metrics = criterion(pred, target_w, target_dei)

    assert float(total.item()) < 1e-8
    assert float(metrics["loss_w"].item()) < 1e-8
    assert float(metrics["loss_dei"].item()) < 1e-8


def test_sampling_helper_shapes_and_constraints():
    fm = ForwardModel3CNonGaussian(n_b=16, n_restrict_terms=80)
    out = sample_3c_nongaussian_inverse_dataset(
        forward_model=fm,
        n_samples=3,
        seed=7,
    )

    signals = out["signals_noisy"]
    clean = out["signals_clean"]
    w = out["pathway_weights"]
    dei = out["dei"]

    assert signals.shape == (3, 16, 16)
    assert clean.shape == (3, 16, 16)
    assert w.shape == (3, 9)
    assert dei.shape == (3,)

    # Softmax-like probability simplex target from forward model
    np.testing.assert_allclose(w.sum(axis=1), np.ones(3, dtype=np.float32), atol=1e-6)

    # Normalization target: S(0,0)=1 for noisy and clean outputs
    np.testing.assert_allclose(signals[:, 0, 0], np.ones(3, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(clean[:, 0, 0], np.ones(3, dtype=np.float32), atol=1e-6)

    # DEI in output should match formula from sampled pathway weights
    dei_re = compute_dei_from_pathway_weights(torch.from_numpy(w)).numpy()
    np.testing.assert_allclose(dei, dei_re, rtol=1e-6, atol=1e-6)

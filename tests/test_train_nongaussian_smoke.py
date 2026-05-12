"""Smoke test for direct training entry of non-Gaussian inverse model."""

from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models_nonGaussian import train_nongaussian_inverse_model


def test_train_nongaussian_inverse_smoke(tmp_path: Path):
    model, history, datasets, forward_model, run_dir = train_nongaussian_inverse_model(
        output_dir=tmp_path,
        n_train=12,
        n_val=4,
        n_test=4,
        epochs=1,
        batch_size=4,
        n_restrict_terms=60,
        base_channels=8,
        hidden_dim=64,
        seed=11,
        device="cpu",
    )

    assert model is not None
    assert forward_model.n_b == 16
    assert len(history["train_loss"]) >= 1
    assert len(datasets["train"]["signals_noisy"]) == 12

    assert run_dir.exists()
    assert (run_dir / "best_model.pt").exists()
    assert (run_dir / "final_model.pt").exists()
    assert (run_dir / "history.json").exists()

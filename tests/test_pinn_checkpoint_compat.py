from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models_2d.pinn import PINN2D, PINNInferencePipeline  # noqa: E402
from models_2d.pinn.inference import load_trained_model  # noqa: E402


class PINNCheckpointCompatibilityTest(unittest.TestCase):
    def test_load_trained_model_accepts_current_architecture_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "pinn_good.pt"
            model = PINN2D(signal_size=64, in_channels=3, base_filters=8)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "signal_size": 64,
                        "in_channels": 3,
                        "base_filters": 8,
                        "architecture": "simple_encoder_decoder_v2",
                    },
                },
                checkpoint_path,
            )

            loaded_model, metadata = load_trained_model(checkpoint_path, device="cpu")
            self.assertIsNotNone(loaded_model)
            self.assertEqual(metadata["model_name"], "pinn")

    def test_pipeline_rejects_incompatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "pinn_bad.pt"
            torch.save(
                {
                    "model_state_dict": {
                        "signal_encoder.0.weight": torch.randn(32, 64 * 64),
                        "signal_encoder.0.bias": torch.randn(32),
                    },
                    "config": {
                        "architecture": "legacy_pinn_v1",
                    },
                },
                checkpoint_path,
            )

            with self.assertRaises(RuntimeError):
                PINNInferencePipeline(checkpoint_path=checkpoint_path, device="cpu")


if __name__ == "__main__":
    unittest.main()

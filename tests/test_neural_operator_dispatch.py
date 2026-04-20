from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from improved_2d_dexsy import DEXSYInferencePipeline  # noqa: E402
from models_2d.neural_operators.fno import FNO2D  # noqa: E402


class NeuralOperatorDispatchTest(unittest.TestCase):
    def test_fno_model_name_maps_to_fno_pipeline_type(self):
        pipeline = DEXSYInferencePipeline(
            model_name="fno",
            checkpoint_path=None,
            device="cpu",
        )
        self.assertEqual(pipeline.get_model_name(), "fno")
        self.assertEqual(pipeline.get_model_info()["model_type"], "fno")

    def test_fno_wrapper_uses_three_channel_inputs_for_prediction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "fno_test.pt"
            model = FNO2D(in_channels=3, hidden_channels=8, n_layers=1, modes=4)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "model_type": "fno",
                        "in_channels": 3,
                        "hidden_channels": 8,
                        "n_layers": 1,
                        "modes": 4,
                    },
                },
                checkpoint_path,
            )

            pipeline = DEXSYInferencePipeline(
                model_name="fno",
                checkpoint_path=checkpoint_path,
                device="cpu",
            )
            signal = np.random.rand(64, 64).astype(np.float32)
            result = pipeline.predict_from_signal(signal, include_figure=False)

            self.assertEqual(result.reconstructed_spectrum.shape, (64, 64))
            self.assertTrue(np.isfinite(result.dei))


if __name__ == "__main__":
    unittest.main()

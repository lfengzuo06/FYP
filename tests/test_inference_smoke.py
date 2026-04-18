from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from improved_2d_dexsy import ForwardModel2D, predict_from_signal  # noqa: E402


class InferenceSmokeTest(unittest.TestCase):
    def test_predict_from_signal_returns_normalized_spectrum(self):
        forward_model = ForwardModel2D(n_d=64, n_b=64)
        _, noisy_signal, _, _ = forward_model.generate_sample(
            n_compartments=2,
            return_reference_signal=True,
        )

        result = predict_from_signal(
            noisy_signal,
            forward_model=forward_model,
            include_figure=False,
        )

        self.assertEqual(result.reconstructed_spectrum.shape, (64, 64))
        self.assertAlmostEqual(result.summary_metrics["prediction_mass"], 1.0, places=5)
        self.assertIn("dei", result.summary_metrics)
        self.assertEqual(result.metadata["model_name"], "attention_unet")


if __name__ == "__main__":
    unittest.main()

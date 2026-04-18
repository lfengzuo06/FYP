from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from improved_2d_dexsy import ForwardModel2D, validate_signal_grid  # noqa: E402


class ShapeValidationTest(unittest.TestCase):
    def setUp(self):
        self.forward_model = ForwardModel2D(n_d=64, n_b=64)

    def test_accepts_single_64x64_signal(self):
        signal = np.ones((64, 64), dtype=np.float32)
        validated = validate_signal_grid(signal, self.forward_model)
        self.assertEqual(validated.shape, (1, 1, 64, 64))

    def test_accepts_batched_signal_tensor(self):
        signal = np.ones((3, 64, 64), dtype=np.float32)
        validated = validate_signal_grid(signal, self.forward_model)
        self.assertEqual(validated.shape, (3, 1, 64, 64))

    def test_rejects_wrong_grid_size(self):
        signal = np.ones((32, 32), dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_signal_grid(signal, self.forward_model)


if __name__ == "__main__":
    unittest.main()

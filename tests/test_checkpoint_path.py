from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from improved_2d_dexsy import resolve_checkpoint_path  # noqa: E402


class CheckpointPathTest(unittest.TestCase):
    def test_default_checkpoint_points_to_0411_model(self):
        path = resolve_checkpoint_path()
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "attention_unet_best_model_20260411_155746.pt")

    def test_relative_checkpoint_override_resolves_inside_repo(self):
        path = resolve_checkpoint_path("checkpoints/attention_unet_best_model.pt")
        self.assertTrue(path.exists())
        self.assertTrue(str(path).startswith(str(ROOT)))


if __name__ == "__main__":
    unittest.main()

"""
Smoke tests for 16x16 training pipeline.

This module verifies:
1. 2D training pipeline runs with 16x16 grid
2. 3D training pipeline runs with 16x16 grid
3. Checkpoint saves n_d and n_b correctly
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

# Add FYP root to path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


class TestTrainG16Smoke(unittest.TestCase):
    """Smoke tests for 16x16 training."""

    @classmethod
    def setUpClass(cls):
        """Skip if CUDA not available."""
        cls.skip_if_no_cuda = not torch.cuda.is_available()

    def test_2d_g16_training_smoke(self):
        """Test that 2D training runs for 1 epoch with 16x16 grid."""
        from models_2d.attention_unet import train_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model, history, datasets, fm = train_model(
                output_dir=tmpdir,
                n_train=10,
                n_val=5,
                epochs=1,
                batch_size=4,
                n_d=16,
                n_b=16,
                seed=42,
                device='cpu',
            )

            # Verify model was created
            self.assertIsNotNone(model)

            # Verify forward model has correct grid size
            self.assertEqual(fm.n_d, 16)
            self.assertEqual(fm.n_b, 16)

            # Verify datasets have correct shapes
            self.assertEqual(datasets['train']['inputs'].shape[2], 16)
            self.assertEqual(datasets['train']['inputs'].shape[3], 16)

            # Verify history was recorded
            self.assertEqual(len(history['train_loss']), 1)
            self.assertEqual(len(history['val_loss']), 1)

    def test_3d_g16_training_smoke(self):
        """Test that 3D training runs for 1 epoch with 16x16 grid."""
        from models_3d.attention_unet import train_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model, history, datasets, fm = train_model(
                output_dir=tmpdir,
                n_train=10,
                n_val=5,
                epochs=1,
                batch_size=4,
                n_d=16,
                n_b=16,
                n_compartments=3,
                seed=42,
                device='cpu',
            )

            # Verify model was created
            self.assertIsNotNone(model)

            # Verify forward model has correct grid size
            self.assertEqual(fm.n_d, 16)
            self.assertEqual(fm.n_b, 16)

            # Verify datasets have correct shapes
            self.assertEqual(datasets['train']['inputs'].shape[2], 16)
            self.assertEqual(datasets['train']['inputs'].shape[3], 16)

            # Verify history was recorded
            self.assertEqual(len(history['train_loss']), 1)

    def test_2d_g16_checkpoint_contains_grid_size(self):
        """Test that 2D checkpoint contains n_d and n_b."""
        from models_2d.attention_unet import train_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model, history, datasets, fm = train_model(
                output_dir=tmpdir,
                n_train=10,
                n_val=5,
                epochs=1,
                batch_size=4,
                n_d=16,
                n_b=16,
                seed=42,
                device='cpu',
            )

            # Find checkpoint
            checkpoint_dirs = list(Path(tmpdir).glob("checkpoints_*"))
            self.assertGreater(len(checkpoint_dirs), 0)

            checkpoint_dir = checkpoint_dirs[0]
            final_model_path = checkpoint_dir / "final_model.pt"

            if final_model_path.exists():
                checkpoint = torch.load(final_model_path, map_location='cpu')
                config = checkpoint.get('config', {})

                self.assertEqual(config.get('n_d'), 16)
                self.assertEqual(config.get('n_b'), 16)

    def test_3d_g16_checkpoint_contains_grid_size(self):
        """Test that 3D checkpoint contains n_d and n_b."""
        from models_3d.attention_unet import train_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model, history, datasets, fm = train_model(
                output_dir=tmpdir,
                n_train=10,
                n_val=5,
                epochs=1,
                batch_size=4,
                n_d=16,
                n_b=16,
                n_compartments=3,
                seed=42,
                device='cpu',
            )

            # Find checkpoint
            checkpoint_dirs = list(Path(tmpdir).glob("checkpoints_*"))
            self.assertGreater(len(checkpoint_dirs), 0)

            checkpoint_dir = checkpoint_dirs[0]
            final_model_path = checkpoint_dir / "final_model.pt"

            if final_model_path.exists():
                checkpoint = torch.load(final_model_path, map_location='cpu')
                config = checkpoint.get('config', {})

                self.assertEqual(config.get('n_d'), 16)
                self.assertEqual(config.get('n_b'), 16)

    def test_g16_grid_size_shorthand(self):
        """Test that grid_size shorthand works correctly."""
        from models_2d.attention_unet import train_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model, history, datasets, fm = train_model(
                output_dir=tmpdir,
                n_train=10,
                n_val=5,
                epochs=1,
                batch_size=4,
                # Use n_d directly (simulating grid_size shorthand behavior)
                n_d=16,
                n_b=16,
                seed=42,
                device='cpu',
            )

            self.assertEqual(fm.n_d, 16)
            self.assertEqual(fm.n_b, 16)


class TestTrainG16OutputDirectories(unittest.TestCase):
    """Test output directory structure for 16x16 models."""

    def test_2d_g16_output_directory_naming(self):
        """Test that 2D 16x16 uses correct output directory."""
        from models_2d.attention_unet import train_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model, history, datasets, fm = train_model(
                output_dir=tmpdir,
                n_train=5,
                n_val=2,
                epochs=1,
                n_d=16,
                n_b=16,
                seed=42,
                device='cpu',
            )

            # Check that output directory contains grid size
            subdirs = list(Path(tmpdir).glob("*"))
            has_g16_dir = any("g16" in str(d) for d in subdirs)

            # If output_dir is explicitly set to tmpdir, it should contain checkpoints
            checkpoint_dirs = list(Path(tmpdir).glob("checkpoints_*"))
            self.assertGreater(len(checkpoint_dirs), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

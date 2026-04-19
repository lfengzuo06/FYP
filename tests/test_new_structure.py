"""
Tests for the new modular structure: dexsy_core, models_2d, benchmarks_2d.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestDexsyCore(unittest.TestCase):
    """Test dexsy_core module imports and basic functionality."""

    def test_imports(self):
        """dexsy_core can be imported and its components work."""
        import numpy as np
        from dexsy_core import ForwardModel2D, compute_dei
        from dexsy_core.metrics import (
            compute_mse, compute_mae, compute_ssim,
            compute_dssim, compute_metrics_dict
        )
        from dexsy_core.preprocessing import (
            build_model_inputs,
            build_position_channel,
            validate_signal_grid,
        )
        self.assertTrue(True)  # All imports succeeded

    def test_forward_model(self):
        """ForwardModel2D generates valid samples."""
        import numpy as np
        from dexsy_core import ForwardModel2D, compute_dei

        fm = ForwardModel2D(n_d=64, n_b=64)
        f, s, params = fm.generate_sample(n_compartments=2)

        self.assertEqual(f.shape, (64, 64))
        self.assertEqual(s.shape, (64, 64))
        self.assertGreater(f.sum(), 0.9)
        self.assertLess(f.sum(), 1.1)
        self.assertGreaterEqual(compute_dei(f), 0)

    def test_metrics(self):
        """Metrics functions compute correctly."""
        import numpy as np
        from dexsy_core.metrics import compute_mse, compute_mae, compute_ssim, compute_dei

        # Use random but similar images for SSIM test
        np.random.seed(42)
        y_true = np.random.rand(64, 64).astype(np.float32) * 0.5 + 0.25
        y_pred = y_true + 0.01 * np.random.randn(64, 64).astype(np.float32)

        self.assertLess(compute_mse(y_true, y_pred), 0.001)
        self.assertLess(compute_mae(y_true, y_pred), 0.05)
        self.assertGreater(compute_ssim(y_true, y_pred), 0.9)
        
        # Test DEI computation
        spectrum = np.random.rand(64, 64).astype(np.float32)
        spectrum = spectrum / spectrum.sum()
        self.assertGreaterEqual(compute_dei(spectrum), 0)

    def test_preprocessing(self):
        """Preprocessing functions work correctly."""
        import numpy as np
        from dexsy_core import ForwardModel2D
        from dexsy_core.preprocessing import build_model_inputs

        fm = ForwardModel2D(n_d=64, n_b=64)
        s = np.random.rand(64, 64).astype(np.float32)
        inputs = build_model_inputs(s, fm)

        self.assertEqual(inputs.shape, (1, 3, 64, 64))


class TestModels2D(unittest.TestCase):
    """Test models_2d module imports and basic functionality."""

    def test_imports(self):
        """models_2d.attention_unet can be imported."""
        from models_2d.attention_unet import (
            AttentionUNet2D,
            InferencePipeline,
            train_model,
            predict,
        )
        self.assertTrue(True)  # All imports succeeded

    def test_plain_unet_imports(self):
        """models_2d.plain_unet can be imported."""
        from models_2d.plain_unet import (
            PlainUNet2D,
            PlainUNetLoss,
            train_model,
        )
        self.assertTrue(True)  # All imports succeeded

    def test_model_forward(self):
        """AttentionUNet2D forward pass works."""
        import torch
        from models_2d.attention_unet import AttentionUNet2D

        model = AttentionUNet2D(in_channels=3, base_filters=16)
        x = torch.randn(2, 3, 64, 64)
        y = model(x)

        self.assertEqual(y.shape, (2, 1, 64, 64))
        self.assertTrue((y >= 0).all())  # Non-negative output
        self.assertTrue((y.sum(dim=(2, 3)) > 0.9).all())  # Sum to ~1

    def test_plain_unet_forward(self):
        """PlainUNet2D forward pass works."""
        import torch
        from models_2d.plain_unet import PlainUNet2D

        model = PlainUNet2D(in_channels=3, base_filters=16)
        x = torch.randn(2, 3, 64, 64)
        y = model(x)

        self.assertEqual(y.shape, (2, 1, 64, 64))
        self.assertTrue((y >= 0).all())  # Non-negative output
        self.assertTrue((y.sum(dim=(2, 3)) > 0.9).all())  # Sum to ~1

    def test_plain_unet_smaller_than_attention(self):
        """Plain U-Net has fewer parameters than Attention U-Net."""
        import torch
        from models_2d.attention_unet import AttentionUNet2D
        from models_2d.plain_unet import PlainUNet2D

        attn = AttentionUNet2D(in_channels=3, base_filters=32)
        plain = PlainUNet2D(in_channels=3, base_filters=32)

        attn_params = sum(p.numel() for p in attn.parameters())
        plain_params = sum(p.numel() for p in plain.parameters())

        self.assertLess(plain_params, attn_params)  # Plain should be smaller
        print(f"\n  Plain U-Net: {plain_params:,} params")
        print(f"  Attention U-Net: {attn_params:,} params")

    def test_inference_pipeline_init(self):
        """InferencePipeline can be initialized without a checkpoint."""
        from models_2d.attention_unet import InferencePipeline
        from dexsy_core import ForwardModel2D

        fm = ForwardModel2D(n_d=64, n_b=64)
        pipeline = InferencePipeline(forward_model=fm)

        self.assertEqual(pipeline.get_model_name(), "attention_unet")
        self.assertIn("model_type", pipeline.get_model_info())


class TestBenchmarks2D(unittest.TestCase):
    """Test benchmarks_2d module imports and basic functionality."""

    def test_imports(self):
        """benchmarks_2d can be imported."""
        from benchmarks_2d import ILTInferencePipeline, predict_ilt
        self.assertTrue(True)  # All imports succeeded

    def test_ilt_pipeline_init(self):
        """ILTInferencePipeline can be initialized."""
        from benchmarks_2d import ILTInferencePipeline
        from dexsy_core import ForwardModel2D

        fm = ForwardModel2D(n_d=64, n_b=64)
        pipeline = ILTInferencePipeline(forward_model=fm)

        self.assertEqual(pipeline.get_model_name(), "2d_ilt")
        self.assertIn("2D ILT", pipeline.get_model_info()["model_type"])

    def test_ilt_predict(self):
        """ILTInferencePipeline.predict works on synthetic data."""
        from benchmarks_2d import ILTInferencePipeline
        from dexsy_core import ForwardModel2D

        fm = ForwardModel2D(n_d=64, n_b=64)
        f, s, params = fm.generate_sample(n_compartments=2)

        pipeline = ILTInferencePipeline(forward_model=fm)
        result = pipeline.predict(s, include_figure=False)

        self.assertEqual(result.reconstructed_spectrum.shape, (64, 64))
        self.assertGreater(result.inference_time, 0)
        self.assertGreaterEqual(result.dei, 0)


class TestBackwardCompatibility(unittest.TestCase):
    """Test that improved_2d_dexsy thin wrapper works correctly."""

    def test_improved_2d_dexsy_imports(self):
        """improved_2d_dexsy still exports all expected names."""
        from improved_2d_dexsy import (
            ForwardModel2D,
            compute_dei,
            DEXSYInferencePipeline,
            InferencePipeline,
            available_models,
            InferenceConfig,
            CHECKPOINTS_DIR,
            # I/O
            load_matrix,
            save_prediction_result,
            to_serializable,
        )
        self.assertTrue(True)  # All imports succeeded

    def test_forward_model_via_wrapper(self):
        """ForwardModel2D works via improved_2d_dexsy."""
        from improved_2d_dexsy import ForwardModel2D

        fm = ForwardModel2D(n_d=64, n_b=64)
        f, s, params = fm.generate_sample()
        self.assertEqual(f.shape, (64, 64))

    def test_inference_pipeline_via_wrapper(self):
        """InferencePipeline works via improved_2d_dexsy."""
        from improved_2d_dexsy import InferencePipeline
        from improved_2d_dexsy import ForwardModel2D

        fm = ForwardModel2D(n_d=64, n_b=64)
        pipeline = InferencePipeline(forward_model=fm)
        self.assertEqual(pipeline.get_model_name(), "attention_unet")


if __name__ == "__main__":
    import numpy as np
    unittest.main()

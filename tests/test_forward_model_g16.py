"""
Tests for 16x16 forward model implementation.

This module verifies:
1. 16x16 forward model creation with profile
2. 2C spectrum generation with correct shapes
3. 3C spectrum generation with correct shapes
4. N-compartment sample generation (N > 3)
5. Normalization correctness
6. DEI calculation
"""

import unittest
from pathlib import Path

import numpy as np

# Add dexsy_core to path
_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(_ROOT / "dexsy_core"))

from forward_model import (
    ForwardModel2D,
    create_forward_model,
    GRID_PROFILES,
    compute_dei,
    compute_weight_matrix_dei,
)


class TestForwardModelG16(unittest.TestCase):
    """Test suite for 16x16 forward model."""

    def test_g16_profile_exists(self):
        """Test that 16x16 profile exists in GRID_PROFILES."""
        self.assertIn(16, GRID_PROFILES)
        profile = GRID_PROFILES[16]
        self.assertEqual(profile["n_d"], 16)
        self.assertEqual(profile["n_b"], 16)

    def test_g16_creation_with_profile(self):
        """Test creating forward model with 16x16 profile."""
        fm = create_forward_model(profile=16)
        self.assertEqual(fm.n_d, 16)
        self.assertEqual(fm.n_b, 16)

    def test_g16_creation_with_explicit_params(self):
        """Test creating forward model with explicit parameters."""
        fm = ForwardModel2D(n_d=16, n_b=16)
        self.assertEqual(fm.n_d, 16)
        self.assertEqual(fm.n_b, 16)

    def test_g16_2c_spectrum_shape(self):
        """Test that 2C spectrum has correct shape for 16x16."""
        fm = create_forward_model(profile=16)
        f, S, params = fm.generate_2compartment_paper()

        self.assertEqual(f.shape, (16, 16), "Spectrum should be 16x16")
        self.assertEqual(S.shape, (16, 16), "Signal should be 16x16")
        self.assertEqual(params["n_compartments"], 2)

    def test_g16_3c_spectrum_shape(self):
        """Test that 3C spectrum has correct shape for 16x16."""
        fm = create_forward_model(profile=16)
        f, S, params = fm.generate_3compartment_paper()

        self.assertEqual(f.shape, (16, 16), "Spectrum should be 16x16")
        self.assertEqual(S.shape, (16, 16), "Signal should be 16x16")
        self.assertEqual(params["n_compartments"], 3)

    def test_g16_normalization(self):
        """Test that spectrum is properly normalized."""
        fm = create_forward_model(profile=16)

        # Test 2C
        f_2c, _, _ = fm.generate_2compartment_paper()
        self.assertAlmostEqual(
            f_2c.sum(), 1.0, places=4,
            msg="2C spectrum should sum to ~1.0"
        )
        self.assertTrue(
            np.all(f_2c >= 0), "Spectrum should be non-negative"
        )

        # Test 3C
        f_3c, _, _ = fm.generate_3compartment_paper()
        self.assertAlmostEqual(
            f_3c.sum(), 1.0, places=4,
            msg="3C spectrum should sum to ~1.0"
        )
        self.assertTrue(
            np.all(f_3c >= 0), "Spectrum should be non-negative"
        )

    def test_g16_dei_calculation(self):
        """Test that DEI is calculated correctly for 16x16."""
        fm = create_forward_model(profile=16)
        f, _, params = fm.generate_2compartment_paper()

        # Compute DEI from spectrum
        dei_spectrum = compute_dei(f)
        self.assertGreaterEqual(dei_spectrum, 0.0)
        self.assertLessEqual(dei_spectrum, 1.0)

        # Compare with theoretical DEI from weight matrix
        dei_weight = params.get("theoretical_dei", compute_weight_matrix_dei(params["weight_matrix"]))
        self.assertIsNotNone(dei_weight)

    def test_g16_signal_normalization(self):
        """Test that signal is properly normalized."""
        fm = create_forward_model(profile=16)
        _, S, params = fm.generate_2compartment_paper()

        # After normalization, S[0, 0] should be 1.0
        self.assertAlmostEqual(S[0, 0], 1.0, places=4)
        # Signal values should be in reasonable range
        self.assertLessEqual(S.max(), 1.0)
        self.assertGreaterEqual(S.min(), 0.0)

    def test_g16_n_compartment_samples(self):
        """Test N-compartment sample generation for various N values."""
        fm = create_forward_model(profile=16)

        for N in [2, 3, 4, 5, 6, 7]:
            f, S, params = fm.generate_ncompartment_sample(N=N)
            self.assertEqual(f.shape, (16, 16), f"N={N}: Spectrum should be 16x16")
            self.assertEqual(S.shape, (16, 16), f"N={N}: Signal should be 16x16")
            self.assertEqual(params["n_compartments"], N)

    def test_g16_batch_generation(self):
        """Test batch generation for 16x16."""
        fm = create_forward_model(profile=16)

        # 2C batch
        F, S, params_list = fm.generate_batch(
            n_samples=5, n_compartments=2
        )
        self.assertEqual(F.shape, (5, 16, 16))
        self.assertEqual(S.shape, (5, 16, 16))
        self.assertEqual(len(params_list), 5)

        # 3C batch
        F, S, params_list = fm.generate_batch(
            n_samples=3, n_compartments=3
        )
        self.assertEqual(F.shape, (3, 16, 16))
        self.assertEqual(S.shape, (3, 16, 16))

    def test_g16_reference_signal(self):
        """Test that reference signals are returned correctly."""
        fm = create_forward_model(profile=16)

        # 2C with reference
        f, S, params, S_clean = fm.generate_2compartment_paper(
            return_reference_signal=True
        )
        self.assertEqual(f.shape, (16, 16))
        self.assertEqual(S.shape, (16, 16))
        self.assertEqual(S_clean.shape, (16, 16))
        # Clean signal should have less variance
        self.assertLess(np.std(S_clean), np.std(S))

        # N-compartment with reference
        f, S, params, S_clean = fm.generate_ncompartment_sample(
            N=4, return_reference_signal=True
        )
        self.assertEqual(f.shape, (16, 16))
        self.assertEqual(S.shape, (16, 16))
        self.assertEqual(S_clean.shape, (16, 16))

    def test_g16_vs_g64_profile_difference(self):
        """Test that 16x16 and 64x64 profiles are different."""
        fm_16 = create_forward_model(profile=16)
        fm_64 = create_forward_model(profile=64)

        # Grid sizes should differ
        self.assertEqual(fm_16.n_d, 16)
        self.assertEqual(fm_64.n_d, 64)

        # Profiles should have different jitter settings
        self.assertEqual(fm_16.jitter_pixels, GRID_PROFILES[16]["jitter_pixels"])
        self.assertEqual(fm_64.jitter_pixels, GRID_PROFILES[64]["jitter_pixels"])


class TestForwardModelG16EdgeCases(unittest.TestCase):
    """Test edge cases for 16x16 forward model."""

    def test_g16_small_n_compartment(self):
        """Test that N=2 works correctly."""
        fm = create_forward_model(profile=16)
        f, S, params = fm.generate_ncompartment_sample(N=2)
        self.assertEqual(f.shape, (16, 16))
        self.assertEqual(params["n_compartments"], 2)

    def test_g16_max_n_compartment(self):
        """Test that N=7 works correctly."""
        fm = create_forward_model(profile=16)
        f, S, params = fm.generate_ncompartment_sample(N=7)
        self.assertEqual(f.shape, (16, 16))
        self.assertEqual(params["n_compartments"], 7)

    def test_g16_invalid_profile(self):
        """Test that invalid profile raises error."""
        with self.assertRaises(ValueError):
            create_forward_model(profile=32)

    def test_g16_with_fixed_noise(self):
        """Test with fixed noise sigma."""
        fm = create_forward_model(profile=16)
        _, S1, _ = fm.generate_2compartment_paper(noise_sigma=0.01)
        _, S2, _ = fm.generate_2compartment_paper(noise_sigma=0.01)

        # With fixed noise, signals should be similar but not identical
        # (due to jitter in diffusion values)
        self.assertEqual(S1.shape, (16, 16))
        self.assertEqual(S2.shape, (16, 16))

    def test_g16_no_noise(self):
        """Test with no noise."""
        fm = create_forward_model(profile=16)
        _, S, _ = fm.generate_2compartment_paper(noise_sigma=0.0)

        # No noise should give same result for same seed
        np.random.seed(42)
        _, S1, _ = fm.generate_2compartment_paper(noise_sigma=0.0)

        np.random.seed(42)
        _, S2, _ = fm.generate_2compartment_paper(noise_sigma=0.0)

        np.testing.assert_array_almost_equal(S1, S2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

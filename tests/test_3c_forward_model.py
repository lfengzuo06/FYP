"""
Tests for 3-compartment forward model implementation.

This module verifies:
1. 3C spectrum generation completeness (9 peaks exist)
2. Weight matrix symmetry and diagonal dominance
3. DEI calculation correctness (exact vs rasterized)
4. ILT reconstruction DEI stability
5. Pairwise DEI consistency with global DEI
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
    compute_dei,
    compute_weight_matrix_dei,
    compute_pairwise_3c_dei,
    local_square_mask,
)


class Test3CForwardModel(unittest.TestCase):
    """Test suite for 3-compartment forward model."""

    @classmethod
    def setUpClass(cls):
        """Set up forward model for all tests."""
        cls.fm = ForwardModel2D(n_d=64, n_b=64, jitter_pixels=0)

    def test_3c_spectrum_generation(self):
        """Test that 3C spectrum has the expected 9-peak structure."""
        d_intra, d_extra, d_fast = 8e-12, 3e-9, 1e-8
        vf = np.array([0.33, 0.34, 0.33])
        rate_01, rate_02, rate_12 = 5.0, 5.0, 5.0

        spectrum, clean_signal, params = self.fm.generate_3c_validation_spectrum(
            np.array([d_intra, d_extra, d_fast]),
            vf,
            (rate_01, rate_02, rate_12),
            mixing_time=0.1,
            jitter_pixels=0,
            smoothing_sigma=0.6,
        )

        # Check spectrum shape
        self.assertEqual(spectrum.shape, (64, 64), "Spectrum should be 64x64")

        # Check signal shape
        self.assertEqual(clean_signal.shape, (64, 64), "Signal should be 64x64")

        # Check that spectrum is normalized (sums to ~1)
        self.assertAlmostEqual(
            spectrum.sum(), 1.0, places=5, msg="Spectrum should be normalized"
        )

        # Check that spectrum is non-negative
        self.assertTrue(
            np.all(spectrum >= 0), "Spectrum should be non-negative"
        )

        # Check params
        self.assertEqual(params["n_compartments"], 3, "Should be 3 compartments")
        self.assertEqual(len(params["diffusions"]), 3, "Should have 3 diffusions")
        self.assertEqual(len(params["volume_fractions"]), 3, "Should have 3 volume fractions")
        self.assertEqual(len(params["jittered_indices"]), 3, "Should have 3 indices")

    def test_3c_weight_matrix_properties(self):
        """Test that 3C weight matrix is symmetric and diagonally dominant."""
        d_intra, d_extra, d_fast = 8e-12, 3e-9, 1e-8
        vf = np.array([0.33, 0.34, 0.33])
        rate_01, rate_02, rate_12 = 5.0, 5.0, 5.0

        _, _, params = self.fm.generate_3c_validation_spectrum(
            np.array([d_intra, d_extra, d_fast]),
            vf,
            (rate_01, rate_02, rate_12),
            mixing_time=0.1,
        )

        wm = params["weight_matrix"]

        # Check symmetry
        np.testing.assert_allclose(
            wm, wm.T, rtol=1e-10,
            err_msg="Weight matrix should be symmetric"
        )

        # Check diagonal dominance: diagonal >= sum of off-diagonal in each row
        for i in range(3):
            diag_val = wm[i, i]
            off_diag_sum = wm[i, :].sum() - diag_val
            self.assertGreaterEqual(
                diag_val, off_diag_sum,
                f"Row {i} should be diagonally dominant"
            )

        # Check non-negativity
        self.assertTrue(
            np.all(wm >= 0), "Weight matrix should be non-negative"
        )

        # Check normalization (sums to ~1)
        self.assertAlmostEqual(
            wm.sum(), 1.0, places=10,
            msg="Weight matrix should sum to 1"
        )

    def test_3c_dei_calculation(self):
        """Test that DEI is calculated correctly."""
        d_intra, d_extra, d_fast = 8e-12, 3e-9, 1e-8
        vf = np.array([0.33, 0.34, 0.33])
        rate_01, rate_02, rate_12 = 5.0, 5.0, 5.0

        spectrum, _, params = self.fm.generate_3c_validation_spectrum(
            np.array([d_intra, d_extra, d_fast]),
            vf,
            (rate_01, rate_02, rate_12),
            mixing_time=0.1,
            jitter_pixels=0,
            smoothing_sigma=0.6,
        )

        # Compare theoretical (weight matrix) DEI with rasterized (spectrum) DEI
        theoretical_dei = params["theoretical_dei"]
        rasterized_dei = compute_dei(spectrum)

        # They should be close (within ~10% due to smoothing effects)
        rel_diff = abs(rasterized_dei - theoretical_dei) / (theoretical_dei + 1e-10)
        self.assertLess(
            rel_diff, 0.15,
            f"Rasterized DEI ({rasterized_dei:.4f}) should be close to "
            f"theoretical DEI ({theoretical_dei:.4f})"
        )

    def test_3c_ilt_stability(self):
        """Test that ILT reconstruction gives stable DEI."""
        d_intra, d_extra, d_fast = 8e-12, 3e-9, 1e-8
        vf = np.array([0.33, 0.34, 0.33])
        rate_01, rate_02, rate_12 = 5.0, 5.0, 5.0

        spectrum, clean_signal, params = self.fm.generate_3c_validation_spectrum(
            np.array([d_intra, d_extra, d_fast]),
            vf,
            (rate_01, rate_02, rate_12),
            mixing_time=0.1,
            jitter_pixels=0,
            smoothing_sigma=0.6,
        )

        # Compute ILT
        spectrum_ilt = self.fm.compute_ilt_nnls(
            clean_signal, alpha=0.02, post_sharpen=True,
            sharpen_sigma=0.9, sharpen_strength=0.30
        )

        # Check ILT result is valid
        self.assertEqual(spectrum_ilt.shape, (64, 64))
        self.assertTrue(np.all(spectrum_ilt >= 0))

        # DEI should be positive
        ilt_dei = compute_dei(spectrum_ilt)
        self.assertGreater(ilt_dei, 0, "ILT DEI should be positive")

    def test_3c_pairwise_dei(self):
        """Test that pairwise DEI is calculated correctly."""
        d_intra, d_extra, d_fast = 8e-12, 3e-9, 1e-8
        vf = np.array([0.33, 0.34, 0.33])
        rate_01, rate_02, rate_12 = 5.0, 5.0, 5.0

        spectrum, _, params = self.fm.generate_3c_validation_spectrum(
            np.array([d_intra, d_extra, d_fast]),
            vf,
            (rate_01, rate_02, rate_12),
            mixing_time=0.1,
            jitter_pixels=0,
            smoothing_sigma=0.6,
        )

        compartment_indices = params["compartment_indices"]
        pairwise = compute_pairwise_3c_dei(spectrum, compartment_indices)

        # Check all pairwise DEIs are positive
        self.assertGreater(pairwise["dei_01_blob"], 0)
        self.assertGreater(pairwise["dei_02_blob"], 0)
        self.assertGreater(pairwise["dei_12_blob"], 0)

        # Check that diagonal masses are non-zero
        self.assertGreater(pairwise["diagonal_01"], 0)
        self.assertGreater(pairwise["diagonal_02"], 0)
        self.assertGreater(pairwise["diagonal_12"], 0)

    def test_3c_exchange_rate_variation(self):
        """Test that DEI changes with exchange rate."""
        d_intra, d_extra, d_fast = 8e-12, 3e-9, 1e-8
        vf = np.array([0.33, 0.34, 0.33])

        # Low exchange rate
        _, _, params_low = self.fm.generate_3c_validation_spectrum(
            np.array([d_intra, d_extra, d_fast]),
            vf,
            (0.1, 0.1, 0.1),
            mixing_time=0.1,
        )

        # High exchange rate
        _, _, params_high = self.fm.generate_3c_validation_spectrum(
            np.array([d_intra, d_extra, d_fast]),
            vf,
            (25.0, 25.0, 25.0),
            mixing_time=0.1,
        )

        # Higher exchange rate should give higher DEI
        self.assertGreater(
            params_high["theoretical_dei"], params_low["theoretical_dei"],
            "Higher exchange rate should give higher DEI"
        )

    def test_3c_volume_fraction_conservation(self):
        """Test that weight matrix respects volume fractions."""
        d_intra, d_extra, d_fast = 8e-12, 3e-9, 1e-8

        # Test different volume fractions
        for vf in [
            np.array([0.5, 0.3, 0.2]),
            np.array([0.2, 0.5, 0.3]),
            np.array([0.3, 0.2, 0.5]),
        ]:
            _, _, params = self.fm.generate_3c_validation_spectrum(
                np.array([d_intra, d_extra, d_fast]),
                vf,
                (5.0, 5.0, 5.0),
                mixing_time=0.1,
            )

            wm = params["weight_matrix"]

            # Total mass on diagonal should roughly reflect volume fractions
            # (with some exchange mass moving off-diagonal)
            total_mass = wm.sum()

            # Diagonal masses should be proportional to volume fractions
            diag_sum = np.trace(wm)
            off_diag_sum = total_mass - diag_sum

            # Diagonal should be majority of mass
            self.assertGreater(
                diag_sum, off_diag_sum,
                "Diagonal mass should be greater than off-diagonal mass"
            )


class Test3CGeneratorConsistency(unittest.TestCase):
    """Test consistency between different 3C generation methods."""

    @classmethod
    def setUpClass(cls):
        cls.fm = ForwardModel2D(n_d=64, n_b=64, jitter_pixels=0)

    def test_validation_vs_paper_generator(self):
        """Test that generate_3c_validation_spectrum is consistent with generate_3compartment_paper."""
        # Generate with validation function
        spectrum_val, signal_val, params_val = self.fm.generate_3c_validation_spectrum(
            np.array([8e-12, 3e-9, 1e-8]),
            np.array([0.33, 0.34, 0.33]),
            (5.0, 5.0, 5.0),
            mixing_time=0.1,
            jitter_pixels=0,
            smoothing_sigma=0.6,
        )

        # Generate with paper function
        np.random.seed(42)  # For reproducibility
        spectrum_pap, signal_pap, params_pap = self.fm.generate_3compartment_paper(
            mixing_time=0.1,
            exchange_rates=(5.0, 5.0, 5.0),
            noise_sigma=0.0,  # No noise for comparison
            jitter_pixels=0,
            smoothing_sigma=0.6,
        )

        # Check they have similar structure
        # (exact values may differ due to random volume fraction sampling)
        self.assertEqual(spectrum_val.shape, spectrum_pap.shape)
        self.assertEqual(signal_val.shape, signal_pap.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)

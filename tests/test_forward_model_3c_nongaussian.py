"""Tests for the Stanisz-style 3C non-Gaussian forward model."""

import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(_ROOT / "dexsy_core"))

from forward_model_3c_nongaussian import (
    ForwardModel3CNonGaussian,
    PATHWAYS_3C,
    GRID_PROFILES_3C_NG,
    create_forward_model_3c_nongaussian,
)


class TestForwardModel3CNonGaussian(unittest.TestCase):
    """Validation tests for E/T/S 3C non-Gaussian model."""

    @classmethod
    def setUpClass(cls):
        cls.fm = ForwardModel3CNonGaussian(
            n_b=16,
            n_restrict_terms=120,
            gradient_spacing="linear",
        )

    def test_restricted_kernel_boundary(self):
        """Restricted kernels should satisfy K(g=0)=1 and remain non-negative."""
        kernels = self.fm.compartment_kernels(
            g=self.fm.G1,
            extracellular_diffusivity=1.6e-9,
            intracellular_diffusivity=0.8e-9,
            axon_restricted_length=1.0e-6,
            sphere_radius=4.0e-6,
        )
        self.assertAlmostEqual(float(kernels["T"][0]), 1.0, places=8)
        self.assertAlmostEqual(float(kernels["S"][0]), 1.0, places=8)
        self.assertTrue(np.all(kernels["T"] >= 0.0))
        self.assertTrue(np.all(kernels["S"] >= 0.0))

    def test_default_and_profile_16(self):
        """Default model and profile factory should use 16x16 by default."""
        fm_default = ForwardModel3CNonGaussian()
        self.assertEqual(fm_default.n_b, 16)
        self.assertIn(16, GRID_PROFILES_3C_NG)
        fm_profile = create_forward_model_3c_nongaussian(profile=16)
        self.assertEqual(fm_profile.n_b, 16)

    def test_generator_from_permeability_detailed_balance(self):
        """Permeability mapping should enforce detailed-balance back-rates."""
        phi = np.array([0.45, 0.35, 0.20], dtype=np.float64)  # E,T,S
        q, rates = ForwardModel3CNonGaussian.build_generator_from_permeability(
            phi=phi,
            sphere_diameter=8e-6,
            sphere_permeability=1e-5,
            axon_surface_to_volume=2.0e6,
            axon_permeability=5e-7,
        )
        # phi_S * k_SE == phi_E * k_ES
        lhs_s = phi[2] * rates["k_SE"]
        rhs_s = phi[0] * rates["k_ES"]
        self.assertAlmostEqual(lhs_s, rhs_s, places=10)
        # phi_T * k_TE == phi_E * k_ET
        lhs_t = phi[1] * rates["k_TE"]
        rhs_t = phi[0] * rates["k_ET"]
        self.assertAlmostEqual(lhs_t, rhs_t, places=10)
        self.assertEqual(q.shape, (3, 3))

    def test_permeability_rates_feed_directly_to_signal(self):
        """Rates returned by permeability helper should be accepted directly by compute_signal."""
        phi = np.array([0.45, 0.35, 0.20], dtype=np.float64)
        _q, rates = ForwardModel3CNonGaussian.build_generator_from_permeability(
            phi=phi,
            sphere_diameter=8e-6,
            sphere_permeability=1e-5,
            axon_surface_to_volume=2.0e6,
            axon_permeability=5e-7,
        )
        signal, details = self.fm.compute_signal(
            phi=phi,
            rates=rates,  # pass helper output directly
            mixing_time=0.08,
            extracellular_diffusivity=1.7e-9,
            intracellular_diffusivity=0.7e-9,
            axon_restricted_length=0.9e-6,
            sphere_radius=3.8e-6,
        )
        self.assertEqual(signal.shape, (16, 16))
        self.assertAlmostEqual(float(signal[0, 0]), 1.0, places=8)
        self.assertAlmostEqual(float(details["weight_matrix"].sum()), 1.0, places=8)

    def test_zero_exchange_gives_diagonal_weights(self):
        """With Q=0, P=I and only EE/TT/SS pathways remain."""
        phi = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        q = np.zeros((3, 3), dtype=np.float64)
        w, p, phi_n = self.fm.compute_weight_matrix(phi=phi, q=q, mixing_time=0.1)

        np.testing.assert_allclose(p, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.diag(w), phi_n, atol=1e-10)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=10)
        self.assertAlmostEqual(float(w[0, 1] + w[0, 2] + w[1, 0] + w[1, 2] + w[2, 0] + w[2, 1]), 0.0, places=10)

    def test_signal_shape_and_pathways(self):
        """Signal should be n_b x n_b with all 9 pathways available."""
        phi = np.array([0.42, 0.33, 0.25], dtype=np.float64)
        q = self.fm.build_generator(
            k_et=3.0,
            k_te=2.0,
            k_es=1.5,
            k_se=1.2,
            k_ts=0.0,
            k_st=0.0,
        )
        signal, details = self.fm.compute_signal(
            phi=phi,
            q=q,
            mixing_time=0.08,
            extracellular_diffusivity=1.7e-9,
            intracellular_diffusivity=0.7e-9,
            axon_restricted_length=0.9e-6,
            sphere_radius=3.8e-6,
            return_pathway_signals=True,
        )

        self.assertEqual(signal.shape, (16, 16))
        self.assertAlmostEqual(float(signal[0, 0]), 1.0, places=8)
        self.assertEqual(len(details["pathway_weights"]), 9)
        self.assertEqual(set(details["pathway_weights"].keys()), set(PATHWAYS_3C))
        self.assertIn("pathway_signals", details)

        pathway_sum = np.zeros_like(signal)
        for mat in details["pathway_signals"].values():
            pathway_sum += mat

        np.testing.assert_allclose(pathway_sum, signal, rtol=1e-6, atol=1e-8)
        self.assertAlmostEqual(sum(details["pathway_weights"].values()), 1.0, places=8)

    def test_dei_functions(self):
        """DEI computed from matrix and pathways should match."""
        phi = np.array([0.42, 0.33, 0.25], dtype=np.float64)
        q = self.fm.build_generator(k_et=3.0, k_te=2.0, k_es=1.5, k_se=1.2)
        _, details = self.fm.compute_signal(
            phi=phi,
            q=q,
            mixing_time=0.08,
            extracellular_diffusivity=1.7e-9,
            intracellular_diffusivity=0.7e-9,
            axon_restricted_length=0.9e-6,
            sphere_radius=3.8e-6,
        )
        dei_w = self.fm.compute_dei_from_weight_matrix(details["weight_matrix"])
        dei_p = self.fm.compute_dei_from_pathway_weights(details["pathway_weights"])
        self.assertAlmostEqual(float(dei_w), float(dei_p), places=10)

    def test_add_rician_noise(self):
        """Rician-noisy output should keep shape and S(0,0)=1 after normalization."""
        phi = np.array([0.42, 0.33, 0.25], dtype=np.float64)
        q = self.fm.build_generator(k_et=3.0, k_te=2.0, k_es=1.5, k_se=1.2)
        clean, _ = self.fm.compute_signal(
            phi=phi,
            q=q,
            mixing_time=0.08,
            extracellular_diffusivity=1.7e-9,
            intracellular_diffusivity=0.7e-9,
            axon_restricted_length=0.9e-6,
            sphere_radius=3.8e-6,
            normalize=False,
        )
        noisy = self.fm.add_rician_noise(clean, noise_sigma=0.01, normalize=True, rng=np.random.default_rng(0))
        self.assertEqual(noisy.shape, clean.shape)
        self.assertAlmostEqual(float(noisy[0, 0]), 1.0, places=8)

    def test_sample_dataset_shapes(self):
        """Dataset sampler should return correctly shaped tensors and metadata."""
        signals, params_list, clean = self.fm.sample_dataset(
            n_samples=4,
            noise_sigma=0.01,
            seed=123,
            return_clean_signals=True,
        )
        self.assertEqual(signals.shape, (4, 16, 16))
        self.assertEqual(clean.shape, (4, 16, 16))
        self.assertEqual(len(params_list), 4)
        self.assertIn("dei_weight_matrix", params_list[0])
        self.assertIn("pathway_weights", params_list[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

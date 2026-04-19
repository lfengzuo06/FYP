"""
Paper-aligned 2D DEXSY forward model.

This module implements the same core Laplace forward operator as the earlier
student code, but the default synthetic data generator now follows the
simulation framework described in steven_submission.pdf:

1. 64x64 diffusion and signal grids
2. Log-spaced compartment diffusivities spanning intracellular, extracellular,
   and fast 3C pools
3. Log-grid exchange rates converted to mixing-time dependent probabilities
4. Symmetric, diagonally dominant weight matrices
5. Symmetric peak broadening via pixel jitter and Gaussian smoothing
6. Rician noise followed by noisy b=0 baseline normalisation
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
except ImportError:  # pragma: no cover - exercised only in lighter environments.
    _scipy_gaussian_filter = None


def _gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing with a scipy fallback-free numpy implementation."""
    if sigma <= 0:
        return image
    if _scipy_gaussian_filter is not None:
        return _scipy_gaussian_filter(image, sigma=sigma, mode="nearest")

    radius = max(1, int(np.ceil(3 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(grid ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()

    padded = np.pad(image, ((0, 0), (radius, radius)), mode="edge")
    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"),
        axis=1,
        arr=padded,
    )
    padded = np.pad(smoothed, ((radius, radius), (0, 0)), mode="edge")
    smoothed = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="valid"),
        axis=0,
        arr=padded,
    )
    return smoothed.astype(np.float32)


def _discrete_gaussian_1d_kernel(sigma: float, radius: int) -> np.ndarray:
    """Normalised 1D Gaussian weights for integer offsets in [-radius, radius]."""
    if sigma <= 0:
        k = np.zeros(2 * radius + 1, dtype=np.float64)
        k[radius] = 1.0
        return k
    offs = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-0.5 * (offs / sigma) ** 2)
    g /= g.sum() + 1e-12
    return g


def _discrete_gaussian_2d_kernel(sigma: float, radius: int) -> np.ndarray:
    """Normalised 2D Gaussian on an integer grid."""
    if sigma <= 0:
        k = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
        k[radius, radius] = 1.0
        return k
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    gx = np.exp(-0.5 * (ax / sigma) ** 2)
    g2 = np.outer(gx, gx)
    g2 /= g2.sum() + 1e-12
    return g2


def _unsharp_mask_nonnegative(f: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """Mild sharpening (paper-style Gaussian deconvolution surrogate)."""
    if sigma <= 0 or strength <= 0:
        return f
    blurred = _gaussian_filter(f.astype(np.float64), sigma)
    out = f.astype(np.float64) + strength * (f.astype(np.float64) - blurred)
    return np.clip(out, 0.0, None)


def local_square_mask(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    """Build a square mask around one peak centre on the discrete D-grid."""
    mask = np.zeros(shape, dtype=bool)
    ci, cj = center
    i0, i1 = max(0, ci - radius), min(shape[0], ci + radius + 1)
    j0, j1 = max(0, cj - radius), min(shape[1], cj + radius + 1)
    mask[i0:i1, j0:j1] = True
    return mask


def compute_weight_matrix_dei(weight_matrix: np.ndarray) -> float:
    """
    Compute an exact DEI from a compartment-level 2C or 3C weight matrix.

    This is the cleanest generator-side ground truth because it is measured
    before rasterisation, smoothing, or NNLS blur.
    """
    weight_matrix = np.asarray(weight_matrix, dtype=np.float64)
    diag_sum = float(np.trace(weight_matrix))
    off_diag_sum = float(weight_matrix.sum() - diag_sum)
    return off_diag_sum / (diag_sum + 1e-10)


def compute_pair_blob_masses(
    f: np.ndarray,
    pair_indices: tuple[int, int],
    radius: int = 4,
) -> dict[str, float]:
    """
    Sum local diagonal and off-diagonal peak masses for one 2C peak pair.

    The masks are anchored at the expected peak centres, matching the "peak
    intensity" interpretation used in the validation section of the report.
    """
    i, j = pair_indices
    diag_mask = local_square_mask(f.shape, (i, i), radius) | local_square_mask(f.shape, (j, j), radius)
    off_mask = local_square_mask(f.shape, (i, j), radius) | local_square_mask(f.shape, (j, i), radius)
    return {
        "diagonal": float(f[diag_mask].sum()),
        "off_diagonal": float(f[off_mask].sum()),
    }


def compute_pair_blob_dei(
    f: np.ndarray,
    pair_indices: tuple[int, int],
    radius: int = 4,
) -> float:
    """Compute a local blob-wise DEI for one expected 2C peak pair."""
    masses = compute_pair_blob_masses(f, pair_indices, radius=radius)
    return masses["off_diagonal"] / (masses["diagonal"] + 1e-10)


class ForwardModel2D:
    """2D DEXSY forward model for paper-style synthetic dataset generation."""

    def __init__(
        self,
        n_d: int = 64,
        n_b: int = 64,
        d_min: float = 5e-12,
        d_max: float = 5e-8,
        g_max: float = 1.5,
        delta: float = 0.003,
        DELTA: float = 0.01,
        gamma: float = 267.52e6,
        gradient_spacing: str = "linear",
        normalize_signals: bool = True,
        mixing_time_range: tuple = (0.015, 0.300),
        exchange_rate_range: tuple = (0.1, 30.0),
        exchange_rate_grid_size: int = 24,
        jitter_pixels: int = 1,
        smoothing_sigma_range: tuple = (0.65, 1.15),
        min_index_separation: int = 4,
        spectral_broadening_mode: str = "directional",
    ):
        """
        Initialise the paper-style 2D DEXSY forward model.

        Args:
            n_d: Number of diffusion coefficient grid points.
            n_b: Number of gradient / b-value grid points.
            d_min: Minimum diffusion coefficient.
            d_max: Maximum diffusion coefficient.
            g_max: Maximum gradient strength in T/m.
            delta: Gradient pulse duration in seconds.
            DELTA: Diffusion time in seconds.
            gamma: Gyromagnetic ratio in rad/s/T.
            gradient_spacing: "linear" or "log".
            normalize_signals: If True, normalise by the noisy b=0 entry.
            mixing_time_range: Mixing-time sampling range in seconds.
            exchange_rate_range: Exchange-rate sampling range in s^-1.
            exchange_rate_grid_size: Number of log-spaced rate candidates.
            jitter_pixels: Peak jitter in pixels.
            smoothing_sigma_range: Gaussian broadening sigma range (pixels).
            min_index_separation: Minimum projected grid separation between
                compartment diffusivities.
            spectral_broadening_mode: "directional" (default) places self-retention
                mass only along the D1=D2 diagonal on the grid, and exchange mass
                with a local 2D Gaussian. This avoids the large artificial DEI
                inflation caused by isotropic blurring of diagonal peaks into
                off-diagonal bins. Use "isotropic" for the legacy whole-image
                Gaussian convolution.
        """
        self.n_d = n_d
        self.n_b = n_b
        self.delta = delta
        self.DELTA = DELTA
        self.gamma = gamma
        self.gradient_spacing = gradient_spacing
        self.normalize_signals = normalize_signals
        self.mixing_time_range = mixing_time_range
        self.exchange_rate_range = exchange_rate_range
        self.exchange_rate_grid_size = exchange_rate_grid_size
        self.jitter_pixels = jitter_pixels
        self.smoothing_sigma_range = smoothing_sigma_range
        self.min_index_separation = min_index_separation
        if spectral_broadening_mode not in ("directional", "isotropic"):
            raise ValueError(f"Unknown spectral_broadening_mode: {spectral_broadening_mode}")
        self.spectral_broadening_mode = spectral_broadening_mode

        self.compartment_ranges = {
            "intracellular": (5e-12, 3e-11),
            "extracellular": (3e-11, 5e-9),
            "fast": (5e-9, 5e-8),
        }

        self.D1 = np.logspace(np.log10(d_min), np.log10(d_max), n_d)
        self.D2 = np.logspace(np.log10(d_min), np.log10(d_max), n_d)

        if gradient_spacing == "linear":
            self.G1 = np.linspace(0.0, g_max, n_b)
            self.G2 = np.linspace(0.0, g_max, n_b)
        elif gradient_spacing == "log":
            positive_g = np.logspace(
                np.log10(max(g_max / 1e4, 1e-8)),
                np.log10(g_max),
                n_b - 1,
            )
            self.G1 = np.concatenate(([0.0], positive_g))
            self.G2 = np.concatenate(([0.0], positive_g))
        else:
            raise ValueError(f"Unknown gradient spacing: {gradient_spacing}")

        coeff = (gamma ** 2) * (delta ** 2) * (DELTA - delta / 3.0)
        self.b1 = (self.G1 ** 2) * coeff
        self.b2 = (self.G2 ** 2) * coeff
        self.exchange_rate_grid = np.logspace(
            np.log10(exchange_rate_range[0]),
            np.log10(exchange_rate_range[1]),
            exchange_rate_grid_size,
        )

        self._compute_kernel()

    def _compute_kernel(self):
        """Precompute the exponential kernel tensor."""
        D1G, D2G = np.meshgrid(self.D1, self.D2, indexing="ij")
        b1G, b2G = np.meshgrid(self.b1, self.b2, indexing="ij")
        kernel = np.exp(-b1G[..., None, None] * D1G) * np.exp(-b2G[..., None, None] * D2G)
        self._D1G = D1G
        self._D2G = D2G
        self._b1G = b1G
        self._b2G = b2G
        self._kernel = kernel.astype(np.float32)
        self.kernel_matrix = self._kernel.reshape(self.n_b * self.n_b, self.n_d * self.n_d)

    def _sample_log_uniform(self, value_range: tuple) -> float:
        """Sample one value uniformly in log space."""
        low, high = value_range
        return float(np.exp(np.random.uniform(np.log(low), np.log(high))))

    def _sample_uniform(self, value_range: tuple) -> float:
        """Sample one value uniformly in linear space."""
        low, high = value_range
        return float(np.random.uniform(low, high))

    def _sample_noise_sigma(self, noise_sigma: float = None, noise_sigma_range: tuple = (0.005, 0.015)) -> float:
        """Sample noise sigma uniformly, matching the paper protocol."""
        if noise_sigma is not None:
            return float(noise_sigma)
        return self._sample_uniform(noise_sigma_range)

    def _sample_mixing_time(self, mixing_time: float = None) -> float:
        """Sample mixing time in seconds."""
        if mixing_time is not None:
            return float(mixing_time)
        return self._sample_log_uniform(self.mixing_time_range)

    def _sample_exchange_rate(self, exchange_rate: float = None) -> float:
        """Sample exchange rate from a logarithmic grid."""
        if exchange_rate is not None:
            return float(exchange_rate)
        return float(np.random.choice(self.exchange_rate_grid))

    def _exchange_probability(self, rate: float, mixing_time: float) -> float:
        """Convert exchange rate to mixing-time dependent exchange probability."""
        return float(1.0 - np.exp(-rate * mixing_time))

    def _nearest_diffusion_index(self, diffusion_value: float) -> int:
        """Project a diffusion value onto the log-spaced reconstruction grid."""
        log_grid = np.log(self.D1)
        return int(np.argmin(np.abs(log_grid - np.log(diffusion_value))))

    def _sample_compartment_diffusions(self, compartment_names: tuple[str, ...]) -> np.ndarray:
        """
        Sample compartment diffusivities while keeping them distinct on the
        reconstruction grid so the intended peaks remain separable.
        """
        min_separation = max(int(self.min_index_separation), 2 * int(self.jitter_pixels) + 1)

        for _ in range(256):
            diffusions = np.array(
                [self._sample_log_uniform(self.compartment_ranges[name]) for name in compartment_names],
                dtype=np.float64,
            )
            indices = np.array([self._nearest_diffusion_index(d) for d in diffusions], dtype=int)
            if len(indices) <= 1:
                return diffusions
            if np.min(np.abs(indices[:, None] - indices[None, :]) + np.eye(len(indices)) * self.n_d) >= min_separation:
                return diffusions

        return diffusions

    def _jitter_index(self, idx: int, jitter_pixels: int) -> int:
        """Apply bounded integer jitter to a grid index."""
        if jitter_pixels <= 0:
            return idx
        offset = int(np.random.randint(-jitter_pixels, jitter_pixels + 1))
        return int(np.clip(idx + offset, 0, self.n_d - 1))

    def _build_weight_matrix(self, volume_fractions: np.ndarray, exchange_probs: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Construct a symmetric, diagonally dominant weight matrix.

        Off-diagonal entries are proportional to the sampled exchange
        probabilities and equilibrium compartment volumes. A single global scale
        factor is applied only when needed to preserve diagonal dominance.
        """
        n_compartments = len(volume_fractions)
        offdiag = np.zeros((n_compartments, n_compartments), dtype=np.float64)

        for i in range(n_compartments):
            for j in range(i + 1, n_compartments):
                offdiag_mass = exchange_probs[i, j] * volume_fractions[i] * volume_fractions[j]
                offdiag[i, j] = offdiag_mass
                offdiag[j, i] = offdiag_mass

        row_offdiag = offdiag.sum(axis=1)
        allowed = 0.49 * volume_fractions
        valid_rows = row_offdiag > 0
        scale = 1.0
        if np.any(valid_rows & (row_offdiag > allowed)):
            scale = float(np.min(allowed[valid_rows] / row_offdiag[valid_rows]))
            scale = min(scale, 1.0)
            offdiag *= scale

        diag = volume_fractions - offdiag.sum(axis=1)
        diag = np.clip(diag, 1e-8, None)

        weight_matrix = offdiag
        np.fill_diagonal(weight_matrix, diag)
        weight_matrix = np.clip(weight_matrix, 0.0, None)
        weight_matrix /= weight_matrix.sum() + 1e-12
        return weight_matrix, float(scale)

    def _pair_exchange_masses(self, weight_matrix: np.ndarray) -> dict[str, float]:
        """Summarise the actual off-diagonal mass carried by each exchange pair."""
        masses: dict[str, float] = {}
        for i in range(weight_matrix.shape[0]):
            for j in range(i + 1, weight_matrix.shape[1]):
                masses[f"{i}-{j}"] = float(weight_matrix[i, j] + weight_matrix[j, i])
        return masses

    def _project_weight_matrix(
        self,
        diffusions: np.ndarray,
        weight_matrix: np.ndarray,
        jitter_pixels: int = None,
        smoothing_sigma: float = None,
    ) -> tuple:
        """
        Project a compartment-level weight matrix onto the 2D D-grid.

        Each compartment pair is placed near its ideal grid location, then the
        spectrum is broadened with Gaussian smoothing to match the paper's
        structural spectrum generation stage.
        """
        if jitter_pixels is None:
            jitter_pixels = self.jitter_pixels
        if smoothing_sigma is None:
            smoothing_sigma = float(np.random.uniform(*self.smoothing_sigma_range))

        spectrum = np.zeros((self.n_d, self.n_d), dtype=np.float64)
        base_indices = [self._nearest_diffusion_index(d) for d in diffusions]
        jittered_indices = [self._jitter_index(idx, jitter_pixels) for idx in base_indices]

        if self.spectral_broadening_mode == "directional":
            sigma = float(smoothing_sigma)
            radius = max(1, int(np.ceil(3.0 * sigma)))
            g1 = _discrete_gaussian_1d_kernel(sigma, radius)
            g2 = _discrete_gaussian_2d_kernel(sigma, radius)
            for i in range(weight_matrix.shape[0]):
                for j in range(weight_matrix.shape[1]):
                    weight = float(weight_matrix[i, j])
                    if weight <= 0:
                        continue
                    ia = jittered_indices[i]
                    jb = jittered_indices[j]
                    if ia == jb:
                        for k, gk in enumerate(g1):
                            kk = k - radius
                            ii = ia + kk
                            if 0 <= ii < self.n_d:
                                spectrum[ii, ii] += weight * gk
                    else:
                        for di in range(2 * radius + 1):
                            for dj in range(2 * radius + 1):
                                ii = ia + di - radius
                                jj = jb + dj - radius
                                if 0 <= ii < self.n_d and 0 <= jj < self.n_d:
                                    spectrum[ii, jj] += weight * g2[di, dj]
        else:
            for i in range(weight_matrix.shape[0]):
                for j in range(weight_matrix.shape[1]):
                    weight = float(weight_matrix[i, j])
                    if weight <= 0:
                        continue
                    idx_i = jittered_indices[i]
                    idx_j = jittered_indices[j]
                    spectrum[idx_i, idx_j] += weight

            spectrum = _gaussian_filter(spectrum.astype(np.float32), sigma=smoothing_sigma)

        spectrum = np.clip(spectrum, 0.0, None)
        spectrum /= spectrum.sum() + 1e-12
        return spectrum.astype(np.float32), float(smoothing_sigma), tuple(int(i) for i in jittered_indices)

    def compute_signal(
        self,
        f: np.ndarray,
        noise_sigma: float = 0.0,
        rician: bool = True,
        normalize: bool = None,
        noise_model: str = "rician",
    ) -> np.ndarray:
        """
        Compute a DEXSY signal from a diffusion distribution.

        The paper normalises after noise addition, so the b=0 scaling is always
        applied after any noise model has been injected.
        """
        if f.ndim == 2:
            f = f[None, ...]

        s = np.einsum("bij,mnij->bmn", f.astype(np.float32), self._kernel, optimize=True)

        if noise_sigma > 0:
            if noise_model is None:
                noise_model = "rician" if rician else "gaussian"
            if noise_model == "rician":
                noise_real = np.random.randn(*s.shape).astype(np.float32) * noise_sigma
                noise_imag = np.random.randn(*s.shape).astype(np.float32) * noise_sigma
                s = np.sqrt((s + noise_real) ** 2 + noise_imag ** 2)
            elif noise_model == "gaussian":
                s = s + np.random.randn(*s.shape).astype(np.float32) * noise_sigma
            elif noise_model == "legacy_uniform":
                noise_real = -2.0 * noise_sigma * np.random.rand(*s.shape).astype(np.float32)
                noise_imag = -2.0 * noise_sigma * np.random.rand(*s.shape).astype(np.float32)
                s = np.sqrt((s + noise_real) ** 2 + noise_imag ** 2)
            else:
                raise ValueError(f"Unknown noise model: {noise_model}")

        if normalize is None:
            normalize = self.normalize_signals
        if normalize:
            s0 = np.maximum(s[:, 0, 0], 1e-10)
            s = s / s0[:, None, None]

        return s[0] if f.shape[0] == 1 else s

    def _generate_paper_spectrum(
        self,
        diffusions: np.ndarray,
        volume_fractions: np.ndarray,
        exchange_rates: np.ndarray,
        mixing_time: float,
        jitter_pixels: int = None,
        smoothing_sigma: float = None,
    ) -> tuple:
        """Generate a broadened paper-style 2D spectrum and metadata."""
        exchange_probs = np.zeros_like(exchange_rates, dtype=np.float64)
        for i in range(exchange_rates.shape[0]):
            for j in range(i + 1, exchange_rates.shape[1]):
                prob = self._exchange_probability(exchange_rates[i, j], mixing_time)
                exchange_probs[i, j] = prob
                exchange_probs[j, i] = prob

        weight_matrix, exchange_scale = self._build_weight_matrix(volume_fractions, exchange_probs)
        spectrum, used_sigma, jittered_indices = self._project_weight_matrix(
            diffusions=diffusions,
            weight_matrix=weight_matrix,
            jitter_pixels=jitter_pixels,
            smoothing_sigma=smoothing_sigma,
        )
        return spectrum, weight_matrix, exchange_probs, used_sigma, exchange_scale, jittered_indices

    def generate_2compartment_paper(
        self,
        mixing_time: float = None,
        exchange_rate: float = None,
        noise_sigma: float = None,
        noise_sigma_range: tuple = (0.005, 0.015),
        noise_model: str = "rician",
        normalize: bool = None,
        jitter_pixels: int = None,
        smoothing_sigma: float = None,
        return_reference_signal: bool = False,
    ) -> tuple:
        """Generate one paper-style 2-compartment DEXSY sample."""
        mixing_time = self._sample_mixing_time(mixing_time)
        noise_sigma = self._sample_noise_sigma(noise_sigma, noise_sigma_range)

        diffusions = self._sample_compartment_diffusions(("intracellular", "extracellular"))
        volume_fractions = np.random.dirichlet(np.array([2.0, 2.0], dtype=np.float64))
        rate = self._sample_exchange_rate(exchange_rate)

        exchange_rates = np.zeros((2, 2), dtype=np.float64)
        exchange_rates[0, 1] = rate
        exchange_rates[1, 0] = rate

        f, weight_matrix, exchange_probs, used_sigma, exchange_scale, jittered_indices = self._generate_paper_spectrum(
            diffusions=diffusions,
            volume_fractions=volume_fractions,
            exchange_rates=exchange_rates,
            mixing_time=mixing_time,
            jitter_pixels=jitter_pixels,
            smoothing_sigma=smoothing_sigma,
        )
        clean_signal = self.compute_signal(f, noise_sigma=0.0, normalize=normalize, noise_model=None)
        s = self.compute_signal(f, noise_sigma=noise_sigma, normalize=normalize, noise_model=noise_model)

        params = {
            "n_compartments": 2,
            "mixing_time": float(mixing_time),
            "noise_sigma": float(noise_sigma),
            "baseline_snr": float(1.0 / max(noise_sigma, 1e-12)),
            "diffusions": diffusions.tolist(),
            "volume_fractions": volume_fractions.tolist(),
            "exchange_rates": {"0-1": float(rate)},
            "exchange_probabilities": {"0-1": float(exchange_probs[0, 1])},
            "exchange_peak_masses": self._pair_exchange_masses(weight_matrix),
            "exchange_probability_scale": float(exchange_scale),
            "weight_matrix": weight_matrix.copy(),
            "smoothing_sigma": float(used_sigma),
            "jitter_pixels": int(self.jitter_pixels if jitter_pixels is None else jitter_pixels),
            "jittered_indices": tuple(int(i) for i in jittered_indices),
        }
        if return_reference_signal:
            return f, s, params, clean_signal
        return f, s, params

    def generate_3compartment_paper(
        self,
        mixing_time: float = None,
        exchange_rates: tuple = None,
        noise_sigma: float = None,
        noise_sigma_range: tuple = (0.005, 0.015),
        noise_model: str = "rician",
        normalize: bool = None,
        jitter_pixels: int = None,
        smoothing_sigma: float = None,
        return_reference_signal: bool = False,
    ) -> tuple:
        """Generate one paper-style 3-compartment DEXSY sample with nine peaks."""
        mixing_time = self._sample_mixing_time(mixing_time)
        noise_sigma = self._sample_noise_sigma(noise_sigma, noise_sigma_range)

        diffusions = self._sample_compartment_diffusions(("intracellular", "extracellular", "fast"))
        volume_fractions = np.random.dirichlet(np.array([2.0, 2.0, 2.0], dtype=np.float64))

        if exchange_rates is None:
            rate_01 = self._sample_exchange_rate()
            rate_02 = self._sample_exchange_rate()
            rate_12 = self._sample_exchange_rate()
        else:
            rate_01, rate_02, rate_12 = exchange_rates

        exchange_rate_matrix = np.zeros((3, 3), dtype=np.float64)
        exchange_rate_matrix[0, 1] = exchange_rate_matrix[1, 0] = rate_01
        exchange_rate_matrix[0, 2] = exchange_rate_matrix[2, 0] = rate_02
        exchange_rate_matrix[1, 2] = exchange_rate_matrix[2, 1] = rate_12

        f, weight_matrix, exchange_probs, used_sigma, exchange_scale, jittered_indices = self._generate_paper_spectrum(
            diffusions=diffusions,
            volume_fractions=volume_fractions,
            exchange_rates=exchange_rate_matrix,
            mixing_time=mixing_time,
            jitter_pixels=jitter_pixels,
            smoothing_sigma=smoothing_sigma,
        )
        clean_signal = self.compute_signal(f, noise_sigma=0.0, normalize=normalize, noise_model=None)
        s = self.compute_signal(f, noise_sigma=noise_sigma, normalize=normalize, noise_model=noise_model)

        params = {
            "n_compartments": 3,
            "mixing_time": float(mixing_time),
            "noise_sigma": float(noise_sigma),
            "baseline_snr": float(1.0 / max(noise_sigma, 1e-12)),
            "diffusions": diffusions.tolist(),
            "volume_fractions": volume_fractions.tolist(),
            "exchange_rates": {
                "0-1": float(rate_01),
                "0-2": float(rate_02),
                "1-2": float(rate_12),
            },
            "exchange_probabilities": {
                "0-1": float(exchange_probs[0, 1]),
                "0-2": float(exchange_probs[0, 2]),
                "1-2": float(exchange_probs[1, 2]),
            },
            "exchange_peak_masses": self._pair_exchange_masses(weight_matrix),
            "exchange_probability_scale": float(exchange_scale),
            "weight_matrix": weight_matrix.copy(),
            "smoothing_sigma": float(used_sigma),
            "jitter_pixels": int(self.jitter_pixels if jitter_pixels is None else jitter_pixels),
            "jittered_indices": tuple(int(i) for i in jittered_indices),
        }
        if return_reference_signal:
            return f, s, params, clean_signal
        return f, s, params

    def generate_2compartment_sparse(self, **kwargs) -> tuple:
        """
        Backwards-compatible wrapper.

        The name is retained so existing notebooks do not break, but the output
        now follows the paper-style broadened 2C simulation rather than the old
        four-delta sparse generator.
        """
        return self.generate_2compartment_paper(**kwargs)

    def generate_sample(self, n_compartments: int = 2, **kwargs) -> tuple:
        """Generate one 2C or 3C paper-style sample."""
        if n_compartments == 2:
            return self.generate_2compartment_paper(**kwargs)
        if n_compartments == 3:
            return self.generate_3compartment_paper(**kwargs)
        raise ValueError(f"Unsupported number of compartments: {n_compartments}")

    def generate_batch(
        self,
        n_samples: int,
        noise_sigma: float = None,
        noise_sigma_range: tuple = (0.005, 0.015),
        n_compartments: int = 2,
        return_reference_signal: bool = False,
        **kwargs,
    ) -> tuple:
        """
        Generate a batch of paper-style DEXSY samples.

        Args:
            n_samples: Number of samples to generate.
            noise_sigma: Optional fixed noise level.
            noise_sigma_range: Paper-style continuous noise range.
            n_compartments: 2 or 3.
            **kwargs: Passed to the selected generator.
        """
        F = np.zeros((n_samples, self.n_d, self.n_d), dtype=np.float32)
        S = np.zeros((n_samples, self.n_b, self.n_b), dtype=np.float32)
        S_clean = np.zeros((n_samples, self.n_b, self.n_b), dtype=np.float32) if return_reference_signal else None
        params_list = []

        for i in range(n_samples):
            generated = self.generate_sample(
                n_compartments=n_compartments,
                noise_sigma=noise_sigma,
                noise_sigma_range=noise_sigma_range,
                return_reference_signal=return_reference_signal,
                **kwargs,
            )
            if return_reference_signal:
                f, s, params, clean_signal = generated
                S_clean[i] = clean_signal
            else:
                f, s, params = generated
            F[i] = f
            S[i] = s
            params_list.append(params)

        if return_reference_signal:
            return F, S, params_list, S_clean
        return F, S, params_list

    def compute_ilt_nnls(
        self,
        s: np.ndarray,
        alpha: float = 0.02,
        *,
        post_sharpen: bool = False,
        sharpen_sigma: float = 0.85,
        sharpen_strength: float = 0.38,
        renorm: bool = True,
    ) -> np.ndarray:
        """
        Compute a 2D ILT baseline using non-negative (Tikhonov-regularised) least squares.

        Optional mild unsharp masking approximates the Gaussian deconvolution stage
        described for the Python 2D ILT pipeline in the dissertation (reduces
        over-smoothed off-diagonal leakage in DEI).
        """
        K = self.kernel_matrix
        s_flat = s.flatten().astype(np.float64)

        K_reg = np.vstack([K.astype(np.float64), alpha * np.eye(self.n_d * self.n_d)])
        s_reg = np.concatenate([s_flat, np.zeros(self.n_d * self.n_d)])
        try:
            from scipy.optimize import nnls

            f_flat, _ = nnls(K_reg, s_reg)
        except ImportError:  # pragma: no cover - exercised only in lighter environments.
            # Projected gradient NNLS fallback so evaluation still works without scipy.
            f_flat = np.zeros(K_reg.shape[1], dtype=np.float64)
            spectral_norm = np.linalg.norm(K_reg, ord=2)
            step = 1.0 / (spectral_norm ** 2 + 1e-8)
            for _ in range(300):
                gradient = K_reg.T @ (K_reg @ f_flat - s_reg)
                updated = np.maximum(f_flat - step * gradient, 0.0)
                if np.linalg.norm(updated - f_flat) < 1e-8 * (np.linalg.norm(f_flat) + 1.0):
                    f_flat = updated
                    break
                f_flat = updated
        f_est = f_flat.reshape(self.n_d, self.n_d)
        if post_sharpen:
            f_est = _unsharp_mask_nonnegative(f_est, sharpen_sigma, sharpen_strength)
        if renorm:
            f_est = np.clip(f_est, 0.0, None)
            f_est /= f_est.sum() + 1e-12
        return f_est.astype(np.float64)

    def generate_2c_validation_spectrum(
        self,
        diffusions: np.ndarray,
        volume_fractions: np.ndarray,
        exchange_rate: float,
        mixing_time: float,
        *,
        jitter_pixels: int = 0,
        smoothing_sigma: float | None = None,
        normalize: bool | None = True,
    ) -> tuple:
        """
        Build a noise-free 2-compartment spectrum and signal for Section 3.1-style checks.

        Uses the same exchange and projection path as ``generate_2compartment_paper`` so
        validation matches the training forward model (no duplicate notebook physics).
        """
        diffusions = np.asarray(diffusions, dtype=np.float64).reshape(2)
        volume_fractions = np.asarray(volume_fractions, dtype=np.float64).reshape(2)
        pair_indices = tuple(int(self._nearest_diffusion_index(d)) for d in diffusions)
        exchange_rates = np.zeros((2, 2), dtype=np.float64)
        exchange_rates[0, 1] = exchange_rate
        exchange_rates[1, 0] = exchange_rate

        f, weight_matrix, exchange_probs, used_sigma, exchange_scale, jittered_indices = self._generate_paper_spectrum(
            diffusions=diffusions,
            volume_fractions=volume_fractions,
            exchange_rates=exchange_rates,
            mixing_time=mixing_time,
            jitter_pixels=jitter_pixels,
            smoothing_sigma=smoothing_sigma,
        )
        clean_signal = self.compute_signal(f, noise_sigma=0.0, normalize=normalize, noise_model=None)
        theoretical_peak_masses = {
            "diagonal": float(weight_matrix[0, 0] + weight_matrix[1, 1]),
            "off_diagonal": float(weight_matrix[0, 1] + weight_matrix[1, 0]),
        }
        params = {
            "n_compartments": 2,
            "mixing_time": float(mixing_time),
            "diffusions": diffusions.tolist(),
            "volume_fractions": volume_fractions.tolist(),
            "exchange_rates": {"0-1": float(exchange_rate)},
            "exchange_probabilities": {"0-1": float(exchange_probs[0, 1])},
            "smoothing_sigma": float(used_sigma),
            "exchange_probability_scale": float(exchange_scale),
            "weight_matrix": weight_matrix.copy(),
            "pair_indices": pair_indices,
            "expected_peak_centres": {
                "diagonal": ((pair_indices[0], pair_indices[0]), (pair_indices[1], pair_indices[1])),
                "off_diagonal": ((pair_indices[0], pair_indices[1]), (pair_indices[1], pair_indices[0])),
            },
            "theoretical_peak_masses": theoretical_peak_masses,
            "theoretical_dei": compute_weight_matrix_dei(weight_matrix),
            "jittered_indices": tuple(int(i) for i in jittered_indices),
        }
        return f, clean_signal, params


def compute_dei(f: np.ndarray, diagonal_band_width: int = 5) -> float:
    """
    Compute Diffusion Exchange Index for broadened spectra.

    Mass with |i - j| <= ``diagonal_band_width`` is treated as lying in the
    D1 \\approx D2 (self-diffusion) band on the discrete grid; the rest counts
    toward exchange. A width of ~5 matches the spatial spread of NNLS-based
    2D ILT on this 64x64 grid so that ground-truth and inverted spectra are
    compared on the same footing (a width of 2 is often too narrow once
    broadening or ILT blur is present).
    """
    n = f.shape[0]
    ii, jj = np.indices((n, n))
    diagonal_mask = np.abs(ii - jj) <= diagonal_band_width
    diag_sum = float(f[diagonal_mask].sum())
    off_diag_sum = float(f[~diagonal_mask].sum())
    return off_diag_sum / (diag_sum + 1e-10)


if __name__ == "__main__":
    fm = ForwardModel2D()
    print("Forward Model Parameters:")
    print(f"  D range: {fm.D1[0]:.2e} to {fm.D1[-1]:.2e}")
    print(f"  G range: {fm.G1[0]:.2e} to {fm.G1[-1]:.2e}")
    print(f"  b range: {fm.b1[0]:.2e} to {fm.b1[-1]:.2e}")
    print()

    for n_compartments in (2, 3):
        f, s, params = fm.generate_sample(n_compartments=n_compartments)
        print(f"{n_compartments}C sample:")
        print(f"  spectrum shape: {f.shape}, signal shape: {s.shape}")
        print(f"  signal range: [{s.min():.4f}, {s.max():.4f}]")
        print(f"  DEI: {compute_dei(f):.4f}")
        print(f"  mixing time: {params['mixing_time']:.4f}s")
        print(f"  baseline SNR: {params['baseline_snr']:.2f}")
        print()

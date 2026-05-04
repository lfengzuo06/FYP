"""
N-Compartment DEXSY Forward Model.

Implements the general N-compartment forward model based on the mathematical
formulation:

Inputs:
    N: Number of compartments
    {phi_i}_{i=1}^{N}: Volume fractions
    {d_i}_{i=1}^{N}: Diffusion coefficients
    {kappa_ij}_{i<j}: Exchange rates between compartments
    t_m: Mixing time
    D: Diffusion grid
    G1, G2: Gradient strengths
    gamma, delta, DELTA: NMR parameters
    SNR: Signal-to-noise ratio

Steps:
    1. Compute b-values: b_1(m), b_2(n)
    2. Construct exponent matrices: A_1(m,u), A_2(n,v)
    3. Construct exchange probabilities: p_ij
    4. Compute unscaled weights: w_hat_ij
    5. Compute scaling factor: lambda
    6. Obtain final weight matrix W
    7. Project to diffusion grid
    8. Compute noiseless signal: S = A_1 f A_2^T
    9. Add Rician noise
    10. Normalize
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Literal


def _gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing."""
    if sigma <= 0:
        return image

    from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
    return _scipy_gaussian_filter(image, sigma=sigma, mode="nearest")


class ForwardModelNC:
    """
    N-compartment DEXSY forward model.

    Implements the general N-compartment forward model where N can be any
    positive integer. The model follows the mathematical formulation:
        - b-values are computed from gradient strengths
        - Exchange probabilities derived from exchange rates and mixing time
        - Weight matrix constructed with scaling factor for diagonal dominance
        - Signal computed via matrix multiplication with exponent matrices
    """

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
        mixing_time_range: tuple[float, float] = (0.015, 0.300),
        exchange_rate_range: tuple[float, float] = (0.1, 30.0),
        jitter_pixels: int = 1,
        smoothing_sigma_range: tuple[float, float] = (0.65, 1.15),
        min_index_separation: int = 4,
        spectral_broadening_mode: str = "directional",
        alpha: float = 0.5,
        epsilon: float = 1e-10,
    ):
        """
        Initialize the N-compartment DEXSY forward model.

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
            normalize_signals: If True, normalize by the noisy b=0 entry.
            mixing_time_range: Mixing-time sampling range in seconds.
            exchange_rate_range: Exchange-rate sampling range in s^-1.
            jitter_pixels: Peak jitter in pixels.
            smoothing_sigma_range: Gaussian broadening sigma range (pixels).
            min_index_separation: Minimum projected grid separation between
                compartment diffusivities.
            spectral_broadening_mode: "directional" or "isotropic".
            alpha: Scaling factor parameter for diagonal dominance constraint.
                Default 0.5 ensures w_ii >= 0.5 * phi_i.
            epsilon: Small constant for numerical stability.
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
        self.jitter_pixels = jitter_pixels
        self.smoothing_sigma_range = smoothing_sigma_range
        self.min_index_separation = min_index_separation
        self.spectral_broadening_mode = spectral_broadening_mode
        self.alpha = alpha
        self.epsilon = epsilon

        # Compartment diffusion ranges
        self.compartment_ranges = {
            "intracellular": (5e-12, 3e-11),
            "extracellular": (3e-11, 5e-9),
            "fast": (5e-9, 5e-8),
        }

        # Build diffusion grid (log-spaced)
        self.D = np.logspace(np.log10(d_min), np.log10(d_max), n_d)

        # Build gradient grids
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

        # Precompute b-values (Step 1)
        coeff = (gamma ** 2) * (delta ** 2) * (DELTA - delta / 3.0)
        self.b1 = (self.G1 ** 2) * coeff
        self.b2 = (self.G2 ** 2) * coeff

        # Exchange rate grid for sampling
        self.exchange_rate_grid = np.logspace(
            np.log10(exchange_rate_range[0]),
            np.log10(exchange_rate_range[1]),
            24,
        )

        # Precompute exponent matrices (Step 2)
        self._compute_exponent_matrices()

        # Compute kernel matrix for physics-informed loss
        self._compute_kernel()

    def _compute_kernel(self):
        """Precompute the exponential kernel tensor."""
        D1G, D2G = np.meshgrid(self.D, self.D, indexing="ij")
        b1G, b2G = np.meshgrid(self.b1, self.b2, indexing="ij")
        kernel = np.exp(-b1G[..., None, None] * D1G) * np.exp(-b2G[..., None, None] * D2G)
        self._kernel = kernel.astype(np.float32)
        self.kernel_matrix = self._kernel.reshape(self.n_b * self.n_b, self.n_d * self.n_d)

    def _compute_exponent_matrices(self):
        """
        Compute the exponent matrices A_1 and A_2.

        Step 2: A_1(m,u) = exp(-b_1(m) * D_u)
                 A_2(n,v) = exp(-b_2(n) * D_v)
        """
        self.A1 = np.exp(-self.b1[:, None] * self.D[None, :])  # (n_b, n_d)
        self.A2 = np.exp(-self.b2[:, None] * self.D[None, :])  # (n_b, n_d)

    def _sample_log_uniform(self, value_range: tuple[float, float]) -> float:
        """Sample one value uniformly in log space."""
        low, high = value_range
        return float(np.exp(np.random.uniform(np.log(low), np.log(high))))

    def _sample_uniform(self, value_range: tuple[float, float]) -> float:
        """Sample one value uniformly in linear space."""
        low, high = value_range
        return float(np.random.uniform(low, high))

    def _sample_mixing_time(self, mixing_time: Optional[float] = None) -> float:
        """Sample mixing time in seconds."""
        if mixing_time is not None:
            return float(mixing_time)
        return self._sample_log_uniform(self.mixing_time_range)

    def _sample_exchange_rate(self, exchange_rate: Optional[float] = None) -> float:
        """Sample exchange rate from a logarithmic grid."""
        if exchange_rate is not None:
            return float(exchange_rate)
        return float(np.random.choice(self.exchange_rate_grid))

    def _nearest_diffusion_index(self, diffusion_value: float) -> int:
        """Project a diffusion value onto the log-spaced reconstruction grid."""
        log_grid = np.log(self.D)
        return int(np.argmin(np.abs(log_grid - np.log(diffusion_value))))

    def _jitter_index(self, idx: int, jitter_pixels: Optional[int] = None) -> int:
        """Apply bounded integer jitter to a grid index."""
        jitter_pixels = self.jitter_pixels if jitter_pixels is None else jitter_pixels
        if jitter_pixels <= 0:
            return idx
        offset = int(np.random.randint(-jitter_pixels, jitter_pixels + 1))
        return int(np.clip(idx + offset, 0, self.n_d - 1))

    def build_weight_matrix(
        self,
        N: int,
        phi: np.ndarray,
        kappa: np.ndarray,
        tm: float,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Build the N-compartment weight matrix.

        Steps 3-6 of the forward model:

        Step 3: p_ij = 1 - exp(-kappa_ij * tm) for i < j
                p_ji = p_ij (symmetric)

        Step 4: w_hat_ij = phi_i * phi_j * p_ij for i != j
                w_hat_ii = 0

        Step 5: lambda = min(1, min_i (alpha * phi_i / (sum_j!=i w_hat_ij + epsilon)))

        Step 6: w_ij = lambda * w_hat_ij for i != j
                w_ii = phi_i - sum_j!=i w_ij

        Args:
            N: Number of compartments.
            phi: Volume fractions of shape (N,).
            kappa: Exchange rate matrix of shape (N, N).
                Only upper triangular (i < j) is used.
            tm: Mixing time in seconds.

        Returns:
            W: Weight matrix of shape (N, N).
            lam: Scaling factor lambda.
            p: Exchange probability matrix of shape (N, N).
        """
        phi = np.asarray(phi, dtype=np.float64)
        kappa = np.asarray(kappa, dtype=np.float64)

        # Step 3: Compute exchange probabilities
        p = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                prob = 1.0 - np.exp(-kappa[i, j] * tm)
                p[i, j] = prob
                p[j, i] = prob

        # Step 4: Compute unscaled off-diagonal weights
        w_hat = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                offdiag_mass = phi[i] * phi[j] * p[i, j]
                w_hat[i, j] = offdiag_mass
                w_hat[j, i] = offdiag_mass

        # Step 5: Compute scaling factor lambda
        row_offdiag = w_hat.sum(axis=1)  # sum over j != i
        valid_rows = row_offdiag > 0

        if np.any(valid_rows):
            # lambda = min(1, min_i (alpha * phi_i / (row_offdiag_i + epsilon)))
            ratios = np.full(N, np.inf, dtype=np.float64)
            ratios[valid_rows] = (self.alpha * phi[valid_rows]) / (row_offdiag[valid_rows] + self.epsilon)
            lam = min(1.0, float(np.min(ratios)))
        else:
            lam = 1.0

        # Step 6: Apply scaling and compute diagonal entries
        w = lam * w_hat
        diag = phi - w.sum(axis=1)
        np.fill_diagonal(w, diag)

        # Ensure non-negative
        w = np.clip(w, 0.0, None)

        return w, lam, p

    def project_to_diffusion_grid(
        self,
        W: np.ndarray,
        D: np.ndarray,
        jitter_indices: Optional[list[int]] = None,
        smoothing_sigma: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        """
        Project compartment weight matrix to 2D diffusion grid.

        Step 7: Project W to the (n_d x n_d) diffusion grid.

        Args:
            W: Weight matrix of shape (N, N).
            D: Compartment diffusion values of shape (N,).
            jitter_indices: Jittered grid indices for each compartment.
            smoothing_sigma: Gaussian smoothing sigma.
            mode: Projection mode - "directional" or "isotropic".

        Returns:
            f: 2D spectrum of shape (n_d, n_d).
        """
        if mode is None:
            mode = self.spectral_broadening_mode
        if smoothing_sigma is None:
            smoothing_sigma = float(np.random.uniform(*self.smoothing_sigma_range))

        N = W.shape[0]
        spectrum = np.zeros((self.n_d, self.n_d), dtype=np.float64)

        # Get or compute jittered indices
        if jitter_indices is None:
            base_indices = [self._nearest_diffusion_index(d) for d in D]
            jitter_indices = [self._jitter_index(idx, self.jitter_pixels) for idx in base_indices]

        if mode == "directional":
            sigma = float(smoothing_sigma)
            radius = max(1, int(np.ceil(3.0 * sigma)))

            # 1D Gaussian kernel for diagonal
            g1 = self._discrete_gaussian_1d_kernel(sigma, radius)
            # 2D Gaussian kernel for off-diagonal
            g2 = self._discrete_gaussian_2d_kernel(sigma, radius)

            for i in range(N):
                for j in range(N):
                    weight = float(W[i, j])
                    if weight <= 0:
                        continue
                    ia = jitter_indices[i]
                    jb = jitter_indices[j]

                    if ia == jb:
                        # Diagonal: 1D Gaussian along the diagonal
                        for k, gk in enumerate(g1):
                            kk = k - radius
                            ii = ia + kk
                            if 0 <= ii < self.n_d:
                                spectrum[ii, ii] += weight * gk
                    else:
                        # Off-diagonal: 2D Gaussian
                        for di in range(2 * radius + 1):
                            for dj in range(2 * radius + 1):
                                ii = ia + di - radius
                                jj = jb + dj - radius
                                if 0 <= ii < self.n_d and 0 <= jj < self.n_d:
                                    spectrum[ii, jj] += weight * g2[di, dj]
        else:
            # Isotropic mode
            for i in range(N):
                for j in range(N):
                    weight = float(W[i, j])
                    if weight <= 0:
                        continue
                    idx_i = jitter_indices[i]
                    idx_j = jitter_indices[j]
                    spectrum[idx_i, idx_j] += weight

            spectrum = _gaussian_filter(spectrum.astype(np.float32), sigma=smoothing_sigma)

        spectrum = np.clip(spectrum, 0.0, None)
        spectrum /= spectrum.sum() + 1e-12
        return spectrum.astype(np.float32)

    @staticmethod
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

    @staticmethod
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

    def compute_signal(
        self,
        f: np.ndarray,
        noise_sigma: float = 0.0,
        noise_model: str = "rician",
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Compute DEXSY signal from diffusion distribution.

        Step 8: S = A_1 @ f @ A_2^T
        Step 9: Add Rician noise
        Step 10: Normalize by S[0,0]

        Args:
            f: 2D spectrum of shape (n_d, n_d).
            noise_sigma: Standard deviation of noise.
            noise_model: "rician", "gaussian", or "none".
            normalize: Whether to normalize signal.

        Returns:
            S: Signal of shape (n_b, n_b).
        """
        # Step 8: Compute noiseless signal S = A_1 @ f @ A_2^T
        f = f.astype(np.float32)
        S = self.A1 @ f @ self.A2.T  # (n_b, n_b)

        # Step 9: Add noise
        if noise_sigma > 0:
            if noise_model == "rician":
                noise_real = np.random.randn(*S.shape).astype(np.float32) * noise_sigma
                noise_imag = np.random.randn(*S.shape).astype(np.float32) * noise_sigma
                S = np.sqrt((S + noise_real) ** 2 + noise_imag ** 2)
            elif noise_model == "gaussian":
                S = S + np.random.randn(*S.shape).astype(np.float32) * noise_sigma
            elif noise_model == "none":
                pass
            else:
                raise ValueError(f"Unknown noise model: {noise_model}")

        # Step 10: Normalize
        if normalize is None:
            normalize = self.normalize_signals
        if normalize:
            S0 = np.maximum(S[0, 0], 1e-10)
            S = S / S0

        return S

    def generate_ncompartment_sample(
        self,
        N: int,
        phi: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
        kappa: Optional[np.ndarray] = None,
        mixing_time: Optional[float] = None,
        noise_sigma: Optional[float] = None,
        jitter_pixels: Optional[int] = None,
        smoothing_sigma: Optional[float] = None,
        noise_model: str = "rician",
        normalize: Optional[bool] = None,
        return_reference_signal: bool = False,
    ) -> tuple:
        """
        Generate one N-compartment DEXSY sample.

        Args:
            N: Number of compartments (2 or more).
            phi: Volume fractions of shape (N,). If None, sampled from Dirichlet.
            D: Compartment diffusion values of shape (N,). If None, sampled.
            kappa: Exchange rate matrix of shape (N, N). If None, sampled.
            mixing_time: Mixing time in seconds.
            noise_sigma: Noise standard deviation.
            jitter_pixels: Peak jitter in pixels.
            smoothing_sigma: Gaussian broadening sigma.
            noise_model: "rician", "gaussian", or "none".
            normalize: Whether to normalize.
            return_reference_signal: If True, also return clean signal.

        Returns:
            f: 2D spectrum (n_d, n_d).
            S: Signal (n_b, n_b).
            params: Dictionary with ground truth parameters.
            [S_clean]: Optional clean signal if return_reference_signal=True.
        """
        if N < 2:
            raise ValueError("N must be at least 2")

        # Sample mixing time and noise
        mixing_time = self._sample_mixing_time(mixing_time)
        noise_sigma = self._sample_noise_sigma(noise_sigma)

        # Sample volume fractions (Dirichlet)
        if phi is None:
            phi = np.random.dirichlet(np.ones(N) * 2.0)

        # Determine effective jitter_pixels (use instance default if not specified)
        effective_jitter = self.jitter_pixels if jitter_pixels is None else jitter_pixels

        # Sample diffusion values (pass effective jitter for proper separation calculation)
        if D is None:
            D = self._sample_compartment_diffusions(N, jitter_pixels=effective_jitter)

        # Sample exchange rates
        if kappa is None:
            kappa = self._sample_exchange_rate_matrix(N)

        # Build weight matrix (Steps 3-6)
        W, lam, p = self.build_weight_matrix(N, phi, kappa, mixing_time)

        # Project to diffusion grid (Step 7)
        f, used_sigma, jitter_indices = self._project_weight_matrix(
            W=W,
            D=D,
            jitter_pixels=jitter_pixels,
            smoothing_sigma=smoothing_sigma,
        )

        # Compute signals (Steps 8-10)
        S_clean = self.compute_signal(f, noise_sigma=0.0, normalize=normalize, noise_model="none")
        S = self.compute_signal(f, noise_sigma=noise_sigma, normalize=normalize, noise_model=noise_model)

        # Build params dictionary
        params = self._build_params_dict(
            N=N,
            phi=phi,
            D=D,
            kappa=kappa,
            mixing_time=mixing_time,
            noise_sigma=noise_sigma,
            W=W,
            lam=lam,
            p=p,
            used_sigma=used_sigma,
            jitter_indices=jitter_indices,
        )

        if return_reference_signal:
            return f, S, params, S_clean
        return f, S, params

    def _sample_noise_sigma(self, noise_sigma: Optional[float] = None) -> float:
        """Sample noise sigma."""
        if noise_sigma is not None:
            return float(noise_sigma)
        return self._sample_uniform((0.005, 0.015))

    def _sample_compartment_diffusions(
        self,
        N: int,
        jitter_pixels: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample compartment diffusivities while keeping them distinct on the grid.

        For N compartments, we cycle through the three predefined ranges:
        (intracellular, extracellular, fast) and sample within each.
        Uses deterministic spacing as fallback to guarantee separation.
        """
        if jitter_pixels is None:
            jitter_pixels = self.jitter_pixels

        ranges = list(self.compartment_ranges.values())
        n_ranges = len(ranges)
        diffusions = []

        for i in range(N):
            range_idx = i % n_ranges
            d = self._sample_log_uniform(ranges[range_idx])
            diffusions.append(d)

        diffusions = np.array(diffusions, dtype=np.float64)

        # Ensure separation on grid using the ACTUAL jitter_pixels that will be used
        min_separation = max(
            int(self.min_index_separation),
            2 * int(jitter_pixels) + 1
        )
        indices = np.array([self._nearest_diffusion_index(d) for d in diffusions], dtype=int)

        # Try random sampling first (up to 512 attempts)
        for _ in range(512):
            if len(indices) <= 1:
                break
            min_dist = np.min(
                np.abs(indices[:, None] - indices[None, :]) + np.eye(N) * self.n_d
            )
            if min_dist >= min_separation:
                break

            # Resample
            for i in range(N):
                range_idx = i % n_ranges
                diffusions[i] = self._sample_log_uniform(ranges[range_idx])
                indices[i] = self._nearest_diffusion_index(diffusions[i])

        # Fallback: use deterministic log-spaced placement if random fails
        # For high N, we need to ensure minimum spacing across the entire grid
        if len(indices) > 1:
            min_dist = np.min(
                np.abs(indices[:, None] - indices[None, :]) + np.eye(N) * self.n_d
            )
            if min_dist < min_separation:
                # Deterministic fallback: distribute compartments evenly across grid
                # Each compartment needs at least min_separation pixels
                # Total needed: N * min_separation <= n_d for non-overlapping
                # We distribute compartments as evenly as possible
                total_needed = N * min_separation
                if total_needed > self.n_d:
                    # Reduce min_separation proportionally
                    effective_min_sep = self.n_d // N
                    effective_min_sep = max(effective_min_sep, 1)
                else:
                    effective_min_sep = min_separation

                # Place compartments at evenly spaced indices
                # Start from middle of first slot, then every effective_min_sep pixels
                start_offset = effective_min_sep // 2
                for i in range(N):
                    idx = start_offset + i * effective_min_sep
                    idx = min(idx, self.n_d - 1)  # Clamp to valid range
                    diffusions[i] = self.D[idx]
                    indices[i] = idx

        return diffusions

    def _sample_exchange_rate_matrix(self, N: int) -> np.ndarray:
        """Sample a symmetric exchange rate matrix."""
        kappa = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                rate = self._sample_exchange_rate()
                kappa[i, j] = rate
                kappa[j, i] = rate
        return kappa

    def _project_weight_matrix(
        self,
        W: np.ndarray,
        D: np.ndarray,
        jitter_pixels: Optional[int] = None,
        smoothing_sigma: Optional[float] = None,
    ) -> tuple:
        """Project weight matrix to diffusion grid with jitter and smoothing."""
        if jitter_pixels is None:
            jitter_pixels = self.jitter_pixels
        if smoothing_sigma is None:
            smoothing_sigma = float(np.random.uniform(*self.smoothing_sigma_range))

        N = W.shape[0]
        base_indices = [self._nearest_diffusion_index(d) for d in D]
        jitter_indices = [self._jitter_index(idx, jitter_pixels) for idx in base_indices]

        f = self.project_to_diffusion_grid(W, D, jitter_indices, smoothing_sigma)

        return f, float(smoothing_sigma), tuple(jitter_indices)

    def _build_params_dict(
        self,
        N: int,
        phi: np.ndarray,
        D: np.ndarray,
        kappa: np.ndarray,
        mixing_time: float,
        noise_sigma: float,
        W: np.ndarray,
        lam: float,
        p: np.ndarray,
        used_sigma: float,
        jitter_indices: tuple,
    ) -> dict:
        """Build parameters dictionary for output."""
        params = {
            "n_compartments": N,
            "mixing_time": float(mixing_time),
            "noise_sigma": float(noise_sigma),
            "baseline_snr": float(1.0 / max(noise_sigma, 1e-12)),
            "diffusions": D.tolist(),
            "volume_fractions": phi.tolist(),
            "exchange_rates": {},
            "exchange_probabilities": {},
            "lambda": float(lam),
            "smoothing_sigma": float(used_sigma),
            "jitter_indices": jitter_indices,
            "weight_matrix": W.copy(),
        }

        # Add exchange rates and probabilities
        for i in range(N):
            for j in range(i + 1, N):
                key = f"{i}-{j}"
                params["exchange_rates"][key] = float(kappa[i, j])
                params["exchange_probabilities"][key] = float(p[i, j])

        # Compute theoretical DEI
        diag_sum = float(np.trace(W))
        off_diag_sum = float(W.sum() - diag_sum)
        params["theoretical_dei"] = off_diag_sum / (diag_sum + 1e-10)

        return params

    def generate_batch(
        self,
        n_samples: int,
        N: int,
        noise_sigma: Optional[float] = None,
        noise_sigma_range: Optional[tuple[float, float]] = None,
        return_reference_signal: bool = False,
        **kwargs,
    ) -> tuple:
        """
        Generate a batch of N-compartment DEXSY samples.

        Args:
            n_samples: Number of samples.
            N: Number of compartments.
            noise_sigma: Fixed noise level.
            noise_sigma_range: Range for random noise sampling (low, high).
            return_reference_signal: Whether to return clean signals.
            **kwargs: Passed to generate_ncompartment_sample.

        Returns:
            F: Spectra of shape (n_samples, n_d, n_d).
            S: Signals of shape (n_samples, n_b, n_b).
            params_list: List of parameter dicts.
            [S_clean]: Optional clean signals.
        """
        F = np.zeros((n_samples, self.n_d, self.n_d), dtype=np.float32)
        S = np.zeros((n_samples, self.n_b, self.n_b), dtype=np.float32)
        S_clean = (
            np.zeros((n_samples, self.n_b, self.n_b), dtype=np.float32)
            if return_reference_signal
            else None
        )
        params_list = []

        # Sample noise levels if range is provided
        if noise_sigma_range is not None and noise_sigma is None:
            noise_sigmas = np.random.uniform(noise_sigma_range[0], noise_sigma_range[1], n_samples)
        else:
            noise_sigmas = [noise_sigma] * n_samples

        for i in range(n_samples):
            if return_reference_signal:
                f, s, params, clean = self.generate_ncompartment_sample(
                    N=N,
                    noise_sigma=noise_sigmas[i],
                    return_reference_signal=True,
                    **kwargs,
                )
                S_clean[i] = clean
            else:
                f, s, params = self.generate_ncompartment_sample(
                    N=N,
                    noise_sigma=noise_sigmas[i],
                    **kwargs,
                )
            F[i] = f
            S[i] = s
            params_list.append(params)

        if return_reference_signal:
            return F, S, params_list, S_clean
        return F, S, params_list


def compute_nc_weight_matrix_dei(weight_matrix: np.ndarray) -> float:
    """
    Compute DEI from an N-compartment weight matrix.

    DEI = sum(off-diagonal) / sum(diagonal)
    """
    weight_matrix = np.asarray(weight_matrix, dtype=np.float64)
    diag_sum = float(np.trace(weight_matrix))
    off_diag_sum = float(weight_matrix.sum() - diag_sum)
    return off_diag_sum / (diag_sum + 1e-10)


if __name__ == "__main__":
    # Test the N-compartment forward model
    print("Testing N-Compartment Forward Model")
    print("=" * 50)

    fm = ForwardModelNC()

    # Test with different N values
    for N in [2, 3, 4]:
        print(f"\n--- Testing N={N} ---")
        f, S, params = fm.generate_ncompartment_sample(N=N)

        print(f"  Spectrum shape: {f.shape}")
        print(f"  Signal shape: {S.shape}")
        print(f"  Signal range: [{S.min():.4f}, {S.max():.4f}]")
        print(f"  DEI: {params['theoretical_dei']:.4f}")
        print(f"  Lambda: {params['lambda']:.4f}")
        print(f"  Mixing time: {params['mixing_time']:.4f}s")
        print(f"  SNR: {params['baseline_snr']:.2f}")

        print(f"  Diffusions: {[f'{d:.2e}' for d in params['diffusions']]}")
        print(f"  Volume fractions: {[f'{v:.3f}' for v in params['volume_fractions']]}")

        # Show weight matrix
        W = params["weight_matrix"]
        print(f"  Weight matrix:\n{W}")

"""
3-compartment DEXSY forward model with non-Gaussian restricted kernels.

This module implements a Stanisz-style three-pool signal model with:
    E: extracellular Gaussian compartment
    T: restricted ellipsoid/cylinder (axon-like) compartment
    S: restricted sphere (glial/cell-like) compartment

Signal model:
    S(g1, g2) / S0 = sum_{i,j in {E,T,S}} w_ij * K_i(g1) * K_j(g2)
where:
    w_ij = phi_i * [exp(Q * t_m)]_ij
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.linalg import expm as _scipy_expm
except Exception:  # pragma: no cover - fallback used when scipy is unavailable.
    _scipy_expm = None


COMPARTMENTS_3C = ("E", "T", "S")
PATHWAYS_3C = tuple(f"{i}{j}" for i in COMPARTMENTS_3C for j in COMPARTMENTS_3C)

GRID_PROFILES_3C_NG = {
    16: {
        "n_b": 16,
        "g_max": 1.5,
        "delta": 0.003,
        "DELTA": 0.01,
        "gamma": 267.52e6,
        "gradient_spacing": "linear",
        "normalize_signals": True,
        "n_restrict_terms": 500,
        "mixing_time_range": (0.015, 0.300),
        "series_epsilon": 1e-24,
        "small_x_threshold": 1e-8,
    },
    64: {
        "n_b": 64,
        "g_max": 1.5,
        "delta": 0.003,
        "DELTA": 0.01,
        "gamma": 267.52e6,
        "gradient_spacing": "linear",
        "normalize_signals": True,
        "n_restrict_terms": 500,
        "mixing_time_range": (0.015, 0.300),
        "series_epsilon": 1e-24,
        "small_x_threshold": 1e-8,
    },
}


def create_forward_model_3c_nongaussian(
    n_b: int = 16,
    profile: int | None = None,
    **kwargs,
) -> "ForwardModel3CNonGaussian":
    """
    Factory for 3C non-Gaussian forward model with 16/64 profile support.

    If ``profile`` is None and ``n_b`` matches a known profile (16 or 64),
    that profile is auto-selected.
    """
    if profile is None and n_b in GRID_PROFILES_3C_NG:
        profile = n_b

    if profile is not None:
        profile_size = int(profile)
        if profile_size not in GRID_PROFILES_3C_NG:
            raise ValueError(f"Unknown profile: {profile_size}. Use 16 or 64.")
        config = GRID_PROFILES_3C_NG[profile_size].copy()
        config.update(kwargs)
        return ForwardModel3CNonGaussian(**config)

    return ForwardModel3CNonGaussian(n_b=n_b, **kwargs)


class ForwardModel3CNonGaussian:
    """Stanisz-style 3C forward model for DEXSY with restricted kernels."""

    def __init__(
        self,
        n_b: int = 16,
        g_max: float = 1.5,
        delta: float = 0.003,
        DELTA: float = 0.01,
        gamma: float = 267.52e6,
        gradient_spacing: str = "linear",
        normalize_signals: bool = True,
        n_restrict_terms: int = 500,
        mixing_time_range: tuple[float, float] = (0.015, 0.300),
        series_epsilon: float = 1e-24,
        small_x_threshold: float = 1e-8,
    ):
        """
        Args:
            n_b: Number of gradient/b-value points along each encoding axis.
            g_max: Maximum gradient strength in T/m.
            delta: Pulse duration (s).
            DELTA: Diffusion time (s).
            gamma: Gyromagnetic ratio (rad/s/T).
            gradient_spacing: "linear" or "log".
            normalize_signals: If True, divide output by noisy/clean S(0,0).
            n_restrict_terms: Number of terms in restricted series summation.
            mixing_time_range: Default random sampling range for t_m if needed.
            series_epsilon: Numerical stabilizer for restricted denominator.
            small_x_threshold: Threshold for q*l -> 0 handling.
        """
        self.n_b = int(n_b)
        self.g_max = float(g_max)
        self.delta = float(delta)
        self.DELTA = float(DELTA)
        self.gamma = float(gamma)
        self.gradient_spacing = gradient_spacing
        self.normalize_signals = bool(normalize_signals)
        self.n_restrict_terms = int(n_restrict_terms)
        self.mixing_time_range = mixing_time_range
        self.series_epsilon = float(series_epsilon)
        self.small_x_threshold = float(small_x_threshold)

        if self.n_restrict_terms < 1:
            raise ValueError("n_restrict_terms must be >= 1")

        if gradient_spacing == "linear":
            self.G1 = np.linspace(0.0, self.g_max, self.n_b, dtype=np.float64)
            self.G2 = np.linspace(0.0, self.g_max, self.n_b, dtype=np.float64)
        elif gradient_spacing == "log":
            if self.n_b < 2:
                raise ValueError("n_b must be >= 2 for log gradient spacing.")
            positive_g = np.logspace(
                np.log10(max(self.g_max / 1e4, 1e-8)),
                np.log10(self.g_max),
                self.n_b - 1,
                dtype=np.float64,
            )
            self.G1 = np.concatenate(([0.0], positive_g))
            self.G2 = np.concatenate(([0.0], positive_g))
        else:
            raise ValueError(f"Unknown gradient_spacing: {gradient_spacing}")

        self.b1 = self.compute_b_values(self.G1, self.gamma, self.delta, self.DELTA)
        self.b2 = self.compute_b_values(self.G2, self.gamma, self.delta, self.DELTA)

    @staticmethod
    def compute_b_values(g: np.ndarray, gamma: float, delta: float, DELTA: float) -> np.ndarray:
        """PGSE b-value formula: b = gamma^2 * g^2 * delta^2 * (DELTA - delta/3)."""
        coeff = (gamma ** 2) * (delta ** 2) * (DELTA - delta / 3.0)
        return (np.asarray(g, dtype=np.float64) ** 2) * coeff

    @staticmethod
    def _validate_phi(phi: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi, dtype=np.float64).reshape(-1)
        if phi.shape[0] != 3:
            raise ValueError("phi must have shape (3,) in E/T/S order.")
        if np.any(phi < 0):
            raise ValueError("phi entries must be non-negative.")
        s = float(phi.sum())
        if s <= 0:
            raise ValueError("phi sum must be positive.")
        return phi / s

    @staticmethod
    def build_generator(
        k_et: float,
        k_te: float,
        k_es: float,
        k_se: float,
        k_ts: float = 0.0,
        k_st: float = 0.0,
    ) -> np.ndarray:
        """
        Build 3-state CTMC generator Q in E/T/S order.

        Q =
            [[-(kET+kES),  kET,         kES      ],
             [kTE,         -(kTE+kTS),  kTS      ],
             [kSE,          kST,        -(kSE+kST)]]
        """
        k_et = float(k_et)
        k_te = float(k_te)
        k_es = float(k_es)
        k_se = float(k_se)
        k_ts = float(k_ts)
        k_st = float(k_st)

        rates = np.array([k_et, k_te, k_es, k_se, k_ts, k_st], dtype=np.float64)
        if np.any(rates < 0):
            raise ValueError("Exchange rates must be non-negative.")

        q = np.array(
            [
                [-(k_et + k_es), k_et, k_es],
                [k_te, -(k_te + k_ts), k_ts],
                [k_se, k_st, -(k_se + k_st)],
            ],
            dtype=np.float64,
        )
        return q

    @classmethod
    def build_generator_from_permeability(
        cls,
        phi: np.ndarray,
        sphere_diameter: float,
        sphere_permeability: float,
        axon_surface_to_volume: float,
        axon_permeability: float,
        k_ts: float = 0.0,
        k_st: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """
        Build Q from permeability + detailed balance (Stanisz-style mapping).

        Sphere:
            k_SE = 6 * P_S / a_S

        Axon/ellipsoid:
            k_TE = P_T * (A_T / V_T)

        Detailed balance:
            phi_S k_SE = phi_E k_ES
            phi_T k_TE = phi_E k_ET
        """
        phi_n = cls._validate_phi(phi)
        phi_e, phi_t, phi_s = float(phi_n[0]), float(phi_n[1]), float(phi_n[2])
        if phi_e <= 0:
            raise ValueError("phi_E must be > 0 for detailed-balance back-rates.")
        if sphere_diameter <= 0:
            raise ValueError("sphere_diameter must be positive.")
        if sphere_permeability < 0 or axon_permeability < 0 or axon_surface_to_volume < 0:
            raise ValueError("Permeability and surface/volume terms must be non-negative.")

        k_se = 6.0 * float(sphere_permeability) / float(sphere_diameter)
        k_te = float(axon_permeability) * float(axon_surface_to_volume)
        k_es = (phi_s / phi_e) * k_se
        k_et = (phi_t / phi_e) * k_te

        q = cls.build_generator(
            k_et=k_et,
            k_te=k_te,
            k_es=k_es,
            k_se=k_se,
            k_ts=k_ts,
            k_st=k_st,
        )
        rates = {
            "k_et": float(k_et),
            "k_te": float(k_te),
            "k_es": float(k_es),
            "k_se": float(k_se),
            "k_ts": float(k_ts),
            "k_st": float(k_st),
            "k_ET": float(k_et),
            "k_TE": float(k_te),
            "k_ES": float(k_es),
            "k_SE": float(k_se),
            "k_TS": float(k_ts),
            "k_ST": float(k_st),
        }
        return q, rates

    def _matrix_exponential(self, q: np.ndarray, mixing_time: float) -> np.ndarray:
        """Compute P(tm) = exp(Q * tm) with scipy or eig fallback."""
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (3, 3):
            raise ValueError("q must have shape (3, 3).")
        if mixing_time < 0:
            raise ValueError("mixing_time must be non-negative.")

        a = q * float(mixing_time)
        if _scipy_expm is not None:
            p = _scipy_expm(a)
        else:  # pragma: no cover - only used if scipy is unavailable.
            vals, vecs = np.linalg.eig(a)
            p = vecs @ np.diag(np.exp(vals)) @ np.linalg.inv(vecs)

        p = np.real_if_close(p, tol=1e5).astype(np.float64)
        p = np.clip(p, 0.0, None)
        row_sums = p.sum(axis=1, keepdims=True)
        zero_rows = row_sums[:, 0] <= 0
        if np.any(zero_rows):
            p[zero_rows] = np.eye(3, dtype=np.float64)[zero_rows]
            row_sums = p.sum(axis=1, keepdims=True)
        p = p / row_sums
        return p

    def _restricted_kernel(self, g: np.ndarray, restricted_length: float, diffusivity: float) -> np.ndarray:
        """
        Restricted attenuation kernel K_restrict(g, l, D, DELTA, delta).

        K =
            2(1-cos(ql)) / (ql)^2
            + 4(ql)^2 * sum_{n=1}^N
                exp(-n^2*pi^2*D*DELTA/l^2) *
                [1 - (-1)^n cos(ql)] / [((ql)^2 - (n*pi)^2)^2]
        """
        if restricted_length <= 0:
            raise ValueError("restricted_length must be positive.")
        if diffusivity <= 0:
            raise ValueError("diffusivity must be positive.")

        g = np.asarray(g, dtype=np.float64)
        q = self.gamma * g * self.delta
        x = q * float(restricted_length)
        x2 = x ** 2

        first = np.ones_like(x, dtype=np.float64)
        nz = np.abs(x) > self.small_x_threshold
        first[nz] = 2.0 * (1.0 - np.cos(x[nz])) / x2[nz]

        n = np.arange(1, self.n_restrict_terms + 1, dtype=np.float64)
        n_pi = n * np.pi
        n_pi2 = n_pi ** 2

        decay = np.exp(-(n_pi2 * diffusivity * self.DELTA) / (restricted_length ** 2))
        parity = np.where((n.astype(np.int64) % 2) == 0, 1.0, -1.0)  # (-1)^n
        numer = 1.0 - parity[None, :] * np.cos(x)[:, None]
        denom = (x2[:, None] - n_pi2[None, :]) ** 2 + self.series_epsilon
        series = np.sum(decay[None, :] * numer / denom, axis=1)

        kernel = first + 4.0 * x2 * series
        kernel = np.clip(kernel, 0.0, 1.0)
        kernel[~nz] = 1.0
        return kernel

    def compartment_kernels(
        self,
        g: np.ndarray,
        extracellular_diffusivity: float,
        intracellular_diffusivity: float,
        axon_restricted_length: float,
        sphere_radius: float,
    ) -> dict[str, np.ndarray]:
        """Compute K_E, K_T, K_S over a gradient vector."""
        g = np.asarray(g, dtype=np.float64)
        b = self.compute_b_values(g, self.gamma, self.delta, self.DELTA)

        k_e = np.exp(-b * float(extracellular_diffusivity))
        k_t = self._restricted_kernel(
            g=g,
            restricted_length=float(axon_restricted_length),
            diffusivity=float(intracellular_diffusivity),
        )
        k_s = self._restricted_kernel(
            g=g,
            restricted_length=float(sphere_radius),
            diffusivity=float(intracellular_diffusivity),
        )
        return {"E": k_e, "T": k_t, "S": k_s}

    def compute_weight_matrix(
        self,
        phi: np.ndarray,
        q: np.ndarray,
        mixing_time: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute W with:
            P(tm) = exp(Q * tm)
            W_ij = phi_i * P_ij(tm)
        """
        phi_n = self._validate_phi(phi)
        p = self._matrix_exponential(q=q, mixing_time=mixing_time)
        w = phi_n[:, None] * p
        return w, p, phi_n

    @staticmethod
    def compute_dei_from_weight_matrix(weight_matrix: np.ndarray) -> float:
        """
        Compute DEI from a 3x3 pathway weight matrix:
            DEI = sum(off-diagonal) / sum(diagonal)
        """
        w = np.asarray(weight_matrix, dtype=np.float64)
        if w.shape != (3, 3):
            raise ValueError("weight_matrix must have shape (3, 3).")
        diag_sum = float(np.trace(w))
        off_diag_sum = float(w.sum() - diag_sum)
        return off_diag_sum / (diag_sum + 1e-12)

    @staticmethod
    def compute_dei_from_pathway_weights(pathway_weights: dict[str, float]) -> float:
        """
        Compute DEI from pathway dictionary with keys:
            EE, ET, ES, TE, TT, TS, SE, ST, SS
        """
        required = set(PATHWAYS_3C)
        keys = set(pathway_weights.keys())
        if required - keys:
            missing = sorted(required - keys)
            raise ValueError(f"Missing pathway keys for DEI: {missing}")
        diag_sum = float(pathway_weights["EE"] + pathway_weights["TT"] + pathway_weights["SS"])
        off_diag_sum = float(sum(pathway_weights[k] for k in PATHWAYS_3C if k not in ("EE", "TT", "SS")))
        return off_diag_sum / (diag_sum + 1e-12)

    @staticmethod
    def add_rician_noise(
        signal: np.ndarray,
        noise_sigma: float,
        normalize: bool = True,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Add Rician noise to a DEXSY signal matrix.

        Args:
            signal: Clean signal array of shape (n_b, n_b).
            noise_sigma: Noise standard deviation.
            normalize: If True, normalize by noisy S(0,0).
            rng: Optional NumPy random generator.
        """
        s = np.asarray(signal, dtype=np.float64)
        if s.ndim != 2:
            raise ValueError("signal must be a 2D array.")
        if noise_sigma < 0:
            raise ValueError("noise_sigma must be non-negative.")

        if noise_sigma == 0:
            out = s.copy()
        else:
            if rng is None:
                rng = np.random.default_rng()
            noise_real = rng.standard_normal(size=s.shape) * float(noise_sigma)
            noise_imag = rng.standard_normal(size=s.shape) * float(noise_sigma)
            out = np.sqrt((s + noise_real) ** 2 + noise_imag ** 2)

        if normalize:
            s0 = max(float(out[0, 0]), 1e-12)
            out = out / s0
        return out.astype(np.float64)

    @staticmethod
    def _sample_uniform(rng: np.random.Generator, value_range: tuple[float, float]) -> float:
        low, high = value_range
        return float(rng.uniform(low, high))

    @staticmethod
    def _sample_log_uniform(rng: np.random.Generator, value_range: tuple[float, float]) -> float:
        low, high = value_range
        if low <= 0 or high <= 0:
            raise ValueError("Log-uniform sampling range must be strictly positive.")
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))

    def compute_signal(
        self,
        phi: np.ndarray,
        mixing_time: float,
        extracellular_diffusivity: float,
        intracellular_diffusivity: float,
        axon_restricted_length: float,
        sphere_radius: float,
        q: np.ndarray | None = None,
        rates: dict[str, float] | None = None,
        normalize: bool | None = None,
        return_pathway_signals: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """
        Compute 3C non-Gaussian DEXSY signal on the current (g1, g2) grid.

        Provide either:
            - q (3x3 generator), or
            - rates dict with keys k_et, k_te, k_es, k_se and optional k_ts, k_st.
        If neither is provided, no-exchange Q=0 is used.
        """
        if q is None:
            if rates is None:
                q = np.zeros((3, 3), dtype=np.float64)
            else:
                def _rate(name_lower: str, name_upper: str, default: float | None = None) -> float:
                    if name_lower in rates:
                        return float(rates[name_lower])
                    if name_upper in rates:
                        return float(rates[name_upper])
                    if default is not None:
                        return float(default)
                    raise KeyError(
                        f"Missing exchange rate '{name_lower}' (or '{name_upper}') in rates dict."
                    )

                q = self.build_generator(
                    k_et=_rate("k_et", "k_ET"),
                    k_te=_rate("k_te", "k_TE"),
                    k_es=_rate("k_es", "k_ES"),
                    k_se=_rate("k_se", "k_SE"),
                    k_ts=_rate("k_ts", "k_TS", 0.0),
                    k_st=_rate("k_st", "k_ST", 0.0),
                )

        w, p, phi_n = self.compute_weight_matrix(phi=phi, q=q, mixing_time=mixing_time)

        kernels_1 = self.compartment_kernels(
            g=self.G1,
            extracellular_diffusivity=extracellular_diffusivity,
            intracellular_diffusivity=intracellular_diffusivity,
            axon_restricted_length=axon_restricted_length,
            sphere_radius=sphere_radius,
        )
        kernels_2 = self.compartment_kernels(
            g=self.G2,
            extracellular_diffusivity=extracellular_diffusivity,
            intracellular_diffusivity=intracellular_diffusivity,
            axon_restricted_length=axon_restricted_length,
            sphere_radius=sphere_radius,
        )

        k1 = np.stack([kernels_1[c] for c in COMPARTMENTS_3C], axis=1)  # (n_b, 3)
        k2 = np.stack([kernels_2[c] for c in COMPARTMENTS_3C], axis=1)  # (n_b, 3)
        signal = k1 @ w @ k2.T
        signal = np.clip(signal, 0.0, None)

        if normalize is None:
            normalize = self.normalize_signals
        if normalize:
            s0 = max(float(signal[0, 0]), 1e-12)
            signal = signal / s0

        pathway_weights = {
            pathway: float(w[i, j])
            for i, ci in enumerate(COMPARTMENTS_3C)
            for j, cj in enumerate(COMPARTMENTS_3C)
            for pathway in [f"{ci}{cj}"]
        }

        details = {
            "compartments": COMPARTMENTS_3C,
            "pathways": PATHWAYS_3C,
            "phi": phi_n.copy(),
            "mixing_time": float(mixing_time),
            "generator_q": np.asarray(q, dtype=np.float64).copy(),
            "transition_matrix": p.copy(),
            "weight_matrix": w.copy(),
            "pathway_weights": pathway_weights,
            "b1": self.b1.copy(),
            "b2": self.b2.copy(),
            "g1": self.G1.copy(),
            "g2": self.G2.copy(),
            "model_parameters": {
                "D_E": float(extracellular_diffusivity),
                "D_I": float(intracellular_diffusivity),
                "l_T": float(axon_restricted_length),
                "R_S": float(sphere_radius),
                "n_restrict_terms": int(self.n_restrict_terms),
            },
        }

        if return_pathway_signals:
            pathway_signals: dict[str, np.ndarray] = {}
            for i, ci in enumerate(COMPARTMENTS_3C):
                for j, cj in enumerate(COMPARTMENTS_3C):
                    pathway = f"{ci}{cj}"
                    pathway_signals[pathway] = (
                        w[i, j] * np.outer(kernels_1[ci], kernels_2[cj])
                    ).astype(np.float64)
            details["pathway_signals"] = pathway_signals

        return signal.astype(np.float64), details

    def sample_dataset(
        self,
        n_samples: int,
        *,
        noise_sigma: float | None = None,
        noise_sigma_range: tuple[float, float] = (0.005, 0.015),
        mixing_time: float | None = None,
        mixing_time_range: tuple[float, float] | None = None,
        phi_alpha: tuple[float, float, float] = (2.0, 2.0, 2.0),
        extracellular_diffusivity_range: tuple[float, float] = (1.0e-9, 2.5e-9),
        intracellular_diffusivity_range: tuple[float, float] = (0.4e-9, 1.2e-9),
        axon_restricted_length_range: tuple[float, float] = (0.5e-6, 2.0e-6),
        sphere_radius_range: tuple[float, float] = (1.0e-6, 6.0e-6),
        use_permeability_rates: bool = True,
        sphere_permeability_range: tuple[float, float] = (5e-6, 3e-5),
        axon_permeability_range: tuple[float, float] = (2e-7, 2e-6),
        axon_surface_to_volume_range: tuple[float, float] = (0.5e6, 3.0e6),
        direct_rate_range: tuple[float, float] = (0.1, 30.0),
        allow_ts_st_exchange: bool = False,
        normalize: bool = True,
        return_clean_signals: bool = False,
        seed: int | None = None,
    ) -> tuple[np.ndarray, list[dict], np.ndarray | None]:
        """
        Sample a synthetic 3C non-Gaussian DEXSY dataset.

        Returns:
            signals_noisy: (n_samples, n_b, n_b)
            params_list: per-sample metadata list
            signals_clean (optional): clean signals with same shape as noisy
        """
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1.")

        rng = np.random.default_rng(seed)
        tm_range = self.mixing_time_range if mixing_time_range is None else mixing_time_range
        phi_alpha_arr = np.asarray(phi_alpha, dtype=np.float64)
        if phi_alpha_arr.shape != (3,) or np.any(phi_alpha_arr <= 0):
            raise ValueError("phi_alpha must be a length-3 tuple of positive values.")

        signals_noisy = np.zeros((n_samples, self.n_b, self.n_b), dtype=np.float64)
        signals_clean = (
            np.zeros((n_samples, self.n_b, self.n_b), dtype=np.float64)
            if return_clean_signals
            else None
        )
        params_list: list[dict] = []

        for idx in range(n_samples):
            phi = rng.dirichlet(phi_alpha_arr)
            tm = float(mixing_time) if mixing_time is not None else self._sample_log_uniform(rng, tm_range)

            d_e = self._sample_log_uniform(rng, extracellular_diffusivity_range)
            d_i = self._sample_log_uniform(rng, intracellular_diffusivity_range)
            l_t = self._sample_log_uniform(rng, axon_restricted_length_range)
            r_s = self._sample_log_uniform(rng, sphere_radius_range)

            if use_permeability_rates:
                p_s = self._sample_log_uniform(rng, sphere_permeability_range)
                p_t = self._sample_log_uniform(rng, axon_permeability_range)
                a_over_v_t = self._sample_log_uniform(rng, axon_surface_to_volume_range)
                k_ts = self._sample_log_uniform(rng, direct_rate_range) if allow_ts_st_exchange else 0.0
                k_st = self._sample_log_uniform(rng, direct_rate_range) if allow_ts_st_exchange else 0.0
                q, rates = self.build_generator_from_permeability(
                    phi=phi,
                    sphere_diameter=2.0 * r_s,
                    sphere_permeability=p_s,
                    axon_surface_to_volume=a_over_v_t,
                    axon_permeability=p_t,
                    k_ts=k_ts,
                    k_st=k_st,
                )
                rate_metadata = {
                    "construction": "permeability",
                    "sphere_permeability": float(p_s),
                    "axon_permeability": float(p_t),
                    "axon_surface_to_volume": float(a_over_v_t),
                    "rates": rates,
                }
            else:
                rates = {
                    "k_et": self._sample_log_uniform(rng, direct_rate_range),
                    "k_te": self._sample_log_uniform(rng, direct_rate_range),
                    "k_es": self._sample_log_uniform(rng, direct_rate_range),
                    "k_se": self._sample_log_uniform(rng, direct_rate_range),
                    "k_ts": self._sample_log_uniform(rng, direct_rate_range) if allow_ts_st_exchange else 0.0,
                    "k_st": self._sample_log_uniform(rng, direct_rate_range) if allow_ts_st_exchange else 0.0,
                }
                q = self.build_generator(
                    k_et=rates["k_et"],
                    k_te=rates["k_te"],
                    k_es=rates["k_es"],
                    k_se=rates["k_se"],
                    k_ts=rates["k_ts"],
                    k_st=rates["k_st"],
                )
                rate_metadata = {"construction": "direct_rates", "rates": rates}

            signal_clean_un, details = self.compute_signal(
                phi=phi,
                mixing_time=tm,
                extracellular_diffusivity=d_e,
                intracellular_diffusivity=d_i,
                axon_restricted_length=l_t,
                sphere_radius=r_s,
                q=q,
                normalize=False,
                return_pathway_signals=False,
            )

            sampled_noise_sigma = float(noise_sigma) if noise_sigma is not None else self._sample_uniform(
                rng, noise_sigma_range
            )
            signal_noisy = self.add_rician_noise(
                signal=signal_clean_un,
                noise_sigma=sampled_noise_sigma,
                normalize=normalize,
                rng=rng,
            )

            if normalize:
                clean_s0 = max(float(signal_clean_un[0, 0]), 1e-12)
                signal_clean_out = signal_clean_un / clean_s0
            else:
                signal_clean_out = signal_clean_un

            signals_noisy[idx] = signal_noisy
            if signals_clean is not None:
                signals_clean[idx] = signal_clean_out

            dei = self.compute_dei_from_weight_matrix(details["weight_matrix"])
            params = {
                "sample_index": int(idx),
                "phi": np.asarray(details["phi"], dtype=np.float64).tolist(),
                "mixing_time": float(tm),
                "noise_sigma": float(sampled_noise_sigma),
                "D_E": float(d_e),
                "D_I": float(d_i),
                "l_T": float(l_t),
                "R_S": float(r_s),
                "generator_q": np.asarray(details["generator_q"], dtype=np.float64).tolist(),
                "weight_matrix": np.asarray(details["weight_matrix"], dtype=np.float64).tolist(),
                "pathway_weights": dict(details["pathway_weights"]),
                "dei_weight_matrix": float(dei),
            }
            params.update(rate_metadata)
            params_list.append(params)

        return signals_noisy.astype(np.float32), params_list, (
            signals_clean.astype(np.float32) if signals_clean is not None else None
        )


def compute_3c_weight_matrix_dei(weight_matrix: np.ndarray) -> float:
    """Module-level convenience wrapper for 3C DEI from weight matrix."""
    return ForwardModel3CNonGaussian.compute_dei_from_weight_matrix(weight_matrix)

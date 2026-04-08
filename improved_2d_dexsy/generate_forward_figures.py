"""
Generate 20 2C + 20 3C forward model figures with full parameter annotations.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
try:
    from .forward_model_2d import ForwardModel2D
except ImportError:  # pragma: no cover - allows running as a script.
    from forward_model_2d import ForwardModel2D

REPO_ROOT = Path(__file__).resolve().parents[1]
out_dir = REPO_ROOT / "outputs" / "forward_model_figures"
out_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(99)   # reproducible but distinct from any previous run

fm = ForwardModel2D(
    n_d=64, n_b=64,
    d_min=5e-12, d_max=5e-8,
    g_max=1.5,
    delta=0.003, DELTA=0.01, gamma=267.52e6,
    gradient_spacing="linear",
    normalize_signals=True,
    mixing_time_range=(0.015, 0.300),
    exchange_rate_range=(0.1, 30.0),
    exchange_rate_grid_size=24,
    jitter_pixels=1,
    smoothing_sigma_range=(1.2, 2.2),
)

def fmt_val(v, fmt=".4f"):
    return format(v, fmt).rstrip('0').rstrip('.')


def fmt_diffusion(v):
    return f"{v:.2e}"

def make_fig(i, n_comp, f, s, params):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # True distribution
    ax = axes[0]
    vmax = f.max() * 1.15
    im0 = ax.imshow(f, cmap="hot", origin="lower",
                    extent=[0, 63, 0, 63], vmin=0, vmax=vmax)
    ax.set_xlabel("D₂ index", fontsize=12)
    ax.set_ylabel("D₁ index", fontsize=12)
    ax.set_title(f"True Distribution ({n_comp}C)", fontsize=14, pad=10)
    plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04).set_label("Volume Fraction", fontsize=11)

    # Signal
    ax = axes[1]
    im1 = ax.imshow(s, cmap="viridis", origin="lower",
                    extent=[0, 63, 0, 63])
    ax.set_xlabel("b₂ index", fontsize=12)
    ax.set_ylabel("b₁ index", fontsize=12)
    ax.set_title(f"Noisy Signal (SNR={params['baseline_snr']:.1f})", fontsize=14, pad=10)
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04).set_label("Normalised Signal", fontsize=11)

    # Parameter table (as a centered subtitle block)
    if n_comp == 2:
        exchange_mass = params['exchange_peak_masses']['0-1']
        param_lines = [
            f"Mixing time : {params['mixing_time']*1000:.2f} ms  "
            f"|  Noise σ : {fmt_val(params['noise_sigma'])}  "
            f"|  SNR : {fmt_val(params['baseline_snr'], '.2f')}  "
            f"|  Smoothing σ : {fmt_val(params['smoothing_sigma'])}  "
            f"|  Jitter : {params['jitter_pixels']} px",
            f"D₁ : {fmt_diffusion(params['diffusions'][0])} m²/s  "
            f"|  D₂ : {fmt_diffusion(params['diffusions'][1])} m²/s  "
            f"|  vf₁ : {fmt_val(params['volume_fractions'][0])}  "
            f"|  vf₂ : {fmt_val(params['volume_fractions'][1])}  "
            f"|  κ : {fmt_val(params['exchange_rates']['0-1'])} s⁻¹",
            f"P_ex(sampled) : {fmt_val(params['exchange_probabilities']['0-1'])}  "
            f"|  M_ex(actual) : {fmt_val(exchange_mass)}  "
            f"|  scale : {fmt_val(params['exchange_probability_scale'])}",
        ]
    else:
        kr = params['exchange_rates']
        kp = params['exchange_probabilities']
        km = params['exchange_peak_masses']
        param_lines = [
            f"Mixing time : {params['mixing_time']*1000:.2f} ms  "
            f"|  Noise σ : {fmt_val(params['noise_sigma'])}  "
            f"|  SNR : {fmt_val(params['baseline_snr'], '.2f')}  "
            f"|  Smoothing σ : {fmt_val(params['smoothing_sigma'])}  "
            f"|  Jitter : {params['jitter_pixels']} px",
            f"D₁ : {fmt_diffusion(params['diffusions'][0])} m²/s  "
            f"|  D₂ : {fmt_diffusion(params['diffusions'][1])} m²/s  "
            f"|  D₃ : {fmt_diffusion(params['diffusions'][2])} m²/s",
            f"vf : {fmt_val(params['volume_fractions'][0])}, "
            f"{fmt_val(params['volume_fractions'][1])}, "
            f"{fmt_val(params['volume_fractions'][2])}  |  "
            f"κ₀₁ : {fmt_val(kr['0-1'])} s⁻¹  "
            f"κ₀₂ : {fmt_val(kr['0-2'])} s⁻¹  "
            f"κ₁₂ : {fmt_val(kr['1-2'])} s⁻¹",
            f"P_ex(sampled) : {fmt_val(kp['0-1'])}, {fmt_val(kp['0-2'])}, {fmt_val(kp['1-2'])}",
            f"M_ex(actual) : {fmt_val(km['0-1'])}, {fmt_val(km['0-2'])}, {fmt_val(km['1-2'])}  "
            f"|  scale : {fmt_val(params['exchange_probability_scale'])}",
        ]

    fig.text(0.5, 0.02, "\n".join(param_lines),
             ha="center", va="bottom", fontsize=9.5,
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="lightgrey"))

    fig.suptitle(f"Sample {i:02d}  —  {n_comp}C",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.14, 1, 0.97])

    tag = "2C" if n_comp == 2 else "3C"
    filename = f"sample_{i:02d}_{tag}.png"
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  {filename}")


# ── 20 × 2-compartment ────────────────────────────────────────────────
print("\n=== 2C samples ===")
for i in range(1, 21):
    f, s, params = fm.generate_sample(n_compartments=2)
    make_fig(i, 2, f, s, params)

# ── 20 × 3-compartment ────────────────────────────────────────────────
print("\n=== 3C samples ===")
for i in range(1, 21):
    f, s, params = fm.generate_sample(n_compartments=3)
    make_fig(i, 3, f, s, params)

print(f"\nAll 40 figures saved to:\n  {out_dir}")

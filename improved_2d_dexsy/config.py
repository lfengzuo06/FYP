"""Configuration helpers for the 2D DEXSY inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch


def resolve_repo_root() -> Path:
    """Resolve the repository root from the package location."""
    return Path(__file__).resolve().parent.parent


REPO_ROOT = resolve_repo_root()
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints_2d"
CHECKPOINTS_DIR_3D = REPO_ROOT / "checkpoints_3d"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "inference"
DEFAULT_MODEL_NAME = "attention_unet"

# 2C models (using checkpoints_2d)
DEFAULT_CHECKPOINTS = {
    "attention_unet": "attention_unet/attention_unet_best_model.pt",
    "plain_unet": "plain_unet/best_model.pt",
    "pinn": "pinn_run_20260422/checkpoints/pinn_20260422_220103.pt",
    "deep_unfolding": "deep_unfolding/best_model.pt",
    "deeponet": "deeponet/best_model.pt",
    "fno": "fno/best_model.pt",
    "2d_ilt": None,  # ILT doesn't require a checkpoint
}

# 3C models (using checkpoints_3d)
DEFAULT_CHECKPOINTS_3D = {
    "attention_unet_3c": "attention_unet_3c",
    "plain_unet_3c": "plain_unet_3c",
    "pinn_3c": "pinn_3c",
    "deep_unfolding_3c": "deep_unfolding_3c",
    "diffusion_refiner": "diffusion_refiner",
    "3d_ilt": None,  # ILT doesn't require a checkpoint
}

# All models combined
ALL_MODELS = {**DEFAULT_CHECKPOINTS, **DEFAULT_CHECKPOINTS_3D}


def is_3c_model(model_name: str) -> bool:
    """Check if a model is a 3C model."""
    return model_name in DEFAULT_CHECKPOINTS_3D


def available_models(include_3c: bool = True) -> list[str]:
    """Return supported model names."""
    models = list(ALL_MODELS.keys()) if include_3c else list(DEFAULT_CHECKPOINTS.keys())
    return sorted(models)


def list_available_checkpoints(checkpoints_dir: str | Path | None = None) -> list[Path]:
    """List bundled checkpoint files for 2C models."""
    directory = CHECKPOINTS_DIR if checkpoints_dir is None else Path(checkpoints_dir)
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*.pt") if path.is_file())


def list_available_checkpoints_3d() -> list[Path]:
    """List bundled checkpoint files for 3C models."""
    if not CHECKPOINTS_DIR_3D.exists():
        return []
    return sorted(path for path in CHECKPOINTS_DIR_3D.rglob("*.pt") if path.is_file())


def resolve_checkpoint_path(
    checkpoint_path: str | Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Path | None:
    """Resolve an explicit or bundled checkpoint path."""
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        return path

    # ILT doesn't need a checkpoint
    if model_name in ("2d_ilt", "3d_ilt"):
        return None

    # Check if 3C model
    if model_name in DEFAULT_CHECKPOINTS_3D:
        model_dir = DEFAULT_CHECKPOINTS_3D[model_name]
        if model_dir is None:
            return None
        root_3d = CHECKPOINTS_DIR_3D / model_dir
        if root_3d.exists():
            # Find best_model.pt in timestamped subdirectories
            best_models = list(root_3d.glob("*/best_model.pt"))
            if best_models:
                return sorted(best_models, key=lambda p: p.stat().st_mtime)[-1]
            # Fallback: look for any .pt file
            all_pts = list(root_3d.glob("**/*.pt"))
            if all_pts:
                return sorted(all_pts, key=lambda p: p.stat().st_mtime)[-1]
        return None

    # 2C model
    try:
        filename = DEFAULT_CHECKPOINTS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available_models()}"
        ) from exc
    if filename is None:
        return None
    return CHECKPOINTS_DIR / filename


def resolve_output_root(output_dir: str | Path | None = None) -> Path:
    """Resolve the base output directory for inference artifacts."""
    if output_dir is None:
        return DEFAULT_OUTPUT_ROOT
    path = Path(output_dir)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def create_run_output_dir(
    output_dir: str | Path | None = None,
    prefix: str = "run",
) -> Path:
    """Create a timestamped output directory for one inference run."""
    root = resolve_output_root(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{prefix}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a torch device, treating ``auto`` as CUDA-if-available."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass(frozen=True)
class InferenceConfig:
    """Lightweight config object shared by scripts and interface code."""

    model_name: str = DEFAULT_MODEL_NAME
    checkpoint_path: str | Path | None = None
    output_dir: str | Path | None = None
    device: str | torch.device | None = None
    batch_size: int = 16
    include_figures: bool = True

    @property
    def resolved_checkpoint_path(self) -> Path:
        return resolve_checkpoint_path(self.checkpoint_path, model_name=self.model_name)

    @property
    def resolved_output_root(self) -> Path:
        return resolve_output_root(self.output_dir)

    @property
    def resolved_device(self) -> torch.device:
        return resolve_device(self.device)

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
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "inference"
DEFAULT_MODEL_NAME = "attention_unet"
DEFAULT_CHECKPOINTS = {
    "attention_unet": "attention_unet_best_model_20260411_155746.pt",
    "plain_unet": "plain_unet/checkpoints_20260419_185116/best_model.pt",
    "deep_unfolding": "deep_unfolding/run_fixed_v1/best_model.pt",
    "deeponet": None,  # Not trained yet
    "fno": None,  # Not trained yet
    "2d_ilt": None,  # ILT doesn't require a checkpoint
}


def available_models() -> list[str]:
    """Return supported high-level model names."""
    return sorted(DEFAULT_CHECKPOINTS.keys())


def list_available_checkpoints(checkpoints_dir: str | Path | None = None) -> list[Path]:
    """List bundled checkpoint files."""
    directory = CHECKPOINTS_DIR if checkpoints_dir is None else Path(checkpoints_dir)
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*.pt") if path.is_file())


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
    if model_name == "2d_ilt":
        return None

    try:
        filename = DEFAULT_CHECKPOINTS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available_models()}"
        ) from exc
    if filename is None:
        return None  # e.g., 2d_ilt
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

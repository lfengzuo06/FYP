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
OTHER_MODELS_DIR = REPO_ROOT / "checkpoints_other"  # User-trained models
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "inference"
DEFAULT_MODEL_NAME = "attention_unet"

# Grid size support for each model
# 64: supports 64x64, 16: supports 16x16, [16, 64]: supports both
MODEL_GRID_SUPPORT = {
    # 64x64 only models
    "pinn": [64],
    "deeponet": [64],
    "pinn_3c": [64],
    # Both sizes supported
    "attention_unet": [16, 64],
    "attention_unet_g16": [16],
    "plain_unet": [16, 64],
    "plain_unet_g16": [16],
    "deep_unfolding": [16, 64],
    "deep_unfolding_g16": [16],
    "fno": [16, 64],
    "attention_unet_3c": [16, 64],
    "attention_unet_3c_g16": [16],
    "plain_unet_3c": [16, 64],
    "plain_unet_3c_g16": [16],
    "deep_unfolding_3c": [16, 64],
    "deep_unfolding_3c_g16": [16],
    # No checkpoint needed
    "2d_ilt": [16, 64],
    "3d_ilt": [16, 64],
    # Special models
    "diffusion_refiner": [64],
    "pinn_g16": [16],
    "pinn_3c_g16": [16],
}

# 2C models (using checkpoints_2d)
DEFAULT_CHECKPOINTS = {
    "attention_unet": "attention_unet/attention_unet_best_model.pt",
    "attention_unet_g16": "attention_unet_g16/checkpoints_20260507_203149/best_model.pt",
    "plain_unet": "plain_unet/best_model.pt",
    "plain_unet_g16": "plain_unet_g16_2c/checkpoints_20260508_172800/best_model.pt",
    "pinn": "pinn_run_20260422/checkpoints/pinn_20260422_220103.pt",
    "pinn_g16": "pinn_g16_2c/checkpoints/pinn_20260508_172729.pt",
    "deep_unfolding": "deep_unfolding/best_model.pt",
    "deep_unfolding_g16": "deep_unfolding_g16_2c/best_model.pt",
    "deeponet": "deeponet/best_model.pt",
    "fno": "fno/best_model.pt",
    "2d_ilt": None,  # ILT doesn't require a checkpoint
}

# 3C models (using checkpoints_3d)
DEFAULT_CHECKPOINTS_3D = {
    "attention_unet_3c": "attention_unet_3c/checkpoints_20260428_164228/best_model.pt",
    "attention_unet_3c_g16": "attention_unet_3c_g16/checkpoints_20260507_205915/best_model.pt",
    "plain_unet_3c": "plain_unet_3c/checkpoints_20260430_203333/best_model.pt",
    "plain_unet_3c_g16": "plain_unet_3c_g16/checkpoints_20260509_071946/best_model.pt",
    "pinn_3c": "pinn_3c/checkpoints_20260502_095025/best_model.pt",
    "pinn_3c_g16": "pinn_3c_g16/checkpoints_20260509_071819/best_model.pt",
    "deep_unfolding_3c": "deep_unfolding_3c/checkpoints_20260501_121714/best_model.pt",
    "deep_unfolding_3c_g16": "deep_unfolding_3c_g16/checkpoints_20260509_075729/best_model.pt",
    "diffusion_refiner": "diffusion_refiner/checkpoints_20260430_161942/best_model.pt",
    "3d_ilt": None,  # ILT doesn't require a checkpoint
}

# All models combined
ALL_MODELS = {**DEFAULT_CHECKPOINTS, **DEFAULT_CHECKPOINTS_3D}


def is_3c_model(model_name: str) -> bool:
    """Check if a model is a 3C model."""
    # Handle g16 suffix
    base_name = model_name.replace("_g16", "")
    
    if model_name.startswith("other_"):
        # Check from other_models directory
        custom_model_name = model_name[len("other_"):]
        other_checkpoint = OTHER_MODELS_DIR / custom_model_name / "best_model.pt"
        if other_checkpoint.exists():
            try:
                import torch
                ckpt = torch.load(other_checkpoint, map_location='cpu')
                config = ckpt.get('config', {})
                return config.get('n_compartments', 2) == 3
            except Exception:
                pass
        return False
    return base_name in DEFAULT_CHECKPOINTS_3D


def available_models(include_3c: bool = True, include_other: bool = True) -> list[str]:
    """Return supported model names."""
    models = list(ALL_MODELS.keys()) if include_3c else list(DEFAULT_CHECKPOINTS.keys())
    
    # Add other_models if they exist
    if include_other:
        other_models = list_other_models_by_name()
        for model_dir in other_models.keys():
            other_name = f"other_{model_dir}"
            if other_name not in models:
                # Check n_compartments and add to appropriate list
                try:
                    import torch
                    checkpoint_path = OTHER_MODELS_DIR / model_dir / "best_model.pt"
                    ckpt = torch.load(checkpoint_path, map_location='cpu')
                    config = ckpt.get('config', {})
                    n_comp = config.get('n_compartments', 2)
                    if n_comp == 3 and not include_3c:
                        continue
                except Exception:
                    pass
                models.append(other_name)
    
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

    # Handle other_models
    if model_name.startswith("other_"):
        custom_model_name = model_name[len("other_"):]
        other_checkpoint = OTHER_MODELS_DIR / custom_model_name / "best_model.pt"
        if other_checkpoint.exists():
            return other_checkpoint
        return None

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


def get_model_grid_support(model_name: str) -> list[int]:
    """Get supported grid sizes for a model."""
    return MODEL_GRID_SUPPORT.get(model_name, [64])


def list_other_models() -> list[Path]:
    """List user-trained model checkpoints from other_models directory."""
    if not OTHER_MODELS_DIR.exists():
        return []
    return sorted(path for path in OTHER_MODELS_DIR.rglob("best_model.pt") if path.is_file())


def list_other_models_by_name() -> dict[str, list[str]]:
    """List other models grouped by model directory name."""
    if not OTHER_MODELS_DIR.exists():
        return {}
    result = {}
    for path in OTHER_MODELS_DIR.rglob("best_model.pt"):
        model_dir = path.parent.name
        rel_path = str(path.relative_to(OTHER_MODELS_DIR))
        if model_dir not in result:
            result[model_dir] = []
        result[model_dir].append(rel_path)
    return result

"""
Slim 2D DEXSY package for training and Colab inference.

This package is now a thin wrapper around the modular structure:
- dexsy_core/: Shared scientific core (forward model, metrics, preprocessing)
- models_2d/: Model implementations (attention_unet, plain_unet, etc.)
- benchmarks_2d/: Benchmark infrastructure (ILT baseline, evaluation)

All implementations are delegated to their respective modules.
"""

from __future__ import annotations
from pathlib import Path

# Import configuration helpers (these remain here as they're app-specific)
from .config import (
    CHECKPOINTS_DIR,
    DEFAULT_CHECKPOINTS,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_ROOT,
    InferenceConfig,
    create_run_output_dir,
    list_available_checkpoints,
    resolve_device,
    resolve_output_root,
    resolve_repo_root,
)

# Re-export from dexsy_core for backwards compatibility
from dexsy_core import (
    ForwardModel2D,
    compute_dei,
)
from dexsy_core.preprocessing import (
    build_model_inputs,
    build_position_channel,
    validate_signal_grid,
)

# Re-export from models_2d for backwards compatibility
from models_2d.attention_unet import (
    AttentionUNet2D,
    InferencePipeline as DEXSYInferencePipeline,
    InferencePipeline,
    PhysicsInformedLoss,
    predict as predict_from_signal,
    predict_batch as predict_batch_from_signals,
    train_model,
)
from models_2d.attention_unet.model import get_model as get_attention_model
from models_2d.attention_unet.inference import load_trained_model

# Re-export Plain U-Net
from models_2d.plain_unet import (
    PlainUNet2D,
    InferencePipeline as PlainInferencePipeline,
    PlainUNetLoss,
    predict as predict_plain_unet,
    predict_batch as predict_batch_plain_unet,
    train_model as train_plain_unet,
)
from models_2d.plain_unet.model import get_model as get_plain_model

# Re-export from benchmarks_2d for convenience
from benchmarks_2d import ILTInferencePipeline

# Import I/O helpers (these remain here as they're app-specific)
from .io_2d import (
    create_output_archive,
    load_matrix,
    load_named_matrices_from_directory,
    save_batch_results,
    save_json,
    save_prediction_result,
    to_serializable,
)


def available_models() -> list[str]:
    """Return supported model names."""
    return sorted(DEFAULT_CHECKPOINTS.keys())


def resolve_checkpoint_path(
    checkpoint_path: str | Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Path:
    """Resolve an explicit or bundled checkpoint path."""
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.is_absolute():
            path = (resolve_repo_root() / path).resolve()
        return path

    try:
        filename = DEFAULT_CHECKPOINTS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available_models()}"
        ) from exc
    return CHECKPOINTS_DIR / filename


__all__ = [
    # Configuration
    "CHECKPOINTS_DIR",
    "DEFAULT_CHECKPOINTS",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "InferenceConfig",
    "available_models",
    "create_run_output_dir",
    "list_available_checkpoints",
    "resolve_checkpoint_path",
    "resolve_device",
    "resolve_output_root",
    "resolve_repo_root",
    # Core (from dexsy_core)
    "ForwardModel2D",
    "compute_dei",
    "build_model_inputs",
    "build_position_channel",
    "validate_signal_grid",
    # Attention U-Net
    "AttentionUNet2D",
    "DEXSYInferencePipeline",
    "PhysicsInformedLoss",
    "get_attention_model",
    "predict_from_signal",
    "predict_batch_from_signals",
    "train_model",
    "load_trained_model",
    # Plain U-Net
    "PlainUNet2D",
    "PlainInferencePipeline",
    "PlainUNetLoss",
    "get_plain_model",
    "predict_plain_unet",
    "predict_batch_plain_unet",
    "train_plain_unet",
    # Benchmarks
    "ILTInferencePipeline",
    # I/O
    "create_output_archive",
    "load_matrix",
    "load_named_matrices_from_directory",
    "save_batch_results",
    "save_json",
    "save_prediction_result",
    "to_serializable",
]

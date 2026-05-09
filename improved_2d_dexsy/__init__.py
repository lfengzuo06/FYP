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

import torch

# Import configuration helpers (these remain here as they're app-specific)
from .config import (
    CHECKPOINTS_DIR,
    CHECKPOINTS_DIR_3D,
    DEFAULT_CHECKPOINTS,
    DEFAULT_CHECKPOINTS_3D,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_ROOT,
    InferenceConfig,
    MODEL_GRID_SUPPORT,
    available_models,
    create_run_output_dir,
    get_model_grid_support,
    is_3c_model,
    list_available_checkpoints,
    list_available_checkpoints_3d,
    list_other_models,
    list_other_models_by_name,
    OTHER_MODELS_DIR,
    resolve_checkpoint_path,
    resolve_device,
    resolve_output_root,
    resolve_repo_root,
)

# Re-export from dexsy_core for backwards compatibility
from dexsy_core import (
    ForwardModel2D,
    create_forward_model,
    GRID_PROFILES,
    compute_dei,
)
from dexsy_core.preprocessing import (
    build_model_inputs,
    build_position_channel,
    validate_signal_grid,
)

# Import ILT baseline
from benchmarks_2d.ilt_baseline import (
    ILTInferencePipeline,
    predict_ilt,
)

# Import individual model inference pipelines
from models_2d.attention_unet.inference import InferencePipeline as AttentionInferencePipeline
from models_2d.attention_unet import (
    AttentionUNet2D,
    PhysicsInformedLoss,
    predict as predict_from_signal,
    predict_batch as predict_batch_from_signals,
    train_model,
)
from models_2d.attention_unet.model import get_model as get_attention_model
from models_2d.attention_unet.inference import load_trained_model

# Re-export Plain U-Net
from models_2d.plain_unet.inference import InferencePipeline as PlainInferencePipeline
from models_2d.plain_unet import (
    PlainUNet2D,
    PlainUNetLoss,
    predict as predict_plain_unet,
    predict_batch as predict_batch_plain_unet,
    train_model as train_plain_unet,
)
from models_2d.plain_unet.model import get_model as get_plain_model

# Import PINN
from models_2d.pinn.inference import PINNInferencePipeline
from models_2d.pinn.model import PINN2D, PINNLoss
from models_2d.pinn.train import train_model as train_pinn
from models_2d.pinn.inference import (
    predict as predict_pinn,
    predict_batch as predict_batch_pinn,
    load_trained_model as load_trained_pinn_model,
)

# Import Deep Unfolding
from models_2d.deep_unfolding.inference import InferencePipeline as DeepUnfoldingInferencePipeline
from models_2d.deep_unfolding.model import (
    DeepUnfolding2D,
    get_model as get_deep_unfolding_model,
)
from models_2d.deep_unfolding import inference as deep_unfolding_inference
from models_2d.deep_unfolding import train as deep_unfolding_train

# Import Neural Operators
from models_2d.neural_operators.inference import InferencePipeline as NeuralOpInferencePipeline
from models_2d.neural_operators.model import (
    DeepONet2D,
    FNO2D,
    get_deeponet_model,
    get_fno_model,
)
from models_2d.neural_operators import train as neural_op_train
from models_2d.neural_operators import inference as neural_op_inference

# Import 3D model inference pipelines
from models_3d.attention_unet.inference import InferencePipeline3C as AttentionInferencePipeline3D
from models_3d.plain_unet.inference import InferencePipelinePlain3C as PlainInferencePipeline3D
from models_3d.pinn.inference import PINNInferencePipeline3C as PINNInferencePipeline3D
from models_3d.deep_unfolding.inference import InferencePipeline3C as DeepUnfoldingInferencePipeline3D
from models_3d.diffusion_refiner.inference import UncertaintyEstimator as DiffusionRefinerPipeline


class DEXSYInferencePipeline:
    """
    Unified inference pipeline that supports multiple model types.

    This class wraps the model-specific inference pipelines and provides
    a consistent interface with model_name dispatching.

    Args:
        model_name: Model type ('attention_unet', 'plain_unet', 'pinn', 'deep_unfolding',
                    'deeponet', 'fno', '2d_ilt', 'attention_unet_3c', 'plain_unet_3c',
                    'pinn_3c', 'deep_unfolding_3c', 'diffusion_refiner', '3d_ilt')
        checkpoint_path: Path to model checkpoint (not needed for ILT)
        device: Device to use ('cuda', 'cpu', or None for auto)
        forward_model: ForwardModel2D instance (creates new if None)
        alpha: ILT regularization parameter (only for 2d_ilt/3d_ilt)
        model_type: Neural operator type ('deeponet' or 'fno', only for neural operators)
        grid_size: Grid size for inference (16 or 64). Creates matching forward model if forward_model is None.
    """

    _MODEL_REGISTRY_2D = {
        "attention_unet": AttentionInferencePipeline,
        "plain_unet": PlainInferencePipeline,
        "pinn": PINNInferencePipeline,
        "deep_unfolding": DeepUnfoldingInferencePipeline,
        "deeponet": NeuralOpInferencePipeline,
        "fno": NeuralOpInferencePipeline,
        "2d_ilt": ILTInferencePipeline,
    }

    _MODEL_REGISTRY_3D = {
        "attention_unet_3c": AttentionInferencePipeline3D,
        "plain_unet_3c": PlainInferencePipeline3D,
        "pinn_3c": PINNInferencePipeline3D,
        "deep_unfolding_3c": DeepUnfoldingInferencePipeline3D,
        "diffusion_refiner": DiffusionRefinerPipeline,
        "3d_ilt": None,  # Will be handled separately
    }

    _MODEL_REGISTRY = {**_MODEL_REGISTRY_2D, **_MODEL_REGISTRY_3D}

    def __init__(
        self,
        model_name: str = "attention_unet",
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        forward_model: ForwardModel2D | None = None,
        alpha: float = 0.02,
        model_type: str | None = None,  # For neural operators
        grid_size: int | None = None,  # Grid size for inference
    ):
        # Handle custom "other" models
        if model_name.startswith("other_"):
            # Custom trained model from OTHER_MODELS_DIR
            custom_model_name = model_name[len("other_"):]
            other_checkpoint = OTHER_MODELS_DIR / custom_model_name / "best_model.pt"
            
            if other_checkpoint.exists():
                checkpoint_path = str(other_checkpoint)
                # Try to determine actual model type from checkpoint
                try:
                    ckpt = torch.load(other_checkpoint, map_location='cpu')
                    config = ckpt.get('config', {})
                    actual_model_type = config.get('model_type', 'attention_unet')
                    grid_size = config.get('grid_size', grid_size)
                    model_name = actual_model_type  # Use actual model type
                except Exception:
                    actual_model_type = 'attention_unet'
            else:
                raise ValueError(f"Custom model '{custom_model_name}' not found at {other_checkpoint}")

        if model_name not in self._MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {list(self._MODEL_REGISTRY.keys())}"
            )

        pipeline_class = self._MODEL_REGISTRY[model_name]

        # Create forward model based on grid_size or checkpoint config
        if forward_model is None and model_name not in ("2d_ilt", "3d_ilt"):
            # Try to read grid_size from checkpoint if not explicitly provided
            if grid_size is None and checkpoint_path is not None:
                checkpoint_path_obj = Path(checkpoint_path) if checkpoint_path else None
                if checkpoint_path_obj and checkpoint_path_obj.exists():
                    try:
                        checkpoint = torch.load(checkpoint_path_obj, map_location='cpu')
                        config = checkpoint.get('config', {})
                        grid_size = config.get('n_d') or config.get('grid_size')
                    except Exception:
                        pass  # Fall back to default

            if grid_size is not None:
                forward_model = create_forward_model(profile=grid_size)
            else:
                # Default to 64x64 for backward compatibility
                forward_model = ForwardModel2D(n_d=64, n_b=64)

        # 3D ILT doesn't need checkpoint_path (but ILTInferencePipeline is 2D only)
        # For 3d_ilt, we'll need to check if there's a 3D ILT implementation
        if model_name in ("2d_ilt", "3d_ilt"):
            if model_name == "2d_ilt":
                self._pipeline = pipeline_class(
                    alpha=alpha,
                    forward_model=forward_model,
                )
            else:
                # 3D ILT - use 3D ILT implementation if available
                raise NotImplementedError(
                    "3D ILT is not yet implemented. "
                    "Please use a trained 3D model instead."
                )
        # Neural operators need model_type parameter
        elif model_name in ("deeponet", "fno"):
            if model_type is not None and model_type != model_name:
                raise ValueError(
                    f"model_name='{model_name}' is incompatible with model_type='{model_type}'. "
                    "Use matching values, or omit model_type."
                )
            resolved_model_type = model_name if model_type is None else model_type
            self._pipeline = pipeline_class(
                model_type=resolved_model_type,
                checkpoint_path=checkpoint_path,
                device=device,
                forward_model=forward_model,
            )
        else:
            self._pipeline = pipeline_class(
                checkpoint_path=checkpoint_path,
                device=device,
                forward_model=forward_model,
            )
        self._model_name = model_name

    def predict(
        self,
        signal,
        *,
        true_spectrum=None,
        include_figure=True,
        source_name=None,
        **kwargs,
    ):
        """Delegate to the underlying pipeline's predict method."""
        # ILT doesn't use true_spectrum
        if self._model_name == "2d_ilt":
            return self._pipeline.predict(
                signal,
                include_figure=include_figure,
                source_name=source_name,
                **kwargs,
            )
        return self._pipeline.predict(
            signal,
            true_spectrum=true_spectrum,
            include_figure=include_figure,
            source_name=source_name,
            **kwargs,
        )

    def predict_batch(
        self,
        signals,
        *,
        true_spectra=None,
        source_names=None,
        include_figures=False,
        batch_size=16,
        **kwargs,
    ):
        """Delegate to the underlying pipeline's predict_batch method."""
        # ILT doesn't use true_spectra
        if self._model_name == "2d_ilt":
            return self._pipeline.predict_batch(
                signals,
                source_names=source_names,
                include_figures=include_figures,
                **kwargs,
            )
        return self._pipeline.predict_batch(
            signals,
            true_spectra=true_spectra,
            source_names=source_names,
            include_figures=include_figures,
            batch_size=batch_size,
            **kwargs,
        )

    def predict_batch_from_signals(
        self,
        signals,
        *,
        true_spectra=None,
        source_names=None,
        include_figures=False,
        batch_size=16,
        **kwargs,
    ):
        """Alias for predict_batch (backwards compatibility)."""
        return self.predict_batch(
            signals,
            true_spectra=true_spectra,
            source_names=source_names,
            include_figures=include_figures,
            batch_size=batch_size,
            **kwargs,
        )

    def predict_from_signal(
        self,
        signal,
        *,
        true_spectrum=None,
        include_figure=True,
        source_name=None,
        **kwargs,
    ):
        """Alias for predict (backwards compatibility)."""
        return self.predict(
            signal,
            true_spectrum=true_spectrum,
            include_figure=include_figure,
            source_name=source_name,
            **kwargs,
        )

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def get_model_name(self) -> str:
        """Return the model name (for interface compatibility)."""
        return self._pipeline.get_model_name()

    def get_model_info(self) -> dict:
        """Return model information."""
        return self._pipeline.get_model_info()

    @property
    def forward_model(self) -> ForwardModel2D:
        """Return the forward model (for interface compatibility)."""
        return self._pipeline.forward_model

    @property
    def device(self):
        """Return the device (for interface compatibility)."""
        return self._pipeline.device

    @property
    def model(self):
        """Return the underlying model (for interface compatibility)."""
        return self._pipeline.model


# Alias InferencePipeline for backwards compatibility
InferencePipeline = DEXSYInferencePipeline

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


def available_models_2d() -> list[str]:
    """Return supported 2D model names."""
    return sorted(DEFAULT_CHECKPOINTS.keys())


def available_models_3d() -> list[str]:
    """Return supported 3D model names."""
    return sorted(DEFAULT_CHECKPOINTS_3D.keys())


def available_models_all() -> list[str]:
    """Return all supported model names (2D + 3D)."""
    return sorted(set(DEFAULT_CHECKPOINTS.keys()) | set(DEFAULT_CHECKPOINTS_3D.keys()))


# For backwards compatibility, make available_models return all models
def available_models() -> list[str]:
    """Return all supported model names."""
    return available_models_all()


def is_3c_model(model_name: str) -> bool:
    """Check if a model is a 3C model."""
    return model_name in DEFAULT_CHECKPOINTS_3D


def resolve_checkpoint_path(
    checkpoint_path: str | Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Path | None:
    """Resolve an explicit or bundled checkpoint path."""
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.is_absolute():
            path = (resolve_repo_root() / path).resolve()
        return path

    # Handle other_models (custom trained models)
    if model_name.startswith("other_"):
        custom_model_name = model_name[len("other_"):]
        other_checkpoint = OTHER_MODELS_DIR / custom_model_name / "best_model.pt"
        if other_checkpoint.exists():
            return other_checkpoint
        return None

    # ILT doesn't need checkpoint
    if model_name in ("2d_ilt", "3d_ilt"):
        return None

    # Check 3C models first
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

    # 2D model
    try:
        filename = DEFAULT_CHECKPOINTS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available_models()}"
        ) from exc
    if filename is None:
        return None
    return CHECKPOINTS_DIR / filename


def list_available_checkpoints_3d() -> list[Path]:
    """List available 3D checkpoint files."""
    if not CHECKPOINTS_DIR_3D.exists():
        return []
    return sorted(path for path in CHECKPOINTS_DIR_3D.rglob("*.pt") if path.is_file())


__all__ = [
    # Configuration
    "CHECKPOINTS_DIR",
    "CHECKPOINTS_DIR_3D",
    "DEFAULT_CHECKPOINTS",
    "DEFAULT_CHECKPOINTS_3D",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "InferenceConfig",
    "available_models",
    "available_models_2d",
    "available_models_3d",
    "available_models_all",
    "create_run_output_dir",
    "is_3c_model",
    "list_available_checkpoints",
    "list_available_checkpoints_3d",
    "resolve_checkpoint_path",
    "resolve_device",
    "resolve_output_root",
    "resolve_repo_root",
    # Core (from dexsy_core)
    "ForwardModel2D",
    "create_forward_model",
    "GRID_PROFILES",
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
    # PINN
    "PINN2D",
    "PINNLoss",
    "PINNInferencePipeline",
    "predict_pinn",
    "predict_batch_pinn",
    "train_pinn",
    "load_trained_pinn_model",
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

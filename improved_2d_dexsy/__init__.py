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


class DEXSYInferencePipeline:
    """
    Unified inference pipeline that supports multiple model types.

    This class wraps the model-specific inference pipelines and provides
    a consistent interface with model_name dispatching.

    Args:
        model_name: Model type ('attention_unet', 'plain_unet', 'deep_unfolding',
                    'deeponet', 'fno', or '2d_ilt')
        checkpoint_path: Path to model checkpoint (not needed for ILT)
        device: Device to use ('cuda', 'cpu', or None for auto)
        forward_model: ForwardModel2D instance
        alpha: ILT regularization parameter (only for 2d_ilt)
        model_type: Neural operator type ('deeponet' or 'fno', only for neural operators)
    """

    _MODEL_REGISTRY = {
        "attention_unet": AttentionInferencePipeline,
        "plain_unet": PlainInferencePipeline,
        "deep_unfolding": DeepUnfoldingInferencePipeline,
        "deeponet": NeuralOpInferencePipeline,
        "fno": NeuralOpInferencePipeline,
        "2d_ilt": ILTInferencePipeline,
    }

    def __init__(
        self,
        model_name: str = "attention_unet",
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        forward_model: ForwardModel2D | None = None,
        alpha: float = 0.02,
        model_type: str | None = None,  # For neural operators
    ):
        if model_name not in self._MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {list(self._MODEL_REGISTRY.keys())}"
            )

        pipeline_class = self._MODEL_REGISTRY[model_name]

        # ILT doesn't need checkpoint_path
        if model_name == "2d_ilt":
            self._pipeline = pipeline_class(
                alpha=alpha,
                forward_model=forward_model,
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


def available_models() -> list[str]:
    """Return supported model names."""
    return sorted(DEFAULT_CHECKPOINTS.keys())


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

    if model_name == "2d_ilt":
        return None

    try:
        filename = DEFAULT_CHECKPOINTS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available_models()}"
        ) from exc
    if filename is None:
        return None
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

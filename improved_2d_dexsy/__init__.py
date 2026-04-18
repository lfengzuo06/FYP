"""Slim 2D DEXSY package for training and Colab inference."""

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
)
from .forward_model_2d import ForwardModel2D, compute_dei
from .inference_2d import (
    DEXSYInferencePipeline,
    PredictionResult,
    available_models,
    build_model_inputs,
    build_position_channel,
    load_trained_model,
    predict_batch_from_signals,
    predict_from_signal,
    predict_distribution,
    resolve_checkpoint_path,
)
from .io_2d import (
    create_output_archive,
    load_matrix,
    load_named_matrices_from_directory,
    save_batch_results,
    save_json,
    save_prediction_result,
    to_serializable,
)
from .model_2d import (
    AttentionUNet2D,
    MultiscaleUNet2D,
    PhysicsInformedLoss,
    get_model,
)
from .preprocessing_2d import validate_signal_grid

__all__ = [
    "AttentionUNet2D",
    "CHECKPOINTS_DIR",
    "DEFAULT_CHECKPOINTS",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "DEXSYInferencePipeline",
    "InferenceConfig",
    "MultiscaleUNet2D",
    "PhysicsInformedLoss",
    "ForwardModel2D",
    "PredictionResult",
    "available_models",
    "build_model_inputs",
    "build_position_channel",
    "compute_dei",
    "create_output_archive",
    "create_run_output_dir",
    "get_model",
    "list_available_checkpoints",
    "load_matrix",
    "load_named_matrices_from_directory",
    "load_trained_model",
    "predict_batch_from_signals",
    "predict_from_signal",
    "predict_distribution",
    "resolve_checkpoint_path",
    "resolve_device",
    "resolve_output_root",
    "save_batch_results",
    "save_json",
    "save_prediction_result",
    "to_serializable",
    "validate_signal_grid",
]

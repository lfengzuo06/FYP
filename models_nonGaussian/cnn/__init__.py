"""
CNN module for non-Gaussian 3C inverse modeling.

Exports:
- model definitions and loss
- training pipeline
- inference pipeline
"""

from .model import (
    PATHWAY_ORDER_3C,
    DIAGONAL_PATHWAYS_3C,
    NonGaussian3CPrediction,
    reshape_pathway_vector_to_matrix,
    flatten_pathway_matrix_to_vector,
    compute_dei_from_pathway_weights,
    compute_dei_from_weight_matrix,
    NonGaussian3CInverseNet,
    NonGaussian3CLoss,
)

from .train import (
    NonGaussian3CDataset,
    set_seed,
    pathway_weights_from_params_list,
    sample_3c_nongaussian_inverse_dataset,
    generate_nongaussian_inverse_splits,
    train_nongaussian_inverse_model,
)

from .inference import (
    NonGaussian3CInferenceResult,
    InferencePipeline,
    predict,
)

__all__ = [
    "PATHWAY_ORDER_3C",
    "DIAGONAL_PATHWAYS_3C",
    "NonGaussian3CPrediction",
    "reshape_pathway_vector_to_matrix",
    "flatten_pathway_matrix_to_vector",
    "compute_dei_from_pathway_weights",
    "compute_dei_from_weight_matrix",
    "NonGaussian3CInverseNet",
    "NonGaussian3CLoss",
    "NonGaussian3CDataset",
    "set_seed",
    "pathway_weights_from_params_list",
    "sample_3c_nongaussian_inverse_dataset",
    "generate_nongaussian_inverse_splits",
    "train_nongaussian_inverse_model",
    "NonGaussian3CInferenceResult",
    "InferencePipeline",
    "predict",
]

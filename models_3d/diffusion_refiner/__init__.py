"""
Diffusion Refiner for 3C Model.

This module provides a conditional diffusion model that refines the output of the
3C Attention UNet by learning to correct artifacts and fill in details.

Usage:
    # Training
    from models_3d.diffusion_refiner import train_refiner

    model, history = train_refiner(
        baseline_model=unet_model,
        forward_model=forward_model,
        epochs=100,
    )

    # Inference
    from models_3d.diffusion_refiner import RefinementInference, load_refiner

    inference = load_refiner(
        refiner_checkpoint='path/to/refiner.pt',
        baseline_checkpoint='path/to/baseline.pt',
        forward_model=forward_model,
    )

    result = inference.refine(signal)

Modules:
    config: Configuration classes for diffusion training
    scheduler: DDPM/DDIM noise schedule implementation
    model: Conditional U-Net denoiser architecture
    dataset: Dataset classes for refinement training
    loss: Combined loss functions
    train: Training pipeline
    inference: Inference with uncertainty estimation
    evaluate: Evaluation and comparison utilities
"""

from .config import (
    DiffusionConfig,
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    RefinerConfig,
)

from .scheduler import (
    DDPMScheduler,
    DDIMScheduler,
    SinusoidalPosEmb,
)

from .model import (
    ConditionalUNetDenoiser,
    RefinerWithBaseline,
    TimeEmbedding,
    FiLMLayer,
    ConditionalResidualDenseBlock,
    get_denoiser,
)

from .inference import (
    RefinementInference,
    RefinementResult,
    load_refiner,
)

from .dataset import (
    RefinementDataset,
    RefinementDataLoader,
    generate_refinement_dataset,
)

from .train import train_refiner

from .loss import (
    RefinementLoss,
    RefinementLossSimple,
    AdaptiveRefinementLoss,
    create_loss,
)

from .evaluate import (
    evaluate_refiner,
    ComparisonResult,
    compute_spectrum_metrics,
    plot_comparison_samples,
    plot_error_distribution,
    plot_uncertainty_calibration,
)

from .train import train_refiner

__all__ = [
    # Config
    'DiffusionConfig',
    'ModelConfig',
    'TrainingConfig',
    'InferenceConfig',
    'RefinerConfig',
    # Scheduler
    'DDPMScheduler',
    'DDIMScheduler',
    'SinusoidalPosEmb',
    # Model
    'ConditionalUNetDenoiser',
    'RefinerWithBaseline',
    'TimeEmbedding',
    'FiLMLayer',
    'ConditionalResidualDenseBlock',
    'get_denoiser',
    # Inference
    'RefinementInference',
    'RefinementResult',
    'load_refiner',
    # Dataset
    'RefinementDataset',
    'RefinementDataLoader',
    'generate_refinement_dataset',
    # Loss
    'RefinementLoss',
    'RefinementLossSimple',
    'AdaptiveRefinementLoss',
    'create_loss',
    # Evaluate
    'evaluate_refiner',
    'ComparisonResult',
    'compute_spectrum_metrics',
    'plot_comparison_samples',
    'plot_error_distribution',
    'plot_uncertainty_calibration',
    # Train
    'train_refiner',
]

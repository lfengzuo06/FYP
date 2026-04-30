"""
Diffusion Refiner for 3C Model.

This module provides a conditional diffusion model for uncertainty estimation.
The baseline UNet provides the primary prediction, and diffusion sampling
provides uncertainty quantification.

Usage:
    # Training
    from models_3d.diffusion_refiner import train_refiner

    model, history = train_refiner(
        baseline_model=unet_model,
        forward_model=forward_model,
        epochs=100,
    )

    # Uncertainty Estimation (new interface)
    from models_3d.diffusion_refiner import UncertaintyEstimator, load_estimator

    estimator = load_estimator(
        baseline_checkpoint='path/to/baseline.pt',
        diffusion_checkpoint='path/to/refiner.pt',
        forward_model=forward_model,
    )

    result = estimator.predict(signal)
    print(f"DEI: {result.dei:.4f}, Confidence: {result.confidence:.4f}")
    high_risk = result.get_high_risk_regions(threshold=0.9)

Modules:
    config: Configuration classes for diffusion training
    scheduler: DDPM/DDIM noise schedule implementation
    model: Conditional U-Net denoiser architecture
    dataset: Dataset classes for refinement training
    loss: Combined loss functions
    train: Training pipeline
    inference: Uncertainty estimation via diffusion sampling
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
    UncertaintyEstimator,
    UncertaintyResult,
    RefinementInference,
    RefinementResult,
    load_estimator,
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
    # Inference (Uncertainty Estimation)
    'UncertaintyEstimator',
    'UncertaintyResult',
    'RefinementInference',  # Backward compatibility
    'RefinementResult',     # Backward compatibility
    'load_estimator',
    'load_refiner',         # Backward compatibility
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

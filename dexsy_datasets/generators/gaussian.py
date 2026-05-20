"""
Generator for Gaussian datasets (2C, 3C, and NC).

Uses the ForwardModel2D from dexsy_core.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np

from dexsy_core.forward_model import create_forward_model
from dexsy_core.forward_model_nc import create_forward_model_nc

from dexsy_datasets.config import DatasetConfig
from dexsy_datasets.core import ImmutableDataset, compute_dataset_id
from dexsy_datasets.seeds import SeedManager


def generate_gaussian_dataset(
    config: Union[DatasetConfig, Dict[str, Any]],
    output_dir: Optional[str] = None,
    generator_version: str = "1.0.0",
) -> ImmutableDataset:
    """
    Generate a Gaussian DEXSY dataset (2C/3C/NC).

    Args:
        config: DatasetConfig or config dict
        output_dir: If provided, save the dataset to this directory
        generator_version: Version string for reproducibility

    Returns:
        ImmutableDataset with signals and spectra
    """
    if isinstance(config, dict):
        config = DatasetConfig.from_dict(config)

    if config.task_type != "reconstruction":
        raise ValueError(
            f"gaussian generator requires task_type='reconstruction', "
            f"got '{config.task_type}'"
        )

    if config.model_type not in ["gaussian_2c", "gaussian_3c", "gaussian_nc"]:
        raise ValueError(
            f"gaussian generator requires model_type in "
            f"['gaussian_2c', 'gaussian_3c', 'gaussian_nc'], "
            f"got '{config.model_type}'"
        )

    if config.model_type == "gaussian_2c":
        n_compartments = 2
        forward_model = create_forward_model(n_b=config.n_b, profile=config.n_b)
    elif config.model_type == "gaussian_3c":
        n_compartments = 3
        forward_model = create_forward_model(n_b=config.n_b, profile=config.n_b)
    else:
        n_compartments = int(config.params.get("n_compartments", 3))
        forward_model = create_forward_model_nc(n_b=config.n_b, profile=config.n_b)

    seed_manager = SeedManager(config.seed)

    signals_list = []
    spectra_list = []
    params_list = []

    for split_name in ["train", "val", "test"]:
        n_split = getattr(config, f"n_{split_name}")
        if n_split <= 0:
            continue

        split_seed = seed_manager.get_generator_seed(split_name)

        if config.model_type == "gaussian_nc":
            split_spectra, split_signals, split_params, _ = forward_model.generate_batch(
                n_samples=n_split,
                N=n_compartments,
                noise_sigma=None,
                noise_sigma_range=config.params.get("noise_sigma_range", (0.005, 0.015)),
                return_reference_signal=True,
                seed=split_seed,
            )
        else:
            split_spectra, split_signals, split_params, _ = forward_model.generate_batch(
                n_samples=n_split,
                noise_sigma=None,
                noise_sigma_range=config.params.get("noise_sigma_range", (0.005, 0.015)),
                n_compartments=n_compartments,
                return_reference_signal=True,
                seed=split_seed,
            )

        signals_list.append(split_signals.astype(np.float32))
        spectra_list.append(split_spectra.astype(np.float32))
        params_list.extend(split_params)

    signals = np.concatenate(signals_list, axis=0).astype(np.float32)
    spectra = np.concatenate(spectra_list, axis=0).astype(np.float32)

    n_train = config.n_train
    n_val = config.n_val
    n_test = config.n_test

    splits = {
        "train": list(range(n_train)),
        "val": list(range(n_train, n_train + n_val)),
        "test": list(range(n_train + n_val, n_train + n_val + n_test)),
    }

    dataset = ImmutableDataset(
        dataset_id=compute_dataset_id(config.to_dict()),
        signals=signals,
        spectra=spectra,
        splits=splits,
        config=config.to_dict(),
        metadata={
            "task_type": config.task_type,
            "model_type": config.model_type,
            "n_b": config.n_b,
            "n_compartments": n_compartments,
        },
    )

    if output_dir is not None:
        from dexsy_datasets.storage import save_dataset
        save_dataset(dataset, output_dir, generator_version)

    return dataset

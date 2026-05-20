"""
Generator for 3-compartment non-Gaussian datasets.

Uses the ForwardModel3CNonGaussian from dexsy_core.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np

from dexsy_core.forward_model_3c_nongaussian import ForwardModel3CNonGaussian

from dexsy_datasets.config import DatasetConfig
from dexsy_datasets.core import ImmutableDataset, compute_dataset_id
from dexsy_datasets.seeds import SeedManager


def generate_nongaussian_dataset(
    config: Union[DatasetConfig, Dict[str, Any]],
    output_dir: Optional[str] = None,
    generator_version: str = "1.0.0",
) -> ImmutableDataset:
    """
    Generate a 3-compartment non-Gaussian DEXSY dataset.

    Args:
        config: DatasetConfig or config dict
        output_dir: If provided, save the dataset to this directory
        generator_version: Version string for reproducibility

    Returns:
        ImmutableDataset with signals, pathway_weights, and DEI
    """
    if isinstance(config, dict):
        config = DatasetConfig.from_dict(config)

    if config.task_type != "pathway_regression":
        raise ValueError(
            f"nongaussian generator requires task_type='pathway_regression', "
            f"got '{config.task_type}'"
        )

    if config.model_type != "nongaussian_3c":
        raise ValueError(
            f"nongaussian generator requires model_type='nongaussian_3c', "
            f"got '{config.model_type}'"
        )

    params = config.params

    forward_model = ForwardModel3CNonGaussian(
        n_b=config.n_b,
        mixing_time_range=params.get("mixing_time_range", (0.015, 0.300)),
    )

    seed_manager = SeedManager(config.seed)

    signals_list = []
    pathway_weights_list = []
    dei_list = []
    params_list = []

    for split_name in ["train", "val", "test"]:
        n_split = getattr(config, f"n_{split_name}")
        if n_split <= 0:
            continue

        split_seed = seed_manager.get_generator_seed(split_name)

        split_signals, split_params, _ = forward_model.sample_dataset(
            n_samples=n_split,
            seed=split_seed,
            return_clean_signals=False,
        )

        signals_list.append(split_signals.astype(np.float32))
        params_list.extend(split_params)

        for p in split_params:
            w_matrix = np.array(p.get("weight_matrix", np.eye(3)), dtype=np.float32)
            pathway_weights_list.append(w_matrix)

            dei_value = float(p.get("theoretical_dei", 0.0))
            dei_list.append(dei_value)

    signals = np.concatenate(signals_list, axis=0).astype(np.float32)
    pathway_weights = np.stack(pathway_weights_list, axis=0).astype(np.float32)
    dei = np.array(dei_list, dtype=np.float32)

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
        pathway_weights=pathway_weights,
        dei=dei,
        splits=splits,
        config=config.to_dict(),
        metadata={
            "task_type": config.task_type,
            "model_type": config.model_type,
            "n_b": config.n_b,
        },
    )

    if output_dir is not None:
        from dexsy_datasets.storage import save_dataset
        save_dataset(dataset, output_dir, generator_version)

    return dataset

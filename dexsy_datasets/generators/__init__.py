"""
Dataset generators.

Generates immutable datasets using the forward models.
"""

from dexsy_datasets.generators.nongaussian import generate_nongaussian_dataset
from dexsy_datasets.generators.gaussian import generate_gaussian_dataset

__all__ = ["generate_nongaussian_dataset", "generate_gaussian_dataset"]

"""Compatibility exports for canonical synthetic dataset generation."""

from tcspc_toolkit.simulation.datasets import (
    FloatArray,
    IntArray,
    SyntheticDataset,
    generate_monoexponential_dataset,
    generate_grouped_monoexponential_dataset,
)

__all__ = [
    "FloatArray",
    "IntArray",
    "SyntheticDataset",
    "generate_monoexponential_dataset",
    "generate_grouped_monoexponential_dataset",
]

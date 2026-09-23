"""
Dataset abstraction layer for the AI Driver Safety Platform.

Provides a pluggable interface so any dataset can be swapped in without
touching model or training code.  No dataset-specific files are committed here.
"""

from .base import DatasetAdapter, Sample, DatasetMetadata, DatasetSplit
from .config import DatasetConfig, load_dataset_config
from .registry import DatasetRegistry

__all__ = [
    "DatasetAdapter",
    "Sample",
    "DatasetMetadata",
    "DatasetSplit",
    "DatasetConfig",
    "load_dataset_config",
    "DatasetRegistry",
]

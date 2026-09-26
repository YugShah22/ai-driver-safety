"""
ml/cnn/dataset.py — compatibility shim.

The canonical Dataset class and transform factories now live in:
  ml/datasets/pytorch_dataset.py  ->  CNNImageDataset  (= FrameDataset)
  ml/datasets/transforms.py       ->  get_train_transforms / get_validation_transforms

This module re-exports everything under the original names so that
existing callers (trainer, evaluator, infer, tests) require no changes.
"""
from __future__ import annotations

from typing import Callable

from .config import CNNConfig
from ml.datasets.pytorch_dataset import CNNImageDataset
from ml.datasets.transforms import (
    get_train_transforms,
    get_validation_transforms,
)


# ── CNNImageDataset re-exported as FrameDataset ───────────────────────────────
FrameDataset = CNNImageDataset


# ── Transform factories that accept CNNConfig ─────────────────────────────────

def build_train_transforms(config: CNNConfig) -> Callable:
    """Training transform pipeline (delegates to get_train_transforms)."""
    return get_train_transforms(image_size=config.image_size)


def build_eval_transforms(config: CNNConfig) -> Callable:
    """Deterministic eval/inference transform pipeline (delegates to get_validation_transforms)."""
    return get_validation_transforms(image_size=config.image_size)


__all__ = ["FrameDataset", "build_train_transforms", "build_eval_transforms"]

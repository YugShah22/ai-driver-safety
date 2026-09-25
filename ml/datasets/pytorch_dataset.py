"""
pytorch_dataset.py — PyTorch Dataset adapter for standardized Sample objects.

Responsibility:
    Bridge the DatasetAdapter interface (which yields Sample objects) to
    torch.utils.data.Dataset, so that any adapter — regardless of the
    underlying dataset — can be fed directly into a DataLoader.

This module does NOT:
    - Contain dataset-specific parsing logic.
    - Read annotation files (JSON, tar, Hugging Face, etc.).
    - Implement augmentation, normalization, or resizing.
    - Know anything about the Indian Road Dataset or any other dataset.

Those concerns belong to the adapter (DatasetAdapter subclass) and the
transform pipeline respectively. This class only wraps a pre-built list
of Sample objects and applies an optional transform to each image.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

from ml.datasets.base import Sample


class CNNImageDataset(Dataset):
    """
    A generic PyTorch Dataset that wraps a list of Sample objects.

    Responsibilities:
        - Store a flat list of Sample objects produced by any DatasetAdapter.
        - Return (image_tensor, label) pairs for consumption by a DataLoader.
        - Optionally apply a transform (e.g. resize + normalize) to each image.

    This class is intentionally dataset-agnostic. It only knows about the
    Sample contract (image_path, label, metadata) — not about any specific
    dataset format, annotation structure, or file layout.

    Usage::

        samples = list(adapter.get_training_samples())
        dataset = CNNImageDataset(samples, transform=train_transform)
        loader  = DataLoader(dataset, batch_size=32, shuffle=True)

    Args:
        samples:   A list of Sample objects from a DatasetAdapter split.
        transform: An optional callable applied to the PIL image before it
                   is converted to a tensor. Expected signature:
                   ``transform(PIL.Image) -> torch.Tensor``
                   If None, the image will need to be converted manually
                   or a transform must be supplied before training.
    """

    def __init__(
        self,
        samples: list[Sample],
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image,sample.label

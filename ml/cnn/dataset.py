"""
PyTorch Dataset wrapper that bridges a DatasetAdapter to a DataLoader.

Responsibilities:
  - Accept a list of Sample objects from a DatasetAdapter split.
  - Apply configurable transforms (resize, normalise, augment).
  - Return (image_tensor, label) pairs compatible with DataLoader.

No dataset-specific preprocessing is hard-coded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import CNNConfig
from ml.datasets.base import Sample


def build_train_transforms(config: CNNConfig) -> Callable:
    """
    Return a transform pipeline suitable for training.

    Includes random flips and colour jitter as generic augmentations.
    Dataset-specific augmentations belong in the dataset adapter, not here.
    """
    h, w = config.image_size
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


def build_eval_transforms(config: CNNConfig) -> Callable:
    """Return a deterministic transform pipeline for validation / test / inference."""
    h, w = config.image_size
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


class FrameDataset(Dataset):
    """
    PyTorch Dataset wrapping a list of Sample objects from a DatasetAdapter.

    Args:
        samples:   Flat list of Sample objects (all from one split).
        transform: Torchvision-compatible transform applied to each PIL image.
    """

    def __init__(
        self,
        samples: list[Sample],
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        img = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, sample.label

    @staticmethod
    def from_adapter_split(
        samples_iter,
        transform: Optional[Callable] = None,
    ) -> "FrameDataset":
        """
        Build a FrameDataset by exhausting a sample iterator from an adapter.

        Args:
            samples_iter: Iterator[Sample] from DatasetAdapter.get_*_samples().
            transform:    Transform to apply.
        """
        return FrameDataset(list(samples_iter), transform=transform)

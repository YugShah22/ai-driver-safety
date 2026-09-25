"""
transforms.py — Image preprocessing pipelines for CNN training and inference.

Responsibility:
    Provide separate, composable transform pipelines for each data split:
      - Training:   may include data augmentation.
      - Validation: deterministic, no augmentation.
      - Test:       identical to validation (deterministic).

Design decisions:
    - All three functions accept an ``image_size`` argument so the pipeline
      dimensions are controlled by CNNConfig rather than hard-coded here.
    - The return type is ``torchvision.transforms.Compose``, which is
      compatible with CNNImageDataset's ``transform`` argument.
    - Augmentation parameters (flip probability, colour jitter strength, etc.)
      will be added as keyword arguments when implemented — the signatures
      are designed to accommodate them without breaking callers.
    - Normalisation statistics use ImageNet defaults as a starting point;
      they may be replaced with dataset-specific statistics once computed.

This module does NOT:
    - Know about the Indian Road Dataset or any specific dataset.
    - Read annotation files or images.
    - Perform model training.
"""
from __future__ import annotations

from typing import Tuple

from torchvision import transforms
from typing import Tuple


# Canonical ImageNet normalisation statistics — used as defaults.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def get_validation_transforms(
    image_size: Tuple[int, int] = (224, 224),
) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def get_test_transforms(
    image_size: Tuple[int, int] = (224, 224),
) -> transforms.Compose:
    return get_validation_transforms(image_size)

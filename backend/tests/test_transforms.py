"""
Unit tests for ml/datasets/transforms.py (Phase 5.4).

Covers currently implemented functions:
  - get_validation_transforms()
  - get_test_transforms()

Training augmentation is not tested here.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))  # d:\ai driver safety

from ml.datasets.transforms import get_validation_transforms, get_test_transforms,get_train_transforms
from torchvision import transforms as T


# ── helper ────────────────────────────────────────────────────────────────────

def small_rgb_image() -> Image.Image:
    """Return a tiny 8×8 RGB PIL image created entirely in memory."""
    return Image.new("RGB", (8, 8), color=(100, 150, 200))


# ── get_validation_transforms ─────────────────────────────────────────────────

class TestGetValidationTransforms:

    def test_returns_compose(self):
        pipeline = get_validation_transforms()
        assert isinstance(pipeline, T.Compose)

    def test_output_is_tensor(self):
        pipeline = get_validation_transforms()
        tensor = pipeline(small_rgb_image())
        assert isinstance(tensor, torch.Tensor)

    def test_default_output_shape(self):
        pipeline = get_validation_transforms()
        tensor = pipeline(small_rgb_image())
        assert tensor.shape == (3, 224, 224)

    def test_custom_image_size(self):
        pipeline = get_validation_transforms(image_size=(128, 128))
        tensor = pipeline(small_rgb_image())
        assert tensor.shape == (3, 128, 128)

    def test_is_deterministic(self):
        """Same image → same tensor on repeated calls."""
        pipeline = get_validation_transforms()
        img = small_rgb_image()
        t1 = pipeline(img)
        t2 = pipeline(img)
        assert torch.allclose(t1, t2)


# ── get_test_transforms ───────────────────────────────────────────────────────

class TestGetTestTransforms:

    def test_returns_compose(self):
        pipeline = get_test_transforms()
        assert isinstance(pipeline, T.Compose)

    def test_output_is_tensor(self):
        pipeline = get_test_transforms()
        tensor = pipeline(small_rgb_image())
        assert isinstance(tensor, torch.Tensor)

    def test_default_output_shape(self):
        pipeline = get_test_transforms()
        tensor = pipeline(small_rgb_image())
        assert tensor.shape == (3, 224, 224)

    def test_custom_image_size(self):
        pipeline = get_test_transforms(image_size=(64, 64))
        tensor = pipeline(small_rgb_image())
        assert tensor.shape == (3, 64, 64)

    def test_matches_validation_output(self):
        """Test transforms must produce identical output to validation transforms."""
        img = small_rgb_image()
        val_tensor  = get_validation_transforms()(img)
        test_tensor = get_test_transforms()(img)
        assert torch.allclose(val_tensor, test_tensor)

class TestGetTrainTransforms:
    def test_returns_compose(self):
        pipeline = get_train_transforms()
        assert isinstance(pipeline, T.Compose)

    def test_output_is_tensor(self):
        pipeline = get_train_transforms()
        tensor = pipeline(small_rgb_image())
        assert isinstance(tensor, torch.Tensor)

    def test_default_output_shape(self):
        pipeline = get_train_transforms()
        tensor = pipeline(small_rgb_image())
        assert tensor.shape == (3, 224, 224)

    def test_custom_image_size(self):
        pipeline = get_train_transforms(image_size=(128, 128))
        tensor = pipeline(small_rgb_image())
        assert tensor.shape == (3, 128, 128)

    def test_contains_horizontal_flip(self):
        pipeline = get_train_transforms()
        transform_types = [type(transform) for transform in pipeline.transforms]

        assert T.RandomHorizontalFlip in transform_types

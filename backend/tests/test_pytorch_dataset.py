"""
Unit tests for CNNImageDataset (Phase 5.4).

Covers the behaviour currently implemented:
  - len(dataset) returns the correct number of samples.
  - dataset[0] returns a (image, label) tuple.
  - The image is a PIL.Image.Image when no transform is supplied.
  - The label matches the integer stored on the Sample.

No torchvision transforms, DataLoader, or dataset-specific logic is tested here.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))  # d:\ai driver safety

from ml.datasets.base import Sample
from ml.datasets.pytorch_dataset import CNNImageDataset
from torchvision import transforms


# ── helpers ───────────────────────────────────────────────────────────────────

def make_jpeg(directory: Path, filename: str = "frame.jpg") -> Path:
    """Write a tiny 4×4 RGB JPEG into *directory* and return its path."""
    path = directory / filename
    img = Image.new("RGB", (4, 4), color=(128, 64, 32))
    img.save(path, format="JPEG")
    return path


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def single_sample(tmp_path: Path) -> tuple[Sample, Path]:
    img_path = make_jpeg(tmp_path)
    sample = Sample(image_path=img_path, label=3)
    return sample, img_path


@pytest.fixture
def multi_sample(tmp_path: Path) -> list[Sample]:
    samples = []
    for i in range(5):
        img_path = make_jpeg(tmp_path, f"frame_{i:02d}.jpg")
        samples.append(Sample(image_path=img_path, label=i % 6))
    return samples


# ── tests ─────────────────────────────────────────────────────────────────────

class TestCNNImageDatasetLen:

    def test_len_single_sample(self, single_sample):
        sample, _ = single_sample
        ds = CNNImageDataset([sample])
        assert len(ds) == 1

    def test_len_multiple_samples(self, multi_sample):
        ds = CNNImageDataset(multi_sample)
        assert len(ds) == 5

    def test_len_empty(self):
        ds = CNNImageDataset([])
        assert len(ds) == 0


class TestCNNImageDatasetGetItem:

    def test_getitem_returns_tuple(self, single_sample):
        sample, _ = single_sample
        ds = CNNImageDataset([sample])
        result = ds[0]
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_image_is_pil_without_transform(self, single_sample):
        sample, _ = single_sample
        ds = CNNImageDataset([sample])
        image, _ = ds[0]
        assert isinstance(image, Image.Image)

    def test_image_mode_is_rgb(self, single_sample):
        sample, _ = single_sample
        ds = CNNImageDataset([sample])
        image, _ = ds[0]
        assert image.mode == "RGB"

    def test_label_matches_sample(self, single_sample):
        sample, _ = single_sample
        ds = CNNImageDataset([sample])
        _, label = ds[0]
        assert label == 3

    def test_label_type_is_int(self, single_sample):
        sample, _ = single_sample
        ds = CNNImageDataset([sample])
        _, label = ds[0]
        assert isinstance(label, int)

    def test_correct_label_for_each_index(self, multi_sample):
        ds = CNNImageDataset(multi_sample)
        for i, sample in enumerate(multi_sample):
            _, label = ds[i]
            assert label == sample.label
        
    def test_transform_converts_image_to_tensor(self, single_sample):

        sample, _ = single_sample

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        ds = CNNImageDataset([sample], transform=transform)

        image, label = ds[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert label == 3

    
        

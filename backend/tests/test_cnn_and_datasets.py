"""
Tests for the dataset abstraction layer and CNN infrastructure.

All tests use tiny synthetic data — no real dataset files are required.
Supabase/storage calls are not involved in this phase's tests.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))  # d:\ai driver safety

from ml.datasets.base import DatasetAdapter, DatasetMetadata, DatasetSplit, Sample
from ml.datasets.config import DatasetConfig, load_dataset_config
from ml.datasets.registry import DatasetRegistry
from ml.datasets.adapters.generic import GenericFolderAdapter
from ml.cnn.config import CNNConfig
from ml.cnn.model import SceneCNN
from ml.cnn.dataset import FrameDataset, build_train_transforms, build_eval_transforms
from ml.cnn.infer import Inferencer, InferenceResult
from ml.cnn.trainer import Trainer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tmp_dataset(tmp_path: Path) -> Path:
    """
    Build a minimal folder-per-class dataset under tmp_path:

        images/
            train/
                cat/  (3 images)
                dog/  (3 images)
            val/
                cat/  (1 image)
                dog/  (1 image)
            test/
                cat/  (1 image)
                dog/  (1 image)
    """
    classes = ["cat", "dog"]
    splits  = {"train": 3, "val": 1, "test": 1}

    for split, n in splits.items():
        for cls in classes:
            cls_dir = tmp_path / "images" / split / cls
            cls_dir.mkdir(parents=True)
            for i in range(n):
                img = Image.fromarray(
                    (np.random.randint(0, 255, (32, 32, 3))).astype("uint8"), "RGB"
                )
                img.save(cls_dir / f"img_{i:02d}.jpg")

    return tmp_path


@pytest.fixture
def dataset_config(tmp_dataset: Path) -> DatasetConfig:
    return DatasetConfig(
        name="synthetic_test",
        root=tmp_dataset,
        image_dir="images",
        annot_dir="annotations",
        train_split="train",
        val_split="val",
        test_split="test",
        adapter="generic",
    )


@pytest.fixture
def loaded_adapter(dataset_config: DatasetConfig) -> GenericFolderAdapter:
    adapter = GenericFolderAdapter(dataset_config)
    adapter.load_annotations()
    return adapter


@pytest.fixture
def small_cfg() -> CNNConfig:
    """Tiny CNNConfig that runs fast on CPU."""
    return CNNConfig(
        num_classes=2,
        image_size=(32, 32),
        batch_size=2,
        learning_rate=1e-3,
        epochs=1,
        device="cpu",
        checkpoint_dir=tempfile.mkdtemp(),
        num_workers=0,
    )


@pytest.fixture
def small_model(small_cfg: CNNConfig) -> SceneCNN:
    return SceneCNN(small_cfg)


@pytest.fixture
def synthetic_image() -> Image.Image:
    arr = (np.random.randint(0, 255, (64, 64, 3))).astype("uint8")
    return Image.fromarray(arr, "RGB")


# =============================================================================
# Dataset Adapter Interface
# =============================================================================

class TestDatasetAdapterInterface:
    """Verify the abstract interface is enforced correctly."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            DatasetAdapter(config=MagicMock())

    def test_concrete_must_implement_all_methods(self):
        """A partial implementation raises TypeError at instantiation."""
        class Partial(DatasetAdapter):
            def get_metadata(self): ...
            # missing 5 abstract methods

        with pytest.raises(TypeError):
            Partial(config=MagicMock())

    def test_dataset_split_constants(self):
        assert DatasetSplit.TRAIN == "train"
        assert DatasetSplit.VAL   == "val"
        assert DatasetSplit.TEST  == "test"

    def test_get_samples_dispatches_correctly(self, loaded_adapter):
        train_samples = list(loaded_adapter.get_samples(DatasetSplit.TRAIN))
        assert len(train_samples) == 6  # 3 cat + 3 dog

    def test_get_samples_raises_on_unknown_split(self, loaded_adapter):
        with pytest.raises(ValueError, match="Unknown split"):
            list(loaded_adapter.get_samples("unknown_split"))


# =============================================================================
# GenericFolderAdapter
# =============================================================================

class TestGenericFolderAdapter:

    def test_load_annotations_discovers_classes(self, loaded_adapter):
        names = loaded_adapter.get_class_names()
        assert set(names) == {"cat", "dog"}

    def test_load_annotations_requires_existing_dir(self, dataset_config, tmp_path):
        bad_cfg = DatasetConfig(
            name="bad", root=tmp_path / "nonexistent",
            train_split="train", adapter="generic",
        )
        adapter = GenericFolderAdapter(bad_cfg)
        with pytest.raises(FileNotFoundError):
            adapter.load_annotations()

    def test_must_call_load_annotations_first(self, dataset_config):
        adapter = GenericFolderAdapter(dataset_config)
        with pytest.raises(RuntimeError, match="load_annotations"):
            adapter.get_class_names()

    def test_training_samples_count(self, loaded_adapter):
        samples = list(loaded_adapter.get_training_samples())
        assert len(samples) == 6  # 2 classes x 3 images

    def test_validation_samples_count(self, loaded_adapter):
        samples = list(loaded_adapter.get_validation_samples())
        assert len(samples) == 2

    def test_test_samples_count(self, loaded_adapter):
        samples = list(loaded_adapter.get_test_samples())
        assert len(samples) == 2

    def test_sample_fields(self, loaded_adapter):
        sample = next(loaded_adapter.get_training_samples())
        assert isinstance(sample.image_path, Path)
        assert sample.image_path.exists()
        assert sample.label in (0, 1)

    def test_metadata(self, loaded_adapter):
        meta = loaded_adapter.get_metadata()
        assert meta.num_classes == 2
        assert meta.total_samples == 10  # 6 train + 2 val + 2 test
        assert meta.name == "synthetic_test"

    def test_num_classes_helper(self, loaded_adapter):
        assert loaded_adapter.num_classes() == 2


# =============================================================================
# DatasetConfig
# =============================================================================

class TestDatasetConfig:

    def test_default_values(self):
        cfg = DatasetConfig()
        assert cfg.train_split == "train"
        assert cfg.val_split   == "val"
        assert cfg.test_split  == "test"

    def test_image_path_is_root_slash_image_dir(self, tmp_path):
        cfg = DatasetConfig(root=tmp_path, image_dir="frames")
        assert cfg.image_path == tmp_path / "frames"

    def test_load_from_json(self, tmp_path):
        cfg_file = tmp_path / "dataset.json"
        cfg_file.write_text(json.dumps({
            "name":       "JsonDataset",
            "root":       str(tmp_path),
            "train_split":"train",
            "val_split":  "val",
            "test_split": "test",
            "adapter":    "generic",
        }))
        cfg = load_dataset_config(cfg_file)
        assert cfg.name == "JsonDataset"
        assert cfg.root == tmp_path

    def test_load_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATASET_NAME", "EnvDataset")
        monkeypatch.setenv("DATASET_ROOT", str(tmp_path))
        cfg = load_dataset_config()
        assert cfg.name == "EnvDataset"

    def test_load_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_dataset_config("/no/such/path/dataset.json")


# =============================================================================
# DatasetRegistry
# =============================================================================

class TestDatasetRegistry:

    def test_generic_is_pre_registered(self):
        assert "generic" in DatasetRegistry.list_registered()

    def test_get_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            DatasetRegistry.get("totally_unknown_adapter_xyz")

    def test_build_instantiates_correct_class(self, dataset_config):
        adapter = DatasetRegistry.build(dataset_config)
        assert isinstance(adapter, GenericFolderAdapter)


# =============================================================================
# CNN Model
# =============================================================================

class TestSceneCNN:

    def test_forward_pass_shape(self, small_model, small_cfg):
        x = torch.randn(2, 3, *small_cfg.image_size)
        out = small_model(x)
        assert out.shape == (2, small_cfg.num_classes)

    def test_parameter_count_positive(self, small_model):
        assert small_model.count_parameters() > 0

    def test_different_num_classes(self, small_cfg):
        cfg = CNNConfig(num_classes=5, image_size=(32, 32), device="cpu")
        model = SceneCNN(cfg)
        x = torch.randn(1, 3, 32, 32)
        assert model(x).shape == (1, 5)


# =============================================================================
# FrameDataset & Transforms
# =============================================================================

class TestFrameDataset:

    def test_len(self, loaded_adapter, small_cfg):
        samples = list(loaded_adapter.get_training_samples())
        ds = FrameDataset(samples, transform=build_eval_transforms(small_cfg))
        assert len(ds) == 6

    def test_getitem_returns_tensor_and_label(self, loaded_adapter, small_cfg):
        samples = list(loaded_adapter.get_training_samples())
        ds = FrameDataset(samples, transform=build_eval_transforms(small_cfg))
        img, lbl = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, *small_cfg.image_size)
        assert isinstance(lbl, int)

    def test_train_transforms_produce_correct_shape(self, small_cfg, synthetic_image):
        t = build_train_transforms(small_cfg)
        tensor = t(synthetic_image)
        assert tensor.shape == (3, *small_cfg.image_size)

    def test_eval_transforms_are_deterministic(self, small_cfg, synthetic_image):
        t = build_eval_transforms(small_cfg)
        a = t(synthetic_image)
        b = t(synthetic_image)
        assert torch.allclose(a, b)


# =============================================================================
# Training Configuration & Checkpoints
# =============================================================================

class TestTrainerAndCheckpoints:

    def test_save_and_load_checkpoint(self, small_model, small_cfg, tmp_path):
        small_cfg.checkpoint_dir = str(tmp_path)
        ckpt_path = tmp_path / "test.pt"
        torch.save({
            "model_state": small_model.state_dict(),
            "num_classes": small_cfg.num_classes,
            "image_size":  small_cfg.image_size,
        }, ckpt_path)

        loaded = Trainer.load_checkpoint(ckpt_path, small_cfg, device="cpu")
        assert isinstance(loaded, SceneCNN)
        loaded.eval()

    def test_checkpoint_class_mismatch_raises(self, small_model, small_cfg, tmp_path):
        ckpt_path = tmp_path / "mismatch.pt"
        torch.save({
            "model_state": small_model.state_dict(),
            "num_classes": 99,  # wrong
            "image_size":  small_cfg.image_size,
        }, ckpt_path)
        with pytest.raises(ValueError, match="num_classes"):
            Trainer.load_checkpoint(ckpt_path, small_cfg, device="cpu")

    def test_cnn_config_resolve_device(self):
        cfg = CNNConfig(device="cpu")
        assert cfg.resolve_device() == "cpu"

    def test_cnn_config_auto_device(self):
        cfg = CNNConfig(device="auto")
        result = cfg.resolve_device()
        assert result in ("cpu", "cuda")


# =============================================================================
# Inferencer
# =============================================================================

class TestInferencer:

    def test_predict_returns_inference_result(self, small_model, small_cfg, synthetic_image):
        infer = Inferencer(small_model, small_cfg, class_names=["cat", "dog"])
        result = infer.predict(synthetic_image)

        assert isinstance(result, InferenceResult)
        assert result.predicted_class in (0, 1)
        assert result.predicted_label in ("cat", "dog")
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.probabilities) == 2
        assert abs(sum(result.probabilities) - 1.0) < 1e-4

    def test_predict_from_file_path(self, small_model, small_cfg, tmp_path, synthetic_image):
        img_path = tmp_path / "test_frame.jpg"
        synthetic_image.save(str(img_path))

        infer = Inferencer(small_model, small_cfg, class_names=["cat", "dog"])
        result = infer.predict(img_path)
        assert isinstance(result, InferenceResult)

    def test_from_checkpoint_factory(self, small_model, small_cfg, tmp_path, synthetic_image):
        ckpt_path = tmp_path / "model.pt"
        torch.save({
            "model_state": small_model.state_dict(),
            "num_classes": small_cfg.num_classes,
            "image_size":  small_cfg.image_size,
        }, ckpt_path)
        small_cfg.checkpoint_dir = str(tmp_path)

        infer = Inferencer.from_checkpoint(ckpt_path, small_cfg, class_names=["cat", "dog"])
        result = infer.predict(synthetic_image)
        assert isinstance(result, InferenceResult)

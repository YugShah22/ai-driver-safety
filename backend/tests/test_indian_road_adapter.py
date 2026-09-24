"""
Tests for IndianRoadAdapter — skeleton / interface level only.

These tests verify:
  - The adapter is registered in DatasetRegistry under "indian_road"
  - The class exposes the correct interface (is a DatasetAdapter subclass)
  - SCENE_CLASSES has the correct 6 values in the correct order
  - The constructor accepts a DatasetConfig without error
  - _ensure_loaded() raises RuntimeError before load_annotations()
  - get_class_names() returns SCENE_CLASSES without requiring load_annotations()
  - load_annotations() raises NotImplementedError (not yet implemented)
  - get_metadata(), get_training_samples(), etc. raise RuntimeError before
    load_annotations(), and NotImplementedError after (TODO stubs)

No real dataset files are required.
No HuggingFace network access is made.
No model training is performed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).parents[2]))  # d:\ai driver safety

from ml.datasets.base import DatasetAdapter, DatasetSplit
from ml.datasets.config import DatasetConfig
from ml.datasets.registry import DatasetRegistry
from ml.datasets.adapters.indian_road import IndianRoadAdapter, SCENE_CLASSES


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def minimal_config(tmp_path: Path) -> DatasetConfig:
    """Minimal DatasetConfig pointing at a temp root — no real files needed."""
    return DatasetConfig(
        name="indian-road-test",
        root=tmp_path,
        annot_dir="annotations",
        adapter="indian_road",
        extra={"source": "local"},
    )


@pytest.fixture
def adapter(minimal_config: DatasetConfig) -> IndianRoadAdapter:
    """Unloaded IndianRoadAdapter (load_annotations() NOT called)."""
    return IndianRoadAdapter(minimal_config)


# =============================================================================
# Registry
# =============================================================================

class TestIndianRoadRegistration:

    def test_registered_under_correct_key(self):
        assert "indian_road" in DatasetRegistry.list_registered()

    def test_registry_returns_correct_class(self, minimal_config):
        cls = DatasetRegistry.get("indian_road")
        assert cls is IndianRoadAdapter

    def test_registry_build_returns_instance(self, minimal_config):
        instance = DatasetRegistry.build(minimal_config)
        assert isinstance(instance, IndianRoadAdapter)


# =============================================================================
# Class structure / inheritance
# =============================================================================

class TestIndianRoadClassStructure:

    def test_is_dataset_adapter_subclass(self):
        assert issubclass(IndianRoadAdapter, DatasetAdapter)

    def test_constructor_accepts_config(self, minimal_config):
        adapter = IndianRoadAdapter(minimal_config)
        assert adapter.config is minimal_config

    def test_config_stored_correctly(self, minimal_config):
        adapter = IndianRoadAdapter(minimal_config)
        assert adapter.config.name == "indian-road-test"
        assert adapter.config.adapter == "indian_road"


# =============================================================================
# SCENE_CLASSES — confirmed from real dataset inspection
# =============================================================================

class TestSceneClasses:

    def test_exactly_six_classes(self):
        assert len(SCENE_CLASSES) == 6

    def test_class_names_exact_strings(self):
        expected = [
            "city street",
            "highway",
            "parking lot",
            "residential road",
            "tunnel",
            "village road",
        ]
        assert SCENE_CLASSES == expected

    def test_classes_are_alphabetically_ordered(self):
        assert SCENE_CLASSES == sorted(SCENE_CLASSES)

    def test_label_indices_match_list_position(self):
        # city street → 0, highway → 1, etc.
        assert SCENE_CLASSES.index("city street")       == 0
        assert SCENE_CLASSES.index("highway")           == 1
        assert SCENE_CLASSES.index("parking lot")       == 2
        assert SCENE_CLASSES.index("residential road")  == 3
        assert SCENE_CLASSES.index("tunnel")            == 4
        assert SCENE_CLASSES.index("village road")      == 5

    def test_get_class_names_returns_scene_classes(self, adapter):
        # get_class_names() must work WITHOUT calling load_annotations()
        assert adapter.get_class_names() == SCENE_CLASSES

    def test_num_classes_is_six(self, adapter):
        # num_classes() is a concrete helper on DatasetAdapter
        assert adapter.num_classes() == 6


# =============================================================================
# Before load_annotations() is called
# =============================================================================

class TestBeforeLoadAnnotations:

    def test_ensure_loaded_raises_runtime_error(self, adapter):
        with pytest.raises(RuntimeError, match="load_annotations"):
            adapter._ensure_loaded()

    def test_get_metadata_raises_runtime_error(self, adapter):
        with pytest.raises(RuntimeError, match="load_annotations"):
            adapter.get_metadata()

    def test_get_training_samples_raises_runtime_error(self, adapter):
        with pytest.raises(RuntimeError, match="load_annotations"):
            next(iter(adapter.get_training_samples()))

    def test_get_validation_samples_raises_runtime_error(self, adapter):
        with pytest.raises(RuntimeError, match="load_annotations"):
            next(iter(adapter.get_validation_samples()))

    def test_get_test_samples_raises_runtime_error(self, adapter):
        with pytest.raises(RuntimeError, match="load_annotations"):
            next(iter(adapter.get_test_samples()))


# =============================================================================
# load_annotations() — stub behavior (NotImplementedError until you implement)
# =============================================================================

class TestLoadAnnotationStub:

    def test_load_annotations_missing_scene_file_raises_file_not_found(self, adapter):
        with pytest.raises(FileNotFoundError, match="scene_attributes.json not found"):
            adapter.load_annotations()

# =============================================================================
# Config extra key defaults
# =============================================================================

class TestConfigDefaults:

    def test_default_source_is_local(self, adapter):
        assert adapter._source == "local"

    def test_default_hf_repo(self, adapter):
        assert "thirdeyelabs" in adapter._hf_repo

    def test_default_min_confidence(self, adapter):
        assert adapter._min_confidence == 0.5

    def test_custom_min_confidence(self, minimal_config, tmp_path):
        cfg = DatasetConfig(
            name="test", root=tmp_path,
            adapter="indian_road",
            extra={"source": "local", "min_confidence": "0.7"},
        )
        a = IndianRoadAdapter(cfg)
        assert a._min_confidence == pytest.approx(0.7)

    def test_custom_source_hf_hub(self, tmp_path):
        cfg = DatasetConfig(
            name="test", root=tmp_path,
            adapter="indian_road",
            extra={"source": "hf_hub"},
        )
        a = IndianRoadAdapter(cfg)
        assert a._source == "hf_hub"

class TestLocalFrameIterator:
    def test_iter_frames_local(self, adapter, tmp_path):
        frames_dir = tmp_path / "frames" / "clip_001"
        frames_dir.mkdir(parents=True)

        for frame_name in ["0000.jpg", "0001.jpg", "0002.jpg"]:
            (frames_dir / frame_name).touch()

        adapter.config.data_root = tmp_path

        results = list(
            adapter._iter_frames_local(["clip_001"])
        )

        assert len(results) == 3
        assert results[0][1] == "clip_001"
        assert results[0][2] == 0
        assert results[1][2] == 1
        assert results[2][2] == 2

    def test_iter_frames_tar(self, adapter, tmp_path):
        shards_dir = tmp_path / "shards"
        shards_dir.mkdir(parents=True)

        tar_path = shards_dir / "train-00000-of-00646.tar"

        clip_1 = "a1b2c3d4-e5f6-7890-abcd-123456789abc"
        clip_2 = "b1c2d3e4-f5a6-7890-bcde-234567890abc"

        with tarfile.open(tar_path, "w") as tar:
            for name in [
                f"{clip_1}_0000.jpg",
                f"{clip_1}_0001.jpg",
                f"{clip_2}_0000.jpg",
                "unrelated.txt",
            ]:
                file_path = tmp_path / name
                file_path.write_bytes(b"fake image data")
                tar.add(file_path, arcname=name)

        adapter.config.data_root = tmp_path
        adapter._shards_subdir = "shards"

        results = list(
            adapter._iter_frames_tar([clip_1])
        )

        assert len(results) == 2

        assert results[0][1] == clip_1
        assert results[0][2] == 0

        assert results[1][1] == clip_1
        assert results[1][2] == 1

        assert results[0][0].exists()
        assert results[1][0].exists()

    def test_iter_frames_hf_hub(self, adapter, tmp_path, monkeypatch):
        clip_1 = "a1b2c3d4-e5f6-7890-abcd-123456789abc"
        clip_2 = "b1c2d3e4-f5a6-7890-bcde-234567890abc"

        tar_path = tmp_path / "fake_hf_shard.tar"

        with tarfile.open(tar_path, "w") as tar:
            for name in [
                f"{clip_1}_0000.jpg",
                f"{clip_1}_0001.jpg",
                f"{clip_2}_0000.jpg",
                "unrelated.txt",
            ]:
                file_path = tmp_path / name
                file_path.write_bytes(b"fake image data")
                tar.add(file_path, arcname=name)

        def fake_hf_hub_download(
            repo_id,
            filename,
            repo_type,
        ):
            return str(tar_path)

        monkeypatch.setattr(
            "ml.datasets.adapters.indian_road.hf_hub_download",
            fake_hf_hub_download,
        )

        adapter._hf_repo = "fake/repo"

        results = []

        for result in adapter._iter_frames_hf_hub([clip_1]):
            results.append(result)

            if len(results) == 2:
                break

        assert len(results) == 2

        assert results[0][1] == clip_1
        assert results[0][2] == 0

        assert results[1][1] == clip_1
        assert results[1][2] == 1

        assert results[0][0].exists()
        assert results[1][0].exists()
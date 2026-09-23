"""
Dataset configuration — loaded from environment variables or a YAML/JSON file.

No dataset-specific paths are hard-coded here; the project owner supplies them
via a .env file or an explicit config dict.

Environment variables (all prefixed with DATASET_):
    DATASET_NAME       — human-readable dataset name
    DATASET_ROOT       — root directory that contains the dataset
    DATASET_IMAGE_DIR  — subdirectory with raw images/frames (relative to root)
    DATASET_ANNOT_DIR  — subdirectory with annotation files (relative to root)
    DATASET_TRAIN_SPLIT— name of the training split directory or file stem
    DATASET_VAL_SPLIT  — name of the validation split directory or file stem
    DATASET_TEST_SPLIT — name of the test split directory or file stem
    DATASET_ADAPTER    — registry key of the adapter class to use
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DatasetConfig:
    """
    All dataset-level settings consumed by DatasetAdapter subclasses.

    Attributes:
        name:         Human-readable dataset identifier.
        root:         Absolute path to the dataset root directory on disk.
        image_dir:    Path to images relative to root (default: "images").
        annot_dir:    Path to annotations relative to root (default: "annotations").
        train_split:  Name of the training split (default: "train").
        val_split:    Name of the validation split (default: "val").
        test_split:   Name of the test split (default: "test").
        adapter:      Registry key of the DatasetAdapter subclass to load.
        extra:        Escape hatch for dataset-specific key/value settings.
    """

    name: str = "unknown"
    root: Path = field(default_factory=lambda: Path("."))
    image_dir: str = "images"
    annot_dir: str = "annotations"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    adapter: str = "generic"
    extra: dict[str, str] = field(default_factory=dict)

    # ---- Derived helpers ----------------------------------------------------

    @property
    def image_path(self) -> Path:
        """Full path to the image directory."""
        return self.root / self.image_dir

    @property
    def annot_path(self) -> Path:
        """Full path to the annotation directory."""
        return self.root / self.annot_dir

    @property
    def train_image_path(self) -> Path:
        return self.image_path / self.train_split

    @property
    def val_image_path(self) -> Path:
        return self.image_path / self.val_split

    @property
    def test_image_path(self) -> Path:
        return self.image_path / self.test_split


def load_dataset_config(path: Optional[str | Path] = None) -> DatasetConfig:
    """
    Load a DatasetConfig from a JSON file *or* from environment variables.

    Priority (highest to lowest):
      1. Explicit JSON/dict file at ``path``.
      2. Environment variables prefixed with ``DATASET_``.
      3. Built-in defaults.

    Args:
        path: Optional path to a JSON config file.  If None, only env vars
              and defaults are used.

    Returns:
        A populated DatasetConfig instance.

    Example JSON file::

        {
          "name": "MyDashcamDataset",
          "root": "/data/mydashcam",
          "image_dir": "frames",
          "annot_dir": "labels",
          "train_split": "train",
          "val_split":   "val",
          "test_split":  "test",
          "adapter":     "generic"
        }
    """
    raw: dict[str, str] = {}

    # 1. Load from JSON file if provided
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

    # 2. Overlay / fill from environment variables
    env_map = {
        "DATASET_NAME":        "name",
        "DATASET_ROOT":        "root",
        "DATASET_IMAGE_DIR":   "image_dir",
        "DATASET_ANNOT_DIR":   "annot_dir",
        "DATASET_TRAIN_SPLIT": "train_split",
        "DATASET_VAL_SPLIT":   "val_split",
        "DATASET_TEST_SPLIT":  "test_split",
        "DATASET_ADAPTER":     "adapter",
    }
    for env_key, cfg_key in env_map.items():
        val = os.environ.get(env_key)
        if val:
            raw[cfg_key] = val

    # Collect DATASET_EXTRA_* env vars into the extra dict
    extra: dict[str, str] = {
        k[len("DATASET_EXTRA_"):].lower(): v
        for k, v in os.environ.items()
        if k.startswith("DATASET_EXTRA_")
    }
    if extra:
        raw.setdefault("extra", {})
        raw["extra"].update(extra)

    # Convert root to Path
    if "root" in raw:
        raw["root"] = Path(raw["root"])

    return DatasetConfig(**{k: v for k, v in raw.items()})

"""
GenericFolderAdapter — a dataset-agnostic adapter for the common layout:

    <root>/
        images/
            train/
                class_a/img1.jpg
                class_b/img2.jpg
            val/
                class_a/img3.jpg
            test/
                class_a/img4.jpg

Each class is represented by a sub-folder name under the split directory.
No annotation files are required — the folder structure IS the annotation.

This is intentionally dataset-neutral.  When the project owner chooses a
specific dataset, they either:
  a) Organise it into this folder structure, OR
  b) Write a dedicated adapter that replaces this one.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from ..base import DatasetAdapter, DatasetMetadata, DatasetSplit, Sample
from ..config import DatasetConfig
from ..registry import DatasetRegistry

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class GenericFolderAdapter(DatasetAdapter):
    """
    Pluggable adapter for any image-classification dataset organised as
    one sub-folder per class under each split directory.

    The class list is derived at runtime from the folder names — no labels
    are hard-coded here.
    """

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self._class_names: list[str] = []
        self._annotations_loaded: bool = False

    # -------------------------------------------------------------------------

    def load_annotations(self) -> None:
        """
        Discover class names from the training split folder structure.
        Must be called before iterating samples.
        """
        train_dir = self.config.train_image_path
        if not train_dir.exists():
            raise FileNotFoundError(
                f"Training split directory not found: {train_dir}. "
                f"Ensure DATASET_ROOT and DATASET_TRAIN_SPLIT are configured correctly."
            )

        # Class names = sorted sub-folder names (folders only, no hidden dirs)
        self._class_names = sorted(
            d.name for d in train_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        if not self._class_names:
            raise ValueError(
                f"No class sub-folders found in {train_dir}. "
                f"Expected at least one sub-folder per class."
            )

        self._annotations_loaded = True
        logger.info(
            "[GenericFolderAdapter] Loaded %d classes from %s: %s",
            len(self._class_names), train_dir,
            ", ".join(self._class_names[:5]) + (" ..." if len(self._class_names) > 5 else ""),
        )

    def get_class_names(self) -> list[str]:
        self._ensure_loaded()
        return list(self._class_names)

    def get_metadata(self) -> DatasetMetadata:
        self._ensure_loaded()
        split_counts: dict[str, int] = {}
        for split_name in (
            self.config.train_split,
            self.config.val_split,
            self.config.test_split,
        ):
            split_counts[split_name] = self._count_split(
                self.config.image_path / split_name
            )

        total = sum(split_counts.values())
        return DatasetMetadata(
            name=self.config.name,
            version="n/a",
            num_classes=len(self._class_names),
            class_names=self._class_names,
            total_samples=total,
            split_counts=split_counts,
        )

    def get_training_samples(self) -> Iterator[Sample]:
        return self._iter_split(self.config.train_image_path)

    def get_validation_samples(self) -> Iterator[Sample]:
        return self._iter_split(self.config.val_image_path)

    def get_test_samples(self) -> Iterator[Sample]:
        return self._iter_split(self.config.test_image_path)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._annotations_loaded:
            raise RuntimeError(
                "load_annotations() must be called before accessing samples or metadata."
            )

    def _iter_split(self, split_dir: Path) -> Iterator[Sample]:
        """Yield Sample objects for every image under ``split_dir``."""
        self._ensure_loaded()
        if not split_dir.exists():
            logger.warning("[GenericFolderAdapter] Split dir does not exist: %s", split_dir)
            return

        class_index = {name: idx for idx, name in enumerate(self._class_names)}

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith("."):
                continue
            label = class_index.get(class_dir.name)
            if label is None:
                logger.warning(
                    "[GenericFolderAdapter] Unknown class folder '%s' in %s - skipping",
                    class_dir.name, split_dir,
                )
                continue
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in SUPPORTED_EXTS:
                    yield Sample(
                        image_path=img_file,
                        label=label,
                        metadata={"class_name": class_dir.name, "split": split_dir.name},
                    )

    def _count_split(self, split_dir: Path) -> int:
        if not split_dir.exists():
            return 0
        return sum(
            1
            for p in split_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        )


# Auto-register under the key "generic"
DatasetRegistry.register("generic", GenericFolderAdapter)

"""
Abstract base class for all dataset adapters.

Every dataset plug-in must subclass DatasetAdapter and implement the
abstract methods.  The core CNN training/inference code only ever talks
to this interface — never to a concrete dataset module.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional


@dataclass
class Sample:
    """
    One labelled data sample returned by a dataset adapter.

    Attributes:
        image_path:  Absolute path to the image file on disk.
        label:       Integer class index.  The mapping from index to name is
                     defined by DatasetAdapter.get_class_names().
        metadata:    Optional dict of arbitrary per-sample metadata
                     (e.g. scene type, weather, time-of-day).  Adapters may
                     leave this empty; model code must never require it.
    """

    image_path: Path
    label: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetMetadata:
    """
    Top-level metadata returned by DatasetAdapter.get_metadata().

    Attributes:
        name:          Human-readable dataset name (e.g. "MyDashcamDataset-v1").
        version:       Dataset version string.
        num_classes:   Total number of output classes.
        class_names:   Ordered list of class name strings.
        total_samples: Number of samples across all splits combined.
        split_counts:  Per-split sample counts keyed by split name.
        description:   Optional free-text description.
    """

    name: str
    version: str
    num_classes: int
    class_names: list[str]
    total_samples: int
    split_counts: dict[str, int] = field(default_factory=dict)
    description: Optional[str] = None


class DatasetSplit(str):
    """String constants for the three canonical splits."""

    TRAIN = "train"
    VAL   = "val"
    TEST  = "test"


class DatasetAdapter(abc.ABC):
    """
    Abstract interface every dataset adapter must implement.

    Subclasses are responsible for:
    - Locating files and annotations under their own dataset root.
    - Converting dataset-specific label formats into integer class indices.
    - Returning Sample objects with valid, existing image_path values.

    The adapter must NOT perform heavy preprocessing (e.g. resizing, normalisation);
    that is delegated to the CNN data-loading pipeline.

    Example subclass skeleton::

        class MyDatasetAdapter(DatasetAdapter):
            def get_metadata(self) -> DatasetMetadata: ...
            def load_annotations(self) -> None: ...
            def get_class_names(self) -> list[str]: ...
            def get_training_samples(self) -> Iterator[Sample]: ...
            def get_validation_samples(self) -> Iterator[Sample]: ...
            def get_test_samples(self) -> Iterator[Sample]: ...
    """

    def __init__(self, config: "DatasetConfig") -> None:  # noqa: F821
        self.config = config

    # -------------------------------------------------------------------------
    # Abstract interface — every adapter must implement these
    # -------------------------------------------------------------------------

    @abc.abstractmethod
    def get_metadata(self) -> DatasetMetadata:
        """
        Return static metadata about this dataset (name, version, splits, ...).
        This must not require loading all samples into memory.
        """

    @abc.abstractmethod
    def load_annotations(self) -> None:
        """
        Parse the dataset's annotation files and build internal lookup
        structures.  Called once before iterating samples.
        Raises FileNotFoundError if the annotation files are missing.
        """

    @abc.abstractmethod
    def get_class_names(self) -> list[str]:
        """Return an ordered list of class name strings."""

    @abc.abstractmethod
    def get_training_samples(self) -> Iterator[Sample]:
        """Yield Sample objects from the training split."""

    @abc.abstractmethod
    def get_validation_samples(self) -> Iterator[Sample]:
        """Yield Sample objects from the validation split."""

    @abc.abstractmethod
    def get_test_samples(self) -> Iterator[Sample]:
        """Yield Sample objects from the test split."""

    # -------------------------------------------------------------------------
    # Concrete helpers — subclasses may override but need not
    # -------------------------------------------------------------------------

    def get_samples(self, split: str) -> Iterator[Sample]:
        """
        Dispatch to the correct split iterator by name.

        Args:
            split: One of DatasetSplit.TRAIN / VAL / TEST.

        Raises:
            ValueError: For an unknown split name.
        """
        if split == DatasetSplit.TRAIN:
            return self.get_training_samples()
        if split == DatasetSplit.VAL:
            return self.get_validation_samples()
        if split == DatasetSplit.TEST:
            return self.get_test_samples()
        raise ValueError(
            f"Unknown split '{split}'. "
            f"Expected one of: {DatasetSplit.TRAIN!r}, {DatasetSplit.VAL!r}, {DatasetSplit.TEST!r}"
        )

    def num_classes(self) -> int:
        """Convenience: return the number of classes."""
        return len(self.get_class_names())

    def __repr__(self) -> str:
        meta = self.get_metadata()
        return (
            f"{self.__class__.__name__}("
            f"dataset={meta.name!r}, "
            f"num_classes={meta.num_classes})"
        )

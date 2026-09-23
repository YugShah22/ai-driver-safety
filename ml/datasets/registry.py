"""
Dataset registry — maps adapter name strings to concrete DatasetAdapter classes.

When the project owner has decided on a dataset:
  1. Create a new module under ml/datasets/adapters/.
  2. Subclass DatasetAdapter.
  3. Call DatasetRegistry.register("mykey", MyAdapter) anywhere at import time.

The training script then loads the right adapter by reading the "adapter" key
from DatasetConfig — no changes needed to the CNN training code.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from .base import DatasetAdapter
    from .config import DatasetConfig


class DatasetRegistry:
    """
    Simple name -> class registry for DatasetAdapter subclasses.

    Usage::

        # In your adapter module
        DatasetRegistry.register("bdd100k", BDD100KAdapter)

        # In the training script
        adapter_cls = DatasetRegistry.get("bdd100k")
        adapter = adapter_cls(config)
    """

    _registry: dict[str, Type["DatasetAdapter"]] = {}

    @classmethod
    def register(cls, name: str, adapter_cls: Type["DatasetAdapter"]) -> None:
        """
        Register a DatasetAdapter subclass under ``name``.

        Args:
            name:        Unique lowercase key used in DatasetConfig.adapter.
            adapter_cls: Concrete subclass of DatasetAdapter.

        Raises:
            ValueError: If ``name`` is already registered (prevents silent overwrites).
        """
        if name in cls._registry:
            raise ValueError(
                f"DatasetAdapter key '{name}' is already registered "
                f"({cls._registry[name].__name__}). Use a different key."
            )
        cls._registry[name] = adapter_cls

    @classmethod
    def get(cls, name: str) -> Type["DatasetAdapter"]:
        """
        Retrieve a registered DatasetAdapter class by name.

        Raises:
            KeyError: If no adapter is registered under ``name``.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys())) or "<none>"
            raise KeyError(
                f"No DatasetAdapter registered for '{name}'. "
                f"Available adapters: {available}."
            )
        return cls._registry[name]

    @classmethod
    def build(cls, config: "DatasetConfig") -> "DatasetAdapter":
        """
        Instantiate the adapter named in ``config.adapter``.

        Args:
            config: DatasetConfig with a valid .adapter key.

        Returns:
            A concrete DatasetAdapter bound to the given config.
        """
        adapter_cls = cls.get(config.adapter)
        return adapter_cls(config)

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return sorted list of all registered adapter names."""
        return sorted(cls._registry.keys())

    @classmethod
    def _clear(cls) -> None:
        """Clear the registry — for use in tests only."""
        cls._registry.clear()

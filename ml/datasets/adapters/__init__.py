"""
Dataset adapters sub-package.

Importing from this package auto-registers all built-in adapters.
Dataset-specific adapters added later should also be imported here.
"""

from .generic import GenericFolderAdapter  # registers "generic"
from .indian_road import IndianRoadAdapter  # registers "indian_road"

__all__ = ["GenericFolderAdapter", "IndianRoadAdapter"]

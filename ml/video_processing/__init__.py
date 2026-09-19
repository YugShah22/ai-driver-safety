"""
Video Processing module for the AI Driver Safety Platform.

VideoProcessor: extracts frames, generates thumbnails, and uploads to Supabase Storage.
No AI/ML inference is performed here — this is pure video preprocessing.
"""

from .processor import VideoProcessor
from .models import (
    VideoMetadata,
    ExtractedFrame,
    ProcessingConfig,
    ResizeMode,
)

__all__ = [
    "VideoProcessor",
    "VideoMetadata",
    "ExtractedFrame",
    "ProcessingConfig",
    "ResizeMode",
]

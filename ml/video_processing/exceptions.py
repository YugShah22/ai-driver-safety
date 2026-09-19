"""
Custom exceptions for the video processing pipeline.
"""


class VideoProcessingError(RuntimeError):
    """Base class for all video-processing errors."""


class VideoNotFoundError(VideoProcessingError):
    """Raised when the video file does not exist on disk."""


class CorruptVideoError(VideoProcessingError):
    """Raised when OpenCV cannot open or read the video file."""


class UnsupportedFormatError(VideoProcessingError):
    """Raised when the video container or codec is not supported."""


class InvalidMetadataError(VideoProcessingError):
    """Raised when essential metadata (FPS, dimensions) cannot be determined."""


class StorageUploadError(VideoProcessingError):
    """Raised when uploading frames or thumbnails to Supabase Storage fails."""


class DatabaseError(VideoProcessingError):
    """Raised when inserting frame records into the database fails."""

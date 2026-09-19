"""
Data models for the video processing pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ResizeMode(str, Enum):
    """How to resize extracted frames."""

    ORIGINAL = "original"          # No resizing — keep native resolution.
    TARGET = "target"              # Exact width × height (may distort unless ratio matches).
    MAX_DIM = "max_dim"            # Scale so the larger dimension equals max_dimension,
                                   # preserving aspect ratio.


@dataclass
class VideoMetadata:
    """
    Metadata extracted from a dashcam video file.

    Attributes:
        filename:    Base name of the file.
        file_size:   File size in bytes.
        width:       Frame width in pixels.
        height:      Frame height in pixels.
        fps:         Actual frame rate (None if the file reports 0 or is missing).
        frame_count: Total number of frames (None if the container cannot report it).
        duration:    Video duration in seconds (None if indeterminate).
        codec:       Four-character codec string (e.g. "H264").
        container:   Container/format string (e.g. "mp4").
    """

    filename: str
    file_size: int
    width: int
    height: int
    fps: Optional[float]
    frame_count: Optional[int]
    duration: Optional[float]
    codec: Optional[str]
    container: Optional[str]


@dataclass
class ExtractedFrame:
    """
    A single frame extracted from a video, held in memory before saving.

    Attributes:
        frame_number: 1-based index of this frame among ALL extracted frames.
        timestamp:    Position in seconds from the start of the video.
        width:        Width in pixels (after any resizing).
        height:       Height in pixels (after any resizing).
        data:         Raw BGR image array (NumPy ndarray) from OpenCV.
    """

    frame_number: int
    timestamp: float
    width: int
    height: int
    data: object  # numpy.ndarray — avoid importing numpy at model definition time


@dataclass
class ProcessingConfig:
    """
    Configurable parameters for the VideoProcessor.

    Sampling:
        sample_interval:  Extract one frame every N seconds.  Takes precedence
                          over target_fps when both are provided.
        target_fps:       Extract frames at this rate (e.g. 2.0 → 2 fps).
                          Ignored if sample_interval is set.
        max_frames:       Stop after extracting this many frames (0 = unlimited).

    Resizing:
        resize_mode:      Which resize strategy to use (see ResizeMode enum).
        target_width:     Target width when resize_mode == TARGET.
        target_height:    Target height when resize_mode == TARGET.
        max_dimension:    Upper bound when resize_mode == MAX_DIM.

    Output:
        output_format:    "jpg" or "png".
        jpeg_quality:     JPEG quality (1–100).

    Thumbnails:
        thumbnail_width:  Thumbnail width in pixels.
        thumbnail_height: Thumbnail height in pixels.
    """

    # Sampling
    sample_interval: Optional[float] = None   # seconds between frames
    target_fps: float = 1.0                   # frames per second to extract
    max_frames: int = 0                        # 0 = no limit

    # Resizing
    resize_mode: ResizeMode = ResizeMode.ORIGINAL
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    max_dimension: Optional[int] = None

    # Output format
    output_format: str = "jpg"
    jpeg_quality: int = 85

    # Thumbnails
    thumbnail_width: int = 320
    thumbnail_height: int = 180

    def effective_sample_interval(self, video_fps: Optional[float]) -> float:
        """
        Return the number of *source* seconds between extracted frames.

        Uses sample_interval if set; otherwise derives it from target_fps.
        Falls back to extracting every second if no FPS info is available
        and no explicit interval was configured.
        """
        if self.sample_interval is not None:
            return max(self.sample_interval, 1e-6)  # guard against 0 / negative

        # Derive from target_fps
        if self.target_fps > 0:
            return 1.0 / self.target_fps

        # Last-resort fallback
        return 1.0

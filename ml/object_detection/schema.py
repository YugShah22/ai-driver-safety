from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates (x1, y1, x2, y2)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Detection:
    """
    One detected object in a single frame.

    Fields:
        category:    Human-readable class label (e.g. "car", "person").
        class_id:    Integer class index in the model's vocabulary.
        confidence:  Score in [0, 1].
        box:         Bounding box in pixel coordinates.
        frame_id:    Identifier for the source frame (path stem or timestamp).
        track_id:    Optional tracker ID — populated in Phase 8.
    """
    category: str
    class_id: int
    confidence: float
    box: BoundingBox
    frame_id: str = ""
    track_id: Optional[int] = None


@dataclass
class FrameDetections:
    """All detections from a single frame, plus metadata for downstream use."""
    frame_id: str
    detections: list[Detection] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0

    def __len__(self) -> int:
        return len(self.detections)

    def filter_by_confidence(self, threshold: float) -> "FrameDetections":
        """Return a new FrameDetections keeping only boxes above threshold."""
        filtered = [d for d in self.detections if d.confidence >= threshold]
        return FrameDetections(
            frame_id=self.frame_id,
            detections=filtered,
            image_width=self.image_width,
            image_height=self.image_height,
        )

    def filter_by_category(self, categories: set[str]) -> "FrameDetections":
        """Return a new FrameDetections keeping only the specified categories."""
        filtered = [d for d in self.detections if d.category in categories]
        return FrameDetections(
            frame_id=self.frame_id,
            detections=filtered,
            image_width=self.image_width,
            image_height=self.image_height,
        )

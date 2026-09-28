from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ml.object_detection.schema import BoundingBox


@dataclass
class Track:
    """
    A single tracked object across one or more consecutive frames.

    Fields:
        track_id:     Unique integer ID assigned when the track is created.
        category:     Class name from the detector (e.g. "car", "person").
        class_id:     Integer class index.
        box:          Bounding box in the most recent frame.
        confidence:   Detection confidence in the most recent frame.
        frame_index:  Frame number where this track was last updated.
        age:          Total number of frames this track has existed.
        hits:         Number of frames this track has been matched to a detection.
        misses:       Consecutive frames with no matching detection.
        history:      Ordered list of (frame_index, BoundingBox) for all matched frames.
        is_active:    False once the track is terminated.
        track_id:     Set on Detection.track_id whenever the tracker updates it.
    """
    track_id: int
    category: str
    class_id: int
    box: BoundingBox
    confidence: float
    frame_index: int
    age: int = 1
    hits: int = 1
    misses: int = 0
    history: list[tuple[int, BoundingBox]] = field(default_factory=list)
    is_active: bool = True

    def __post_init__(self) -> None:
        if not self.history:
            self.history = [(self.frame_index, self.box)]

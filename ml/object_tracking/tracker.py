from __future__ import annotations

import logging
from typing import Optional

from ml.object_detection.schema import BoundingBox, Detection, FrameDetections
from .schema import Track

logger = logging.getLogger(__name__)


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection over Union for two bounding boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h

    if intersection == 0.0:
        return 0.0

    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


class IOUTracker:
    """
    Frame-by-frame IoU-based multi-object tracker.

    Each frame's detections are matched to existing tracks by computing
    IoU between detection boxes and track boxes. The highest-IoU pair
    above the threshold is matched greedily. Unmatched detections start
    new tracks; tracks with no match for more than `max_misses` consecutive
    frames are terminated.

    This is intentionally simple — no Kalman filter, no ReID.
    It works well for dashcam footage with consistent frame rates and
    relatively stable object positions between frames.

    Args:
        iou_threshold:  Minimum IoU to consider a detection/track match.
        max_misses:     Consecutive frames without a match before a track ends.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_misses: int = 3,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses

        self._tracks: dict[int, Track] = {}
        self._next_id: int = 1

    # -------------------------------------------------------------------------

    @property
    def active_tracks(self) -> list[Track]:
        return [t for t in self._tracks.values() if t.is_active]

    @property
    def all_tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def reset(self) -> None:
        """Clear all tracks and reset the ID counter."""
        self._tracks.clear()
        self._next_id = 1

    # -------------------------------------------------------------------------

    def update(self, frame_detections: FrameDetections, frame_index: int) -> list[Track]:
        """
        Match detections to existing tracks, create new tracks, terminate lost ones.

        Side effect: sets Detection.track_id on every matched detection.

        Args:
            frame_detections: Output from ObjectDetector.detect() for one frame.
            frame_index:      Monotonically increasing frame counter.

        Returns:
            List of currently active tracks after processing this frame.
        """
        detections = frame_detections.detections
        active = self.active_tracks

        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()

        # Greedy IoU matching: for each active track find the best detection
        for track in active:
            best_iou = self.iou_threshold
            best_det_idx = -1

            for i, det in enumerate(detections):
                if i in matched_det_indices:
                    continue
                if det.class_id != track.class_id:
                    continue
                score = _iou(track.box, det.box)
                if score > best_iou:
                    best_iou = score
                    best_det_idx = i

            if best_det_idx >= 0:
                det = detections[best_det_idx]
                track.box = det.box
                track.confidence = det.confidence
                track.frame_index = frame_index
                track.age += 1
                track.hits += 1
                track.misses = 0
                track.history.append((frame_index, det.box))
                det.track_id = track.track_id

                matched_track_ids.add(track.track_id)
                matched_det_indices.add(best_det_idx)

        # Increment misses for unmatched tracks; terminate if over limit
        for track in active:
            if track.track_id not in matched_track_ids:
                track.misses += 1
                track.age += 1
                if track.misses > self.max_misses:
                    track.is_active = False
                    logger.debug("Track %d terminated after %d misses", track.track_id, track.misses)

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i in matched_det_indices:
                continue
            new_track = Track(
                track_id=self._next_id,
                category=det.category,
                class_id=det.class_id,
                box=det.box,
                confidence=det.confidence,
                frame_index=frame_index,
            )
            det.track_id = self._next_id
            self._tracks[self._next_id] = new_track
            self._next_id += 1
            logger.debug("Track %d created at frame %d", new_track.track_id, frame_index)

        return self.active_tracks

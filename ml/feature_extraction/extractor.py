from __future__ import annotations

import logging
from typing import Optional

from ml.object_detection.schema import FrameDetections
from ml.object_tracking.schema import Track
from ml.segmentation.schema import LaneResult

from .schema import FrameFeatures, ObjectFeature, LaneFeature

logger = logging.getLogger(__name__)


def _object_feature(
    det_or_track,
    image_width: int,
    image_height: int,
) -> ObjectFeature:
    """Build an ObjectFeature from either a Detection or a Track."""
    box = det_or_track.box
    image_area = image_width * image_height

    cx = (box.x1 + box.x2) / 2.0
    cy = (box.y1 + box.y2) / 2.0
    box_area = max(0.0, box.width) * max(0.0, box.height)
    area_frac = box_area / image_area if image_area > 0 else 0.0

    track_id = getattr(det_or_track, "track_id", None)
    track_age  = getattr(det_or_track, "age",  None)
    track_hits = getattr(det_or_track, "hits", None)

    return ObjectFeature(
        track_id=track_id,
        category=det_or_track.category,
        confidence=det_or_track.confidence,
        box_x1=box.x1,
        box_y1=box.y1,
        box_x2=box.x2,
        box_y2=box.y2,
        box_center_x_px=cx,
        box_center_y_px=cy,
        box_area_fraction=area_frac,
        is_in_lower_half=(cy > image_height / 2.0),
        track_age=track_age,
        track_hits=track_hits,
    )


def _lane_feature(lane_result: Optional[LaneResult]) -> Optional[LaneFeature]:
    if lane_result is None:
        return None
    return LaneFeature(
        is_valid=lane_result.is_valid,
        status=lane_result.status,
        lane_center_x_px=lane_result.lane_center_x_px,
        image_center_x_px=lane_result.image_center_x_px,
        left_boundary_x_px=lane_result.left_boundary_x_px,
        right_boundary_x_px=lane_result.right_boundary_x_px,
        lateral_offset_px=lane_result.lateral_offset_px,
        lane_departure_indicator=lane_result.lane_departure_indicator,
        road_pixel_fraction=lane_result.road_pixel_fraction,
    )


def _count_by_category(objects: list[ObjectFeature]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obj in objects:
        counts[obj.category] = counts.get(obj.category, 0) + 1
    return counts


class FeatureExtractor:
    """
    Aggregates outputs from Phases 5–9 into a single FrameFeatures object.

    The extractor is stateful: it remembers the active track IDs from the
    previous frame so it can populate new_track_ids and lost_track_ids.

    Usage::

        extractor = FeatureExtractor()

        for frame_index, frame in enumerate(frames):
            scene_class    = cnn_model.predict(frame)
            frame_dets     = detector.detect(frame, frame_id=...)
            active_tracks  = tracker.update(frame_dets, frame_index)
            lane_result    = lane_analyzer.analyze(seg_model.segment(frame))

            features = extractor.extract(
                frame_id=frame_id,
                frame_index=frame_index,
                timestamp_s=frame.timestamp,
                image_width=frame.width,
                image_height=frame.height,
                scene_class=scene_class,
                scene_class_id=scene_class_id,
                active_tracks=active_tracks,
                frame_detections=frame_dets,   # used when no tracker
                lane_result=lane_result,
            )
    """

    def __init__(self) -> None:
        self._prev_track_ids: set[int] = set()

    def reset(self) -> None:
        """Clear frame-to-frame state (e.g. at clip boundaries)."""
        self._prev_track_ids.clear()

    def extract(
        self,
        *,
        frame_id: str,
        frame_index: int,
        image_width: int,
        image_height: int,
        timestamp_s: Optional[float] = None,
        scene_class: Optional[str] = None,
        scene_class_id: Optional[int] = None,
        active_tracks: Optional[list[Track]] = None,
        frame_detections: Optional[FrameDetections] = None,
        lane_result: Optional[LaneResult] = None,
    ) -> FrameFeatures:
        """
        Build a FrameFeatures from upstream model outputs.

        Prefer active_tracks (from Phase 8) when available; fall back to
        frame_detections (from Phase 7) when no tracker is running.
        """
        # Build object features
        objects: list[ObjectFeature] = []

        if active_tracks:
            for track in active_tracks:
                objects.append(_object_feature(track, image_width, image_height))
        elif frame_detections:
            for det in frame_detections.detections:
                objects.append(_object_feature(det, image_width, image_height))

        # Temporal: new / lost track IDs
        current_ids: set[int] = {
            obj.track_id for obj in objects if obj.track_id is not None
        }
        new_ids  = sorted(current_ids - self._prev_track_ids)
        lost_ids = sorted(self._prev_track_ids - current_ids)
        self._prev_track_ids = current_ids

        return FrameFeatures(
            frame_id=frame_id,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            image_width=image_width,
            image_height=image_height,
            scene_class=scene_class,
            scene_class_id=scene_class_id,
            objects=objects,
            object_counts=_count_by_category(objects),
            lane=_lane_feature(lane_result),
            new_track_ids=new_ids,
            lost_track_ids=lost_ids,
        )

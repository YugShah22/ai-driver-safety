from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ObjectFeature:
    """
    Features for a single detected/tracked object in a frame.

    All spatial values are in pixel coordinates.
    box_area_fraction is the fraction of the total image area covered by the box.
    """
    track_id: Optional[int]
    category: str
    confidence: float
    box_x1: float
    box_y1: float
    box_x2: float
    box_y2: float
    box_center_x_px: float
    box_center_y_px: float
    box_area_fraction: float   # box area / image area — a size proxy, not a distance
    is_in_lower_half: bool     # True if box centre y > image height/2
    track_age: Optional[int]   # frames this track has existed; None if untracked
    track_hits: Optional[int]  # frames matched to a detection


@dataclass
class LaneFeature:
    """
    Lane/road geometry derived from Phase 9 segmentation.

    All values are pixel-space proxies — not physical measurements.
    is_valid=False means geometry could not be estimated.
    """
    is_valid: bool
    status: str
    lane_center_x_px: Optional[float] = None
    image_center_x_px: Optional[float] = None
    left_boundary_x_px: Optional[float] = None
    right_boundary_x_px: Optional[float] = None
    lateral_offset_px: Optional[float] = None
    lane_departure_indicator: Optional[bool] = None
    road_pixel_fraction: Optional[float] = None


@dataclass
class FrameFeatures:
    """
    Consolidated driving features for a single video frame.

    This is the output of Phase 10 and the input to Phase 11 (risk scoring).

    Fields marked with "proxy" or "_px" are pixel-space estimates.
    No physical quantities (speed, distance in metres, TTC, g-force) are
    included unless they can actually be derived from available data.
    """
    # --- Frame identity ---
    frame_id: str
    frame_index: int
    timestamp_s: Optional[float]        # seconds from video start; None if unavailable
    image_width: int
    image_height: int

    # --- Scene (from CNN scene classifier) ---
    scene_class: Optional[str]          # e.g. "highway", "residential road"
    scene_class_id: Optional[int]

    # --- Objects (from YOLO + tracker) ---
    objects: list[ObjectFeature] = field(default_factory=list)

    # --- Aggregate object counts per category ---
    object_counts: dict[str, int] = field(default_factory=dict)

    # --- Lane (from segmentation / lane analyser) ---
    lane: Optional[LaneFeature] = None

    # --- Derived temporal features (require previous frame) ---
    # These are only populated by FeatureExtractor when a previous frame exists.
    new_track_ids: list[int] = field(default_factory=list)      # tracks that appeared this frame
    lost_track_ids: list[int] = field(default_factory=list)     # tracks that disappeared

    # --- Convenience properties ---

    @property
    def total_object_count(self) -> int:
        return len(self.objects)

    @property
    def has_lane_info(self) -> bool:
        return self.lane is not None and self.lane.is_valid

    @property
    def lane_departure_flagged(self) -> bool:
        return (
            self.has_lane_info
            and self.lane.lane_departure_indicator is True
        )

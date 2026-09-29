from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SegmentationResult:
    """
    Output of a segmentation model for a single frame.

    Fields:
        frame_id:     Identifier from the source (path stem, timestamp, etc.).
        mask:         Integer class-index mask of shape (H, W), dtype int64.
                      Each pixel value is the argmax class index.
        width:        Image width in pixels.
        height:       Image height in pixels.
        num_classes:  Number of output classes the model was configured with.
        class_names:  Optional mapping from class index to human-readable name.
                      If None, the semantic meaning of pixel values is UNKNOWN.
        logits:       Optional raw model output of shape (num_classes, H, W).
                      Retained when the caller needs confidence values.

    IMPORTANT: `class_names` is None by default. A None class_names means the
    semantic mapping has NOT been verified. Do not assume any pixel value
    corresponds to a specific road feature without a verified mapping.
    """
    frame_id: str
    mask: np.ndarray           # shape (H, W), dtype int64
    width: int
    height: int
    num_classes: int
    class_names: Optional[dict[int, str]] = None
    logits: Optional[np.ndarray] = None  # shape (num_classes, H, W)

    def pixels_for_class(self, class_id: int) -> np.ndarray:
        """Return a boolean mask selecting pixels of the given class."""
        return self.mask == class_id

    def class_pixel_counts(self) -> dict[int, int]:
        """Return pixel counts per class index."""
        unique, counts = np.unique(self.mask, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    @property
    def semantic_mapping_verified(self) -> bool:
        """True only when class_names has been explicitly provided."""
        return self.class_names is not None


@dataclass
class LaneResult:
    """
    Lane/road geometry estimated from a segmentation mask for a single frame.

    All spatial values are in pixel coordinates — they are NOT physically
    calibrated measurements. A pixel-space lateral offset is not equivalent
    to metres from lane centre, a steering angle, or a lateral acceleration.

    Fields:
        frame_id:               Source frame identifier.
        is_valid:               False when geometry could not be estimated
                                (e.g., no road pixels found, unknown mapping).
        status:                 Human-readable reason for validity state.
        lane_center_x_px:       Estimated centre of the road/lane region, x-axis.
        image_center_x_px:      Geometric centre of the image, x-axis.
        left_boundary_x_px:     Leftmost road pixel x in the analysis row band.
        right_boundary_x_px:    Rightmost road pixel x in the analysis row band.
        lateral_offset_px:      image_center_x_px - lane_center_x_px.
                                Positive → image centre is to the right of lane centre.
        lane_departure_indicator: True when |lateral_offset_px| exceeds the
                                  configured threshold. This is a geometry heuristic
                                  only — not a definitive safety assertion.
        road_pixel_fraction:    Fraction of pixels in the analysis band classified
                                as road/lane. Indicator of mask coverage quality.
    """
    frame_id: str
    is_valid: bool
    status: str
    lane_center_x_px: Optional[float] = None
    image_center_x_px: Optional[float] = None
    left_boundary_x_px: Optional[float] = None
    right_boundary_x_px: Optional[float] = None
    lateral_offset_px: Optional[float] = None
    lane_departure_indicator: Optional[bool] = None
    road_pixel_fraction: Optional[float] = None

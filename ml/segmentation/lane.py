from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .schema import SegmentationResult, LaneResult

logger = logging.getLogger(__name__)


class LaneAnalyzer:
    """
    Derives lane/road geometry from a segmentation mask.

    This is a visual-geometry heuristic — not a calibrated measurement system.
    All outputs are in pixel space.

    The analyzer requires the caller to specify which class index (or set of
    indices) represents the road/lane region. This is intentional: the
    Indian Road dataset mask semantics have NOT been verified, and silently
    assuming a class mapping would produce meaningless results.

    If no road_class_ids are provided, every LaneResult will have
    is_valid=False with a clear status message.

    Args:
        road_class_ids:       Set of class index values that represent the
                              road or lane region. Must be verified externally
                              before use.
        analysis_band_frac:   Fraction of the image height, from the bottom,
                              to analyse for lane boundaries (default 0.3 = 30%).
        departure_threshold_px: Minimum |lateral_offset_px| to set
                                lane_departure_indicator=True.
    """

    def __init__(
        self,
        road_class_ids: Optional[set[int]] = None,
        analysis_band_frac: float = 0.3,
        departure_threshold_px: float = 50.0,
    ) -> None:
        self.road_class_ids = road_class_ids
        self.analysis_band_frac = analysis_band_frac
        self.departure_threshold_px = departure_threshold_px

    def analyze(self, result: SegmentationResult) -> LaneResult:
        """
        Estimate lane geometry from a SegmentationResult.

        Returns LaneResult with is_valid=False and an explanatory status when:
        - no road_class_ids have been configured
        - the configured class IDs produce no pixels in the analysis band
        - the mask is empty or malformed
        """
        frame_id = result.frame_id

        if not self.road_class_ids:
            return LaneResult(
                frame_id=frame_id,
                is_valid=False,
                status="No road_class_ids configured. Semantic mapping must be "
                       "verified before lane analysis is possible.",
            )

        mask = result.mask
        h, w = mask.shape

        if h == 0 or w == 0:
            return LaneResult(
                frame_id=frame_id,
                is_valid=False,
                status="Empty or zero-dimension mask.",
            )

        # Select the bottom analysis band
        band_rows = max(1, int(h * self.analysis_band_frac))
        band = mask[h - band_rows:, :]   # shape (band_rows, w)

        # Build road-pixel boolean mask from all configured class IDs
        road_mask = np.zeros(band.shape, dtype=bool)
        for cid in self.road_class_ids:
            road_mask |= (band == cid)

        road_pixel_count = int(road_mask.sum())
        total_band_pixels = band_rows * w
        road_pixel_fraction = road_pixel_count / total_band_pixels if total_band_pixels > 0 else 0.0

        if road_pixel_count == 0:
            return LaneResult(
                frame_id=frame_id,
                is_valid=False,
                status=f"No road pixels found in bottom {self.analysis_band_frac:.0%} "
                       f"of frame for class IDs {self.road_class_ids}.",
                road_pixel_fraction=0.0,
            )

        # Find column extents of road pixels across all rows in the band
        road_cols = np.where(road_mask.any(axis=0))[0]
        left_boundary_x  = float(road_cols.min())
        right_boundary_x = float(road_cols.max())
        lane_center_x    = (left_boundary_x + right_boundary_x) / 2.0
        image_center_x   = (w - 1) / 2.0
        lateral_offset   = image_center_x - lane_center_x

        departure = abs(lateral_offset) > self.departure_threshold_px

        return LaneResult(
            frame_id=frame_id,
            is_valid=True,
            status="ok",
            lane_center_x_px=lane_center_x,
            image_center_x_px=image_center_x,
            left_boundary_x_px=left_boundary_x,
            right_boundary_x_px=right_boundary_x,
            lateral_offset_px=lateral_offset,
            lane_departure_indicator=departure,
            road_pixel_fraction=road_pixel_fraction,
        )

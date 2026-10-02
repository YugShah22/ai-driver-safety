from __future__ import annotations

import numpy as np

from ml.feature_extraction.schema import FrameFeatures
from .schema import FEATURE_NAMES, N_FEATURES, LabeledSample


def vectorize(frame: FrameFeatures) -> np.ndarray:
    """
    Convert a FrameFeatures into a flat float32 vector of shape (N_FEATURES,).

    Missing values are encoded as 0.0 (for counts/flags) or -1.0 (for IDs).
    No imputation is applied — callers that need sklearn pipelines should
    wrap the matrix in a StandardScaler or similar after vectorization.
    """
    counts = frame.object_counts
    objects = frame.objects

    max_area = max((o.box_area_fraction for o in objects), default=0.0)
    n_lower  = sum(1 for o in objects if o.is_in_lower_half)

    lane = frame.lane
    has_lane      = 1.0 if (lane is not None and lane.is_valid) else 0.0
    offset_px     = float(lane.lateral_offset_px or 0.0) if has_lane else 0.0
    road_frac     = float(lane.road_pixel_fraction or 0.0) if has_lane else 0.0
    departure     = 1.0 if (has_lane and lane.lane_departure_indicator) else 0.0

    scene_id = float(frame.scene_class_id) if frame.scene_class_id is not None else -1.0

    vec = np.array([
        float(len(objects)),
        float(counts.get("person", 0)),
        float(counts.get("car", 0)),
        float(counts.get("truck", 0)),
        float(counts.get("bus", 0)),
        float(counts.get("motorcycle", 0)),
        float(counts.get("bicycle", 0)),
        float(counts.get("traffic light", 0)),
        float(counts.get("stop sign", 0)),
        max_area,
        float(n_lower),
        float(len(frame.new_track_ids)),
        float(len(frame.lost_track_ids)),
        has_lane,
        offset_px,
        road_frac,
        departure,
        scene_id,
    ], dtype=np.float32)

    assert vec.shape == (N_FEATURES,), f"Expected {N_FEATURES} features, got {vec.shape[0]}"
    return vec


def build_matrix(
    frames: list[FrameFeatures],
    labels: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a feature matrix X and label vector y from a list of FrameFeatures.

    Args:
        frames: List of FrameFeatures (one per sample).
        labels: Integer labels aligned with frames.

    Returns:
        X: float32 array of shape (n_samples, N_FEATURES).
        y: int64 array of shape (n_samples,).
    """
    if len(frames) != len(labels):
        raise ValueError(
            f"frames ({len(frames)}) and labels ({len(labels)}) must have the same length."
        )
    X = np.stack([vectorize(f) for f in frames], axis=0)
    y = np.array(labels, dtype=np.int64)
    return X, y


def make_labeled_samples(
    frames: list[FrameFeatures],
    labels: list[int],
    label_names: dict[int, str] | None = None,
) -> list[LabeledSample]:
    """Convenience wrapper — returns LabeledSample objects for traceability."""
    if len(frames) != len(labels):
        raise ValueError("frames and labels must have the same length.")
    return [
        LabeledSample(
            label=lbl,
            feature_vec=vectorize(f),
            frame_id=f.frame_id,
            label_name=label_names.get(lbl) if label_names else None,
        )
        for f, lbl in zip(frames, labels)
    ]

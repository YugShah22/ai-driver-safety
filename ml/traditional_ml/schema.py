from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# Ordered list of feature names produced by FeatureVectorizer.
# Changing this list changes the serialised model's expected input — treat as an API.
FEATURE_NAMES: list[str] = [
    # object counts
    "n_objects",
    "n_persons",
    "n_cars",
    "n_trucks",
    "n_buses",
    "n_motorcycles",
    "n_bicycles",
    "n_traffic_lights",
    "n_stop_signs",
    # object spatial proxies
    "max_box_area_fraction",   # largest object's area fraction — coarse closeness proxy
    "n_objects_lower_half",    # objects in lower screen half (typically closer)
    # temporal
    "n_new_tracks",
    "n_lost_tracks",
    # lane
    "has_lane_info",           # 1.0 / 0.0
    "lateral_offset_px",       # 0.0 when unavailable
    "road_pixel_fraction",     # 0.0 when unavailable
    "lane_departure",          # 1.0 / 0.0
    # scene
    "scene_class_id",          # -1.0 when unknown
]

N_FEATURES = len(FEATURE_NAMES)


@dataclass
class LabeledSample:
    """
    A single training/evaluation sample.

    label:         Integer class label assigned externally (e.g. human annotation).
                   This is NOT derived from model pseudo-predictions.
    label_name:    Optional human-readable name for the label.
    feature_vec:   Flat float32 numpy array of shape (N_FEATURES,).
    frame_id:      Source frame identifier for traceability.
    """
    label: int
    feature_vec: np.ndarray
    frame_id: str = ""
    label_name: Optional[str] = None


@dataclass
class ModelEvaluation:
    """
    Evaluation results for a trained RiskModel.

    These metrics are only meaningful relative to the label set used.
    They do NOT represent real-world driving-risk accuracy unless the
    labels are derived from validated ground-truth annotation.
    """
    accuracy: float
    f1_weighted: float
    confusion_matrix: np.ndarray       # shape (n_classes, n_classes)
    class_names: Optional[list[str]]
    n_samples: int
    label_note: str = (
        "Metrics computed on the provided label set. "
        "Real-world validity depends on label quality."
    )

"""
System Capabilities and Operational Limitations for Driving Risk Assessment.

Explicitly defines what the AI Driver Safety platform CAN and CANNOT infer
from forward-facing camera vision outputs (YOLO detections, SORT tracking,
UNet segmentation, CNN scene classification).

CRITICAL ARCHITECTURAL DIRECTIVE:
Do NOT claim or imply physical quantities or driver internal states that
cannot be rigorously established from monocular video data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final


CAN_INFER: Final[list[str]] = [
    "2D object bounding boxes, object categories (cars, pedestrians, bicycles, trucks, etc.), and detection confidence.",
    "2D visual scale proxies: bounding box area as a fraction of image area (large area indicates visual proximity / prominence).",
    "2D spatial image positions: vertical position (lower half vs. upper half) and horizontal corridor centering.",
    "Object tracking persistence across consecutive frames: track identity, track age (frames survived), and appearance/loss events.",
    "Road lane geometry proxies: lateral pixel offset from image center to detected lane center, and lane boundary crossing indicators.",
    "Environmental scene categories: highway, residential, parking, adverse weather (fog, rain, snow, night) from CNN classifier.",
    "Traffic density proxies: aggregate object counts and distribution of objects across image regions.",
]

CANNOT_INFER: Final[list[str]] = [
    "Exact physical metric distance in meters (requires calibrated stereo camera, LiDAR, or depth sensor).",
    "Exact vehicle speed, acceleration, deceleration, or braking g-forces (requires CAN bus, IMU, or GPS telemetry).",
    "Exact physical time-to-collision (TTC) in seconds (requires calibrated 3D range and relative velocity vectors).",
    "Exact physical collision probability percentage (requires calibrated physical vehicle dynamics and friction modeling).",
    "Driver attention, gaze direction, head pose, distraction, or drowsiness (requires an interior cabin-facing camera).",
    "Driver reaction time or physical cognitive latency (requires driver behavioral telemetry).",
    "Driver medical, psychological, emotional, or physiological state (no biometric sensors available).",
    "Road surface metric friction coefficient / mu (requires dedicated chassis sensors or wheel slip telemetry).",
]

SCORE_MEANING_DOCUMENTATION: Final[str] = (
    "The composite visual risk score is a deterministic heuristic index on a 0–100 scale "
    "representing the cumulative severity of detected visual hazard indicators "
    "(LOW: 10, MEDIUM: 25, HIGH: 50, CRITICAL: 80, capped at 100). "
    "It reflects observable visual hazard density in the 2D forward camera frame. "
    "It does NOT represent a physical collision probability, distance in meters, "
    "or driver attentiveness."
)

DISCLAIMER_NOTICE: Final[str] = (
    "DISCLAIMER: This driving assessment is derived exclusively from 2D optical proxies "
    "captured by a monocular forward-facing camera. The system does not measure physical "
    "distance, vehicle speed, driver attention, or collision probability. It provides "
    "situational visual indicators for driver assistance and safety analytics."
)


@dataclass(frozen=True)
class SystemCapabilities:
    """Documented capabilities and boundaries of the risk engine."""
    can_infer: list[str] = field(default_factory=lambda: list(CAN_INFER))
    cannot_infer: list[str] = field(default_factory=lambda: list(CANNOT_INFER))
    score_meaning: str = SCORE_MEANING_DOCUMENTATION
    disclaimer: str = DISCLAIMER_NOTICE

    @classmethod
    def get_summary(cls) -> dict[str, object]:
        return {
            "can_infer": list(CAN_INFER),
            "cannot_infer": list(CANNOT_INFER),
            "score_meaning": SCORE_MEANING_DOCUMENTATION,
            "disclaimer": DISCLAIMER_NOTICE,
        }

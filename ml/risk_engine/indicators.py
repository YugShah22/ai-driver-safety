"""
Documented Risk Indicator Evaluators.

Every indicator implemented here has an explicit, documented empirical basis
grounded in 2D camera geometry and upstream perception outputs.
No arbitrary or magical rules are used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .schema import IndicatorId, RiskIndicator, Severity


@dataclass
class RiskIndicatorConfig:
    """Configurable thresholds for deterministic indicator evaluation."""
    # Nearby vehicle optical area proxies
    nearby_vehicle_area_threshold: float = 0.12
    nearby_vehicle_high_area: float = 0.25

    # Vulnerable road user (VRU) in path
    vru_area_threshold: float = 0.015
    vru_critical_area: float = 0.06
    vru_corridor_x_min: float = 0.25
    vru_corridor_x_max: float = 0.75

    # Lane departure
    lane_departure_offset_px: float = 60.0
    lane_departure_critical_offset_px: float = 120.0

    # Persistent obstruction
    persistent_obstruction_min_age: int = 15
    persistent_obstruction_min_area: float = 0.04

    # Sudden cut-in
    cut_in_min_area: float = 0.08

    # Traffic density
    high_density_object_count: int = 8
    high_density_lower_count: int = 5

    # Adverse environment
    adverse_scene_classes: set[str] = field(
        default_factory=lambda: {
            "rain", "rainy", "fog", "foggy", "snow", "snowy",
            "night", "storm", "adverse_weather", "heavy_rain",
        }
    )

    # Recognised object categories
    vehicle_categories: set[str] = field(
        default_factory=lambda: {"car", "truck", "bus", "motorcycle", "vehicle"}
    )
    vru_categories: set[str] = field(
        default_factory=lambda: {"person", "bicycle", "pedestrian", "cyclist"}
    )


def evaluate_nearby_vehicles(
    objects: list[Any],
    config: RiskIndicatorConfig,
) -> list[RiskIndicator]:
    """
    Evaluates visual closeness proxy for vehicles in the lower half of the frame.

    Basis: In monocular forward dashcam geometry, a large bounding box area in the lower
    half of the image indicates visual closeness to the ego-vehicle.
    """
    indicators: list[RiskIndicator] = []

    for obj in objects:
        cat = getattr(obj, "category", "") or ""
        cat_lower = str(cat).lower()
        if cat_lower not in config.vehicle_categories:
            continue

        is_lower = getattr(obj, "is_in_lower_half", False)
        area_frac = getattr(obj, "box_area_fraction", 0.0) or 0.0

        if is_lower and area_frac >= config.nearby_vehicle_area_threshold:
            severity = (
                Severity.HIGH
                if area_frac >= config.nearby_vehicle_high_area
                else Severity.MEDIUM
            )
            track_id = getattr(obj, "track_id", None)
            conf = getattr(obj, "confidence", 0.0)

            basis = (
                f"2D optical proxy: Vehicle '{cat}' (track_id={track_id}) occupies "
                f"{area_frac:.1%} of frame area in the lower half (>= {config.nearby_vehicle_area_threshold:.1%} threshold). "
                f"In forward camera geometry, large bounding box area indicates close visual proximity. "
                f"NOTE: This is an optical scale proxy, not a calibrated metric distance in meters."
            )

            indicators.append(
                RiskIndicator(
                    indicator_id=IndicatorId.NEARBY_VEHICLE.value,
                    name="Nearby Vehicle in Forward View",
                    severity=severity,
                    triggered=True,
                    basis=basis,
                    source_data={
                        "category": cat,
                        "track_id": track_id,
                        "box_area_fraction": round(float(area_frac), 4),
                        "confidence": round(float(conf), 3) if conf else None,
                        "is_in_lower_half": True,
                    },
                )
            )

    return indicators


def evaluate_vulnerable_road_users(
    objects: list[Any],
    image_width: int,
    config: RiskIndicatorConfig,
) -> list[RiskIndicator]:
    """
    Evaluates vulnerable road users (pedestrians, cyclists) in the projected travel corridor.

    Basis: Vulnerable road users lack external impact protection. Detection in the central
    corridor of the lower frame indicates an acute trajectory hazard.
    """
    indicators: list[RiskIndicator] = []
    width = max(1, image_width)

    for obj in objects:
        cat = getattr(obj, "category", "") or ""
        cat_lower = str(cat).lower()
        if cat_lower not in config.vru_categories:
            continue

        is_lower = getattr(obj, "is_in_lower_half", False)
        area_frac = getattr(obj, "box_area_fraction", 0.0) or 0.0
        cx = getattr(obj, "box_center_x_px", width / 2.0)
        x_ratio = cx / width

        in_corridor = config.vru_corridor_x_min <= x_ratio <= config.vru_corridor_x_max

        if is_lower and in_corridor and area_frac >= config.vru_area_threshold:
            severity = (
                Severity.CRITICAL
                if area_frac >= config.vru_critical_area
                else Severity.HIGH
            )
            track_id = getattr(obj, "track_id", None)
            conf = getattr(obj, "confidence", 0.0)

            basis = (
                f"Vulnerable road user ('{cat}', track_id={track_id}) detected in central vehicle travel "
                f"corridor (x_pos={cx:.1f}px / {width}px, lower frame). Area fraction is {area_frac:.1%} "
                f"(>= {config.vru_area_threshold:.1%} threshold). Pedestrians and cyclists lack crash protection, "
                f"making forward path presence an acute safety concern."
            )

            indicators.append(
                RiskIndicator(
                    indicator_id=IndicatorId.VULNERABLE_ROAD_USER_PATH.value,
                    name="Vulnerable Road User in Path",
                    severity=severity,
                    triggered=True,
                    basis=basis,
                    source_data={
                        "category": cat,
                        "track_id": track_id,
                        "box_area_fraction": round(float(area_frac), 4),
                        "box_center_x_px": round(float(cx), 1),
                        "corridor_ratio": round(float(x_ratio), 3),
                        "confidence": round(float(conf), 3) if conf else None,
                    },
                )
            )

    return indicators


def evaluate_lane_departure(
    lane: Optional[Any],
    config: RiskIndicatorConfig,
) -> Optional[RiskIndicator]:
    """
    Evaluates lane departure and significant lateral offset from lane centerline.

    Basis: Road segmentation lateral offset and lane boundary flags indicate trajectory departure.
    """
    if lane is None:
        return None

    is_valid = getattr(lane, "is_valid", False)
    if not is_valid:
        return None

    departure_flag = getattr(lane, "lane_departure_indicator", False) or False
    offset_px = getattr(lane, "lateral_offset_px", 0.0) or 0.0
    abs_offset = abs(float(offset_px))

    is_triggered = departure_flag or (abs_offset >= config.lane_departure_offset_px)

    if not is_triggered:
        return None

    severity = (
        Severity.HIGH
        if abs_offset >= config.lane_departure_critical_offset_px
        else Severity.MEDIUM
    )

    basis = (
        f"Lane segmentation indicates vehicle departure from driving corridor: "
        f"departure_flag={departure_flag}, lateral_offset={offset_px:.1f}px "
        f"(threshold={config.lane_departure_offset_px:.1f}px). Ego-vehicle visual centerline "
        f"deviates from detected road lane markers."
    )

    return RiskIndicator(
        indicator_id=IndicatorId.LANE_DEPARTURE.value,
        name="Lane Departure Detected",
        severity=severity,
        triggered=True,
        basis=basis,
        source_data={
            "lane_departure_indicator": departure_flag,
            "lateral_offset_px": round(float(offset_px), 1),
            "status": getattr(lane, "status", "unknown"),
        },
    )


def evaluate_persistent_obstructions(
    objects: list[Any],
    image_width: int,
    config: RiskIndicatorConfig,
) -> list[RiskIndicator]:
    """
    Evaluates persistent tracked objects blocking the forward corridor.

    Basis: Objects tracked across >= 15 consecutive frames in forward path without clearing
    indicate a stopped vehicle or stationary obstacle.
    """
    indicators: list[RiskIndicator] = []
    width = max(1, image_width)

    for obj in objects:
        age = getattr(obj, "track_age", None)
        if age is None or age < config.persistent_obstruction_min_age:
            continue

        is_lower = getattr(obj, "is_in_lower_half", False)
        area_frac = getattr(obj, "box_area_fraction", 0.0) or 0.0
        cx = getattr(obj, "box_center_x_px", width / 2.0)
        x_ratio = cx / width

        in_corridor = config.vru_corridor_x_min <= x_ratio <= config.vru_corridor_x_max

        if is_lower and in_corridor and area_frac >= config.persistent_obstruction_min_area:
            track_id = getattr(obj, "track_id", None)
            cat = getattr(obj, "category", "object")

            basis = (
                f"Persistent obstruction: Tracked object '{cat}' (track_id={track_id}) "
                f"has persisted continuously in the forward path for {age} frames "
                f"(>= {config.persistent_obstruction_min_age} threshold) with area fraction {area_frac:.1%}. "
                f"Indicates a stationary vehicle or enduring roadway obstacle."
            )

            indicators.append(
                RiskIndicator(
                    indicator_id=IndicatorId.PERSISTENT_OBSTRUCTION.value,
                    name="Persistent Obstruction in Travel Corridor",
                    severity=Severity.MEDIUM,
                    triggered=True,
                    basis=basis,
                    source_data={
                        "category": cat,
                        "track_id": track_id,
                        "track_age": age,
                        "box_area_fraction": round(float(area_frac), 4),
                        "corridor_ratio": round(float(x_ratio), 3),
                    },
                )
            )

    return indicators


def evaluate_adverse_environment(
    scene_class: Optional[str],
    lane: Optional[Any],
    config: RiskIndicatorConfig,
) -> Optional[RiskIndicator]:
    """
    Evaluates degraded environmental conditions from scene classification.

    Basis: Rain, fog, snow, or night reduce camera contrast and tire traction.
    """
    if scene_class is None:
        return None

    scene_norm = str(scene_class).strip().lower()
    is_adverse = any(term in scene_norm for term in config.adverse_scene_classes)

    if not is_adverse:
        return None

    # Storm/fog/snow gets MEDIUM; rain/night gets LOW
    severe_terms = {"fog", "snow", "storm", "heavy_rain"}
    is_severe = any(term in scene_norm for term in severe_terms)
    severity = Severity.MEDIUM if is_severe else Severity.LOW

    basis = (
        f"Environmental scene classification identified adverse driving condition '{scene_class}'. "
        f"Adverse weather or poor illumination reduces visual visibility, increases stopping distances, "
        f"and degrades perceptual clarity."
    )

    return RiskIndicator(
        indicator_id=IndicatorId.ADVERSE_ENVIRONMENT.value,
        name="Adverse Environmental Conditions",
        severity=severity,
        triggered=True,
        basis=basis,
        source_data={
            "scene_class": scene_class,
            "severity_basis": "severe_weather" if is_severe else "moderate_adverse_condition",
        },
    )


def evaluate_sudden_cut_in(
    objects: list[Any],
    new_track_ids: list[int],
    image_width: int,
    config: RiskIndicatorConfig,
) -> list[RiskIndicator]:
    """
    Evaluates newly appeared tracks with substantial area in the forward corridor.

    Basis: A new track suddenly appearing in the lower central frame is consistent
    with a cut-in vehicle or rapid intrusion.
    """
    if not new_track_ids:
        return []

    indicators: list[RiskIndicator] = []
    width = max(1, image_width)
    new_set = set(new_track_ids)

    for obj in objects:
        tid = getattr(obj, "track_id", None)
        if tid is None or tid not in new_set:
            continue

        is_lower = getattr(obj, "is_in_lower_half", False)
        area_frac = getattr(obj, "box_area_fraction", 0.0) or 0.0
        cx = getattr(obj, "box_center_x_px", width / 2.0)
        x_ratio = cx / width

        in_corridor = config.vru_corridor_x_min <= x_ratio <= config.vru_corridor_x_max

        if is_lower and in_corridor and area_frac >= config.cut_in_min_area:
            cat = getattr(obj, "category", "object")
            basis = (
                f"Sudden track appearance: New track {tid} ('{cat}') appeared abruptly in forward "
                f"travel corridor with significant visual area ({area_frac:.1%} >= {config.cut_in_min_area:.1%}). "
                f"Consistent with a sudden lane change or vehicle cut-in."
            )

            indicators.append(
                RiskIndicator(
                    indicator_id=IndicatorId.SUDDEN_CUT_IN.value,
                    name="Sudden Forward Path Intrusion",
                    severity=Severity.MEDIUM,
                    triggered=True,
                    basis=basis,
                    source_data={
                        "track_id": tid,
                        "category": cat,
                        "box_area_fraction": round(float(area_frac), 4),
                        "corridor_ratio": round(float(x_ratio), 3),
                    },
                )
            )

    return indicators


def evaluate_traffic_density(
    objects: list[Any],
    config: RiskIndicatorConfig,
) -> Optional[RiskIndicator]:
    """
    Evaluates elevated traffic density in the forward visual field.

    Basis: High object counts increase visual clutter and occlusion hazards.
    """
    total = len(objects)
    lower_count = sum(1 for o in objects if getattr(o, "is_in_lower_half", False))

    if total >= config.high_density_object_count or lower_count >= config.high_density_lower_count:
        basis = (
            f"High traffic density: {total} total road objects detected ({lower_count} in lower forward view, "
            f"thresholds: total>={config.high_density_object_count}, lower>={config.high_density_lower_count}). "
            f"High spatial clutter elevates situational complexity and visual occlusion risks."
        )

        return RiskIndicator(
            indicator_id=IndicatorId.HIGH_TRAFFIC_DENSITY.value,
            name="High Traffic Clutter Density",
            severity=Severity.LOW,
            triggered=True,
            basis=basis,
            source_data={
                "total_objects": total,
                "objects_in_lower_half": lower_count,
            },
        )

    return None

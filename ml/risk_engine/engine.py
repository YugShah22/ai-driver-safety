"""
Core Risk Engine orchestrator for Phase 13.

Consumes structured driving features and produces a documented, explainable
driving assessment with explicit 5-layer separation:
1. Raw observations
2. Derived features
3. Risk indicators (each with a documented empirical basis)
4. Final assessment (with documented score meaning)
5. Explanations (human-readable evidence breakdown)
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Union

from ml.feature_extraction.schema import FrameFeatures, LaneFeature, ObjectFeature

from .capabilities import DISCLAIMER_NOTICE, SCORE_MEANING_DOCUMENTATION
from .indicators import (
    RiskIndicatorConfig,
    evaluate_adverse_environment,
    evaluate_lane_departure,
    evaluate_nearby_vehicles,
    evaluate_persistent_obstructions,
    evaluate_sudden_cut_in,
    evaluate_traffic_density,
    evaluate_vulnerable_road_users,
)
from .schema import (
    Assessment,
    AssessmentLevel,
    DrivingAssessmentResult,
    Observation,
    RiskIndicator,
    Severity,
)

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Deterministic, explainable risk assessment engine.

    Consumes FrameFeatures (or a dictionary representation) and synthesizes
    an evidence-grounded driving assessment with no arbitrary/magical scores.
    """

    def __init__(self, config: Optional[RiskIndicatorConfig] = None) -> None:
        self.config = config or RiskIndicatorConfig()

    def evaluate(
        self,
        frame_input: Union[FrameFeatures, dict[str, Any]],
    ) -> DrivingAssessmentResult:
        """
        Evaluate driving risk for a single frame or structured feature dictionary.

        Returns a DrivingAssessmentResult adhering strictly to the 5-layer schema.
        """
        # Step 0: Input validation and normalization
        normalized = self._validate_and_normalize(frame_input)

        # Layer 1: Raw Observations
        observations = self._extract_observations(normalized)

        # Layer 2: Derived Features
        features = self._extract_derived_features(normalized)

        # Layer 3: Risk Indicators (each with documented empirical basis)
        indicators = self._evaluate_indicators(normalized)

        # Layer 4: Final Assessment
        assessment = self._synthesize_assessment(indicators)

        # Layer 5: Explanations
        explanations = self._generate_explanations(indicators, assessment)

        # Metadata
        metadata = {
            "frame_id": normalized["frame_id"],
            "frame_index": normalized["frame_index"],
            "timestamp_s": normalized.get("timestamp_s"),
            "engine_version": "13.0.0",
            "disclaimer": DISCLAIMER_NOTICE,
        }

        return DrivingAssessmentResult(
            observations=[obs.to_dict() for obs in observations],
            features=features,
            risk_indicators=[ind.to_dict() for ind in indicators],
            assessment=assessment.to_dict(),
            explanations=explanations,
            metadata=metadata,
        )

    # ─────────────────────────────────────────────────────────────
    # Input Validation & Normalization
    # ─────────────────────────────────────────────────────────────

    def _validate_and_normalize(
        self,
        frame_input: Union[FrameFeatures, dict[str, Any]],
    ) -> dict[str, Any]:
        """Validate input types, structural integrity, and normalize to standard dict."""
        if frame_input is None:
            raise ValueError("Input to RiskEngine.evaluate cannot be None.")

        if isinstance(frame_input, FrameFeatures):
            return self._normalize_from_dataclass(frame_input)

        if isinstance(frame_input, dict):
            return self._normalize_from_dict(frame_input)

        # Check for duck-typed objects with frame_id and image_width
        if hasattr(frame_input, "frame_id") and hasattr(frame_input, "image_width"):
            return self._normalize_from_dataclass(frame_input)

        raise TypeError(
            f"Expected FrameFeatures instance or dict, got {type(frame_input).__name__}."
        )

    def _normalize_from_dataclass(self, frame: Any) -> dict[str, Any]:
        width = getattr(frame, "image_width", 0)
        height = getattr(frame, "image_height", 0)

        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError(
                f"image_width ({width}) and image_height ({height}) must be positive integers."
            )

        objects = getattr(frame, "objects", [])
        if not isinstance(objects, list):
            raise ValueError("Frame 'objects' must be a list.")

        self._validate_objects(objects, width, height)

        return {
            "frame_id": str(getattr(frame, "frame_id", "frame_0")),
            "frame_index": int(getattr(frame, "frame_index", 0)),
            "timestamp_s": getattr(frame, "timestamp_s", None),
            "image_width": width,
            "image_height": height,
            "scene_class": getattr(frame, "scene_class", None),
            "scene_class_id": getattr(frame, "scene_class_id", None),
            "objects": objects,
            "object_counts": getattr(frame, "object_counts", {}),
            "lane": getattr(frame, "lane", None),
            "new_track_ids": getattr(frame, "new_track_ids", []),
            "lost_track_ids": getattr(frame, "lost_track_ids", []),
        }

    def _normalize_from_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        if "image_width" not in data or "image_height" not in data:
            raise ValueError("Dictionary input must contain 'image_width' and 'image_height'.")

        width = data["image_width"]
        height = data["image_height"]

        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError(
                f"image_width ({width}) and image_height ({height}) must be positive integers."
            )

        raw_objects = data.get("objects", [])
        if not isinstance(raw_objects, list):
            raise ValueError("'objects' in dictionary input must be a list.")

        # Convert dict objects to ObjectFeature if needed
        parsed_objects = []
        for i, obj in enumerate(raw_objects):
            if isinstance(obj, ObjectFeature):
                parsed_objects.append(obj)
            elif isinstance(obj, dict):
                parsed_objects.append(self._dict_to_object_feature(obj, width, height, i))
            elif hasattr(obj, "category") and hasattr(obj, "box_area_fraction"):
                parsed_objects.append(obj)
            else:
                raise TypeError(
                    f"Object at index {i} is malformed: expected ObjectFeature or dict, got {type(obj).__name__}."
                )

        self._validate_objects(parsed_objects, width, height)

        raw_lane = data.get("lane")
        parsed_lane = None
        if isinstance(raw_lane, LaneFeature):
            parsed_lane = raw_lane
        elif isinstance(raw_lane, dict):
            parsed_lane = LaneFeature(
                is_valid=bool(raw_lane.get("is_valid", False)),
                status=str(raw_lane.get("status", "unknown")),
                lane_center_x_px=raw_lane.get("lane_center_x_px"),
                image_center_x_px=raw_lane.get("image_center_x_px"),
                left_boundary_x_px=raw_lane.get("left_boundary_x_px"),
                right_boundary_x_px=raw_lane.get("right_boundary_x_px"),
                lateral_offset_px=raw_lane.get("lateral_offset_px"),
                lane_departure_indicator=raw_lane.get("lane_departure_indicator"),
                road_pixel_fraction=raw_lane.get("road_pixel_fraction"),
            )
        elif raw_lane is not None:
            # Duck typed or unknown
            parsed_lane = raw_lane

        counts = data.get("object_counts")
        if not isinstance(counts, dict):
            counts = {}
            for o in parsed_objects:
                c = getattr(o, "category", "")
                counts[c] = counts.get(c, 0) + 1

        new_ids = data.get("new_track_ids", [])
        if not isinstance(new_ids, list):
            new_ids = []

        lost_ids = data.get("lost_track_ids", [])
        if not isinstance(lost_ids, list):
            lost_ids = []

        return {
            "frame_id": str(data.get("frame_id", "frame_0")),
            "frame_index": int(data.get("frame_index", 0)),
            "timestamp_s": data.get("timestamp_s"),
            "image_width": width,
            "image_height": height,
            "scene_class": data.get("scene_class"),
            "scene_class_id": data.get("scene_class_id"),
            "objects": parsed_objects,
            "object_counts": counts,
            "lane": parsed_lane,
            "new_track_ids": new_ids,
            "lost_track_ids": lost_ids,
        }

    def _dict_to_object_feature(
        self,
        d: dict[str, Any],
        width: int,
        height: int,
        index: int,
    ) -> ObjectFeature:
        if "category" not in d:
            raise ValueError(f"Object at index {index} is missing required 'category' key.")

        box_x1 = float(d.get("box_x1", 0.0))
        box_y1 = float(d.get("box_y1", 0.0))
        box_x2 = float(d.get("box_x2", 0.0))
        box_y2 = float(d.get("box_y2", 0.0))

        cx = float(d.get("box_center_x_px", (box_x1 + box_x2) / 2.0))
        cy = float(d.get("box_center_y_px", (box_y1 + box_y2) / 2.0))

        area_frac = d.get("box_area_fraction")
        if area_frac is None:
            box_area = max(0.0, box_x2 - box_x1) * max(0.0, box_y2 - box_y1)
            total_area = width * height
            area_frac = box_area / total_area if total_area > 0 else 0.0
        else:
            area_frac = float(area_frac)

        is_lower = d.get("is_in_lower_half")
        if is_lower is None:
            is_lower = cy > (height / 2.0)
        else:
            is_lower = bool(is_lower)

        return ObjectFeature(
            track_id=d.get("track_id"),
            category=str(d["category"]),
            confidence=float(d.get("confidence", 1.0)),
            box_x1=box_x1,
            box_y1=box_y1,
            box_x2=box_x2,
            box_y2=box_y2,
            box_center_x_px=cx,
            box_center_y_px=cy,
            box_area_fraction=area_frac,
            is_in_lower_half=is_lower,
            track_age=d.get("track_age"),
            track_hits=d.get("track_hits"),
        )

    def _validate_objects(self, objects: list[Any], width: int, height: int) -> None:
        """Validate numeric coordinates and ranges for all objects."""
        for i, obj in enumerate(objects):
            area_frac = getattr(obj, "box_area_fraction", 0.0)
            if area_frac is not None and (area_frac < 0.0 or area_frac > 1.5):
                raise ValueError(
                    f"Object {i} has invalid box_area_fraction ({area_frac}): expected between 0.0 and 1.0."
                )

    # ─────────────────────────────────────────────────────────────
    # Layer 1: Raw Observations
    # ─────────────────────────────────────────────────────────────

    def _extract_observations(self, normalized: dict[str, Any]) -> list[Observation]:
        """Extract explicit raw perception facts from upstream models."""
        obs: list[Observation] = []

        # Object detections
        for obj in normalized["objects"]:
            obs.append(
                Observation(
                    observation_type="detected_object",
                    source="yolo_tracker",
                    details={
                        "category": getattr(obj, "category", "unknown"),
                        "track_id": getattr(obj, "track_id", None),
                        "confidence": round(float(getattr(obj, "confidence", 0.0)), 3),
                        "box": [
                            round(float(getattr(obj, "box_x1", 0.0)), 1),
                            round(float(getattr(obj, "box_y1", 0.0)), 1),
                            round(float(getattr(obj, "box_x2", 0.0)), 1),
                            round(float(getattr(obj, "box_y2", 0.0)), 1),
                        ],
                        "box_area_fraction": round(float(getattr(obj, "box_area_fraction", 0.0)), 4),
                        "is_in_lower_half": getattr(obj, "is_in_lower_half", False),
                    },
                )
            )

        # Lane observation
        lane = normalized.get("lane")
        if lane is not None:
            obs.append(
                Observation(
                    observation_type="lane_geometry",
                    source="unet_segmenter",
                    details={
                        "is_valid": getattr(lane, "is_valid", False),
                        "status": getattr(lane, "status", "unknown"),
                        "lateral_offset_px": getattr(lane, "lateral_offset_px", None),
                        "lane_departure_indicator": getattr(lane, "lane_departure_indicator", None),
                        "road_pixel_fraction": getattr(lane, "road_pixel_fraction", None),
                    },
                )
            )

        # Scene observation
        scene_class = normalized.get("scene_class")
        if scene_class is not None:
            obs.append(
                Observation(
                    observation_type="scene_classification",
                    source="cnn_classifier",
                    details={
                        "scene_class": scene_class,
                        "scene_class_id": normalized.get("scene_class_id"),
                    },
                )
            )

        return obs

    # ─────────────────────────────────────────────────────────────
    # Layer 2: Derived Features
    # ─────────────────────────────────────────────────────────────

    def _extract_derived_features(self, normalized: dict[str, Any]) -> dict[str, Any]:
        """Produce structured summary of derived features."""
        objects = normalized["objects"]
        lane = normalized.get("lane")

        max_area = max((getattr(o, "box_area_fraction", 0.0) or 0.0 for o in objects), default=0.0)
        n_lower = sum(1 for o in objects if getattr(o, "is_in_lower_half", False))

        has_lane = bool(lane is not None and getattr(lane, "is_valid", False))
        offset_px = getattr(lane, "lateral_offset_px", None) if has_lane else None
        departure = bool(lane is not None and getattr(lane, "lane_departure_indicator", False))
        road_frac = getattr(lane, "road_pixel_fraction", None) if has_lane else None

        return {
            "n_objects": len(objects),
            "category_counts": dict(normalized.get("object_counts", {})),
            "max_box_area_fraction": round(float(max_area), 4),
            "n_objects_lower_half": n_lower,
            "n_new_tracks": len(normalized.get("new_track_ids", [])),
            "n_lost_tracks": len(normalized.get("lost_track_ids", [])),
            "has_lane_info": has_lane,
            "lateral_offset_px": round(float(offset_px), 1) if offset_px is not None else None,
            "lane_departure_flagged": departure,
            "road_pixel_fraction": round(float(road_frac), 4) if road_frac is not None else None,
            "scene_class": normalized.get("scene_class"),
        }

    # ─────────────────────────────────────────────────────────────
    # Layer 3: Risk Indicators
    # ─────────────────────────────────────────────────────────────

    def _evaluate_indicators(self, normalized: dict[str, Any]) -> list[RiskIndicator]:
        """Evaluate all documented risk indicators against normalized inputs."""
        objects = normalized["objects"]
        width = normalized["image_width"]
        lane = normalized.get("lane")
        scene_class = normalized.get("scene_class")
        new_track_ids = normalized.get("new_track_ids", [])

        indicators: list[RiskIndicator] = []

        # 1. Nearby vehicles
        indicators.extend(evaluate_nearby_vehicles(objects, self.config))

        # 2. Vulnerable road users in forward corridor
        indicators.extend(evaluate_vulnerable_road_users(objects, width, self.config))

        # 3. Lane departure
        lane_ind = evaluate_lane_departure(lane, self.config)
        if lane_ind is not None:
            indicators.append(lane_ind)

        # 4. Persistent obstructions
        indicators.extend(evaluate_persistent_obstructions(objects, width, self.config))

        # 5. Adverse environment
        env_ind = evaluate_adverse_environment(scene_class, lane, self.config)
        if env_ind is not None:
            indicators.append(env_ind)

        # 6. Sudden cut-in
        indicators.extend(evaluate_sudden_cut_in(objects, new_track_ids, width, self.config))

        # 7. Traffic density
        density_ind = evaluate_traffic_density(objects, self.config)
        if density_ind is not None:
            indicators.append(density_ind)

        return indicators

    # ─────────────────────────────────────────────────────────────
    # Layer 4: Final Assessment
    # ─────────────────────────────────────────────────────────────

    def _synthesize_assessment(self, indicators: list[RiskIndicator]) -> Assessment:
        """Synthesize overall driving risk assessment from active indicators."""
        active = [ind for ind in indicators if ind.triggered]
        count = len(active)

        if count == 0:
            return Assessment(
                level=AssessmentLevel.NOMINAL,
                summary="Nominal driving conditions. No visual risk indicators triggered.",
                active_indicators_count=0,
                highest_severity=None,
                composite_score=0.0,
                score_meaning=SCORE_MEANING_DOCUMENTATION,
            )

        # Determine highest severity
        severity_rank = {
            Severity.LOW: 1,
            Severity.MEDIUM: 2,
            Severity.HIGH: 3,
            Severity.CRITICAL: 4,
        }
        highest = max(active, key=lambda ind: severity_rank[ind.severity]).severity

        # Deterministic composite score: weighted sum capped at 100
        raw_score = sum(ind.severity.score_weight for ind in active)
        composite_score = min(100.0, float(raw_score))

        # Map to assessment level
        if highest == Severity.CRITICAL or composite_score >= 80.0:
            level = AssessmentLevel.CRITICAL_RISK
        elif highest == Severity.HIGH or composite_score >= 50.0:
            level = AssessmentLevel.HIGH_RISK
        elif highest == Severity.MEDIUM or composite_score >= 25.0:
            level = AssessmentLevel.ELEVATED_CAUTION
        else:
            level = AssessmentLevel.LOW_CAUTION

        # Build readable summary
        hazard_names = [ind.name for ind in active[:3]]
        summary = (
            f"Assessment: {level.value}. {count} active visual hazard indicator(s) detected: "
            f"{', '.join(hazard_names)}."
        )

        return Assessment(
            level=level,
            summary=summary,
            active_indicators_count=count,
            highest_severity=highest,
            composite_score=composite_score,
            score_meaning=SCORE_MEANING_DOCUMENTATION,
        )

    # ─────────────────────────────────────────────────────────────
    # Layer 5: Explanations
    # ─────────────────────────────────────────────────────────────

    def _generate_explanations(
        self,
        indicators: list[RiskIndicator],
        assessment: Assessment,
    ) -> list[str]:
        """Generate human-readable explanations citing specific evidence and limitations."""
        explanations: list[str] = []
        active = [ind for ind in indicators if ind.triggered]

        if not active:
            explanations.append(
                "Nominal state: All detected objects, lane markings, and scene conditions "
                "remain within safe visual operating parameters."
            )
        else:
            for ind in active:
                explanations.append(f"[{ind.severity.value}] {ind.name}: {ind.basis}")

        # Add explicit capability boundary disclosure
        explanations.append(
            "Boundary disclosure: Assessment is based entirely on 2D forward-camera optical proxies. "
            "Physical metric distance, ego-vehicle speed, driver attention, and exact collision "
            "probability are NOT measured or inferred."
        )

        return explanations

"""
Tests for Phase 13 Risk Engine and Driving Insights.

Covers:
- System capabilities, boundaries, and documented claims
- Output structure compliance (observations, features, risk_indicators, assessment, explanations)
- No-risk scenario
- One-risk-indicator scenarios (each indicator type)
- Multiple compounding indicators
- Missing features handling
- Malformed input validation
- Dict vs FrameFeatures input compatibility
- Configurable indicator thresholds
"""
from __future__ import annotations

import pytest

from ml.feature_extraction.schema import FrameFeatures, LaneFeature, ObjectFeature
from ml.risk_engine import (
    CAN_INFER,
    CANNOT_INFER,
    DISCLAIMER_NOTICE,
    SCORE_MEANING_DOCUMENTATION,
    AssessmentLevel,
    IndicatorId,
    RiskEngine,
    RiskIndicatorConfig,
    Severity,
    SystemCapabilities,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_nominal_frame() -> FrameFeatures:
    """A clean driving frame with no hazards."""
    lane = LaneFeature(
        is_valid=True,
        status="detected",
        lane_center_x_px=640.0,
        image_center_x_px=640.0,
        lateral_offset_px=0.0,
        lane_departure_indicator=False,
        road_pixel_fraction=0.35,
    )
    return FrameFeatures(
        frame_id="frame_001",
        frame_index=1,
        timestamp_s=0.1,
        image_width=1280,
        image_height=720,
        scene_class="highway",
        scene_class_id=0,
        objects=[],
        object_counts={},
        lane=lane,
        new_track_ids=[],
        lost_track_ids=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. System Capabilities & Explicit Operational Boundaries
# ─────────────────────────────────────────────────────────────────────────────

class TestCapabilitiesAndBoundaries:
    def test_can_infer_is_documented(self):
        assert len(CAN_INFER) >= 5
        # Verify it mentions 2D visual proxies
        assert any("2D" in c for c in CAN_INFER)
        assert any("proxies" in c for c in CAN_INFER)

    def test_cannot_infer_explicitly_forbids_unvalidated_claims(self):
        # Must explicitly forbid metric distance, speed, TTC, collision prob, driver attention, medical
        disallowed = [
            "Exact physical metric distance",
            "Exact vehicle speed",
            "Exact physical time-to-collision",
            "Exact physical collision probability",
            "Driver attention",
            "Driver medical",
        ]
        for term in disallowed:
            assert any(term in claim for claim in CANNOT_INFER), f"Missing prohibited claim: {term}"

    def test_score_meaning_documentation_is_explicit(self):
        assert "0–100" in SCORE_MEANING_DOCUMENTATION
        assert "NOT represent a physical collision probability" in SCORE_MEANING_DOCUMENTATION

    def test_system_capabilities_summary(self):
        summary = SystemCapabilities.get_summary()
        assert "can_infer" in summary
        assert "cannot_infer" in summary
        assert "score_meaning" in summary
        assert "disclaimer" in summary


# ─────────────────────────────────────────────────────────────────────────────
# 2. Output Schema Contract
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputContract:
    def test_output_contains_all_five_separated_layers(self):
        engine = RiskEngine()
        frame = make_nominal_frame()
        result = engine.evaluate(frame)
        data = result.to_dict()

        # Contract: { observations: [...], features: {...], risk_indicators: [...], assessment: ..., explanations: [...] }
        assert "observations" in data
        assert isinstance(data["observations"], list)

        assert "features" in data
        assert isinstance(data["features"], dict)

        assert "risk_indicators" in data
        assert isinstance(data["risk_indicators"], list)

        assert "assessment" in data
        assert isinstance(data["assessment"], dict)

        assert "explanations" in data
        assert isinstance(data["explanations"], list)

    def test_assessment_contains_documented_score(self):
        engine = RiskEngine()
        frame = make_nominal_frame()
        res = engine.evaluate(frame).to_dict()
        assessment = res["assessment"]

        assert "level" in assessment
        assert "composite_score" in assessment
        assert "score_meaning" in assessment
        assert assessment["score_meaning"] == SCORE_MEANING_DOCUMENTATION


# ─────────────────────────────────────────────────────────────────────────────
# 3. No-Risk Scenario
# ─────────────────────────────────────────────────────────────────────────────

class TestNoRiskScenario:
    def test_nominal_frame_produces_no_risk(self):
        engine = RiskEngine()
        frame = make_nominal_frame()
        result = engine.evaluate(frame)
        data = result.to_dict()

        assert data["assessment"]["level"] == AssessmentLevel.NOMINAL.value
        assert data["assessment"]["active_indicators_count"] == 0
        assert data["assessment"]["highest_severity"] is None
        assert data["assessment"]["composite_score"] == 0.0
        assert len(data["risk_indicators"]) == 0

        # Explanations explain nominal state and provide boundary disclosure
        explanations = data["explanations"]
        assert len(explanations) >= 2
        assert "Nominal state" in explanations[0]
        assert "Boundary disclosure" in explanations[-1]


# ─────────────────────────────────────────────────────────────────────────────
# 4. One-Risk-Indicator Scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleIndicatorScenarios:
    def test_nearby_vehicle_indicator(self):
        engine = RiskEngine()
        frame = make_nominal_frame()
        # Add vehicle with large area fraction in lower half
        frame.objects = [
            ObjectFeature(
                track_id=10,
                category="car",
                confidence=0.92,
                box_x1=400.0,
                box_y1=400.0,
                box_x2=880.0,
                box_y2=680.0,
                box_center_x_px=640.0,
                box_center_y_px=540.0,
                box_area_fraction=0.15,  # >= 0.12 threshold
                is_in_lower_half=True,
                track_age=5,
                track_hits=5,
            )
        ]
        frame.object_counts = {"car": 1}

        result = engine.evaluate(frame)
        data = result.to_dict()

        assert data["assessment"]["active_indicators_count"] == 1
        assert data["assessment"]["highest_severity"] == Severity.MEDIUM.value
        assert data["assessment"]["level"] == AssessmentLevel.ELEVATED_CAUTION.value
        assert data["assessment"]["composite_score"] == 25.0

        ind = data["risk_indicators"][0]
        assert ind["indicator_id"] == IndicatorId.NEARBY_VEHICLE.value
        assert ind["severity"] == Severity.MEDIUM.value
        assert "2D optical proxy" in ind["basis"]
        assert ind["source_data"]["category"] == "car"
        assert ind["source_data"]["box_area_fraction"] == 0.15

        # Explanation cites the indicator
        assert any(IndicatorId.NEARBY_VEHICLE.value in exp or "Nearby Vehicle" in exp for exp in data["explanations"])

    def test_vulnerable_road_user_in_path_indicator(self):
        engine = RiskEngine()
        frame = make_nominal_frame()
        # Pedestrian in central corridor
        frame.objects = [
            ObjectFeature(
                track_id=7,
                category="person",
                confidence=0.88,
                box_x1=600.0,
                box_y1=450.0,
                box_x2=680.0,
                box_y2=650.0,
                box_center_x_px=640.0,
                box_center_y_px=550.0,
                box_area_fraction=0.03,  # >= 0.015 threshold
                is_in_lower_half=True,
                track_age=3,
                track_hits=3,
            )
        ]
        frame.object_counts = {"person": 1}

        result = engine.evaluate(frame)
        data = result.to_dict()

        assert data["assessment"]["active_indicators_count"] == 1
        assert data["assessment"]["highest_severity"] == Severity.HIGH.value
        assert data["assessment"]["level"] == AssessmentLevel.HIGH_RISK.value
        assert data["assessment"]["composite_score"] == 50.0

        ind = data["risk_indicators"][0]
        assert ind["indicator_id"] == IndicatorId.VULNERABLE_ROAD_USER_PATH.value
        assert ind["severity"] == Severity.HIGH.value
        assert "Vulnerable road user" in ind["basis"]

    def test_lane_departure_indicator(self):
        engine = RiskEngine()
        frame = make_nominal_frame()
        # Lane departure flag active
        frame.lane = LaneFeature(
            is_valid=True,
            status="departure",
            lane_center_x_px=550.0,
            image_center_x_px=640.0,
            lateral_offset_px=-90.0,
            lane_departure_indicator=True,
            road_pixel_fraction=0.30,
        )

        result = engine.evaluate(frame)
        data = result.to_dict()

        assert data["assessment"]["active_indicators_count"] == 1
        assert data["assessment"]["level"] == AssessmentLevel.ELEVATED_CAUTION.value
        ind = data["risk_indicators"][0]
        assert ind["indicator_id"] == IndicatorId.LANE_DEPARTURE.value
        assert ind["source_data"]["lane_departure_indicator"] is True
        assert ind["source_data"]["lateral_offset_px"] == -90.0

    def test_persistent_obstruction_indicator(self):
        engine = RiskEngine()
        frame = make_nominal_frame()
        # Persistent object tracked for 20 frames in forward corridor
        frame.objects = [
            ObjectFeature(
                track_id=4,
                category="truck",
                confidence=0.85,
                box_x1=500.0,
                box_y1=420.0,
                box_x2=780.0,
                box_y2=600.0,
                box_center_x_px=640.0,
                box_center_y_px=510.0,
                box_area_fraction=0.06,  # >= 0.04 threshold
                is_in_lower_half=True,
                track_age=20,            # >= 15 threshold
                track_hits=20,
            )
        ]
        frame.object_counts = {"truck": 1}

        result = engine.evaluate(frame)
        data = result.to_dict()

        assert any(ind["indicator_id"] == IndicatorId.PERSISTENT_OBSTRUCTION.value for ind in data["risk_indicators"])
        p_ind = next(ind for ind in data["risk_indicators"] if ind["indicator_id"] == IndicatorId.PERSISTENT_OBSTRUCTION.value)
        assert p_ind["source_data"]["track_age"] == 20
        assert "Persistent obstruction" in p_ind["basis"]

    def test_adverse_environment_indicator(self):
        engine = RiskEngine()
        frame = make_nominal_frame()
        frame.scene_class = "heavy_rain"

        result = engine.evaluate(frame)
        data = result.to_dict()

        assert data["assessment"]["active_indicators_count"] == 1
        ind = data["risk_indicators"][0]
        assert ind["indicator_id"] == IndicatorId.ADVERSE_ENVIRONMENT.value
        assert "Adverse Environmental" in ind["name"]
        assert "heavy_rain" in ind["basis"]

    def test_sudden_cut_in_indicator(self):
        engine = RiskEngine()
        frame = make_nominal_frame()
        frame.new_track_ids = [99]
        frame.objects = [
            ObjectFeature(
                track_id=99,
                category="car",
                confidence=0.90,
                box_x1=450.0,
                box_y1=400.0,
                box_x2=750.0,
                box_y2=600.0,
                box_center_x_px=600.0,
                box_center_y_px=500.0,
                box_area_fraction=0.09,  # >= 0.08 threshold
                is_in_lower_half=True,
                track_age=1,
                track_hits=1,
            )
        ]

        result = engine.evaluate(frame)
        data = result.to_dict()

        assert any(ind["indicator_id"] == IndicatorId.SUDDEN_CUT_IN.value for ind in data["risk_indicators"])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Multiple Compounding Indicators
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleIndicatorsScenario:
    def test_multiple_compounding_hazards(self):
        engine = RiskEngine()
        frame = make_nominal_frame()

        # 1. Pedestrian in path (HIGH, weight=50)
        vru = ObjectFeature(
            track_id=1,
            category="person",
            confidence=0.91,
            box_x1=620.0,
            box_y1=450.0,
            box_x2=700.0,
            box_y2=660.0,
            box_center_x_px=660.0,
            box_center_y_px=555.0,
            box_area_fraction=0.03,
            is_in_lower_half=True,
            track_age=2,
            track_hits=2,
        )
        # 2. Nearby vehicle (MEDIUM, weight=25)
        veh = ObjectFeature(
            track_id=2,
            category="car",
            confidence=0.95,
            box_x1=200.0,
            box_y1=420.0,
            box_x2=550.0,
            box_y2=680.0,
            box_center_x_px=375.0,
            box_center_y_px=550.0,
            box_area_fraction=0.14,
            is_in_lower_half=True,
            track_age=10,
            track_hits=10,
        )
        frame.objects = [vru, veh]
        frame.object_counts = {"person": 1, "car": 1}

        # 3. Lane departure (MEDIUM, weight=25)
        frame.lane.lane_departure_indicator = True
        frame.lane.lateral_offset_px = 75.0

        # 4. Adverse weather ("fog", MEDIUM, weight=25)
        frame.scene_class = "fog"

        result = engine.evaluate(frame)
        data = result.to_dict()

        # Active indicators: VRU, Nearby Vehicle, Lane Departure, Fog (4 indicators)
        assert data["assessment"]["active_indicators_count"] == 4
        # Total weight: 50 + 25 + 25 + 25 = 125, capped at 100.0
        assert data["assessment"]["composite_score"] == 100.0
        assert data["assessment"]["highest_severity"] == Severity.HIGH.value
        # Since score >= 80, level elevates to CRITICAL_RISK
        assert data["assessment"]["level"] == AssessmentLevel.CRITICAL_RISK.value

        # Explanations explain all triggered items
        explanations = data["explanations"]
        assert len(explanations) == 5  # 4 indicators + 1 boundary disclosure
        assert any("Vulnerable Road User" in exp for exp in explanations)
        assert any("Nearby Vehicle" in exp for exp in explanations)
        assert any("Lane Departure" in exp for exp in explanations)
        assert any("Adverse Environmental" in exp for exp in explanations)
        assert "Boundary disclosure" in explanations[-1]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Missing Features Handling
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingFeatures:
    def test_missing_lane_handled_gracefully(self):
        engine = RiskEngine()
        frame = make_nominal_frame()
        frame.lane = None

        result = engine.evaluate(frame)
        data = result.to_dict()

        assert data["features"]["has_lane_info"] is False
        assert data["features"]["lateral_offset_px"] is None
        assert data["assessment"]["level"] == AssessmentLevel.NOMINAL.value

    def test_missing_scene_handled_gracefully(self):
        engine = RiskEngine()
        frame = make_nominal_frame()
        frame.scene_class = None
        frame.scene_class_id = None

        result = engine.evaluate(frame)
        data = result.to_dict()

        assert data["features"]["scene_class"] is None
        assert data["assessment"]["level"] == AssessmentLevel.NOMINAL.value

    def test_empty_objects_handled_gracefully(self):
        engine = RiskEngine()
        frame = make_nominal_frame()
        frame.objects = []
        frame.object_counts = {}

        result = engine.evaluate(frame)
        data = result.to_dict()

        assert data["features"]["n_objects"] == 0
        assert data["features"]["max_box_area_fraction"] == 0.0

    def test_missing_timestamp_handled_gracefully(self):
        engine = RiskEngine()
        frame = make_nominal_frame()
        frame.timestamp_s = None

        result = engine.evaluate(frame)
        data = result.to_dict()
        assert data["metadata"]["timestamp_s"] is None

    def test_dict_with_minimal_keys(self):
        engine = RiskEngine()
        # Bare minimal dictionary with only dimensions
        min_dict = {
            "image_width": 1920,
            "image_height": 1080,
        }
        result = engine.evaluate(min_dict)
        data = result.to_dict()

        assert data["assessment"]["level"] == AssessmentLevel.NOMINAL.value
        assert data["features"]["n_objects"] == 0
        assert data["features"]["has_lane_info"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 7. Malformed Inputs Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestMalformedInputs:
    def test_none_input_raises_value_error(self):
        engine = RiskEngine()
        with pytest.raises(ValueError, match="cannot be None"):
            engine.evaluate(None)  # type: ignore

    def test_invalid_type_raises_type_error(self):
        engine = RiskEngine()
        with pytest.raises(TypeError, match="Expected FrameFeatures instance or dict"):
            engine.evaluate([1, 2, 3])  # type: ignore

    def test_non_positive_image_dimensions_raises_value_error(self):
        engine = RiskEngine()
        with pytest.raises(ValueError, match="must be positive integers"):
            engine.evaluate({"image_width": 0, "image_height": 720})

        with pytest.raises(ValueError, match="must be positive integers"):
            engine.evaluate({"image_width": 1280, "image_height": -10})

    def test_missing_image_dimensions_in_dict_raises_value_error(self):
        engine = RiskEngine()
        with pytest.raises(ValueError, match="must contain 'image_width' and 'image_height'"):
            engine.evaluate({"frame_id": "test"})

    def test_malformed_objects_type_raises_value_error(self):
        engine = RiskEngine()
        with pytest.raises(ValueError, match="must be a list"):
            engine.evaluate({"image_width": 640, "image_height": 480, "objects": "not-a-list"})

    def test_malformed_object_element_raises_type_error(self):
        engine = RiskEngine()
        with pytest.raises(TypeError, match="Object at index 0 is malformed"):
            engine.evaluate({"image_width": 640, "image_height": 480, "objects": [12345]})

    def test_dict_object_missing_category_raises_value_error(self):
        engine = RiskEngine()
        with pytest.raises(ValueError, match="missing required 'category' key"):
            engine.evaluate({
                "image_width": 640,
                "image_height": 480,
                "objects": [{"box_x1": 10.0, "box_y1": 20.0}],
            })

    def test_invalid_box_area_fraction_raises_value_error(self):
        engine = RiskEngine()
        with pytest.raises(ValueError, match="invalid box_area_fraction"):
            engine.evaluate({
                "image_width": 640,
                "image_height": 480,
                "objects": [{"category": "car", "box_area_fraction": 2.5}],
            })


# ─────────────────────────────────────────────────────────────────────────────
# 8. Dictionary vs DataClass Input Compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestDictVsDataclassEquivalence:
    def test_dict_input_produces_identical_assessment(self):
        engine = RiskEngine()
        frame = make_nominal_frame()
        frame.objects = [
            ObjectFeature(
                track_id=5,
                category="car",
                confidence=0.90,
                box_x1=300.0,
                box_y1=400.0,
                box_x2=700.0,
                box_y2=650.0,
                box_center_x_px=500.0,
                box_center_y_px=525.0,
                box_area_fraction=0.18,
                is_in_lower_half=True,
                track_age=8,
                track_hits=8,
            )
        ]
        frame.object_counts = {"car": 1}

        res_dc = engine.evaluate(frame).to_dict()

        dict_input = {
            "frame_id": frame.frame_id,
            "frame_index": frame.frame_index,
            "timestamp_s": frame.timestamp_s,
            "image_width": frame.image_width,
            "image_height": frame.image_height,
            "scene_class": frame.scene_class,
            "scene_class_id": frame.scene_class_id,
            "objects": [
                {
                    "track_id": 5,
                    "category": "car",
                    "confidence": 0.90,
                    "box_x1": 300.0,
                    "box_y1": 400.0,
                    "box_x2": 700.0,
                    "box_y2": 650.0,
                    "box_area_fraction": 0.18,
                    "is_in_lower_half": True,
                    "track_age": 8,
                    "track_hits": 8,
                }
            ],
            "lane": {
                "is_valid": True,
                "status": "detected",
                "lateral_offset_px": 0.0,
                "lane_departure_indicator": False,
            },
        }

        res_dict = engine.evaluate(dict_input).to_dict()

        assert res_dc["assessment"]["level"] == res_dict["assessment"]["level"]
        assert res_dc["assessment"]["composite_score"] == res_dict["assessment"]["composite_score"]
        assert len(res_dc["risk_indicators"]) == len(res_dict["risk_indicators"])
        assert res_dc["risk_indicators"][0]["indicator_id"] == res_dict["risk_indicators"][0]["indicator_id"]


# ─────────────────────────────────────────────────────────────────────────────
# 9. Configurable Thresholds
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigurableThresholds:
    def test_custom_thresholds_adjust_indicator_trigger(self):
        # Default threshold is 0.12. An object with area 0.10 wouldn't trigger default.
        custom_config = RiskIndicatorConfig(nearby_vehicle_area_threshold=0.08)
        engine_custom = RiskEngine(config=custom_config)
        engine_default = RiskEngine()

        frame = make_nominal_frame()
        frame.objects = [
            ObjectFeature(
                track_id=1,
                category="car",
                confidence=0.9,
                box_x1=400.0,
                box_y1=400.0,
                box_x2=700.0,
                box_y2=600.0,
                box_center_x_px=550.0,
                box_center_y_px=500.0,
                box_area_fraction=0.10,
                is_in_lower_half=True,
                track_age=3,
                track_hits=3,
            )
        ]
        frame.object_counts = {"car": 1}

        res_default = engine_default.evaluate(frame).to_dict()
        res_custom = engine_custom.evaluate(frame).to_dict()

        # Default engine: area 0.10 < 0.12 -> NOT triggered
        assert res_default["assessment"]["active_indicators_count"] == 0

        # Custom engine: area 0.10 >= 0.08 -> TRIGGERED
        assert res_custom["assessment"]["active_indicators_count"] == 1
        assert res_custom["risk_indicators"][0]["indicator_id"] == IndicatorId.NEARBY_VEHICLE.value

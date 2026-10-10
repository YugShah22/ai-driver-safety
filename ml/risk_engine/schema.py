"""
Schema and data models for Phase 13 Risk Engine and Driving Insights.

Provides strongly-typed schemas separating:
1. Raw observations
2. Derived features
3. Risk indicators
4. Final assessment
5. Explanation
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

from .capabilities import DISCLAIMER_NOTICE, SCORE_MEANING_DOCUMENTATION


class Severity(str, Enum):
    """Severity levels for individual risk indicators."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def score_weight(self) -> float:
        """Deterministic score contribution of this severity."""
        weights = {
            Severity.LOW: 10.0,
            Severity.MEDIUM: 25.0,
            Severity.HIGH: 50.0,
            Severity.CRITICAL: 80.0,
        }
        return weights[self]


class AssessmentLevel(str, Enum):
    """Categorized summary driving risk assessment level."""
    NOMINAL = "NOMINAL"
    LOW_CAUTION = "LOW_CAUTION"
    ELEVATED_CAUTION = "ELEVATED_CAUTION"
    HIGH_RISK = "HIGH_RISK"
    CRITICAL_RISK = "CRITICAL_RISK"


class IndicatorId(str, Enum):
    """Documented indicator identifiers."""
    NEARBY_VEHICLE = "NEARBY_VEHICLE"
    VULNERABLE_ROAD_USER_PATH = "VULNERABLE_ROAD_USER_PATH"
    LANE_DEPARTURE = "LANE_DEPARTURE"
    PERSISTENT_OBSTRUCTION = "PERSISTENT_OBSTRUCTION"
    ADVERSE_ENVIRONMENT = "ADVERSE_ENVIRONMENT"
    SUDDEN_CUT_IN = "SUDDEN_CUT_IN"
    HIGH_TRAFFIC_DENSITY = "HIGH_TRAFFIC_DENSITY"


@dataclass
class Observation:
    """
    Layer 1: Raw Observation directly from upstream perception models.
    """
    observation_type: str        # e.g., "detected_object", "lane_geometry", "scene_classification"
    source: str                  # e.g., "yolo_detector", "sort_tracker", "unet_segmenter", "cnn_classifier"
    details: dict[str, Any]      # raw model detection facts

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_type": self.observation_type,
            "source": self.source,
            "details": self.details,
        }


@dataclass
class RiskIndicator:
    """
    Layer 3: Risk Indicator with an explicit, documented empirical basis.
    """
    indicator_id: str            # IndicatorId value
    name: str                    # Human-readable title
    severity: Severity           # LOW, MEDIUM, HIGH, CRITICAL
    triggered: bool              # True if threshold condition was met
    basis: str                   # Documented empirical/mathematical rationale
    source_data: dict[str, Any]  # The specific feature/observation evidence that triggered it

    def to_dict(self) -> dict[str, Any]:
        return {
            "indicator_id": self.indicator_id,
            "name": self.name,
            "severity": self.severity.value,
            "triggered": self.triggered,
            "basis": self.basis,
            "source_data": self.source_data,
        }


@dataclass
class Assessment:
    """
    Layer 4: Final synthesized driving assessment.
    """
    level: AssessmentLevel
    summary: str
    active_indicators_count: int
    highest_severity: Optional[Severity] = None
    composite_score: float = 0.0
    score_meaning: str = SCORE_MEANING_DOCUMENTATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "summary": self.summary,
            "active_indicators_count": self.active_indicators_count,
            "highest_severity": self.highest_severity.value if self.highest_severity else None,
            "composite_score": round(self.composite_score, 1),
            "score_meaning": self.score_meaning,
        }


@dataclass
class DrivingAssessmentResult:
    """
    Complete explainable driving assessment conforming to the 5-layer architecture.
    """
    observations: list[dict[str, Any]]
    features: dict[str, Any]
    risk_indicators: list[dict[str, Any]]
    assessment: dict[str, Any]
    explanations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Produce the standard serialized dictionary representation."""
        res: dict[str, Any] = {
            "observations": self.observations,
            "features": self.features,
            "risk_indicators": self.risk_indicators,
            "assessment": self.assessment,
            "explanations": self.explanations,
        }
        if self.metadata:
            res["metadata"] = self.metadata
        return res

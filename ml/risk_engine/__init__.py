"""
Risk Engine module — Driving risk assessment orchestrator.

Phase 13 implements:
- RiskEngine: Orchestrates feature input -> 5-layer assessment
- DrivingAssessmentResult: Explainable driving assessment output container
- RiskIndicator, Assessment, Observation: 5-layer component models
- RiskIndicatorConfig: Tunable empirical thresholds
- SystemCapabilities, CAN_INFER, CANNOT_INFER: Explicit operational capabilities & boundaries
- Severity, AssessmentLevel, IndicatorId: Typed enumerations
"""
from __future__ import annotations

from .capabilities import (
    CAN_INFER,
    CANNOT_INFER,
    DISCLAIMER_NOTICE,
    SCORE_MEANING_DOCUMENTATION,
    SystemCapabilities,
)
from .engine import RiskEngine
from .indicators import RiskIndicatorConfig
from .schema import (
    Assessment,
    AssessmentLevel,
    DrivingAssessmentResult,
    IndicatorId,
    Observation,
    RiskIndicator,
    Severity,
)

__all__ = [
    "CAN_INFER",
    "CANNOT_INFER",
    "DISCLAIMER_NOTICE",
    "SCORE_MEANING_DOCUMENTATION",
    "SystemCapabilities",
    "RiskEngine",
    "RiskIndicatorConfig",
    "Assessment",
    "AssessmentLevel",
    "DrivingAssessmentResult",
    "IndicatorId",
    "Observation",
    "RiskIndicator",
    "Severity",
]

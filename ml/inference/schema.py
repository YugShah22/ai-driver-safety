"""
Schema for the Phase 14 end-to-end pipeline report.

Defines the structured output of one full video analysis run, separating
video metadata, per-stage status, per-frame results, and aggregated summaries.

Every field is explicitly documented. No physical quantities
(distance, speed, TTC, collision probability) are claimed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class StageStatus(str, Enum):
    """Outcome of a single pipeline stage."""
    COMPLETED = "completed"      # Stage ran successfully.
    SKIPPED   = "skipped"        # Stage skipped because it is disabled in config.
    UNAVAILABLE = "unavailable"  # Stage required a checkpoint/mapping that is absent.
    FAILED    = "failed"         # Stage raised an unexpected exception.


@dataclass
class StageResult:
    """Status record for a single pipeline stage."""
    stage: str
    status: StageStatus
    reason: Optional[str] = None          # For UNAVAILABLE / FAILED / SKIPPED.
    extra: Optional[dict[str, Any]] = None  # Additional stage-specific metadata.

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"stage": self.stage, "status": self.status.value}
        if self.reason:
            d["reason"] = self.reason
        if self.extra:
            d["extra"] = self.extra
        return d


@dataclass
class FrameResult:
    """
    Consolidated analysis result for a single extracted frame.

    All spatial values are pixel-space proxies (not physical measurements).
    """
    frame_index: int
    frame_id: str
    timestamp_s: float

    # Per-frame stage outputs (None when stage did not run)
    scene_class: Optional[str] = None
    scene_class_id: Optional[int] = None
    scene_confidence: Optional[float] = None

    detections: list[dict[str, Any]] = field(default_factory=list)  # {category, confidence, box}
    tracks: list[dict[str, Any]] = field(default_factory=list)       # {track_id, category, box, age}
    lane_valid: Optional[bool] = None
    lane_status: Optional[str] = None
    lateral_offset_px: Optional[float] = None
    lane_departure: Optional[bool] = None

    risk_level: Optional[str] = None
    risk_score: Optional[float] = None          # 0–100 visual hazard index (NOT collision probability)
    risk_indicators: list[dict[str, Any]] = field(default_factory=list)

    traditional_ml_prediction: Optional[int] = None  # class index if model ran
    ann_prediction: Optional[int] = None              # class index if model ran

    error: Optional[str] = None  # Non-None if this frame failed to process

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "scene": {
                "class": self.scene_class,
                "class_id": self.scene_class_id,
                "confidence": self.scene_confidence,
            } if self.scene_class is not None else None,
            "detections": self.detections,
            "tracks": self.tracks,
            "lane": {
                "valid": self.lane_valid,
                "status": self.lane_status,
                "lateral_offset_px": self.lateral_offset_px,
                "departure": self.lane_departure,
            } if self.lane_valid is not None else None,
            "risk": {
                "level": self.risk_level,
                "score": self.risk_score,
                "indicators": self.risk_indicators,
            } if self.risk_level is not None else None,
            "ml_predictions": {
                "traditional_ml_class": self.traditional_ml_prediction,
                "ann_class": self.ann_prediction,
            } if (self.traditional_ml_prediction is not None or self.ann_prediction is not None) else None,
            "error": self.error,
        }


@dataclass
class VideoAnalysisReport:
    """
    Complete structured report from one end-to-end video analysis run.

    Limitations are explicitly included. No overall safety guarantee is produced.
    """

    # ── Video metadata ──────────────────────────────────────────────────────────
    video_path: str
    video_filename: str
    video_width: Optional[int]
    video_height: Optional[int]
    video_fps: Optional[float]
    video_frame_count: Optional[int]
    video_duration_s: Optional[float]
    video_codec: Optional[str]

    # ── Sampling configuration ──────────────────────────────────────────────────
    target_fps: float
    max_frames_config: int

    # ── Pipeline status ─────────────────────────────────────────────────────────
    pipeline_succeeded: bool
    pipeline_error: Optional[str]
    stage_results: list[StageResult] = field(default_factory=list)

    # ── Frame counts ────────────────────────────────────────────────────────────
    frames_extracted: int = 0
    frames_processed: int = 0
    frames_failed: int = 0

    # ── Per-frame results ───────────────────────────────────────────────────────
    frame_results: list[FrameResult] = field(default_factory=list)

    # ── Aggregated summary ──────────────────────────────────────────────────────
    summary: dict[str, Any] = field(default_factory=dict)

    # ── Documented limitations ──────────────────────────────────────────────────
    limitations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": {
                "path": self.video_path,
                "filename": self.video_filename,
                "width": self.video_width,
                "height": self.video_height,
                "fps": self.video_fps,
                "frame_count": self.video_frame_count,
                "duration_s": self.video_duration_s,
                "codec": self.video_codec,
            },
            "sampling": {
                "target_fps": self.target_fps,
                "max_frames_config": self.max_frames_config,
            },
            "pipeline": {
                "succeeded": self.pipeline_succeeded,
                "error": self.pipeline_error,
                "stages": [s.to_dict() for s in self.stage_results],
            },
            "frames": {
                "extracted": self.frames_extracted,
                "processed": self.frames_processed,
                "failed": self.frames_failed,
            },
            "frame_results": [f.to_dict() for f in self.frame_results],
            "summary": self.summary,
            "limitations": self.limitations,
            "warnings": self.warnings,
        }

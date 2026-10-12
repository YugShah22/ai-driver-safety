"""
ml/inference — End-to-end AI Driver Safety Pipeline (Phase 14).

Public API:
    AnalysisPipeline   — orchestrates all phases end-to-end
    PipelineConfig     — all configuration options
    VideoAnalysisReport — complete structured video analysis report
    StageResult        — per-stage status record
    FrameResult        — per-frame consolidated results
    StageStatus        — stage outcome enum
"""
from __future__ import annotations

from .config import PipelineConfig
from .pipeline import AnalysisPipeline
from .schema import (
    FrameResult,
    StageResult,
    StageStatus,
    VideoAnalysisReport,
)

__all__ = [
    "AnalysisPipeline",
    "PipelineConfig",
    "VideoAnalysisReport",
    "StageResult",
    "FrameResult",
    "StageStatus",
]

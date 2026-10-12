"""
End-to-End AI Driver Safety Pipeline — Phase 14.

Orchestrates existing Phase 5–13 components into a full video analysis pipeline.

Stage order:
  1. Validate video input and config.
  2. Extract sampled frames via VideoProcessor.
  3. CNN scene classification (optional — requires trained checkpoint).
  4. YOLO object detection (via ObjectDetector / yolov8n.pt).
  5. Object tracking (via IOUTracker).
  6. Segmentation + lane analysis (optional — requires verified class mapping).
  7. Feature extraction (via FeatureExtractor).
  8. Traditional ML prediction (optional — requires fitted RiskModel checkpoint).
  9. ANN prediction (optional — requires trained RiskANN checkpoint).
 10. Risk Engine assessment (Phase 13 deterministic engine — always runs).
 11. Aggregate per-frame results into a VideoAnalysisReport.

IMPORTANT HONESTY CONSTRAINTS (enforced in code, not just documentation):
 - No stage result from an untrained model is presented as meaningful driving-risk.
 - No physical quantities (metric distance, TTC, speed, collision probability) are
   computed or claimed from monocular camera data.
 - Missing checkpoints produce UNAVAILABLE status with explicit reasons.
 - Per-frame failures do NOT abort the remaining frames.
"""
from __future__ import annotations

import logging
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from ml.feature_extraction.extractor import FeatureExtractor
from ml.feature_extraction.schema import FrameFeatures
from ml.object_detection.detector import ObjectDetector
from ml.object_tracking.tracker import IOUTracker
from ml.risk_engine.capabilities import CANNOT_INFER, DISCLAIMER_NOTICE, SCORE_MEANING_DOCUMENTATION
from ml.risk_engine.engine import RiskEngine
from ml.risk_engine.indicators import RiskIndicatorConfig
from ml.segmentation.lane import LaneAnalyzer
from ml.traditional_ml.features import vectorize
from ml.traditional_ml.schema import FEATURE_NAMES, N_FEATURES
from ml.video_processing.exceptions import (
    CorruptVideoError,
    InvalidMetadataError,
    UnsupportedFormatError,
    VideoNotFoundError,
)
from ml.video_processing.models import ExtractedFrame
from ml.video_processing.processor import VideoProcessor

from .config import PipelineConfig
from .schema import (
    FrameResult,
    StageResult,
    StageStatus,
    VideoAnalysisReport,
)

logger = logging.getLogger(__name__)

# ─── Static limitations text ─────────────────────────────────────────────────

_BASE_LIMITATIONS: list[str] = [
    "All risk assessments are based on 2D forward-camera optical proxies from a monocular camera.",
    "Physical metric distances (meters), vehicle speed (km/h), and time-to-collision (seconds) "
    "are NOT measured or estimated. No calibrated 3D sensing is available.",
    "The composite visual risk score (0–100) is a deterministic heuristic index over active "
    "visual hazard indicators. It does NOT represent a calibrated collision probability.",
    "Traditional ML and ANN predictions are only meaningful relative to the quality of human "
    "annotation labels used to train those models. Untrained models are not applied.",
    "Segmentation-based lane analysis requires a verified class-ID-to-road-region mapping. "
    "Without it, lane analysis returns is_valid=False to prevent false reporting.",
    DISCLAIMER_NOTICE,
]


# ─── Pipeline implementation ─────────────────────────────────────────────────

class AnalysisPipeline:
    """
    End-to-end video analysis pipeline.

    Each public stage method is independently testable. The full pipeline is
    invoked via :meth:`run`.

    Args:
        config: PipelineConfig controlling all stages.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._risk_engine = RiskEngine(
            config=RiskIndicatorConfig()
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, video_path: str | Path) -> VideoAnalysisReport:
        """
        Analyse a local video file end-to-end.

        Args:
            video_path: Absolute or relative path to a supported video file.

        Returns:
            A fully populated :class:`VideoAnalysisReport`. Never raises —
            all exceptions are captured into the report.
        """
        video_path = Path(video_path)
        stage_results: list[StageResult] = []
        warnings: list[str] = []

        # ── Stage 1: Validate input ────────────────────────────────────────────
        validation_result, processor = self._stage_validate(video_path, stage_results)
        if not validation_result:
            return VideoAnalysisReport(
                video_path=str(video_path),
                video_filename=video_path.name,
                video_width=None,
                video_height=None,
                video_fps=None,
                video_frame_count=None,
                video_duration_s=None,
                video_codec=None,
                target_fps=self.config.processing_config.target_fps,
                max_frames_config=self.config.processing_config.max_frames,
                pipeline_succeeded=False,
                pipeline_error=stage_results[-1].reason,
                stage_results=stage_results,
                limitations=_BASE_LIMITATIONS,
                warnings=warnings,
            )

        metadata = processor.get_video_metadata()

        # ── Stage 2: Extract frames ────────────────────────────────────────────
        frames = self._stage_extract_frames(processor, metadata, stage_results)
        frames_extracted = len(frames)

        # Build base report skeleton
        report = VideoAnalysisReport(
            video_path=str(video_path),
            video_filename=metadata.filename,
            video_width=metadata.width,
            video_height=metadata.height,
            video_fps=metadata.fps,
            video_frame_count=metadata.frame_count,
            video_duration_s=metadata.duration,
            video_codec=metadata.codec,
            target_fps=self.config.processing_config.target_fps,
            max_frames_config=self.config.processing_config.max_frames,
            pipeline_succeeded=True,
            pipeline_error=None,
            stage_results=stage_results,
            frames_extracted=frames_extracted,
            limitations=_BASE_LIMITATIONS,
            warnings=warnings,
        )

        if not frames:
            report.pipeline_succeeded = False
            report.pipeline_error = "No frames could be extracted from the video."
            report.summary = self._build_summary([], stage_results)
            return report

        # ── Stages 3–10: Load models, process each frame ───────────────────────
        cnn_inferencer = self._load_cnn(stage_results, warnings)
        detector = self._load_detector(stage_results, warnings)
        tracker = self._create_tracker()
        seg_model, lane_analyzer = self._load_segmentation(stage_results, warnings)
        trad_model = self._load_traditional_ml(stage_results, warnings)
        ann_model = self._load_ann(stage_results, warnings)

        feature_extractor = FeatureExtractor()
        frame_results: list[FrameResult] = []

        for idx, frame in enumerate(frames):
            fr = self._process_frame(
                idx=idx,
                frame=frame,
                cnn_inferencer=cnn_inferencer,
                detector=detector,
                tracker=tracker,
                seg_model=seg_model,
                lane_analyzer=lane_analyzer,
                feature_extractor=feature_extractor,
                trad_model=trad_model,
                ann_model=ann_model,
            )
            frame_results.append(fr)

        # ── Stage 11: Aggregate ────────────────────────────────────────────────
        processed = sum(1 for f in frame_results if f.error is None)
        failed = sum(1 for f in frame_results if f.error is not None)

        report.frames_processed = processed
        report.frames_failed = failed
        report.frame_results = frame_results
        report.summary = self._build_summary(frame_results, stage_results)
        stage_results.append(StageResult(
            stage="aggregation",
            status=StageStatus.COMPLETED,
            extra={"frames_processed": processed, "frames_failed": failed},
        ))

        logger.info(
            "[AnalysisPipeline] Completed: %d frames processed, %d failed.",
            processed, failed,
        )
        return report

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1: Input validation
    # ─────────────────────────────────────────────────────────────────────────

    def _stage_validate(
        self,
        video_path: Path,
        stage_results: list[StageResult],
    ) -> tuple[bool, Optional[VideoProcessor]]:
        """Return (ok, processor). Appends StageResult."""
        try:
            processor = VideoProcessor(video_path, self.config.processing_config)
            processor.get_video_metadata()  # validates early
            stage_results.append(StageResult(
                stage="input_validation",
                status=StageStatus.COMPLETED,
                extra={"video_path": str(video_path)},
            ))
            # Re-create processor for caller to use
            processor = VideoProcessor(video_path, self.config.processing_config)
            return True, processor
        except (VideoNotFoundError, CorruptVideoError, UnsupportedFormatError, InvalidMetadataError) as exc:
            stage_results.append(StageResult(
                stage="input_validation",
                status=StageStatus.FAILED,
                reason=str(exc),
            ))
            logger.error("[AnalysisPipeline] Input validation failed: %s", exc)
            return False, None
        except Exception as exc:
            stage_results.append(StageResult(
                stage="input_validation",
                status=StageStatus.FAILED,
                reason=f"Unexpected error during validation: {exc}",
            ))
            logger.exception("[AnalysisPipeline] Unexpected validation error: %s", exc)
            return False, None

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2: Frame extraction
    # ─────────────────────────────────────────────────────────────────────────

    def _stage_extract_frames(
        self,
        processor: VideoProcessor,
        metadata,
        stage_results: list[StageResult],
    ) -> list[ExtractedFrame]:
        """Extract and materialise frames from the video generator."""
        try:
            frames: list[ExtractedFrame] = []
            corrupt_count = 0

            for frame in processor.extract_frames(metadata):
                # Guard against corrupt frames (null or zero-area data)
                if frame.data is None or frame.width <= 0 or frame.height <= 0:
                    corrupt_count += 1
                    logger.warning(
                        "[AnalysisPipeline] Skipping corrupt frame #%d", frame.frame_number
                    )
                    continue
                frames.append(frame)

            extra: dict[str, Any] = {"frames_extracted": len(frames)}
            if corrupt_count:
                extra["corrupt_frames_skipped"] = corrupt_count

            stage_results.append(StageResult(
                stage="frame_extraction",
                status=StageStatus.COMPLETED,
                extra=extra,
            ))
            logger.info("[AnalysisPipeline] Extracted %d frames.", len(frames))
            return frames

        except Exception as exc:
            stage_results.append(StageResult(
                stage="frame_extraction",
                status=StageStatus.FAILED,
                reason=str(exc),
            ))
            logger.exception("[AnalysisPipeline] Frame extraction failed: %s", exc)
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3: CNN scene classification (conditional)
    # ─────────────────────────────────────────────────────────────────────────

    def _load_cnn(
        self,
        stage_results: list[StageResult],
        warnings: list[str],
    ) -> Optional[Any]:
        """Load the CNN Inferencer if a valid checkpoint is configured."""
        cfg = self.config
        if not cfg.cnn_checkpoint:
            stage_results.append(StageResult(
                stage="cnn_scene_classification",
                status=StageStatus.UNAVAILABLE,
                reason="No CNN checkpoint configured (cnn_checkpoint is None).",
            ))
            return None

        ckpt_path = Path(cfg.cnn_checkpoint)
        if not ckpt_path.is_file():
            stage_results.append(StageResult(
                stage="cnn_scene_classification",
                status=StageStatus.UNAVAILABLE,
                reason=f"CNN checkpoint not found: {ckpt_path}. "
                       "Train a CNN model first (Phase 5 or Phase 6).",
            ))
            warnings.append(
                f"CNN scene classification unavailable: checkpoint not found at '{ckpt_path}'."
            )
            return None

        try:
            from ml.cnn.config import CNNConfig
            from ml.cnn.infer import Inferencer

            cnn_config = CNNConfig(
                num_classes=cfg.cnn_num_classes,
                image_size=cfg.cnn_image_size,
                device=cfg.cnn_device,
            )
            inferencer = Inferencer.from_checkpoint(
                checkpoint_path=ckpt_path,
                config=cnn_config,
                class_names=cfg.cnn_class_names,
            )
            stage_results.append(StageResult(
                stage="cnn_scene_classification",
                status=StageStatus.COMPLETED,
                extra={
                    "checkpoint": str(ckpt_path),
                    "num_classes": cfg.cnn_num_classes,
                    "class_names": cfg.cnn_class_names,
                },
            ))
            logger.info("[AnalysisPipeline] CNN loaded from %s", ckpt_path)
            return inferencer

        except Exception as exc:
            stage_results.append(StageResult(
                stage="cnn_scene_classification",
                status=StageStatus.FAILED,
                reason=f"Failed to load CNN checkpoint: {exc}",
            ))
            warnings.append(f"CNN scene classification will be skipped: {exc}")
            logger.warning("[AnalysisPipeline] CNN load failed: %s", exc)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 4: YOLO object detection
    # ─────────────────────────────────────────────────────────────────────────

    def _load_detector(
        self,
        stage_results: list[StageResult],
        warnings: list[str],
    ) -> Optional[ObjectDetector]:
        try:
            detector = ObjectDetector(
                model_path=self.config.yolo_model_path,
                conf_threshold=self.config.yolo_conf_threshold,
                device=self.config.yolo_device,
            )
            stage_results.append(StageResult(
                stage="object_detection",
                status=StageStatus.COMPLETED,
                extra={
                    "model": self.config.yolo_model_path,
                    "conf_threshold": self.config.yolo_conf_threshold,
                },
            ))
            return detector
        except Exception as exc:
            stage_results.append(StageResult(
                stage="object_detection",
                status=StageStatus.FAILED,
                reason=str(exc),
            ))
            warnings.append(f"YOLO object detection unavailable: {exc}")
            logger.warning("[AnalysisPipeline] YOLO load failed: %s", exc)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 5: Object tracking
    # ─────────────────────────────────────────────────────────────────────────

    def _create_tracker(self) -> IOUTracker:
        return IOUTracker(
            iou_threshold=self.config.tracker_iou_threshold,
            max_misses=self.config.tracker_max_misses,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 6: Segmentation + lane analysis (conditional)
    # ─────────────────────────────────────────────────────────────────────────

    def _load_segmentation(
        self,
        stage_results: list[StageResult],
        warnings: list[str],
    ) -> tuple[Optional[Any], Optional[LaneAnalyzer]]:
        cfg = self.config

        if not cfg.segmentation_enabled:
            stage_results.append(StageResult(
                stage="segmentation_lane",
                status=StageStatus.SKIPPED,
                reason="Segmentation disabled in config (segmentation_enabled=False).",
            ))
            return None, None

        try:
            from ml.segmentation.model import SegmentationModel

            seg_model = SegmentationModel(
                num_classes=cfg.segmentation_num_classes,
                pretrained=cfg.segmentation_pretrained,
                device=cfg.segmentation_device,
            )

            lane_analyzer = LaneAnalyzer(
                road_class_ids=cfg.segmentation_road_class_ids,
                departure_threshold_px=cfg.lane_departure_threshold_px,
            )

            road_mapping_note = (
                "Verified road class mapping provided."
                if cfg.segmentation_road_class_ids
                else "No road_class_ids configured — lane analysis will return is_valid=False. "
                     "This prevents false reporting from unverified class mappings."
            )

            stage_results.append(StageResult(
                stage="segmentation_lane",
                status=StageStatus.COMPLETED,
                extra={
                    "num_classes": cfg.segmentation_num_classes,
                    "pretrained": cfg.segmentation_pretrained,
                    "road_class_ids": (
                        list(cfg.segmentation_road_class_ids)
                        if cfg.segmentation_road_class_ids
                        else None
                    ),
                    "road_mapping_note": road_mapping_note,
                },
            ))

            if not cfg.segmentation_road_class_ids:
                warnings.append(
                    "Segmentation model is loaded, but no road_class_ids are configured. "
                    "Lane analysis will run but return is_valid=False for all frames. "
                    "Provide verified class IDs to enable lane geometry analysis."
                )

            return seg_model, lane_analyzer

        except Exception as exc:
            stage_results.append(StageResult(
                stage="segmentation_lane",
                status=StageStatus.FAILED,
                reason=str(exc),
            ))
            warnings.append(f"Segmentation unavailable: {exc}")
            logger.warning("[AnalysisPipeline] Segmentation load failed: %s", exc)
            return None, None

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 8: Traditional ML (conditional)
    # ─────────────────────────────────────────────────────────────────────────

    def _load_traditional_ml(
        self,
        stage_results: list[StageResult],
        warnings: list[str],
    ) -> Optional[Any]:
        cfg = self.config
        if not cfg.traditional_ml_checkpoint:
            stage_results.append(StageResult(
                stage="traditional_ml",
                status=StageStatus.UNAVAILABLE,
                reason="No traditional ML checkpoint configured (traditional_ml_checkpoint is None).",
            ))
            return None

        ckpt_path = Path(cfg.traditional_ml_checkpoint)
        if not ckpt_path.is_file():
            stage_results.append(StageResult(
                stage="traditional_ml",
                status=StageStatus.UNAVAILABLE,
                reason=f"Traditional ML checkpoint not found: {ckpt_path}. "
                       "Train a RiskModel first (Phase 11).",
            ))
            warnings.append(
                f"Traditional ML predictions unavailable: checkpoint not found at '{ckpt_path}'."
            )
            return None

        try:
            from ml.traditional_ml.model import RiskModel
            model = RiskModel.load(ckpt_path)
            stage_results.append(StageResult(
                stage="traditional_ml",
                status=StageStatus.COMPLETED,
                extra={"checkpoint": str(ckpt_path), "kind": model.kind},
            ))
            logger.info("[AnalysisPipeline] Traditional ML loaded from %s", ckpt_path)
            return model
        except Exception as exc:
            stage_results.append(StageResult(
                stage="traditional_ml",
                status=StageStatus.FAILED,
                reason=f"Failed to load RiskModel: {exc}",
            ))
            warnings.append(f"Traditional ML predictions will be skipped: {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 9: ANN (conditional)
    # ─────────────────────────────────────────────────────────────────────────

    def _load_ann(
        self,
        stage_results: list[StageResult],
        warnings: list[str],
    ) -> Optional[Any]:
        cfg = self.config
        if not cfg.ann_checkpoint:
            stage_results.append(StageResult(
                stage="ann",
                status=StageStatus.UNAVAILABLE,
                reason="No ANN checkpoint configured (ann_checkpoint is None).",
            ))
            return None

        ckpt_path = Path(cfg.ann_checkpoint)
        if not ckpt_path.is_file():
            stage_results.append(StageResult(
                stage="ann",
                status=StageStatus.UNAVAILABLE,
                reason=f"ANN checkpoint not found: {ckpt_path}. "
                       "Train a RiskANN model first (Phase 12).",
            ))
            warnings.append(f"ANN predictions unavailable: checkpoint not found at '{ckpt_path}'.")
            return None

        try:
            import torch
            from ml.ann.model import RiskANN
            ann_model = RiskANN.load(ckpt_path)
            stage_results.append(StageResult(
                stage="ann",
                status=StageStatus.COMPLETED,
                extra={"checkpoint": str(ckpt_path)},
            ))
            logger.info("[AnalysisPipeline] ANN loaded from %s", ckpt_path)
            return ann_model
        except Exception as exc:
            stage_results.append(StageResult(
                stage="ann",
                status=StageStatus.FAILED,
                reason=f"Failed to load RiskANN: {exc}",
            ))
            warnings.append(f"ANN predictions will be skipped: {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Per-frame processing (Stages 3–10 applied to one frame)
    # ─────────────────────────────────────────────────────────────────────────

    def _process_frame(
        self,
        idx: int,
        frame: ExtractedFrame,
        cnn_inferencer: Optional[Any],
        detector: Optional[ObjectDetector],
        tracker: IOUTracker,
        seg_model: Optional[Any],
        lane_analyzer: Optional[LaneAnalyzer],
        feature_extractor: FeatureExtractor,
        trad_model: Optional[Any],
        ann_model: Optional[Any],
    ) -> FrameResult:
        frame_id = f"frame_{frame.frame_number:06d}"

        fr = FrameResult(
            frame_index=idx,
            frame_id=frame_id,
            timestamp_s=frame.timestamp,
        )

        try:
            # Convert BGR numpy frame to PIL RGB for models that need PIL
            import cv2
            bgr = frame.data
            rgb_array = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_array)

            # ── Stage 3: CNN scene classification ─────────────────────────────
            scene_class: Optional[str] = None
            scene_class_id: Optional[int] = None

            if cnn_inferencer is not None:
                try:
                    result = cnn_inferencer.predict(pil_image)
                    scene_class = result.predicted_label
                    scene_class_id = result.predicted_class
                    fr.scene_class = scene_class
                    fr.scene_class_id = scene_class_id
                    fr.scene_confidence = round(result.confidence, 4)
                except Exception as exc:
                    logger.warning("[AnalysisPipeline] CNN failed on frame %s: %s", frame_id, exc)

            # ── Stage 4: YOLO detection ────────────────────────────────────────
            from ml.object_detection.schema import FrameDetections
            frame_detections = FrameDetections(
                frame_id=frame_id,
                detections=[],
                image_width=frame.width,
                image_height=frame.height,
            )

            if detector is not None:
                try:
                    frame_detections = detector.detect(pil_image, frame_id=frame_id)
                    fr.detections = [
                        {
                            "category": d.category,
                            "confidence": round(d.confidence, 4),
                            "box": [round(d.box.x1, 1), round(d.box.y1, 1),
                                    round(d.box.x2, 1), round(d.box.y2, 1)],
                            "track_id": d.track_id,
                        }
                        for d in frame_detections.detections
                    ]
                except Exception as exc:
                    logger.warning("[AnalysisPipeline] YOLO failed on frame %s: %s", frame_id, exc)

            # ── Stage 5: Tracking ──────────────────────────────────────────────
            active_tracks = tracker.update(frame_detections, frame_index=idx)
            fr.tracks = [
                {
                    "track_id": t.track_id,
                    "category": t.category,
                    "confidence": round(t.confidence, 4),
                    "box": [round(t.box.x1, 1), round(t.box.y1, 1),
                            round(t.box.x2, 1), round(t.box.y2, 1)],
                    "age": t.age,
                    "hits": t.hits,
                }
                for t in active_tracks
            ]

            # ── Stage 6: Segmentation + lane ──────────────────────────────────
            lane_result = None
            if seg_model is not None and lane_analyzer is not None:
                try:
                    seg_result = seg_model.segment(pil_image, frame_id=frame_id)
                    lane_result = lane_analyzer.analyze(seg_result)
                    fr.lane_valid = lane_result.is_valid
                    fr.lane_status = lane_result.status
                    fr.lateral_offset_px = lane_result.lateral_offset_px
                    fr.lane_departure = lane_result.lane_departure_indicator
                except Exception as exc:
                    logger.warning(
                        "[AnalysisPipeline] Segmentation failed on frame %s: %s", frame_id, exc
                    )

            # ── Stage 7: Feature extraction ────────────────────────────────────
            features: FrameFeatures = feature_extractor.extract(
                frame_id=frame_id,
                frame_index=idx,
                timestamp_s=frame.timestamp,
                image_width=frame.width,
                image_height=frame.height,
                scene_class=scene_class,
                scene_class_id=scene_class_id,
                active_tracks=active_tracks if active_tracks else None,
                frame_detections=frame_detections if not active_tracks else None,
                lane_result=lane_result,
            )

            # ── Stage 8: Traditional ML prediction (conditional) ───────────────
            if trad_model is not None:
                try:
                    vec = vectorize(features).reshape(1, -1)
                    pred = trad_model.predict(vec)
                    fr.traditional_ml_prediction = int(pred[0])
                except Exception as exc:
                    logger.warning(
                        "[AnalysisPipeline] Traditional ML failed on frame %s: %s", frame_id, exc
                    )

            # ── Stage 9: ANN prediction (conditional) ──────────────────────────
            if ann_model is not None:
                try:
                    import torch
                    vec_t = torch.from_numpy(
                        vectorize(features).reshape(1, -1)
                    ).float()
                    with torch.no_grad():
                        ann_pred = ann_model.predict(vec_t)
                    fr.ann_prediction = int(ann_pred.item())
                except Exception as exc:
                    logger.warning(
                        "[AnalysisPipeline] ANN failed on frame %s: %s", frame_id, exc
                    )

            # ── Stage 10: Risk Engine assessment (always runs) ─────────────────
            risk_result = self._risk_engine.evaluate(features)
            risk_data = risk_result.to_dict()
            assessment = risk_data.get("assessment", {})
            fr.risk_level = assessment.get("level")
            fr.risk_score = assessment.get("composite_score")
            fr.risk_indicators = risk_data.get("risk_indicators", [])

        except Exception as exc:
            fr.error = f"Frame {frame_id} processing failed: {traceback.format_exc()}"
            logger.error("[AnalysisPipeline] Unexpected frame error: %s", exc)

        return fr

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 11: Aggregation
    # ─────────────────────────────────────────────────────────────────────────

    def _build_summary(
        self,
        frame_results: list[FrameResult],
        stage_results: list[StageResult],
    ) -> dict[str, Any]:
        """Build deterministic summary statistics from per-frame results."""
        if not frame_results:
            return {
                "total_frames": 0,
                "processed_frames": 0,
                "failed_frames": 0,
                "risk_level_counts": {},
                "indicator_counts": {},
                "scene_class_counts": {},
                "lane_departures_detected": 0,
                "stages": {s.stage: s.status.value for s in stage_results},
                "score_note": SCORE_MEANING_DOCUMENTATION,
            }

        risk_level_counts: Counter = Counter()
        indicator_counts: Counter = Counter()
        scene_counts: Counter = Counter()
        departures = 0

        for fr in frame_results:
            if fr.error:
                continue
            if fr.risk_level:
                risk_level_counts[fr.risk_level] += 1
            for ind in fr.risk_indicators:
                if ind.get("triggered"):
                    indicator_counts[ind.get("name", ind.get("indicator_id", "unknown"))] += 1
            if fr.scene_class:
                scene_counts[fr.scene_class] += 1
            if fr.lane_departure:
                departures += 1

        return {
            "total_frames": len(frame_results),
            "processed_frames": sum(1 for f in frame_results if f.error is None),
            "failed_frames": sum(1 for f in frame_results if f.error is not None),
            "risk_level_counts": dict(risk_level_counts),
            "indicator_counts": dict(indicator_counts),
            "scene_class_counts": dict(scene_counts),
            "lane_departures_detected": departures,
            "stages": {s.stage: s.status.value for s in stage_results},
            "score_note": SCORE_MEANING_DOCUMENTATION,
        }

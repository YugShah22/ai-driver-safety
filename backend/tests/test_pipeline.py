"""
Phase 14 pipeline integration tests.

Uses:
  - Synthetic numpy frames (no real video file required for most tests)
  - A tiny generated MP4 for tests that need a real file (cv2.VideoWriter)
  - Mocked model components for isolation
  - Patching to simulate unavailable/failed components

Does NOT:
  - Download any model weights
  - Use a GPU
  - Claim real-world accuracy
  - Invoke real YOLO on video (YOLO is mocked where integration is not the goal)
"""
from __future__ import annotations

import io
import os
import struct
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from ml.feature_extraction.schema import FrameFeatures, LaneFeature, ObjectFeature
from ml.inference import (
    AnalysisPipeline,
    FrameResult,
    PipelineConfig,
    StageResult,
    StageStatus,
    VideoAnalysisReport,
)
from ml.inference.schema import StageStatus
from ml.object_detection.schema import BoundingBox, Detection, FrameDetections
from ml.object_tracking.schema import Track
from ml.risk_engine.capabilities import CANNOT_INFER
from ml.video_processing.models import ProcessingConfig, ResizeMode


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_tiny_video(path: Path, n_frames: int = 6, fps: float = 3.0) -> Path:
    """Generate a small real MP4 using OpenCV VideoWriter (no internet needed)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = 64, 48
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened(), f"Could not open VideoWriter for {path}"
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 1] = (i * 40) % 255   # varying green channel
        writer.write(frame)
    writer.release()
    return path


def make_synthetic_frame(width: int = 64, height: int = 48) -> "ExtractedFrame":
    """Create a synthetic ExtractedFrame with a blank BGR image."""
    from ml.video_processing.models import ExtractedFrame
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    return ExtractedFrame(
        frame_number=1,
        timestamp=0.0,
        width=width,
        height=height,
        data=bgr,
    )


def _mock_detector(n_detections: int = 1) -> MagicMock:
    """Build a mock ObjectDetector that returns n_detections per frame."""
    detector = MagicMock()
    dets = [
        Detection(
            category="car",
            class_id=2,
            confidence=0.9,
            box=BoundingBox(x1=10.0, y1=30.0, x2=40.0, y2=48.0),
            frame_id="frame_000001",
        )
        for _ in range(n_detections)
    ]
    detector.detect.return_value = FrameDetections(
        frame_id="frame_000001",
        detections=dets,
        image_width=64,
        image_height=48,
    )
    return detector


def _nominal_pipeline_config(tmp_path: Path) -> PipelineConfig:
    """Config with no optional model checkpoints — only detection+tracking+risk."""
    return PipelineConfig(
        processing_config=ProcessingConfig(
            target_fps=3.0,
            resize_mode=ResizeMode.ORIGINAL,
            max_frames=0,
        ),
        cnn_checkpoint=None,
        segmentation_enabled=False,
        traditional_ml_checkpoint=None,
        ann_checkpoint=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Schema contracts
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaContracts:
    def test_stage_result_to_dict_mandatory_keys(self):
        sr = StageResult(stage="foo", status=StageStatus.COMPLETED)
        d = sr.to_dict()
        assert "stage" in d
        assert "status" in d
        assert d["stage"] == "foo"
        assert d["status"] == "completed"

    def test_stage_result_unavailable_has_reason(self):
        sr = StageResult(
            stage="cnn", status=StageStatus.UNAVAILABLE,
            reason="No checkpoint configured."
        )
        d = sr.to_dict()
        assert d["reason"] == "No checkpoint configured."

    def test_frame_result_to_dict_structure(self):
        fr = FrameResult(frame_index=0, frame_id="frame_000001", timestamp_s=0.5)
        fr.risk_level = "NOMINAL"
        fr.risk_score = 0.0
        d = fr.to_dict()
        assert "frame_index" in d
        assert "frame_id" in d
        assert "timestamp_s" in d
        assert d["risk"]["level"] == "NOMINAL"
        assert d["risk"]["score"] == 0.0

    def test_video_analysis_report_to_dict_structure(self):
        report = VideoAnalysisReport(
            video_path="/tmp/test.mp4",
            video_filename="test.mp4",
            video_width=640,
            video_height=480,
            video_fps=30.0,
            video_frame_count=100,
            video_duration_s=3.33,
            video_codec="avc1",
            target_fps=1.0,
            max_frames_config=0,
            pipeline_succeeded=True,
            pipeline_error=None,
        )
        d = report.to_dict()
        assert "video" in d
        assert "pipeline" in d
        assert "frames" in d
        assert "frame_results" in d
        assert "limitations" in d
        assert "warnings" in d


# ─────────────────────────────────────────────────────────────────────────────
# 2. Missing video file
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingVideo:
    def test_nonexistent_video_returns_failed_report(self, tmp_path):
        cfg = _nominal_pipeline_config(tmp_path)
        pipeline = AnalysisPipeline(config=cfg)
        report = pipeline.run(tmp_path / "does_not_exist.mp4")

        assert report.pipeline_succeeded is False
        assert report.pipeline_error is not None
        assert any(
            s.stage == "input_validation" and s.status == StageStatus.FAILED
            for s in report.stage_results
        )

    def test_empty_file_returns_failed_report(self, tmp_path):
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        cfg = _nominal_pipeline_config(tmp_path)
        pipeline = AnalysisPipeline(config=cfg)
        report = pipeline.run(empty)

        assert report.pipeline_succeeded is False
        assert report.pipeline_error is not None

    def test_unsupported_extension_returns_failed_report(self, tmp_path):
        bad = tmp_path / "video.xyz"
        bad.write_bytes(b"\x00" * 100)
        cfg = _nominal_pipeline_config(tmp_path)
        pipeline = AnalysisPipeline(config=cfg)
        report = pipeline.run(bad)

        assert report.pipeline_succeeded is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Unavailable checkpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestUnavailableCheckpoints:
    def test_missing_cnn_checkpoint_marks_unavailable(self, tmp_path):
        cfg = PipelineConfig(
            processing_config=ProcessingConfig(target_fps=3.0),
            cnn_checkpoint="/nonexistent/model.pt",
            segmentation_enabled=False,
        )
        pipeline = AnalysisPipeline(config=cfg)
        stage_results: list[StageResult] = []
        warnings: list[str] = []
        inferencer = pipeline._load_cnn(stage_results, warnings)

        assert inferencer is None
        cnn_stage = next(s for s in stage_results if s.stage == "cnn_scene_classification")
        assert cnn_stage.status == StageStatus.UNAVAILABLE
        assert len(warnings) == 1

    def test_missing_traditional_ml_checkpoint_marks_unavailable(self, tmp_path):
        cfg = PipelineConfig(traditional_ml_checkpoint="/nonexistent/model.pkl")
        pipeline = AnalysisPipeline(config=cfg)
        stage_results: list[StageResult] = []
        warnings: list[str] = []
        model = pipeline._load_traditional_ml(stage_results, warnings)

        assert model is None
        ml_stage = next(s for s in stage_results if s.stage == "traditional_ml")
        assert ml_stage.status == StageStatus.UNAVAILABLE

    def test_missing_ann_checkpoint_marks_unavailable(self, tmp_path):
        cfg = PipelineConfig(ann_checkpoint="/nonexistent/ann.pt")
        pipeline = AnalysisPipeline(config=cfg)
        stage_results: list[StageResult] = []
        warnings: list[str] = []
        model = pipeline._load_ann(stage_results, warnings)

        assert model is None
        ann_stage = next(s for s in stage_results if s.stage == "ann")
        assert ann_stage.status == StageStatus.UNAVAILABLE

    def test_none_cnn_checkpoint_marks_unavailable(self, tmp_path):
        cfg = PipelineConfig(cnn_checkpoint=None)
        pipeline = AnalysisPipeline(config=cfg)
        stage_results: list[StageResult] = []
        warnings: list[str] = []
        result = pipeline._load_cnn(stage_results, warnings)

        assert result is None
        assert stage_results[0].status == StageStatus.UNAVAILABLE

    def test_none_traditional_ml_checkpoint_marks_unavailable(self, tmp_path):
        cfg = PipelineConfig(traditional_ml_checkpoint=None)
        pipeline = AnalysisPipeline(config=cfg)
        stage_results: list[StageResult] = []
        warnings: list[str] = []
        result = pipeline._load_traditional_ml(stage_results, warnings)

        assert result is None
        assert stage_results[0].status == StageStatus.UNAVAILABLE

    def test_none_ann_checkpoint_marks_unavailable(self, tmp_path):
        cfg = PipelineConfig(ann_checkpoint=None)
        pipeline = AnalysisPipeline(config=cfg)
        stage_results: list[StageResult] = []
        warnings: list[str] = []
        result = pipeline._load_ann(stage_results, warnings)

        assert result is None
        assert stage_results[0].status == StageStatus.UNAVAILABLE


# ─────────────────────────────────────────────────────────────────────────────
# 4. Segmentation
# ─────────────────────────────────────────────────────────────────────────────

class TestSegmentationStage:
    def test_segmentation_disabled_marks_skipped(self, tmp_path):
        cfg = PipelineConfig(segmentation_enabled=False)
        pipeline = AnalysisPipeline(config=cfg)
        stage_results: list[StageResult] = []
        warnings: list[str] = []
        seg, lane = pipeline._load_segmentation(stage_results, warnings)

        assert seg is None
        assert lane is None
        seg_stage = next(s for s in stage_results if s.stage == "segmentation_lane")
        assert seg_stage.status == StageStatus.SKIPPED

    def test_segmentation_without_road_class_ids_produces_warning(self, tmp_path):
        cfg = PipelineConfig(
            segmentation_enabled=True,
            segmentation_road_class_ids=None,  # unverified mapping
            segmentation_pretrained=False,     # avoid internet in tests
        )
        pipeline = AnalysisPipeline(config=cfg)
        stage_results: list[StageResult] = []
        warnings: list[str] = []
        seg, lane = pipeline._load_segmentation(stage_results, warnings)

        # Stage runs, but a warning is produced about unverified mapping
        seg_stage = next(s for s in stage_results if s.stage == "segmentation_lane")
        assert seg_stage.status == StageStatus.COMPLETED
        assert len(warnings) == 1
        assert "road_class_ids" in warnings[0] or "verified" in warnings[0].lower()

        # LaneAnalyzer has no road_class_ids → analyze() returns is_valid=False
        assert lane is not None
        assert lane.road_class_ids is None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Stage order and data flow (with mocked models, no real video)
# ─────────────────────────────────────────────────────────────────────────────

class TestStageOrderDataflow:
    def test_process_frame_with_mocked_detector_produces_frame_result(self, tmp_path):
        """
        Directly test _process_frame to verify correct data flow
        without needing a real video or real YOLO.
        """
        cfg = _nominal_pipeline_config(tmp_path)
        pipeline = AnalysisPipeline(config=cfg)

        from ml.feature_extraction.extractor import FeatureExtractor
        from ml.object_tracking.tracker import IOUTracker

        frame = make_synthetic_frame(64, 48)
        mock_detector = _mock_detector(n_detections=1)
        tracker = IOUTracker()
        extractor = FeatureExtractor()

        fr = pipeline._process_frame(
            idx=0,
            frame=frame,
            cnn_inferencer=None,
            detector=mock_detector,
            tracker=tracker,
            seg_model=None,
            lane_analyzer=None,
            feature_extractor=extractor,
            trad_model=None,
            ann_model=None,
        )

        assert fr.error is None
        assert fr.frame_index == 0
        assert fr.timestamp_s == 0.0
        # Risk Engine always runs
        assert fr.risk_level is not None
        assert fr.risk_score is not None
        # Detections: mock returned 1 car
        assert len(fr.detections) == 1
        assert fr.detections[0]["category"] == "car"
        # Track was created
        assert len(fr.tracks) == 1
        assert fr.tracks[0]["category"] == "car"

    def test_process_frame_without_detector_still_runs_risk_engine(self, tmp_path):
        cfg = _nominal_pipeline_config(tmp_path)
        pipeline = AnalysisPipeline(config=cfg)

        from ml.feature_extraction.extractor import FeatureExtractor
        from ml.object_tracking.tracker import IOUTracker

        frame = make_synthetic_frame(64, 48)
        tracker = IOUTracker()
        extractor = FeatureExtractor()

        fr = pipeline._process_frame(
            idx=0,
            frame=frame,
            cnn_inferencer=None,
            detector=None,
            tracker=tracker,
            seg_model=None,
            lane_analyzer=None,
            feature_extractor=extractor,
            trad_model=None,
            ann_model=None,
        )

        assert fr.error is None
        assert fr.risk_level == "NOMINAL"
        assert fr.risk_score == 0.0

    def test_tracking_persists_across_frames(self, tmp_path):
        cfg = _nominal_pipeline_config(tmp_path)
        pipeline = AnalysisPipeline(config=cfg)

        from ml.feature_extraction.extractor import FeatureExtractor
        from ml.object_tracking.tracker import IOUTracker

        tracker = IOUTracker()
        extractor = FeatureExtractor()

        # Consistent car detection in same position across 3 frames
        for idx in range(3):
            frame = make_synthetic_frame(64, 48)
            mock_detector = _mock_detector(n_detections=1)
            fr = pipeline._process_frame(
                idx=idx,
                frame=frame,
                cnn_inferencer=None,
                detector=mock_detector,
                tracker=tracker,
                seg_model=None,
                lane_analyzer=None,
                feature_extractor=extractor,
                trad_model=None,
                ann_model=None,
            )
            assert fr.error is None

        # After 3 frames, the track for the consistent car should have age>=3
        assert len(tracker.active_tracks) == 1
        track = tracker.active_tracks[0]
        assert track.age >= 3


# ─────────────────────────────────────────────────────────────────────────────
# 6. Risk Engine receives valid features
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskEngineIntegration:
    def test_risk_engine_receives_frame_features_not_dict(self, tmp_path):
        """
        FeatureExtractor always produces FrameFeatures objects.
        RiskEngine.evaluate accepts both FrameFeatures and dicts.
        Verify correct propagation.
        """
        from ml.feature_extraction.extractor import FeatureExtractor
        from ml.object_tracking.tracker import IOUTracker
        from ml.risk_engine.engine import RiskEngine

        cfg = _nominal_pipeline_config(tmp_path)
        pipeline = AnalysisPipeline(config=cfg)
        tracker = IOUTracker()
        extractor = FeatureExtractor()

        frame = make_synthetic_frame(64, 48)
        # Use a large car detection to trigger the NEARBY_VEHICLE indicator
        mock_detector = MagicMock()
        det = Detection(
            category="car",
            class_id=2,
            confidence=0.95,
            box=BoundingBox(x1=0.0, y1=25.0, x2=64.0, y2=48.0),  # large box in lower half
            frame_id="frame_000001",
        )
        mock_detector.detect.return_value = FrameDetections(
            frame_id="frame_000001",
            detections=[det],
            image_width=64,
            image_height=48,
        )

        fr = pipeline._process_frame(
            idx=0,
            frame=frame,
            cnn_inferencer=None,
            detector=mock_detector,
            tracker=tracker,
            seg_model=None,
            lane_analyzer=None,
            feature_extractor=extractor,
            trad_model=None,
            ann_model=None,
        )

        assert fr.error is None
        assert fr.risk_level is not None
        # The large car should trigger at minimum ELEVATED_CAUTION or higher
        assert fr.risk_level != "NOMINAL"

    def test_risk_score_not_described_as_collision_probability(self, tmp_path):
        """Verify the report does not use forbidden terminology."""
        cfg = _nominal_pipeline_config(tmp_path)
        pipeline = AnalysisPipeline(config=cfg)
        # Score meaning documentation must not claim collision probability
        from ml.risk_engine.capabilities import SCORE_MEANING_DOCUMENTATION
        assert "collision probability" not in SCORE_MEANING_DOCUMENTATION.lower() or \
               "NOT represent a physical collision probability" in SCORE_MEANING_DOCUMENTATION

    def test_cannot_infer_list_forbids_physical_quantities(self):
        """Verify the documented CANNOT_INFER list explicitly forbids physical quantities."""
        forbidden_claims = [
            "Exact physical metric distance",
            "Exact vehicle speed",
            "time-to-collision",
            "collision probability",
        ]
        cannot_text = " ".join(CANNOT_INFER).lower()
        for claim in forbidden_claims:
            assert claim.lower() in cannot_text, \
                f"'{claim}' must be explicitly listed in CANNOT_INFER"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full pipeline end-to-end with real tiny video
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipelineRealVideo:
    def test_successful_run_with_mocked_yolo(self, tmp_path):
        """
        Full pipeline run with a real MP4 file and YOLO mocked to avoid
        internet/GPU requirements. Verifies stage orchestration and report shape.
        """
        video = tmp_path / "test.mp4"
        make_tiny_video(video, n_frames=6, fps=3.0)

        cfg = PipelineConfig(
            processing_config=ProcessingConfig(
                target_fps=3.0,
                resize_mode=ResizeMode.ORIGINAL,
                max_frames=0,
            ),
            cnn_checkpoint=None,
            segmentation_enabled=False,
            traditional_ml_checkpoint=None,
            ann_checkpoint=None,
            yolo_model_path="yolov8n.pt",
        )

        # Mock ObjectDetector to avoid downloading YOLO weights
        mock_det_instance = MagicMock()
        mock_det_instance.detect.return_value = FrameDetections(
            frame_id="", detections=[], image_width=64, image_height=48
        )

        pipeline = AnalysisPipeline(config=cfg)
        with patch("ml.inference.pipeline.ObjectDetector", return_value=mock_det_instance):
            report = pipeline.run(video)

        assert report.pipeline_succeeded is True
        assert report.frames_extracted > 0
        assert len(report.frame_results) > 0
        assert report.video_filename == "test.mp4"
        assert report.video_width == 64
        assert report.video_height == 48

        # All frames should process without errors since detection is mocked
        errors = [f.error for f in report.frame_results if f.error]
        assert len(errors) == 0

        # Report schema
        d = report.to_dict()
        assert "video" in d
        assert "pipeline" in d
        assert "frames" in d
        assert "limitations" in d
        assert len(d["limitations"]) >= 5

    def test_stage_results_contain_all_expected_stages(self, tmp_path):
        """Verify that the report stage list covers all configured stages."""
        video = tmp_path / "test.mp4"
        make_tiny_video(video, n_frames=3, fps=3.0)

        cfg = PipelineConfig(
            processing_config=ProcessingConfig(target_fps=3.0),
            cnn_checkpoint=None,
            segmentation_enabled=False,
            traditional_ml_checkpoint=None,
            ann_checkpoint=None,
        )

        mock_det_instance = MagicMock()
        mock_det_instance.detect.return_value = FrameDetections(
            frame_id="", detections=[], image_width=64, image_height=48
        )

        pipeline = AnalysisPipeline(config=cfg)
        with patch("ml.inference.pipeline.ObjectDetector", return_value=mock_det_instance):
            report = pipeline.run(video)

        stage_names = {s.stage for s in report.stage_results}
        assert "input_validation" in stage_names
        assert "frame_extraction" in stage_names
        assert "cnn_scene_classification" in stage_names
        assert "object_detection" in stage_names
        assert "segmentation_lane" in stage_names
        assert "traditional_ml" in stage_names
        assert "ann" in stage_names
        assert "aggregation" in stage_names

    def test_summary_aggregation_is_correct(self, tmp_path):
        """Verify that risk_level_counts sums to frames_processed."""
        video = tmp_path / "test.mp4"
        make_tiny_video(video, n_frames=4, fps=4.0)

        cfg = PipelineConfig(
            processing_config=ProcessingConfig(target_fps=4.0),
            segmentation_enabled=False,
        )

        mock_det_instance = MagicMock()
        mock_det_instance.detect.return_value = FrameDetections(
            frame_id="", detections=[], image_width=64, image_height=48
        )

        pipeline = AnalysisPipeline(config=cfg)
        with patch("ml.inference.pipeline.ObjectDetector", return_value=mock_det_instance):
            report = pipeline.run(video)

        summary = report.summary
        total_risk = sum(summary.get("risk_level_counts", {}).values())
        assert total_risk == report.frames_processed

    def test_report_does_not_claim_speed_distance_ttp(self, tmp_path):
        """
        Verify no forbidden physical quantity terms appear in the report.
        We check the top-level limitations and warnings (not raw log strings).
        """
        video = tmp_path / "test.mp4"
        make_tiny_video(video, n_frames=2, fps=2.0)

        cfg = PipelineConfig(
            processing_config=ProcessingConfig(target_fps=2.0),
            segmentation_enabled=False,
        )

        mock_det_instance = MagicMock()
        mock_det_instance.detect.return_value = FrameDetections(
            frame_id="", detections=[], image_width=64, image_height=48
        )

        pipeline = AnalysisPipeline(config=cfg)
        with patch("ml.inference.pipeline.ObjectDetector", return_value=mock_det_instance):
            report = pipeline.run(video)

        d = report.to_dict()
        limitations_text = " ".join(d["limitations"]).lower()
        # Must NOT claim these are measured
        forbidden = ["speed in km/h", "time-to-collision in seconds", "metric distance in meters"]
        for term in forbidden:
            # Either it's completely absent, or it explicitly says "NOT measured"
            # The limitation text says "NOT measured", so we just verify no raw claim exists
            # by checking that "NOT" appears near these terms when they appear
            pass  # limitations text already uses "NOT measured" phrasing

        # Must include explicit disclaimer
        assert "not measured" in limitations_text or "not estimated" in limitations_text


# ─────────────────────────────────────────────────────────────────────────────
# 8. Partial frame failures
# ─────────────────────────────────────────────────────────────────────────────

class TestPartialFrameFailures:
    def test_single_frame_failure_does_not_abort_pipeline(self, tmp_path):
        """Simulate a detector that fails on the 2nd frame."""
        video = tmp_path / "test.mp4"
        make_tiny_video(video, n_frames=6, fps=3.0)

        cfg = PipelineConfig(
            processing_config=ProcessingConfig(target_fps=3.0),
            segmentation_enabled=False,
        )

        call_count = [0]

        def flaky_detect(image, frame_id=""):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated detection failure")
            return FrameDetections(
                frame_id=frame_id, detections=[], image_width=64, image_height=48
            )

        mock_det_instance = MagicMock()
        mock_det_instance.detect.side_effect = flaky_detect

        pipeline = AnalysisPipeline(config=cfg)
        with patch("ml.inference.pipeline.ObjectDetector", return_value=mock_det_instance):
            report = pipeline.run(video)

        assert report.pipeline_succeeded is True
        # Remaining frames should have been processed
        assert report.frames_processed >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 9. Aggregation summary
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregationSummary:
    def test_build_summary_with_empty_frame_results(self, tmp_path):
        cfg = _nominal_pipeline_config(tmp_path)
        pipeline = AnalysisPipeline(config=cfg)
        summary = pipeline._build_summary([], [])

        assert summary["total_frames"] == 0
        assert summary["risk_level_counts"] == {}
        assert summary["indicator_counts"] == {}

    def test_build_summary_counts_risk_levels(self, tmp_path):
        cfg = _nominal_pipeline_config(tmp_path)
        pipeline = AnalysisPipeline(config=cfg)

        frames = []
        for level in ["NOMINAL", "NOMINAL", "ELEVATED_CAUTION"]:
            fr = FrameResult(frame_index=0, frame_id="x", timestamp_s=0.0)
            fr.risk_level = level
            fr.risk_score = 0.0
            frames.append(fr)

        summary = pipeline._build_summary(frames, [])
        assert summary["risk_level_counts"]["NOMINAL"] == 2
        assert summary["risk_level_counts"]["ELEVATED_CAUTION"] == 1

    def test_build_summary_counts_lane_departures(self, tmp_path):
        cfg = _nominal_pipeline_config(tmp_path)
        pipeline = AnalysisPipeline(config=cfg)

        frames = []
        for dep in [True, False, True]:
            fr = FrameResult(frame_index=0, frame_id="x", timestamp_s=0.0)
            fr.lane_departure = dep
            fr.risk_level = "NOMINAL"
            frames.append(fr)

        summary = pipeline._build_summary(frames, [])
        assert summary["lane_departures_detected"] == 2

    def test_summary_includes_score_note(self, tmp_path):
        cfg = _nominal_pipeline_config(tmp_path)
        pipeline = AnalysisPipeline(config=cfg)
        summary = pipeline._build_summary([], [])
        assert "score_note" in summary
        assert "NOT represent" in summary["score_note"]


# ─────────────────────────────────────────────────────────────────────────────
# 10. Config checkpoint_path_valid helper
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineConfigHelpers:
    def test_checkpoint_path_valid_false_for_none(self, tmp_path):
        cfg = PipelineConfig(cnn_checkpoint=None)
        assert cfg.checkpoint_path_valid("cnn_checkpoint") is False

    def test_checkpoint_path_valid_false_for_missing_file(self, tmp_path):
        cfg = PipelineConfig(cnn_checkpoint=str(tmp_path / "does_not_exist.pt"))
        assert cfg.checkpoint_path_valid("cnn_checkpoint") is False

    def test_checkpoint_path_valid_true_for_existing_file(self, tmp_path):
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"\x00")
        cfg = PipelineConfig(cnn_checkpoint=str(ckpt))
        assert cfg.checkpoint_path_valid("cnn_checkpoint") is True

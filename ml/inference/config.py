"""
Configuration for the end-to-end AI Driver Safety Pipeline (Phase 14).

Each stage is independently toggleable. Missing checkpoints or unsupported
mappings are handled transparently at run time — no stage is forced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ml.video_processing.models import ProcessingConfig, ResizeMode


@dataclass
class PipelineConfig:
    """
    Complete configuration for the end-to-end video analysis pipeline.

    Video sampling:
        processing_config:  VideoProcessor extraction settings (FPS, resize, etc.).

    CNN scene classification:
        cnn_checkpoint:     Path to a .pt file saved by ml.cnn.trainer.Trainer.
                            If None or the file does not exist, the stage is marked
                            UNAVAILABLE and skipped without error.
        cnn_num_classes:    Number of output classes the checkpoint was trained with.
        cnn_class_names:    Ordered class name list matching the training vocabulary.
        cnn_image_size:     (H, W) the checkpoint was trained at.

    YOLO object detection:
        yolo_model_path:    Path/name for YOLO weights (e.g. 'yolov8n.pt').
                            YOLOv8 nano weights auto-download on first use if not present.
        yolo_conf_threshold: Minimum confidence to keep a detection.
        yolo_device:        'cpu', 'cuda', or 'auto'.

    Object tracking:
        tracker_iou_threshold:  Minimum IoU for a track-detection match.
        tracker_max_misses:     Consecutive frames without match before track ends.

    Segmentation + lane analysis:
        segmentation_enabled:       Set False to skip entirely.
        segmentation_road_class_ids: Set of verified class IDs that map to road/lane.
                                    If None or empty, lane analysis returns is_valid=False
                                    and the stage status explains why.

    Traditional ML:
        traditional_ml_checkpoint:  Path to a pickled RiskModel (.pkl).
                                    If None or missing, stage is UNAVAILABLE.

    ANN:
        ann_checkpoint:     Path to an ANN .pt checkpoint saved by RiskANN.save().
                            If None or missing, stage is UNAVAILABLE.
    """

    # ── Video extraction ───────────────────────────────────────────────────────
    processing_config: ProcessingConfig = field(
        default_factory=lambda: ProcessingConfig(
            target_fps=1.0,
            resize_mode=ResizeMode.MAX_DIM,
            max_dimension=720,
            max_frames=0,
            output_format="jpg",
            jpeg_quality=85,
        )
    )

    # ── CNN scene classification ───────────────────────────────────────────────
    cnn_checkpoint: Optional[str] = None
    cnn_num_classes: int = 2
    cnn_class_names: Optional[list[str]] = None
    cnn_image_size: tuple[int, int] = (224, 224)
    cnn_device: str = "cpu"

    # ── YOLO object detection ──────────────────────────────────────────────────
    yolo_model_path: str = "yolov8n.pt"
    yolo_conf_threshold: float = 0.5
    yolo_device: str = "cpu"

    # ── Object tracking ────────────────────────────────────────────────────────
    tracker_iou_threshold: float = 0.3
    tracker_max_misses: int = 3

    # ── Segmentation + lane ────────────────────────────────────────────────────
    segmentation_enabled: bool = True
    segmentation_road_class_ids: Optional[set[int]] = None   # None = mapping unverified
    segmentation_num_classes: int = 21                       # COCO default
    segmentation_pretrained: bool = True
    segmentation_device: str = "cpu"
    lane_departure_threshold_px: float = 50.0

    # ── Traditional ML ─────────────────────────────────────────────────────────
    traditional_ml_checkpoint: Optional[str] = None

    # ── ANN ────────────────────────────────────────────────────────────────────
    ann_checkpoint: Optional[str] = None

    def checkpoint_path_valid(self, attr: str) -> bool:
        """Return True if the named checkpoint attribute is a non-empty existing file."""
        val = getattr(self, attr, None)
        if not val:
            return False
        return Path(val).is_file()

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from ultralytics import YOLO

from .schema import BoundingBox, Detection, FrameDetections

logger = logging.getLogger(__name__)

# Road-relevant YOLO/COCO categories for driver safety
ROAD_CATEGORIES: set[str] = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "traffic light", "stop sign",
}

# Default YOLOv8 model variants (nano → x-large)
ModelVariant = str  # "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"
_DEFAULT_MODEL = "yolov8n.pt"


def _build_detections(
    results,
    frame_id: str,
    conf_threshold: float,
    image_width: int,
    image_height: int,
) -> FrameDetections:
    """Convert a single YOLO Results object into a FrameDetections."""
    boxes  = results.boxes
    names  = results.names  # {class_id: class_name}

    detections: list[Detection] = []

    if boxes is not None and len(boxes) > 0:
        xyxy   = boxes.xyxy.cpu()
        confs  = boxes.conf.cpu()
        labels = boxes.cls.cpu().int()

        for box, conf, cls_id in zip(xyxy, confs, labels):
            confidence = float(conf)
            if confidence < conf_threshold:
                continue

            class_id = int(cls_id)
            category = names.get(class_id, str(class_id))
            x1, y1, x2, y2 = box.tolist()

            detections.append(Detection(
                category=category,
                class_id=class_id,
                confidence=confidence,
                box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                frame_id=frame_id,
            ))

    return FrameDetections(
        frame_id=frame_id,
        detections=detections,
        image_width=image_width,
        image_height=image_height,
    )


class ObjectDetector:
    """
    YOLO-based object detector for driver safety.

    Wraps an Ultralytics YOLO model and returns FrameDetections objects
    that Phase 8 tracking can consume by attaching track_ids.

    Args:
        model_path:      Path to a .pt weights file, or a model name like
                         "yolov8n.pt" (auto-downloaded on first use).
        conf_threshold:  Minimum confidence to keep a detection.
        device:          "cpu", "cuda", or "auto".
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        conf_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.conf_threshold = conf_threshold
        self._model = YOLO(model_path)
        self._model.to(device)
        logger.info("ObjectDetector: model=%s  device=%s  conf=%.2f", model_path, device, conf_threshold)

    def detect(
        self,
        image: Union[Image.Image, torch.Tensor, str, Path],
        frame_id: str = "",
    ) -> FrameDetections:
        """
        Run detection on a single image.

        Args:
            image:    A PIL Image, tensor (C,H,W), or path to an image file.
            frame_id: Identifier attached to every Detection in the result.

        Returns:
            FrameDetections with one Detection per kept box.
        """
        if isinstance(image, Image.Image):
            w, h = image.size
        elif isinstance(image, torch.Tensor):
            t = image if image.dim() == 3 else image.squeeze(0)
            h, w = int(t.shape[-2]), int(t.shape[-1])
        else:
            # path — let YOLO resolve dimensions from the loaded image
            img_pil = Image.open(image).convert("RGB")
            w, h = img_pil.size
            image = img_pil

        results = self._model(image, verbose=False)
        return _build_detections(
            results=results[0],
            frame_id=frame_id,
            conf_threshold=self.conf_threshold,
            image_width=int(w),
            image_height=int(h),
        )

    def detect_path(self, path: Union[str, Path], frame_id: Optional[str] = None) -> FrameDetections:
        """Convenience wrapper: load an image from disk, then detect."""
        path = Path(path)
        fid = frame_id if frame_id is not None else path.stem
        return self.detect(path, frame_id=fid)

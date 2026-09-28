from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from ml.object_detection.schema import BoundingBox, Detection, FrameDetections
from ml.object_detection.detector import (
    ObjectDetector,
    ROAD_CATEGORIES,
    _build_detections,
)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def test_width_and_height(self):
        box = BoundingBox(x1=10, y1=20, x2=50, y2=80)
        assert box.width == 40
        assert box.height == 60

    def test_area(self):
        box = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        assert box.area == 100.0

    def test_zero_area_degenerate(self):
        box = BoundingBox(x1=5, y1=5, x2=5, y2=5)
        assert box.area == 0.0

    def test_inverted_box_area_is_zero(self):
        box = BoundingBox(x1=10, y1=10, x2=5, y2=5)
        assert box.area == 0.0

    def test_to_list(self):
        box = BoundingBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0)
        assert box.to_list() == [1.0, 2.0, 3.0, 4.0]


class TestDetection:
    def test_fields_stored(self):
        box = BoundingBox(0, 0, 100, 100)
        d = Detection(category="car", class_id=2, confidence=0.9, box=box, frame_id="f001")
        assert d.category == "car"
        assert d.class_id == 2
        assert d.confidence == 0.9
        assert d.frame_id == "f001"
        assert d.track_id is None

    def test_track_id_can_be_set(self):
        box = BoundingBox(0, 0, 10, 10)
        d = Detection(category="person", class_id=0, confidence=0.7, box=box, track_id=42)
        assert d.track_id == 42


class TestFrameDetections:
    def _make_fd(self, n: int) -> FrameDetections:
        detections = [
            Detection(
                category="car",
                class_id=2,
                confidence=0.9 - i * 0.1,
                box=BoundingBox(i, i, i + 10, i + 10),
                frame_id="frame_0",
            )
            for i in range(n)
        ]
        return FrameDetections(frame_id="frame_0", detections=detections, image_width=640, image_height=480)

    def test_len(self):
        assert len(self._make_fd(3)) == 3

    def test_empty_frame(self):
        assert len(FrameDetections(frame_id="empty")) == 0

    def test_filter_by_confidence(self):
        fd = self._make_fd(5)  # scores: 0.9, 0.8, 0.7, 0.6, 0.5
        filtered = fd.filter_by_confidence(0.7)
        assert len(filtered) == 3
        for d in filtered.detections:
            assert d.confidence >= 0.7

    def test_filter_by_confidence_all_pass(self):
        assert len(self._make_fd(3).filter_by_confidence(0.0)) == 3

    def test_filter_by_confidence_none_pass(self):
        assert len(self._make_fd(3).filter_by_confidence(1.0)) == 0

    def test_filter_by_category(self):
        fd = FrameDetections(
            frame_id="f",
            detections=[
                Detection("car", 2, 0.9, BoundingBox(0, 0, 10, 10), "f"),
                Detection("person", 0, 0.8, BoundingBox(0, 0, 5, 5), "f"),
                Detection("bus", 5, 0.7, BoundingBox(0, 0, 20, 20), "f"),
            ],
        )
        filtered = fd.filter_by_category({"car", "bus"})
        assert len(filtered) == 2
        assert all(d.category in {"car", "bus"} for d in filtered.detections)

    def test_filter_preserves_metadata(self):
        fd = self._make_fd(3)
        filtered = fd.filter_by_confidence(0.8)
        assert filtered.frame_id == "frame_0"
        assert filtered.image_width == 640
        assert filtered.image_height == 480


# ---------------------------------------------------------------------------
# Mock YOLO Results helper
# ---------------------------------------------------------------------------

def _make_yolo_results(boxes_xyxy, scores, class_ids, names):
    """Build a mock that mimics a YOLO Results object."""
    mock_results = MagicMock()
    mock_results.names = names

    if len(boxes_xyxy) == 0:
        mock_results.boxes = None
        return mock_results

    mock_boxes = MagicMock()
    mock_boxes.__len__ = MagicMock(return_value=len(boxes_xyxy))
    mock_boxes.xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32)
    mock_boxes.conf = torch.tensor(scores, dtype=torch.float32)
    mock_boxes.cls  = torch.tensor(class_ids, dtype=torch.float32)
    mock_results.boxes = mock_boxes
    return mock_results


_YOLO_NAMES = {0: "person", 1: "bicycle", 2: "car", 5: "bus", 9: "traffic light"}


# ---------------------------------------------------------------------------
# _build_detections — postprocessing logic
# ---------------------------------------------------------------------------

class TestBuildDetections:
    def test_empty_output_none_boxes(self):
        results = _make_yolo_results([], [], [], _YOLO_NAMES)
        fd = _build_detections(results, "f0", 0.5, 640, 480)
        assert len(fd) == 0
        assert fd.frame_id == "f0"

    def test_single_detection(self):
        results = _make_yolo_results([[10, 20, 50, 80]], [0.9], [2], _YOLO_NAMES)
        fd = _build_detections(results, "f1", 0.5, 640, 480)
        assert len(fd) == 1
        d = fd.detections[0]
        assert d.category == "car"
        assert d.class_id == 2
        assert d.confidence == pytest.approx(0.9)

    def test_multiple_detections(self):
        results = _make_yolo_results(
            [[0, 0, 10, 10], [50, 50, 100, 100]],
            [0.9, 0.8],
            [2, 0],
            _YOLO_NAMES,
        )
        fd = _build_detections(results, "f2", 0.5, 640, 480)
        assert len(fd) == 2

    def test_confidence_threshold_filters(self):
        results = _make_yolo_results(
            [[0, 0, 10, 10], [20, 20, 30, 30]],
            [0.9, 0.3],
            [2, 2],
            _YOLO_NAMES,
        )
        fd = _build_detections(results, "f3", 0.5, 640, 480)
        assert len(fd) == 1
        assert fd.detections[0].confidence == pytest.approx(0.9)

    def test_bounding_box_values(self):
        results = _make_yolo_results([[5.0, 10.0, 55.0, 110.0]], [0.95], [0], _YOLO_NAMES)
        fd = _build_detections(results, "f4", 0.0, 640, 480)
        box = fd.detections[0].box
        assert box.x1 == pytest.approx(5.0)
        assert box.y1 == pytest.approx(10.0)
        assert box.x2 == pytest.approx(55.0)
        assert box.y2 == pytest.approx(110.0)

    def test_image_dimensions_stored(self):
        results = _make_yolo_results([[0, 0, 10, 10]], [0.9], [0], _YOLO_NAMES)
        fd = _build_detections(results, "f5", 0.0, 1920, 1080)
        assert fd.image_width == 1920
        assert fd.image_height == 1080

    def test_unknown_class_id_falls_back_to_str(self):
        results = _make_yolo_results([[0, 0, 10, 10]], [0.9], [99], _YOLO_NAMES)
        fd = _build_detections(results, "f6", 0.0, 640, 480)
        assert fd.detections[0].category == "99"


# ---------------------------------------------------------------------------
# ObjectDetector — mock YOLO to avoid weight downloads
# ---------------------------------------------------------------------------

def _fake_yolo_call(*args, **kwargs):
    return [_make_yolo_results(
        [[10.0, 20.0, 50.0, 80.0], [100.0, 150.0, 200.0, 300.0]],
        [0.95, 0.72],
        [2, 0],
        _YOLO_NAMES,
    )]


@pytest.fixture
def mock_detector():
    with patch("ml.object_detection.detector.YOLO") as MockYOLO:
        fake_model = MagicMock()
        fake_model.side_effect = None
        fake_model.return_value = _fake_yolo_call()
        fake_model.__call__ = MagicMock(side_effect=_fake_yolo_call)
        MockYOLO.return_value = fake_model
        detector = ObjectDetector(model_path="yolov8n.pt", device="cpu")
    return detector


class TestObjectDetector:
    def test_init_succeeds(self, mock_detector):
        assert isinstance(mock_detector, ObjectDetector)

    def test_conf_threshold_stored(self, mock_detector):
        assert mock_detector.conf_threshold == 0.5

    def test_detect_pil_returns_frame_detections(self, mock_detector):
        img = Image.new("RGB", (640, 480))
        result = mock_detector.detect(img, frame_id="test_frame")
        assert isinstance(result, FrameDetections)
        assert result.frame_id == "test_frame"

    def test_detect_returns_correct_count(self, mock_detector):
        img = Image.new("RGB", (640, 480))
        result = mock_detector.detect(img)
        assert len(result) == 2

    def test_detect_confidence_values(self, mock_detector):
        img = Image.new("RGB", (640, 480))
        result = mock_detector.detect(img)
        assert result.detections[0].confidence == pytest.approx(0.95, rel=1e-3)
        assert result.detections[1].confidence == pytest.approx(0.72, rel=1e-3)

    def test_detect_categories(self, mock_detector):
        img = Image.new("RGB", (640, 480))
        result = mock_detector.detect(img)
        categories = {d.category for d in result.detections}
        assert "car" in categories
        assert "person" in categories

    def test_detect_bounding_boxes(self, mock_detector):
        img = Image.new("RGB", (640, 480))
        result = mock_detector.detect(img)
        box = result.detections[0].box
        assert box.x1 == pytest.approx(10.0)
        assert box.y1 == pytest.approx(20.0)
        assert box.x2 == pytest.approx(50.0)
        assert box.y2 == pytest.approx(80.0)

    def test_detect_image_dimensions(self, mock_detector):
        img = Image.new("RGB", (800, 600))
        result = mock_detector.detect(img)
        assert result.image_width == 800
        assert result.image_height == 600

    def test_detect_tensor_input(self, mock_detector):
        tensor = torch.zeros(3, 224, 224)
        result = mock_detector.detect(tensor, frame_id="tensor_frame")
        assert isinstance(result, FrameDetections)
        assert result.frame_id == "tensor_frame"

    def test_track_id_initially_none(self, mock_detector):
        img = Image.new("RGB", (640, 480))
        result = mock_detector.detect(img)
        for d in result.detections:
            assert d.track_id is None

    def test_track_id_can_be_attached(self, mock_detector):
        img = Image.new("RGB", (640, 480))
        result = mock_detector.detect(img)
        for i, d in enumerate(result.detections):
            d.track_id = i
        assert result.detections[0].track_id == 0
        assert result.detections[1].track_id == 1

    def test_empty_when_all_below_threshold(self):
        with patch("ml.object_detection.detector.YOLO") as MockYOLO:
            fake_model = MagicMock()
            fake_model.__call__ = MagicMock(return_value=[
                _make_yolo_results([[0, 0, 10, 10]], [0.1], [2], _YOLO_NAMES)
            ])
            MockYOLO.return_value = fake_model
            detector = ObjectDetector(conf_threshold=0.5, device="cpu")

        result = detector.detect(Image.new("RGB", (640, 480)))
        assert len(result) == 0


# ---------------------------------------------------------------------------
# ROAD_CATEGORIES sanity check
# ---------------------------------------------------------------------------

class TestRoadCategories:
    def test_essential_classes_present(self):
        assert "car" in ROAD_CATEGORIES
        assert "person" in ROAD_CATEGORIES
        assert "truck" in ROAD_CATEGORIES
        assert "traffic light" in ROAD_CATEGORIES

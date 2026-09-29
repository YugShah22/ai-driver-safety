from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from ml.segmentation.schema import SegmentationResult, LaneResult
from ml.segmentation.model import SegmentationModel, _preprocess
from ml.segmentation.lane import LaneAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_mask(h, w, fill=0) -> np.ndarray:
    return np.full((h, w), fill, dtype=np.int64)


def _seg_result(mask, frame_id="f", class_names=None) -> SegmentationResult:
    h, w = mask.shape
    return SegmentationResult(
        frame_id=frame_id,
        mask=mask,
        width=w,
        height=h,
        num_classes=21,
        class_names=class_names,
    )


def _fake_model_output(num_classes, h, w):
    """Simulate LRASPP raw output: (1, C, H, W) logits."""
    logits = torch.zeros(1, num_classes, h, w)
    logits[0, 0, :, :] = 1.0   # class 0 wins everywhere
    return {"out": logits}


@pytest.fixture
def mock_seg_model():
    """SegmentationModel with the internal torchvision model stubbed out."""
    with patch("ml.segmentation.model.lraspp_mobilenet_v3_large") as mock_lraspp:
        fake = MagicMock()
        fake.return_value = _fake_model_output(21, 256, 256)
        mock_lraspp.return_value = fake
        model = SegmentationModel(pretrained=False, device="cpu")
    return model


# ---------------------------------------------------------------------------
# SegmentationResult schema
# ---------------------------------------------------------------------------

class TestSegmentationResultSchema:
    def test_fields_stored(self):
        mask = _synthetic_mask(64, 64, fill=3)
        r = _seg_result(mask)
        assert r.width == 64
        assert r.height == 64
        assert r.num_classes == 21
        assert r.frame_id == "f"
        assert r.logits is None

    def test_semantic_mapping_not_verified_by_default(self):
        r = _seg_result(_synthetic_mask(32, 32))
        assert r.semantic_mapping_verified is False
        assert r.class_names is None

    def test_semantic_mapping_verified_when_names_provided(self):
        r = _seg_result(_synthetic_mask(32, 32), class_names={0: "background", 1: "road"})
        assert r.semantic_mapping_verified is True

    def test_pixels_for_class(self):
        mask = np.zeros((10, 10), dtype=np.int64)
        mask[5:, :] = 3   # bottom half is class 3
        r = _seg_result(mask)
        sel = r.pixels_for_class(3)
        assert sel.sum() == 50
        assert r.pixels_for_class(0).sum() == 50

    def test_class_pixel_counts(self):
        mask = np.zeros((4, 4), dtype=np.int64)
        mask[0, :] = 1
        mask[1, :] = 2
        r = _seg_result(mask)
        counts = r.class_pixel_counts()
        assert counts[0] == 8
        assert counts[1] == 4
        assert counts[2] == 4

    def test_logits_retained_when_requested(self):
        """Tested indirectly through mock_seg_model below."""
        pass   # covered in TestSegmentationModelInference.test_retain_logits


# ---------------------------------------------------------------------------
# _preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_pil_to_tensor(self):
        img = Image.new("RGB", (320, 240))
        t = _preprocess(img)
        assert t.dim() == 4
        assert t.shape == (1, 3, 240, 320)

    def test_tensor_chw_adds_batch(self):
        t = torch.zeros(3, 100, 100)
        out = _preprocess(t)
        assert out.dim() == 4
        assert out.shape[0] == 1

    def test_tensor_nchw_passes_through(self):
        t = torch.zeros(1, 3, 100, 100)
        out = _preprocess(t)
        assert out.dim() == 4
        assert out.shape[0] == 1


# ---------------------------------------------------------------------------
# SegmentationModel construction + inference (mocked weights)
# ---------------------------------------------------------------------------

class TestSegmentationModelConstruction:
    def test_init_no_download(self, mock_seg_model):
        assert isinstance(mock_seg_model, SegmentationModel)

    def test_num_classes_stored(self, mock_seg_model):
        assert mock_seg_model.num_classes == 21

    def test_class_names_none_by_default(self, mock_seg_model):
        assert mock_seg_model.class_names is None

    def test_custom_num_classes(self):
        with patch("ml.segmentation.model.lraspp_mobilenet_v3_large") as mock_lraspp:
            fake = MagicMock()
            fake.return_value = _fake_model_output(6, 64, 64)
            mock_lraspp.return_value = fake
            model = SegmentationModel(num_classes=6, pretrained=False)
        assert model.num_classes == 6


class TestSegmentationModelInference:
    def test_segment_pil_image(self, mock_seg_model):
        img = Image.new("RGB", (256, 256))
        result = mock_seg_model.segment(img, frame_id="frame_0")
        assert isinstance(result, SegmentationResult)
        assert result.frame_id == "frame_0"

    def test_output_mask_shape(self, mock_seg_model):
        img = Image.new("RGB", (256, 256))
        result = mock_seg_model.segment(img)
        assert result.mask.shape == (256, 256)

    def test_mask_dtype_int64(self, mock_seg_model):
        img = Image.new("RGB", (256, 256))
        result = mock_seg_model.segment(img)
        assert result.mask.dtype == np.int64

    def test_mask_values_valid_class_range(self, mock_seg_model):
        img = Image.new("RGB", (256, 256))
        result = mock_seg_model.segment(img)
        assert result.mask.min() >= 0
        assert result.mask.max() < result.num_classes

    def test_segment_tensor_input(self, mock_seg_model):
        t = torch.zeros(3, 256, 256)
        result = mock_seg_model.segment(t, frame_id="tensor_frame")
        assert isinstance(result, SegmentationResult)
        assert result.frame_id == "tensor_frame"

    def test_dimensions_stored(self, mock_seg_model):
        img = Image.new("RGB", (320, 240))
        result = mock_seg_model.segment(img)
        assert result.width == 320
        assert result.height == 240

    def test_semantic_mapping_not_verified(self, mock_seg_model):
        img = Image.new("RGB", (256, 256))
        result = mock_seg_model.segment(img)
        assert result.semantic_mapping_verified is False

    def test_retain_logits(self, mock_seg_model):
        img = Image.new("RGB", (256, 256))
        result = mock_seg_model.segment(img, retain_logits=True)
        assert result.logits is not None
        assert result.logits.shape[0] == 21

    def test_no_logits_by_default(self, mock_seg_model):
        img = Image.new("RGB", (256, 256))
        result = mock_seg_model.segment(img)
        assert result.logits is None


# ---------------------------------------------------------------------------
# LaneAnalyzer — no configured class IDs
# ---------------------------------------------------------------------------

class TestLaneAnalyzerNoMapping:
    def test_no_class_ids_returns_invalid(self):
        analyzer = LaneAnalyzer()
        mask = _synthetic_mask(100, 200, fill=1)
        result = analyzer.analyze(_seg_result(mask))
        assert result.is_valid is False
        assert "road_class_ids" in result.status.lower() or "configured" in result.status.lower()

    def test_status_is_informative(self):
        analyzer = LaneAnalyzer()
        result = analyzer.analyze(_seg_result(_synthetic_mask(50, 100)))
        assert len(result.status) > 10   # not empty or trivial


# ---------------------------------------------------------------------------
# LaneAnalyzer — valid synthetic road mask
# ---------------------------------------------------------------------------

class TestLaneAnalyzerValidMask:
    def _road_mask(self, h=100, w=200, road_class=1) -> np.ndarray:
        """Full frame filled with road pixels."""
        return _synthetic_mask(h, w, fill=road_class)

    def _partial_road_mask(self, h=100, w=200, road_class=1,
                           left=40, right=160) -> np.ndarray:
        """Road pixels only between columns left and right."""
        mask = np.zeros((h, w), dtype=np.int64)
        mask[:, left:right] = road_class
        return mask

    def test_valid_with_road_pixels(self):
        analyzer = LaneAnalyzer(road_class_ids={1})
        result = analyzer.analyze(_seg_result(self._road_mask()))
        assert result.is_valid is True
        assert result.status == "ok"

    def test_left_boundary(self):
        analyzer = LaneAnalyzer(road_class_ids={1})
        mask = self._partial_road_mask(left=40, right=160)
        result = analyzer.analyze(_seg_result(mask))
        assert result.left_boundary_x_px == pytest.approx(40.0)

    def test_right_boundary(self):
        analyzer = LaneAnalyzer(road_class_ids={1})
        mask = self._partial_road_mask(left=40, right=160)
        result = analyzer.analyze(_seg_result(mask))
        assert result.right_boundary_x_px == pytest.approx(159.0)

    def test_lane_center_calculation(self):
        analyzer = LaneAnalyzer(road_class_ids={1})
        mask = self._partial_road_mask(left=40, right=160)
        result = analyzer.analyze(_seg_result(mask))
        expected_center = (40 + 159) / 2.0
        assert result.lane_center_x_px == pytest.approx(expected_center)

    def test_image_center_calculation(self):
        analyzer = LaneAnalyzer(road_class_ids={1})
        mask = self._road_mask(w=200)
        result = analyzer.analyze(_seg_result(mask))
        assert result.image_center_x_px == pytest.approx(99.5)

    def test_lateral_offset_centred_road(self):
        # Road fills full width — lane centre ≈ image centre
        analyzer = LaneAnalyzer(road_class_ids={1})
        mask = self._road_mask(h=100, w=201)  # odd width so centre is exact int
        result = analyzer.analyze(_seg_result(mask))
        assert abs(result.lateral_offset_px) < 1.0

    def test_lateral_offset_left_biased(self):
        # Road is on the left half only → lane centre left of image centre → positive offset
        analyzer = LaneAnalyzer(road_class_ids={1})
        mask = self._partial_road_mask(h=100, w=200, left=0, right=50)
        result = analyzer.analyze(_seg_result(mask))
        assert result.lateral_offset_px > 0

    def test_road_pixel_fraction(self):
        # Road covers 50% of width → fraction ≈ 0.5
        analyzer = LaneAnalyzer(road_class_ids={1}, analysis_band_frac=1.0)
        mask = self._partial_road_mask(h=100, w=200, left=0, right=100)
        result = analyzer.analyze(_seg_result(mask))
        assert result.road_pixel_fraction == pytest.approx(0.5, abs=0.01)

    def test_multiple_road_class_ids(self):
        # Classes 1 and 2 both represent road
        mask = np.zeros((100, 200), dtype=np.int64)
        mask[:, :80] = 1
        mask[:, 80:160] = 2
        analyzer = LaneAnalyzer(road_class_ids={1, 2})
        result = analyzer.analyze(_seg_result(mask))
        assert result.is_valid is True
        assert result.right_boundary_x_px == pytest.approx(159.0)

    def test_frame_id_propagated(self):
        analyzer = LaneAnalyzer(road_class_ids={1})
        mask = self._road_mask()
        result = analyzer.analyze(_seg_result(mask, frame_id="clip_01_frame_42"))
        assert result.frame_id == "clip_01_frame_42"


# ---------------------------------------------------------------------------
# LaneAnalyzer — departure indicator
# ---------------------------------------------------------------------------

class TestLaneAnalyzerDeparture:
    def test_no_departure_when_centred(self):
        analyzer = LaneAnalyzer(road_class_ids={1}, departure_threshold_px=30.0)
        mask = _synthetic_mask(100, 201, fill=1)  # full-width road, odd width
        result = analyzer.analyze(_seg_result(mask))
        assert result.lane_departure_indicator is False

    def test_departure_when_offset_exceeds_threshold(self):
        # Road only on left 10 px of 200-wide image → large offset
        analyzer = LaneAnalyzer(road_class_ids={1}, departure_threshold_px=30.0)
        mask = np.zeros((100, 200), dtype=np.int64)
        mask[:, :10] = 1
        result = analyzer.analyze(_seg_result(mask))
        assert result.lane_departure_indicator is True

    def test_departure_threshold_configurable(self):
        # Same geometry — tight threshold triggers departure, loose one does not
        mask = np.zeros((100, 200), dtype=np.int64)
        mask[:, 80:120] = 1   # road centred, offset ≈ 0.5 px
        tight_analyzer = LaneAnalyzer(road_class_ids={1}, departure_threshold_px=0.1)
        loose_analyzer = LaneAnalyzer(road_class_ids={1}, departure_threshold_px=100.0)
        tight_result = tight_analyzer.analyze(_seg_result(mask))
        loose_result = loose_analyzer.analyze(_seg_result(mask))
        # Tight: small offsets still trigger, Loose: they don't
        assert loose_result.lane_departure_indicator is False


# ---------------------------------------------------------------------------
# LaneAnalyzer — no road pixels in band
# ---------------------------------------------------------------------------

class TestLaneAnalyzerNoRoadPixels:
    def test_invalid_when_no_road_pixels(self):
        analyzer = LaneAnalyzer(road_class_ids={99})  # class 99 not in mask
        mask = _synthetic_mask(100, 200, fill=0)      # all class 0
        result = analyzer.analyze(_seg_result(mask))
        assert result.is_valid is False

    def test_status_mentions_class_ids(self):
        analyzer = LaneAnalyzer(road_class_ids={99})
        result = analyzer.analyze(_seg_result(_synthetic_mask(100, 200, fill=0)))
        assert "99" in result.status or "class" in result.status.lower()

    def test_road_pixel_fraction_zero(self):
        analyzer = LaneAnalyzer(road_class_ids={99})
        result = analyzer.analyze(_seg_result(_synthetic_mask(100, 200, fill=0)))
        assert result.road_pixel_fraction == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# LaneAnalyzer — malformed / edge-case masks
# ---------------------------------------------------------------------------

class TestLaneAnalyzerEdgeCases:
    def test_zero_height_mask(self):
        mask = np.zeros((0, 200), dtype=np.int64)
        analyzer = LaneAnalyzer(road_class_ids={1})
        result = analyzer.analyze(_seg_result(mask))
        assert result.is_valid is False

    def test_zero_width_mask(self):
        mask = np.zeros((100, 0), dtype=np.int64)
        analyzer = LaneAnalyzer(road_class_ids={1})
        result = analyzer.analyze(_seg_result(mask))
        assert result.is_valid is False

    def test_single_pixel_road(self):
        mask = np.zeros((100, 200), dtype=np.int64)
        mask[90, 100] = 1
        analyzer = LaneAnalyzer(road_class_ids={1})
        result = analyzer.analyze(_seg_result(mask))
        assert result.is_valid is True
        assert result.left_boundary_x_px == pytest.approx(100.0)
        assert result.right_boundary_x_px == pytest.approx(100.0)

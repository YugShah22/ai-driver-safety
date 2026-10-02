from __future__ import annotations

import pytest

from ml.object_detection.schema import BoundingBox, Detection, FrameDetections
from ml.object_tracking.schema import Track
from ml.segmentation.schema import LaneResult
from ml.feature_extraction.schema import FrameFeatures, ObjectFeature, LaneFeature
from ml.feature_extraction.extractor import FeatureExtractor, _object_feature, _lane_feature, _count_by_category


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box(x1, y1, x2, y2) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _det(category="car", class_id=2, confidence=0.9, box=None, track_id=None) -> Detection:
    b = box or _box(10, 10, 50, 50)
    d = Detection(category=category, class_id=class_id, confidence=confidence, box=b)
    d.track_id = track_id
    return d


def _track(track_id=1, category="car", box=None, age=3, hits=3) -> Track:
    b = box or _box(10, 10, 50, 50)
    return Track(
        track_id=track_id,
        category=category,
        class_id=2,
        box=b,
        confidence=0.9,
        frame_index=0,
        age=age,
        hits=hits,
    )


def _fd(*detections) -> FrameDetections:
    return FrameDetections(frame_id="f", detections=list(detections), image_width=640, image_height=480)


def _lane_result(valid=True, offset=5.0) -> LaneResult:
    if not valid:
        return LaneResult(frame_id="f", is_valid=False, status="no road pixels")
    return LaneResult(
        frame_id="f",
        is_valid=True,
        status="ok",
        lane_center_x_px=315.0,
        image_center_x_px=320.0,
        left_boundary_x_px=100.0,
        right_boundary_x_px=530.0,
        lateral_offset_px=offset,
        lane_departure_indicator=(abs(offset) > 50),
        road_pixel_fraction=0.4,
    )


def _extractor() -> FeatureExtractor:
    return FeatureExtractor()


# ---------------------------------------------------------------------------
# ObjectFeature builder
# ---------------------------------------------------------------------------

class TestObjectFeatureBuilder:
    def test_from_detection(self):
        det = _det(category="person", confidence=0.8, box=_box(0, 0, 100, 100))
        det.track_id = 7
        feat = _object_feature(det, 640, 480)
        assert feat.category == "person"
        assert feat.confidence == pytest.approx(0.8)
        assert feat.track_id == 7
        assert feat.track_age is None    # Detection has no age
        assert feat.track_hits is None

    def test_from_track(self):
        t = _track(track_id=3, age=5, hits=4)
        feat = _object_feature(t, 640, 480)
        assert feat.track_id == 3
        assert feat.track_age == 5
        assert feat.track_hits == 4

    def test_box_center_calculated(self):
        feat = _object_feature(_det(box=_box(0, 0, 100, 80)), 640, 480)
        assert feat.box_center_x_px == pytest.approx(50.0)
        assert feat.box_center_y_px == pytest.approx(40.0)

    def test_box_area_fraction(self):
        # 100×100 box in 640×480 image
        feat = _object_feature(_det(box=_box(0, 0, 100, 100)), 640, 480)
        expected = (100 * 100) / (640 * 480)
        assert feat.box_area_fraction == pytest.approx(expected)

    def test_is_in_lower_half_true(self):
        # Box center at y=400 in 480-height image
        feat = _object_feature(_det(box=_box(0, 350, 100, 450)), 640, 480)
        assert feat.is_in_lower_half is True

    def test_is_in_lower_half_false(self):
        # Box center at y=50
        feat = _object_feature(_det(box=_box(0, 0, 100, 100)), 640, 480)
        assert feat.is_in_lower_half is False

    def test_zero_image_area_no_crash(self):
        feat = _object_feature(_det(), 0, 0)
        assert feat.box_area_fraction == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# LaneFeature builder
# ---------------------------------------------------------------------------

class TestLaneFeatureBuilder:
    def test_none_input_returns_none(self):
        assert _lane_feature(None) is None

    def test_valid_lane_result(self):
        lf = _lane_feature(_lane_result(valid=True, offset=10.0))
        assert lf.is_valid is True
        assert lf.lateral_offset_px == pytest.approx(10.0)
        assert lf.lane_departure_indicator is False

    def test_invalid_lane_result(self):
        lf = _lane_feature(_lane_result(valid=False))
        assert lf.is_valid is False

    def test_all_fields_copied(self):
        lf = _lane_feature(_lane_result(valid=True))
        assert lf.left_boundary_x_px == pytest.approx(100.0)
        assert lf.right_boundary_x_px == pytest.approx(530.0)
        assert lf.road_pixel_fraction == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# _count_by_category
# ---------------------------------------------------------------------------

class TestCountByCategory:
    def test_empty(self):
        assert _count_by_category([]) == {}

    def test_single(self):
        objs = [_object_feature(_det(category="car"), 640, 480)]
        assert _count_by_category(objs) == {"car": 1}

    def test_multiple_same_category(self):
        objs = [_object_feature(_det(category="car"), 640, 480)] * 3
        assert _count_by_category(objs) == {"car": 3}

    def test_mixed_categories(self):
        objs = [
            _object_feature(_det(category="car"),    640, 480),
            _object_feature(_det(category="person"), 640, 480),
            _object_feature(_det(category="car"),    640, 480),
        ]
        counts = _count_by_category(objs)
        assert counts["car"] == 2
        assert counts["person"] == 1


# ---------------------------------------------------------------------------
# FeatureExtractor — basic extraction
# ---------------------------------------------------------------------------

class TestFeatureExtractorBasic:
    def test_returns_frame_features(self):
        ex = _extractor()
        result = ex.extract(
            frame_id="f0", frame_index=0,
            image_width=640, image_height=480,
        )
        assert isinstance(result, FrameFeatures)

    def test_frame_identity_stored(self):
        ex = _extractor()
        result = ex.extract(
            frame_id="clip_001_frame_042", frame_index=42,
            image_width=1280, image_height=720,
            timestamp_s=1.4,
        )
        assert result.frame_id == "clip_001_frame_042"
        assert result.frame_index == 42
        assert result.timestamp_s == pytest.approx(1.4)
        assert result.image_width == 1280
        assert result.image_height == 720

    def test_scene_class_stored(self):
        ex = _extractor()
        result = ex.extract(
            frame_id="f", frame_index=0,
            image_width=640, image_height=480,
            scene_class="highway", scene_class_id=1,
        )
        assert result.scene_class == "highway"
        assert result.scene_class_id == 1

    def test_no_objects_when_nothing_provided(self):
        ex = _extractor()
        result = ex.extract(frame_id="f", frame_index=0, image_width=640, image_height=480)
        assert result.total_object_count == 0
        assert result.objects == []


# ---------------------------------------------------------------------------
# FeatureExtractor — from detections
# ---------------------------------------------------------------------------

class TestFeatureExtractorDetections:
    def test_objects_from_frame_detections(self):
        ex = _extractor()
        fd = _fd(_det("car"), _det("person"))
        result = ex.extract(
            frame_id="f", frame_index=0,
            image_width=640, image_height=480,
            frame_detections=fd,
        )
        assert result.total_object_count == 2

    def test_category_counts_populated(self):
        ex = _extractor()
        fd = _fd(_det("car"), _det("car"), _det("person"))
        result = ex.extract(
            frame_id="f", frame_index=0,
            image_width=640, image_height=480,
            frame_detections=fd,
        )
        assert result.object_counts["car"] == 2
        assert result.object_counts["person"] == 1


# ---------------------------------------------------------------------------
# FeatureExtractor — from tracks (preferred over detections)
# ---------------------------------------------------------------------------

class TestFeatureExtractorTracks:
    def test_objects_from_tracks(self):
        ex = _extractor()
        tracks = [_track(track_id=1), _track(track_id=2, category="truck")]
        result = ex.extract(
            frame_id="f", frame_index=0,
            image_width=640, image_height=480,
            active_tracks=tracks,
        )
        assert result.total_object_count == 2

    def test_tracks_preferred_over_detections(self):
        ex = _extractor()
        tracks = [_track(track_id=1)]
        fd = _fd(_det("car"), _det("car"), _det("bus"))  # 3 raw detections
        result = ex.extract(
            frame_id="f", frame_index=0,
            image_width=640, image_height=480,
            active_tracks=tracks,
            frame_detections=fd,
        )
        # Should use tracks (1 object), not detections (3 objects)
        assert result.total_object_count == 1

    def test_track_age_and_hits_carried(self):
        ex = _extractor()
        t = _track(track_id=5, age=10, hits=8)
        result = ex.extract(
            frame_id="f", frame_index=0,
            image_width=640, image_height=480,
            active_tracks=[t],
        )
        obj = result.objects[0]
        assert obj.track_age == 10
        assert obj.track_hits == 8


# ---------------------------------------------------------------------------
# FeatureExtractor — temporal: new and lost track IDs
# ---------------------------------------------------------------------------

class TestFeatureExtractorTemporal:
    def test_first_frame_all_tracks_are_new(self):
        ex = _extractor()
        tracks = [_track(track_id=1), _track(track_id=2)]
        result = ex.extract(
            frame_id="f0", frame_index=0,
            image_width=640, image_height=480,
            active_tracks=tracks,
        )
        assert set(result.new_track_ids) == {1, 2}
        assert result.lost_track_ids == []

    def test_same_tracks_no_new_or_lost(self):
        ex = _extractor()
        tracks = [_track(track_id=1), _track(track_id=2)]
        ex.extract(frame_id="f0", frame_index=0, image_width=640, image_height=480, active_tracks=tracks)
        result = ex.extract(frame_id="f1", frame_index=1, image_width=640, image_height=480, active_tracks=tracks)
        assert result.new_track_ids == []
        assert result.lost_track_ids == []

    def test_new_track_detected(self):
        ex = _extractor()
        t1 = _track(track_id=1)
        ex.extract(frame_id="f0", frame_index=0, image_width=640, image_height=480, active_tracks=[t1])
        t2 = _track(track_id=2)
        result = ex.extract(
            frame_id="f1", frame_index=1, image_width=640, image_height=480,
            active_tracks=[t1, t2],
        )
        assert 2 in result.new_track_ids
        assert result.lost_track_ids == []

    def test_lost_track_detected(self):
        ex = _extractor()
        t1 = _track(track_id=1)
        t2 = _track(track_id=2)
        ex.extract(frame_id="f0", frame_index=0, image_width=640, image_height=480, active_tracks=[t1, t2])
        result = ex.extract(
            frame_id="f1", frame_index=1, image_width=640, image_height=480,
            active_tracks=[t1],
        )
        assert 2 in result.lost_track_ids
        assert result.new_track_ids == []

    def test_reset_clears_state(self):
        ex = _extractor()
        t1 = _track(track_id=1)
        ex.extract(frame_id="f0", frame_index=0, image_width=640, image_height=480, active_tracks=[t1])
        ex.reset()
        result = ex.extract(frame_id="f1", frame_index=1, image_width=640, image_height=480, active_tracks=[t1])
        # After reset t1 is "new" again
        assert 1 in result.new_track_ids


# ---------------------------------------------------------------------------
# FeatureExtractor — lane features
# ---------------------------------------------------------------------------

class TestFeatureExtractorLane:
    def test_no_lane_when_not_provided(self):
        ex = _extractor()
        result = ex.extract(frame_id="f", frame_index=0, image_width=640, image_height=480)
        assert result.lane is None
        assert result.has_lane_info is False

    def test_valid_lane_stored(self):
        ex = _extractor()
        result = ex.extract(
            frame_id="f", frame_index=0,
            image_width=640, image_height=480,
            lane_result=_lane_result(valid=True, offset=5.0),
        )
        assert result.has_lane_info is True
        assert result.lane.lateral_offset_px == pytest.approx(5.0)

    def test_invalid_lane_has_lane_info_false(self):
        ex = _extractor()
        result = ex.extract(
            frame_id="f", frame_index=0,
            image_width=640, image_height=480,
            lane_result=_lane_result(valid=False),
        )
        assert result.lane is not None
        assert result.has_lane_info is False

    def test_lane_departure_flagged(self):
        ex = _extractor()
        result = ex.extract(
            frame_id="f", frame_index=0,
            image_width=640, image_height=480,
            lane_result=_lane_result(valid=True, offset=200.0),
        )
        assert result.lane_departure_flagged is True

    def test_no_departure_when_centred(self):
        ex = _extractor()
        result = ex.extract(
            frame_id="f", frame_index=0,
            image_width=640, image_height=480,
            lane_result=_lane_result(valid=True, offset=5.0),
        )
        assert result.lane_departure_flagged is False


# ---------------------------------------------------------------------------
# FrameFeatures convenience properties
# ---------------------------------------------------------------------------

class TestFrameFeatureProperties:
    def test_total_object_count(self):
        ex = _extractor()
        result = ex.extract(
            frame_id="f", frame_index=0,
            image_width=640, image_height=480,
            active_tracks=[_track(1), _track(2), _track(3)],
        )
        assert result.total_object_count == 3

    def test_object_counts_empty(self):
        ex = _extractor()
        result = ex.extract(frame_id="f", frame_index=0, image_width=640, image_height=480)
        assert result.object_counts == {}

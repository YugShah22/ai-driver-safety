from __future__ import annotations

import pytest

from ml.object_detection.schema import BoundingBox, Detection, FrameDetections
from ml.object_tracking.schema import Track
from ml.object_tracking.tracker import IOUTracker, _iou


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box(x1, y1, x2, y2) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _det(category, class_id, confidence, box, frame_id="") -> Detection:
    return Detection(category=category, class_id=class_id, confidence=confidence,
                     box=box, frame_id=frame_id)


def _fd(detections, frame_id="f") -> FrameDetections:
    return FrameDetections(frame_id=frame_id, detections=detections)


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

class TestIOU:
    def test_perfect_overlap(self):
        box = _box(0, 0, 10, 10)
        assert _iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _iou(_box(0, 0, 10, 10), _box(20, 20, 30, 30)) == pytest.approx(0.0)

    def test_half_overlap(self):
        # Two boxes sharing a 5x10 strip, each 10x10 = 50 area
        a = _box(0, 0, 10, 10)
        b = _box(5, 0, 15, 10)
        # intersection = 5x10 = 50, union = 150
        assert _iou(a, b) == pytest.approx(50 / 150)

    def test_contained_box(self):
        outer = _box(0, 0, 20, 20)   # area 400
        inner = _box(5, 5, 15, 15)   # area 100, fully inside
        # intersection = 100, union = 400
        assert _iou(outer, inner) == pytest.approx(100 / 400)

    def test_adjacent_boxes_no_overlap(self):
        assert _iou(_box(0, 0, 10, 10), _box(10, 0, 20, 10)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Track schema
# ---------------------------------------------------------------------------

class TestTrackSchema:
    def test_fields(self):
        t = Track(track_id=1, category="car", class_id=2,
                  box=_box(0, 0, 10, 10), confidence=0.9, frame_index=0)
        assert t.track_id == 1
        assert t.category == "car"
        assert t.is_active is True
        assert t.misses == 0
        assert t.hits == 1
        assert t.age == 1

    def test_history_initialised(self):
        t = Track(track_id=1, category="car", class_id=2,
                  box=_box(0, 0, 10, 10), confidence=0.9, frame_index=5)
        assert len(t.history) == 1
        assert t.history[0][0] == 5


# ---------------------------------------------------------------------------
# IOUTracker — single object
# ---------------------------------------------------------------------------

class TestTrackerSingleObject:
    def test_new_detection_creates_track(self):
        tracker = IOUTracker()
        fd = _fd([_det("car", 2, 0.9, _box(0, 0, 50, 50))])
        tracks = tracker.update(fd, frame_index=0)
        assert len(tracks) == 1
        assert tracks[0].track_id == 1
        assert tracks[0].category == "car"

    def test_detection_sets_track_id_on_detection(self):
        tracker = IOUTracker()
        det = _det("car", 2, 0.9, _box(0, 0, 50, 50))
        tracker.update(_fd([det]), frame_index=0)
        assert det.track_id == 1

    def test_same_object_across_frames_same_track_id(self):
        tracker = IOUTracker()
        # Slightly shifted box each frame — high IoU
        for i in range(4):
            fd = _fd([_det("car", 2, 0.9, _box(i, i, 50 + i, 50 + i))])
            tracks = tracker.update(fd, frame_index=i)
        assert len(tracks) == 1
        assert tracks[0].track_id == 1
        assert tracks[0].hits == 4

    def test_track_age_increments_each_frame(self):
        tracker = IOUTracker()
        for i in range(5):
            fd = _fd([_det("car", 2, 0.9, _box(0, 0, 50, 50))])
            tracker.update(fd, frame_index=i)
        assert tracker.active_tracks[0].age == 5

    def test_history_grows_with_matches(self):
        tracker = IOUTracker()
        for i in range(3):
            fd = _fd([_det("car", 2, 0.9, _box(i, i, 50 + i, 50 + i))])
            tracker.update(fd, frame_index=i)
        assert len(tracker.active_tracks[0].history) == 3


# ---------------------------------------------------------------------------
# IOUTracker — two objects
# ---------------------------------------------------------------------------

class TestTrackerTwoObjects:
    def test_two_separate_objects_get_different_ids(self):
        tracker = IOUTracker()
        dets = [
            _det("car", 2, 0.9, _box(0, 0, 50, 50)),
            _det("person", 0, 0.8, _box(200, 200, 230, 280)),
        ]
        tracks = tracker.update(_fd(dets), frame_index=0)
        ids = {t.track_id for t in tracks}
        assert len(ids) == 2

    def test_two_objects_continue_correctly(self):
        tracker = IOUTracker()
        for i in range(3):
            dets = [
                _det("car", 2, 0.9, _box(i, i, 50 + i, 50 + i)),
                _det("person", 0, 0.8, _box(200 + i, 200, 230 + i, 280)),
            ]
            tracker.update(_fd(dets), frame_index=i)

        active = tracker.active_tracks
        assert len(active) == 2
        for t in active:
            assert t.hits == 3

    def test_two_objects_have_distinct_track_ids(self):
        tracker = IOUTracker()
        dets = [
            _det("car", 2, 0.9, _box(0, 0, 50, 50)),
            _det("bus", 5, 0.7, _box(300, 0, 400, 80)),
        ]
        tracker.update(_fd(dets), frame_index=0)
        ids = [t.track_id for t in tracker.active_tracks]
        assert ids[0] != ids[1]


# ---------------------------------------------------------------------------
# Track termination / disappearance
# ---------------------------------------------------------------------------

class TestTrackerTermination:
    def test_track_terminated_after_max_misses(self):
        tracker = IOUTracker(max_misses=2)
        # Frame 0: create track
        tracker.update(_fd([_det("car", 2, 0.9, _box(0, 0, 50, 50))]), frame_index=0)
        # Frames 1-3: no detections
        for i in range(1, 4):
            tracker.update(_fd([]), frame_index=i)

        assert len(tracker.active_tracks) == 0
        assert tracker.all_tracks[0].is_active is False

    def test_track_survives_within_max_misses(self):
        tracker = IOUTracker(max_misses=3)
        tracker.update(_fd([_det("car", 2, 0.9, _box(0, 0, 50, 50))]), frame_index=0)
        # Miss for 2 frames — within limit
        tracker.update(_fd([]), frame_index=1)
        tracker.update(_fd([]), frame_index=2)
        assert len(tracker.active_tracks) == 1

    def test_miss_count_increments(self):
        tracker = IOUTracker(max_misses=5)
        tracker.update(_fd([_det("car", 2, 0.9, _box(0, 0, 50, 50))]), frame_index=0)
        tracker.update(_fd([]), frame_index=1)
        tracker.update(_fd([]), frame_index=2)
        assert tracker.active_tracks[0].misses == 2

    def test_miss_resets_on_rematch(self):
        tracker = IOUTracker(max_misses=5)
        tracker.update(_fd([_det("car", 2, 0.9, _box(0, 0, 50, 50))]), frame_index=0)
        tracker.update(_fd([]), frame_index=1)  # miss
        tracker.update(_fd([_det("car", 2, 0.8, _box(2, 2, 52, 52))]), frame_index=2)  # rematch
        assert tracker.active_tracks[0].misses == 0


# ---------------------------------------------------------------------------
# New object appears mid-sequence
# ---------------------------------------------------------------------------

class TestTrackerNewObject:
    def test_new_object_gets_new_track_id(self):
        tracker = IOUTracker()
        tracker.update(_fd([_det("car", 2, 0.9, _box(0, 0, 50, 50))]), frame_index=0)
        # New object appears far away
        tracker.update(_fd([
            _det("car", 2, 0.9, _box(0, 0, 50, 50)),
            _det("person", 0, 0.8, _box(300, 300, 330, 380)),
        ]), frame_index=1)

        ids = {t.track_id for t in tracker.active_tracks}
        assert len(ids) == 2
        assert 2 in ids  # new person track

    def test_new_object_track_id_increments(self):
        tracker = IOUTracker()
        tracker.update(_fd([_det("car", 2, 0.9, _box(0, 0, 50, 50))]), frame_index=0)
        tracker.update(_fd([
            _det("car", 2, 0.9, _box(0, 0, 50, 50)),
            _det("bus", 5, 0.7, _box(400, 0, 500, 80)),
        ]), frame_index=1)
        ids = sorted(t.track_id for t in tracker.active_tracks)
        assert ids == [1, 2]


# ---------------------------------------------------------------------------
# Class consistency
# ---------------------------------------------------------------------------

class TestTrackerClassConsistency:
    def test_category_preserved_across_frames(self):
        tracker = IOUTracker()
        for i in range(3):
            fd = _fd([_det("truck", 7, 0.85, _box(i, i, 80 + i, 60 + i))])
            tracker.update(fd, frame_index=i)
        assert tracker.active_tracks[0].category == "truck"

    def test_different_class_does_not_match(self):
        # Car and person at the same location — different class_id, no IoU match
        tracker = IOUTracker(iou_threshold=0.3)
        tracker.update(_fd([_det("car", 2, 0.9, _box(0, 0, 50, 50))]), frame_index=0)
        # Same box but different class — should NOT match, creates new track
        tracker.update(_fd([_det("person", 0, 0.9, _box(0, 0, 50, 50))]), frame_index=1)
        all_tracks = tracker.all_tracks
        assert len(all_tracks) == 2


# ---------------------------------------------------------------------------
# Tracker reset
# ---------------------------------------------------------------------------

class TestTrackerReset:
    def test_reset_clears_tracks(self):
        tracker = IOUTracker()
        tracker.update(_fd([_det("car", 2, 0.9, _box(0, 0, 50, 50))]), frame_index=0)
        tracker.reset()
        assert tracker.active_tracks == []
        assert tracker.all_tracks == []

    def test_reset_restarts_id_counter(self):
        tracker = IOUTracker()
        tracker.update(_fd([_det("car", 2, 0.9, _box(0, 0, 50, 50))]), frame_index=0)
        tracker.reset()
        tracker.update(_fd([_det("car", 2, 0.9, _box(0, 0, 50, 50))]), frame_index=0)
        assert tracker.active_tracks[0].track_id == 1

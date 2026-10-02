from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ml.feature_extraction.schema import FrameFeatures, ObjectFeature, LaneFeature
from ml.traditional_ml.schema import FEATURE_NAMES, N_FEATURES, LabeledSample, ModelEvaluation
from ml.traditional_ml.features import vectorize, build_matrix, make_labeled_samples
from ml.traditional_ml.model import RiskModel


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _frame(
    n_cars=0, n_persons=0,
    has_lane=False, offset_px=0.0, departure=False,
    scene_class_id=None,
    n_new=0, n_lost=0,
) -> FrameFeatures:
    objects = []
    for _ in range(n_cars):
        objects.append(ObjectFeature(
            track_id=None, category="car", confidence=0.9,
            box_x1=0, box_y1=0, box_x2=50, box_y2=50,
            box_center_x_px=25, box_center_y_px=300,
            box_area_fraction=0.01, is_in_lower_half=True,
            track_age=None, track_hits=None,
        ))
    for _ in range(n_persons):
        objects.append(ObjectFeature(
            track_id=None, category="person", confidence=0.8,
            box_x1=0, box_y1=0, box_x2=30, box_y2=60,
            box_center_x_px=15, box_center_y_px=100,
            box_area_fraction=0.005, is_in_lower_half=False,
            track_age=None, track_hits=None,
        ))
    counts = {}
    if n_cars:    counts["car"]    = n_cars
    if n_persons: counts["person"] = n_persons

    lane = None
    if has_lane:
        lane = LaneFeature(
            is_valid=True, status="ok",
            lane_center_x_px=320.0 - offset_px,
            image_center_x_px=320.0,
            left_boundary_x_px=100.0,
            right_boundary_x_px=540.0,
            lateral_offset_px=offset_px,
            lane_departure_indicator=departure,
            road_pixel_fraction=0.3,
        )

    return FrameFeatures(
        frame_id="f", frame_index=0,
        timestamp_s=None,
        image_width=640, image_height=480,
        scene_class=None, scene_class_id=scene_class_id,
        objects=objects, object_counts=counts, lane=lane,
        new_track_ids=list(range(n_new)),
        lost_track_ids=list(range(n_lost)),
    )


def _make_dataset(n=60, n_classes=2, seed=0):
    """Synthetic training set: alternating labels with structured features."""
    rng = np.random.default_rng(seed)
    frames, labels = [], []
    for i in range(n):
        lbl = i % n_classes
        frames.append(_frame(
            n_cars=lbl * 2,
            n_persons=lbl,
            has_lane=(lbl == 0),
            offset_px=float(lbl * 30),
            scene_class_id=lbl,
        ))
        labels.append(lbl)
    return frames, labels


# ---------------------------------------------------------------------------
# FEATURE_NAMES / schema
# ---------------------------------------------------------------------------

class TestFeatureNames:
    def test_length_matches_n_features(self):
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_names_are_unique(self):
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)

    def test_expected_names_present(self):
        assert "n_objects" in FEATURE_NAMES
        assert "lateral_offset_px" in FEATURE_NAMES
        assert "lane_departure" in FEATURE_NAMES
        assert "scene_class_id" in FEATURE_NAMES


# ---------------------------------------------------------------------------
# vectorize
# ---------------------------------------------------------------------------

class TestVectorize:
    def test_output_shape(self):
        vec = vectorize(_frame())
        assert vec.shape == (N_FEATURES,)

    def test_output_dtype_float32(self):
        assert vectorize(_frame()).dtype == np.float32

    def test_empty_frame_defaults(self):
        vec = vectorize(_frame())
        assert vec[FEATURE_NAMES.index("n_objects")] == pytest.approx(0.0)
        assert vec[FEATURE_NAMES.index("has_lane_info")] == pytest.approx(0.0)
        assert vec[FEATURE_NAMES.index("scene_class_id")] == pytest.approx(-1.0)

    def test_car_count(self):
        vec = vectorize(_frame(n_cars=3))
        assert vec[FEATURE_NAMES.index("n_cars")] == pytest.approx(3.0)

    def test_person_count(self):
        vec = vectorize(_frame(n_persons=2))
        assert vec[FEATURE_NAMES.index("n_persons")] == pytest.approx(2.0)

    def test_n_objects_total(self):
        vec = vectorize(_frame(n_cars=2, n_persons=1))
        assert vec[FEATURE_NAMES.index("n_objects")] == pytest.approx(3.0)

    def test_has_lane_info(self):
        vec = vectorize(_frame(has_lane=True))
        assert vec[FEATURE_NAMES.index("has_lane_info")] == pytest.approx(1.0)

    def test_lateral_offset_stored(self):
        vec = vectorize(_frame(has_lane=True, offset_px=42.5))
        assert vec[FEATURE_NAMES.index("lateral_offset_px")] == pytest.approx(42.5)

    def test_lateral_offset_zero_when_no_lane(self):
        vec = vectorize(_frame(has_lane=False))
        assert vec[FEATURE_NAMES.index("lateral_offset_px")] == pytest.approx(0.0)

    def test_lane_departure_flag(self):
        vec = vectorize(_frame(has_lane=True, departure=True))
        assert vec[FEATURE_NAMES.index("lane_departure")] == pytest.approx(1.0)

    def test_no_departure(self):
        vec = vectorize(_frame(has_lane=True, departure=False))
        assert vec[FEATURE_NAMES.index("lane_departure")] == pytest.approx(0.0)

    def test_scene_class_id_stored(self):
        vec = vectorize(_frame(scene_class_id=3))
        assert vec[FEATURE_NAMES.index("scene_class_id")] == pytest.approx(3.0)

    def test_new_tracks_count(self):
        vec = vectorize(_frame(n_new=4))
        assert vec[FEATURE_NAMES.index("n_new_tracks")] == pytest.approx(4.0)

    def test_lost_tracks_count(self):
        vec = vectorize(_frame(n_lost=2))
        assert vec[FEATURE_NAMES.index("n_lost_tracks")] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# build_matrix
# ---------------------------------------------------------------------------

class TestBuildMatrix:
    def test_output_shapes(self):
        frames, labels = _make_dataset(n=10)
        X, y = build_matrix(frames, labels)
        assert X.shape == (10, N_FEATURES)
        assert y.shape == (10,)

    def test_label_dtype(self):
        frames, labels = _make_dataset(n=5)
        _, y = build_matrix(frames, labels)
        assert y.dtype == np.int64

    def test_mismatched_lengths_raises(self):
        frames, labels = _make_dataset(n=5)
        with pytest.raises(ValueError, match="same length"):
            build_matrix(frames, labels[:3])

    def test_feature_dtype_float32(self):
        frames, labels = _make_dataset(n=4)
        X, _ = build_matrix(frames, labels)
        assert X.dtype == np.float32


# ---------------------------------------------------------------------------
# make_labeled_samples
# ---------------------------------------------------------------------------

class TestMakeLabeledSamples:
    def test_returns_labeled_samples(self):
        frames, labels = _make_dataset(n=5)
        samples = make_labeled_samples(frames, labels)
        assert len(samples) == 5
        assert all(isinstance(s, LabeledSample) for s in samples)

    def test_label_names_attached(self):
        frames, labels = _make_dataset(n=4, n_classes=2)
        samples = make_labeled_samples(frames, labels, label_names={0: "safe", 1: "risky"})
        for s in samples:
            assert s.label_name in ("safe", "risky")

    def test_mismatched_raises(self):
        frames, labels = _make_dataset(n=5)
        with pytest.raises(ValueError):
            make_labeled_samples(frames, labels[:3])


# ---------------------------------------------------------------------------
# RiskModel — construction
# ---------------------------------------------------------------------------

class TestRiskModelConstruction:
    @pytest.mark.parametrize("kind", ["random_forest", "logistic_regression", "xgboost"])
    def test_init_all_kinds(self, kind):
        model = RiskModel(kind=kind)
        assert model.kind == kind
        assert model._is_fitted is False

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown model kind"):
            RiskModel(kind="svm")

    def test_predict_before_fit_raises(self):
        model = RiskModel()
        X = np.zeros((5, N_FEATURES), dtype=np.float32)
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(X)


# ---------------------------------------------------------------------------
# RiskModel — training + prediction
# ---------------------------------------------------------------------------

class TestRiskModelFitPredict:
    @pytest.fixture(params=["random_forest", "logistic_regression", "xgboost"])
    def fitted_model(self, request):
        frames, labels = _make_dataset(n=60)
        X, y = build_matrix(frames, labels)
        model = RiskModel(kind=request.param, random_state=0)
        model.fit(X, y, class_names=["safe", "risky"])
        return model, X, y

    def test_predict_shape(self, fitted_model):
        model, X, _ = fitted_model
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_dtype(self, fitted_model):
        model, X, _ = fitted_model
        assert model.predict(X).dtype == np.int64

    def test_predictions_within_label_range(self, fitted_model):
        model, X, y = fitted_model
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_wrong_feature_count_raises(self):
        model = RiskModel()
        frames, labels = _make_dataset(n=20)
        X, y = build_matrix(frames, labels)
        model.fit(X, y)
        bad_X = np.zeros((5, N_FEATURES - 1), dtype=np.float32)
        with pytest.raises(Exception):  # sklearn raises ValueError
            model.predict(bad_X)

    def test_fit_wrong_feature_count_raises(self):
        model = RiskModel()
        bad_X = np.zeros((10, N_FEATURES + 1), dtype=np.float32)
        y = np.zeros(10, dtype=np.int64)
        with pytest.raises(ValueError, match="features"):
            model.fit(bad_X, y)


# ---------------------------------------------------------------------------
# RiskModel — predict_proba
# ---------------------------------------------------------------------------

class TestRiskModelProba:
    @pytest.mark.parametrize("kind", ["random_forest", "logistic_regression", "xgboost"])
    def test_proba_shape(self, kind):
        frames, labels = _make_dataset(n=40)
        X, y = build_matrix(frames, labels)
        model = RiskModel(kind=kind, random_state=0).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape[0] == len(X)
        assert proba.shape[1] == 2   # 2 classes

    @pytest.mark.parametrize("kind", ["random_forest", "logistic_regression", "xgboost"])
    def test_proba_sums_to_one(self, kind):
        frames, labels = _make_dataset(n=20)
        X, y = build_matrix(frames, labels)
        model = RiskModel(kind=kind, random_state=0).fit(X, y)
        proba = model.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# RiskModel — evaluation
# ---------------------------------------------------------------------------

class TestRiskModelEvaluation:
    def test_evaluate_returns_model_evaluation(self):
        frames, labels = _make_dataset(n=40)
        X, y = build_matrix(frames, labels)
        model = RiskModel(random_state=0).fit(X, y)
        result = model.evaluate(X, y)
        assert isinstance(result, ModelEvaluation)

    def test_accuracy_in_range(self):
        frames, labels = _make_dataset(n=40)
        X, y = build_matrix(frames, labels)
        model = RiskModel(random_state=0).fit(X, y)
        result = model.evaluate(X, y)
        assert 0.0 <= result.accuracy <= 1.0

    def test_confusion_matrix_shape(self):
        frames, labels = _make_dataset(n=40, n_classes=2)
        X, y = build_matrix(frames, labels)
        model = RiskModel(random_state=0).fit(X, y)
        result = model.evaluate(X, y)
        assert result.confusion_matrix.shape == (2, 2)

    def test_n_samples_correct(self):
        frames, labels = _make_dataset(n=20)
        X, y = build_matrix(frames, labels)
        model = RiskModel(random_state=0).fit(X, y)
        result = model.evaluate(X, y)
        assert result.n_samples == 20

    def test_label_note_present(self):
        frames, labels = _make_dataset(n=20)
        X, y = build_matrix(frames, labels)
        model = RiskModel(random_state=0).fit(X, y)
        result = model.evaluate(X, y)
        assert len(result.label_note) > 0


# ---------------------------------------------------------------------------
# RiskModel — feature importances
# ---------------------------------------------------------------------------

class TestFeatureImportances:
    def test_random_forest_has_importances(self):
        frames, labels = _make_dataset(n=30)
        X, y = build_matrix(frames, labels)
        model = RiskModel(kind="random_forest", random_state=0).fit(X, y)
        imps = model.feature_importances()
        assert imps is not None
        assert imps.shape == (N_FEATURES,)
        assert np.allclose(imps.sum(), 1.0, atol=1e-5)

    def test_xgboost_has_importances(self):
        frames, labels = _make_dataset(n=30)
        X, y = build_matrix(frames, labels)
        model = RiskModel(kind="xgboost", random_state=0).fit(X, y)
        imps = model.feature_importances()
        assert imps is not None
        assert imps.shape == (N_FEATURES,)


# ---------------------------------------------------------------------------
# RiskModel — serialization
# ---------------------------------------------------------------------------

class TestRiskModelSerialization:
    @pytest.mark.parametrize("kind", ["random_forest", "logistic_regression", "xgboost"])
    def test_save_and_load(self, kind, tmp_path):
        frames, labels = _make_dataset(n=40)
        X, y = build_matrix(frames, labels)
        model = RiskModel(kind=kind, random_state=0).fit(X, y, class_names=["safe", "risky"])

        path = tmp_path / f"model_{kind}.pkl"
        model.save(path)

        loaded = RiskModel.load(path)
        assert loaded.kind == kind
        assert loaded.class_names == ["safe", "risky"]
        assert loaded._is_fitted is True

        # Predictions must match
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))

    def test_unfitted_save_raises(self, tmp_path):
        model = RiskModel()
        with pytest.raises(RuntimeError, match="fitted"):
            model.save(tmp_path / "model.pkl")

    def test_feature_names_mismatch_raises(self, tmp_path):
        frames, labels = _make_dataset(n=20)
        X, y = build_matrix(frames, labels)
        model = RiskModel(random_state=0).fit(X, y)
        path = tmp_path / "model.pkl"
        model.save(path)

        # Tamper with stored feature names
        with open(path, "rb") as f:
            data = pickle.load(f)
        data["feature_names"] = ["wrong_feature"]
        with open(path, "wb") as f:
            pickle.dump(data, f)

        with pytest.raises(ValueError, match="FEATURE_NAMES"):
            RiskModel.load(path)

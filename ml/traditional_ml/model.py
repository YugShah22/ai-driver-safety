from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .schema import FEATURE_NAMES, N_FEATURES, ModelEvaluation

logger = logging.getLogger(__name__)

ModelKind = str   # "random_forest" | "logistic_regression" | "xgboost"


def _build_estimator(kind: ModelKind, random_state: int, **kwargs):
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", None),
            random_state=random_state,
            n_jobs=-1,
        )
    if kind == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=kwargs.get("max_iter", 1000),
                random_state=random_state,
                C=kwargs.get("C", 1.0),
            )),
        ])
    if kind == "xgboost":
        return XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 4),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        )
    raise ValueError(f"Unknown model kind: {kind!r}. Choose from: random_forest, logistic_regression, xgboost")


class RiskModel:
    """
    Thin wrapper around a sklearn/XGBoost classifier for driving-risk scoring.

    The model takes a feature matrix produced by ml.traditional_ml.features.build_matrix()
    and outputs an integer risk class per frame.

    IMPORTANT: The risk class labels are defined externally by whoever annotates
    the training data. They are NOT automatically derived from pseudo-predictions
    of the detection/segmentation models. Model accuracy is only meaningful
    relative to the quality of those external labels.

    Args:
        kind:         One of "random_forest", "logistic_regression", "xgboost".
        random_state: Seed for reproducibility.
        **kwargs:     Forwarded to the underlying estimator.
    """

    def __init__(
        self,
        kind: ModelKind = "random_forest",
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self.kind = kind
        self.random_state = random_state
        self._estimator = _build_estimator(kind, random_state, **kwargs)
        self._is_fitted = False
        self.class_names: Optional[list[str]] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: Optional[list[str]] = None,
    ) -> "RiskModel":
        """Train on feature matrix X with labels y."""
        if X.shape[1] != N_FEATURES:
            raise ValueError(
                f"Expected {N_FEATURES} features (matching FEATURE_NAMES), got {X.shape[1]}."
            )
        self._estimator.fit(X, y)
        self._is_fitted = True
        self.class_names = class_names
        logger.info("RiskModel (%s) fitted on %d samples.", self.kind, len(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return integer class predictions of shape (n_samples,)."""
        self._check_fitted()
        return self._estimator.predict(X).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probability estimates of shape (n_samples, n_classes).
        Not available for all model kinds.
        """
        self._check_fitted()
        if hasattr(self._estimator, "predict_proba"):
            return self._estimator.predict_proba(X)
        raise NotImplementedError(f"{self.kind} does not support predict_proba.")

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
    ) -> ModelEvaluation:
        """Compute accuracy, weighted F1, and confusion matrix on a held-out set."""
        self._check_fitted()
        y_pred = self.predict(X)
        cm = confusion_matrix(y_true, y_pred)
        return ModelEvaluation(
            accuracy=float(accuracy_score(y_true, y_pred)),
            f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            confusion_matrix=cm,
            class_names=self.class_names,
            n_samples=len(y_true),
        )

    def feature_importances(self) -> Optional[np.ndarray]:
        """
        Return feature importances if the underlying model supports them.
        Shape: (N_FEATURES,). Returns None for logistic regression.
        """
        self._check_fitted()
        est = self._estimator
        # Unwrap sklearn Pipeline
        if hasattr(est, "named_steps"):
            est = est.named_steps.get("clf", est)
        if hasattr(est, "feature_importances_"):
            return est.feature_importances_
        return None

    def save(self, path: Union[str, Path]) -> None:
        """Serialise the fitted model to disk with pickle."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "kind": self.kind,
                "random_state": self.random_state,
                "estimator": self._estimator,
                "class_names": self.class_names,
                "feature_names": FEATURE_NAMES,
            }, f)
        logger.info("RiskModel saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RiskModel":
        """Load a previously saved RiskModel from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        stored_features = data.get("feature_names", [])
        if stored_features and stored_features != FEATURE_NAMES:
            raise ValueError(
                "Saved model was trained with a different FEATURE_NAMES. "
                "Re-train before loading."
            )
        model = cls.__new__(cls)
        model.kind = data["kind"]
        model.random_state = data["random_state"]
        model._estimator = data["estimator"]
        model.class_names = data.get("class_names")
        model._is_fitted = True
        return model

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("RiskModel must be fitted before calling this method.")

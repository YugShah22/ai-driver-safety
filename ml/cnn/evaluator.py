"""
CNN Evaluator — accuracy, precision, recall, F1-score, and confusion matrix.

Metrics are computed from actual model predictions — never fabricated.
Requires scikit-learn.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .config import CNNConfig
from .model import SceneCNN
from .dataset import FrameDataset

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Results of running the evaluator on one split.

    All metrics are computed from real model predictions.
    """
    accuracy:         float
    precision_macro:  float
    recall_macro:     float
    f1_macro:         float
    confusion_matrix: list[list[int]]   # shape: [num_classes x num_classes]
    class_names:      list[str]
    num_samples:      int


class Evaluator:
    """
    Evaluates a trained SceneCNN on a FrameDataset.

    Args:
        model:       SceneCNN in eval mode.
        config:      CNNConfig (must match the model's training config).
        class_names: Ordered list of class name strings.
    """

    def __init__(
        self,
        model: SceneCNN,
        config: CNNConfig,
        class_names: Optional[list[str]] = None,
    ) -> None:
        self.model       = model
        self.config      = config
        self.device      = torch.device(config.resolve_device())
        self.class_names = class_names or [str(i) for i in range(config.num_classes)]

    def evaluate(self, dataset: FrameDataset) -> EvaluationResult:
        """
        Run inference over the entire dataset and compute metrics.

        Args:
            dataset: FrameDataset (val or test split).

        Returns:
            EvaluationResult populated from actual predictions.
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        all_preds:  list[int] = []
        all_labels: list[int] = []

        self.model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                logits = self.model(images)
                preds  = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        acc  = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        cm   = confusion_matrix(all_labels, all_preds,
                                labels=list(range(self.config.num_classes)))

        logger.info(
            "[Evaluator] accuracy=%.4f  precision=%.4f  recall=%.4f  F1=%.4f  "
            "(samples=%d)",
            acc, prec, rec, f1, len(all_labels),
        )

        return EvaluationResult(
            accuracy=float(acc),
            precision_macro=float(prec),
            recall_macro=float(rec),
            f1_macro=float(f1),
            confusion_matrix=cm.tolist(),
            class_names=self.class_names,
            num_samples=len(all_labels),
        )

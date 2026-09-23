"""
CNN Inference interface — accepts a single image / frame and returns predictions.

The Inferencer is dataset-agnostic: it takes a model checkpoint and a list
of class names, returning the predicted class, confidence, and full probability
distribution.  It does NOT interact with the risk engine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image

from .config import CNNConfig
from .dataset import build_eval_transforms
from .model import SceneCNN
from .trainer import Trainer

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """
    Output of a single Inferencer call.

    Attributes:
        predicted_class:  Integer index of the most-probable class.
        predicted_label:  Human-readable class name (if class_names was provided).
        confidence:       Probability of the predicted class (0-1).
        probabilities:    Full softmax probability distribution over all classes.
    """
    predicted_class:  int
    predicted_label:  str
    confidence:       float
    probabilities:    list[float]


class Inferencer:
    """
    Runs inference on individual frames using a trained SceneCNN.

    Args:
        model:       SceneCNN in eval mode.
        config:      CNNConfig (image size must match the trained model).
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
        self.transform   = build_eval_transforms(config)
        self.model.to(self.device)
        self.model.eval()

    # -------------------------------------------------------------------------

    def predict(self, image: Union[str, Path, "Image.Image"]) -> InferenceResult:
        """
        Run inference on one image.

        Args:
            image: A PIL Image, or a file path to a JPEG/PNG frame.

        Returns:
            InferenceResult with class index, label, confidence, and probabilities.
        """
        if not isinstance(image, Image.Image):
            image = Image.open(str(image)).convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)  # (1, C, H, W)

        with torch.no_grad():
            logits = self.model(tensor)            # (1, num_classes)
            probs  = F.softmax(logits, dim=1)[0]  # (num_classes,)

        pred_idx    = int(probs.argmax().item())
        confidence  = float(probs[pred_idx].item())
        pred_label  = self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx)
        prob_list   = probs.tolist()

        logger.debug(
            "[Inferencer] predicted=%s  confidence=%.4f",
            pred_label, confidence,
        )

        return InferenceResult(
            predicted_class=pred_idx,
            predicted_label=pred_label,
            confidence=confidence,
            probabilities=prob_list,
        )

    # -------------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        config: CNNConfig,
        class_names: Optional[list[str]] = None,
    ) -> "Inferencer":
        """
        Factory method: load a checkpoint and return a ready Inferencer.

        Args:
            checkpoint_path: Path to a .pt file saved by Trainer.save_checkpoint().
            config:          CNNConfig that matches the checkpoint's num_classes.
            class_names:     Optional class name list.
        """
        model = Trainer.load_checkpoint(checkpoint_path, config)
        return cls(model, config, class_names=class_names)

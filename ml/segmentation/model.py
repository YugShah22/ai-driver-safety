from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    LRASPP_MobileNet_V3_Large_Weights,
)

from .schema import SegmentationResult

logger = logging.getLogger(__name__)


def _preprocess(image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
    """
    Convert PIL image or tensor to a normalised NCHW float tensor.
    Uses ImageNet statistics — same as the existing CNN pipeline.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if isinstance(image, Image.Image):
        t = F.to_tensor(image.convert("RGB"))
    else:
        t = image.float()
        if t.max() > 1.0:
            t = t / 255.0
        if t.dim() == 4:
            # Already NCHW
            t = F.normalize(t, mean, std)
            return t
        # CHW — add batch dim after normalise
    t = F.normalize(t, mean, std)
    return t.unsqueeze(0)   # (1, C, H, W)


class SegmentationModel:
    """
    Wraps torchvision LRASPP-MobileNetV3-Large for semantic segmentation.

    LRASPP is the lightest pretrained segmentation model in torchvision,
    runs well on CPU, and matches the project's existing MobileNetV3 backbone
    already used in Phase 6 TransferCNN.

    The model returns one argmax class per pixel. Without a verified class
    vocabulary, the resulting indices have UNKNOWN semantic meaning.

    Args:
        num_classes:  Number of output classes (default 21 = COCO).
        pretrained:   Load COCO weights (requires internet on first run).
        device:       "cpu", "cuda", or "auto".
        class_names:  Optional {class_id: label} dict. If not provided,
                      SegmentationResult.semantic_mapping_verified is False.
    """

    def __init__(
        self,
        num_classes: int = 21,
        pretrained: bool = True,
        device: str = "cpu",
        class_names: Optional[dict[int, str]] = None,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.class_names = class_names

        if pretrained and num_classes == 21:
            weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        else:
            weights = None

        self._model = lraspp_mobilenet_v3_large(
            weights=weights,
            num_classes=num_classes,
        )
        self._model.to(self.device)
        self._model.eval()

        logger.info(
            "SegmentationModel: LRASPP MobileNetV3-Large  classes=%d  device=%s  pretrained=%s",
            num_classes, device, pretrained,
        )

    def segment(
        self,
        image: Union[Image.Image, torch.Tensor],
        frame_id: str = "",
        retain_logits: bool = False,
    ) -> SegmentationResult:
        """
        Run segmentation on a single image.

        Args:
            image:         PIL Image or float/uint8 tensor (C,H,W) or (1,C,H,W).
            frame_id:      Identifier attached to the result.
            retain_logits: When True, raw logits are stored in the result.

        Returns:
            SegmentationResult with argmax mask of the same spatial size as input.
        """
        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            t = image if image.dim() == 3 else image.squeeze(0)
            h, w = int(t.shape[-2]), int(t.shape[-1])

        inp = _preprocess(image).to(self.device)

        with torch.no_grad():
            raw = self._model(inp)["out"]   # (1, C, H, W)

        logits_np = raw.squeeze(0).cpu().numpy()          # (C, H, W)
        mask_np   = logits_np.argmax(axis=0).astype(np.int64)   # (H, W)

        return SegmentationResult(
            frame_id=frame_id,
            mask=mask_np,
            width=int(w),
            height=int(h),
            num_classes=self.num_classes,
            class_names=self.class_names,
            logits=logits_np if retain_logits else None,
        )

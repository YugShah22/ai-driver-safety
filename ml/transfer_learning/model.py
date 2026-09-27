import logging
from typing import Literal

import torch
import torch.nn as nn
from torchvision import models

from ml.cnn.config import CNNConfig

logger = logging.getLogger(__name__)

# Supported pre-trained models from torchvision
BackboneName = Literal[
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "mobilenet_v3_small",
]

_DEFAULT_BACKBONE: BackboneName = "resnet18"


def _probe_features(backbone: nn.Module) -> int:
    """Run a dummy forward pass to find the output feature dimension."""
    backbone.eval()
    with torch.no_grad():
        out = backbone(torch.zeros(1, 3, 224, 224))
    return out.shape[1] if out.dim() == 2 else int(out.flatten(1).shape[1])


def _build_backbone(name: BackboneName, pretrained: bool) -> tuple[nn.Module, int]:
    """Loads a torchvision backbone and removes its classification head."""
    
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, in_features

    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, in_features

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        backbone.classifier = nn.Identity()
        in_features = _probe_features(backbone)
        return backbone, in_features

    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        backbone.classifier = nn.Identity()
        in_features = _probe_features(backbone)
        return backbone, in_features

    raise ValueError(f"Unsupported backbone '{name}'")


class TransferCNN(nn.Module):
    """
    CNN that uses a pre-trained backbone with a custom classification head.
    Supports freezing the backbone for two-stage fine-tuning.
    """
    
    def __init__(
        self,
        config: CNNConfig,
        backbone_name: BackboneName = _DEFAULT_BACKBONE,
        pretrained: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.config = config

        self.backbone, in_features = _build_backbone(backbone_name, pretrained)
        self._backbone_name = backbone_name
        
        # New classification head matched to our specific number of classes
        self.head = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(in_features, config.num_classes),
        )

        if freeze:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Lock backbone weights to train only the classification head."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unlock all weights for full network fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

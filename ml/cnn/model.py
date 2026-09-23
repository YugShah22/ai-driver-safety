"""
SceneCNN — a compact, configurable CNN for image classification.

Architecture:
  * Feature extractor: 3 convolutional blocks (Conv -> BN -> ReLU -> MaxPool)
  * Adaptive average pooling -> flatten
  * Classification head: Linear -> ReLU -> Dropout -> Linear(num_classes)

The number of output classes and all major dimensions come from CNNConfig.
No dataset-specific knowledge is baked in.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import CNNConfig


class _ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SceneCNN(nn.Module):
    """
    Compact CNN for scene-level image classification.

    Args:
        config: CNNConfig instance.  The model reads num_classes and dropout.

    Example::

        cfg   = CNNConfig(num_classes=4)
        model = SceneCNN(cfg)
        out   = model(torch.randn(1, 3, 224, 224))   # shape: (1, 4)
    """

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        self.config = config

        # Feature extractor
        self.features = nn.Sequential(
            _ConvBlock(3,   32,  3, pool=True),   # 224->112
            _ConvBlock(32,  64,  3, pool=True),   # 112->56
            _ConvBlock(64,  128, 3, pool=True),   # 56->28
            _ConvBlock(128, 256, 3, pool=True),   # 28->14
        )

        # Pooling
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # always 256x4x4 = 4096

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (B, 3, H, W).

        Returns:
            Raw logits of shape (B, num_classes).
        """
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

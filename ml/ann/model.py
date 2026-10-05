from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from .config import ANNConfig


class RiskANN(nn.Module):
    """
    Feed-forward neural network for driving-risk classification from tabular features.

    Architecture: Linear → BN → ReLU → Dropout, repeated for each hidden layer,
    followed by a final Linear projection to num_classes logits.

    The model accepts a flat float tensor of shape (B, input_dim) and returns
    raw logits of shape (B, num_classes). Apply softmax or argmax downstream
    for probabilities or class predictions.

    Args:
        config: ANNConfig controlling all architectural hyperparameters.
    """

    def __init__(self, config: ANNConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_dim = config.input_dim

        for hidden in config.hidden_dims:
            layers += [
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=config.dropout),
            ]
            in_dim = hidden

        layers.append(nn.Linear(in_dim, config.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor of shape (B, input_dim).
        Returns:
            Logits of shape (B, num_classes).
        """
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return integer class predictions of shape (B,)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
            "input_dim":   self.config.input_dim,
            "hidden_dims": self.config.hidden_dims,
            "num_classes": self.config.num_classes,
        }, path)

    @classmethod
    def load(cls, path: Union[str, Path], config: Optional[ANNConfig] = None) -> "RiskANN":
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        if config is None:
            config = ANNConfig(
                input_dim=ckpt["input_dim"],
                hidden_dims=tuple(ckpt["hidden_dims"]),
                num_classes=ckpt["num_classes"],
            )
        model = cls(config)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model

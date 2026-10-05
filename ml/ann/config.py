from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ANNConfig:
    """
    Configuration for the feed-forward ANN risk predictor.

    This mirrors the CNNConfig style but contains only fields relevant
    to tabular feature vectors — no image dimensions or DataLoader settings.

    Attributes:
        input_dim:      Number of input features. Should equal
                        ml.traditional_ml.schema.N_FEATURES (18) unless
                        the feature set has been extended.
        hidden_dims:    Sequence of hidden layer sizes.
        num_classes:    Number of output classes.
        dropout:        Dropout rate applied after each hidden layer.
        learning_rate:  Initial learning rate.
        weight_decay:   L2 regularisation coefficient.
        epochs:         Training epochs.
        batch_size:     Mini-batch size.
        optimizer:      "adam" or "sgd".
        device:         "cpu", "cuda", or "auto".
        checkpoint_dir: Directory for saved checkpoints.
        best_ckpt_name: Filename for best-val-loss checkpoint.
        seed:           Random seed for reproducibility.
    """
    input_dim: int = 18
    hidden_dims: tuple[int, ...] = (64, 32)
    num_classes: int = 2

    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 32

    optimizer: Literal["adam", "sgd"] = "adam"
    device: Literal["cpu", "cuda", "auto"] = "auto"

    checkpoint_dir: str = "checkpoints/ann"
    best_ckpt_name: str = "ann_best.pt"
    seed: int = 42

    def resolve_device(self) -> str:
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

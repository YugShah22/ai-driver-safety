"""
CNN training / inference configuration.

All hyperparameters are stored here.  Nothing is hard-coded in model or trainer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class CNNConfig:
    """
    Complete configuration for CNN training and inference.

    Attributes:
        num_classes:    Number of output classes.  Set from DatasetAdapter.num_classes().
        image_size:     (height, width) that frames are resized to before the network.
        batch_size:     Mini-batch size for training and evaluation.
        learning_rate:  Initial learning rate.
        epochs:         Maximum training epochs.
        optimizer:      Optimizer name — "adam" or "sgd".
        loss:           Loss function — "cross_entropy" or "label_smoothing".
        label_smoothing:Smoothing factor used when loss == "label_smoothing".
        weight_decay:   L2 regularisation coefficient.
        device:         "cpu", "cuda", or "auto" (picks CUDA if available).
        checkpoint_dir: Directory where model checkpoints are saved.
        best_ckpt_name: Filename for the best validation checkpoint.
        seed:           Random seed for reproducibility.
        dropout:        Dropout rate applied in the classification head.
        num_workers:    DataLoader worker processes (0 = main process only).
    """

    # Dataset
    num_classes: int = 2  # must be overridden from DatasetAdapter.num_classes()

    # Image
    image_size: tuple[int, int] = (224, 224)  # (H, W)

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 20
    weight_decay: float = 1e-4
    dropout: float = 0.3

    # Optimizer & Loss
    optimizer: Literal["adam", "sgd"] = "adam"
    loss: Literal["cross_entropy", "label_smoothing"] = "cross_entropy"
    label_smoothing: float = 0.1

    # Device
    device: Literal["cpu", "cuda", "auto"] = "auto"

    # IO
    checkpoint_dir: str = "checkpoints"
    best_ckpt_name: str = "best_model.pt"
    seed: int = 42

    # Data loading
    num_workers: int = 0  # 0 is safe on all platforms in the early phases

    def resolve_device(self) -> str:
        """Return the actual device string to use (resolves 'auto')."""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

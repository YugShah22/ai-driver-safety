"""
CNN module — Convolutional Neural Network for scene classification.

Modules:
  model     — Architecture definition (flexible backbone + head)
  config    — Training/inference hyperparameter dataclass
  dataset   — PyTorch Dataset wrapper around DatasetAdapter
  trainer   — Training loop with validation and checkpointing
  evaluator — Precision / recall / F1 / confusion matrix
  infer     — Single-image inference interface

No dataset-specific code lives here.  The adapter is injected at runtime.
"""

from .config import CNNConfig
from .model import SceneCNN
from .infer import Inferencer

__all__ = ["CNNConfig", "SceneCNN", "Inferencer"]

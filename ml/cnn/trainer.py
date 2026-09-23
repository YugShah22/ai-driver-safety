"""
CNN Trainer — clean training loop with validation, checkpointing, and best-model selection.

No dataset-specific logic lives here.  The trainer consumes FrameDataset
objects and a SceneCNN model built from CNNConfig.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .config import CNNConfig
from .model import SceneCNN
from .dataset import FrameDataset

logger = logging.getLogger(__name__)


@torch.no_grad()
def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def _build_optimizer(model: SceneCNN, cfg: CNNConfig) -> torch.optim.Optimizer:
    params = model.parameters()
    if cfg.optimizer == "sgd":
        return SGD(params, lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay)
    return Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def _build_criterion(cfg: CNNConfig) -> nn.Module:
    if cfg.loss == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    return nn.CrossEntropyLoss()


class TrainingResult:
    """Holds the per-epoch history returned by Trainer.train()."""

    def __init__(self) -> None:
        self.train_loss: list[float] = []
        self.val_loss:   list[float] = []
        self.train_acc:  list[float] = []
        self.val_acc:    list[float] = []

    def best_val_acc(self) -> float:
        return max(self.val_acc) if self.val_acc else 0.0

    def best_epoch(self) -> int:
        if not self.val_acc:
            return -1
        return int(torch.tensor(self.val_acc).argmax().item())


class Trainer:
    """
    Manages the training loop for SceneCNN.

    Args:
        model:       Instantiated SceneCNN.
        config:      CNNConfig controlling training hyperparameters.
        train_ds:    FrameDataset for the training split.
        val_ds:      FrameDataset for the validation split.
    """

    def __init__(
        self,
        model: SceneCNN,
        config: CNNConfig,
        train_ds: FrameDataset,
        val_ds: FrameDataset,
    ) -> None:
        self.model   = model
        self.config  = config
        self.device  = torch.device(config.resolve_device())
        self.model.to(self.device)

        self.criterion = _build_criterion(config)
        self.optimizer = _build_optimizer(model, config)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        self.ckpt_dir = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._best_val_loss = float("inf")

    # -------------------------------------------------------------------------

    def train(self) -> TrainingResult:
        """
        Run the full training loop for config.epochs epochs.

        Returns a TrainingResult with per-epoch metrics recorded
        from actual model output — never fabricated.
        """
        result = TrainingResult()
        torch.manual_seed(self.config.seed)

        for epoch in range(1, self.config.epochs + 1):
            t_loss, t_acc = self._train_epoch()
            v_loss, v_acc = self._val_epoch()
            self.scheduler.step(v_loss)

            result.train_loss.append(t_loss)
            result.val_loss.append(v_loss)
            result.train_acc.append(t_acc)
            result.val_acc.append(v_acc)

            logger.info(
                "Epoch %d/%d  train_loss=%.4f  train_acc=%.4f  "
                "val_loss=%.4f  val_acc=%.4f",
                epoch, self.config.epochs, t_loss, t_acc, v_loss, v_acc,
            )

            # Save checkpoint every epoch
            self.save_checkpoint(f"epoch_{epoch:03d}.pt")

            # Save best model
            if v_loss < self._best_val_loss:
                self._best_val_loss = v_loss
                self.save_checkpoint(self.config.best_ckpt_name)
                logger.info(
                    "  -> New best val_loss=%.4f — saved %s",
                    v_loss, self.config.best_ckpt_name,
                )

        return result

    # -------------------------------------------------------------------------

    def _train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_acc  = 0.0
        n_batches  = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_acc  += _accuracy(logits, labels)
            n_batches  += 1

        if n_batches == 0:
            return 0.0, 0.0
        return total_loss / n_batches, total_acc / n_batches

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_acc  = 0.0
        n_batches  = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss   = self.criterion(logits, labels)

            total_loss += loss.item()
            total_acc  += _accuracy(logits, labels)
            n_batches  += 1

        if n_batches == 0:
            return 0.0, 0.0
        return total_loss / n_batches, total_acc / n_batches

    # -------------------------------------------------------------------------

    def save_checkpoint(self, name: str) -> Path:
        """Save model + optimizer state and config to disk."""
        path = self.ckpt_dir / name
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "num_classes":     self.config.num_classes,
            "image_size":      self.config.image_size,
        }, path)
        return path

    @staticmethod
    def load_checkpoint(
        path: str | Path,
        config: CNNConfig,
        device: Optional[str] = None,
    ) -> SceneCNN:
        """
        Load a SceneCNN from a checkpoint saved by save_checkpoint().

        Args:
            path:   Path to the .pt file.
            config: CNNConfig must have num_classes matching the checkpoint.
            device: Optional override; otherwise uses config.resolve_device().

        Returns:
            Loaded SceneCNN in eval mode.
        """
        dev  = device or config.resolve_device()
        ckpt = torch.load(str(path), map_location=dev, weights_only=True)

        ckpt_classes = ckpt.get("num_classes")
        if ckpt_classes is not None and ckpt_classes != config.num_classes:
            raise ValueError(
                f"Checkpoint has num_classes={ckpt_classes} "
                f"but config has num_classes={config.num_classes}."
            )

        model = SceneCNN(config)
        model.load_state_dict(ckpt["model_state"])
        model.to(dev)
        model.eval()
        return model

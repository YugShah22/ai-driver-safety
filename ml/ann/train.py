from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from .config import ANNConfig
from .model import RiskANN

logger = logging.getLogger(__name__)


class ANNTrainingResult:
    """Per-epoch training history returned by ANNTrainer.train()."""

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
        return int(np.argmax(self.val_acc))


@torch.no_grad()
def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


class ANNTrainer:
    """
    Minimal training loop for RiskANN on tabular feature matrices.

    Accepts numpy arrays directly — no DataLoader wiring required from the caller.
    Mirrors the CNN Trainer's interface (train → ANNTrainingResult,
    save_checkpoint, load_checkpoint) so Phase 12 is navigable by analogy.

    Args:
        model:    RiskANN instance.
        config:   ANNConfig controlling all training hyperparameters.
        X_train:  float32 array (n_train, input_dim).
        y_train:  int64 array  (n_train,).
        X_val:    float32 array (n_val,   input_dim).
        y_val:    int64 array  (n_val,).
    """

    def __init__(
        self,
        model: RiskANN,
        config: ANNConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        self.model  = model
        self.config = config
        self.device = torch.device(config.resolve_device())
        self.model.to(self.device)

        self.criterion    = nn.CrossEntropyLoss()
        params = model.parameters()
        if config.optimizer == "sgd":
            self.optimizer = SGD(params, lr=config.learning_rate,
                                 momentum=0.9, weight_decay=config.weight_decay)
        else:
            self.optimizer = Adam(params, lr=config.learning_rate,
                                  weight_decay=config.weight_decay)

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)

        self.train_loader = _make_loader(X_train, y_train, config.batch_size, shuffle=True)
        self.val_loader   = _make_loader(X_val,   y_val,   config.batch_size, shuffle=False)

        self.ckpt_dir = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._best_val_loss = float("inf")

    def train(self) -> ANNTrainingResult:
        """Run the full training loop. Returns per-epoch metrics from real model output."""
        result = ANNTrainingResult()
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
                "Epoch %d/%d  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f",
                epoch, self.config.epochs, t_loss, t_acc, v_loss, v_acc,
            )

            self.save_checkpoint(f"epoch_{epoch:03d}.pt")

            if v_loss < self._best_val_loss:
                self._best_val_loss = v_loss
                self.save_checkpoint(self.config.best_ckpt_name)

        return result

    def _train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss = total_acc = n = 0.0
        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_acc  += _accuracy(logits, y_batch)
            n += 1
        return (total_loss / n, total_acc / n) if n else (0.0, 0.0)

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = total_acc = n = 0.0
        for X_batch, y_batch in self.val_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            logits = self.model(X_batch)
            loss   = self.criterion(logits, y_batch)
            total_loss += loss.item()
            total_acc  += _accuracy(logits, y_batch)
            n += 1
        return (total_loss / n, total_acc / n) if n else (0.0, 0.0)

    def save_checkpoint(self, name: str) -> Path:
        path = self.ckpt_dir / name
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "input_dim":       self.config.input_dim,
            "hidden_dims":     self.config.hidden_dims,
            "num_classes":     self.config.num_classes,
        }, path)
        return path

    @staticmethod
    def load_checkpoint(path: str | Path, config: ANNConfig) -> RiskANN:
        ckpt = torch.load(str(path), map_location=config.resolve_device(), weights_only=True)
        model = RiskANN(config)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model

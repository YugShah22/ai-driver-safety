"""
ml/training/train_cnn.py — Training entry point for the Indian Road CNN.

Connects the existing components in this exact order:
    DatasetConfig
    → IndianRoadAdapter
    → load_annotations()
    → adapter.num_classes()
    → CNNConfig
    → CNNImageDataset (+ existing transforms)
    → SceneCNN
    → Trainer
    → trainer.train()

Smoke-test mode (default):
    --max-train   32     # tiny training set
    --max-val     16     # tiny validation set
    --epochs      1      # single epoch
    --device      cpu    # no GPU required

Full training example:
    python -m ml.training.train_cnn \\
        --data-root /path/to/dataset \\
        --source local \\
        --max-train 0 \\
        --max-val   0 \\
        --epochs   20 \\
        --device  auto

No Dataset classes, transforms, Trainer, or CNN model are defined here.
This file ONLY wires together existing components.
"""
from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

from ml.datasets.config import DatasetConfig
from ml.datasets.adapters.indian_road import IndianRoadAdapter
from ml.datasets.pytorch_dataset import CNNImageDataset
from ml.datasets.transforms import get_train_transforms, get_validation_transforms

from ml.cnn.config import CNNConfig
from ml.cnn.model import SceneCNN
from ml.cnn.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SceneCNN on the Indian Road Driving Dataset."
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--data-root",
        default="YOUR_DATASET_ROOT",
        help="Absolute path to the dataset root directory.",
    )
    p.add_argument(
        "--source",
        default="local",
        choices=["local", "tar", "hf_hub"],
        help="Data source: 'local' (extracted frames), 'tar' (local shards), 'hf_hub' (HuggingFace).",
    )
    p.add_argument(
        "--annot-dir",
        default="annotations",
        help="Subdirectory of data-root containing annotation JSON files.",
    )
    p.add_argument(
        "--frames-subdir",
        default="frames",
        help="Subdirectory of data-root containing extracted frame images (local/tar modes).",
    )

    # ── Smoke-test sample limits ──────────────────────────────────────────────
    p.add_argument(
        "--max-train",
        type=int,
        default=32,
        help="Max training samples.  0 = no limit (full dataset).  Default: 32 (smoke test).",
    )
    p.add_argument(
        "--max-val",
        type=int,
        default=16,
        help="Max validation samples.  0 = no limit (full dataset).  Default: 16 (smoke test).",
    )

    # ── Training hyperparameters ──────────────────────────────────────────────
    p.add_argument("--epochs",      type=int,   default=1,     help="Number of training epochs.  Default: 1.")
    p.add_argument("--batch-size",  type=int,   default=8,     help="Mini-batch size.")
    p.add_argument("--lr",          type=float, default=1e-3,  help="Initial learning rate.")
    p.add_argument("--image-size",  type=int,   default=224,   help="Square image size fed to the network.")
    p.add_argument("--device",      default="cpu",             help="'cpu', 'cuda', or 'auto'.  Default: cpu.")
    p.add_argument("--checkpoint-dir", default="checkpoints/indian_road",
                   help="Directory where checkpoints are saved.")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── 1. Dataset configuration ──────────────────────────────────────────────
    dataset_config = DatasetConfig(
        name="indian-road",
        root=Path(args.data_root),
        annot_dir=args.annot_dir,
        adapter="indian_road",
        extra={
            "source":        args.source,
            "frames_subdir": args.frames_subdir,
        },
    )

    # ── 2. Load adapter & annotations ────────────────────────────────────────
    logger.info("Loading IndianRoadAdapter from: %s", args.data_root)
    adapter = IndianRoadAdapter(dataset_config)
    adapter.load_annotations()
    logger.info("Adapter loaded.  num_classes=%d", adapter.num_classes())

    # ── 3. CNN configuration (num_classes comes from the adapter) ─────────────
    image_size = (args.image_size, args.image_size)
    cnn_config = CNNConfig(
        num_classes=adapter.num_classes(),   # must be 6 for Indian Road
        image_size=image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )
    logger.info("CNNConfig: num_classes=%d  image_size=%s  epochs=%d  device=%s",
                cnn_config.num_classes, cnn_config.image_size,
                cnn_config.epochs, cnn_config.resolve_device())

    # ── 4. Build datasets (with optional sample cap for smoke tests) ──────────
    train_iter = adapter.get_training_samples()
    val_iter   = adapter.get_validation_samples()

    if args.max_train > 0:
        logger.info("Smoke-test mode: capping training samples at %d", args.max_train)
        train_iter = itertools.islice(train_iter, args.max_train)

    if args.max_val > 0:
        logger.info("Smoke-test mode: capping validation samples at %d", args.max_val)
        val_iter = itertools.islice(val_iter, args.max_val)

    train_ds = CNNImageDataset(
        list(train_iter),
        transform=get_train_transforms(image_size),
    )
    val_ds = CNNImageDataset(
        list(val_iter),
        transform=get_validation_transforms(image_size),
    )
    logger.info("Datasets built.  train=%d  val=%d", len(train_ds), len(val_ds))

    # ── 5. Build model ────────────────────────────────────────────────────────
    model = SceneCNN(cnn_config)
    logger.info("SceneCNN created.  Output logits: %d", cnn_config.num_classes)

    # ── 6. Train via existing Trainer ─────────────────────────────────────────
    trainer = Trainer(model, cnn_config, train_ds, val_ds)
    result  = trainer.train()

    # ── 7. Summary ───────────────────────────────────────────────────────────
    logger.info("Training complete.")
    print(f"\nBest val acc : {result.best_val_acc():.4f}")
    print(f"Best epoch   : {result.best_epoch() + 1}")


if __name__ == "__main__":
    main()
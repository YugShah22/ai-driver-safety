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
    p = argparse.ArgumentParser(description="Train SceneCNN on Indian Road Dataset")
    
    # Dataset locations and parsing
    p.add_argument("--data-root", default="YOUR_DATASET_ROOT")
    p.add_argument("--source", default="local", choices=["local", "tar", "hf_hub"])
    p.add_argument("--annot-dir", default="annotations")
    p.add_argument("--frames-subdir", default="frames")

    # Sample limits (useful for quick smoke tests)
    p.add_argument("--max-train", type=int, default=32)
    p.add_argument("--max-val", type=int, default=16)

    # Standard hyperparameters
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint-dir", default="checkpoints/indian_road")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # 1. Setup the dataset config
    dataset_config = DatasetConfig(
        name="indian-road",
        root=Path(args.data_root),
        annot_dir=args.annot_dir,
        adapter="indian_road",
        extra={
            "source": args.source,
            "frames_subdir": args.frames_subdir,
        },
    )

    # 2. Parse labels from the annotation files
    logger.info(f"Loading adapter from: {args.data_root}")
    adapter = IndianRoadAdapter(dataset_config)
    adapter.load_annotations()

    # 3. Configure the CNN and trainer
    image_size = (args.image_size, args.image_size)
    cnn_config = CNNConfig(
        num_classes=adapter.num_classes(),
        image_size=image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    train_iter = adapter.get_training_samples()
    val_iter = adapter.get_validation_samples()

    # Limit dataset size if requested (e.g., for smoke testing)
    if args.max_train > 0:
        train_iter = itertools.islice(train_iter, args.max_train)
    if args.max_val > 0:
        val_iter = itertools.islice(val_iter, args.max_val)

    # 4. Create PyTorch datasets with appropriate transforms
    train_ds = CNNImageDataset(
        list(train_iter),
        transform=get_train_transforms(image_size),
    )
    val_ds = CNNImageDataset(
        list(val_iter),
        transform=get_validation_transforms(image_size),
    )

    # 5. Build model and begin training
    model = SceneCNN(cnn_config)
    trainer = Trainer(model, cnn_config, train_ds, val_ds)
    result = trainer.train()

    print(f"\nBest val acc : {result.best_val_acc():.4f}")
    print(f"Best epoch   : {result.best_epoch() + 1}")


if __name__ == "__main__":
    main()
# Data Directory

This directory holds raw and processed datasets.

> ⚠️ Raw video data and large datasets are excluded from version control.

## Structure (Phase 2+)

```
data/
├── raw/            # Original dashcam footage (never committed)
├── processed/      # Pre-processed frames and tensors
├── annotations/    # YOLO-format label files
├── splits/         # Train / val / test split manifests (CSV)
└── samples/        # Small sample clips for development & testing
```

## Dataset Sources (Phase 3+)

- BDD100K (Berkeley DeepDrive)
- nuScenes
- KITTI
- Custom dashcam footage

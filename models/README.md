# Models Directory

This directory stores trained model weights and checkpoints.

## Contents (Phase 3+)

```
models/
├── yolo/           # YOLO object detection weights (.pt)
├── cnn/            # Custom CNN checkpoints (.pth)
├── transfer/       # Fine-tuned transfer learning weights (.pth)
├── ann/            # ANN model weights (.pth)
├── xgboost/        # XGBoost booster files (.json / .ubj)
└── onnx/           # ONNX export files for production inference
```

> ⚠️ Model weights are excluded from version control via `.gitignore`.
> Use the training scripts in `/ml/training/` to reproduce models,
> or download pre-trained checkpoints from the project's model registry.

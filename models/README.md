# Models

Place ONNX model files in this directory before running the Track 2 evaluation notebook.

## Structure

Each model should be placed in its own subdirectory named after the model:

```text
models/
├── baselines/               # Official baseline models
│   ├── Baseline_NatureCNN/
│   │   └── model.onnx
│   ├── Baseline_ResNet/
│   │   └── model.onnx
│   ├── Baseline_SimpleCNN/
│   │   └── model.onnx
│   └── Baseline_Xu2023/
│       └── model.onnx
└── MyModel/                 # Example: your own model
    └── model.onnx
```

## Usage

Set `MODEL_DIR` in the User Settings cell of `02_track2_evaluation.ipynb` to point to this directory (default: `models/`).

Set `MODEL_NAME` to the subfolder name of the model you want to evaluate (e.g., `"Baseline_NatureCNN"`).

To evaluate all models in the directory at once, set `MODEL_MODE = "all_models"`.

## Model format

Models must be in ONNX format and accept a single input tensor of shape `[B, C, H, W]` (float32), where `H=86` and `W=155` (the competition frame dimensions).

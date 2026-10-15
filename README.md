# Parkinson disease image classification

This repository contains a reproducible TensorFlow notebook for a **binary image-classification experiment**. It is an educational/research artifact, not a clinical or production decision system.

## Dataset contract

The dataset is intentionally not committed because of size/licensing constraints. The notebook requires this layout:

```text
<PARKINSON_ROOT>/
├── train images/
│   ├── control/
│   └── pd/
└── test images/
    ├── control/
    └── pd/
```

Class directories are discovered and sorted by name; the same order is used for labels and reports. The notebook fails early with a clear error if the root, split directories, or exactly two matching class directories are absent.

Set `PARKINSON_ROOT` to the directory containing `train images/` and `test images/`:

```bash
export PARKINSON_ROOT=/absolute/path/to/dataset
```

If the dataset is unavailable, **no training or metric claim can be made**. Do not replace the missing data with invented or synthetic results.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PARKINSON_ROOT=/absolute/path/to/dataset
jupyter notebook Parkinson.ipynb
```

The default run uses `MobileNetV2` with `PARKINSON_WEIGHTS=none`, so it does not implicitly download ImageNet weights. Optional configuration:

```bash
PARKINSON_BACKBONE=EfficientNetB0 PARKINSON_WEIGHTS=imagenet PARKINSON_EPOCHS=5 jupyter notebook Parkinson.ipynb
```

Run Jupyter from the repository root so the default relative paths resolve as documented. Generated models are written to `artifacts/` and ignored by Git; set `PARKINSON_ARTIFACT_DIR` when a different output location is required. Use the cells from top to bottom. The notebook validates that every class has at least one supported image (`.bmp`, `.gif`, `.jpeg`, `.jpg`, or `.png`), uses the same seed for the training/validation split, keeps the test loader unshuffled, and evaluates only the checkpoint produced by the current run.

The selected backbone applies its matching `tf.keras.applications` preprocessing function. With `PARKINSON_WEIGHTS=imagenet`, the convolutional base is frozen for this baseline; with `PARKINSON_WEIGHTS=none`, it is trained from scratch. The default `none` mode avoids any implicit network download.

## Static validation

Run the dependency-light checker before committing notebook edits:

```bash
python validate_notebook.py
```

It validates notebook JSON, code-cell syntax, clean notebook state (no outputs or execution counts), configurable paths, non-stale checkpoint handling, model-specific preprocessing, the shared train/validation split seed, deterministic test ordering, and safe metric-report configuration. It does not train a model.

## Reproducibility and limitations

- `SEED=1337` seeds Python, NumPy, and TensorFlow, and TensorFlow deterministic operations are requested where supported. Exact results can still vary with hardware and library versions.
- The split is image-directory based. Patient-level independence and duplicate-person leakage cannot be verified without patient identifiers; use a group-aware split when those identifiers are available.
- The held-out test directory is not used for fitting or model selection.
- Reported metrics are produced only by running the notebook on a real, correctly structured dataset. They are not medical advice or evidence of clinical validity.

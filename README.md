# Parkinson's disease image-classification notebook

This repository contains `Parkinson.ipynb`, an exploratory TensorFlow/Keras image-classification notebook. **It is not a clinical or diagnostic tool.** No dataset, trained model, dependency lockfile, or validated performance result is included.

## Dataset setup

The notebook no longer requires Google Colab or a mounted Drive. Place the dataset in `./data`, or set `PARKINSON_DATA_DIR` to the directory containing these folders:

```text
<data-root>/
├── train images/
│   ├── control/
│   └── pd/
└── test images/
    ├── control/
    └── pd/
```

For example, before launching Jupyter from the repository root:

```bash
export PARKINSON_DATA_DIR=/path/to/dataset
jupyter notebook Parkinson.ipynb
```

The notebook uses the `control` and `pd` subdirectory names as class labels. It creates a 60%/40% training/validation split from `train images` and loads the separate `test images` directory. The validation and training subsets now use the same fixed split seed, preventing the prior mismatch that could make the subsets overlap or omit samples. Keep the test set separate from model selection and tuning.

The dataset is intentionally not bundled. Obtain it only from a source for which you have permission, and confirm its label definitions and subject-level split policy before interpreting results. If multiple images belong to one person, split by person rather than by image to avoid leakage.

## Environment and execution

The notebook imports TensorFlow/Keras, NumPy, pandas, scikit-learn, Matplotlib, seaborn, h5py, and tqdm. No dependency versions are pinned because the original runtime/version information is not available; create a Python environment with mutually compatible TensorFlow and Keras versions for your platform before running it. Image-model training may require substantial compute, disk space, and network access to fetch pretrained ImageNet weights.

Captured notebook outputs and execution counts have been cleared to avoid distributing stale metrics, plots, and dataset-derived output. **No training or performance validation was run as part of this maintenance change** because the dataset and runtime dependencies are not included. Results from any future run must be independently evaluated and are not evidence of clinical utility.

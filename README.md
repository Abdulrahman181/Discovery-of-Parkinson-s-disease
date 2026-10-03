# Parkinson's disease image experiment

This repository contains an exploratory binary image-classification workflow and a notebook wrapper. **It is not a clinical or diagnostic tool. Do not use it for patient care, screening, treatment, or medical decisions.** No dataset, trained model, validated performance result, license grant, or external validation is provided. Nothing in this repository establishes clinical utility, safety, fairness, or generalizability.

## Data and leakage controls

The dataset is deliberately not included. Use only data you are authorized to use and understand its label definitions and provenance. The supported workflow requires a CSV manifest with exactly these columns:

```csv
path,label,group_id,split
```

- `path`: normalized relative POSIX path under the data root; accepted extensions are JPEG/JFIF, PNG, and BMP.
- `label`: exactly `control` or `pd` (the latter is the positive class).
- `group_id`: a stable **anonymized** subject/group identifier used only to check split isolation. Do not use names, dates of birth, medical record numbers, or other direct identifiers.
- `split`: exactly `train`, `validation`, or `test`.

Every split must include both labels. The validator rejects duplicate image paths, unsafe/missing files, inconsistent labels within a group, and any group appearing in multiple splits. Group-level splitting is essential when images are repeated or otherwise related; an image-level random split can leak subject-specific information and produce misleading estimates. Keep the manifest and dataset private; neither identifiers nor per-image predictions are printed or included in saved model metadata.

## Setup

Python 3.10 or newer is required. The core validator uses the standard library. Install training and development dependencies in a virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,test]'
```

TensorFlow is an optional, relatively large training dependency (`tensorflow>=2.16,<3`). The project does not provide a lockfile; resolve and record the exact environment used for any research run. Training uses a small CNN initialized from scratch and does not download pretrained weights. Image size is 224×224. CPU/GPU support, performance, and deterministic behavior can vary by TensorFlow build and hardware.

## Validate, train, and evaluate

Set paths explicitly; no machine-specific or cloud-drive paths are embedded in the code:

```bash
export PARKINSON_DATA_DIR=/path/to/authorized/images
export PARKINSON_MANIFEST=/path/to/private/manifest.csv
python -m parkinson.cli validate --manifest "$PARKINSON_MANIFEST" --data-root "$PARKINSON_DATA_DIR"
```

The validation summary contains aggregate counts only. To train after reviewing the split policy:

```bash
python -m parkinson.cli train \
  --manifest "$PARKINSON_MANIFEST" \
  --data-root "$PARKINSON_DATA_DIR" \
  --output-dir ./artifacts/run-1
```

Training and early stopping use only the `train` and `validation` splits. The `test` images are not decoded or used for model selection. Training uses a fixed configurable random seed (default `1337`), deterministic dataset ordering where supported, class weighting, early stopping on validation loss, and a learning-rate reduction callback. Choose a new, private output directory outside the data root; existing model artifacts are not overwritten. The generated `.keras` file and metadata are ignored by Git. Metadata contains only the model hash, architecture/version schema, class labels, input size, threshold, and seed—not paths, groups, or individual predictions.

After freezing the model and analysis plan, run the separate holdout evaluation once:

```bash
python -m parkinson.cli evaluate \
  --manifest "$PARKINSON_MANIFEST" \
  --data-root "$PARKINSON_DATA_DIR" \
  --model-dir ./artifacts/run-1
```

Evaluation reads only the `test` images for inference and reports aggregate accuracy, sensitivity, specificity, precision, and sample count; it does not emit per-image results or save test outputs. Do not use those test results to select or tune a model. Repeated experimentation invalidates a holdout estimate. A group-disjoint split alone does not prove that acquisition, site, device, or other confounders are controlled; a trustworthy clinical study requires a prespecified protocol and appropriately independent external validation.

`Parkinson.ipynb` provides an interactive manifest-validation step and documents the same separate CLI workflow. It contains no captured execution outputs. Do not commit datasets, manifests, model files, patient-derived images, or private identifiers.

## Checks

Run the automated checks without a dataset or TensorFlow installation:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
ruff check src tests
```

Continuous integration runs these checks, validates notebook JSON/output hygiene, and uses read-only repository permissions. The included tests use temporary placeholder files only; they do not validate image decoding, training, scientific performance, or clinical claims.

## Provenance and limitations

The original repository contained a research notebook with repeated architecture experiments and a shared test-set evaluation pattern. The current notebook is a transparent wrapper around a small, auditable group-aware workflow rather than a claim that the previous models or results were reproduced. No external dataset, metrics, license, clinical review, or independent validation has been invented or inferred. Confirm permissions and provenance independently before use.

# Engineering review — Parkinson disease image classification

**Repository:** `Abdulrahman181/Discovery-of-Parkinson-s-disease`  
**Review scope:** `Parkinson.ipynb`, `README.md`, `requirements.txt`, `.gitignore`, and static validation support  
**Review date:** 2026-10-13  
**Execution policy:** local repository changes only; no GitHub push performed

## Executive summary

The original notebook was not safely runnable as committed. It contained a literal dataset path instead of the configured variable, used different random seeds for the training and validation subsets, passed callbacks as a nested list, assumed model files existed, hard-coded class names in a potentially reversed order, and trained four large backbones sequentially without a single explicit experiment contract. The notebook also lacked valid `outputs` fields for Ruff's notebook parser, and the dependency manifest included both `tensorflow` and standalone `keras` without version bounds.

The notebook has been rewritten as a smaller, deterministic baseline pipeline. It now validates the data root and class directories before importing TensorFlow, uses one shared seed and split for complementary train/validation subsets, keeps test ordering stable, discovers the class order from the directories, saves and evaluates the best validation-loss checkpoint, and reports metrics only when real data is supplied and training is actually run. Old cell outputs were removed; all code cells now have empty output arrays and no fabricated metrics.

## Findings and remediation

| Severity | Finding | Remediation |
|---|---|---|
| Critical | The original loader used literal strings such as `PARKINSON_ROOT/train images/...`; the `PARKINSON_ROOT` variable was never used. | All paths now use `Path` objects derived from `PARKINSON_ROOT`, with a clear missing-root/split error. |
| Critical | Training used `seed=1337` while validation used `seed=333` for the same directory and split fraction. This can create overlapping/non-complementary subsets and invalidate validation metrics. | Both subsets use the same `VALIDATION_SPLIT=0.20` and `SEED=1337`; test data is a separate unshuffled loader. |
| Critical | `model.fit(..., callbacks=[callbacks])` passed a list inside a list. | The rewritten notebook passes `callbacks=callbacks` as a flat list. |
| High | Multiple cells loaded `model.*.h5` files without a robust, explicit save/load contract; four heavyweight model experiments ran sequentially and reused mutable names. | One configurable baseline writes `artifacts/best_model.keras`, restores the best checkpoint, then evaluates it. The artifact directory is ignored. |
| High | Classification reports hard-coded target names in inconsistent orders (`non parkinson`/`parkinson` versus `Parkinson`/`Non Parkinson`). | Class directories are discovered, sorted, checked to match across train/test, and passed to all reports and confusion matrices in the same order. |
| High | Reproducibility was incomplete: no effective global seed setup, mutable experiment state, and inconsistent batch-size constants. | Python, NumPy, and TensorFlow seeds are set; TensorFlow deterministic operations are requested where supported; configuration is centralized. Exact hardware/version determinism remains a caveat. |
| Medium | The notebook imported both `keras` and `tensorflow.keras`, plus many unused libraries, creating compatibility and maintenance risk. | Imports were reduced to NumPy, TensorFlow, and scikit-learn; standalone `keras`, pandas, plotting, and unused model utilities are no longer required. |
| Medium | The original notebook's code cells did not consistently satisfy Ruff's notebook schema (`outputs` was missing), and it had stale training artifacts/assumptions. | The notebook was regenerated as valid nbformat 4 JSON with explicit empty outputs and no stored results. |
| Medium | Requirements were unbounded and included a separate `keras` package alongside TensorFlow. | `requirements.txt` now bounds NumPy, scikit-learn, TensorFlow, Jupyter, and ipykernel; TensorFlow's bundled Keras API is used. |
| Low | Generated models and local outputs were only partially ignored. | Added `artifacts/`, logs, pytest/Ruff caches, and existing model extensions to `.gitignore`. |

## Data and leakage assessment

The corrected pipeline prevents accidental overlap from the directory-level train/validation split by using TensorFlow's same-seed complementary subsets. The test loader is not shuffled, and the label array is collected from that same loader before predictions are generated. A length assertion prevents silent label/prediction misalignment.

This repository does not include patient identifiers. Therefore, the review **cannot verify patient-level independence**, duplicate people across folders, or leakage caused by multiple images from the same patient. If patient IDs are available, the experiment should be upgraded to a group-aware split before interpreting metrics.

## Files changed

- `Parkinson.ipynb` — replaced unsafe multi-model notebook with a portable, single-baseline pipeline; removed stale outputs and hard-coded paths/labels.
- `README.md` — documented the dataset contract, configuration, no-data limitation, execution steps, reproducibility caveats, and static validation.
- `requirements.txt` — added bounded compatible dependencies and removed the conflicting standalone Keras requirement.
- `.gitignore` — added generated artifacts and common local caches.
- `validate_notebook.py` — added dependency-light JSON/AST and invariant checks.
- `ENGINEERING_REVIEW.md` — this report.

## Validation performed

All checks were run in the Sandbox against the local working tree:

1. `python3 validate_notebook.py` — passed: `12 cells, 6 code cells`.
2. `python3 -m py_compile validate_notebook.py` — passed.
3. AST parsing of every notebook code cell — passed for all 6 code cells.
4. `python3 -m json.tool Parkinson.ipynb` — passed.
5. `ruff check Parkinson.ipynb validate_notebook.py --select E9,F63,F7,F82` — passed (`All checks passed!`).
6. Notebook inspection confirmed zero non-empty/stale outputs, two shared `validation_split=VALIDATION_SPLIT` uses, two shared `seed=SEED` uses, `shuffle=False` for test, flat `callbacks=callbacks`, and `zero_division=0` in reporting.
7. `git diff --check` — run before commit and required to pass.

## Not run / limitations

- No full training or evaluation was claimed. The repository has no `data/` directory, and this Sandbox lacks TensorFlow, scikit-learn, nbformat, and the other heavy runtime dependencies. Installing TensorFlow and downloading ImageNet weights would be unnecessary for this static audit.
- Consequently, there are no new accuracy, loss, AUC, precision, recall, or confusion-matrix results. Any results must be generated by the user after supplying the real dataset and installing dependencies.
- The notebook's optional `PARKINSON_WEIGHTS=imagenet` mode requires network access for Keras weights; the default `none` mode avoids that implicit download.

## Commit policy

A local commit is created after validation. No changes are pushed to GitHub, and no other repository is modified.

## Second Senior Review

**Review date:** 2026-10-15
**Scope:** all tracked files, the complete notebook JSON and code cells, repository history/status/diff, configuration, dependency declaration, and static validation support.

### Additional findings and fixes

| Severity | Finding | Remediation |
|---|---|---|
| High | Raw `image_dataset_from_directory` pixels were sent directly to every backbone. This violates the expected input contract for ImageNet backbones and makes the optional pretrained mode unreliable. | Added an explicit matching Keras application preprocessor for every supported backbone. ImageNet mode now freezes the convolutional base; scratch mode remains trainable. |
| High | `artifacts/best_model.keras` could be left over from a previous experiment, and the evaluation cell did not explicitly verify that the current fit produced a checkpoint. | The training cell removes the prior checkpoint before fitting; evaluation checks `best_model_path.is_file()` before loading. Output location is configurable through `PARKINSON_ARTIFACT_DIR`. |
| Medium | Empty class directories were left to fail later inside TensorFlow with a less actionable error. | Added supported-image counting for both splits and a clear early error naming each empty class. |
| Medium | The static checker did not enforce clean execution state or the newly required model/artifact contracts. | Extended `validate_notebook.py` to reject non-empty outputs/non-null execution counts and to check preprocessing, image validation, configurable artifacts, stale-checkpoint cleanup, and checkpoint existence. |

### Second-review validation

- `python3 validate_notebook.py` — passed after the hardening changes.
- `python3 -m py_compile validate_notebook.py` — passed.
- `python3 -m json.tool Parkinson.ipynb` — passed.
- AST parsing of every notebook code cell — passed.
- `git diff --check` — passed.
- Notebook inspection confirmed six code cells with `execution_count: null` and empty `outputs`, matching train/validation seeds and split fractions, unshuffled test data, model-specific preprocessing, and no fabricated metrics.
- No end-to-end training was run: the repository intentionally contains no dataset, and TensorFlow/scikit-learn are not installed in this audit environment. No accuracy, loss, AUC, or clinical claim is made.

### Remaining risks

- Patient-level independence, duplicate images, and any subject leakage remain unverifiable without patient identifiers and metadata; a group-aware split is still required where identifiers exist.
- The split remains directory/image based and is not stratified by patient. Very small or imbalanced datasets may produce unstable validation estimates.
- Exact numerical reproducibility can still vary by TensorFlow version, CPU/GPU kernels, hardware, and optional ImageNet weight downloads.
- No runtime artifact compatibility test could be performed without TensorFlow and a real dataset; checkpoint loading is guarded but not empirically exercised here.

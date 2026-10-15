#!/usr/bin/env python3
"""Static validation for the Parkinson notebook.

This intentionally does not import TensorFlow or access the dataset. It catches
malformed notebook JSON, Python syntax errors, stale literal paths, dirty
notebook state, and the important data/evaluation invariants in a lightweight
environment.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


NOTEBOOK = Path(__file__).with_name("Parkinson.ipynb")


def fail(message: str) -> None:
    raise SystemExit(f"validation failed: {message}")


def main() -> int:
    try:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read valid notebook JSON: {exc}")

    if notebook.get("nbformat", 0) < 4:
        fail("notebook must use nbformat 4 or newer")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        fail("notebook must contain cells")

    code = []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") not in {"code", "markdown", "raw"}:
            fail(f"cell {index} has an unknown cell_type")
        if cell.get("cell_type") == "code":
            if not isinstance(cell.get("outputs"), list):
                fail(f"code cell {index} has invalid outputs")
            if cell["outputs"]:
                fail(f"code cell {index} contains stale outputs")
            if cell.get("execution_count") is not None:
                fail(f"code cell {index} has a non-null execution count")
            source = "".join(cell.get("source", []))
            try:
                ast.parse(source, filename=f"Parkinson.ipynb[cell {index}]")
            except SyntaxError as exc:
                fail(f"cell {index} has Python syntax error: {exc}")
            code.append(source)

    combined = "\n\n".join(code)
    forbidden_fragments = (
        '"PARKINSON_ROOT/train images',
        "'PARKINSON_ROOT/train images",
        "model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[callbacks])",
    )
    for fragment in forbidden_fragments:
        if fragment in combined:
            fail(f"stale or unsafe fragment found: {fragment}")

    if 'os.environ.get("PARKINSON_ROOT"' not in combined:
        fail("dataset root must be configurable through PARKINSON_ROOT")
    if 'PARKINSON_ARTIFACT_DIR' not in combined:
        fail("artifact output must be configurable through PARKINSON_ARTIFACT_DIR")
    if "SUPPORTED_IMAGE_EXTENSIONS" not in combined or "empty_classes" not in combined:
        fail("train/test class directories must be checked for supported images")
    if combined.count("validation_split=VALIDATION_SPLIT") < 2:
        fail("training and validation loaders must use the same validation split")
    if combined.count("seed=SEED") < 2:
        fail("training and validation loaders must use the same seed")
    if "shuffle=False" not in combined:
        fail("test loader must disable shuffling for reproducible evaluation")
    if "callbacks=callbacks" not in combined:
        fail("fit must pass callbacks as a flat callback list")
    if "PREPROCESSORS[MODEL_NAME]" not in combined:
        fail("the selected backbone must receive its matching preprocessing function")
    if "base_model.trainable = WEIGHTS == \"none\"" not in combined:
        fail("ImageNet transfer-learning mode must freeze the convolutional base")
    if "best_model_path.unlink()" not in combined or "best_model_path.is_file()" not in combined:
        fail("checkpoint loading must not reuse a stale artifact and must verify creation")
    if "classification_report" not in combined or "zero_division=0" not in combined:
        fail("evaluation must produce a safe classification report")

    print(f"validated {NOTEBOOK.name}: {len(cells)} cells, {len(code)} code cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())

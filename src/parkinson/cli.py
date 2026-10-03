"""CLI for validating manifests and running explicitly separated experiments."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import sys
from pathlib import Path
from typing import Any

from parkinson.data import CLASSES, ImageExample, ManifestError, read_manifest, summarize_manifest

IMAGE_SIZE = (224, 224)
ARTIFACT_SCHEMA = 1


class DependencyError(RuntimeError):
    """Optional training dependency is unavailable."""


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise DependencyError(
            "TensorFlow is required; install the optional training dependencies"
        ) from exc
    return tf


def _dataset(
    tf: Any,
    examples: list[ImageExample],
    batch_size: int,
    seed: int,
    shuffle: bool,
    class_weights: tuple[float, float] | None = None,
) -> Any:
    paths = [str(example.path) for example in examples]
    labels = [example.label for example in examples]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path: Any, label: Any) -> Any:
        encoded = tf.io.read_file(path)
        image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
        label = tf.cast(label, tf.float32)
        if class_weights is None:
            return image, label
        weights = tf.constant(class_weights, dtype=tf.float32)
        sample_weight = tf.gather(weights, tf.cast(label, tf.int32))
        return image, label, sample_weight

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(examples), 4096), seed=seed, reshuffle_each_iteration=True
        )
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    return (
        dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def _build_model(tf: Any) -> Any:
    layers = tf.keras.layers
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pd_probability")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="parkinson_image_cnn_v1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def _train(args: argparse.Namespace, examples: list[ImageExample]) -> None:
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    train_examples = [item for item in examples if item.split == "train"]
    validation_examples = [item for item in examples if item.split == "validation"]
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve(strict=True)
    try:
        output_dir.relative_to(data_root)
    except ValueError:
        pass
    else:
        raise ManifestError("artifact output directory must be outside the dataset root")

    model_path = output_dir / "model.keras"
    metadata_path = output_dir / "metadata.json"
    if (
        model_path.exists()
        or model_path.is_symlink()
        or metadata_path.exists()
        or metadata_path.is_symlink()
    ):
        raise ManifestError("artifact files already exist; choose a new output directory")

    class_counts = [
        sum(item.label == index for item in train_examples) for index in range(len(CLASSES))
    ]
    class_weights = tuple(len(train_examples) / (len(CLASSES) * count) for count in class_counts)
    train_data = _dataset(
        tf, train_examples, args.batch_size, args.seed, shuffle=True, class_weights=class_weights
    )
    validation_data = _dataset(tf, validation_examples, args.batch_size, args.seed, shuffle=False)
    model = _build_model(tf)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, mode="min"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, mode="min"
        ),
    ]
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": ARTIFACT_SCHEMA,
        "model_file": "model.keras",
        "model_sha256": digest,
        "architecture": "parkinson_image_cnn_v1",
        "class_names": list(CLASSES),
        "positive_class": "pd",
        "input_size": list(IMAGE_SIZE),
        "threshold": 0.5,
        "seed": args.seed,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    best_epoch = (
        min(range(len(history.history["val_loss"])), key=history.history["val_loss"].__getitem__)
        + 1
    )
    print(
        f"Training complete. Best validation-loss epoch: {best_epoch} of {len(history.history['val_loss'])}."
    )
    print("Model and privacy-minimized metadata saved to the selected output directory.")
    print(
        "Validation results are for model development only; the holdout test split was not loaded."
    )


def _load_artifact(tf: Any, model_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    directory = Path(model_dir).expanduser().resolve(strict=True)
    model_path = directory / "model.keras"
    metadata_path = directory / "metadata.json"
    if (
        model_path.is_symlink()
        or metadata_path.is_symlink()
        or not model_path.is_file()
        or not metadata_path.is_file()
    ):
        raise ManifestError(
            "artifact directory must contain regular model.keras and metadata.json files"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError("artifact metadata is invalid") from exc
    expected_keys = {
        "schema_version",
        "model_file",
        "model_sha256",
        "architecture",
        "class_names",
        "positive_class",
        "input_size",
        "threshold",
        "seed",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_keys:
        raise ManifestError("artifact metadata schema is invalid")
    if (
        metadata.get("schema_version") != ARTIFACT_SCHEMA
        or metadata.get("model_file") != "model.keras"
        or metadata.get("architecture") != "parkinson_image_cnn_v1"
        or metadata.get("class_names") != list(CLASSES)
        or metadata.get("positive_class") != "pd"
        or metadata.get("input_size") != list(IMAGE_SIZE)
        or not isinstance(metadata.get("threshold"), (int, float))
        or not 0.0 <= metadata["threshold"] <= 1.0
        or not isinstance(metadata.get("model_sha256"), str)
    ):
        raise ManifestError("artifact metadata values are invalid")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if not hmac.compare_digest(digest, metadata["model_sha256"]):
        raise ManifestError("model checksum does not match artifact metadata")
    try:
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
    except Exception as exc:
        raise ManifestError("saved model could not be loaded safely") from exc
    return model, metadata


def _evaluate(args: argparse.Namespace, examples: list[ImageExample]) -> None:
    tf = _tensorflow()
    model, metadata = _load_artifact(tf, args.model_dir)
    test_examples = [item for item in examples if item.split == "test"]
    probabilities = model.predict(
        _dataset(tf, test_examples, args.batch_size, seed=0, shuffle=False), verbose=0
    )
    scores = [float(value) for value in probabilities.reshape(-1)]
    if len(scores) != len(test_examples) or not all(
        math.isfinite(value) and 0 <= value <= 1 for value in scores
    ):
        raise ManifestError("model returned invalid prediction scores")
    threshold = float(metadata["threshold"])
    tp = tn = fp = fn = 0
    for example, score in zip(test_examples, scores, strict=True):
        predicted = int(score >= threshold)
        if example.label == 1 and predicted == 1:
            tp += 1
        elif example.label == 0 and predicted == 0:
            tn += 1
        elif example.label == 0:
            fp += 1
        else:
            fn += 1
    count = len(test_examples)
    accuracy = (tp + tn) / count
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    print("Descriptive holdout test results (aggregate only; not clinical evidence):")
    print(
        f"n={count}; accuracy={accuracy:.4f}; sensitivity={sensitivity:.4f}; specificity={specificity:.4f}; precision={precision:.4f}"
    )
    print(
        "Do not use these results for model selection. Any tuning invalidates this holdout evaluation."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research-only Parkinson image experiment utilities."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument(
            "--manifest", required=True, help="CSV with path,label,group_id,split columns"
        )
        command.add_argument("--data-root", required=True, help="root for relative image paths")
        command.set_defaults(handler=name)
        if name == "train":
            command.add_argument(
                "--output-dir", required=True, help="new, non-data output directory"
            )
            command.add_argument("--epochs", type=int, default=30)
            command.add_argument("--batch-size", type=int, default=16)
            command.add_argument("--seed", type=int, default=1337)
        elif name == "evaluate":
            command.add_argument(
                "--model-dir",
                required=True,
                help="directory containing model.keras and metadata.json",
            )
            command.add_argument("--batch-size", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if getattr(args, "epochs", 1) < 1 or getattr(args, "batch_size", 1) < 1:
        print("error: epochs and batch size must be positive integers", file=sys.stderr)
        return 2
    if getattr(args, "seed", 0) < 0:
        print("error: seed must be a non-negative integer", file=sys.stderr)
        return 2
    try:
        examples = read_manifest(args.manifest, args.data_root)
        if args.handler == "validate":
            print(summarize_manifest(examples))
            print(
                "Group-disjoint train/validation/test splits verified; no file paths or group IDs were printed."
            )
        elif args.handler == "train":
            _train(args, examples)
        else:
            _evaluate(args, examples)
        return 0
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except DependencyError as exc:
        print(f"error: {exc}; install with: pip install -e '.[train]'", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001 -- suppress path-bearing runtime tracebacks.
        print(
            "error: operation failed; diagnostic details were suppressed to avoid disclosing local paths or filenames",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

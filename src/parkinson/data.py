"""Strict, privacy-conscious manifest handling for image experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final

CLASSES: Final[tuple[str, str]] = ("control", "pd")
SPLITS: Final[tuple[str, str, str]] = ("train", "validation", "test")
MANIFEST_COLUMNS: Final[tuple[str, str, str, str]] = (
    "path",
    "label",
    "group_id",
    "split",
)
SUPPORTED_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".jfif", ".png", ".bmp"}
)


class ManifestError(ValueError):
    """A manifest is invalid; messages deliberately omit sample paths and IDs."""


@dataclass(frozen=True, slots=True)
class ImageExample:
    """One validated manifest row. Do not log or serialize this object."""

    path: Path
    label: int
    group_id: str
    split: str


def read_manifest(manifest_path: str | Path, data_root: str | Path) -> list[ImageExample]:
    """Read and validate a CSV manifest with paths relative to ``data_root``.

    Required columns are exactly ``path,label,group_id,split``. Group IDs are
    used only in-memory to enforce leakage-free splits and are never included
    in validation summaries or model metadata.
    """
    try:
        root = Path(data_root).resolve(strict=True)
        manifest = Path(manifest_path).resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ManifestError("dataset root or manifest is unavailable") from exc
    if not root.is_dir():
        raise ManifestError("dataset root must be a directory")
    if not manifest.is_file():
        raise ManifestError("manifest must be a regular file")

    examples: list[ImageExample] = []
    seen_paths: set[Path] = set()
    group_splits: dict[str, str] = {}
    group_labels: dict[str, str] = {}
    split_class_counts = {split: {label: 0 for label in CLASSES} for split in SPLITS}

    try:
        stream = manifest.open("r", encoding="utf-8-sig", newline="")
    except (OSError, UnicodeError) as exc:
        raise ManifestError("manifest could not be read as UTF-8 CSV") from exc

    with stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ManifestError("manifest is empty") from exc
        except csv.Error as exc:
            raise ManifestError("manifest contains invalid CSV") from exc
        if tuple(header) != MANIFEST_COLUMNS:
            raise ManifestError("manifest columns must be exactly: path,label,group_id,split")

        try:
            for row_number, values in enumerate(reader, start=2):
                if len(values) != len(MANIFEST_COLUMNS):
                    raise ManifestError(f"row {row_number} must contain exactly four fields")
                raw_path, raw_label, raw_group, raw_split = values
                relative = raw_path.strip()
                label_name = raw_label.strip().lower()
                group_id = raw_group.strip()
                split = raw_split.strip().lower()

                if not relative or not group_id:
                    raise ManifestError(f"row {row_number} has a blank path or group ID")
                posix_path = PurePosixPath(relative)
                if (
                    posix_path.is_absolute()
                    or bool(PureWindowsPath(relative).drive)
                    or posix_path.as_posix() != relative
                    or "\\" in relative
                    or any(part in {"", ".", ".."} for part in posix_path.parts)
                ):
                    raise ManifestError(
                        f"row {row_number} path must be a normalized relative POSIX path"
                    )
                if label_name not in CLASSES:
                    raise ManifestError(f"row {row_number} has an unsupported class label")
                if split not in SPLITS:
                    raise ManifestError(f"row {row_number} has an unsupported split")
                if Path(relative).suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                    raise ManifestError(f"row {row_number} has an unsupported image file type")

                try:
                    image_path = (root / Path(*posix_path.parts)).resolve(strict=True)
                    image_path.relative_to(root)
                except (OSError, RuntimeError, ValueError) as exc:
                    raise ManifestError(
                        f"row {row_number} image is missing or outside the dataset root"
                    ) from exc
                if not image_path.is_file():
                    raise ManifestError(f"row {row_number} image is not a regular file")
                if image_path in seen_paths:
                    raise ManifestError(f"row {row_number} duplicates an image path")
                seen_paths.add(image_path)

                prior_split = group_splits.setdefault(group_id, split)
                if prior_split != split:
                    raise ManifestError(
                        "a group ID occurs in more than one split; regroup data before training"
                    )
                prior_label = group_labels.setdefault(group_id, label_name)
                if prior_label != label_name:
                    raise ManifestError("a group ID has inconsistent class labels")

                label = CLASSES.index(label_name)
                split_class_counts[split][label_name] += 1
                examples.append(ImageExample(image_path, label, group_id, split))
        except csv.Error as exc:
            raise ManifestError("manifest contains invalid CSV") from exc

    if not examples:
        raise ManifestError("manifest contains no image rows")
    for split in SPLITS:
        if not any(example.split == split for example in examples):
            raise ManifestError(f"manifest must contain a {split} split")
        missing = [label for label, count in split_class_counts[split].items() if count == 0]
        if missing:
            raise ManifestError(f"each split must contain examples from both classes ({split})")
    return examples


def summarize_manifest(examples: list[ImageExample]) -> str:
    """Return aggregate counts only; never include local paths or group IDs."""
    totals = {split: {label: 0 for label in CLASSES} for split in SPLITS}
    for example in examples:
        totals[example.split][CLASSES[example.label]] += 1
    chunks = [f"{split}={sum(totals[split].values())}" for split in SPLITS]
    return f"Valid manifest: {len(examples)} images; " + ", ".join(chunks)

from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from parkinson.cli import main
from parkinson.data import MANIFEST_COLUMNS, ManifestError, read_manifest, summarize_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "images"
        self.root.mkdir()
        self.manifest = Path(self.temp.name) / "manifest.csv"
        self.rows = [
            ("train/control-a.jpg", "control", "anon-control-train", "train"),
            ("train/pd-a.jpg", "pd", "anon-pd-train", "train"),
            ("validation/control-b.png", "control", "anon-control-val", "validation"),
            ("validation/pd-b.png", "pd", "anon-pd-val", "validation"),
            ("test/control-c.jpg", "control", "anon-control-test", "test"),
            ("test/pd-c.jpg", "pd", "anon-pd-test", "test"),
        ]
        for relative, _label, _group, _split in self.rows:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"placeholder")
        self._write()

    def _write(self, rows: list[tuple[str, str, str, str]] | None = None) -> None:
        with self.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(MANIFEST_COLUMNS)
            writer.writerows(self.rows if rows is None else rows)

    def test_accepts_group_disjoint_complete_splits(self) -> None:
        examples = read_manifest(self.manifest, self.root)
        self.assertEqual(len(examples), 6)
        self.assertTrue(all(example.path.is_absolute() for example in examples))
        summary = summarize_manifest(examples)
        self.assertIn("train=2", summary)
        self.assertIn("validation=2", summary)
        self.assertIn("test=2", summary)
        self.assertNotIn("anon-", summary)
        self.assertNotIn(str(self.root), summary)

    def test_rejects_group_crossing_split_boundary(self) -> None:
        rows = list(self.rows)
        rows[0] = (rows[0][0], rows[0][1], "same-subject", "train")
        rows[2] = (rows[2][0], rows[2][1], "same-subject", "validation")
        self._write(rows)
        with self.assertRaisesRegex(ManifestError, "more than one split"):
            read_manifest(self.manifest, self.root)

    def test_rejects_group_with_conflicting_labels(self) -> None:
        rows = list(self.rows)
        rows[1] = (rows[1][0], rows[1][1], rows[0][2], "train")
        self._write(rows)
        with self.assertRaisesRegex(ManifestError, "inconsistent class labels"):
            read_manifest(self.manifest, self.root)

    def test_rejects_duplicate_image(self) -> None:
        rows = list(self.rows)
        rows[-1] = (rows[0][0], rows[-1][1], rows[-1][2], rows[-1][3])
        self._write(rows)
        with self.assertRaisesRegex(ManifestError, "duplicates an image"):
            read_manifest(self.manifest, self.root)

    def test_rejects_traversal_and_absolute_paths(self) -> None:
        for unsafe in (
            "../outside.jpg",
            "/tmp/outside.jpg",
            "C:/outside.jpg",
            "nested\\outside.jpg",
            "train//control-a.jpg",
            "./train/control-a.jpg",
        ):
            rows = list(self.rows)
            rows[0] = (unsafe, rows[0][1], rows[0][2], rows[0][3])
            self._write(rows)
            with self.subTest(path=unsafe), self.assertRaises(ManifestError):
                read_manifest(self.manifest, self.root)

    def test_rejects_incomplete_classes_or_splits(self) -> None:
        self._write(self.rows[:-1])
        with self.assertRaisesRegex(ManifestError, "both classes"):
            read_manifest(self.manifest, self.root)

    def test_rejects_symlink_escape(self) -> None:
        outside = Path(self.temp.name) / "outside.jpg"
        outside.write_bytes(b"placeholder")
        link = self.root / "escape.jpg"
        link.symlink_to(outside)
        rows = list(self.rows)
        rows[0] = ("escape.jpg", rows[0][1], rows[0][2], rows[0][3])
        self._write(rows)
        with self.assertRaisesRegex(ManifestError, "outside the dataset root"):
            read_manifest(self.manifest, self.root)

    def test_rejects_extra_schema_columns(self) -> None:
        with self.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow((*MANIFEST_COLUMNS, "patient_name"))
            writer.writerow((*self.rows[0], "must-not-be-accepted"))
        with self.assertRaisesRegex(ManifestError, "columns must be exactly"):
            read_manifest(self.manifest, self.root)

    def test_validate_cli_prints_aggregate_counts_only(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(
                [
                    "validate",
                    "--manifest",
                    str(self.manifest),
                    "--data-root",
                    str(self.root),
                ]
            )
        self.assertEqual(result, 0)
        self.assertIn("train=2", output.getvalue())
        self.assertNotIn("anon-", output.getvalue())
        self.assertNotIn(str(self.root), output.getvalue())

    def test_cli_rejects_negative_seed_before_data_access(self) -> None:
        error = io.StringIO()
        with redirect_stderr(error):
            result = main(
                [
                    "train",
                    "--manifest",
                    "unused.csv",
                    "--data-root",
                    "unused-data",
                    "--output-dir",
                    "unused-artifacts",
                    "--seed",
                    "-1",
                ]
            )
        self.assertEqual(result, 2)
        self.assertIn("non-negative", error.getvalue())


if __name__ == "__main__":
    unittest.main()

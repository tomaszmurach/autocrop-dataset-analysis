"""CLI, path-safety, manifest, exit-code, and privacy tests."""

from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

from autocrop_analysis import cli


def save_image(path: Path, *, format_name: str = "PNG") -> None:
    Image.new("RGB", (10, 6), color=(20, 40, 60)).save(path, format=format_name)


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.originals = self.base / "originals"
        self.cropped = self.base / "cropped"
        self.output_directory = self.base / "private-output"
        self.originals.mkdir()
        self.cropped.mkdir()
        self.output_directory.mkdir()

    def output(self, name: str = "audit_manifest.private.json") -> Path:
        return self.output_directory / name

    def argv(self, output: Path | None = None) -> list[str]:
        return [
            "--originals",
            str(self.originals),
            "--cropped",
            str(self.cropped),
            "--output",
            str(output or self.output()),
        ]

    def test_missing_required_argument_uses_argparse_exit_two(self) -> None:
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as caught:
                cli.main([])
        self.assertEqual(caught.exception.code, 2)

    def test_missing_and_non_directory_roots_are_rejected(self) -> None:
        missing = self.base / "missing"
        with self.assertRaises(cli.ConfigurationError):
            cli.validate_paths(missing, self.cropped, self.output())

        regular_file = self.base / "file"
        regular_file.write_text("synthetic", encoding="utf-8")
        with self.assertRaises(cli.ConfigurationError):
            cli.validate_paths(regular_file, self.cropped, self.output())

    def test_equal_and_containing_roots_are_rejected(self) -> None:
        with self.assertRaises(cli.ConfigurationError):
            cli.validate_paths(self.originals, self.originals, self.output())

        nested = self.originals / "nested"
        nested.mkdir()
        with self.assertRaises(cli.ConfigurationError):
            cli.validate_paths(self.originals, nested, self.output())
        with self.assertRaises(cli.ConfigurationError):
            cli.validate_paths(nested, self.originals, self.output())

    def test_output_inside_either_root_is_rejected(self) -> None:
        with self.assertRaises(cli.ConfigurationError):
            cli.validate_paths(
                self.originals,
                self.cropped,
                self.originals / "audit.private.json",
            )
        with self.assertRaises(cli.ConfigurationError):
            cli.validate_paths(
                self.originals,
                self.cropped,
                self.cropped / "audit.private.json",
            )

    def test_output_parent_suffix_and_existing_output_are_validated(self) -> None:
        with self.assertRaises(cli.ConfigurationError):
            cli.validate_paths(
                self.originals,
                self.cropped,
                self.base / "missing" / "audit.private.json",
            )
        with self.assertRaises(cli.ConfigurationError):
            cli.validate_paths(self.originals, self.cropped, self.output("audit.json"))

        output = self.output()
        output.write_text("existing", encoding="utf-8")
        with self.assertRaises(cli.ConfigurationError):
            cli.validate_paths(self.originals, self.cropped, output)

    def test_success_creates_expected_private_manifest_and_aggregate_stdout(self) -> None:
        original_name = "synthetic-original.png"
        cropped_name = "synthetic-crop.png"
        save_image(self.originals / original_name)
        save_image(self.cropped / cropped_name)
        stdout = StringIO()
        stderr = StringIO()

        exit_code = cli.main(self.argv(), stdout=stdout, stderr=stderr)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertTrue(self.output().is_file())
        self.assertNotIn(original_name, stdout.getvalue())
        self.assertNotIn(cropped_name, stdout.getvalue())
        self.assertNotIn(str(self.originals.resolve()), stdout.getvalue())
        self.assertNotIn(str(self.cropped.resolve()), stdout.getvalue())
        manifest = json.loads(self.output().read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], cli.SCHEMA_VERSION)
        self.assertEqual(
            set(manifest),
            {
                "schema_version",
                "tool_version",
                "runtime",
                "roots",
                "summary",
                "items",
                "scan_issues",
                "collisions",
                "pairs",
            },
        )
        self.assertEqual(len(manifest["items"]), 2)
        self.assertEqual(manifest["summary"]["pairs"]["UNMATCHED"], 1)
        self.assertEqual(
            manifest["runtime"].keys(), {"python_version", "pillow_version"}
        )

    def test_manifest_repeats_absolute_roots_only_in_roots_section(self) -> None:
        save_image(self.originals / "photo.png")
        save_image(self.cropped / "photo.png")
        self.assertEqual(cli.main(self.argv(), stdout=StringIO(), stderr=StringIO()), 0)
        manifest = json.loads(self.output().read_text(encoding="utf-8"))

        roots = manifest.pop("roots")
        serialized_remainder = json.dumps(manifest, sort_keys=True)

        self.assertEqual(roots["ORIGINAL"], str(self.originals.resolve()))
        self.assertEqual(roots["CROPPED"], str(self.cropped.resolve()))
        self.assertNotIn(str(self.originals.resolve()), serialized_remainder)
        self.assertNotIn(str(self.cropped.resolve()), serialized_remainder)

    def test_raw_exception_messages_are_not_serialized(self) -> None:
        malformed = self.originals / "private-broken-name.jpg"
        malformed.write_bytes(b"not a jpeg")
        self.assertEqual(cli.main(self.argv(), stdout=StringIO(), stderr=StringIO()), 0)
        manifest = json.loads(self.output().read_text(encoding="utf-8"))
        record = manifest["items"][0]

        self.assertEqual(record["error_type"], "UnidentifiedImageError")
        self.assertNotIn("error_message", record)
        self.assertNotIn(str(self.originals.resolve()), json.dumps(record))

    def test_dataset_imperfections_still_exit_zero(self) -> None:
        (self.originals / "broken.jpg").write_bytes(b"broken")
        (self.cropped / "unknown.jpg").write_bytes(b"broken")
        exit_code = cli.main(self.argv(), stdout=StringIO(), stderr=StringIO())
        self.assertEqual(exit_code, 0)

    def test_configuration_failure_returns_two(self) -> None:
        stderr = StringIO()
        exit_code = cli.main(
            self.argv(self.output("not-private.json")),
            stdout=StringIO(),
            stderr=stderr,
        )
        self.assertEqual(exit_code, 2)
        self.assertIn("configuration error", stderr.getvalue())

    def test_atomic_publication_succeeds_with_complete_json(self) -> None:
        output = self.output()
        manifest = {"complete": True, "records": [1, 2, 3]}

        cli.write_manifest_atomic(output, manifest)

        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), manifest)
        self.assertEqual(list(self.output_directory.glob("*.tmp")), [])

    def test_atomic_publication_preserves_preexisting_destination(self) -> None:
        output = self.output()
        existing_bytes = b"preexisting manifest bytes"
        output.write_bytes(existing_bytes)

        with self.assertRaises(cli.OutputFailure) as caught:
            cli.write_manifest_atomic(output, {"replacement": True})

        self.assertEqual(caught.exception.error_type, "FileExistsError")
        self.assertEqual(output.read_bytes(), existing_bytes)
        self.assertEqual(list(self.output_directory.glob("*.tmp")), [])

    def test_atomic_publication_race_does_not_clobber_and_cleans_temp(self) -> None:
        output = self.output()
        competing_bytes = b"concurrently published bytes"
        real_link = os.link

        def competing_publication(source, destination):
            prepared = Path(source)
            self.assertEqual(
                json.loads(prepared.read_text(encoding="utf-8")),
                {"ours": "complete"},
            )
            Path(destination).write_bytes(competing_bytes)
            return real_link(source, destination)

        with mock.patch(
            "autocrop_analysis.cli.os.link", side_effect=competing_publication
        ):
            with self.assertRaises(cli.OutputFailure) as caught:
                cli.write_manifest_atomic(output, {"ours": "complete"})

        self.assertEqual(caught.exception.error_type, "FileExistsError")
        self.assertEqual(output.read_bytes(), competing_bytes)
        self.assertEqual(list(self.output_directory.glob("*.tmp")), [])

    def test_publication_collision_returns_three_and_cleans_temporary_file(self) -> None:
        save_image(self.originals / "photo.png")
        competing_bytes = b"concurrent CLI destination"
        real_link = os.link

        def competing_publication(source, destination):
            Path(destination).write_bytes(competing_bytes)
            return real_link(source, destination)

        with mock.patch(
            "autocrop_analysis.cli.os.link", side_effect=competing_publication
        ):
            exit_code = cli.main(self.argv(), stdout=StringIO(), stderr=StringIO())

        self.assertEqual(exit_code, 3)
        self.assertEqual(self.output().read_bytes(), competing_bytes)
        self.assertEqual(list(self.output_directory.glob("*.tmp")), [])

    def test_unexpected_failure_returns_one(self) -> None:
        with mock.patch(
            "autocrop_analysis.cli.audit_datasets", side_effect=RuntimeError("private")
        ):
            stderr = StringIO()
            exit_code = cli.main(self.argv(), stdout=StringIO(), stderr=stderr)
        self.assertEqual(exit_code, 1)
        self.assertNotIn("private", stderr.getvalue())

    def test_manifest_is_deterministic_except_for_runtime_block(self) -> None:
        save_image(self.originals / "photo.png")
        save_image(self.cropped / "photo.png")
        first_output = self.output("first.private.json")
        second_output = self.output("second.private.json")
        self.assertEqual(
            cli.main(self.argv(first_output), stdout=StringIO(), stderr=StringIO()), 0
        )
        self.assertEqual(
            cli.main(self.argv(second_output), stdout=StringIO(), stderr=StringIO()), 0
        )
        first = json.loads(first_output.read_text(encoding="utf-8"))
        second = json.loads(second_output.read_text(encoding="utf-8"))
        first.pop("runtime")
        second.pop("runtime")
        self.assertEqual(first, second)

    def test_no_source_side_artifacts_are_created(self) -> None:
        save_image(self.originals / "photo.png")
        save_image(self.cropped / "photo.png")
        originals_before = sorted(path.name for path in self.originals.iterdir())
        cropped_before = sorted(path.name for path in self.cropped.iterdir())

        self.assertEqual(cli.main(self.argv(), stdout=StringIO(), stderr=StringIO()), 0)

        self.assertEqual(
            sorted(path.name for path in self.originals.iterdir()), originals_before
        )
        self.assertEqual(
            sorted(path.name for path in self.cropped.iterdir()), cropped_before
        )

    def test_private_suffix_matches_repository_ignore_policy(self) -> None:
        ignore_text = (Path(__file__).parents[1] / ".gitignore").read_text(
            encoding="utf-8"
        )
        self.assertIn("*.private.json", ignore_text.splitlines())
        validated = cli.validate_paths(self.originals, self.cropped, self.output())
        self.assertTrue(validated.output.name.endswith(".private.json"))


if __name__ == "__main__":
    unittest.main()

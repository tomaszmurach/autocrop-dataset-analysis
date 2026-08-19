"""CLI, privacy, and path-safety tests for crop reconstruction."""

from __future__ import annotations

import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from autocrop_analysis import crop_reconstruction_cli
from autocrop_analysis.cli import OutputFailure
from test_crop_reconstruction import provenance_manifest


class CropReconstructionCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.results = self.base / "results"
        self.results.mkdir()
        self.provenance = self.base / "private-source-name.private.json"

    def write_provenance(self, raw: bytes | None = None) -> bytes:
        content = raw or (
            json.dumps(
                provenance_manifest(),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        self.provenance.write_bytes(content)
        return content

    def output(self, name: str = "reconstruction.private.json") -> Path:
        return self.results / name

    def argv(self, output: Path | None = None, provenance: Path | None = None) -> list[str]:
        return [
            "--provenance",
            str(provenance or self.provenance),
            "--output",
            str(output or self.output()),
        ]

    def run_cli(self, output: Path | None = None, provenance: Path | None = None):
        stdout = StringIO()
        stderr = StringIO()
        exit_code = crop_reconstruction_cli.main(
            self.argv(output, provenance), stdout=stdout, stderr=stderr
        )
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_valid_synthetic_reconstruction_and_atomic_publication(self) -> None:
        self.write_provenance()
        exit_code, stdout, stderr = self.run_cli()
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(
            stdout,
            "CROP_RECONSTRUCTION: items=1 reconstructed=1 not_reconstructed=0\n",
        )
        manifest = json.loads(self.output().read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "1.0")
        self.assertEqual(manifest["items"][0]["status"], "RECONSTRUCTED")

    def test_missing_input_is_configuration_error(self) -> None:
        missing = self.base / "missing.private.json"
        exit_code, _, stderr = self.run_cli(provenance=missing)
        self.assertEqual(exit_code, 2)
        self.assertEqual(stderr, "configuration error: provenance input must exist\n")

    def test_wrong_input_suffix_is_rejected(self) -> None:
        wrong = self.base / "private-name.json"
        wrong.write_text("{}", encoding="utf-8")
        exit_code, _, stderr = self.run_cli(provenance=wrong)
        self.assertEqual(exit_code, 2)
        self.assertNotIn(wrong.name, stderr)

    def test_wrong_output_suffix_is_rejected(self) -> None:
        self.write_provenance()
        output = self.results / "result.json"
        exit_code, _, stderr = self.run_cli(output=output)
        self.assertEqual(exit_code, 2)
        self.assertFalse(output.exists())
        self.assertNotIn(output.name, stderr)

    def test_input_and_output_must_differ(self) -> None:
        self.write_provenance()
        exit_code, _, stderr = self.run_cli(output=self.provenance)
        self.assertEqual(exit_code, 2)
        self.assertEqual(stderr, "configuration error: provenance input and output must differ\n")

    def test_existing_output_is_not_clobbered(self) -> None:
        self.write_provenance()
        self.output().write_text("preexisting", encoding="utf-8")
        exit_code, _, _ = self.run_cli()
        self.assertEqual(exit_code, 2)
        self.assertEqual(self.output().read_text(encoding="utf-8"), "preexisting")

    def test_missing_output_parent_is_rejected(self) -> None:
        self.write_provenance()
        output = self.base / "missing" / "result.private.json"
        exit_code, _, stderr = self.run_cli(output=output)
        self.assertEqual(exit_code, 2)
        self.assertEqual(stderr, "configuration error: output parent must exist\n")

    def test_publication_failure_is_sanitized(self) -> None:
        self.write_provenance()
        with mock.patch(
            "autocrop_analysis.crop_reconstruction_cli.write_manifest_atomic",
            side_effect=OutputFailure("SyntheticOutputFailure"),
        ):
            exit_code, stdout, stderr = self.run_cli()
        self.assertEqual(exit_code, 3)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "result output failure: SyntheticOutputFailure\n")
        self.assertFalse(self.output().exists())

    def test_malformed_json_is_sanitized(self) -> None:
        self.write_provenance(b'{"private-secret":')
        exit_code, stdout, stderr = self.run_cli()
        self.assertEqual(exit_code, 4)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "input manifest error: INVALID_JSON\n")
        self.assertFalse(self.output().exists())

    def test_invalid_utf8_is_sanitized(self) -> None:
        self.write_provenance(b"\xff\xfe")
        exit_code, _, stderr = self.run_cli()
        self.assertEqual(exit_code, 4)
        self.assertEqual(stderr, "input manifest error: INVALID_UTF8\n")

    def test_non_finite_json_constants_are_rejected(self) -> None:
        for constant in (b"NaN", b"Infinity", b"-Infinity", b"1e999"):
            with self.subTest(constant=constant):
                provenance = self.base / f"nonfinite-{len(constant)}-{constant[0]}.private.json"
                provenance.write_bytes(b'{"value":' + constant + b"}")
                output = self.results / f"nonfinite-{len(constant)}-{constant[0]}.private.json"
                exit_code, _, stderr = self.run_cli(output=output, provenance=provenance)
                self.assertEqual(exit_code, 4)
                self.assertEqual(stderr, "input manifest error: NON_FINITE_JSON_NUMBER\n")
                self.assertFalse(output.exists())

    def test_console_contains_aggregates_only(self) -> None:
        self.write_provenance()
        exit_code, stdout, stderr = self.run_cli()
        self.assertEqual(exit_code, 0)
        for private_value in (
            self.provenance.name,
            "crop.png",
            "original.png",
            str(self.provenance.resolve()),
            str(self.results.resolve()),
        ):
            self.assertNotIn(private_value, stdout)
            self.assertNotIn(private_value, stderr)

    def test_source_sha256_links_exact_input_bytes(self) -> None:
        raw = self.write_provenance()
        self.assertEqual(self.run_cli()[0], 0)
        manifest = json.loads(self.output().read_text(encoding="utf-8"))
        linkage = manifest["source_provenance_manifest"]
        self.assertEqual(linkage["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(linkage["path"], str(self.provenance.resolve()))
        serialized_without_link = json.dumps(
            {key: value for key, value in manifest.items() if key != "source_provenance_manifest"}
        )
        self.assertNotIn(str(self.provenance.resolve()), serialized_without_link)

    def test_input_manifest_bytes_are_read_once(self) -> None:
        self.write_provenance()
        real_read_bytes = Path.read_bytes
        reads: list[Path] = []

        def tracked_read_bytes(path: Path) -> bytes:
            reads.append(path)
            return real_read_bytes(path)

        with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=tracked_read_bytes):
            self.assertEqual(self.run_cli()[0], 0)
        self.assertEqual(reads, [self.provenance.resolve()])

    def test_repeated_outputs_are_byte_identical(self) -> None:
        self.write_provenance()
        first = self.output("first.private.json")
        second = self.output("second.private.json")
        self.assertEqual(self.run_cli(output=first)[0], 0)
        self.assertEqual(self.run_cli(output=second)[0], 0)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_actual_module_invocation_smoke(self) -> None:
        self.write_provenance()
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "autocrop_analysis.crop_reconstruction_cli",
                "--provenance",
                str(self.provenance),
                "--output",
                str(self.output()),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout,
            "CROP_RECONSTRUCTION: items=1 reconstructed=1 not_reconstructed=0\n",
        )


if __name__ == "__main__":
    unittest.main()

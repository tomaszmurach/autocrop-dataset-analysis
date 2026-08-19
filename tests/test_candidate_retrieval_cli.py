"""CLI, persistence, privacy, and synthetic retrieval tests."""

from __future__ import annotations

from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from autocrop_analysis import candidate_retrieval_cli
from autocrop_analysis.candidate_retrieval import QueryStatus


def structured_image(seed: int, size: tuple[int, int] = (420, 320)) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = rng.integers(15, 235, (size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(array, "RGB")
    draw = ImageDraw.Draw(image)
    for index in range(24):
        x = int(rng.integers(0, max(1, size[0] - 50)))
        y = int(rng.integers(0, max(1, size[1] - 40)))
        color = tuple(int(value) for value in rng.integers(0, 255, 3))
        draw.rectangle((x, y, x + 35, y + 28), outline=color, width=3)
        draw.line((x, y + 28, x + 35, y), fill=color, width=2)
    return image


class CandidateRetrievalCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.originals = self.base / "private-originals"
        self.cropped = self.base / "private-crops"
        self.results = self.base / "private-results"
        self.originals.mkdir()
        self.cropped.mkdir()
        self.results.mkdir()

    def index(self, name: str = "candidate-index.private.json") -> Path:
        return self.results / name

    def retrieval(self, name: str = "retrieval.private.json") -> Path:
        return self.results / name

    def build_args(self, output: Path | None = None) -> list[str]:
        return [
            "build",
            "--originals",
            str(self.originals),
            "--output",
            str(output or self.index()),
            "--original-max-descriptors",
            "64",
        ]

    def query_args(self, output: Path | None = None) -> list[str]:
        return [
            "query",
            "--index",
            str(self.index()),
            "--cropped",
            str(self.cropped),
            "--output",
            str(output or self.retrieval()),
            "--k",
            "2",
            "--query-max-descriptors",
            "48",
            "--neighbor-depth",
            "32",
        ]

    def create_retrieval_dataset(self) -> str:
        source_name = "private-source.png"
        source = structured_image(101)
        source.save(self.originals / source_name)
        structured_image(201).save(self.originals / "private-distractor-a.png")
        structured_image(202).save(self.originals / "private-distractor-b.png")
        crop = source.crop((80, 55, 350, 260))
        crop.resize((216, 164), Image.Resampling.LANCZOS).save(
            self.cropped / "private-resized.jpg", format="JPEG", quality=64
        )
        ImageEnhance.Contrast(ImageEnhance.Brightness(crop).enhance(1.06)).enhance(1.08).save(
            self.cropped / "private-luminance.png"
        )
        blurred = crop.filter(ImageFilter.GaussianBlur(0.5))
        noisy = np.asarray(blurred, dtype=np.int16)
        noise = np.random.default_rng(303).normal(0, 1.5, noisy.shape)
        Image.fromarray(np.clip(noisy + noise, 0, 255).astype(np.uint8), "RGB").save(
            self.cropped / "private-blur-noise.png"
        )
        return source_name

    def test_end_to_end_build_query_includes_known_source_for_transformations(self) -> None:
        source_name = self.create_retrieval_dataset()
        build_stdout = StringIO()
        query_stdout = StringIO()

        self.assertEqual(
            candidate_retrieval_cli.main(
                self.build_args(), stdout=build_stdout, stderr=StringIO()
            ),
            0,
        )
        self.assertEqual(
            candidate_retrieval_cli.main(
                self.query_args(), stdout=query_stdout, stderr=StringIO()
            ),
            0,
        )

        index_manifest = json.loads(self.index().read_text(encoding="utf-8"))
        retrieval = json.loads(self.retrieval().read_text(encoding="utf-8"))
        self.assertTrue(index_manifest["summary"]["index_corpus_complete"])
        self.assertEqual(retrieval["summary"]["RETRIEVED"], 3)
        for query in retrieval["queries"]:
            candidates = [
                candidate["original"]["relative_path"]
                for candidate in query["ranked_candidates"]
            ]
            self.assertIn(source_name, candidates)
            self.assertNotIn("MATCHED", json.dumps(query))

    def test_index_binary_is_float32_compact_and_hash_linked(self) -> None:
        self.create_retrieval_dataset()
        self.assertEqual(candidate_retrieval_cli.main(self.build_args(), stdout=StringIO(), stderr=StringIO()), 0)
        manifest = json.loads(self.index().read_text(encoding="utf-8"))
        binary = candidate_retrieval_cli.descriptor_path_for_manifest(self.index())
        self.assertEqual(manifest["binary"]["dtype"], "<f4")
        self.assertEqual(manifest["binary"]["descriptor_dimension"], 128)
        self.assertEqual(binary.stat().st_size, manifest["binary"]["byte_size"])
        self.assertLessEqual(manifest["binary"]["total_descriptor_rows"], 3 * 64)
        self.assertTrue(binary.name.endswith(".descriptors.private.f32"))

    def test_repeated_builds_preserve_binary_and_logical_manifest(self) -> None:
        self.create_retrieval_dataset()
        first = self.index("first.private.json")
        second = self.index("second.private.json")
        self.assertEqual(candidate_retrieval_cli.main(self.build_args(first), stdout=StringIO(), stderr=StringIO()), 0)
        self.assertEqual(candidate_retrieval_cli.main(self.build_args(second), stdout=StringIO(), stderr=StringIO()), 0)
        first_binary = candidate_retrieval_cli.descriptor_path_for_manifest(first)
        second_binary = candidate_retrieval_cli.descriptor_path_for_manifest(second)
        self.assertEqual(first_binary.read_bytes(), second_binary.read_bytes())
        first_manifest = json.loads(first.read_text(encoding="utf-8"))
        second_manifest = json.loads(second.read_text(encoding="utf-8"))
        first_manifest["binary"]["filename"] = "normalized"
        second_manifest["binary"]["filename"] = "normalized"
        self.assertEqual(first_manifest, second_manifest)

    def test_no_descriptor_original_marks_index_incomplete(self) -> None:
        structured_image(1).save(self.originals / "source.png")
        Image.new("RGB", (200, 150), (90, 90, 90)).save(self.originals / "blank.png")
        self.assertEqual(candidate_retrieval_cli.main(self.build_args(), stdout=StringIO(), stderr=StringIO()), 0)
        manifest = json.loads(self.index().read_text(encoding="utf-8"))
        self.assertFalse(manifest["summary"]["index_corpus_complete"])
        self.assertEqual(manifest["summary"]["NO_DESCRIPTORS"], 1)

    def test_incomplete_index_produces_explicit_query_status(self) -> None:
        source = structured_image(1)
        source.save(self.originals / "source.png")
        Image.new("RGB", (200, 150), (90, 90, 90)).save(self.originals / "blank.png")
        source.crop((70, 50, 340, 260)).save(self.cropped / "crop.png")
        self.assertEqual(candidate_retrieval_cli.main(self.build_args(), stdout=StringIO(), stderr=StringIO()), 0)
        self.assertEqual(candidate_retrieval_cli.main(self.query_args(), stdout=StringIO(), stderr=StringIO()), 0)
        result = json.loads(self.retrieval().read_text(encoding="utf-8"))["queries"][0]
        self.assertEqual(result["query_status"], QueryStatus.INDEX_INCOMPLETE.value)
        self.assertFalse(result["retrieval_query_complete"])

    def test_corrupt_query_is_explicit_and_does_not_print_filename(self) -> None:
        structured_image(1).save(self.originals / "source.png")
        private_name = "private-corrupt.jpg"
        (self.cropped / private_name).write_bytes(b"not an image")
        self.assertEqual(candidate_retrieval_cli.main(self.build_args(), stdout=StringIO(), stderr=StringIO()), 0)
        stdout = StringIO()
        self.assertEqual(candidate_retrieval_cli.main(self.query_args(), stdout=stdout, stderr=StringIO()), 0)
        result = json.loads(self.retrieval().read_text(encoding="utf-8"))["queries"][0]
        self.assertEqual(result["query_status"], QueryStatus.QUERY_UNAVAILABLE.value)
        self.assertNotIn(private_name, stdout.getvalue())

    def test_zero_descriptor_query_is_explicit(self) -> None:
        structured_image(1).save(self.originals / "source.png")
        Image.new("RGB", (200, 150), (90, 90, 90)).save(
            self.cropped / "blank.png"
        )
        self.assertEqual(candidate_retrieval_cli.main(self.build_args(), stdout=StringIO(), stderr=StringIO()), 0)
        self.assertEqual(candidate_retrieval_cli.main(self.query_args(), stdout=StringIO(), stderr=StringIO()), 0)
        result = json.loads(self.retrieval().read_text(encoding="utf-8"))["queries"][0]
        self.assertEqual(result["query_status"], QueryStatus.NO_QUERY_DESCRIPTORS.value)
        self.assertEqual(result["ranked_candidates"], [])

    def test_query_refuses_existing_output_without_modification(self) -> None:
        source = structured_image(1)
        source.save(self.originals / "source.png")
        source.crop((70, 50, 340, 260)).save(self.cropped / "crop.png")
        self.assertEqual(candidate_retrieval_cli.main(self.build_args(), stdout=StringIO(), stderr=StringIO()), 0)
        self.retrieval().write_bytes(b"existing")
        self.assertEqual(candidate_retrieval_cli.main(self.query_args(), stdout=StringIO(), stderr=StringIO()), 2)
        self.assertEqual(self.retrieval().read_bytes(), b"existing")

    def test_binary_hash_mismatch_is_rejected_with_sanitized_error(self) -> None:
        self.create_retrieval_dataset()
        self.assertEqual(candidate_retrieval_cli.main(self.build_args(), stdout=StringIO(), stderr=StringIO()), 0)
        binary = candidate_retrieval_cli.descriptor_path_for_manifest(self.index())
        with binary.open("r+b") as stream:
            stream.seek(0)
            stream.write(b"\x00\x00\x00\x00")
        stderr = StringIO()
        self.assertEqual(candidate_retrieval_cli.main(self.query_args(), stdout=StringIO(), stderr=stderr), 4)
        self.assertEqual(stderr.getvalue(), "index input error: BINARY_HASH_MISMATCH\n")

    def test_query_rejects_incompatible_index_runtime(self) -> None:
        self.create_retrieval_dataset()
        self.assertEqual(candidate_retrieval_cli.main(self.build_args(), stdout=StringIO(), stderr=StringIO()), 0)
        manifest = json.loads(self.index().read_text(encoding="utf-8"))
        manifest["runtime"]["opencv_version"] = "0.0-incompatible"
        self.index().write_text(
            json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stderr = StringIO()
        self.assertEqual(candidate_retrieval_cli.main(self.query_args(), stdout=StringIO(), stderr=stderr), 4)
        self.assertEqual(stderr.getvalue(), "index input error: INDEX_RUNTIME_MISMATCH\n")

    def test_binary_size_mismatch_is_rejected(self) -> None:
        self.create_retrieval_dataset()
        self.assertEqual(candidate_retrieval_cli.main(self.build_args(), stdout=StringIO(), stderr=StringIO()), 0)
        binary = candidate_retrieval_cli.descriptor_path_for_manifest(self.index())
        with binary.open("ab") as stream:
            stream.write(b"x")
        stderr = StringIO()
        self.assertEqual(candidate_retrieval_cli.main(self.query_args(), stdout=StringIO(), stderr=stderr), 4)
        self.assertEqual(stderr.getvalue(), "index input error: BINARY_SIZE_MISMATCH\n")

    def test_build_refuses_existing_manifest_or_binary(self) -> None:
        structured_image(1).save(self.originals / "source.png")
        self.index().write_text("existing", encoding="utf-8")
        self.assertEqual(candidate_retrieval_cli.main(self.build_args(), stdout=StringIO(), stderr=StringIO()), 2)
        self.index().unlink()
        binary = candidate_retrieval_cli.descriptor_path_for_manifest(self.index())
        binary.write_bytes(b"existing")
        self.assertEqual(candidate_retrieval_cli.main(self.build_args(), stdout=StringIO(), stderr=StringIO()), 2)
        self.assertEqual(binary.read_bytes(), b"existing")

    def test_outputs_must_be_outside_source_roots_and_private_json(self) -> None:
        structured_image(1).save(self.originals / "source.png")
        inside = self.originals / "index.private.json"
        wrong = self.results / "index.json"
        self.assertEqual(candidate_retrieval_cli.main(self.build_args(inside), stdout=StringIO(), stderr=StringIO()), 2)
        self.assertEqual(candidate_retrieval_cli.main(self.build_args(wrong), stdout=StringIO(), stderr=StringIO()), 2)

    def test_successful_console_output_is_aggregate_only(self) -> None:
        private_values = ("private-source.png", "private-crop.png")
        source = structured_image(1)
        source.save(self.originals / private_values[0])
        source.crop((70, 50, 340, 260)).save(self.cropped / private_values[1])
        build_stdout = StringIO()
        query_stdout = StringIO()
        self.assertEqual(candidate_retrieval_cli.main(self.build_args(), stdout=build_stdout, stderr=StringIO()), 0)
        self.assertEqual(candidate_retrieval_cli.main(self.query_args(), stdout=query_stdout, stderr=StringIO()), 0)
        combined = build_stdout.getvalue() + query_stdout.getvalue()
        for value in (*private_values, str(self.originals), str(self.cropped)):
            self.assertNotIn(value, combined)

    def test_manifest_failure_cleans_binary_created_by_invocation(self) -> None:
        structured_image(1).save(self.originals / "source.png")
        paths = candidate_retrieval_cli.validate_build_paths(self.originals, self.index())
        with mock.patch(
            "autocrop_analysis.candidate_retrieval_cli.write_manifest_atomic",
            side_effect=RuntimeError("synthetic"),
        ):
            with self.assertRaises(RuntimeError):
                candidate_retrieval_cli.build_index(
                    paths,
                    candidate_retrieval_cli.IndexParameters(
                        original_max_descriptors=16
                    ),
                )
        self.assertFalse(paths.binary.exists())
        self.assertFalse(paths.manifest.exists())

    def test_module_subprocess_build_and_query_smoke(self) -> None:
        source_name = self.create_retrieval_dataset()
        repository = Path(__file__).parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repository / "src")
        build = subprocess.run(
            [
                sys.executable,
                "-m",
                "autocrop_analysis.candidate_retrieval_cli",
                *self.build_args(),
            ],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(build.returncode, 0, build.stderr)
        self.assertEqual(build.stderr, "")
        query = subprocess.run(
            [
                sys.executable,
                "-m",
                "autocrop_analysis.candidate_retrieval_cli",
                *self.query_args(),
            ],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(query.returncode, 0, query.stderr)
        self.assertEqual(query.stderr, "")
        self.assertNotIn(source_name, build.stdout + query.stdout)
        retrieval = json.loads(self.retrieval().read_text(encoding="utf-8"))
        self.assertEqual(retrieval["summary"]["RETRIEVED"], 3)


if __name__ == "__main__":
    unittest.main()

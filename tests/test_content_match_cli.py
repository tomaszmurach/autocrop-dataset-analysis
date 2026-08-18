"""CLI, privacy, and output-safety tests for content provenance discovery."""

from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

from autocrop_analysis import content_match_cli, content_matching
from autocrop_analysis.content_matching import unavailable_feature_image


def structured_image(seed: int, size: tuple[int, int] = (420, 320)) -> Image.Image:
    width, height = size
    generator = np.random.default_rng(seed)
    y, x = np.indices((height, width))
    noise = generator.integers(0, 55, (height, width), dtype=np.uint8)
    array = np.empty((height, width, 3), dtype=np.uint8)
    array[:, :, 0] = (x // 2 + noise) % 256
    array[:, :, 1] = (y // 2 + np.roll(noise, 5, axis=1)) % 256
    array[:, :, 2] = ((x + y) // 3 + np.roll(noise, 7, axis=0)) % 256
    image = Image.fromarray(array, "RGB")
    draw = ImageDraw.Draw(image)
    for index in range(20):
        left = int(generator.integers(5, width - 60))
        top = int(generator.integers(5, height - 60))
        draw.rectangle((left, top, left + 35, top + 28), outline=(250, index * 9, 20), width=3)
    draw.text((145, 140), f"CLI-{seed}", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return image


class ContentMatchCliTests(unittest.TestCase):
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

    def output(self, name: str = "content-results.private.json") -> Path:
        return self.results / name

    def argv(self, output: Path | None = None) -> list[str]:
        return [
            "--originals",
            str(self.originals),
            "--cropped",
            str(self.cropped),
            "--output",
            str(output or self.output()),
        ]

    def create_dataset(self) -> tuple[str, str, str]:
        source_name = "private-source-frame.png"
        distractor_name = "private-distractor-frame.png"
        crop_name = "private-manual-crop.jpg"
        source = structured_image(11)
        source.save(self.originals / source_name)
        structured_image(12).save(self.originals / distractor_name)
        source.crop((80, 55, 350, 260)).resize((216, 164), Image.Resampling.LANCZOS).save(
            self.cropped / crop_name,
            format="JPEG",
            quality=70,
        )
        return source_name, distractor_name, crop_name

    def test_end_to_end_private_result_and_aggregate_console(self) -> None:
        source_name, distractor_name, crop_name = self.create_dataset()
        originals_before = {path.name: path.read_bytes() for path in self.originals.iterdir()}
        crops_before = {path.name: path.read_bytes() for path in self.cropped.iterdir()}
        stdout = StringIO()
        stderr = StringIO()

        exit_code = content_match_cli.main(self.argv(), stdout=stdout, stderr=stderr)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertTrue(self.output().is_file())
        for private_value in (
            source_name,
            distractor_name,
            crop_name,
            str(self.originals.resolve()),
            str(self.cropped.resolve()),
        ):
            self.assertNotIn(private_value, stdout.getvalue())
        manifest = json.loads(self.output().read_text(encoding="utf-8"))
        self.assertEqual(
            set(manifest),
            {"schema_version", "tool_version", "runtime", "algorithm", "parameters", "roots", "summary", "crops"},
        )
        self.assertEqual(manifest["summary"]["MATCHED"], 1)
        self.assertTrue(manifest["summary"]["candidate_set_complete"])
        self.assertEqual(manifest["summary"]["supplied_original_image_candidates"], 2)
        self.assertEqual(manifest["summary"]["successfully_evaluated_originals"], 2)
        crop_result = manifest["crops"][0]
        self.assertEqual(crop_result["decision"], "MATCHED")
        self.assertEqual(crop_result["provenance_interpretation"], "UNIQUE_STRONG_PROVENANCE")
        self.assertEqual(crop_result["ranked_candidates"][0]["original"]["relative_path"], source_name)
        serialized = json.dumps(manifest).casefold()
        self.assertNotIn("accuracy", serialized)
        self.assertNotIn("ground_truth", serialized)
        self.assertEqual({path.name: path.read_bytes() for path in self.originals.iterdir()}, originals_before)
        self.assertEqual({path.name: path.read_bytes() for path in self.cropped.iterdir()}, crops_before)

    def test_absolute_roots_appear_only_once_each(self) -> None:
        self.create_dataset()
        self.assertEqual(content_match_cli.main(self.argv(), stdout=StringIO(), stderr=StringIO()), 0)
        manifest = json.loads(self.output().read_text(encoding="utf-8"))
        roots = manifest.pop("roots")
        remainder = json.dumps(manifest)
        self.assertEqual(roots["ORIGINAL"], str(self.originals.resolve()))
        self.assertEqual(roots["CROPPED"], str(self.cropped.resolve()))
        self.assertNotIn(str(self.originals.resolve()), remainder)
        self.assertNotIn(str(self.cropped.resolve()), remainder)

    def test_path_safety_and_no_overwrite(self) -> None:
        self.create_dataset()
        inside = self.originals / "result.private.json"
        self.assertEqual(content_match_cli.main(self.argv(inside), stdout=StringIO(), stderr=StringIO()), 2)
        wrong_suffix = self.results / "result.json"
        self.assertEqual(content_match_cli.main(self.argv(wrong_suffix), stdout=StringIO(), stderr=StringIO()), 2)
        output = self.output()
        output.write_text("preexisting", encoding="utf-8")
        self.assertEqual(content_match_cli.main(self.argv(), stdout=StringIO(), stderr=StringIO()), 2)
        self.assertEqual(output.read_text(encoding="utf-8"), "preexisting")

    def test_audit_unreadable_original_forces_ambiguous_without_console_filename(self) -> None:
        self.create_dataset()
        (self.originals / "private-broken.jpg").write_bytes(b"not an image")
        stdout = StringIO()
        stderr = StringIO()
        self.assertEqual(content_match_cli.main(self.argv(), stdout=stdout, stderr=stderr), 0)
        manifest = json.loads(self.output().read_text(encoding="utf-8"))
        summary = manifest["summary"]
        self.assertEqual(summary["supplied_original_image_candidates"], 3)
        self.assertEqual(summary["audit_unreadable_original_candidates"], 1)
        self.assertEqual(summary["successfully_evaluated_originals"], 2)
        self.assertFalse(summary["candidate_set_complete"])
        self.assertEqual(summary["MATCHED"], 0)
        self.assertEqual(manifest["crops"][0]["decision"], "AMBIGUOUS")
        self.assertEqual(
            manifest["crops"][0]["diagnostic_reason"],
            "INCOMPLETE_CANDIDATE_SET",
        )
        self.assertNotIn("private-broken", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_feature_extraction_unavailable_original_forces_ambiguous(self) -> None:
        self.create_dataset()
        unavailable_name = "private-feature-unavailable.png"
        structured_image(13).save(self.originals / unavailable_name)
        real_extract = content_match_cli.extract_features

        def fail_one(path, semantic_reference, parameters, *, retain_grayscale):
            if path.name == unavailable_name:
                return unavailable_feature_image(
                    semantic_reference,
                    "SyntheticFeatureExtractionFailure",
                )
            return real_extract(
                path,
                semantic_reference,
                parameters,
                retain_grayscale=retain_grayscale,
            )

        stdout = StringIO()
        with mock.patch(
            "autocrop_analysis.content_match_cli.extract_features",
            side_effect=fail_one,
        ):
            exit_code = content_match_cli.main(
                self.argv(),
                stdout=stdout,
                stderr=StringIO(),
            )

        self.assertEqual(exit_code, 0)
        manifest = json.loads(self.output().read_text(encoding="utf-8"))
        summary = manifest["summary"]
        self.assertEqual(summary["supplied_original_image_candidates"], 3)
        self.assertEqual(summary["audit_readable_original_candidates"], 3)
        self.assertEqual(summary["feature_extraction_unavailable_originals"], 1)
        self.assertEqual(summary["successfully_evaluated_originals"], 2)
        self.assertFalse(summary["candidate_set_complete"])
        self.assertEqual(manifest["crops"][0]["decision"], "AMBIGUOUS")
        self.assertEqual(
            manifest["crops"][0]["diagnostic_reason"],
            "INCOMPLETE_CANDIDATE_SET",
        )
        self.assertNotIn(unavailable_name, stdout.getvalue())

    def test_lazy_decode_failure_forces_run_wide_ambiguous_finalization(self) -> None:
        lazy_name = "private-lazy-source.png"
        stable_name = "private-stable-source.png"
        crop_a_name = "private-crop-a.jpg"
        crop_b_name = "private-crop-b.jpg"
        lazy_source = structured_image(41)
        stable_source = structured_image(73)
        lazy_source.save(self.originals / lazy_name)
        stable_source.save(self.originals / stable_name)
        lazy_source.crop((70, 45, 360, 275)).resize(
            (232, 184), Image.Resampling.LANCZOS
        ).save(self.cropped / crop_a_name, format="JPEG", quality=78)
        stable_source.crop((65, 50, 355, 280)).resize(
            (232, 184), Image.Resampling.LANCZOS
        ).save(self.cropped / crop_b_name, format="JPEG", quality=78)

        baseline_output = self.output("baseline.private.json")
        self.assertEqual(
            content_match_cli.main(
                self.argv(baseline_output), stdout=StringIO(), stderr=StringIO()
            ),
            0,
        )
        baseline = json.loads(baseline_output.read_text(encoding="utf-8"))
        self.assertEqual(baseline["summary"]["MATCHED"], 2)

        real_decode = content_matching._decode_grayscale
        decode_counts: dict[str, int] = {}

        def fail_lazy_reopen(path: Path):
            decode_counts[path.name] = decode_counts.get(path.name, 0) + 1
            if path.name == lazy_name and decode_counts[path.name] > 1:
                raise OSError("synthetic lazy decode failure")
            return real_decode(path)

        final_output = self.output("final.private.json")
        stdout = StringIO()
        stderr = StringIO()
        with mock.patch(
            "autocrop_analysis.content_matching._decode_grayscale",
            side_effect=fail_lazy_reopen,
        ):
            exit_code = content_match_cli.main(
                self.argv(final_output), stdout=stdout, stderr=stderr
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertTrue(final_output.is_file())
        manifest = json.loads(final_output.read_text(encoding="utf-8"))
        summary = manifest["summary"]
        self.assertFalse(summary["candidate_set_complete"])
        self.assertEqual(summary["photometric_decode_unavailable_originals"], 1)
        self.assertEqual(summary["MATCHED"], 0)
        self.assertEqual(summary["UNIQUE_STRONG_PROVENANCE"], 0)
        self.assertEqual(summary["AMBIGUOUS"], 2)
        self.assertEqual(summary["AMBIGUOUS_PROVENANCE"], 2)
        self.assertEqual(summary["NO_MATCH"], 0)
        self.assertEqual(len(manifest["crops"]), 2)
        self.assertTrue(
            all(result["decision"] == "AMBIGUOUS" for result in manifest["crops"])
        )
        self.assertTrue(
            all(
                result["provenance_interpretation"] == "AMBIGUOUS_PROVENANCE"
                for result in manifest["crops"]
            )
        )
        self.assertTrue(
            all(
                result["diagnostic_reason"] == "INCOMPLETE_CANDIDATE_SET"
                for result in manifest["crops"]
            )
        )
        self.assertTrue(
            all(len(result["ranked_candidates"]) == 2 for result in manifest["crops"])
        )
        crop_a = next(
            result
            for result in manifest["crops"]
            if result["crop"]["relative_path"] == crop_a_name
        )
        self.assertTrue(
            any(
                not candidate["evaluation_complete"]
                and candidate["diagnostic_reason"].startswith(
                    "ORIGINAL_PHOTOMETRIC_DECODE_FAILED:"
                )
                for candidate in crop_a["ranked_candidates"]
            )
        )
        crop_b = next(
            result
            for result in manifest["crops"]
            if result["crop"]["relative_path"] == crop_b_name
        )
        self.assertEqual(
            crop_b["ranked_candidates"][0]["original"]["relative_path"],
            stable_name,
        )
        self.assertTrue(crop_b["ranked_candidates"][0]["strong_provenance"])
        for private_value in (lazy_name, stable_name, crop_a_name, crop_b_name):
            self.assertNotIn(private_value, stdout.getvalue())

    def test_sift_runtime_failure_is_explicit_and_writes_nothing(self) -> None:
        self.create_dataset()
        with mock.patch(
            "autocrop_analysis.content_match_cli.ensure_sift_available",
            side_effect=RuntimeError("SIFT_UNAVAILABLE"),
        ):
            stderr = StringIO()
            exit_code = content_match_cli.main(self.argv(), stdout=StringIO(), stderr=stderr)
        self.assertEqual(exit_code, 1)
        self.assertEqual(stderr.getvalue(), "runtime dependency error: SIFT_UNAVAILABLE\n")
        self.assertFalse(self.output().exists())

    def test_atomic_publication_collision_returns_three_without_clobber(self) -> None:
        self.create_dataset()
        with mock.patch(
            "autocrop_analysis.content_match_cli.write_manifest_atomic",
            side_effect=content_match_cli.OutputFailure("FileExistsError"),
        ):
            exit_code = content_match_cli.main(self.argv(), stdout=StringIO(), stderr=StringIO())
        self.assertEqual(exit_code, 3)
        self.assertFalse(self.output().exists())

    def test_actual_module_invocation_synthetic_smoke(self) -> None:
        source_name, distractor_name, crop_name = self.create_dataset()
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "autocrop_analysis.content_match_cli",
                "--originals",
                str(self.originals),
                "--cropped",
                str(self.cropped),
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
            "CONTENT_PROVENANCE: originals=2 crops=1 comparisons=2 "
            "evaluated_originals=2/2 candidate_set_complete=true "
            "matched_unique=1 ambiguous=0 no_match=0\n",
        )
        for private_value in (source_name, distractor_name, crop_name, str(self.originals)):
            self.assertNotIn(private_value, completed.stdout)
        manifest = json.loads(self.output().read_text(encoding="utf-8"))
        self.assertEqual(manifest["crops"][0]["decision"], "MATCHED")
        self.assertEqual(
            manifest["crops"][0]["ranked_candidates"][0]["original"]["relative_path"],
            source_name,
        )


if __name__ == "__main__":
    unittest.main()

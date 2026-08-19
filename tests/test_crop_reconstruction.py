"""Synthetic tests for manifest-only crop reconstruction."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

from autocrop_analysis.cli import ValidatedPaths
from autocrop_analysis import content_match_cli
from autocrop_analysis.content_matching import MatchingParameters
from autocrop_analysis.crop_reconstruction import (
    BaseTransform,
    ManifestValidationError,
    ReconstructionParameters,
    ReconstructionReason,
    ReconstructionStatus,
    parse_provenance_bytes,
    project_corners,
    reconstruct_geometry,
    reconstruct_manifest,
    validate_provenance_manifest,
)


CROP_WIDTH = 200
CROP_HEIGHT = 100
ORIGINAL_WIDTH = 500
ORIGINAL_HEIGHT = 400
CROP_BOX = (100, 70, 380, 280)


def structured_image(seed: int, size: tuple[int, int] = (480, 360)) -> Image.Image:
    width, height = size
    generator = np.random.default_rng(seed)
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    noise = generator.normal(0, 18, (height, width)).astype(np.float32)
    red = np.clip(35 + 125 * x + 45 * y + noise, 0, 255)
    green = np.clip(45 + 70 * x + 115 * y + np.roll(noise, 9, axis=1), 0, 255)
    blue = np.clip(150 - 80 * x + 65 * y + np.roll(noise, 7, axis=0), 0, 255)
    array = np.stack((red, green, blue), axis=2).astype(np.uint8)
    image = Image.fromarray(array, "RGB")
    draw = ImageDraw.Draw(image)
    for index in range(28):
        left = int(generator.integers(0, max(1, width - 45)))
        top = int(generator.integers(0, max(1, height - 45)))
        right = min(width - 1, left + int(generator.integers(12, 70)))
        bottom = min(height - 1, top + int(generator.integers(12, 70)))
        color = tuple(int(value) for value in generator.integers(15, 245, 3))
        draw.rectangle((left, top, right, bottom), outline=color, width=3)
    draw.text((width // 3, height // 2), f"SYNTH-{seed:04d}", fill=(5, 5, 5))
    return image


def image_reference(
    role: str,
    path: str,
    width: int | None,
    height: int | None,
) -> dict[str, object]:
    return {
        "root_role": role,
        "relative_path": path,
        "display_width": width,
        "display_height": height,
    }


def candidate_record(
    *,
    rank: int = 1,
    name: str = "original.png",
    transform: BaseTransform | None = None,
    complete: bool = True,
    strong: bool = True,
    plausible: bool = True,
    original_width: int | None = ORIGINAL_WIDTH,
    original_height: int | None = ORIGINAL_HEIGHT,
) -> dict[str, object]:
    transform = transform or BaseTransform(1.0, 0.0, 100.0, 70.0)
    corners = project_corners(transform, crop_width=CROP_WIDTH, crop_height=CROP_HEIGHT)
    return {
        "rank": rank,
        "original": image_reference("ORIGINAL", name, original_width, original_height),
        "descriptor_evidence": {},
        "geometric_evidence": {"transform_valid": True},
        "transform_evidence": {
            "plausible": plausible,
            "scale": transform.scale,
            "rotation_degrees": transform.rotation_degrees,
            "translation_x": transform.translation_x,
            "translation_y": transform.translation_y,
            "projected_corners": [list(point) for point in corners],
            "projected_inside_fraction": 1.0,
        },
        "photometric_evidence": {},
        "evaluation_complete": complete,
        "strong_provenance": strong,
        "diagnostic_reason": "SYNTHETIC",
    }


def provenance_manifest(
    *,
    decision: str = "MATCHED",
    interpretation: str | None = None,
    candidate_set_complete: bool = True,
) -> dict[str, object]:
    interpretations = {
        "MATCHED": "UNIQUE_STRONG_PROVENANCE",
        "AMBIGUOUS": "AMBIGUOUS_PROVENANCE",
        "NO_MATCH": "NO_VALID_PROVENANCE",
    }
    crop = {
        "crop": image_reference("CROPPED", "crop.png", CROP_WIDTH, CROP_HEIGHT),
        "decision": decision,
        "provenance_interpretation": interpretation or interpretations.get(decision, "NO_VALID_PROVENANCE"),
        "diagnostic_reason": "SYNTHETIC",
        "best_vs_second_margins": {},
        "ranked_candidates": [candidate_record()],
    }
    unavailable_originals = 0 if candidate_set_complete else 1
    supplied_originals = 1 + unavailable_originals
    manifest: dict[str, object] = {
        "schema_version": "1.1",
        "tool_version": "0.1.0",
        "runtime": {},
        "algorithm": {
            "name": "SIFT_BF_L2_MUTUAL_SIMILARITY_RANSAC_ALIGNED_PHOTOMETRIC",
            "status": "EXPERIMENTAL_UNLABELED_PROVENANCE_DISCOVERY",
            "grayscale_convention": "uint8_0_255_after_exif_transpose",
            "geometric_model": "estimateAffinePartial2D_RANSAC_crop_to_original_display",
            "projected_corner_order": [
                "TOP_LEFT",
                "TOP_RIGHT",
                "BOTTOM_RIGHT",
                "BOTTOM_LEFT",
            ],
        },
        "parameters": {},
        "roots": {"ORIGINAL": "private", "CROPPED": "private"},
        "summary": {
            "originals_analyzed": 1,
            "crops_analyzed": 1,
            "candidate_comparisons": 1,
            "supplied_original_image_candidates": supplied_originals,
            "audit_readable_original_candidates": 1,
            "audit_unreadable_original_candidates": unavailable_originals,
            "audit_unsupported_original_candidates": 0,
            "audit_filesystem_error_original_candidates": 0,
            "feature_extraction_unavailable_originals": 0,
            "photometric_decode_unavailable_originals": 0,
            "successfully_evaluated_originals": 1,
            "candidate_set_complete": candidate_set_complete,
            "audit_unavailable_original_candidates": unavailable_originals,
            "audit_unavailable_crop_candidates": 0,
            "MATCHED": int(decision == "MATCHED"),
            "UNIQUE_STRONG_PROVENANCE": int(decision == "MATCHED"),
            "AMBIGUOUS": int(decision == "AMBIGUOUS"),
            "AMBIGUOUS_PROVENANCE": int(decision == "AMBIGUOUS"),
            "NO_MATCH": int(decision == "NO_MATCH"),
            "NO_VALID_PROVENANCE": int(decision == "NO_MATCH"),
        },
        "crops": [crop],
    }
    return manifest


def validate(manifest: dict[str, object]):
    return validate_provenance_manifest(manifest, ReconstructionParameters())


def set_complete_original_summary(
    manifest: dict[str, object], *, originals: int, crops: int
) -> None:
    summary = manifest["summary"]
    summary.update(
        {
            "originals_analyzed": originals,
            "crops_analyzed": crops,
            "candidate_comparisons": originals * crops,
            "supplied_original_image_candidates": originals,
            "audit_readable_original_candidates": originals,
            "audit_unreadable_original_candidates": 0,
            "audit_unsupported_original_candidates": 0,
            "audit_filesystem_error_original_candidates": 0,
            "feature_extraction_unavailable_originals": 0,
            "photometric_decode_unavailable_originals": 0,
            "successfully_evaluated_originals": originals,
            "candidate_set_complete": True,
            "audit_unavailable_original_candidates": 0,
        }
    )


def set_all_matched_summary(manifest: dict[str, object], count: int) -> None:
    summary = manifest["summary"]
    summary.update(
        {
            "MATCHED": count,
            "UNIQUE_STRONG_PROVENANCE": count,
            "AMBIGUOUS": 0,
            "AMBIGUOUS_PROVENANCE": 0,
            "NO_MATCH": 0,
            "NO_VALID_PROVENANCE": 0,
        }
    )


def reconstruct(manifest: dict[str, object]) -> dict[str, object]:
    raw = json.dumps(manifest, allow_nan=False, sort_keys=True).encode("utf-8")
    validated = validate_provenance_manifest(parse_provenance_bytes(raw))
    return reconstruct_manifest(
        validated,
        source_path=Path("C:/synthetic/provenance.private.json"),
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


class GeometryTests(unittest.TestCase):
    def evaluate(
        self,
        transform: BaseTransform,
        *,
        crop_width: int = CROP_WIDTH,
        crop_height: int = CROP_HEIGHT,
        original_width: int = ORIGINAL_WIDTH,
        original_height: int = ORIGINAL_HEIGHT,
        corners=None,
    ):
        projected = corners or project_corners(
            transform, crop_width=crop_width, crop_height=crop_height
        )
        return reconstruct_geometry(
            projected,
            transform,
            crop_width=crop_width,
            crop_height=crop_height,
            original_width=original_width,
            original_height=original_height,
        )

    def test_perfect_axis_aligned_projected_rectangle(self) -> None:
        result = self.evaluate(BaseTransform(1.0, 0.0, 100.0, 70.0))
        self.assertIs(result.status, ReconstructionStatus.RECONSTRUCTED)
        assert result.rectangle is not None
        self.assertEqual(
            (result.rectangle.left, result.rectangle.top, result.rectangle.right, result.rectangle.bottom),
            (100.0, 70.0, 300.0, 170.0),
        )

    def test_fractional_rectangle_coordinates_remain_floats(self) -> None:
        result = self.evaluate(BaseTransform(1.25, 0.0, 100.25, 70.75))
        assert result.rectangle is not None
        self.assertEqual(
            (result.rectangle.left, result.rectangle.top, result.rectangle.right, result.rectangle.bottom),
            (100.25, 70.75, 350.25, 195.75),
        )

    def test_near_zero_rotation_is_accepted(self) -> None:
        result = self.evaluate(BaseTransform(1.0, 0.1, 100.0, 70.0))
        self.assertIs(result.status, ReconstructionStatus.RECONSTRUCTED)

    def test_exactly_one_degree_is_accepted(self) -> None:
        result = self.evaluate(BaseTransform(1.0, 1.0, 100.0, 70.0))
        self.assertIs(result.status, ReconstructionStatus.RECONSTRUCTED)

    def test_rotation_above_one_degree_is_rejected(self) -> None:
        result = self.evaluate(BaseTransform(1.0, 1.0001, 100.0, 70.0))
        self.assertIs(result.reason, ReconstructionReason.NON_AXIS_ALIGNED_GEOMETRY)

    def test_degenerate_zero_area_geometry_is_rejected_first(self) -> None:
        transform = BaseTransform(0.0, 0.0, 100.0, 70.0)
        result = self.evaluate(transform)
        self.assertIs(result.reason, ReconstructionReason.DEGENERATE_GEOMETRY)

    def test_inconsistent_opposite_sides_are_rejected(self) -> None:
        transform = BaseTransform(1.0, 0.0, 100.0, 70.0)
        corners = ((100.0, 70.0), (300.0, 70.0), (320.0, 170.0), (100.0, 170.0))
        result = self.evaluate(transform, corners=corners)
        self.assertIs(result.reason, ReconstructionReason.INCONSISTENT_GEOMETRY)

    def test_non_parallel_geometry_is_rejected(self) -> None:
        transform = BaseTransform(1.0, 0.0, 100.0, 70.0)
        corners = ((100.0, 70.0), (300.0, 70.0), (290.0, 170.0), (100.0, 170.0))
        result = self.evaluate(transform, corners=corners)
        self.assertIs(result.reason, ReconstructionReason.INCONSISTENT_GEOMETRY)

    def test_non_perpendicular_geometry_is_rejected(self) -> None:
        transform = BaseTransform(1.0, 0.0, 100.0, 70.0)
        corners = ((100.0, 70.0), (300.0, 70.0), (310.0, 170.0), (110.0, 170.0))
        result = self.evaluate(transform, corners=corners)
        self.assertIs(result.reason, ReconstructionReason.INCONSISTENT_GEOMETRY)

    def test_out_of_bounds_projected_corner_is_rejected(self) -> None:
        result = self.evaluate(BaseTransform(1.0, 0.0, -1.0, 70.0))
        self.assertIs(result.reason, ReconstructionReason.OUT_OF_BOUNDS)

    def test_canonical_rectangle_out_of_bounds_is_rejected(self) -> None:
        rotation = 1.0
        translation_x = math.sin(math.radians(rotation))
        result = self.evaluate(
            BaseTransform(1.0, rotation, translation_x, 10.0),
            crop_width=1000,
            crop_height=1,
            original_width=1000,
            original_height=30,
        )
        self.assertTrue(all(0.0 <= x <= 1000 for x, _ in project_corners(
            BaseTransform(1.0, rotation, translation_x, 10.0),
            crop_width=1000,
            crop_height=1,
        )))
        self.assertIs(result.reason, ReconstructionReason.OUT_OF_BOUNDS)

    def test_transform_corner_consistency_is_recorded(self) -> None:
        result = self.evaluate(BaseTransform(0.8, 0.5, 120.0, 80.0))
        self.assertLessEqual(result.diagnostics.transform_corner_max_error, 1e-12)

    def test_aspect_ratio_is_diagnostic_not_a_gate(self) -> None:
        result = self.evaluate(
            BaseTransform(1.0, 0.0, 100.0, 70.0),
            crop_width=200,
            crop_height=50,
        )
        self.assertIs(result.status, ReconstructionStatus.RECONSTRUCTED)
        assert result.rectangle is not None
        self.assertEqual(result.rectangle.aspect_ratio, 4.0)

    def test_repeated_reconstruction_is_deterministic(self) -> None:
        manifest = provenance_manifest()
        self.assertEqual(reconstruct(manifest), reconstruct(manifest))


class ManifestValidationTests(unittest.TestCase):
    def assert_invalid(self, manifest: dict[str, object], code: str | None = None) -> None:
        with self.assertRaises(ManifestValidationError) as caught:
            validate(manifest)
        if code is not None:
            self.assertEqual(caught.exception.code, code)

    def test_wrong_schema_is_rejected(self) -> None:
        manifest = provenance_manifest()
        manifest["schema_version"] = "1.0"
        self.assert_invalid(manifest, "UNSUPPORTED_SCHEMA_VERSION")

    def test_wrong_algorithm_name_is_rejected(self) -> None:
        manifest = provenance_manifest()
        manifest["algorithm"]["name"] = "OTHER"
        self.assert_invalid(manifest, "UNSUPPORTED_ALGORITHM_NAME")

    def test_wrong_algorithm_status_is_rejected(self) -> None:
        manifest = provenance_manifest()
        manifest["algorithm"]["status"] = "OTHER"
        self.assert_invalid(manifest, "UNSUPPORTED_ALGORITHM_STATUS")

    def test_wrong_geometric_model_is_rejected(self) -> None:
        manifest = provenance_manifest()
        manifest["algorithm"]["geometric_model"] = "OTHER"
        self.assert_invalid(manifest, "UNSUPPORTED_GEOMETRIC_MODEL")

    def test_wrong_corner_order_is_rejected(self) -> None:
        manifest = provenance_manifest()
        manifest["algorithm"]["projected_corner_order"].reverse()
        self.assert_invalid(manifest, "UNSUPPORTED_PROJECTED_CORNER_ORDER")

    def test_invalid_decision_is_rejected(self) -> None:
        manifest = provenance_manifest()
        manifest["crops"][0]["decision"] = "UNKNOWN"
        self.assert_invalid(manifest, "UNKNOWN_DECISION")

    def test_decision_interpretation_mismatch_is_rejected(self) -> None:
        manifest = provenance_manifest()
        manifest["crops"][0]["provenance_interpretation"] = "AMBIGUOUS_PROVENANCE"
        self.assert_invalid(manifest, "INCONSISTENT_DECISION_INTERPRETATION")

    def test_duplicate_crop_is_rejected(self) -> None:
        manifest = provenance_manifest()
        manifest["crops"].append(deepcopy(manifest["crops"][0]))
        manifest["summary"]["crops_analyzed"] = 2
        manifest["summary"]["candidate_comparisons"] = 2
        manifest["summary"]["MATCHED"] = 2
        manifest["summary"]["UNIQUE_STRONG_PROVENANCE"] = 2
        self.assert_invalid(manifest, "DUPLICATE_CROP_REFERENCE")

    def test_duplicate_candidate_is_rejected(self) -> None:
        manifest = provenance_manifest()
        duplicate = deepcopy(manifest["crops"][0]["ranked_candidates"][0])
        duplicate["rank"] = 2
        manifest["crops"][0]["ranked_candidates"].append(duplicate)
        manifest["summary"]["candidate_comparisons"] = 2
        self.assert_invalid(manifest, "DUPLICATE_CANDIDATE_REFERENCE")

    def test_invalid_ranks_are_rejected(self) -> None:
        manifest = provenance_manifest()
        manifest["crops"][0]["ranked_candidates"][0]["rank"] = 2
        self.assert_invalid(manifest, "INVALID_CANDIDATE_RANKS")

    def test_candidate_count_must_equal_originals_analyzed(self) -> None:
        manifest = provenance_manifest()
        set_complete_original_summary(manifest, originals=2, crops=1)
        manifest["summary"]["candidate_comparisons"] = 1
        self.assert_invalid(manifest, "INCONSISTENT_CANDIDATE_COUNT")

    def test_uneven_candidate_counts_cannot_preserve_aggregate_total(self) -> None:
        manifest = provenance_manifest()
        second_crop = deepcopy(manifest["crops"][0])
        second_crop["crop"]["relative_path"] = "crop-2.png"
        second_crop["ranked_candidates"].extend(
            (
                candidate_record(rank=2, name="original-2.png"),
                candidate_record(rank=3, name="original-3.png"),
            )
        )
        manifest["crops"].append(second_crop)
        set_complete_original_summary(manifest, originals=2, crops=2)
        set_all_matched_summary(manifest, 2)
        self.assertEqual(
            sum(len(crop["ranked_candidates"]) for crop in manifest["crops"]),
            manifest["summary"]["candidate_comparisons"],
        )
        self.assert_invalid(manifest, "INCONSISTENT_CANDIDATE_COUNT")

    def test_equal_candidate_counts_require_same_original_universe(self) -> None:
        manifest = provenance_manifest()
        first_crop = manifest["crops"][0]
        first_crop["ranked_candidates"].append(
            candidate_record(rank=2, name="original-2.png")
        )
        second_crop = deepcopy(first_crop)
        second_crop["crop"]["relative_path"] = "crop-2.png"
        second_crop["ranked_candidates"][1] = candidate_record(
            rank=2, name="substituted-original.png"
        )
        manifest["crops"].append(second_crop)
        set_complete_original_summary(manifest, originals=2, crops=2)
        set_all_matched_summary(manifest, 2)
        self.assert_invalid(manifest, "INCONSISTENT_CANDIDATE_UNIVERSE")

    def test_falsified_complete_boolean_is_rejected(self) -> None:
        manifest = provenance_manifest()
        summary = manifest["summary"]
        summary["supplied_original_image_candidates"] = 2
        summary["audit_unreadable_original_candidates"] = 1
        summary["audit_unavailable_original_candidates"] = 1
        self.assert_invalid(manifest, "INCONSISTENT_CANDIDATE_SET_COMPLETENESS")

    def test_original_availability_summary_corruption_is_rejected(self) -> None:
        manifest = provenance_manifest()
        manifest["summary"]["audit_unavailable_original_candidates"] = 1
        self.assert_invalid(manifest, "INCONSISTENT_ORIGINAL_AVAILABILITY_SUMMARY")

    def test_valid_complete_candidate_universe_reconstructs(self) -> None:
        manifest = provenance_manifest()
        validated = validate(manifest)
        output = reconstruct_manifest(
            validated,
            source_path=Path("C:/synthetic/provenance.private.json"),
            source_sha256="0" * 64,
        )
        self.assertEqual(output["items"][0]["status"], "RECONSTRUCTED")

    def test_valid_incomplete_candidate_universe_abstains(self) -> None:
        manifest = provenance_manifest(
            decision="AMBIGUOUS", candidate_set_complete=False
        )
        validated = validate(manifest)
        output = reconstruct_manifest(
            validated,
            source_path=Path("C:/synthetic/provenance.private.json"),
            source_sha256="0" * 64,
        )
        self.assertFalse(validated.candidate_set_complete)
        self.assertEqual(output["items"][0]["status"], "NOT_RECONSTRUCTED")
        self.assertEqual(
            output["items"][0]["reason"], "UPSTREAM_PROVENANCE_NOT_MATCHED"
        )

    def test_unsafe_relative_paths_are_rejected(self) -> None:
        unsafe = ("../crop.png", "/crop.png", "C:/crop.png", "a\\crop.png", "a//crop.png", "a/./crop.png")
        for path in unsafe:
            with self.subTest(path=path):
                manifest = provenance_manifest()
                manifest["crops"][0]["crop"]["relative_path"] = path
                self.assert_invalid(manifest, "UNSAFE_RELATIVE_PATH")

    def test_boolean_dimension_is_rejected(self) -> None:
        manifest = provenance_manifest()
        manifest["crops"][0]["crop"]["display_width"] = True
        self.assert_invalid(manifest, "INVALID_DISPLAY_DIMENSION")

    def test_partial_null_dimensions_are_rejected(self) -> None:
        manifest = provenance_manifest(decision="NO_MATCH")
        manifest["crops"][0]["crop"]["display_width"] = None
        self.assert_invalid(manifest, "PARTIAL_NULL_DIMENSIONS")

    def test_matched_with_null_dimensions_is_rejected(self) -> None:
        manifest = provenance_manifest()
        manifest["crops"][0]["crop"]["display_width"] = None
        manifest["crops"][0]["crop"]["display_height"] = None
        self.assert_invalid(manifest, "MATCHED_WITHOUT_DISPLAY_DIMENSIONS")

    def test_incomplete_candidate_set_with_match_is_rejected(self) -> None:
        manifest = provenance_manifest(candidate_set_complete=False)
        self.assert_invalid(manifest, "MATCHED_WITH_INCOMPLETE_CANDIDATE_SET")

    def test_complete_candidate_set_with_incomplete_evidence_is_rejected(self) -> None:
        manifest = provenance_manifest(decision="AMBIGUOUS")
        manifest["crops"][0]["ranked_candidates"][0]["evaluation_complete"] = False
        manifest["crops"][0]["ranked_candidates"][0]["strong_provenance"] = False
        self.assert_invalid(manifest, "COMPLETE_SET_WITH_INCOMPLETE_EVIDENCE")

    def test_malformed_projected_corners_are_rejected(self) -> None:
        manifest = provenance_manifest()
        manifest["crops"][0]["ranked_candidates"][0]["transform_evidence"]["projected_corners"] = [[1.0, 2.0]]
        self.assert_invalid(manifest, "MALFORMED_PROJECTED_CORNERS")

    def test_non_finite_json_numbers_are_rejected(self) -> None:
        for constant in (b"NaN", b"Infinity", b"-Infinity", b"1e999"):
            with self.subTest(constant=constant):
                raw = b'{"value":' + constant + b"}"
                with self.assertRaises(ManifestValidationError) as caught:
                    parse_provenance_bytes(raw)
                self.assertEqual(caught.exception.code, "NON_FINITE_JSON_NUMBER")

    def test_transform_corner_contradiction_is_rejected(self) -> None:
        manifest = provenance_manifest()
        manifest["crops"][0]["ranked_candidates"][0]["transform_evidence"]["projected_corners"][0][0] += 1.0
        self.assert_invalid(manifest, "TRANSFORM_CORNER_CONTRADICTION")

    def test_contradictory_matched_rank_one_is_rejected(self) -> None:
        manifest = provenance_manifest()
        manifest["crops"][0]["ranked_candidates"][0]["strong_provenance"] = False
        self.assert_invalid(manifest, "CONTRADICTORY_MATCHED_RANK_ONE")

    def test_ambiguous_is_preserved_without_selected_geometry(self) -> None:
        output = reconstruct(provenance_manifest(decision="AMBIGUOUS"))
        item = output["items"][0]
        self.assertEqual(item["status"], "NOT_RECONSTRUCTED")
        self.assertEqual(item["reason"], "UPSTREAM_PROVENANCE_NOT_MATCHED")
        for key in ("selected_original", "projected_corners", "base_transform", "rectangle"):
            self.assertIsNone(item[key])

    def test_no_match_is_preserved_without_selected_geometry(self) -> None:
        output = reconstruct(provenance_manifest(decision="NO_MATCH"))
        item = output["items"][0]
        self.assertEqual(item["status"], "NOT_RECONSTRUCTED")
        self.assertEqual(item["reason"], "UPSTREAM_PROVENANCE_NOT_MATCHED")
        for key in ("selected_original", "projected_corners", "base_transform", "rectangle"):
            self.assertIsNone(item[key])


class EndToEndBridgeTests(unittest.TestCase):
    def test_existing_matcher_schema_1_1_to_known_crop_rectangle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            originals = base / "originals"
            crops = base / "crops"
            output = base / "unused.private.json"
            originals.mkdir()
            crops.mkdir()
            source = structured_image(7_001)
            source.save(originals / "source.png")
            structured_image(7_002).save(originals / "distractor.png")
            source.crop(CROP_BOX).resize((224, 168), Image.Resampling.LANCZOS).save(
                crops / "crop.png"
            )
            paths = ValidatedPaths(originals.resolve(), crops.resolve(), output)
            matching_parameters = MatchingParameters()
            prepared = content_match_cli.prepare_inputs(paths, matching_parameters)
            initial = content_match_cli.match_crops(
                prepared.crops,
                prepared.originals,
                matching_parameters,
                candidate_set_complete=prepared.candidate_set_complete,
            )
            results, complete = content_match_cli.finalize_candidate_set_completeness(
                initial,
                preprocessed_candidate_set_complete=prepared.candidate_set_complete,
            )
            upstream = content_match_cli.build_manifest(
                paths,
                prepared,
                results,
                matching_parameters,
                candidate_set_complete=complete,
            )
            self.assertEqual(upstream["crops"][0]["decision"], "MATCHED")
            raw = json.dumps(upstream, allow_nan=False, sort_keys=True).encode("utf-8")
            validated = validate_provenance_manifest(parse_provenance_bytes(raw))
            downstream = reconstruct_manifest(
                validated,
                source_path=output.resolve(),
                source_sha256=hashlib.sha256(raw).hexdigest(),
            )

        item = downstream["items"][0]
        self.assertEqual(item["status"], "RECONSTRUCTED")
        self.assertEqual(item["selected_original"]["relative_path"], "source.png")
        rectangle = item["rectangle"]
        expected = dict(zip(("left", "top", "right", "bottom"), CROP_BOX, strict=True))
        maximum_error = max(abs(rectangle[key] - expected[key]) for key in expected)
        self.assertLessEqual(maximum_error, 3.0)


if __name__ == "__main__":
    unittest.main()

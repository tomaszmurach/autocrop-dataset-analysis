"""Synthetic labeled validation for experimental content provenance matching."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from autocrop_analysis.audit import RootRole, SemanticReference
from autocrop_analysis.content_matching import (
    CandidateEvidence,
    ContentDecision,
    CropMatchResult,
    MatchingParameters,
    ProvenanceInterpretation,
    decide_candidates,
    ensure_sift_available,
    extract_features,
    finalize_candidate_set_completeness,
    match_crop,
)


SOURCE_SIZE = (480, 360)
CROP_BOX = (100, 70, 380, 280)


def structured_image(seed: int, size: tuple[int, int] = SOURCE_SIZE) -> Image.Image:
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
        if index % 3 == 0:
            draw.ellipse((left, top, right, bottom), outline=color, width=3)
        elif index % 3 == 1:
            draw.rectangle((left, top, right, bottom), outline=color, width=3)
        else:
            draw.line((left, top, right, bottom), fill=color, width=4)
    for row in range(24, height, 47):
        draw.line((12, row, width - 13, row + (seed % 7) - 3), fill=(245, 245, 245), width=1)
    draw.text((width // 3, height // 2), f"SYNTH-{seed:04d}", fill=(5, 5, 5), stroke_width=1, stroke_fill=(250, 250, 250))
    return image


def reference(role: RootRole, name: str) -> SemanticReference:
    return SemanticReference(role, name)


def synthetic_evidence(
    original_name: str,
    *,
    strong: bool = True,
    plausible: bool = True,
    luma: float | None = 0.98,
    gradient: float | None = 0.91,
    residual: float | None = 0.08,
) -> CandidateEvidence:
    return CandidateEvidence(
        crop=reference(RootRole.CROPPED, "crop.png"),
        original=reference(RootRole.ORIGINAL, original_name),
        crop_keypoints=100,
        original_keypoints=500,
        raw_knn_count=100,
        ratio_passed_count=60,
        reverse_ratio_passed_count=150,
        mutual_match_count=50,
        transform_valid=plausible,
        inlier_count=45 if plausible else 0,
        inlier_ratio=0.9 if plausible else None,
        reprojection_rmse=0.4 if plausible else None,
        reprojection_median=0.3 if plausible else None,
        reprojection_p95=0.8 if plausible else None,
        bbox_coverage=0.65 if plausible else None,
        grid_coverage=0.75 if plausible else None,
        scale=1.0 if plausible else None,
        rotation_degrees=0.0 if plausible else None,
        translation_x=100.0 if plausible else None,
        translation_y=70.0 if plausible else None,
        projected_corners=((100.0, 70.0), (380.0, 70.0), (380.0, 280.0), (100.0, 280.0)) if plausible else (),
        projected_inside_fraction=1.0 if plausible else None,
        transform_plausible=plausible,
        base_luminance_correlation=luma,
        base_gradient_correlation=gradient,
        base_normalized_residual=residual,
        luminance_correlation=luma,
        gradient_correlation=gradient,
        normalized_residual=residual,
        alignment_refined=False,
        refinement_shift_x=0,
        refinement_shift_y=0,
        evaluation_complete=True,
        strong_provenance=strong,
        diagnostic_reason="STRONG_PROVENANCE_EVIDENCE" if strong else "INSUFFICIENT_EVIDENCE",
    )


class DecisionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = MatchingParameters()
        self.crop = reference(RootRole.CROPPED, "crop.png")

    def decide(self, *candidates: CandidateEvidence):
        return decide_candidates(self.crop, candidates, self.parameters)

    def test_strong_unique_candidate_is_matched(self) -> None:
        result = self.decide(synthetic_evidence("source.png"))
        self.assertIs(result.decision, ContentDecision.MATCHED)
        self.assertIs(result.provenance_interpretation, ProvenanceInterpretation.UNIQUE_STRONG_PROVENANCE)

    def test_incomplete_candidate_set_forces_ambiguous(self) -> None:
        result = decide_candidates(
            self.crop,
            [synthetic_evidence("source.png")],
            self.parameters,
            candidate_set_complete=False,
        )
        self.assertIs(result.decision, ContentDecision.AMBIGUOUS)
        self.assertEqual(result.diagnostic_reason, "INCOMPLETE_CANDIDATE_SET")

    def test_run_wide_finalization_preserves_evidence_and_forces_abstention(self) -> None:
        failed_evidence = replace(
            synthetic_evidence("lazy-failure.png"),
            evaluation_complete=False,
            strong_provenance=False,
            diagnostic_reason="ORIGINAL_PHOTOMETRIC_DECODE_FAILED:OSError",
        )
        crop_a = decide_candidates(
            reference(RootRole.CROPPED, "crop-a.png"),
            [failed_evidence],
            self.parameters,
        )
        crop_b = CropMatchResult(
            crop=reference(RootRole.CROPPED, "crop-b.png"),
            decision=ContentDecision.MATCHED,
            provenance_interpretation=ProvenanceInterpretation.UNIQUE_STRONG_PROVENANCE,
            diagnostic_reason="UNIQUE_STRONG_PROVENANCE_EVIDENCE",
            ranked_candidates=(synthetic_evidence("valid-source.png"),),
        )

        finalized, candidate_set_complete = finalize_candidate_set_completeness(
            (crop_a, crop_b),
            preprocessed_candidate_set_complete=True,
        )

        self.assertFalse(candidate_set_complete)
        self.assertTrue(
            all(result.decision is ContentDecision.AMBIGUOUS for result in finalized)
        )
        self.assertTrue(
            all(
                result.provenance_interpretation
                is ProvenanceInterpretation.AMBIGUOUS_PROVENANCE
                for result in finalized
            )
        )
        self.assertTrue(
            all(
                result.diagnostic_reason == "INCOMPLETE_CANDIDATE_SET"
                for result in finalized
            )
        )
        self.assertIs(finalized[0].ranked_candidates, crop_a.ranked_candidates)
        self.assertIs(finalized[1].ranked_candidates, crop_b.ranked_candidates)

    def test_two_strong_near_equal_candidates_are_ambiguous(self) -> None:
        result = self.decide(
            synthetic_evidence("a.png"),
            synthetic_evidence("b.png", luma=0.975, gradient=0.905, residual=0.085),
        )
        self.assertIs(result.decision, ContentDecision.AMBIGUOUS)

    def test_no_valid_candidate_is_no_match(self) -> None:
        result = self.decide(synthetic_evidence("invalid.png", strong=False, plausible=False, luma=None, gradient=None, residual=None))
        self.assertIs(result.decision, ContentDecision.NO_MATCH)

    def test_strong_geometry_with_weak_photometry_cannot_match(self) -> None:
        result = self.decide(synthetic_evidence("weak.png", strong=False, luma=0.4, gradient=0.2, residual=0.7))
        self.assertIs(result.decision, ContentDecision.NO_MATCH)

    def test_strong_photometry_with_invalid_geometry_cannot_match(self) -> None:
        result = self.decide(synthetic_evidence("invalid.png", strong=False, plausible=False))
        self.assertIs(result.decision, ContentDecision.NO_MATCH)

    def test_insufficient_runner_up_separation_cannot_match(self) -> None:
        best = synthetic_evidence("a.png")
        second = synthetic_evidence("b.png", luma=0.97, gradient=0.89, residual=0.09)
        self.assertIs(self.decide(best, second).decision, ContentDecision.AMBIGUOUS)

    def test_alignment_refinement_winner_change_is_ambiguous(self) -> None:
        best = replace(synthetic_evidence("a.png"), base_luminance_correlation=0.94)
        second = replace(synthetic_evidence("b.png", luma=0.94, gradient=0.80, residual=0.14), base_luminance_correlation=0.99)
        result = self.decide(best, second)
        self.assertIs(result.decision, ContentDecision.AMBIGUOUS)
        self.assertEqual(result.diagnostic_reason, "ALIGNMENT_REFINEMENT_CHANGED_WINNER")

    def test_geometric_and_photometric_leaders_disagree_is_ambiguous(self) -> None:
        best = synthetic_evidence("photometric.png")
        geometry_leader = replace(
            synthetic_evidence("geometric.png", luma=0.93, gradient=0.82, residual=0.13),
            inlier_count=60,
            inlier_ratio=0.98,
        )
        result = self.decide(best, geometry_leader)
        self.assertIs(result.decision, ContentDecision.AMBIGUOUS)
        self.assertEqual(
            result.diagnostic_reason,
            "GEOMETRIC_AND_PHOTOMETRIC_LEADERS_DISAGREE",
        )


class ContentMatchingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_sift_available()

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.parameters = MatchingParameters()

    def save(self, image: Image.Image, name: str, *, format_name: str = "PNG", **save_options) -> Path:
        path = self.base / name
        image.save(path, format=format_name, **save_options)
        return path

    def features(self, path: Path, role: RootRole):
        return extract_features(
            path,
            reference(role, path.name),
            self.parameters,
            retain_grayscale=role is RootRole.CROPPED,
        )

    def result_for(self, crop_path: Path, originals: list[Path]):
        crop = self.features(crop_path, RootRole.CROPPED)
        original_features = [self.features(path, RootRole.ORIGINAL) for path in originals]
        return match_crop(crop, original_features, self.parameters)

    def test_normal_crop_resize_and_export_degradations_select_exact_source(self) -> None:
        source = structured_image(101)
        source_path = self.save(source, "correct.png")
        distractors = [self.save(structured_image(seed), f"distractor-{seed}.png") for seed in (201, 202, 203)]
        extracted = source.crop(CROP_BOX)
        cases = {
            "direct.png": extracted,
            "smaller.png": extracted.resize((196, 147), Image.Resampling.LANCZOS),
            "larger.png": extracted.resize((420, 315), Image.Resampling.BICUBIC),
            "brightness.png": ImageEnhance.Brightness(extracted).enhance(1.08),
            "contrast.png": ImageEnhance.Contrast(extracted).enhance(1.08),
            "blur.png": extracted.filter(ImageFilter.GaussianBlur(0.55)),
            "sharpen.png": extracted.filter(ImageFilter.SHARPEN),
        }
        for name, crop_image in cases.items():
            with self.subTest(case=name):
                crop_path = self.save(crop_image, name)
                result = self.result_for(crop_path, [*distractors, source_path])
                self.assertIs(result.decision, ContentDecision.MATCHED)
                self.assertEqual(result.ranked_candidates[0].original.relative_path, "correct.png")

        jpeg_path = self.save(
            extracted.resize((224, 168), Image.Resampling.LANCZOS),
            "recompressed.jpg",
            format_name="JPEG",
            quality=62,
        )
        jpeg_result = self.result_for(jpeg_path, [*distractors, source_path])
        self.assertIs(jpeg_result.decision, ContentDecision.MATCHED)
        self.assertEqual(jpeg_result.ranked_candidates[0].original.relative_path, "correct.png")

    def test_transform_recovers_known_crop_corners(self) -> None:
        source = structured_image(301)
        source_path = self.save(source, "source.png")
        crop_path = self.save(source.crop(CROP_BOX).resize((224, 168), Image.Resampling.LANCZOS), "crop.png")
        result = self.result_for(crop_path, [source_path])
        self.assertIs(result.decision, ContentDecision.MATCHED)
        actual = np.asarray(result.ranked_candidates[0].projected_corners)
        expected = np.asarray([(100, 70), (380, 70), (380, 280), (100, 280)], dtype=np.float64)
        self.assertLessEqual(float(np.max(np.linalg.norm(actual - expected, axis=1))), 3.0)

    def test_visible_burst_difference_selects_exact_source(self) -> None:
        exact = structured_image(401)
        neighbor = exact.copy()
        draw = ImageDraw.Draw(neighbor)
        draw.rectangle((205, 135, 285, 215), fill=(245, 20, 30), outline=(0, 0, 0), width=5)
        for offset in range(0, 80, 8):
            draw.line((205 + offset, 135, 285, 215 - offset), fill=(255, 255, 0), width=3)
        exact_path = self.save(exact, "frame-a.png")
        neighbor_path = self.save(neighbor, "frame-b.png")
        crop_path = self.save(exact.crop(CROP_BOX), "crop.png")
        result = self.result_for(crop_path, [neighbor_path, exact_path])
        self.assertIs(result.decision, ContentDecision.MATCHED)
        self.assertEqual(result.ranked_candidates[0].original.relative_path, "frame-a.png")

    def test_subtle_visible_burst_change_never_matches_neighbor(self) -> None:
        base = structured_image(451)
        exact = base.copy()
        neighbor = base.copy()
        exact_draw = ImageDraw.Draw(exact)
        neighbor_draw = ImageDraw.Draw(neighbor)
        exact_draw.line((238, 166, 254, 182), fill=(255, 245, 20), width=3)
        exact_draw.line((254, 166, 238, 182), fill=(10, 10, 10), width=3)
        neighbor_draw.line((242, 168, 258, 184), fill=(255, 245, 20), width=3)
        neighbor_draw.line((258, 168, 242, 184), fill=(10, 10, 10), width=3)
        exact_path = self.save(exact, "frame-exact.png")
        neighbor_path = self.save(neighbor, "frame-neighbor.png")
        crop_path = self.save(exact.crop(CROP_BOX), "crop.png")

        result = self.result_for(crop_path, [neighbor_path, exact_path])

        self.assertEqual(
            result.ranked_candidates[0].original.relative_path,
            "frame-exact.png",
        )
        self.assertIn(
            result.decision,
            (ContentDecision.MATCHED, ContentDecision.AMBIGUOUS),
        )

    def test_difference_outside_visible_crop_is_ambiguous(self) -> None:
        first = structured_image(501)
        second = first.copy()
        ImageDraw.Draw(second).rectangle((5, 5, 75, 55), fill=(255, 0, 255))
        first_path = self.save(first, "frame-a.png")
        second_path = self.save(second, "frame-b.png")
        crop_path = self.save(first.crop(CROP_BOX), "crop.png")
        result = self.result_for(crop_path, [first_path, second_path])
        self.assertIs(result.decision, ContentDecision.AMBIGUOUS)

    def test_absent_source_is_no_match(self) -> None:
        crop_path = self.save(structured_image(601).crop(CROP_BOX), "crop.png")
        originals = [self.save(structured_image(seed), f"source-{seed}.png") for seed in (602, 603, 604)]
        self.assertIs(self.result_for(crop_path, originals).decision, ContentDecision.NO_MATCH)

    def test_exif_orientation_is_normalized_in_memory(self) -> None:
        display_source = structured_image(701)
        encoded = display_source.transpose(Image.Transpose.ROTATE_90)
        exif = Image.Exif()
        exif[274] = 6
        source_path = self.save(encoded, "oriented.jpg", format_name="JPEG", quality=96, exif=exif)
        crop_path = self.save(display_source.crop(CROP_BOX), "crop.jpg", format_name="JPEG", quality=90)
        result = self.result_for(crop_path, [source_path])
        self.assertIs(result.decision, ContentDecision.MATCHED)
        self.assertEqual((self.features(source_path, RootRole.ORIGINAL).width, self.features(source_path, RootRole.ORIGINAL).height), SOURCE_SIZE)

    def test_low_texture_crop_abstains(self) -> None:
        source_path = self.save(structured_image(801), "source.png")
        crop_path = self.save(Image.new("RGB", (220, 170), (90, 90, 90)), "crop.png")
        self.assertIs(self.result_for(crop_path, [source_path]).decision, ContentDecision.NO_MATCH)

    def test_small_rotation_is_supported(self) -> None:
        source = structured_image(901)
        source_path = self.save(source, "source.png")
        angle = np.deg2rad(2.0)
        crop_to_source = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 105.0],
                [np.sin(angle), np.cos(angle), 65.0],
            ],
            dtype=np.float32,
        )
        crop_array = cv2.warpAffine(
            np.asarray(source),
            crop_to_source,
            (280, 210),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
        crop = Image.fromarray(crop_array, "RGB")
        crop_path = self.save(crop, "crop.png")
        result = self.result_for(crop_path, [source_path])
        self.assertIs(result.decision, ContentDecision.MATCHED)
        self.assertLessEqual(abs(result.ranked_candidates[0].rotation_degrees or 0.0), self.parameters.max_abs_rotation_degrees)

    def test_non_uniform_scale_does_not_become_normal_match(self) -> None:
        source = structured_image(1_001)
        source_path = self.save(source, "source.png")
        distorted = source.crop(CROP_BOX).resize((330, 165), Image.Resampling.BICUBIC)
        crop_path = self.save(distorted, "distorted.png")
        self.assertIsNot(self.result_for(crop_path, [source_path]).decision, ContentDecision.MATCHED)

    def test_repeated_execution_preserves_ranking_and_decision(self) -> None:
        source = structured_image(1_101)
        source_path = self.save(source, "source.png")
        distractor = self.save(structured_image(1_102), "distractor.png")
        crop_path = self.save(source.crop(CROP_BOX).resize((224, 168), Image.Resampling.LANCZOS), "crop.png")
        first = self.result_for(crop_path, [distractor, source_path])
        second = self.result_for(crop_path, [distractor, source_path])
        self.assertEqual(first, second)

    def test_original_preprocessing_releases_full_grayscale_pixels(self) -> None:
        source = structured_image(1_201)
        source_path = self.save(source, "source.png")
        crop_path = self.save(source.crop(CROP_BOX), "crop.png")
        original = self.features(source_path, RootRole.ORIGINAL)
        crop = self.features(crop_path, RootRole.CROPPED)

        self.assertIsNone(original.grayscale)
        self.assertEqual(original.source_path, source_path)
        self.assertIsNotNone(original.descriptors)
        self.assertIsNotNone(crop.grayscale)
        self.assertIs(match_crop(crop, [original], self.parameters).decision, ContentDecision.MATCHED)

    def test_lazy_original_decode_failure_forces_incomplete_ambiguity(self) -> None:
        source = structured_image(1_251)
        source_path = self.save(source, "source.png")
        crop_path = self.save(source.crop(CROP_BOX), "crop.png")
        original = self.features(source_path, RootRole.ORIGINAL)
        crop = self.features(crop_path, RootRole.CROPPED)
        source_path.unlink()

        result = match_crop(crop, [original], self.parameters)

        self.assertIs(result.decision, ContentDecision.AMBIGUOUS)
        self.assertEqual(result.diagnostic_reason, "INCOMPLETE_CANDIDATE_SET")
        self.assertFalse(result.ranked_candidates[0].evaluation_complete)
        self.assertTrue(
            result.ranked_candidates[0].diagnostic_reason.startswith(
                "ORIGINAL_PHOTOMETRIC_DECODE_FAILED:"
            )
        )

    def test_several_large_originals_keep_only_bounded_feature_state(self) -> None:
        self.assertEqual(self.parameters.sift_nfeatures, 3_000)
        originals = []
        for index, seed in enumerate((1_301, 1_302, 1_303)):
            path = self.save(structured_image(seed, (1_800, 1_200)), f"large-{index}.png")
            originals.append(self.features(path, RootRole.ORIGINAL))

        for original in originals:
            self.assertIsNone(original.grayscale)
            self.assertIsNotNone(original.source_path)
            self.assertLessEqual(len(original.keypoints), self.parameters.sift_nfeatures)
            self.assertIsNotNone(original.descriptors)
            assert original.descriptors is not None
            self.assertLessEqual(len(original.descriptors), self.parameters.sift_nfeatures)


if __name__ == "__main__":
    unittest.main()

"""Synthetic unit tests for deterministic compact-SIFT retrieval."""

from __future__ import annotations

from copy import deepcopy
import unittest

import cv2
import numpy as np

from autocrop_analysis.audit import RootRole, SemanticReference
from autocrop_analysis.candidate_retrieval import (
    IndexParameters,
    IndexStatus,
    IndexValidationError,
    OriginalIndexRecord,
    QueryParameters,
    RetrievalCandidate,
    build_index_manifest,
    recall_at_k,
    retrieve_candidates,
    select_shortlist,
    select_spatially_balanced_descriptors,
    validate_index_manifest,
)
from autocrop_analysis.content_matching import FeatureImage


def reference(name: str, role: RootRole = RootRole.ORIGINAL) -> SemanticReference:
    return SemanticReference(role, name)


def keypoint(x: float, y: float, response: float) -> cv2.KeyPoint:
    result = cv2.KeyPoint(x, y, 3.0)
    result.response = response
    result.angle = 0.0
    result.octave = 0
    result.class_id = -1
    return result


def feature_image(points: list[tuple[float, float, float]]) -> FeatureImage:
    descriptors = np.vstack(
        [np.full(128, index + 1, dtype=np.float32) for index in range(len(points))]
    ) if points else None
    return FeatureImage(
        reference("image.png"),
        None,
        100,
        100,
        None,
        tuple(keypoint(*point) for point in points),
        descriptors,
        None if points else "NO_DESCRIPTORS",
    )


def record(name: str, offset: int, count: int) -> OriginalIndexRecord:
    return OriginalIndexRecord(
        reference(name),
        100,
        100,
        10,
        "a" * 64,
        IndexStatus.INDEXED if count else IndexStatus.NO_DESCRIPTORS,
        count,
        offset,
        count,
    )


class SpatialSamplingTests(unittest.TestCase):
    def test_sampler_never_exceeds_cap_and_balances_cells(self) -> None:
        image = feature_image(
            [
                (10, 10, 9),
                (12, 12, 8),
                (70, 10, 7),
                (72, 12, 6),
                (10, 70, 5),
                (12, 72, 4),
                (70, 70, 3),
                (72, 72, 2),
            ]
        )

        selected = select_spatially_balanced_descriptors(
            image, maximum=4, grid_rows=2, grid_columns=2
        )

        self.assertEqual(selected.shape, (4, 128))
        self.assertEqual([int(row[0]) for row in selected], [1, 3, 5, 7])

    def test_sampler_uses_unused_capacity_in_later_rounds(self) -> None:
        image = feature_image(
            [(10, 10, 9), (12, 12, 8), (14, 14, 7), (70, 70, 6)]
        )
        selected = select_spatially_balanced_descriptors(
            image, maximum=3, grid_rows=2, grid_columns=2
        )
        self.assertEqual([int(row[0]) for row in selected], [1, 4, 2])

    def test_sampler_ties_are_deterministic(self) -> None:
        image = feature_image([(20, 20, 5), (10, 10, 5), (30, 30, 5)])
        first = select_spatially_balanced_descriptors(
            image, maximum=2, grid_rows=1, grid_columns=1
        )
        second = select_spatially_balanced_descriptors(
            image, maximum=2, grid_rows=1, grid_columns=1
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual([int(row[0]) for row in first], [2, 1])

    def test_sampler_returns_all_when_fewer_than_cap(self) -> None:
        selected = select_spatially_balanced_descriptors(
            feature_image([(10, 10, 2), (90, 90, 1)]),
            maximum=8,
            grid_rows=2,
            grid_columns=2,
        )
        self.assertEqual(selected.shape, (2, 128))

    def test_sampler_handles_zero_descriptors(self) -> None:
        selected = select_spatially_balanced_descriptors(
            feature_image([]), maximum=8, grid_rows=2, grid_columns=2
        )
        self.assertEqual(selected.shape, (0, 128))


class RetrievalScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(123)
        self.source = rng.normal(50, 2, (6, 128)).astype(np.float32)
        self.distractor = rng.normal(200, 2, (6, 128)).astype(np.float32)
        self.matrix = np.vstack((self.distractor, self.source)).astype(np.float32)
        self.records = (record("distractor.png", 0, 6), record("source.png", 6, 6))

    def test_known_source_retrieval_and_distractor_ranking(self) -> None:
        query = self.source + 0.1
        ranked = retrieve_candidates(
            query,
            self.matrix,
            self.records,
            QueryParameters(neighbor_depth=2, requested_k=5),
        )
        self.assertEqual(ranked[0].original.relative_path, "source.png")
        self.assertEqual(ranked[0].supporting_query_descriptors, 6)

    def test_repeated_query_is_identical(self) -> None:
        parameters = QueryParameters(neighbor_depth=2, requested_k=5)
        first = retrieve_candidates(self.source, self.matrix, self.records, parameters)
        second = retrieve_candidates(self.source, self.matrix, self.records, parameters)
        self.assertEqual(first, second)

    def test_same_original_gets_at_most_one_vote_per_query_descriptor(self) -> None:
        query = self.source[:1]
        ranked = retrieve_candidates(
            query,
            self.matrix,
            self.records,
            QueryParameters(neighbor_depth=6, requested_k=5),
        )
        self.assertEqual(ranked[0].supporting_query_descriptors, 1)

    def test_near_duplicate_candidates_can_rank_together(self) -> None:
        neighbor = self.source + 0.2
        matrix = np.vstack((self.source, neighbor)).astype(np.float32)
        records = (record("frame-a.png", 0, 6), record("frame-b.png", 6, 6))
        ranked = retrieve_candidates(
            self.source,
            matrix,
            records,
            QueryParameters(neighbor_depth=12, requested_k=2),
        )
        self.assertEqual(
            {candidate.original.relative_path for candidate in ranked},
            {"frame-a.png", "frame-b.png"},
        )

    def test_candidate_tie_uses_semantic_reference(self) -> None:
        zero = np.zeros((1, 128), dtype=np.float32)
        matrix = np.vstack((zero, zero))
        records = (record("b.png", 0, 1), record("a.png", 1, 1))
        ranked = retrieve_candidates(
            zero,
            matrix,
            records,
            QueryParameters(neighbor_depth=2, requested_k=2),
        )
        self.assertEqual([item.original.relative_path for item in ranked], ["a.png", "b.png"])

    def test_positive_vote_boundary_is_extended(self) -> None:
        ranked = tuple(
            RetrievalCandidate(reference(name), votes, distance, distance)
            for name, votes, distance in (
                ("a.png", 3, 1.0),
                ("b.png", 2, 2.0),
                ("c.png", 2, 3.0),
                ("d.png", 1, 4.0),
            )
        )
        selected, extension, boundary = select_shortlist(ranked, 2)
        self.assertEqual([item.original.relative_path for item in selected], ["a.png", "b.png", "c.png"])
        self.assertEqual(extension, 1)
        self.assertEqual(boundary, 2)

    def test_shortlist_does_not_pad_zero_support(self) -> None:
        ranked = (RetrievalCandidate(reference("a.png"), 1, 2.0, 2.0),)
        selected, extension, boundary = select_shortlist(ranked, 50)
        self.assertEqual(selected, ranked)
        self.assertEqual(extension, 0)
        self.assertIsNone(boundary)

    def test_recall_at_k_uses_known_sources_and_tie_extension(self) -> None:
        ranked = (
            RetrievalCandidate(reference("a.png"), 2, 1.0, 1.0),
            RetrievalCandidate(reference("source.png"), 1, 2.0, 2.0),
            RetrievalCandidate(reference("tie.png"), 1, 3.0, 3.0),
        )
        recalls = recall_at_k(((reference("source.png"), ranked),), (1, 2))
        self.assertEqual(recalls, {1: 0.0, 2: 1.0})


class IndexValidationTests(unittest.TestCase):
    def manifest(self) -> dict[str, object]:
        records = (record("a.png", 0, 1), record("b.png", 1, 1))
        return build_index_manifest(
            parameters=IndexParameters(original_max_descriptors=2),
            binary_filename="index.descriptors.private.f32",
            binary_sha256="b" * 64,
            binary_byte_size=2 * 128 * 4,
            originals=records,
            scan_issues=(),
        )

    def validate(self, manifest: dict[str, object]):
        return validate_index_manifest(
            manifest, expected_binary_filename="index.descriptors.private.f32"
        )

    def test_valid_manifest_round_trips(self) -> None:
        metadata = self.validate(self.manifest())
        self.assertTrue(metadata.index_corpus_complete)
        self.assertEqual(metadata.total_descriptor_rows, 2)

    def test_manifest_contains_extractor_runtime_contract(self) -> None:
        runtime = self.manifest()["runtime"]
        self.assertEqual(
            set(runtime),
            {
                "python_version",
                "pillow_version",
                "numpy_version",
                "opencv_version",
            },
        )
        self.assertTrue(all(isinstance(value, str) and value for value in runtime.values()))

    def test_descriptor_count_above_configured_cap_is_rejected(self) -> None:
        manifest = self.manifest()
        manifest["originals"][0]["selected_descriptor_count"] = 3
        manifest["originals"][0]["descriptor_count"] = 3
        with self.assertRaisesRegex(IndexValidationError, "DESCRIPTOR_CAP_EXCEEDED"):
            self.validate(manifest)

    def test_selected_count_above_configured_cap_is_rejected_before_range_mismatch(self) -> None:
        manifest = self.manifest()
        manifest["originals"][0]["selected_descriptor_count"] = 3
        with self.assertRaisesRegex(IndexValidationError, "DESCRIPTOR_CAP_EXCEEDED"):
            self.validate(manifest)

    def test_record_exactly_at_configured_cap_is_accepted(self) -> None:
        manifest = build_index_manifest(
            parameters=IndexParameters(original_max_descriptors=2),
            binary_filename="index.descriptors.private.f32",
            binary_sha256="b" * 64,
            binary_byte_size=2 * 128 * 4,
            originals=(record("at-cap.png", 0, 2),),
            scan_issues=(),
        )
        metadata = self.validate(manifest)
        self.assertEqual(metadata.originals[0].descriptor_count, 2)

    def test_sift_extraction_limit_is_also_a_persisted_descriptor_cap(self) -> None:
        manifest = build_index_manifest(
            parameters=IndexParameters(
                sift_nfeatures=1, original_max_descriptors=2
            ),
            binary_filename="index.descriptors.private.f32",
            binary_sha256="b" * 64,
            binary_byte_size=2 * 128 * 4,
            originals=(record("over-sift-cap.png", 0, 2),),
            scan_issues=(),
        )
        with self.assertRaisesRegex(IndexValidationError, "DESCRIPTOR_CAP_EXCEEDED"):
            self.validate(manifest)

    def test_index_incomplete_when_original_has_no_descriptors(self) -> None:
        manifest = build_index_manifest(
            parameters=IndexParameters(),
            binary_filename="index.descriptors.private.f32",
            binary_sha256="b" * 64,
            binary_byte_size=0,
            originals=(record("blank.png", 0, 0),),
            scan_issues=(),
        )
        metadata = self.validate(manifest)
        self.assertFalse(metadata.index_corpus_complete)

    def test_indexed_record_requires_encoded_size(self) -> None:
        manifest = self.manifest()
        manifest["originals"][0]["size_bytes"] = None
        with self.assertRaisesRegex(IndexValidationError, "CONTRADICTORY_INDEX_STATUS"):
            self.validate(manifest)

    def test_no_descriptor_record_requires_encoded_size(self) -> None:
        manifest = build_index_manifest(
            parameters=IndexParameters(),
            binary_filename="index.descriptors.private.f32",
            binary_sha256="b" * 64,
            binary_byte_size=0,
            originals=(record("blank.png", 0, 0),),
            scan_issues=(),
        )
        manifest["originals"][0]["size_bytes"] = None
        with self.assertRaisesRegex(IndexValidationError, "CONTRADICTORY_INDEX_STATUS"):
            self.validate(manifest)

    def test_unavailable_record_allows_nullable_builder_identity(self) -> None:
        unavailable = OriginalIndexRecord(
            reference("unavailable.png"),
            None,
            None,
            None,
            None,
            IndexStatus.UNAVAILABLE,
            0,
            0,
            0,
        )
        manifest = build_index_manifest(
            parameters=IndexParameters(),
            binary_filename="index.descriptors.private.f32",
            binary_sha256="b" * 64,
            binary_byte_size=0,
            originals=(unavailable,),
            scan_issues=(),
        )
        metadata = self.validate(manifest)
        self.assertFalse(metadata.index_corpus_complete)
        self.assertIsNone(metadata.originals[0].size_bytes)

    def test_incompatible_opencv_runtime_is_rejected(self) -> None:
        manifest = self.manifest()
        manifest["runtime"]["opencv_version"] = "0.0-incompatible"
        with self.assertRaisesRegex(IndexValidationError, "INDEX_RUNTIME_MISMATCH"):
            self.validate(manifest)

    def test_other_runtime_component_changes_are_rejected(self) -> None:
        for field in ("python_version", "pillow_version", "numpy_version"):
            with self.subTest(field=field):
                manifest = self.manifest()
                manifest["runtime"][field] = "0.0-incompatible"
                with self.assertRaisesRegex(
                    IndexValidationError, "INDEX_RUNTIME_MISMATCH"
                ):
                    self.validate(manifest)

    def test_malformed_runtime_contract_is_rejected(self) -> None:
        cases = []
        missing = self.manifest()
        del missing["runtime"]["opencv_version"]
        cases.append(missing)
        wrong_type = self.manifest()
        wrong_type["runtime"]["numpy_version"] = 2
        cases.append(wrong_type)
        extra = self.manifest()
        extra["runtime"]["hostname"] = "not-allowed"
        cases.append(extra)
        for manifest in cases:
            with self.subTest(runtime=manifest["runtime"]):
                with self.assertRaisesRegex(
                    IndexValidationError, "INVALID_INDEX_RUNTIME"
                ):
                    self.validate(manifest)

    def test_invalid_offsets_are_rejected(self) -> None:
        manifest = self.manifest()
        manifest["originals"][1]["descriptor_offset"] = 0
        with self.assertRaisesRegex(IndexValidationError, "INVALID_DESCRIPTOR_RANGE"):
            self.validate(manifest)

    def test_binary_size_shape_mismatch_is_rejected(self) -> None:
        manifest = self.manifest()
        manifest["binary"]["byte_size"] = 1
        with self.assertRaisesRegex(IndexValidationError, "INVALID_BINARY_SIZE"):
            self.validate(manifest)

    def test_unsafe_semantic_reference_is_rejected(self) -> None:
        manifest = self.manifest()
        manifest["originals"][0]["original"]["relative_path"] = "../secret.png"
        with self.assertRaisesRegex(IndexValidationError, "UNSAFE_RELATIVE_PATH"):
            self.validate(manifest)

    def test_corpus_identity_corruption_is_rejected(self) -> None:
        manifest = deepcopy(self.manifest())
        manifest["corpus"]["identity_sha256"] = "0" * 64
        with self.assertRaisesRegex(IndexValidationError, "CORPUS_IDENTITY_MISMATCH"):
            self.validate(manifest)


if __name__ == "__main__":
    unittest.main()

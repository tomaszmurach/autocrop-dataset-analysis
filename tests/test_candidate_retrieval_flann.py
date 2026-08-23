"""Focused tests for the research-only OpenCV FLANN experiment."""

from __future__ import annotations

from statistics import mean, median
import tempfile
from pathlib import Path
from unittest import mock
import unittest

import numpy as np

from autocrop_analysis.audit import RootRole, SemanticReference, semantic_reference_sort_key
from autocrop_analysis.candidate_retrieval import (
    IndexStatus,
    OriginalIndexRecord,
    QueryParameters,
    RetrievalCandidate,
    retrieve_candidates,
    select_shortlist,
)
from autocrop_analysis.candidate_retrieval_flann import (
    DescriptorCorpusIdentity,
    ExperimentalFlannIndex,
    FlannExperimentError,
    FlannNeighborEvidence,
    FlannParameters,
    _validate_flann_results,
    compare_source_rankings,
)


def reference(name: str) -> SemanticReference:
    return SemanticReference(RootRole.ORIGINAL, name)


def record(name: str, offset: int, count: int) -> OriginalIndexRecord:
    return OriginalIndexRecord(
        reference(name),
        100,
        100,
        10,
        "a" * 64,
        IndexStatus.INDEXED,
        count,
        offset,
        count,
    )


def corpus(rows: int, value: str = "a") -> DescriptorCorpusIdentity:
    return DescriptorCorpusIdentity(value * 64, value * 64, rows)


def descriptor_matrix(rows: int = 8) -> np.ndarray:
    matrix = np.zeros((rows, 128), dtype=np.float32)
    for index in range(rows):
        matrix[index, 0] = float(index * 3)
        matrix[index, 1] = float(index * 4)
    return matrix


class ParameterAndValidationTests(unittest.TestCase):
    def test_parameters_require_positive_plain_integers(self) -> None:
        for values in (
            {"trees": 0},
            {"checks": -1},
            {"neighbor_depth": 0},
            {"trees": True},
        ):
            with self.subTest(values=values):
                with self.assertRaisesRegex(
                    FlannExperimentError, "INVALID_FLANN_PARAMETER"
                ):
                    FlannParameters(**values)

    def test_descriptor_matrix_dimension_dtype_and_contiguity_are_strict(self) -> None:
        index = ExperimentalFlannIndex()
        invalid = (
            np.zeros((2, 127), dtype=np.float32),
            np.zeros((2, 128), dtype=np.float64),
            np.zeros((4, 128), dtype=np.float32)[::2],
        )
        for matrix in invalid:
            with self.subTest(shape=matrix.shape, dtype=str(matrix.dtype)):
                with self.assertRaises(FlannExperimentError):
                    index.build(matrix, FlannParameters(trees=1))

    def test_malformed_neighbor_shapes_indices_and_distances_are_rejected(self) -> None:
        valid_indices = np.array([[0, 1]], dtype=np.int32)
        valid_distances = np.array([[0.0, 1.0]], dtype=np.float32)
        invalid_cases = (
            (valid_indices[:, :1], valid_distances),
            (np.array([[0, -1]], dtype=np.int32), valid_distances),
            (np.array([[0, 2]], dtype=np.int32), valid_distances),
            (valid_indices, np.array([[0.0, -1.0]], dtype=np.float32)),
            (valid_indices, np.array([[0.0, np.inf]], dtype=np.float32)),
            (valid_indices.astype(np.float32), valid_distances),
        )
        for indices, distances in invalid_cases:
            with self.subTest(indices=indices.tolist(), distances=distances.tolist()):
                with self.assertRaises(FlannExperimentError):
                    _validate_flann_results(
                        indices,
                        distances,
                        expected_queries=1,
                        expected_depth=2,
                        descriptor_rows=2,
                    )

    def test_query_dimension_dtype_and_finite_values_are_strict(self) -> None:
        matrix = descriptor_matrix()
        parameters = FlannParameters(trees=1, checks=32, neighbor_depth=2)
        index = ExperimentalFlannIndex()
        try:
            index.build(matrix, parameters)
            invalid = (
                np.zeros((1, 127), dtype=np.float32),
                np.zeros((1, 128), dtype=np.float64),
                np.full((1, 128), np.nan, dtype=np.float32),
            )
            for query in invalid:
                with self.subTest(shape=query.shape, dtype=str(query.dtype)):
                    with self.assertRaises(FlannExperimentError):
                        index.search(query, parameters)
        finally:
            index.release()


class DistanceAndLifecycleTests(unittest.TestCase):
    def test_descriptor_row_and_distance_signatures_are_independent(self) -> None:
        rows = np.array([[1, 2]], dtype=np.int32)
        first = FlannNeighborEvidence(
            rows,
            np.array([[1.0, 4.0]], dtype=np.float32),
            np.array([[1.0, 2.0]], dtype=np.float64),
        )
        second = FlannNeighborEvidence(
            rows.copy(),
            np.array([[1.0, 9.0]], dtype=np.float32),
            np.array([[1.0, 3.0]], dtype=np.float64),
        )

        self.assertEqual(
            first.descriptor_row_signature(), second.descriptor_row_signature()
        )
        self.assertNotEqual(
            first.descriptor_distance_signature(),
            second.descriptor_distance_signature(),
        )

    def test_installed_opencv_returns_squared_l2_and_search_normalizes_it(self) -> None:
        matrix = np.zeros((2, 128), dtype=np.float32)
        matrix[1, 0] = 3.0
        matrix[1, 1] = 4.0
        query = np.zeros((1, 128), dtype=np.float32)
        index = ExperimentalFlannIndex()
        try:
            parameters = FlannParameters(trees=1, checks=32, neighbor_depth=2)
            index.build(matrix, parameters)
            evidence = index.search(query, parameters)
        finally:
            index.release()

        order = np.argsort(evidence.row_indices[0])
        np.testing.assert_array_equal(evidence.row_indices[0][order], [0, 1])
        np.testing.assert_allclose(
            evidence.squared_l2_distances[0][order], [0.0, 25.0]
        )
        np.testing.assert_allclose(evidence.l2_distances[0][order], [0.0, 5.0])

    def test_build_search_release_and_zero_query_lifecycle(self) -> None:
        matrix = descriptor_matrix()
        parameters = FlannParameters(trees=1, checks=64, neighbor_depth=3)
        index = ExperimentalFlannIndex()
        index.build(matrix, parameters)
        self.assertTrue(index.is_ready)
        evidence = index.search(matrix[:2], parameters)
        self.assertEqual(evidence.row_indices.shape, (2, 3))
        empty = index.search(np.empty((0, 128), dtype=np.float32), parameters)
        self.assertEqual(empty.row_indices.shape, (0, 3))
        self.assertEqual(
            index.retrieve_candidates(
                np.empty((0, 128), dtype=np.float32),
                (record("all.png", 0, 8),),
                parameters,
            ),
            (),
        )
        index.release()
        self.assertFalse(index.is_ready)
        with self.assertRaisesRegex(FlannExperimentError, "FLANN_INDEX_NOT_READY"):
            index.search(matrix[:1], parameters)

    def test_save_reload_preserves_results_and_validates_corpus_link(self) -> None:
        matrix = descriptor_matrix()
        parameters = FlannParameters(trees=2, checks=64, neighbor_depth=4)
        identity = corpus(len(matrix))
        with tempfile.TemporaryDirectory() as temporary:
            artifact_path = Path(temporary) / "index.flann"
            built = ExperimentalFlannIndex()
            loaded = ExperimentalFlannIndex()
            try:
                built.build(matrix, parameters)
                before = built.search(matrix[:3], parameters)
                artifact = built.save(artifact_path, identity)
                built.release()
                loaded.load(matrix, artifact_path, artifact, identity)
                after = loaded.search(matrix[:3], parameters)
            finally:
                built.release()
                loaded.release()

            np.testing.assert_array_equal(before.row_indices, after.row_indices)
            np.testing.assert_array_equal(
                before.squared_l2_distances, after.squared_l2_distances
            )
            self.assertGreater(artifact.artifact_bytes, 0)
            self.assertEqual(artifact.corpus, identity)

            mismatch = ExperimentalFlannIndex()
            with self.assertRaisesRegex(
                FlannExperimentError, "FLANN_CORPUS_IDENTITY_MISMATCH"
            ):
                mismatch.load(matrix, artifact_path, artifact, corpus(len(matrix), "b"))

    def test_save_failure_removes_partial_artifact(self) -> None:
        matrix = descriptor_matrix()
        parameters = FlannParameters(trees=1, checks=32, neighbor_depth=2)
        with tempfile.TemporaryDirectory() as temporary:
            artifact_path = Path(temporary) / "partial.flann"
            index = ExperimentalFlannIndex()
            try:
                index.build(matrix, parameters)
                with mock.patch(
                    "autocrop_analysis.candidate_retrieval_flann.hash_file",
                    side_effect=OSError("simulated hash failure"),
                ):
                    with self.assertRaises(OSError):
                        index.save(artifact_path, corpus(len(matrix)))
            finally:
                index.release()
            self.assertFalse(artifact_path.exists())

    def test_reload_rejects_tampered_artifact(self) -> None:
        matrix = descriptor_matrix()
        parameters = FlannParameters(trees=1, checks=32, neighbor_depth=2)
        identity = corpus(len(matrix))
        with tempfile.TemporaryDirectory() as temporary:
            artifact_path = Path(temporary) / "tampered.flann"
            built = ExperimentalFlannIndex()
            try:
                built.build(matrix, parameters)
                artifact = built.save(artifact_path, identity)
            finally:
                built.release()
            with artifact_path.open("ab") as stream:
                stream.write(b"tamper")
            loaded = ExperimentalFlannIndex()
            with self.assertRaisesRegex(
                FlannExperimentError, "FLANN_ARTIFACT_INTEGRITY_MISMATCH"
            ):
                loaded.load(matrix, artifact_path, artifact, identity)

    def test_independent_bounded_rebuilds_are_valid_without_assuming_stability(self) -> None:
        matrix = descriptor_matrix(12)
        records = (record("first.png", 0, 6), record("second.png", 6, 6))
        parameters = FlannParameters(trees=2, checks=128, neighbor_depth=6)
        results = []
        rows = []
        for _ in range(2):
            index = ExperimentalFlannIndex()
            try:
                index.build(matrix, parameters)
                evidence = index.search(matrix[7:9], parameters)
                rows.append(evidence.row_indices.tolist())
                results.append(index.retrieve_candidates(matrix[7:9], records, parameters))
            finally:
                index.release()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(len(item) == 2 and len(item[0]) == 6 for item in rows))
        self.assertTrue(
            all(
                ranked and ranked[0].original == records[1].reference
                for ranked in results
            )
        )


class AggregationAndComparisonTests(unittest.TestCase):
    @staticmethod
    def legacy_reference(
        query: np.ndarray,
        matrix: np.ndarray,
        records: tuple[OriginalIndexRecord, ...],
        parameters: QueryParameters,
    ) -> tuple[RetrievalCandidate, ...]:
        import cv2

        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        nearest = [[] for _ in range(query.shape[0])]
        for block_start in range(0, matrix.shape[0], parameters.descriptor_block_rows):
            block = np.ascontiguousarray(
                matrix[block_start : block_start + parameters.descriptor_block_rows],
                dtype=np.float32,
            )
            rows = matcher.knnMatch(
                query, block, k=min(parameters.neighbor_depth, block.shape[0])
            )
            for query_index, matches in enumerate(rows):
                nearest[query_index].extend(
                    (float(match.distance), block_start + int(match.trainIdx))
                    for match in matches
                )
                nearest[query_index].sort(key=lambda item: (item[0], item[1]))
                del nearest[query_index][parameters.neighbor_depth :]
        range_ends = np.array(
            [item.descriptor_offset + item.descriptor_count for item in records]
        )
        support: dict[int, list[float]] = {}
        for matches in nearest:
            best: dict[int, float] = {}
            for distance, row in matches:
                owner = int(np.searchsorted(range_ends, row, side="right"))
                best[owner] = min(distance, best.get(owner, distance))
            for owner, distance in best.items():
                support.setdefault(owner, []).append(distance)
        candidates = tuple(
            RetrievalCandidate(
                records[owner].reference,
                len(distances),
                float(median(distances)),
                float(mean(distances)),
            )
            for owner, distances in support.items()
        )
        return tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.supporting_query_descriptors,
                    candidate.median_best_l2_distance,
                    candidate.mean_best_l2_distance,
                    semantic_reference_sort_key(candidate.original),
                ),
            )
        )

    def test_exact_bf_refactor_preserves_candidates_distances_and_shortlist(self) -> None:
        rng = np.random.default_rng(123)
        matrix = rng.normal(100, 20, (18, 128)).astype(np.float32)
        query = np.ascontiguousarray(matrix[[2, 7, 13]] + 0.25)
        records = (
            record("c.png", 0, 6),
            record("a.png", 6, 6),
            record("b.png", 12, 6),
        )
        parameters = QueryParameters(
            neighbor_depth=9, requested_k=2, descriptor_block_rows=7
        )
        expected = self.legacy_reference(query, matrix, records, parameters)
        actual = retrieve_candidates(query, matrix, records, parameters)
        self.assertEqual(actual, expected)
        self.assertEqual(select_shortlist(actual, 2), select_shortlist(expected, 2))

    def test_identical_neighbor_evidence_produces_equivalent_source_ranking(self) -> None:
        matrix = descriptor_matrix(8)
        records = (record("first.png", 0, 4), record("second.png", 4, 4))
        query = np.ascontiguousarray(matrix[[1, 6]])
        exact = retrieve_candidates(
            query,
            matrix,
            records,
            QueryParameters(neighbor_depth=8, requested_k=2),
        )
        index = ExperimentalFlannIndex()
        try:
            parameters = FlannParameters(trees=1, checks=256, neighbor_depth=8)
            index.build(matrix, parameters)
            approximate = index.retrieve_candidates(query, records, parameters)
        finally:
            index.release()
        self.assertEqual(approximate, exact)

    def test_oracle_comparison_reports_tie_extended_membership(self) -> None:
        exact = (
            RetrievalCandidate(reference("a.png"), 3, 1.0, 1.0),
            RetrievalCandidate(reference("source.png"), 2, 2.0, 2.0),
            RetrievalCandidate(reference("tie.png"), 2, 3.0, 3.0),
        )
        flann = (exact[0], exact[2], exact[1])
        report = compare_source_rankings(
            exact,
            flann,
            k_values=(1, 2),
            known_source=reference("source.png"),
        )
        self.assertTrue(report["exact_top_source_retained"])
        self.assertTrue(report["by_k"]["1"]["exact_top_source_flann_present"])
        self.assertEqual(report["known_source_rank_delta"], 1)
        self.assertTrue(report["by_k"]["2"]["known_source_flann_present"])
        self.assertTrue(report["by_k"]["2"]["exact_contained_by_flann"])
        self.assertEqual(report["by_k"]["2"]["exact_tie_extension"], 1)


if __name__ == "__main__":
    unittest.main()

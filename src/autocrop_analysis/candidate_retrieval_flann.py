"""Research-only OpenCV FLANN feasibility primitives.

This module does not participate in the candidate-retrieval CLI or provenance
pipeline.  Exact BF remains the retrieval oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .candidate_retrieval import (
    DESCRIPTOR_DIMENSION,
    DESCRIPTOR_DTYPE,
    OriginalIndexRecord,
    RetrievalCandidate,
    _rank_descriptor_neighbors,
)
from .candidate_retrieval_profiling import RetrievalProfiler, profiling_stage


FLANN_INDEX_KDTREE = 1
FLANN_DISTANCE_L2 = 1
FLANN_EXPERIMENT_ALGORITHM = "OPENCV_FLANN_RANDOMIZED_KDTREE_L2"
_HASH_CHUNK_SIZE = 1024 * 1024


class FlannExperimentError(ValueError):
    """Structured failure raised by the research-only FLANN path."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class FlannParameters:
    trees: int = 4
    checks: int = 128
    neighbor_depth: int = 32

    def __post_init__(self) -> None:
        for value in (self.trees, self.checks, self.neighbor_depth):
            if type(value) is not int or value <= 0:
                raise FlannExperimentError("INVALID_FLANN_PARAMETER")

    def as_dict(self) -> dict[str, int]:
        return {
            "trees": self.trees,
            "checks": self.checks,
            "neighbor_depth": self.neighbor_depth,
        }


@dataclass(frozen=True, slots=True)
class DescriptorCorpusIdentity:
    corpus_identity_sha256: str
    descriptor_sha256: str
    descriptor_rows: int
    descriptor_dimension: int = DESCRIPTOR_DIMENSION
    descriptor_dtype: str = DESCRIPTOR_DTYPE

    def __post_init__(self) -> None:
        if not _is_sha256(self.corpus_identity_sha256) or not _is_sha256(
            self.descriptor_sha256
        ):
            raise FlannExperimentError("INVALID_DESCRIPTOR_CORPUS_IDENTITY")
        if type(self.descriptor_rows) is not int or self.descriptor_rows <= 0:
            raise FlannExperimentError("INVALID_DESCRIPTOR_CORPUS_IDENTITY")
        if self.descriptor_dimension != DESCRIPTOR_DIMENSION:
            raise FlannExperimentError("INVALID_DESCRIPTOR_CORPUS_IDENTITY")
        if self.descriptor_dtype != DESCRIPTOR_DTYPE:
            raise FlannExperimentError("INVALID_DESCRIPTOR_CORPUS_IDENTITY")

    def as_dict(self) -> dict[str, object]:
        return {
            "corpus_identity_sha256": self.corpus_identity_sha256,
            "descriptor_sha256": self.descriptor_sha256,
            "descriptor_rows": self.descriptor_rows,
            "descriptor_dimension": self.descriptor_dimension,
            "descriptor_dtype": self.descriptor_dtype,
        }


@dataclass(frozen=True, slots=True)
class FlannArtifactRecord:
    corpus: DescriptorCorpusIdentity
    trees: int
    artifact_sha256: str
    artifact_bytes: int
    opencv_version: str

    def __post_init__(self) -> None:
        if type(self.trees) is not int or self.trees <= 0:
            raise FlannExperimentError("INVALID_FLANN_ARTIFACT_RECORD")
        if not _is_sha256(self.artifact_sha256):
            raise FlannExperimentError("INVALID_FLANN_ARTIFACT_RECORD")
        if type(self.artifact_bytes) is not int or self.artifact_bytes <= 0:
            raise FlannExperimentError("INVALID_FLANN_ARTIFACT_RECORD")
        if not isinstance(self.opencv_version, str) or not self.opencv_version:
            raise FlannExperimentError("INVALID_FLANN_ARTIFACT_RECORD")

    def as_dict(self) -> dict[str, object]:
        return {
            "status": "EXPERIMENTAL_TEMPORARY_DERIVED_ARTIFACT",
            "algorithm": FLANN_EXPERIMENT_ALGORITHM,
            "trees": self.trees,
            "artifact_sha256": self.artifact_sha256,
            "artifact_bytes": self.artifact_bytes,
            "opencv_version": self.opencv_version,
            "descriptor_corpus": self.corpus.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class FlannNeighborEvidence:
    row_indices: np.ndarray
    squared_l2_distances: np.ndarray
    l2_distances: np.ndarray

    def nearest(self) -> tuple[tuple[tuple[float, int], ...], ...]:
        return tuple(
            tuple(
                (float(self.l2_distances[query_index, match_index]), int(row_index))
                for match_index, row_index in enumerate(rows)
            )
            for query_index, rows in enumerate(self.row_indices)
        )

    def descriptor_row_signature(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(int(row_index) for row_index in rows) for rows in self.row_indices
        )

    def descriptor_distance_signature(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(float(distance) for distance in distances)
            for distances in self.squared_l2_distances
        )


class ExperimentalFlannIndex:
    """Explicit build/search/save/load/release lifecycle for one FLANN index."""

    def __init__(self) -> None:
        self._index: cv2.flann_Index | None = None
        self._descriptor_matrix: np.ndarray | None = None
        self._trees: int | None = None

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._descriptor_matrix is not None

    @property
    def trees(self) -> int | None:
        return self._trees

    def build(
        self,
        descriptor_matrix: np.ndarray,
        parameters: FlannParameters,
        *,
        profiler: RetrievalProfiler | None = None,
    ) -> None:
        if self.is_ready:
            raise FlannExperimentError("FLANN_INDEX_ALREADY_INITIALIZED")
        matrix = _validate_descriptor_matrix(descriptor_matrix)
        index = cv2.flann_Index()
        try:
            with profiling_stage(profiler, "flann.index_build") as timing:
                index.build(
                    matrix,
                    {"algorithm": FLANN_INDEX_KDTREE, "trees": parameters.trees},
                    FLANN_DISTANCE_L2,
                )
                if profiler is not None:
                    timing.add_work(
                        descriptor_rows=int(matrix.shape[0]),
                        descriptor_dimension=int(matrix.shape[1]),
                        trees=parameters.trees,
                    )
            if int(index.getAlgorithm()) != FLANN_INDEX_KDTREE:
                raise FlannExperimentError("UNEXPECTED_FLANN_INDEX_ALGORITHM")
            if int(index.getDistance()) != FLANN_DISTANCE_L2:
                raise FlannExperimentError("UNEXPECTED_FLANN_DISTANCE")
        except Exception:
            index.release()
            raise
        self._index = index
        self._descriptor_matrix = matrix
        self._trees = parameters.trees

    def save(
        self,
        path: Path,
        corpus: DescriptorCorpusIdentity,
        *,
        profiler: RetrievalProfiler | None = None,
    ) -> FlannArtifactRecord:
        index = self._require_ready()
        resolved = path.resolve(strict=False)
        if resolved.exists():
            raise FlannExperimentError("FLANN_ARTIFACT_ALREADY_EXISTS")
        if not resolved.parent.exists() or not resolved.parent.is_dir():
            raise FlannExperimentError("FLANN_ARTIFACT_PARENT_MISSING")
        if self._descriptor_matrix is None or corpus.descriptor_rows != int(
            self._descriptor_matrix.shape[0]
        ):
            raise FlannExperimentError("FLANN_CORPUS_IDENTITY_MISMATCH")
        try:
            with profiling_stage(profiler, "flann.index_save") as timing:
                index.save(str(resolved))
                artifact_sha256, artifact_bytes = hash_file(resolved)
                if profiler is not None:
                    timing.add_work(artifact_bytes=artifact_bytes)
        except Exception:
            try:
                resolved.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        if artifact_bytes <= 0:
            resolved.unlink(missing_ok=True)
            raise FlannExperimentError("EMPTY_FLANN_ARTIFACT")
        assert self._trees is not None
        return FlannArtifactRecord(
            corpus=corpus,
            trees=self._trees,
            artifact_sha256=artifact_sha256,
            artifact_bytes=artifact_bytes,
            opencv_version=cv2.__version__,
        )

    def load(
        self,
        descriptor_matrix: np.ndarray,
        path: Path,
        artifact: FlannArtifactRecord,
        expected_corpus: DescriptorCorpusIdentity,
        *,
        profiler: RetrievalProfiler | None = None,
    ) -> None:
        if self.is_ready:
            raise FlannExperimentError("FLANN_INDEX_ALREADY_INITIALIZED")
        matrix = _validate_descriptor_matrix(descriptor_matrix)
        resolved = path.resolve(strict=False)
        if artifact.corpus != expected_corpus:
            raise FlannExperimentError("FLANN_CORPUS_IDENTITY_MISMATCH")
        if artifact.opencv_version != cv2.__version__:
            raise FlannExperimentError("FLANN_OPENCV_VERSION_MISMATCH")
        if expected_corpus.descriptor_rows != int(matrix.shape[0]):
            raise FlannExperimentError("FLANN_CORPUS_IDENTITY_MISMATCH")
        try:
            artifact_sha256, artifact_bytes = hash_file(resolved)
        except OSError as exc:
            raise FlannExperimentError("FLANN_ARTIFACT_READ_FAILED") from exc
        if (
            artifact_sha256 != artifact.artifact_sha256
            or artifact_bytes != artifact.artifact_bytes
        ):
            raise FlannExperimentError("FLANN_ARTIFACT_INTEGRITY_MISMATCH")

        index = cv2.flann_Index()
        try:
            with profiling_stage(profiler, "flann.index_load") as timing:
                loaded = bool(index.load(matrix, str(resolved)))
                if profiler is not None:
                    timing.add_work(
                        artifact_bytes=artifact_bytes,
                        descriptor_rows=int(matrix.shape[0]),
                    )
            if not loaded:
                raise FlannExperimentError("FLANN_ARTIFACT_LOAD_FAILED")
            if int(index.getAlgorithm()) != FLANN_INDEX_KDTREE:
                raise FlannExperimentError("UNEXPECTED_FLANN_INDEX_ALGORITHM")
            if int(index.getDistance()) != FLANN_DISTANCE_L2:
                raise FlannExperimentError("UNEXPECTED_FLANN_DISTANCE")
        except Exception:
            index.release()
            raise
        self._index = index
        self._descriptor_matrix = matrix
        self._trees = artifact.trees

    def search(
        self,
        query_descriptors: np.ndarray,
        parameters: FlannParameters,
        *,
        profiler: RetrievalProfiler | None = None,
        item_ordinal: int | None = None,
    ) -> FlannNeighborEvidence:
        index = self._require_ready()
        matrix = self._descriptor_matrix
        assert matrix is not None
        if parameters.trees != self._trees:
            raise FlannExperimentError("FLANN_TREE_COUNT_MISMATCH")
        query = _validate_query_descriptors(query_descriptors)
        depth = min(parameters.neighbor_depth, int(matrix.shape[0]))
        if query.shape[0] == 0:
            return FlannNeighborEvidence(
                np.empty((0, depth), dtype=np.int32),
                np.empty((0, depth), dtype=np.float32),
                np.empty((0, depth), dtype=np.float64),
            )
        with profiling_stage(
            profiler, "query.flann_search", item_ordinal=item_ordinal
        ) as timing:
            indices, squared_distances = index.knnSearch(
                query,
                depth,
                params={"checks": parameters.checks},
            )
            if profiler is not None:
                timing.add_work(
                    selected_query_descriptor_rows=int(query.shape[0]),
                    indexed_descriptor_rows=int(matrix.shape[0]),
                    returned_descriptor_neighbors=int(query.shape[0] * depth),
                    checks=parameters.checks,
                    trees=parameters.trees,
                )
        validated_indices, validated_squared = _validate_flann_results(
            indices,
            squared_distances,
            expected_queries=int(query.shape[0]),
            expected_depth=depth,
            descriptor_rows=int(matrix.shape[0]),
        )
        with profiling_stage(
            profiler,
            "query.flann_distance_normalization",
            item_ordinal=item_ordinal,
        ) as timing:
            l2_distances = np.sqrt(validated_squared.astype(np.float64, copy=False))
            if profiler is not None:
                timing.add_work(normalized_distances=int(l2_distances.size))
        return FlannNeighborEvidence(
            validated_indices, validated_squared, l2_distances
        )

    def retrieve_candidates(
        self,
        query_descriptors: np.ndarray,
        originals: Sequence[OriginalIndexRecord],
        parameters: FlannParameters,
        *,
        profiler: RetrievalProfiler | None = None,
        item_ordinal: int | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        matrix = self._descriptor_matrix
        self._require_ready()
        assert matrix is not None
        query = _validate_query_descriptors(query_descriptors)
        if query.shape[0] == 0:
            return ()
        evidence = self.search(
            query,
            parameters,
            profiler=profiler,
            item_ordinal=item_ordinal,
        )
        return _rank_descriptor_neighbors(
            evidence.nearest(),
            descriptor_rows=int(matrix.shape[0]),
            originals=originals,
            profiler=profiler,
            item_ordinal=item_ordinal,
        )

    def release(self) -> None:
        if self._index is not None:
            self._index.release()
        self._index = None
        self._descriptor_matrix = None
        self._trees = None

    def _require_ready(self) -> cv2.flann_Index:
        if self._index is None or self._descriptor_matrix is None:
            raise FlannExperimentError("FLANN_INDEX_NOT_READY")
        return self._index


def _validate_descriptor_matrix(matrix: np.ndarray) -> np.ndarray:
    if not isinstance(matrix, np.ndarray):
        raise FlannExperimentError("INVALID_FLANN_DESCRIPTOR_MATRIX")
    if matrix.ndim != 2 or matrix.shape[1] != DESCRIPTOR_DIMENSION:
        raise FlannExperimentError("INVALID_FLANN_DESCRIPTOR_MATRIX")
    if matrix.shape[0] <= 0 or matrix.dtype != np.dtype(DESCRIPTOR_DTYPE):
        raise FlannExperimentError("INVALID_FLANN_DESCRIPTOR_MATRIX")
    if not matrix.flags.c_contiguous:
        raise FlannExperimentError("NONCONTIGUOUS_FLANN_DESCRIPTOR_MATRIX")
    return matrix


def _validate_query_descriptors(query: np.ndarray) -> np.ndarray:
    if not isinstance(query, np.ndarray):
        raise FlannExperimentError("INVALID_FLANN_QUERY_DESCRIPTORS")
    if query.ndim != 2 or query.shape[1] != DESCRIPTOR_DIMENSION:
        raise FlannExperimentError("INVALID_FLANN_QUERY_DESCRIPTORS")
    if query.dtype != np.dtype(DESCRIPTOR_DTYPE) or not query.flags.c_contiguous:
        raise FlannExperimentError("INVALID_FLANN_QUERY_DESCRIPTORS")
    if query.size and not bool(np.isfinite(query).all()):
        raise FlannExperimentError("INVALID_FLANN_QUERY_DESCRIPTORS")
    return query


def _validate_flann_results(
    indices: np.ndarray,
    squared_distances: np.ndarray,
    *,
    expected_queries: int,
    expected_depth: int,
    descriptor_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(indices, np.ndarray) or not isinstance(
        squared_distances, np.ndarray
    ):
        raise FlannExperimentError("MALFORMED_FLANN_RESULT")
    expected_shape = (expected_queries, expected_depth)
    if indices.shape != expected_shape or squared_distances.shape != expected_shape:
        raise FlannExperimentError("MALFORMED_FLANN_RESULT")
    if indices.dtype.kind not in "iu" or squared_distances.dtype.kind != "f":
        raise FlannExperimentError("MALFORMED_FLANN_RESULT")
    if squared_distances.size and (
        not bool(np.isfinite(squared_distances).all())
        or bool(np.any(squared_distances < 0.0))
    ):
        raise FlannExperimentError("INVALID_FLANN_DISTANCE")
    if indices.size and (
        bool(np.any(indices < 0)) or bool(np.any(indices >= descriptor_rows))
    ):
        raise FlannExperimentError("INVALID_FLANN_NEIGHBOR_ROW")
    return (
        np.ascontiguousarray(indices, dtype=np.int32),
        np.ascontiguousarray(squared_distances, dtype=np.float32),
    )


def hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def source_rank(
    candidates: Sequence[RetrievalCandidate], source
) -> int | None:
    return next(
        (
            rank
            for rank, candidate in enumerate(candidates, start=1)
            if candidate.original == source
        ),
        None,
    )


def compare_source_rankings(
    exact_candidates: Sequence[RetrievalCandidate],
    flann_candidates: Sequence[RetrievalCandidate],
    *,
    k_values: Sequence[int],
    known_source=None,
) -> dict[str, object]:
    """Compare final source rankings and tie-extended shortlist membership."""

    from .candidate_retrieval import select_shortlist

    if not k_values or any(type(k) is not int or k <= 0 for k in k_values):
        raise FlannExperimentError("INVALID_COMPARISON_K_VALUES")
    exact = tuple(exact_candidates)
    flann = tuple(flann_candidates)
    exact_top = exact[0].original if exact else None
    exact_top_flann_rank = source_rank(flann, exact_top) if exact_top is not None else None
    known_exact_rank = source_rank(exact, known_source) if known_source is not None else None
    known_flann_rank = source_rank(flann, known_source) if known_source is not None else None
    by_k: dict[str, object] = {}
    for k in k_values:
        exact_shortlist, exact_extension, _ = select_shortlist(exact, k)
        flann_shortlist, flann_extension, _ = select_shortlist(flann, k)
        exact_refs = {candidate.original for candidate in exact_shortlist}
        flann_refs = {candidate.original for candidate in flann_shortlist}
        overlap = exact_refs & flann_refs
        by_k[str(k)] = {
            "exact_shortlist_size": len(exact_refs),
            "flann_shortlist_size": len(flann_refs),
            "exact_tie_extension": exact_extension,
            "flann_tie_extension": flann_extension,
            "overlap_count": len(overlap),
            "exact_contained_by_flann": exact_refs <= flann_refs,
            "exact_candidate_retention": (
                len(overlap) / len(exact_refs) if exact_refs else 1.0
            ),
            "exact_top_source_flann_present": (
                exact_top in flann_refs if exact_top is not None else None
            ),
            "known_source_exact_present": (
                known_source in exact_refs if known_source is not None else None
            ),
            "known_source_flann_present": (
                known_source in flann_refs if known_source is not None else None
            ),
        }
    return {
        "exact_top_source": (
            exact_top.relative_path if exact_top is not None else None
        ),
        "exact_top_source_flann_rank": exact_top_flann_rank,
        "exact_top_source_retained": exact_top_flann_rank is not None,
        "known_source_exact_rank": known_exact_rank,
        "known_source_flann_rank": known_flann_rank,
        "known_source_rank_delta": (
            known_flann_rank - known_exact_rank
            if known_exact_rank is not None and known_flann_rank is not None
            else None
        ),
        "by_k": by_k,
    }

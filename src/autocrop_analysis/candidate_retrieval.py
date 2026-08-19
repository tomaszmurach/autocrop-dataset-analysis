"""Deterministic compact-SIFT candidate retrieval primitives.

This subsystem produces ranked retrieval shortlists only.  It deliberately
does not make content-provenance decisions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import platform
from statistics import mean, median
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
import PIL

from . import __version__
from .audit import RootRole, SemanticReference, semantic_reference_sort_key
from .candidate_retrieval_profiling import RetrievalProfiler, profiling_stage
from .content_matching import FeatureImage


INDEX_SCHEMA_VERSION = "1.0"
RETRIEVAL_SCHEMA_VERSION = "1.0"
ALGORITHM_NAME = "COMPACT_SPATIAL_SIFT_EXACT_BF_L2_VOTING"
ALGORITHM_STATUS = "EXPERIMENTAL_CANDIDATE_RETRIEVAL_NOT_PROVENANCE"
DESCRIPTOR_DIMENSION = 128
DESCRIPTOR_DTYPE = "<f4"
DESCRIPTOR_ITEMSIZE = 4
DEFAULT_DESCRIPTOR_BLOCK_ROWS = 65_536
RUNTIME_VERSION_FIELDS = (
    "python_version",
    "pillow_version",
    "numpy_version",
    "opencv_version",
)


class IndexStatus(str, Enum):
    INDEXED = "INDEXED"
    NO_DESCRIPTORS = "NO_DESCRIPTORS"
    UNAVAILABLE = "UNAVAILABLE"


class QueryStatus(str, Enum):
    RETRIEVED = "RETRIEVED"
    NO_QUERY_DESCRIPTORS = "NO_QUERY_DESCRIPTORS"
    QUERY_UNAVAILABLE = "QUERY_UNAVAILABLE"
    INDEX_INCOMPLETE = "INDEX_INCOMPLETE"


class IndexValidationError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class IndexParameters:
    sift_nfeatures: int = 3_000
    original_max_descriptors: int = 128
    grid_rows: int = 4
    grid_columns: int = 4
    random_seed: int = 17_029

    def __post_init__(self) -> None:
        for value in (
            self.sift_nfeatures,
            self.original_max_descriptors,
            self.grid_rows,
            self.grid_columns,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError("INVALID_INDEX_PARAMETER")
        if type(self.random_seed) is not int:
            raise ValueError("INVALID_INDEX_PARAMETER")

    def as_dict(self) -> dict[str, int]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class QueryParameters:
    query_max_descriptors: int = 64
    neighbor_depth: int = 32
    requested_k: int = 50
    descriptor_block_rows: int = DEFAULT_DESCRIPTOR_BLOCK_ROWS

    def __post_init__(self) -> None:
        for value in (
            self.query_max_descriptors,
            self.neighbor_depth,
            self.requested_k,
            self.descriptor_block_rows,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError("INVALID_QUERY_PARAMETER")

    def as_dict(self) -> dict[str, int]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class OriginalIndexRecord:
    reference: SemanticReference
    display_width: int | None
    display_height: int | None
    size_bytes: int | None
    encoded_sha256: str | None
    status: IndexStatus
    selected_descriptor_count: int
    descriptor_offset: int
    descriptor_count: int

    def identity_dict(self) -> dict[str, object]:
        return {
            "root_role": self.reference.root_role.value,
            "relative_path": self.reference.relative_path,
            "display_width": self.display_width,
            "display_height": self.display_height,
            "size_bytes": self.size_bytes,
            "encoded_sha256": self.encoded_sha256,
            "index_status": self.status.value,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "original": {
                "root_role": self.reference.root_role.value,
                "relative_path": self.reference.relative_path,
            },
            "display_width": self.display_width,
            "display_height": self.display_height,
            "size_bytes": self.size_bytes,
            "encoded_sha256": self.encoded_sha256,
            "index_status": self.status.value,
            "selected_descriptor_count": self.selected_descriptor_count,
            "descriptor_offset": self.descriptor_offset,
            "descriptor_count": self.descriptor_count,
        }


@dataclass(frozen=True, slots=True)
class IndexMetadata:
    parameters: IndexParameters
    corpus_identity_sha256: str
    binary_filename: str
    binary_sha256: str
    binary_byte_size: int
    total_descriptor_rows: int
    index_corpus_complete: bool
    originals: tuple[OriginalIndexRecord, ...]
    scan_issues: tuple[dict[str, str], ...]


def current_runtime_contract() -> dict[str, str]:
    """Return the deterministic extractor-runtime compatibility contract."""

    return {
        "python_version": platform.python_version(),
        "pillow_version": PIL.__version__,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
    }


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    original: SemanticReference
    supporting_query_descriptors: int
    median_best_l2_distance: float
    mean_best_l2_distance: float


@dataclass(frozen=True, slots=True)
class RetrievalQueryResult:
    crop: SemanticReference
    display_width: int | None
    display_height: int | None
    status: QueryStatus
    retrieval_query_complete: bool
    extracted_descriptor_count: int
    selected_descriptor_count: int
    requested_k: int
    returned_candidate_count: int
    tie_extension_count: int
    boundary_support_votes: int | None
    ranked_candidates: tuple[RetrievalCandidate, ...]


def select_spatially_balanced_descriptors(
    image: FeatureImage,
    *,
    maximum: int,
    grid_rows: int,
    grid_columns: int,
) -> np.ndarray:
    """Select strong descriptors in balanced deterministic grid rounds."""

    if maximum <= 0 or grid_rows <= 0 or grid_columns <= 0:
        raise ValueError("INVALID_SAMPLING_PARAMETER")
    if (
        image.descriptors is None
        or not image.keypoints
        or image.width <= 0
        or image.height <= 0
    ):
        return _empty_descriptors()
    descriptors = np.asarray(image.descriptors)
    if descriptors.ndim != 2 or descriptors.shape[1] != DESCRIPTOR_DIMENSION:
        raise ValueError("INVALID_SIFT_DESCRIPTOR_SHAPE")
    if len(image.keypoints) != descriptors.shape[0]:
        raise ValueError("KEYPOINT_DESCRIPTOR_COUNT_MISMATCH")

    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, keypoint in enumerate(image.keypoints):
        x, y = float(keypoint.pt[0]), float(keypoint.pt[1])
        column = min(grid_columns - 1, max(0, int(x * grid_columns / image.width)))
        row = min(grid_rows - 1, max(0, int(y * grid_rows / image.height)))
        cells[(row, column)].append(index)

    for indices in cells.values():
        indices.sort(key=lambda index: _sampler_key(image.keypoints[index], index))

    selected: list[int] = []
    depth = 0
    stable_cells = tuple(sorted(cells))
    while len(selected) < maximum:
        round_indices = [
            cells[cell][depth]
            for cell in stable_cells
            if depth < len(cells[cell])
        ]
        if not round_indices:
            break
        round_indices.sort(key=lambda index: _sampler_key(image.keypoints[index], index))
        remaining = maximum - len(selected)
        selected.extend(round_indices[:remaining])
        depth += 1

    compact = np.ascontiguousarray(descriptors[selected], dtype=np.dtype(DESCRIPTOR_DTYPE))
    compact.setflags(write=False)
    return compact


def _sampler_key(keypoint: cv2.KeyPoint, index: int) -> tuple[float | int, ...]:
    return (
        -round(float(keypoint.response), 9),
        round(float(keypoint.pt[1]), 6),
        round(float(keypoint.pt[0]), 6),
        round(float(keypoint.size), 6),
        round(float(keypoint.angle), 6),
        int(keypoint.octave),
        int(keypoint.class_id),
        index,
    )


def _empty_descriptors() -> np.ndarray:
    result = np.empty((0, DESCRIPTOR_DIMENSION), dtype=np.dtype(DESCRIPTOR_DTYPE))
    result.setflags(write=False)
    return result


def retrieve_candidates(
    query_descriptors: np.ndarray,
    descriptor_matrix: np.ndarray,
    originals: Sequence[OriginalIndexRecord],
    parameters: QueryParameters,
    *,
    profiler: RetrievalProfiler | None = None,
    item_ordinal: int | None = None,
) -> tuple[RetrievalCandidate, ...]:
    """Run exact blockwise pooled BF-L2 search and per-original voting."""

    query = np.ascontiguousarray(query_descriptors, dtype=np.float32)
    if query.ndim != 2 or query.shape[1] != DESCRIPTOR_DIMENSION:
        raise ValueError("INVALID_QUERY_DESCRIPTOR_SHAPE")
    matrix = np.asarray(descriptor_matrix)
    if matrix.ndim != 2 or matrix.shape[1] != DESCRIPTOR_DIMENSION:
        raise ValueError("INVALID_INDEX_DESCRIPTOR_SHAPE")
    if query.shape[0] == 0 or matrix.shape[0] == 0:
        return ()

    indexed = tuple(record for record in originals if record.descriptor_count > 0)
    range_ends = np.asarray(
        [record.descriptor_offset + record.descriptor_count for record in indexed],
        dtype=np.int64,
    )
    if not indexed or int(range_ends[-1]) != matrix.shape[0]:
        raise ValueError("INVALID_INDEX_DESCRIPTOR_RANGES")

    nearest: list[list[tuple[float, int]]] = [[] for _ in range(query.shape[0])]
    with profiling_stage(
        profiler, "query.exact_bf_search", item_ordinal=item_ordinal
    ) as search_timing:
        if profiler is not None:
            search_timing.add_work(
                selected_query_descriptor_rows=int(query.shape[0]),
                indexed_descriptor_rows=int(matrix.shape[0]),
                descriptor_distance_work_units=int(query.shape[0] * matrix.shape[0]),
                descriptor_blocks=math.ceil(
                    matrix.shape[0] / parameters.descriptor_block_rows
                ),
            )
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        for block_start in range(0, matrix.shape[0], parameters.descriptor_block_rows):
            block_end = min(matrix.shape[0], block_start + parameters.descriptor_block_rows)
            block = np.ascontiguousarray(matrix[block_start:block_end], dtype=np.float32)
            depth = min(parameters.neighbor_depth, block.shape[0])
            rows = matcher.knnMatch(query, block, k=depth)
            for query_index, matches in enumerate(rows):
                combined = nearest[query_index]
                combined.extend(
                    (float(match.distance), block_start + int(match.trainIdx))
                    for match in matches
                )
                combined.sort(key=lambda item: (item[0], item[1]))
                del combined[parameters.neighbor_depth :]

    with profiling_stage(
        profiler, "query.vote_aggregation_ranking", item_ordinal=item_ordinal
    ) as ranking_timing:
        support: dict[int, list[float]] = defaultdict(list)
        for matches in nearest:
            best_by_owner: dict[int, float] = {}
            for distance, row_index in matches:
                owner = int(np.searchsorted(range_ends, row_index, side="right"))
                previous = best_by_owner.get(owner)
                if previous is None or distance < previous:
                    best_by_owner[owner] = distance
            for owner, distance in best_by_owner.items():
                support[owner].append(distance)

        candidates = [
            RetrievalCandidate(
                original=indexed[owner].reference,
                supporting_query_descriptors=len(distances),
                median_best_l2_distance=float(median(distances)),
                mean_best_l2_distance=float(mean(distances)),
            )
            for owner, distances in support.items()
            if distances
        ]
        ranked = tuple(sorted(candidates, key=retrieval_candidate_sort_key))
        if profiler is not None:
            ranking_timing.add_work(
                selected_query_descriptor_rows=int(query.shape[0]),
                nearest_descriptor_matches=sum(len(matches) for matches in nearest),
                ranked_originals=len(ranked),
            )
        return ranked


def retrieval_candidate_sort_key(candidate: RetrievalCandidate) -> tuple[object, ...]:
    return (
        -candidate.supporting_query_descriptors,
        candidate.median_best_l2_distance,
        candidate.mean_best_l2_distance,
        semantic_reference_sort_key(candidate.original),
    )


def select_shortlist(
    ranked_candidates: Sequence[RetrievalCandidate], requested_k: int
) -> tuple[tuple[RetrievalCandidate, ...], int, int | None]:
    """Return top K plus every positive primary-score tie at the boundary."""

    if type(requested_k) is not int or requested_k <= 0:
        raise ValueError("INVALID_REQUESTED_K")
    ranked = tuple(ranked_candidates)
    if not ranked:
        return (), 0, None
    if len(ranked) < requested_k:
        return ranked, 0, None
    if len(ranked) == requested_k:
        return ranked, 0, ranked[-1].supporting_query_descriptors
    boundary = ranked[requested_k - 1].supporting_query_descriptors
    end = requested_k
    while end < len(ranked) and ranked[end].supporting_query_descriptors == boundary:
        end += 1
    selected = ranked[:end]
    return selected, end - requested_k, boundary


def recall_at_k(
    ranked_by_query: Iterable[tuple[SemanticReference, Sequence[RetrievalCandidate]]],
    k_values: Iterable[int],
) -> dict[int, float]:
    """Calculate known-source synthetic Recall@K with vote-tie extension."""

    cases = tuple(ranked_by_query)
    if not cases:
        return {int(k): 0.0 for k in k_values}
    result: dict[int, float] = {}
    for k in k_values:
        selected_count = 0
        for source, ranked in cases:
            shortlist, _, _ = select_shortlist(ranked, int(k))
            if any(candidate.original == source for candidate in shortlist):
                selected_count += 1
        result[int(k)] = selected_count / len(cases)
    return result


def corpus_identity_sha256(
    originals: Sequence[OriginalIndexRecord], scan_issues: Sequence[Mapping[str, str]]
) -> str:
    payload = {
        "originals": [record.identity_dict() for record in originals],
        "scan_issues": [dict(sorted(issue.items())) for issue in scan_issues],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_index_manifest(
    *,
    parameters: IndexParameters,
    binary_filename: str,
    binary_sha256: str,
    binary_byte_size: int,
    originals: Sequence[OriginalIndexRecord],
    scan_issues: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    stable_originals = tuple(originals)
    stable_issues = tuple(dict(issue) for issue in scan_issues)
    counts = Counter(record.status.value for record in stable_originals)
    total_rows = sum(record.descriptor_count for record in stable_originals)
    complete = bool(stable_originals) and not stable_issues and all(
        record.status is IndexStatus.INDEXED for record in stable_originals
    )
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "tool_version": __version__,
        "algorithm": {
            "name": ALGORITHM_NAME,
            "status": ALGORITHM_STATUS,
            "preprocessing": "uint8_grayscale_after_exif_transpose_then_sift",
            "descriptor_metric": "EXACT_BF_L2",
            "candidate_semantics": "RANKED_RETRIEVAL_SHORTLIST_NOT_PROVENANCE",
        },
        "runtime": current_runtime_contract(),
        "parameters": parameters.as_dict(),
        "corpus": {
            "root_role": RootRole.ORIGINAL.value,
            "identity_sha256": corpus_identity_sha256(stable_originals, stable_issues),
            "scan_issues": [dict(issue) for issue in stable_issues],
        },
        "binary": {
            "filename": binary_filename,
            "dtype": DESCRIPTOR_DTYPE,
            "descriptor_dimension": DESCRIPTOR_DIMENSION,
            "total_descriptor_rows": total_rows,
            "byte_size": binary_byte_size,
            "sha256": binary_sha256,
        },
        "summary": {
            "supplied_original_image_candidates": len(stable_originals),
            "INDEXED": counts[IndexStatus.INDEXED.value],
            "NO_DESCRIPTORS": counts[IndexStatus.NO_DESCRIPTORS.value],
            "UNAVAILABLE": counts[IndexStatus.UNAVAILABLE.value],
            "total_descriptor_rows": total_rows,
            "scan_issue_count": len(stable_issues),
            "index_corpus_complete": complete,
        },
        "originals": [record.as_dict() for record in stable_originals],
    }


def parse_json_bytes(raw_bytes: bytes) -> Mapping[str, object]:
    try:
        decoded = raw_bytes.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _: (_raise_validation("NON_FINITE_JSON_NUMBER")),
        )
    except IndexValidationError:
        raise
    except Exception as exc:
        raise IndexValidationError("INVALID_JSON") from exc
    if not isinstance(value, dict):
        raise IndexValidationError("INVALID_TOP_LEVEL")
    _reject_non_finite(value)
    return value


def validate_index_manifest(
    manifest: Mapping[str, object], *, expected_binary_filename: str
) -> IndexMetadata:
    if _string(manifest, "schema_version") != INDEX_SCHEMA_VERSION:
        raise IndexValidationError("UNSUPPORTED_INDEX_SCHEMA")
    algorithm = _mapping(manifest, "algorithm")
    if _string(algorithm, "name") != ALGORITHM_NAME:
        raise IndexValidationError("UNSUPPORTED_INDEX_ALGORITHM")
    if (
        _string(algorithm, "status") != ALGORITHM_STATUS
        or _string(algorithm, "preprocessing")
        != "uint8_grayscale_after_exif_transpose_then_sift"
        or _string(algorithm, "descriptor_metric") != "EXACT_BF_L2"
        or _string(algorithm, "candidate_semantics")
        != "RANKED_RETRIEVAL_SHORTLIST_NOT_PROVENANCE"
    ):
        raise IndexValidationError("UNSUPPORTED_INDEX_ALGORITHM")
    _validate_runtime_contract(manifest)
    parameters_data = _mapping(manifest, "parameters")
    try:
        parameters = IndexParameters(
            **{field.name: _integer(parameters_data, field.name, positive=field.name != "random_seed") for field in fields(IndexParameters)}
        )
    except (TypeError, ValueError) as exc:
        raise IndexValidationError("INVALID_INDEX_PARAMETERS") from exc

    binary = _mapping(manifest, "binary")
    filename = _string(binary, "filename")
    if filename != expected_binary_filename or not _safe_sibling_filename(filename):
        raise IndexValidationError("UNSAFE_BINARY_FILENAME")
    if _string(binary, "dtype") != DESCRIPTOR_DTYPE:
        raise IndexValidationError("INVALID_BINARY_DTYPE")
    if _integer(binary, "descriptor_dimension", positive=True) != DESCRIPTOR_DIMENSION:
        raise IndexValidationError("INVALID_DESCRIPTOR_DIMENSION")
    rows = _integer(binary, "total_descriptor_rows", positive=False)
    byte_size = _integer(binary, "byte_size", positive=False)
    if byte_size != rows * DESCRIPTOR_DIMENSION * DESCRIPTOR_ITEMSIZE:
        raise IndexValidationError("INVALID_BINARY_SIZE")
    binary_sha = _sha256(binary.get("sha256"))

    originals_data = _list(manifest, "originals")
    originals: list[OriginalIndexRecord] = []
    expected_offset = 0
    seen: set[SemanticReference] = set()
    for value in originals_data:
        data = _as_mapping(value)
        reference_data = _mapping(data, "original")
        if _string(reference_data, "root_role") != RootRole.ORIGINAL.value:
            raise IndexValidationError("INVALID_REFERENCE_ROLE")
        relative_path = _string(reference_data, "relative_path")
        _validate_relative_path(relative_path)
        reference = SemanticReference(RootRole.ORIGINAL, relative_path)
        if reference in seen:
            raise IndexValidationError("DUPLICATE_ORIGINAL_REFERENCE")
        seen.add(reference)
        width = _optional_positive_integer(data, "display_width")
        height = _optional_positive_integer(data, "display_height")
        if (width is None) != (height is None):
            raise IndexValidationError("PARTIAL_DISPLAY_DIMENSIONS")
        size_bytes = _optional_nonnegative_integer(data, "size_bytes")
        encoded_hash = data.get("encoded_sha256")
        encoded_sha = None if encoded_hash is None else _sha256(encoded_hash)
        try:
            status = IndexStatus(_string(data, "index_status"))
        except ValueError as exc:
            raise IndexValidationError("INVALID_INDEX_STATUS") from exc
        selected = _integer(data, "selected_descriptor_count", positive=False)
        offset = _integer(data, "descriptor_offset", positive=False)
        count = _integer(data, "descriptor_count", positive=False)
        effective_descriptor_cap = min(
            parameters.original_max_descriptors, parameters.sift_nfeatures
        )
        if selected > effective_descriptor_cap or count > effective_descriptor_cap:
            raise IndexValidationError("DESCRIPTOR_CAP_EXCEEDED")
        if offset != expected_offset or selected != count:
            raise IndexValidationError("INVALID_DESCRIPTOR_RANGE")
        if status is IndexStatus.INDEXED:
            if (
                count <= 0
                or width is None
                or size_bytes is None
                or encoded_sha is None
            ):
                raise IndexValidationError("CONTRADICTORY_INDEX_STATUS")
        elif status is IndexStatus.NO_DESCRIPTORS:
            if (
                count != 0
                or width is None
                or size_bytes is None
                or encoded_sha is None
            ):
                raise IndexValidationError("CONTRADICTORY_INDEX_STATUS")
        elif count != 0:
            raise IndexValidationError("CONTRADICTORY_INDEX_STATUS")
        record = OriginalIndexRecord(
            reference,
            width,
            height,
            size_bytes,
            encoded_sha,
            status,
            selected,
            offset,
            count,
        )
        originals.append(record)
        expected_offset += count
    if expected_offset != rows:
        raise IndexValidationError("INVALID_DESCRIPTOR_RANGE")
    if tuple(originals) != tuple(sorted(originals, key=lambda record: semantic_reference_sort_key(record.reference))):
        raise IndexValidationError("NONDETERMINISTIC_ORIGINAL_ORDER")

    corpus = _mapping(manifest, "corpus")
    if _string(corpus, "root_role") != RootRole.ORIGINAL.value:
        raise IndexValidationError("INVALID_CORPUS_ROLE")
    issues_data = _list(corpus, "scan_issues")
    scan_issues: list[dict[str, str]] = []
    for value in issues_data:
        issue_data = _as_mapping(value)
        relative_path = issue_data.get("relative_path")
        if not isinstance(relative_path, str):
            raise IndexValidationError("INVALID_SCAN_ISSUE")
        issue = {
            "root_role": _string(issue_data, "root_role"),
            "relative_path": relative_path,
            "category": _string(issue_data, "category"),
        }
        if issue["root_role"] != RootRole.ORIGINAL.value:
            raise IndexValidationError("INVALID_SCAN_ISSUE")
        if issue["relative_path"]:
            _validate_relative_path(issue["relative_path"])
        scan_issues.append(issue)
    identity = _sha256(corpus.get("identity_sha256"))
    if identity != corpus_identity_sha256(originals, scan_issues):
        raise IndexValidationError("CORPUS_IDENTITY_MISMATCH")

    summary = _mapping(manifest, "summary")
    counts = Counter(record.status.value for record in originals)
    if _integer(summary, "supplied_original_image_candidates", positive=False) != len(originals):
        raise IndexValidationError("INVALID_INDEX_SUMMARY")
    for status in IndexStatus:
        if _integer(summary, status.value, positive=False) != counts[status.value]:
            raise IndexValidationError("INVALID_INDEX_SUMMARY")
    if _integer(summary, "total_descriptor_rows", positive=False) != rows:
        raise IndexValidationError("INVALID_INDEX_SUMMARY")
    if _integer(summary, "scan_issue_count", positive=False) != len(scan_issues):
        raise IndexValidationError("INVALID_INDEX_SUMMARY")
    complete = _boolean(summary, "index_corpus_complete")
    expected_complete = bool(originals) and not scan_issues and all(
        record.status is IndexStatus.INDEXED for record in originals
    )
    if complete is not expected_complete:
        raise IndexValidationError("INVALID_INDEX_COMPLETENESS")

    return IndexMetadata(
        parameters,
        identity,
        filename,
        binary_sha,
        byte_size,
        rows,
        complete,
        tuple(originals),
        tuple(scan_issues),
    )


def build_retrieval_manifest(
    *,
    index_manifest_sha256: str,
    metadata: IndexMetadata,
    query_parameters: QueryParameters,
    queries: Sequence[RetrievalQueryResult],
    query_scan_issue_count: int = 0,
) -> dict[str, object]:
    counts = Counter(query.status.value for query in queries)
    return {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "tool_version": __version__,
        "algorithm": {
            "name": ALGORITHM_NAME,
            "status": ALGORITHM_STATUS,
            "output_semantics": "RANKED_RETRIEVAL_SHORTLIST_NOT_PROVENANCE",
        },
        "index_linkage": {
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "index_manifest_sha256": index_manifest_sha256,
            "corpus_identity_sha256": metadata.corpus_identity_sha256,
            "binary_sha256": metadata.binary_sha256,
        },
        "parameters": {
            **query_parameters.as_dict(),
            "grid_rows": metadata.parameters.grid_rows,
            "grid_columns": metadata.parameters.grid_columns,
        },
        "summary": {
            "queries": len(queries),
            "RETRIEVED": counts[QueryStatus.RETRIEVED.value],
            "NO_QUERY_DESCRIPTORS": counts[QueryStatus.NO_QUERY_DESCRIPTORS.value],
            "QUERY_UNAVAILABLE": counts[QueryStatus.QUERY_UNAVAILABLE.value],
            "INDEX_INCOMPLETE": counts[QueryStatus.INDEX_INCOMPLETE.value],
            "retrieval_queries_complete": sum(query.retrieval_query_complete for query in queries),
            "index_corpus_complete": metadata.index_corpus_complete,
            "query_scan_issue_count": query_scan_issue_count,
            "query_set_complete": query_scan_issue_count == 0,
        },
        "queries": [_serialize_query(query) for query in queries],
    }


def _serialize_query(query: RetrievalQueryResult) -> dict[str, object]:
    return {
        "crop": {
            "root_role": query.crop.root_role.value,
            "relative_path": query.crop.relative_path,
        },
        "display_width": query.display_width,
        "display_height": query.display_height,
        "query_status": query.status.value,
        "retrieval_query_complete": query.retrieval_query_complete,
        "extracted_descriptor_count": query.extracted_descriptor_count,
        "selected_descriptor_count": query.selected_descriptor_count,
        "requested_k": query.requested_k,
        "returned_candidate_count": query.returned_candidate_count,
        "tie_extension_count": query.tie_extension_count,
        "boundary_support_votes": query.boundary_support_votes,
        "ranked_candidates": [
            {
                "rank": rank,
                "original": {
                    "root_role": candidate.original.root_role.value,
                    "relative_path": candidate.original.relative_path,
                },
                "supporting_query_descriptors": candidate.supporting_query_descriptors,
                "median_best_l2_distance": candidate.median_best_l2_distance,
                "mean_best_l2_distance": candidate.mean_best_l2_distance,
            }
            for rank, candidate in enumerate(query.ranked_candidates, start=1)
        ],
    }


def make_query_result(
    *,
    crop: FeatureImage,
    compact_descriptors: np.ndarray,
    metadata: IndexMetadata,
    descriptor_matrix: np.ndarray,
    parameters: QueryParameters,
    profiler: RetrievalProfiler | None = None,
    item_ordinal: int | None = None,
) -> RetrievalQueryResult:
    if crop.descriptors is None or crop.diagnostic_reason is not None:
        status = (
            QueryStatus.NO_QUERY_DESCRIPTORS
            if crop.diagnostic_reason == "NO_DESCRIPTORS"
            else QueryStatus.QUERY_UNAVAILABLE
        )
        return RetrievalQueryResult(
            crop.reference,
            crop.width if crop.width > 0 else None,
            crop.height if crop.height > 0 else None,
            status,
            False,
            len(crop.keypoints),
            0,
            parameters.requested_k,
            0,
            0,
            None,
            (),
        )
    ranked = retrieve_candidates(
        compact_descriptors,
        descriptor_matrix,
        metadata.originals,
        parameters,
        profiler=profiler,
        item_ordinal=item_ordinal,
    )
    with profiling_stage(
        profiler, "query.shortlist_construction", item_ordinal=item_ordinal
    ) as shortlist_timing:
        shortlist, extension, boundary = select_shortlist(ranked, parameters.requested_k)
        if profiler is not None:
            shortlist_timing.add_work(
                ranked_originals=len(ranked),
                returned_candidates=len(shortlist),
                tie_extension_count=extension,
            )
    status = QueryStatus.RETRIEVED if metadata.index_corpus_complete else QueryStatus.INDEX_INCOMPLETE
    return RetrievalQueryResult(
        crop.reference,
        crop.width,
        crop.height,
        status,
        metadata.index_corpus_complete,
        len(crop.keypoints),
        int(compact_descriptors.shape[0]),
        parameters.requested_k,
        len(shortlist),
        extension,
        boundary,
        shortlist,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise IndexValidationError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _validate_runtime_contract(manifest: Mapping[str, object]) -> None:
    try:
        runtime = _mapping(manifest, "runtime")
        if set(runtime) != set(RUNTIME_VERSION_FIELDS):
            raise IndexValidationError("INVALID_INDEX_RUNTIME")
        serialized = {
            field: _string(runtime, field) for field in RUNTIME_VERSION_FIELDS
        }
    except IndexValidationError as exc:
        raise IndexValidationError("INVALID_INDEX_RUNTIME") from exc
    if serialized != current_runtime_contract():
        raise IndexValidationError("INDEX_RUNTIME_MISMATCH")


def _raise_validation(code: str) -> None:
    raise IndexValidationError(code)


def _reject_non_finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise IndexValidationError("NON_FINITE_JSON_NUMBER")
    if isinstance(value, dict):
        for nested in value.values():
            _reject_non_finite(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_non_finite(nested)


def _mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    if key not in data:
        raise IndexValidationError("MISSING_REQUIRED_FIELD")
    return _as_mapping(data[key])


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise IndexValidationError("INVALID_FIELD_TYPE")
    return value


def _list(data: Mapping[str, object], key: str) -> list[object]:
    if key not in data or not isinstance(data[key], list):
        raise IndexValidationError("INVALID_FIELD_TYPE")
    return data[key]


def _string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise IndexValidationError("INVALID_FIELD_TYPE")
    return value


def _integer(data: Mapping[str, object], key: str, *, positive: bool) -> int:
    value = data.get(key)
    if type(value) is not int or value < (1 if positive else 0):
        raise IndexValidationError("INVALID_INTEGER")
    return value


def _optional_positive_integer(data: Mapping[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise IndexValidationError("INVALID_INTEGER")
    return value


def _optional_nonnegative_integer(data: Mapping[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise IndexValidationError("INVALID_INTEGER")
    return value


def _boolean(data: Mapping[str, object], key: str) -> bool:
    value = data.get(key)
    if type(value) is not bool:
        raise IndexValidationError("INVALID_FIELD_TYPE")
    return value


def _sha256(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise IndexValidationError("INVALID_SHA256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise IndexValidationError("INVALID_SHA256") from exc
    return value.lower()


def _validate_relative_path(value: str) -> None:
    if not value or "\\" in value or "\x00" in value:
        raise IndexValidationError("UNSAFE_RELATIVE_PATH")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise IndexValidationError("UNSAFE_RELATIVE_PATH")
    posix = PurePosixPath(value)
    if posix.is_absolute() or posix.as_posix() != value or PureWindowsPath(value).drive:
        raise IndexValidationError("UNSAFE_RELATIVE_PATH")


def _safe_sibling_filename(value: str) -> bool:
    return (
        bool(value)
        and "/" not in value
        and "\\" not in value
        and value not in {".", ".."}
        and not PureWindowsPath(value).drive
    )

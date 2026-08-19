"""Descriptor-only scale characterization for the exact-BF retrieval oracle."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
from statistics import median
import sys
import tempfile
from time import perf_counter_ns
from typing import Callable, Sequence, TextIO

import cv2
import numpy as np

from . import __version__
from .audit import RootRole, SemanticReference
from .candidate_retrieval import (
    DEFAULT_DESCRIPTOR_BLOCK_ROWS,
    DESCRIPTOR_DIMENSION,
    DESCRIPTOR_DTYPE,
    IndexStatus,
    OriginalIndexRecord,
    QueryParameters,
    RetrievalCandidate,
    retrieve_candidates,
    select_shortlist,
)
from .candidate_retrieval_profiling import RetrievalProfiler, profiling_stage
from .cli import ConfigurationError, OutputFailure, write_manifest_atomic
from .content_matching import MatchingParameters, configure_opencv


SCALE_BENCHMARK_SCHEMA_VERSION = "1.0"
SCALE_BENCHMARK_SEMANTICS = (
    "SYNTHETIC_DESCRIPTOR_ONLY_EXACT_BF_SCALE_CHARACTERIZATION"
)
DESCRIPTORS_PER_ORIGINAL = 128
DEFAULT_QUERY_DESCRIPTOR_ROWS = 64
DEFAULT_NEIGHBOR_DEPTH = 32
DEFAULT_REQUESTED_K = 50
DEFAULT_WARM_QUERIES = 10
DEFAULT_GENERATION_CHUNK_ROWS = 2_048
SUCCESSFUL_STOP_REASONS = frozenset(
    {
        "COMPLETED_REQUESTED_SCALES",
        "OPERATOR_MAXIMUM_REACHED",
        "WALL_TIME_BUDGET_EXHAUSTED",
    }
)
APPROVED_SCALE_LADDER = (
    10_880,
    65_408,
    65_664,
    262_144,
    524_288,
    1_048_576,
    1_279_872,
)
SENSITIVITY_SCALE_ROWS = 262_144
SENSITIVITY_REPETITIONS = 3
_MIB = 1_048_576


@dataclass(frozen=True, slots=True)
class ScaleBenchmarkConfiguration:
    scale_ladder: tuple[int, ...] = APPROVED_SCALE_LADDER
    maximum_descriptor_rows: int = APPROVED_SCALE_LADDER[-1]
    warm_queries: int = DEFAULT_WARM_QUERIES
    seed: int = 17_029
    wall_time_seconds: float | None = None
    include_sensitivity: bool = False
    generation_chunk_rows: int = DEFAULT_GENERATION_CHUNK_ROWS

    def __post_init__(self) -> None:
        if not self.scale_ladder:
            raise ValueError("SCALE_LADDER_REQUIRED")
        if tuple(sorted(set(self.scale_ladder))) != self.scale_ladder:
            raise ValueError("SCALE_LADDER_MUST_BE_STRICTLY_INCREASING")
        for rows in self.scale_ladder:
            if type(rows) is not int or rows <= 0 or rows % DESCRIPTORS_PER_ORIGINAL:
                raise ValueError("INVALID_SCALE_DESCRIPTOR_ROWS")
        if (
            type(self.maximum_descriptor_rows) is not int
            or self.maximum_descriptor_rows <= 0
        ):
            raise ValueError("INVALID_MAXIMUM_DESCRIPTOR_ROWS")
        if type(self.warm_queries) is not int or self.warm_queries <= 0:
            raise ValueError("INVALID_WARM_QUERY_COUNT")
        if type(self.seed) is not int or not 0 <= self.seed <= 2_147_483_647:
            raise ValueError("INVALID_GENERATION_SEED")
        if self.wall_time_seconds is not None and self.wall_time_seconds <= 0:
            raise ValueError("INVALID_WALL_TIME_BUDGET")
        if type(self.include_sensitivity) is not bool:
            raise ValueError("INVALID_SENSITIVITY_SETTING")
        if (
            type(self.generation_chunk_rows) is not int
            or self.generation_chunk_rows <= 0
        ):
            raise ValueError("INVALID_GENERATION_CHUNK_ROWS")
        selected = self.selected_scales
        if selected and selected[0] // DESCRIPTORS_PER_ORIGINAL <= self.warm_queries:
            raise ValueError("INSUFFICIENT_ORIGINALS_FOR_DISTINCT_QUERIES")

    @property
    def selected_scales(self) -> tuple[int, ...]:
        return tuple(
            rows for rows in self.scale_ladder if rows <= self.maximum_descriptor_rows
        )


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    status: str
    result: dict[str, object] | None = None
    stop_reason: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autocrop_analysis.candidate_retrieval_scale_benchmark",
        description=(
            "Characterize the unchanged exact-BF oracle using deterministic "
            "descriptor-only temporary memmaps."
        ),
    )
    parser.add_argument(
        "--max-descriptor-rows",
        type=int,
        default=APPROVED_SCALE_LADDER[-1],
        help="Stop after the greatest approved scale at or below this row count.",
    )
    parser.add_argument("--warm-queries", type=int, default=DEFAULT_WARM_QUERIES)
    parser.add_argument("--seed", type=int, default=17_029)
    parser.add_argument("--wall-time-seconds", type=float)
    parser.add_argument("--include-sensitivity", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _descriptor_bytes(rows: int) -> int:
    return rows * DESCRIPTOR_DIMENSION * np.dtype(DESCRIPTOR_DTYPE).itemsize


def build_original_records(original_count: int) -> tuple[OriginalIndexRecord, ...]:
    if type(original_count) is not int or original_count <= 0:
        raise ValueError("INVALID_SYNTHETIC_ORIGINAL_COUNT")
    return tuple(
        OriginalIndexRecord(
            reference=SemanticReference(
                RootRole.ORIGINAL, f"synthetic/source-{owner:05d}.descriptor"
            ),
            display_width=1,
            display_height=1,
            size_bytes=DESCRIPTORS_PER_ORIGINAL * DESCRIPTOR_DIMENSION * 4,
            encoded_sha256=f"{owner:064x}",
            status=IndexStatus.INDEXED,
            selected_descriptor_count=DESCRIPTORS_PER_ORIGINAL,
            descriptor_offset=owner * DESCRIPTORS_PER_ORIGINAL,
            descriptor_count=DESCRIPTORS_PER_ORIGINAL,
        )
        for owner in range(original_count)
    )


def generate_descriptor_file(
    path: Path,
    descriptor_rows: int,
    seed: int,
    *,
    chunk_rows: int = DEFAULT_GENERATION_CHUNK_ROWS,
) -> dict[str, int]:
    """Write deterministic SIFT-like float32 rows without a full-size ndarray."""

    if descriptor_rows <= 0 or descriptor_rows % DESCRIPTORS_PER_ORIGINAL:
        raise ValueError("INVALID_SCALE_DESCRIPTOR_ROWS")
    if chunk_rows <= 0:
        raise ValueError("INVALID_GENERATION_CHUNK_ROWS")
    original_count = descriptor_rows // DESCRIPTORS_PER_ORIGINAL
    family_count = math.ceil(original_count / 8)
    family_rng = np.random.default_rng(seed ^ 0x5DEECE66D)
    family_centroids = family_rng.random(
        (family_count, DESCRIPTOR_DIMENSION), dtype=np.float32
    )
    row_rng = np.random.default_rng(seed)
    matrix: np.memmap | None = None
    started = perf_counter_ns()
    try:
        matrix = np.memmap(
            path,
            mode="w+",
            dtype=np.dtype(DESCRIPTOR_DTYPE),
            shape=(descriptor_rows, DESCRIPTOR_DIMENSION),
        )
        for start in range(0, descriptor_rows, chunk_rows):
            end = min(descriptor_rows, start + chunk_rows)
            count = end - start
            values = row_rng.random((count, DESCRIPTOR_DIMENSION), dtype=np.float32)
            owners = np.arange(start, end, dtype=np.int64) // DESCRIPTORS_PER_ORIGINAL
            correlated = owners % 8 < 2
            if bool(np.any(correlated)):
                nearby = family_centroids[(owners[correlated] // 8).astype(np.intp)]
                values[correlated] = (
                    np.float32(0.35) * values[correlated]
                    + np.float32(0.65) * nearby
                )
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            values *= np.float32(512.0) / np.maximum(norms, np.float32(1e-12))
            np.clip(values, np.float32(0.0), np.float32(255.0), out=values)
            matrix[start:end] = np.ascontiguousarray(values, dtype=np.float32)
        matrix.flush()
    finally:
        if matrix is not None and getattr(matrix, "_mmap", None) is not None:
            matrix._mmap.close()
    return {
        "descriptor_rows": descriptor_rows,
        "synthetic_originals": original_count,
        "raw_descriptor_bytes": _descriptor_bytes(descriptor_rows),
        "descriptor_file_bytes": path.stat().st_size,
        "generation_seed": seed,
        "generation_elapsed_ns": max(0, perf_counter_ns() - started),
    }


def make_query_descriptors(
    descriptor_matrix: np.ndarray,
    *,
    source_owner: int,
    query_rows: int,
    seed: int,
    query_ordinal: int,
) -> np.ndarray:
    if not 0 <= source_owner < descriptor_matrix.shape[0] // DESCRIPTORS_PER_ORIGINAL:
        raise ValueError("INVALID_QUERY_SOURCE_OWNER")
    if not 0 < query_rows <= DESCRIPTORS_PER_ORIGINAL:
        raise ValueError("INVALID_QUERY_DESCRIPTOR_ROWS")
    maximum_offset = DESCRIPTORS_PER_ORIGINAL - query_rows
    local_offset = (query_ordinal * 7) % (maximum_offset + 1)
    start = source_owner * DESCRIPTORS_PER_ORIGINAL + local_offset
    query = np.array(
        descriptor_matrix[start : start + query_rows],
        dtype=np.float32,
        order="C",
        copy=True,
    )
    noise_rng = np.random.default_rng(seed + (query_ordinal + 1) * 1_000_003)
    noise = noise_rng.standard_normal(query.shape, dtype=np.float32)
    query += noise * np.float32(0.25)
    np.clip(query, np.float32(0.0), np.float32(255.0), out=query)
    norms = np.linalg.norm(query, axis=1, keepdims=True)
    query *= np.float32(512.0) / np.maximum(norms, np.float32(1e-12))
    return np.ascontiguousarray(query, dtype=np.float32)


def _stage_measurement(
    profiler: RetrievalProfiler, stage: str
) -> tuple[int, dict[str, int]]:
    matching = [item for item in profiler.measurements if item.stage == stage]
    if len(matching) != 1:
        raise RuntimeError("UNEXPECTED_PROFILER_STAGE_COUNT")
    return matching[0].elapsed_ns, dict(matching[0].work_counts)


def measure_retrieval(
    query: np.ndarray,
    descriptor_matrix: np.ndarray,
    originals: Sequence[OriginalIndexRecord],
    parameters: QueryParameters,
    *,
    expected_owner: int,
) -> tuple[dict[str, object], tuple[RetrievalCandidate, ...]]:
    profiler = RetrievalProfiler()
    started = perf_counter_ns()
    ranked = retrieve_candidates(
        query, descriptor_matrix, originals, parameters, profiler=profiler
    )
    with profiling_stage(profiler, "query.shortlist_construction") as timing:
        shortlist, extension, boundary = select_shortlist(ranked, parameters.requested_k)
        timing.add_work(
            ranked_originals=len(ranked),
            returned_candidates=len(shortlist),
            tie_extension_count=extension,
        )
    total_ns = max(0, perf_counter_ns() - started)
    exact_ns, exact_work = _stage_measurement(profiler, "query.exact_bf_search")
    ranking_ns, ranking_work = _stage_measurement(
        profiler, "query.vote_aggregation_ranking"
    )
    shortlist_ns, shortlist_work = _stage_measurement(
        profiler, "query.shortlist_construction"
    )
    expected_reference = originals[expected_owner].reference
    known_source_rank = next(
        (
            rank
            for rank, candidate in enumerate(ranked, start=1)
            if candidate.original == expected_reference
        ),
        None,
    )
    return (
        {
            "total_descriptor_retrieval_ns": total_ns,
            "exact_bf_search_ns": exact_ns,
            "vote_aggregation_ranking_ns": ranking_ns,
            "shortlist_construction_ns": shortlist_ns,
            "work_counts": {
                "exact_bf_search": exact_work,
                "vote_aggregation_ranking": ranking_work,
                "shortlist_construction": shortlist_work,
            },
            "known_source_rank": known_source_rank,
            "known_source_ranked_first": known_source_rank == 1,
            "ranked_originals": len(ranked),
            "returned_candidates": len(shortlist),
            "tie_extension_count": extension,
            "boundary_support_votes": boundary,
        },
        ranked,
    )


def _duration_summary(measurements: Sequence[dict[str, object]]) -> dict[str, object]:
    fields = (
        "total_descriptor_retrieval_ns",
        "exact_bf_search_ns",
        "vote_aggregation_ranking_ns",
        "shortlist_construction_ns",
    )
    result: dict[str, object] = {}
    for field in fields:
        values = [int(item[field]) for item in measurements]
        result[field.removesuffix("_ns")] = {
            "raw_ns": values,
            "median_ns": float(median(values)),
            "p95_ns": float(np.percentile(values, 95)),
            "min_ns": min(values),
            "max_ns": max(values),
        }
    return result


def _descriptor_diagnostics(matrix: np.memmap) -> dict[str, object]:
    sample_end = min(matrix.shape[0], DEFAULT_DESCRIPTOR_BLOCK_ROWS)
    sample = np.ascontiguousarray(matrix[:sample_end], dtype=np.float32)
    return {
        "dtype": matrix.dtype.str,
        "shape": list(matrix.shape),
        "strides": list(matrix.strides),
        "c_contiguous": bool(matrix.flags.c_contiguous),
        "writeable": bool(matrix.flags.writeable),
        "is_memmap": isinstance(matrix, np.memmap),
        "opened_mode": "read_only",
        "representative_coercion_rows": sample_end,
        "representative_coercion_shares_memory": bool(
            np.shares_memory(matrix, sample)
        ),
    }


def _run_sensitivity(
    matrix: np.memmap,
    originals: Sequence[OriginalIndexRecord],
    seed: int,
) -> dict[str, object]:
    def samples(query_rows: int, neighbor_depth: int) -> dict[str, object]:
        measurements = []
        for repetition in range(SENSITIVITY_REPETITIONS):
            owner = repetition + 1
            query = make_query_descriptors(
                matrix,
                source_owner=owner,
                query_rows=query_rows,
                seed=seed,
                query_ordinal=100 + repetition,
            )
            measured, _ = measure_retrieval(
                query,
                matrix,
                originals,
                QueryParameters(
                    query_max_descriptors=query_rows,
                    neighbor_depth=neighbor_depth,
                    requested_k=DEFAULT_REQUESTED_K,
                    descriptor_block_rows=DEFAULT_DESCRIPTOR_BLOCK_ROWS,
                ),
                expected_owner=owner,
            )
            measurements.append(measured)
        return {
            "query_descriptor_rows": query_rows,
            "neighbor_depth": neighbor_depth,
            "repetitions": SENSITIVITY_REPETITIONS,
            "timing": _duration_summary(measurements),
            "all_known_sources_ranked_first": all(
                bool(item["known_source_ranked_first"]) for item in measurements
            ),
        }

    return {
        "query_row_sensitivity": [samples(rows, 32) for rows in (16, 32, 64)],
        "neighbor_depth_sensitivity": [samples(64, depth) for depth in (8, 32, 64)],
    }


def execute_scale_worker(
    descriptor_path: Path,
    *,
    descriptor_rows: int,
    seed: int,
    warm_queries: int,
    include_sensitivity: bool,
) -> dict[str, object]:
    """Measure one scale. This function is executed in a fresh child process."""

    memory_profiler = RetrievalProfiler()
    memory_profiler.snapshot_memory("worker_start")
    configure_opencv(MatchingParameters(random_seed=seed))
    effective_threads = (
        int(cv2.getNumThreads()) if hasattr(cv2, "getNumThreads") else None
    )
    matrix: np.memmap | None = None
    try:
        try:
            matrix = np.memmap(
                descriptor_path,
                mode="r",
                dtype=np.dtype(DESCRIPTOR_DTYPE),
                shape=(descriptor_rows, DESCRIPTOR_DIMENSION),
            )
        except Exception as exc:
            raise RuntimeError("MEMMAP_OPEN_FAILED") from exc
        memory_profiler.snapshot_memory("after_read_only_memmap_open")
        originals = build_original_records(
            descriptor_rows // DESCRIPTORS_PER_ORIGINAL
        )
        diagnostics = _descriptor_diagnostics(matrix)
        parameters = QueryParameters(
            query_max_descriptors=DEFAULT_QUERY_DESCRIPTOR_ROWS,
            neighbor_depth=DEFAULT_NEIGHBOR_DEPTH,
            requested_k=DEFAULT_REQUESTED_K,
            descriptor_block_rows=DEFAULT_DESCRIPTOR_BLOCK_ROWS,
        )

        first_owner = 0
        first_query = make_query_descriptors(
            matrix,
            source_owner=first_owner,
            query_rows=DEFAULT_QUERY_DESCRIPTOR_ROWS,
            seed=seed,
            query_ordinal=0,
        )
        memory_profiler.snapshot_memory("before_first_search")
        first_touch, _ = measure_retrieval(
            first_query,
            matrix,
            originals,
            parameters,
            expected_owner=first_owner,
        )
        memory_profiler.snapshot_memory("after_first_search")
        if not first_touch["known_source_ranked_first"]:
            raise RuntimeError("KNOWN_SOURCE_GUARD_FAILED")

        memory_profiler.snapshot_memory("before_warm_series")
        warm_measurements = []
        for ordinal in range(1, warm_queries + 1):
            source_owner = ordinal
            query = make_query_descriptors(
                matrix,
                source_owner=source_owner,
                query_rows=DEFAULT_QUERY_DESCRIPTOR_ROWS,
                seed=seed,
                query_ordinal=ordinal,
            )
            measured, _ = measure_retrieval(
                query,
                matrix,
                originals,
                parameters,
                expected_owner=source_owner,
            )
            if not measured["known_source_ranked_first"]:
                raise RuntimeError("KNOWN_SOURCE_GUARD_FAILED")
            warm_measurements.append(measured)
        memory_profiler.snapshot_memory("after_warm_series")

        sensitivity = None
        if include_sensitivity and descriptor_rows == SENSITIVITY_SCALE_ROWS:
            sensitivity = _run_sensitivity(matrix, originals, seed)
        matrix._mmap.close()
        matrix = None
        memory_profiler.snapshot_memory("final_state")
        return {
            "descriptor_rows": descriptor_rows,
            "synthetic_originals": len(originals),
            "bf_blocks": math.ceil(
                descriptor_rows / DEFAULT_DESCRIPTOR_BLOCK_ROWS
            ),
            "raw_descriptor_bytes": _descriptor_bytes(descriptor_rows),
            "opencv_configuration": {
                "requested_threads": 1,
                "effective_thread_count": effective_threads,
                "rng_seed": seed,
            },
            "descriptor_diagnostics": diagnostics,
            "correctness_guard": {
                "passed": True,
                "first_touch_known_source_ranked_first": True,
                "all_warm_known_sources_ranked_first": True,
            },
            "first_touch": first_touch,
            "warm_queries": warm_measurements,
            "warm_summary": _duration_summary(warm_measurements),
            "memory": memory_profiler.as_report()["memory"],
            "sensitivity": sensitivity,
        }
    finally:
        if matrix is not None and getattr(matrix, "_mmap", None) is not None:
            matrix._mmap.close()


def _worker_entry(connection, request: dict[str, object]) -> None:
    try:
        result = execute_scale_worker(
            Path(str(request["descriptor_path"])),
            descriptor_rows=int(request["descriptor_rows"]),
            seed=int(request["seed"]),
            warm_queries=int(request["warm_queries"]),
            include_sensitivity=bool(request["include_sensitivity"]),
        )
        connection.send({"status": "completed", "result": result})
    except BaseException as exc:
        known_runtime_reasons = {
            "KNOWN_SOURCE_GUARD_FAILED",
            "MEMMAP_OPEN_FAILED",
            "UNEXPECTED_PROFILER_STAGE_COUNT",
        }
        runtime_reason = str(exc) if isinstance(exc, RuntimeError) else ""
        connection.send(
            {
                "status": "failed",
                "stop_reason": (
                    runtime_reason
                    if runtime_reason in known_runtime_reasons
                    else f"WORKER_{type(exc).__name__.upper()}"
                ),
            }
        )
    finally:
        connection.close()


def run_scale_in_fresh_process(
    descriptor_path: Path,
    *,
    descriptor_rows: int,
    seed: int,
    warm_queries: int,
    include_sensitivity: bool,
    timeout_seconds: float | None,
) -> WorkerOutcome:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_entry,
        args=(
            child,
            {
                "descriptor_path": str(descriptor_path),
                "descriptor_rows": descriptor_rows,
                "seed": seed,
                "warm_queries": warm_queries,
                "include_sensitivity": include_sensitivity,
            },
        ),
    )
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            process.terminate()
            process.join()
            return WorkerOutcome("failed", stop_reason="WALL_TIME_BUDGET_EXHAUSTED")
        payload = parent.recv()
    except (EOFError, OSError):
        process.join()
        return WorkerOutcome("failed", stop_reason="WORKER_EXITED_UNEXPECTEDLY")
    finally:
        parent.close()
    process.join()
    if process.exitcode != 0:
        return WorkerOutcome("failed", stop_reason="WORKER_EXITED_UNEXPECTEDLY")
    if payload.get("status") != "completed":
        return WorkerOutcome(
            "failed", stop_reason=str(payload.get("stop_reason") or "WORKER_FAILED")
        )
    return WorkerOutcome("completed", result=payload["result"])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run_correctness_guards(seed: int) -> dict[str, object]:
    """Exercise oracle equivalence on a tiny pool before scale progression."""

    rows = 4 * DESCRIPTORS_PER_ORIGINAL
    temporary_paths: list[Path] = []
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="autocrop-exact-bf-guard-") as temporary:
        base = Path(temporary)
        first_path = base / "first.f32"
        second_path = base / "second.f32"
        temporary_paths.extend((first_path, second_path))
        generate_descriptor_file(first_path, rows, seed, chunk_rows=128)
        generate_descriptor_file(second_path, rows, seed, chunk_rows=128)
        checks["deterministic_generation"] = _sha256_file(first_path) == _sha256_file(
            second_path
        )
        matrix = np.memmap(
            first_path,
            mode="r",
            dtype=np.dtype(DESCRIPTOR_DTYPE),
            shape=(rows, DESCRIPTOR_DIMENSION),
        )
        try:
            originals = build_original_records(4)
            query = make_query_descriptors(
                matrix,
                source_owner=1,
                query_rows=64,
                seed=seed,
                query_ordinal=0,
            )
            parameters = QueryParameters(
                query_max_descriptors=64,
                neighbor_depth=32,
                requested_k=50,
                descriptor_block_rows=DEFAULT_DESCRIPTOR_BLOCK_ROWS,
            )
            first = retrieve_candidates(query, matrix, originals, parameters)
            repeated = retrieve_candidates(query, matrix, originals, parameters)
            checks["known_source_ranked_first"] = bool(
                first and first[0].original == originals[1].reference
            )
            checks["repeated_retrieval_identical"] = repeated == first
            profiler = RetrievalProfiler()
            profiled = retrieve_candidates(
                query, matrix, originals, parameters, profiler=profiler
            )
            checks["profiled_unprofiled_identical"] = profiled == first
            ndarray_result = retrieve_candidates(
                query, np.array(matrix, copy=True), originals, parameters
            )
            checks["memmap_ndarray_identical"] = ndarray_result == first
        finally:
            matrix._mmap.close()

        tied = tuple(
            RetrievalCandidate(
                SemanticReference(RootRole.ORIGINAL, f"synthetic/tie-{index:02d}"),
                2 if index < 49 else 1,
                float(index),
                float(index),
            )
            for index in range(52)
        )
        shortlist, extension, boundary = select_shortlist(tied, 50)
        checks["shortlist_k50_tie_extension"] = (
            len(shortlist) == 52 and extension == 2 and boundary == 1
        )
    cleanup_complete = all(not path.exists() for path in temporary_paths)
    checks["temporary_files_cleaned"] = cleanup_complete
    return {"passed": all(checks.values()), "checks": checks}


def read_available_physical_memory() -> int | None:
    if sys.platform != "win32":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = (
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        )

    try:
        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys)
    except Exception:
        return None


def _resource_stop_reason(descriptor_rows: int, temporary_root: Path) -> tuple[str | None, dict[str, int | None]]:
    raw_bytes = _descriptor_bytes(descriptor_rows)
    available_memory = read_available_physical_memory()
    available_disk = shutil.disk_usage(temporary_root).free
    resource = {
        "available_physical_memory_bytes": available_memory,
        "available_temporary_disk_bytes": available_disk,
    }
    if available_disk < raw_bytes + 64 * _MIB:
        return "INSUFFICIENT_TEMPORARY_DISK", resource
    one_block_bytes = _descriptor_bytes(
        min(descriptor_rows, DEFAULT_DESCRIPTOR_BLOCK_ROWS)
    )
    if available_memory is not None and available_memory < raw_bytes + one_block_bytes:
        return "INSUFFICIENT_AVAILABLE_MEMORY", resource
    return None, resource


WorkerRunner = Callable[..., WorkerOutcome]
DescriptorGenerator = Callable[..., dict[str, int]]
TemporaryDirectoryFactory = Callable[..., tempfile.TemporaryDirectory]


def run_benchmark(
    configuration: ScaleBenchmarkConfiguration,
    *,
    worker_runner: WorkerRunner = run_scale_in_fresh_process,
    descriptor_generator: DescriptorGenerator = generate_descriptor_file,
    temporary_directory_factory: TemporaryDirectoryFactory = tempfile.TemporaryDirectory,
) -> dict[str, object]:
    started = perf_counter_ns()
    selected_scales = configuration.selected_scales
    correctness = run_correctness_guards(configuration.seed)
    results: list[dict[str, object]] = []
    stop_reason: str | None = None
    next_unattempted: int | None = selected_scales[0] if selected_scales else None

    if not correctness["passed"]:
        stop_reason = "CORRECTNESS_GUARD_FAILED"
    elif not selected_scales:
        stop_reason = "OPERATOR_MAXIMUM_BELOW_FIRST_SCALE"
    else:
        for scale_index, descriptor_rows in enumerate(selected_scales):
            elapsed_seconds = (perf_counter_ns() - started) / 1_000_000_000
            if (
                configuration.wall_time_seconds is not None
                and elapsed_seconds >= configuration.wall_time_seconds
            ):
                stop_reason = "WALL_TIME_BUDGET_EXHAUSTED"
                next_unattempted = descriptor_rows
                break
            try:
                temporary_directory = temporary_directory_factory(
                    prefix="autocrop-exact-bf-scale-"
                )
            except (OSError, MemoryError):
                stop_reason = "DESCRIPTOR_FILE_CREATION_FAILED"
                next_unattempted = descriptor_rows
                break
            temporary_root = Path(temporary_directory.name)
            descriptor_path = temporary_root / "descriptors.f32"
            scale_result: dict[str, object] | None = None
            attempt_stop_reason: str | None = None
            try:
                resource_reason, resource = _resource_stop_reason(
                    descriptor_rows, temporary_root
                )
                if resource_reason is not None:
                    attempt_stop_reason = resource_reason
                else:
                    try:
                        generation = descriptor_generator(
                            descriptor_path,
                            descriptor_rows,
                            configuration.seed,
                            chunk_rows=configuration.generation_chunk_rows,
                        )
                    except (OSError, MemoryError, ValueError):
                        attempt_stop_reason = "DESCRIPTOR_FILE_CREATION_FAILED"
                if attempt_stop_reason is None:
                    elapsed_seconds = (perf_counter_ns() - started) / 1_000_000_000
                    remaining = (
                        None
                        if configuration.wall_time_seconds is None
                        else max(
                            0.0,
                            configuration.wall_time_seconds - elapsed_seconds,
                        )
                    )
                    if remaining == 0.0:
                        attempt_stop_reason = "WALL_TIME_BUDGET_EXHAUSTED"
                if attempt_stop_reason is None:
                    outcome = worker_runner(
                        descriptor_path,
                        descriptor_rows=descriptor_rows,
                        seed=configuration.seed,
                        warm_queries=configuration.warm_queries,
                        include_sensitivity=configuration.include_sensitivity,
                        timeout_seconds=remaining,
                    )
                    if outcome.status != "completed" or outcome.result is None:
                        attempt_stop_reason = outcome.stop_reason or "WORKER_FAILED"
                    else:
                        scale_result = dict(outcome.result)
                        scale_result["descriptor_file_bytes"] = generation[
                            "descriptor_file_bytes"
                        ]
                        scale_result["generation"] = generation
                        scale_result["resource_preflight"] = resource
            finally:
                cleanup_failed = False
                try:
                    temporary_directory.cleanup()
                except Exception:
                    cleanup_failed = True
                    try:
                        temporary_directory.cleanup()
                    except Exception:
                        pass
                try:
                    descriptor_remains = descriptor_path.exists()
                except OSError:
                    descriptor_remains = True
                if descriptor_remains:
                    cleanup_failed = True
            if cleanup_failed:
                stop_reason = "TEMPORARY_DESCRIPTOR_CLEANUP_FAILED"
                next_unattempted = descriptor_rows
                break
            if attempt_stop_reason is not None:
                stop_reason = attempt_stop_reason
                next_unattempted = descriptor_rows
                break
            if scale_result is None:
                stop_reason = "WORKER_FAILED"
                next_unattempted = descriptor_rows
                break
            scale_result["temporary_descriptor_cleanup_complete"] = True
            results.append(scale_result)
            next_unattempted = (
                selected_scales[scale_index + 1]
                if scale_index + 1 < len(selected_scales)
                else None
            )
        else:
            if configuration.maximum_descriptor_rows < configuration.scale_ladder[-1]:
                stop_reason = "OPERATOR_MAXIMUM_REACHED"
                next_unattempted = next(
                    (
                        rows
                        for rows in configuration.scale_ladder
                        if rows > configuration.maximum_descriptor_rows
                    ),
                    None,
                )
            else:
                stop_reason = "COMPLETED_REQUESTED_SCALES"
                next_unattempted = None

    return {
        "schema_version": SCALE_BENCHMARK_SCHEMA_VERSION,
        "benchmark_semantics": SCALE_BENCHMARK_SEMANTICS,
        "metadata": {
            "tool_version": __version__,
            "seed": configuration.seed,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "platform": platform.platform(),
            "processor": (
                platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or None
            ),
            "opencv_configuration": {
                "requested_threads_per_worker": 1,
                "rng_seed": configuration.seed,
                "effective_thread_count": "reported_per_completed_scale",
            },
            "filesystem_page_cache": "UNCONTROLLED",
            "timing_clock": "time.perf_counter_ns",
        },
        "configuration": {
            "descriptor_dimension": DESCRIPTOR_DIMENSION,
            "descriptor_dtype": DESCRIPTOR_DTYPE,
            "descriptors_per_original": DESCRIPTORS_PER_ORIGINAL,
            "query_descriptor_rows": DEFAULT_QUERY_DESCRIPTOR_ROWS,
            "neighbor_depth": DEFAULT_NEIGHBOR_DEPTH,
            "requested_k": DEFAULT_REQUESTED_K,
            "descriptor_block_rows": DEFAULT_DESCRIPTOR_BLOCK_ROWS,
            "requested_scale_ladder": list(configuration.scale_ladder),
            "selected_scale_ladder": list(selected_scales),
            "warm_query_count": configuration.warm_queries,
            "operator_maximum_descriptor_rows": configuration.maximum_descriptor_rows,
            "wall_time_seconds": configuration.wall_time_seconds,
            "include_sensitivity": configuration.include_sensitivity,
            "generation_chunk_rows": configuration.generation_chunk_rows,
        },
        "correctness_guard": correctness,
        "scales": results,
        "completion": {
            "completed_scales": [int(item["descriptor_rows"]) for item in results],
            "last_completed_scale": (
                int(results[-1]["descriptor_rows"]) if results else None
            ),
            "stop_reason": stop_reason,
            "next_unattempted_scale": next_unattempted,
            "elapsed_ns_including_generation_and_workers": max(
                0, perf_counter_ns() - started
            ),
        },
    }


def validate_output_path(output: Path | None) -> Path | None:
    if output is None:
        return None
    if output.suffix.lower() != ".json":
        raise ConfigurationError("output filename must end with .json")
    if output.exists():
        raise ConfigurationError("output must not already exist")
    try:
        parent = output.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigurationError("output parent must exist") from exc
    if not parent.is_dir():
        raise ConfigurationError("output parent must be a directory")
    return parent / output.name


def benchmark_exit_code(report: dict[str, object]) -> int:
    completion = report.get("completion")
    if not isinstance(completion, dict):
        return 1
    stop_reason = completion.get("stop_reason")
    return 0 if stop_reason in SUCCESSFUL_STOP_REASONS else 1


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output_stream = stdout if stdout is not None else sys.stdout
    error_stream = stderr if stderr is not None else sys.stderr
    arguments = build_parser().parse_args(argv)
    try:
        configuration = ScaleBenchmarkConfiguration(
            maximum_descriptor_rows=arguments.max_descriptor_rows,
            warm_queries=arguments.warm_queries,
            seed=arguments.seed,
            wall_time_seconds=arguments.wall_time_seconds,
            include_sensitivity=arguments.include_sensitivity,
        )
        output = validate_output_path(arguments.output)
    except (ConfigurationError, ValueError) as exc:
        print(f"configuration error: {exc}", file=error_stream)
        return 2
    try:
        report = run_benchmark(configuration)
        if output is not None:
            write_manifest_atomic(output, report)
        print(
            json.dumps(
                report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
            ),
            file=output_stream,
        )
        return benchmark_exit_code(report)
    except OutputFailure as exc:
        print(f"report output failure: {exc.error_type}", file=error_stream)
        return 3
    except Exception as exc:
        print(f"benchmark failure: {type(exc).__name__}", file=error_stream)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

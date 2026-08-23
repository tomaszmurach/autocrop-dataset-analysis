"""Descriptor-scale performance benchmark for the research-only FLANN path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from time import perf_counter_ns
from typing import Sequence, TextIO

import cv2
import numpy as np

from . import __version__
from .candidate_retrieval import (
    DESCRIPTOR_DIMENSION,
    DESCRIPTOR_DTYPE,
    _rank_descriptor_neighbors,
    current_runtime_contract,
    select_shortlist,
)
from .candidate_retrieval_flann import (
    DescriptorCorpusIdentity,
    ExperimentalFlannIndex,
    FlannExperimentError,
    FlannParameters,
    hash_file,
    source_rank,
)
from .candidate_retrieval_flann_benchmark import (
    DEFAULT_CHECKS,
    DEFAULT_TREES,
    _parse_positive_ints,
)
from .candidate_retrieval_profiling import RetrievalProfiler, profiling_stage
from .candidate_retrieval_scale_benchmark import (
    APPROVED_SCALE_LADDER,
    DEFAULT_GENERATION_CHUNK_ROWS,
    DEFAULT_QUERY_DESCRIPTOR_ROWS,
    DEFAULT_REQUESTED_K,
    DEFAULT_WARM_QUERIES,
    DESCRIPTORS_PER_ORIGINAL,
    ScaleBenchmarkConfiguration,
    WorkerOutcome,
    _resource_stop_reason,
    build_original_records,
    generate_descriptor_file,
    make_query_descriptors,
    run_scale_in_fresh_process,
)
from .cli import ConfigurationError, OutputFailure, write_manifest_atomic
from .content_matching import MatchingParameters, configure_opencv


FLANN_SCALE_SCHEMA_VERSION = "1.0"
FLANN_SCALE_SEMANTICS = "SYNTHETIC_DESCRIPTOR_ONLY_EXACT_BF_VS_FLANN_PERFORMANCE"


@dataclass(frozen=True, slots=True)
class FlannScaleConfiguration:
    maximum_descriptor_rows: int = APPROVED_SCALE_LADDER[0]
    trees: tuple[int, ...] = DEFAULT_TREES
    checks: tuple[int, ...] = DEFAULT_CHECKS
    neighbor_depth: int = 32
    warm_queries: int = DEFAULT_WARM_QUERIES
    seed: int = 17_029
    fresh_process_repetitions: int = 2
    generation_chunk_rows: int = DEFAULT_GENERATION_CHUNK_ROWS
    worker_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        ScaleBenchmarkConfiguration(
            maximum_descriptor_rows=self.maximum_descriptor_rows,
            warm_queries=self.warm_queries,
            seed=self.seed,
            generation_chunk_rows=self.generation_chunk_rows,
        )
        if not self.trees or not self.checks:
            raise ValueError("FLANN_PARAMETER_GRID_REQUIRED")
        if any(type(value) is not int or value <= 0 for value in self.trees + self.checks):
            raise ValueError("INVALID_FLANN_PARAMETER_GRID")
        if tuple(sorted(set(self.trees))) != self.trees:
            raise ValueError("FLANN_TREES_MUST_BE_STRICTLY_INCREASING")
        if tuple(sorted(set(self.checks))) != self.checks:
            raise ValueError("FLANN_CHECKS_MUST_BE_STRICTLY_INCREASING")
        if type(self.neighbor_depth) is not int or self.neighbor_depth <= 0:
            raise ValueError("INVALID_FLANN_NEIGHBOR_DEPTH")
        if (
            type(self.fresh_process_repetitions) is not int
            or self.fresh_process_repetitions <= 0
        ):
            raise ValueError("INVALID_FRESH_PROCESS_REPETITIONS")
        if self.worker_timeout_seconds is not None and self.worker_timeout_seconds <= 0:
            raise ValueError("INVALID_WORKER_TIMEOUT")

    @property
    def selected_scales(self) -> tuple[int, ...]:
        return tuple(
            rows
            for rows in APPROVED_SCALE_LADDER
            if rows <= self.maximum_descriptor_rows
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autocrop_analysis.candidate_retrieval_flann_scale_benchmark",
        description="Run operator-directed descriptor-scale FLANN feasibility tests.",
    )
    parser.add_argument(
        "--max-descriptor-rows", type=int, default=APPROVED_SCALE_LADDER[0]
    )
    parser.add_argument("--trees", default=",".join(map(str, DEFAULT_TREES)))
    parser.add_argument("--checks", default=",".join(map(str, DEFAULT_CHECKS)))
    parser.add_argument("--neighbor-depth", type=int, default=32)
    parser.add_argument("--warm-queries", type=int, default=DEFAULT_WARM_QUERIES)
    parser.add_argument("--seed", type=int, default=17_029)
    parser.add_argument("--fresh-process-repetitions", type=int, default=2)
    parser.add_argument("--worker-timeout-seconds", type=float)
    parser.add_argument("--output", type=Path)
    return parser


def _duration_summary(values: Sequence[int]) -> dict[str, object]:
    if not values:
        return {
            "raw_ns": [],
            "p50_ns": None,
            "p95_ns": None,
            "min_ns": None,
            "max_ns": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "raw_ns": list(values),
        "p50_ns": float(np.percentile(array, 50)),
        "p95_ns": float(np.percentile(array, 95)),
        "min_ns": min(values),
        "max_ns": max(values),
    }


def _single_stage_ns(profiler: RetrievalProfiler, stage: str) -> int:
    values = [item.elapsed_ns for item in profiler.measurements if item.stage == stage]
    if len(values) != 1:
        raise RuntimeError("UNEXPECTED_FLANN_PROFILER_STAGE_COUNT")
    return values[0]


def _measure_query(
    index: ExperimentalFlannIndex,
    matrix: np.ndarray,
    originals,
    query: np.ndarray,
    parameters: FlannParameters,
    *,
    expected_owner: int,
) -> tuple[
    dict[str, object],
    tuple[tuple[int, ...], ...],
    tuple[tuple[float, ...], ...],
]:
    profiler = RetrievalProfiler()
    started = perf_counter_ns()
    evidence = index.search(query, parameters, profiler=profiler)
    ranked = _rank_descriptor_neighbors(
        evidence.nearest(),
        descriptor_rows=int(matrix.shape[0]),
        originals=originals,
        profiler=profiler,
    )
    with profiling_stage(profiler, "query.shortlist_construction"):
        shortlist, extension, boundary = select_shortlist(ranked, DEFAULT_REQUESTED_K)
    total_ns = max(0, perf_counter_ns() - started)
    expected = originals[expected_owner].reference
    rank = source_rank(ranked, expected)
    descriptor_rows_sha256 = hashlib.sha256(
        np.ascontiguousarray(evidence.row_indices, dtype=np.dtype("<i8")).tobytes()
    ).hexdigest()
    descriptor_distances_sha256 = hashlib.sha256(
        np.ascontiguousarray(
            evidence.squared_l2_distances, dtype=np.dtype("<f4")
        ).tobytes()
    ).hexdigest()
    source_ranking_sha256 = _reference_sequence_sha256(
        [candidate.original.relative_path for candidate in ranked]
    )
    shortlist_sha256 = _reference_sequence_sha256(
        [candidate.original.relative_path for candidate in shortlist]
    )
    return (
        {
            "total_descriptor_retrieval_ns": total_ns,
            "flann_search_ns": _single_stage_ns(profiler, "query.flann_search"),
            "distance_normalization_ns": _single_stage_ns(
                profiler, "query.flann_distance_normalization"
            ),
            "vote_aggregation_ranking_ns": _single_stage_ns(
                profiler, "query.vote_aggregation_ranking"
            ),
            "shortlist_construction_ns": _single_stage_ns(
                profiler, "query.shortlist_construction"
            ),
            "known_source_rank": rank,
            "known_source_ranked_first": rank == 1,
            "ranked_originals": len(ranked),
            "returned_candidates": len(shortlist),
            "tie_extension_count": extension,
            "boundary_support_votes": boundary,
            "descriptor_neighbor_rows_sha256": descriptor_rows_sha256,
            "descriptor_neighbor_distances_sha256": descriptor_distances_sha256,
            "source_ranking_sha256": source_ranking_sha256,
            "shortlist_membership_sha256": shortlist_sha256,
        },
        evidence.descriptor_row_signature(),
        evidence.descriptor_distance_signature(),
    )


def _reference_sequence_sha256(relative_paths: Sequence[str]) -> str:
    encoded = json.dumps(
        list(relative_paths),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _measurement_summary(measurements: Sequence[dict[str, object]]) -> dict[str, object]:
    fields = (
        "total_descriptor_retrieval_ns",
        "flann_search_ns",
        "distance_normalization_ns",
        "vote_aggregation_ranking_ns",
        "shortlist_construction_ns",
    )
    return {
        field.removesuffix("_ns"): _duration_summary(
            [int(item[field]) for item in measurements]
        )
        for field in fields
    }


def execute_flann_scale_worker(
    descriptor_path: Path,
    *,
    descriptor_rows: int,
    descriptor_sha256: str,
    seed: int,
    trees: int,
    checks: Sequence[int],
    neighbor_depth: int,
    warm_queries: int,
) -> dict[str, object]:
    """Measure one tree count in one fresh child process."""

    configure_opencv(MatchingParameters(random_seed=seed))
    memory = RetrievalProfiler()
    memory.snapshot_memory("worker_start")
    matrix: np.memmap | None = None
    built = ExperimentalFlannIndex()
    loaded = ExperimentalFlannIndex()
    artifact_path = descriptor_path.with_name(
        f".{descriptor_path.name}.{os.getpid()}.trees-{trees}.flann"
    )
    try:
        matrix = np.memmap(
            descriptor_path,
            mode="r",
            dtype=np.dtype(DESCRIPTOR_DTYPE),
            shape=(descriptor_rows, DESCRIPTOR_DIMENSION),
        )
        memory.snapshot_memory("after_read_only_memmap_open")
        originals = build_original_records(descriptor_rows // DESCRIPTORS_PER_ORIGINAL)
        identity = DescriptorCorpusIdentity(
            descriptor_sha256,
            descriptor_sha256,
            descriptor_rows,
        )
        lifecycle = RetrievalProfiler()
        parameters = FlannParameters(
            trees=trees,
            checks=int(checks[0]),
            neighbor_depth=neighbor_depth,
        )
        memory.snapshot_memory("before_flann_build")
        built.build(matrix, parameters, profiler=lifecycle)
        memory.snapshot_memory("after_flann_build")

        first_query = make_query_descriptors(
            matrix,
            source_owner=0,
            query_rows=DEFAULT_QUERY_DESCRIPTOR_ROWS,
            seed=seed,
            query_ordinal=0,
        )
        before_reload_results = {}
        for check_count in checks:
            config = FlannParameters(trees, int(check_count), neighbor_depth)
            before_reload_results[str(check_count)] = _measure_query(
                built,
                matrix,
                originals,
                first_query,
                config,
                expected_owner=0,
            )
        artifact = built.save(artifact_path, identity, profiler=lifecycle)
        memory.snapshot_memory("after_flann_save")
        built.release()
        loaded.load(matrix, artifact_path, artifact, identity, profiler=lifecycle)
        memory.snapshot_memory("after_flann_load")

        configurations = []
        for check_count in checks:
            config = FlannParameters(trees, int(check_count), neighbor_depth)
            first_touch, first_row_signature, first_distance_signature = _measure_query(
                loaded,
                matrix,
                originals,
                first_query,
                config,
                expected_owner=0,
            )
            (
                repeated_first,
                repeated_row_signature,
                repeated_distance_signature,
            ) = _measure_query(
                loaded,
                matrix,
                originals,
                first_query,
                config,
                expected_owner=0,
            )
            warm = []
            for ordinal in range(1, warm_queries + 1):
                query = make_query_descriptors(
                    matrix,
                    source_owner=ordinal,
                    query_rows=DEFAULT_QUERY_DESCRIPTOR_ROWS,
                    seed=seed,
                    query_ordinal=ordinal,
                )
                measured, _, _ = _measure_query(
                    loaded,
                    matrix,
                    originals,
                    query,
                    config,
                    expected_owner=ordinal,
                )
                warm.append(measured)
            configurations.append(
                {
                    "parameters": config.as_dict(),
                    "first_touch": first_touch,
                    "warm_queries": warm,
                    "warm_summary": _measurement_summary(warm),
                    "correctness": {
                        "first_known_source_ranked_first": first_touch[
                            "known_source_ranked_first"
                        ],
                        "all_warm_known_sources_ranked_first": all(
                            bool(item["known_source_ranked_first"]) for item in warm
                        ),
                    },
                    "reproducibility": {
                        "same_index_descriptor_rows_stable": first_row_signature
                        == repeated_row_signature,
                        "same_index_descriptor_distances_stable": first_distance_signature
                        == repeated_distance_signature,
                        "same_index_source_ranking_stable": first_touch[
                            "source_ranking_sha256"
                        ]
                        == repeated_first["source_ranking_sha256"],
                        "same_index_shortlist_membership_stable": first_touch[
                            "shortlist_membership_sha256"
                        ]
                        == repeated_first["shortlist_membership_sha256"],
                        "save_reload_descriptor_rows_stable": first_row_signature
                        == before_reload_results[str(check_count)][1],
                        "save_reload_descriptor_distances_stable": first_distance_signature
                        == before_reload_results[str(check_count)][2],
                        "save_reload_source_ranking_stable": first_touch[
                            "source_ranking_sha256"
                        ]
                        == before_reload_results[str(check_count)][0][
                            "source_ranking_sha256"
                        ],
                        "save_reload_shortlist_membership_stable": first_touch[
                            "shortlist_membership_sha256"
                        ]
                        == before_reload_results[str(check_count)][0][
                            "shortlist_membership_sha256"
                        ],
                    },
                }
            )
        loaded.release()
        matrix._mmap.close()
        matrix = None
        artifact_path.unlink(missing_ok=True)
        memory.snapshot_memory("final_state")
        return {
            "descriptor_rows": descriptor_rows,
            "synthetic_originals": len(originals),
            "tree_count": trees,
            "descriptor_diagnostics": _descriptor_diagnostics_for_report(
                descriptor_rows
            ),
            "opencv_configuration": {
                "requested_threads": 1,
                "effective_thread_count": (
                    int(cv2.getNumThreads())
                    if hasattr(cv2, "getNumThreads")
                    else None
                ),
                "rng_seed": seed,
            },
            "lifecycle": {
                "build_ns": _single_stage_ns(lifecycle, "flann.index_build"),
                "save_ns": _single_stage_ns(lifecycle, "flann.index_save"),
                "load_ns": _single_stage_ns(lifecycle, "flann.index_load"),
                "artifact": artifact.as_dict(),
            },
            "configurations": configurations,
            "memory": memory.as_report()["memory"],
            "temporary_artifact_cleanup_complete": not artifact_path.exists(),
        }
    finally:
        loaded.release()
        built.release()
        if matrix is not None and getattr(matrix, "_mmap", None) is not None:
            matrix._mmap.close()
        try:
            artifact_path.unlink(missing_ok=True)
        except OSError:
            pass


def _descriptor_diagnostics_for_report(descriptor_rows: int) -> dict[str, object]:
    return {
        "dtype": DESCRIPTOR_DTYPE,
        "shape": [descriptor_rows, DESCRIPTOR_DIMENSION],
        "opened_mode": "read_only_memmap",
    }


def _worker_entry(connection, request: dict[str, object]) -> None:
    try:
        result = execute_flann_scale_worker(
            Path(str(request["descriptor_path"])),
            descriptor_rows=int(request["descriptor_rows"]),
            descriptor_sha256=str(request["descriptor_sha256"]),
            seed=int(request["seed"]),
            trees=int(request["trees"]),
            checks=tuple(int(item) for item in request["checks"]),
            neighbor_depth=int(request["neighbor_depth"]),
            warm_queries=int(request["warm_queries"]),
        )
        connection.send({"status": "completed", "result": result})
    except Exception as exc:
        try:
            connection.send(
                {
                    "status": "failed",
                    "stop_reason": str(exc) or type(exc).__name__,
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def run_flann_in_fresh_process(
    descriptor_path: Path,
    *,
    descriptor_rows: int,
    descriptor_sha256: str,
    seed: int,
    trees: int,
    checks: Sequence[int],
    neighbor_depth: int,
    warm_queries: int,
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
                "descriptor_sha256": descriptor_sha256,
                "seed": seed,
                "trees": trees,
                "checks": list(checks),
                "neighbor_depth": neighbor_depth,
                "warm_queries": warm_queries,
            },
        ),
    )
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            process.terminate()
            process.join()
            return WorkerOutcome("failed", stop_reason="WORKER_TIMEOUT")
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


def _semantic_projection(result: dict[str, object]) -> dict[str, object]:
    return {
        "descriptor_rows": result["descriptor_rows"],
        "tree_count": result["tree_count"],
        "configurations": [
            {
                "parameters": item["parameters"],
                "first_known_source_rank": item["first_touch"]["known_source_rank"],
                "warm_known_source_ranks": [
                    query["known_source_rank"] for query in item["warm_queries"]
                ],
                "first_returned_candidates": item["first_touch"][
                    "returned_candidates"
                ],
                "first_descriptor_neighbor_rows_sha256": item["first_touch"][
                    "descriptor_neighbor_rows_sha256"
                ],
                "first_descriptor_neighbor_distances_sha256": item["first_touch"][
                    "descriptor_neighbor_distances_sha256"
                ],
                "first_source_ranking_sha256": item["first_touch"][
                    "source_ranking_sha256"
                ],
                "first_shortlist_membership_sha256": item["first_touch"][
                    "shortlist_membership_sha256"
                ],
                "warm_descriptor_neighbor_rows_sha256": [
                    query["descriptor_neighbor_rows_sha256"]
                    for query in item["warm_queries"]
                ],
                "warm_descriptor_neighbor_distances_sha256": [
                    query["descriptor_neighbor_distances_sha256"]
                    for query in item["warm_queries"]
                ],
                "warm_source_ranking_sha256": [
                    query["source_ranking_sha256"]
                    for query in item["warm_queries"]
                ],
                "warm_shortlist_membership_sha256": [
                    query["shortlist_membership_sha256"]
                    for query in item["warm_queries"]
                ],
            }
            for item in result["configurations"]
        ],
    }


def _fresh_process_evidence_projection(
    result: dict[str, object], field: str
) -> tuple[tuple[object, tuple[object, ...]], ...]:
    return tuple(
        (
            item["first_touch"][field],
            tuple(query[field] for query in item["warm_queries"]),
        )
        for item in result["configurations"]
    )


def run_benchmark(
    configuration: FlannScaleConfiguration,
    *,
    exact_worker=run_scale_in_fresh_process,
    flann_worker=run_flann_in_fresh_process,
    descriptor_generator=generate_descriptor_file,
    temporary_directory_factory=tempfile.TemporaryDirectory,
) -> dict[str, object]:
    scales: list[dict[str, object]] = []
    selected = configuration.selected_scales
    stop_reason: str | None = None
    next_unattempted = selected[0] if selected else None
    if not selected:
        stop_reason = "OPERATOR_MAXIMUM_BELOW_FIRST_SCALE"
    else:
        for scale_index, descriptor_rows in enumerate(selected):
            try:
                temporary_directory = temporary_directory_factory(
                    prefix="autocrop-flann-scale-"
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
                resource_reason, resources = _resource_stop_reason(
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
                        descriptor_sha256, _ = hash_file(descriptor_path)
                    except (OSError, MemoryError, ValueError):
                        attempt_stop_reason = "DESCRIPTOR_FILE_CREATION_FAILED"
                if attempt_stop_reason is None:
                    exact = exact_worker(
                        descriptor_path,
                        descriptor_rows=descriptor_rows,
                        seed=configuration.seed,
                        warm_queries=configuration.warm_queries,
                        include_sensitivity=False,
                        timeout_seconds=configuration.worker_timeout_seconds,
                    )
                    if exact.status != "completed" or exact.result is None:
                        attempt_stop_reason = (
                            exact.stop_reason or "EXACT_BF_WORKER_FAILED"
                        )
                tree_results = []
                if attempt_stop_reason is None:
                    for tree_count in configuration.trees:
                        repetitions = []
                        for _ in range(configuration.fresh_process_repetitions):
                            outcome = flann_worker(
                                descriptor_path,
                                descriptor_rows=descriptor_rows,
                                descriptor_sha256=descriptor_sha256,
                                seed=configuration.seed,
                                trees=tree_count,
                                checks=configuration.checks,
                                neighbor_depth=configuration.neighbor_depth,
                                warm_queries=configuration.warm_queries,
                                timeout_seconds=configuration.worker_timeout_seconds,
                            )
                            if outcome.status != "completed" or outcome.result is None:
                                attempt_stop_reason = (
                                    outcome.stop_reason or "FLANN_WORKER_FAILED"
                                )
                                break
                            repetitions.append(outcome.result)
                        if attempt_stop_reason is not None:
                            break
                        evidence_fields = {
                            "fresh_process_descriptor_rows_stable": "descriptor_neighbor_rows_sha256",
                            "fresh_process_descriptor_distances_stable": "descriptor_neighbor_distances_sha256",
                            "fresh_process_source_rankings_stable": "source_ranking_sha256",
                            "fresh_process_shortlist_membership_stable": "shortlist_membership_sha256",
                        }
                        tree_results.append(
                            {
                                "trees": tree_count,
                                "fresh_process_repetitions": repetitions,
                                "reproducibility": {
                                    label: all(
                                        _fresh_process_evidence_projection(
                                            repetition, field
                                        )
                                        == _fresh_process_evidence_projection(
                                            repetitions[0], field
                                        )
                                        for repetition in repetitions[1:]
                                    )
                                    for label, field in evidence_fields.items()
                                },
                            }
                        )
                if attempt_stop_reason is None:
                    scale_result = {
                        "descriptor_rows": descriptor_rows,
                        "synthetic_originals": descriptor_rows
                        // DESCRIPTORS_PER_ORIGINAL,
                        "generation": generation,
                        "resource_preflight": resources,
                        "descriptor_sha256": descriptor_sha256,
                        "exact_bf_reference": exact.result,
                        "flann": tree_results,
                    }
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
                stop_reason = "FLANN_WORKER_FAILED"
                next_unattempted = descriptor_rows
                break
            scale_result["temporary_descriptor_cleanup_complete"] = True
            scales.append(scale_result)
            next_unattempted = (
                selected[scale_index + 1]
                if scale_index + 1 < len(selected)
                else None
            )
        else:
            if configuration.maximum_descriptor_rows < APPROVED_SCALE_LADDER[-1]:
                stop_reason = "OPERATOR_MAXIMUM_REACHED"
                next_unattempted = next(
                    (
                        rows
                        for rows in APPROVED_SCALE_LADDER
                        if rows > configuration.maximum_descriptor_rows
                    ),
                    None,
                )
            else:
                stop_reason = "COMPLETED_REQUESTED_SCALES"
                next_unattempted = None
    return {
        "schema_version": FLANN_SCALE_SCHEMA_VERSION,
        "tool_version": __version__,
        "benchmark_semantics": FLANN_SCALE_SEMANTICS,
        "status": "RESEARCH_ONLY_NOT_INTEGRATED",
        "runtime": current_runtime_contract(),
        "configuration": {
            "maximum_descriptor_rows": configuration.maximum_descriptor_rows,
            "selected_scales": list(configuration.selected_scales),
            "trees": list(configuration.trees),
            "checks": list(configuration.checks),
            "neighbor_depth": configuration.neighbor_depth,
            "warm_queries": configuration.warm_queries,
            "fresh_process_repetitions": configuration.fresh_process_repetitions,
            "primary_safety_k": DEFAULT_REQUESTED_K,
            "descriptor_scale_semantics": "PERFORMANCE_EVIDENCE_NOT_KNOWN_SOURCE_RECALL",
        },
        "scales": scales,
        "completion": {
            "completed_scales": [int(item["descriptor_rows"]) for item in scales],
            "last_completed_scale": (
                int(scales[-1]["descriptor_rows"]) if scales else None
            ),
            "stop_reason": stop_reason,
            "next_unattempted_scale": next_unattempted,
        },
    }


def _validate_output(output: Path | None) -> Path | None:
    if output is None:
        return None
    resolved = output.resolve(strict=False)
    if resolved.suffix.lower() != ".json":
        raise ConfigurationError("output filename must end with .json")
    if not resolved.parent.exists() or not resolved.parent.is_dir():
        raise ConfigurationError("output parent must exist")
    if resolved.exists():
        raise ConfigurationError("output must not already exist")
    return resolved


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    import sys

    out = stdout or sys.stdout
    err = stderr or sys.stderr
    try:
        arguments = build_parser().parse_args(argv)
        configuration = FlannScaleConfiguration(
            maximum_descriptor_rows=arguments.max_descriptor_rows,
            trees=_parse_positive_ints(arguments.trees, "trees"),
            checks=_parse_positive_ints(arguments.checks, "checks"),
            neighbor_depth=arguments.neighbor_depth,
            warm_queries=arguments.warm_queries,
            seed=arguments.seed,
            fresh_process_repetitions=arguments.fresh_process_repetitions,
            worker_timeout_seconds=arguments.worker_timeout_seconds,
        )
        output = _validate_output(arguments.output)
        report = run_benchmark(configuration)
        if output is not None:
            write_manifest_atomic(output, report)
        print(
            json.dumps(
                report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
            ),
            file=out,
        )
        return 0 if report["completion"]["stop_reason"] in {
            "COMPLETED_REQUESTED_SCALES",
            "OPERATOR_MAXIMUM_REACHED",
        } else 1
    except FlannExperimentError as exc:
        print(f"benchmark failure: {exc.code}", file=err)
        return 1
    except (ConfigurationError, ValueError) as exc:
        print(f"configuration error: {exc}", file=err)
        return 2
    except OutputFailure as exc:
        print(f"output error: {exc}", file=err)
        return 3
    except Exception as exc:
        print(f"benchmark failure: {type(exc).__name__}: {exc}", file=err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

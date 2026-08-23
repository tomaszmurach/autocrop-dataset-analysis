"""Image-level exact-BF versus FLANN feasibility benchmark.

Synthetic mode reports genuine known-source Recall@K.  Bounded-real mode
reports prediction-consistency against an existing exhaustive provenance
report.  Neither mode changes the normal candidate-retrieval CLI.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence, TextIO

import numpy as np

from . import __version__
from .audit import ReadStatus, RootRole, SemanticReference, audit_root
from .candidate_retrieval import (
    IndexParameters,
    QueryParameters,
    RetrievalCandidate,
    _rank_descriptor_neighbors,
    current_runtime_contract,
    recall_at_k,
    retrieve_candidates,
    select_shortlist,
    select_spatially_balanced_descriptors,
)
from .candidate_retrieval_benchmark import (
    DEFAULT_STRATA,
    _save_query,
    _structured_image,
)
from .candidate_retrieval_cli import (
    LoadedIndex,
    QueryPaths,
    build_index,
    load_index,
    validate_build_paths,
    validate_query_paths,
)
from .candidate_retrieval_flann import (
    DescriptorCorpusIdentity,
    ExperimentalFlannIndex,
    FlannExperimentError,
    FlannParameters,
    compare_source_rankings,
    source_rank,
)
from .candidate_retrieval_profiling import RetrievalProfiler, profiling_stage
from .cli import ConfigurationError, OutputFailure, write_manifest_atomic
from .content_matching import (
    MatchingParameters,
    configure_opencv,
    extract_features,
    unavailable_feature_image,
)


IMAGE_BENCHMARK_SCHEMA_VERSION = "1.0"
IMAGE_BENCHMARK_SEMANTICS = "EXACT_BF_VS_FLANN_RESEARCH_FEASIBILITY"
DEFAULT_TREES = (1, 4, 8)
DEFAULT_CHECKS = (32, 64, 128, 256)
DEFAULT_K_VALUES = (5, 10, 20, 50)
PRIVATE_JSON_SUFFIX = ".private.json"


@dataclass(frozen=True, slots=True)
class CompactQuery:
    crop: SemanticReference
    descriptors: np.ndarray
    extracted_descriptor_count: int


@dataclass(frozen=True, slots=True)
class HistoricalPrediction:
    crop: SemanticReference
    decision: str
    ranked_originals: tuple[SemanticReference, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autocrop_analysis.candidate_retrieval_flann_benchmark",
        description="Compare research-only FLANN KD-tree retrieval with exact BF.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    synthetic = commands.add_parser(
        "synthetic", help="Run deterministic image-level known-source evaluation."
    )
    synthetic.add_argument("--corpus-size", type=int, default=64)
    synthetic.add_argument("--queries", type=int, default=20)
    synthetic.add_argument("--original-max-descriptors", type=int, default=128)
    synthetic.add_argument("--query-max-descriptors", type=int, default=64)
    synthetic.add_argument("--strata", default=",".join(DEFAULT_STRATA))
    synthetic.add_argument("--seed", type=int, default=17_029)
    _add_common_arguments(synthetic)

    bounded = commands.add_parser(
        "bounded-real",
        help="Compare against an existing exhaustive provenance prediction report.",
    )
    bounded.add_argument("--index", required=True, type=Path)
    bounded.add_argument("--cropped", required=True, type=Path)
    bounded.add_argument("--provenance-predictions", required=True, type=Path)
    bounded.add_argument("--query-max-descriptors", type=int, default=64)
    _add_common_arguments(bounded, output_required=True)
    return parser


def _add_common_arguments(
    parser: argparse.ArgumentParser, *, output_required: bool = False
) -> None:
    parser.add_argument("--trees", default=",".join(map(str, DEFAULT_TREES)))
    parser.add_argument("--checks", default=",".join(map(str, DEFAULT_CHECKS)))
    parser.add_argument("--neighbor-depth", type=int, default=32)
    parser.add_argument("--k-values", default=",".join(map(str, DEFAULT_K_VALUES)))
    parser.add_argument("--output", required=output_required, type=Path)


def _positive_int(value: int, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigurationError(f"{label} must be a positive integer")
    return value


def _parse_positive_ints(value: str, label: str) -> tuple[int, ...]:
    try:
        result = tuple(
            sorted({_positive_int(int(part.strip()), label) for part in value.split(",")})
        )
    except ValueError as exc:
        raise ConfigurationError(f"{label} must be comma-separated integers") from exc
    if not result:
        raise ConfigurationError(f"{label} values are required")
    return result


def _parse_strata(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result or any(item not in DEFAULT_STRATA for item in result):
        raise ConfigurationError("invalid synthetic stratum")
    return result


def _require_standard_k_values(k_values: Sequence[int]) -> None:
    missing = set(DEFAULT_K_VALUES) - set(k_values)
    if missing:
        raise ConfigurationError("K values must include 5,10,20,50")


def _validate_output(
    output: Path | None, *, private: bool = False
) -> Path | None:
    if output is None:
        if private:
            raise ConfigurationError("bounded-real output is required")
        return None
    if private:
        if not output.name.endswith(PRIVATE_JSON_SUFFIX):
            raise ConfigurationError("output filename must end with .private.json")
    elif output.suffix.lower() != ".json":
        raise ConfigurationError("output filename must end with .json")
    if output.exists():
        raise ConfigurationError("output must not already exist")
    try:
        parent = output.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigurationError("output parent must exist") from exc
    if not parent.is_dir():
        raise ConfigurationError("output parent must exist")
    return parent / output.name


def _validate_predictions_path(path: Path) -> Path:
    if not path.name.endswith(PRIVATE_JSON_SUFFIX):
        raise ConfigurationError(
            "provenance predictions filename must end with .private.json"
        )
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigurationError("provenance predictions must exist") from exc
    if not resolved.is_file():
        raise ConfigurationError("provenance predictions must be a regular file")
    if not os.access(resolved, os.R_OK):
        raise ConfigurationError("provenance predictions must be readable")
    try:
        with resolved.open("rb"):
            pass
    except OSError as exc:
        raise ConfigurationError("provenance predictions must be readable") from exc
    return resolved


def _validate_bounded_real_paths(
    arguments: argparse.Namespace, output: Path
) -> tuple[QueryPaths, Path]:
    predictions = _validate_predictions_path(arguments.provenance_predictions)
    paths = validate_query_paths(arguments.index, arguments.cropped, output)
    return paths, predictions


def _load_compact_queries(
    paths: QueryPaths,
    loaded: LoadedIndex,
    *,
    query_max_descriptors: int,
    profiler: RetrievalProfiler,
) -> tuple[CompactQuery, ...]:
    matching = MatchingParameters(
        sift_nfeatures=loaded.metadata.parameters.sift_nfeatures,
        random_seed=loaded.metadata.parameters.random_seed,
    )
    configure_opencv(matching)
    with profiling_stage(profiler, "query.crop_root_audit") as timing:
        audited = audit_root(paths.cropped, RootRole.CROPPED)
        candidates = tuple(item for item in audited.items if item.is_image_candidate)
        timing.add_work(
            audited_regular_files=len(audited.items),
            query_image_candidates=len(candidates),
            scan_issues=len(audited.scan_issues),
        )
    queries: list[CompactQuery] = []
    for ordinal, item in enumerate(candidates):
        with profiling_stage(
            profiler, "query.feature_extraction", item_ordinal=ordinal
        ) as timing:
            if item.read_status is ReadStatus.READABLE:
                feature = extract_features(
                    paths.cropped / Path(item.relative_path),
                    item.reference,
                    matching,
                    retain_grayscale=False,
                )
            else:
                feature = unavailable_feature_image(item.reference, "AUDIT_UNAVAILABLE")
            timing.add_work(
                display_pixels=max(0, feature.width * feature.height),
                extracted_descriptor_rows=len(feature.keypoints),
            )
        with profiling_stage(
            profiler, "query.compact_descriptor_selection", item_ordinal=ordinal
        ) as timing:
            compact = select_spatially_balanced_descriptors(
                feature,
                maximum=query_max_descriptors,
                grid_rows=loaded.metadata.parameters.grid_rows,
                grid_columns=loaded.metadata.parameters.grid_columns,
            )
            timing.add_work(
                extracted_descriptor_rows=len(feature.keypoints),
                selected_query_descriptor_rows=int(compact.shape[0]),
            )
        queries.append(CompactQuery(item.reference, compact, len(feature.keypoints)))
    return tuple(queries)


def _exact_rankings(
    loaded: LoadedIndex,
    queries: Sequence[CompactQuery],
    *,
    neighbor_depth: int,
) -> tuple[tuple[tuple[RetrievalCandidate, ...], ...], RetrievalProfiler]:
    profiler = RetrievalProfiler()
    parameters = QueryParameters(
        query_max_descriptors=max(
            (int(query.descriptors.shape[0]) for query in queries), default=1
        ),
        neighbor_depth=neighbor_depth,
        requested_k=50,
    )
    rankings = []
    for ordinal, query in enumerate(queries):
        with profiling_stage(
            profiler, "query.exact_retrieval_total", item_ordinal=ordinal
        ):
            rankings.append(
                retrieve_candidates(
                    query.descriptors,
                    loaded.descriptors,
                    loaded.metadata.originals,
                    parameters,
                    profiler=profiler,
                    item_ordinal=ordinal,
                )
            )
    return tuple(rankings), profiler


def _retrieve_flann(
    index: ExperimentalFlannIndex,
    loaded: LoadedIndex,
    queries: Sequence[CompactQuery],
    parameters: FlannParameters,
    profiler: RetrievalProfiler | None,
) -> tuple[
    tuple[tuple[RetrievalCandidate, ...], ...],
    tuple[tuple[tuple[int, ...], ...], ...],
    tuple[tuple[tuple[float, ...], ...], ...],
]:
    rankings = []
    row_signatures = []
    distance_signatures = []
    for ordinal, query in enumerate(queries):
        with profiling_stage(
            profiler, "query.flann_retrieval_total", item_ordinal=ordinal
        ):
            if query.descriptors.shape[0] == 0:
                rankings.append(())
                row_signatures.append(())
                distance_signatures.append(())
                continue
            evidence = index.search(
                query.descriptors,
                parameters,
                profiler=profiler,
                item_ordinal=ordinal,
            )
            rankings.append(
                _rank_descriptor_neighbors(
                    evidence.nearest(),
                    descriptor_rows=loaded.metadata.total_descriptor_rows,
                    originals=loaded.metadata.originals,
                    profiler=profiler,
                    item_ordinal=ordinal,
                )
            )
            row_signatures.append(evidence.descriptor_row_signature())
            distance_signatures.append(evidence.descriptor_distance_signature())
    return tuple(rankings), tuple(row_signatures), tuple(distance_signatures)


def _source_ranking_signature(
    rankings: Sequence[Sequence[RetrievalCandidate]],
) -> tuple[tuple[SemanticReference, ...], ...]:
    return tuple(
        tuple(candidate.original for candidate in ranking) for ranking in rankings
    )


def _shortlist_membership_signature(
    rankings: Sequence[Sequence[RetrievalCandidate]], k_values: Sequence[int]
) -> tuple[tuple[frozenset[SemanticReference], ...], ...]:
    return tuple(
        tuple(
            frozenset(candidate.original for candidate in select_shortlist(ranking, k)[0])
            for k in k_values
        )
        for ranking in rankings
    )


def _duration_summary(profiler: RetrievalProfiler, stage: str) -> dict[str, object]:
    values = np.asarray(
        [
            measurement.elapsed_ns
            for measurement in profiler.measurements
            if measurement.stage == stage
        ],
        dtype=np.float64,
    )
    if values.size == 0:
        return {"count": 0, "p50_ns": None, "p95_ns": None, "total_ns": 0}
    return {
        "count": int(values.size),
        "p50_ns": float(np.percentile(values, 50)),
        "p95_ns": float(np.percentile(values, 95)),
        "total_ns": int(values.sum()),
    }


def _aggregate_comparison(
    queries: Sequence[CompactQuery],
    exact_rankings: Sequence[Sequence[RetrievalCandidate]],
    flann_rankings: Sequence[Sequence[RetrievalCandidate]],
    *,
    k_values: Sequence[int],
    known_sources: Mapping[str, SemanticReference] | None,
    strata: Mapping[str, str] | None,
) -> dict[str, object]:
    individual = []
    top_retained = 0
    top_source_cases = 0
    top_retained_by_k = Counter()
    containment_counts = Counter()
    retention_values: dict[int, list[float]] = defaultdict(list)
    known_cases = []
    losses_at_50 = []
    for query, exact, flann in zip(queries, exact_rankings, flann_rankings, strict=True):
        source = (
            known_sources.get(query.crop.relative_path)
            if known_sources is not None
            else None
        )
        comparison = compare_source_rankings(
            exact, flann, k_values=k_values, known_source=source
        )
        if exact:
            top_source_cases += 1
        if comparison["exact_top_source_retained"]:
            top_retained += 1
        for k in k_values:
            item = comparison["by_k"][str(k)]
            containment_counts[k] += int(item["exact_contained_by_flann"])
            top_retained_by_k[k] += int(
                bool(item["exact_top_source_flann_present"])
            )
            retention_values[k].append(float(item["exact_candidate_retention"]))
        if source is not None:
            known_cases.append((source, flann))
            at_50 = comparison["by_k"].get("50")
            if (
                at_50 is not None
                and at_50["known_source_exact_present"]
                and not at_50["known_source_flann_present"]
            ):
                losses_at_50.append(query.crop.relative_path)
        individual.append(
            {
                "crop": query.crop.relative_path,
                "known_source": source.relative_path if source is not None else None,
                "stratum": (
                    strata.get(query.crop.relative_path) if strata is not None else None
                ),
                "comparison": comparison,
            }
        )
    report: dict[str, object] = {
        "query_count": len(queries),
        "exact_top_source_retention": {
            "applicable_query_count": top_source_cases,
            "count": top_retained,
            "rate": top_retained / top_source_cases if top_source_cases else 0.0,
        },
        "oracle_agreement_by_k": {
            str(k): {
                "exact_top_source_retention_count": top_retained_by_k[k],
                "exact_top_source_retention_rate": (
                    top_retained_by_k[k] / top_source_cases
                    if top_source_cases
                    else 0.0
                ),
                "exact_shortlist_containment_count": containment_counts[k],
                "exact_shortlist_containment_rate": (
                    containment_counts[k] / len(queries) if queries else 0.0
                ),
                "mean_exact_candidate_retention": (
                    float(np.mean(retention_values[k]))
                    if retention_values[k]
                    else 0.0
                ),
            }
            for k in k_values
        },
        "individual_queries": individual,
    }
    if known_sources is not None:
        report["synthetic_known_source_recall"] = {
            f"Recall@{k}": value
            for k, value in recall_at_k(known_cases, k_values).items()
        }
        report["known_source_losses_at_k50_where_exact_bf_retained"] = losses_at_50
    return report


def _run_grid(
    loaded: LoadedIndex,
    queries: Sequence[CompactQuery],
    *,
    exact_rankings: Sequence[Sequence[RetrievalCandidate]],
    corpus: DescriptorCorpusIdentity,
    trees: Sequence[int],
    checks: Sequence[int],
    neighbor_depth: int,
    k_values: Sequence[int],
    artifact_directory: Path,
    known_sources: Mapping[str, SemanticReference] | None = None,
    strata: Mapping[str, str] | None = None,
    predictions: Sequence[HistoricalPrediction] | None = None,
) -> list[dict[str, object]]:
    results = []
    for tree_count in trees:
        build_parameters = FlannParameters(
            trees=tree_count,
            checks=checks[0],
            neighbor_depth=neighbor_depth,
        )
        lifecycle_profiler = RetrievalProfiler()
        lifecycle_profiler.snapshot_memory("before_flann_build")
        built = ExperimentalFlannIndex()
        loaded_index = ExperimentalFlannIndex()
        rebuilt = ExperimentalFlannIndex()
        artifact_path = artifact_directory / f"trees-{tree_count}.flann"
        try:
            built.build(loaded.descriptors, build_parameters, profiler=lifecycle_profiler)
            lifecycle_profiler.snapshot_memory("after_flann_build")
            artifact = built.save(
                artifact_path, corpus, profiler=lifecycle_profiler
            )
            lifecycle_profiler.snapshot_memory("after_flann_save")
            built_results = {}
            for check_count in checks:
                parameters = FlannParameters(
                    trees=tree_count,
                    checks=check_count,
                    neighbor_depth=neighbor_depth,
                )
                built_results[check_count] = _retrieve_flann(
                    built, loaded, queries, parameters, None
                )
            built.release()

            loaded_index.load(
                loaded.descriptors,
                artifact_path,
                artifact,
                corpus,
                profiler=lifecycle_profiler,
            )
            lifecycle_profiler.snapshot_memory("after_flann_load")
            rebuilt.build(loaded.descriptors, build_parameters)
            for check_count in checks:
                parameters = FlannParameters(
                    trees=tree_count,
                    checks=check_count,
                    neighbor_depth=neighbor_depth,
                )
                query_profiler = RetrievalProfiler()
                query_profiler.snapshot_memory("before_flann_query_batch")
                (
                    flann_rankings,
                    reloaded_row_signatures,
                    reloaded_distance_signatures,
                ) = _retrieve_flann(
                    loaded_index, loaded, queries, parameters, query_profiler
                )
                query_profiler.snapshot_memory("after_flann_query_batch")
                (
                    repeated_rankings,
                    repeated_row_signatures,
                    repeated_distance_signatures,
                ) = _retrieve_flann(
                    loaded_index, loaded, queries, parameters, None
                )
                (
                    rebuilt_rankings,
                    rebuilt_row_signatures,
                    rebuilt_distance_signatures,
                ) = _retrieve_flann(
                    rebuilt, loaded, queries, parameters, None
                )
                (
                    built_rankings,
                    built_row_signatures,
                    built_distance_signatures,
                ) = built_results[check_count]
                reloaded_source_ranking = _source_ranking_signature(flann_rankings)
                reloaded_shortlist_membership = _shortlist_membership_signature(
                    flann_rankings, k_values
                )
                comparison = _aggregate_comparison(
                    queries,
                    exact_rankings,
                    flann_rankings,
                    k_values=k_values,
                    known_sources=known_sources,
                    strata=strata,
                )
                result: dict[str, object] = {
                    "parameters": parameters.as_dict(),
                    "comparison": comparison,
                    "performance": {
                        "flann_search": _duration_summary(
                            query_profiler, "query.flann_search"
                        ),
                        "distance_normalization": _duration_summary(
                            query_profiler, "query.flann_distance_normalization"
                        ),
                        "vote_aggregation_ranking": _duration_summary(
                            query_profiler, "query.vote_aggregation_ranking"
                        ),
                        "retrieval_excluding_feature_extraction": _duration_summary(
                            query_profiler, "query.flann_retrieval_total"
                        ),
                        "profiling": query_profiler.as_report(),
                    },
                    "reproducibility": {
                        "same_index_descriptor_rows_stable": repeated_row_signatures
                        == reloaded_row_signatures,
                        "same_index_descriptor_distances_stable": repeated_distance_signatures
                        == reloaded_distance_signatures,
                        "same_index_source_rankings_stable": _source_ranking_signature(
                            repeated_rankings
                        )
                        == reloaded_source_ranking,
                        "same_index_shortlist_membership_stable": _shortlist_membership_signature(
                            repeated_rankings, k_values
                        )
                        == reloaded_shortlist_membership,
                        "save_reload_descriptor_rows_stable": built_row_signatures
                        == reloaded_row_signatures,
                        "save_reload_descriptor_distances_stable": built_distance_signatures
                        == reloaded_distance_signatures,
                        "save_reload_source_rankings_stable": _source_ranking_signature(
                            built_rankings
                        )
                        == reloaded_source_ranking,
                        "save_reload_shortlist_membership_stable": _shortlist_membership_signature(
                            built_rankings, k_values
                        )
                        == reloaded_shortlist_membership,
                        "same_process_rebuild_descriptor_rows_stable": rebuilt_row_signatures
                        == reloaded_row_signatures,
                        "same_process_rebuild_descriptor_distances_stable": rebuilt_distance_signatures
                        == reloaded_distance_signatures,
                        "same_process_rebuild_source_rankings_stable": _source_ranking_signature(
                            rebuilt_rankings
                        )
                        == reloaded_source_ranking,
                        "same_process_rebuild_shortlist_membership_stable": _shortlist_membership_signature(
                            rebuilt_rankings, k_values
                        )
                        == reloaded_shortlist_membership,
                    },
                }
                if predictions is not None:
                    result["bounded_real_prediction_consistency"] = (
                        _prediction_consistency(
                            queries,
                            exact_rankings,
                            flann_rankings,
                            predictions,
                            k_values,
                        )
                    )
                results.append(result)
            rebuilt.release()
            loaded_index.release()
            lifecycle_profiler.snapshot_memory("after_flann_release")
            lifecycle_report = {
                "trees": tree_count,
                "artifact": artifact.as_dict(),
                "build": _duration_summary(lifecycle_profiler, "flann.index_build"),
                "save": _duration_summary(lifecycle_profiler, "flann.index_save"),
                "load": _duration_summary(lifecycle_profiler, "flann.index_load"),
                "profiling": lifecycle_profiler.as_report(),
            }
            for result in results:
                if result["parameters"]["trees"] == tree_count:
                    result["lifecycle"] = lifecycle_report
        finally:
            rebuilt.release()
            loaded_index.release()
            built.release()
            try:
                artifact_path.unlink(missing_ok=True)
            except OSError:
                pass
    return results


def _prediction_consistency(
    queries: Sequence[CompactQuery],
    exact_rankings: Sequence[Sequence[RetrievalCandidate]],
    flann_rankings: Sequence[Sequence[RetrievalCandidate]],
    predictions: Sequence[HistoricalPrediction],
    k_values: Sequence[int],
) -> dict[str, object]:
    query_index = {query.crop.relative_path: index for index, query in enumerate(queries)}
    matched = [prediction for prediction in predictions if prediction.decision == "MATCHED"]
    ambiguous = [
        prediction for prediction in predictions if prediction.decision == "AMBIGUOUS"
    ]
    matched_details = []
    exact_counts = Counter()
    flann_counts = Counter()
    for prediction in matched:
        ordinal = query_index[prediction.crop.relative_path]
        expected = prediction.ranked_originals[0]
        exact = exact_rankings[ordinal]
        flann = flann_rankings[ordinal]
        exact_rank = source_rank(exact, expected)
        flann_rank = source_rank(flann, expected)
        presence = {}
        for k in k_values:
            exact_refs = {
                candidate.original for candidate in select_shortlist(exact, k)[0]
            }
            flann_refs = {
                candidate.original for candidate in select_shortlist(flann, k)[0]
            }
            exact_present = expected in exact_refs
            flann_present = expected in flann_refs
            exact_counts[k] += int(exact_present)
            flann_counts[k] += int(flann_present)
            presence[str(k)] = {"exact_bf": exact_present, "flann": flann_present}
        matched_details.append(
            {
                "crop": prediction.crop.relative_path,
                "prior_prediction": expected.relative_path,
                "exact_bf_rank": exact_rank,
                "flann_rank": flann_rank,
                "rank_delta_vs_exact_bf": (
                    flann_rank - exact_rank
                    if exact_rank is not None and flann_rank is not None
                    else None
                ),
                "presence_by_k": presence,
            }
        )
    ambiguous_details = []
    for prediction in ambiguous:
        ordinal = query_index[prediction.crop.relative_path]
        exact = exact_rankings[ordinal]
        flann = flann_rankings[ordinal]
        prior = prediction.ranked_originals[:2]
        exact_50 = {
            candidate.original for candidate in select_shortlist(exact, 50)[0]
        }
        flann_50 = {
            candidate.original for candidate in select_shortlist(flann, 50)[0]
        }
        ambiguous_details.append(
            {
                "crop": prediction.crop.relative_path,
                "prior_ranked_originals": [item.relative_path for item in prior],
                "prior_rank_1_retained_at_k50": {
                    "exact_bf": bool(prior and prior[0] in exact_50),
                    "flann": bool(prior and prior[0] in flann_50),
                },
                "prior_top_2_contained_at_k50": {
                    "exact_bf": len(prior) == 2 and set(prior) <= exact_50,
                    "flann": len(prior) == 2 and set(prior) <= flann_50,
                },
                "rank_changes": [
                    {
                        "original": item.relative_path,
                        "exact_bf_rank": source_rank(exact, item),
                        "flann_rank": source_rank(flann, item),
                    }
                    for item in prior
                ],
            }
        )
    return {
        "terminology": "PREDICTION_CONSISTENCY_NOT_GROUND_TRUTH_ACCURACY",
        "matched_prediction_count": len(matched),
        "ambiguous_prediction_count": len(ambiguous),
        "matched_prediction_consistency": {
            f"prediction-consistency@{k}": {
                "exact_bf": exact_counts[k] / len(matched) if matched else 0.0,
                "flann": flann_counts[k] / len(matched) if matched else 0.0,
            }
            for k in k_values
        },
        "matched_cases": matched_details,
        "ambiguous_cases": ambiguous_details,
    }


def _parse_historical_predictions(path: Path) -> tuple[HistoricalPrediction, ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError("provenance predictions must be readable JSON") from exc
    crops = value.get("crops") if isinstance(value, dict) else None
    if not isinstance(crops, list):
        raise ConfigurationError("provenance predictions must contain crops")
    predictions = []
    crop_references = set()
    for item in crops:
        if not isinstance(item, dict) or not isinstance(item.get("crop"), dict):
            raise ConfigurationError("invalid provenance prediction crop")
        crop_path = item["crop"].get("relative_path")
        decision = item.get("decision")
        ranked = item.get("ranked_candidates")
        if (
            not isinstance(crop_path, str)
            or decision not in {"MATCHED", "AMBIGUOUS", "NO_MATCH"}
            or not isinstance(ranked, list)
        ):
            raise ConfigurationError("invalid provenance prediction crop")
        originals = []
        for candidate in ranked:
            original = candidate.get("original") if isinstance(candidate, dict) else None
            relative_path = original.get("relative_path") if isinstance(original, dict) else None
            if not isinstance(relative_path, str):
                raise ConfigurationError("invalid provenance prediction candidate")
            originals.append(SemanticReference(RootRole.ORIGINAL, relative_path))
        if decision in {"MATCHED", "AMBIGUOUS"} and not originals:
            raise ConfigurationError("prediction lacks ranked original")
        crop = SemanticReference(RootRole.CROPPED, crop_path)
        if crop in crop_references:
            raise ConfigurationError("duplicate provenance prediction crop")
        crop_references.add(crop)
        predictions.append(
            HistoricalPrediction(
                crop,
                decision,
                tuple(originals),
            )
        )
    return tuple(predictions)


def _validate_prediction_artifact_coherence(
    loaded: LoadedIndex,
    queries: Sequence[CompactQuery],
    predictions: Sequence[HistoricalPrediction],
) -> None:
    prediction_crops = [prediction.crop for prediction in predictions]
    if len(set(prediction_crops)) != len(prediction_crops):
        raise ConfigurationError("duplicate provenance prediction crop")

    query_crops = {query.crop for query in queries}
    evaluated = tuple(
        prediction
        for prediction in predictions
        if prediction.decision in {"MATCHED", "AMBIGUOUS"}
    )
    if any(prediction.crop not in query_crops for prediction in evaluated):
        raise ConfigurationError("prediction crops are absent from query corpus")

    indexed_corpus = {original.reference for original in loaded.metadata.originals}
    for prediction in evaluated:
        used_sources = (
            prediction.ranked_originals[:1]
            if prediction.decision == "MATCHED"
            else prediction.ranked_originals[:2]
        )
        if any(source not in indexed_corpus for source in used_sources):
            raise ConfigurationError(
                "prediction sources are absent from candidate-index corpus"
            )


def _base_report(
    *,
    mode: str,
    loaded: LoadedIndex,
    trees: Sequence[int],
    checks: Sequence[int],
    neighbor_depth: int,
    k_values: Sequence[int],
    queries: Sequence[CompactQuery],
    extraction_profiler: RetrievalProfiler,
    exact_profiler: RetrievalProfiler,
    configurations: Sequence[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": IMAGE_BENCHMARK_SCHEMA_VERSION,
        "tool_version": __version__,
        "benchmark_semantics": IMAGE_BENCHMARK_SEMANTICS,
        "mode": mode,
        "status": "RESEARCH_ONLY_NOT_INTEGRATED",
        "oracle": "EXACT_BF_L2",
        "runtime": current_runtime_contract(),
        "configuration": {
            "trees": list(trees),
            "checks": list(checks),
            "neighbor_depth": neighbor_depth,
            "k_values": list(k_values),
            "primary_safety_k": 50,
            "descriptor_rows": loaded.metadata.total_descriptor_rows,
            "query_count": len(queries),
        },
        "distance_semantics": {
            "opencv_flann_raw": "SQUARED_EUCLIDEAN_L2",
            "aggregation_input": "EUCLIDEAN_L2_AFTER_EXPLICIT_SQRT",
        },
        "query_feature_extraction": extraction_profiler.as_report(),
        "exact_bf_reference": {
            "search": _duration_summary(exact_profiler, "query.exact_bf_search"),
            "vote_aggregation_ranking": _duration_summary(
                exact_profiler, "query.vote_aggregation_ranking"
            ),
            "retrieval_excluding_feature_extraction": _duration_summary(
                exact_profiler, "query.exact_retrieval_total"
            ),
            "profiling": exact_profiler.as_report(),
        },
        "evaluated_configurations": list(configurations),
    }


def run_synthetic(arguments: argparse.Namespace) -> dict[str, object]:
    corpus_size = _positive_int(arguments.corpus_size, "corpus size")
    query_count = _positive_int(arguments.queries, "query count")
    original_max = _positive_int(
        arguments.original_max_descriptors, "original descriptor cap"
    )
    query_max = _positive_int(arguments.query_max_descriptors, "query descriptor cap")
    trees = _parse_positive_ints(arguments.trees, "trees")
    checks = _parse_positive_ints(arguments.checks, "checks")
    k_values = _parse_positive_ints(arguments.k_values, "K")
    _require_standard_k_values(k_values)
    neighbor_depth = _positive_int(arguments.neighbor_depth, "neighbor depth")
    strata_values = _parse_strata(arguments.strata)
    rng = np.random.default_rng(arguments.seed)
    with tempfile.TemporaryDirectory(prefix="autocrop-flann-synthetic-") as temporary:
        base = Path(temporary)
        originals = base / "originals"
        crops = base / "crops"
        artifacts = base / "artifacts"
        originals.mkdir()
        crops.mkdir()
        artifacts.mkdir()
        sources = []
        for index in range(corpus_size):
            size = (360 + 20 * (index % 3), 280 + 20 * (index % 4))
            image = _structured_image(arguments.seed + index * 101, size)
            image.save(originals / f"source-{index:05d}.png")
            sources.append(image)
        known_sources: dict[str, SemanticReference] = {}
        query_strata: dict[str, str] = {}
        for ordinal in range(query_count):
            source_index = ordinal % corpus_size
            stratum = strata_values[ordinal % len(strata_values)]
            suffix = ".jpg" if stratum in {"jpeg", "exif"} else ".png"
            filename = f"query-{ordinal:05d}-{stratum}{suffix}"
            _save_query(sources[source_index], crops / filename, stratum, rng)
            known_sources[filename] = SemanticReference(
                RootRole.ORIGINAL, f"source-{source_index:05d}.png"
            )
            query_strata[filename] = stratum
        index_path = base / "exact-index.private.json"
        build_paths = validate_build_paths(originals, index_path)
        build_index(
            build_paths,
            IndexParameters(
                original_max_descriptors=original_max,
                random_seed=arguments.seed,
            ),
        )
        paths = validate_query_paths(
            index_path, crops, base / "unused-retrieval.private.json"
        )
        loaded = load_index(paths)
        try:
            extraction_profiler = RetrievalProfiler()
            queries = _load_compact_queries(
                paths,
                loaded,
                query_max_descriptors=query_max,
                profiler=extraction_profiler,
            )
            exact_rankings, exact_profiler = _exact_rankings(
                loaded, queries, neighbor_depth=neighbor_depth
            )
            corpus = DescriptorCorpusIdentity(
                loaded.metadata.corpus_identity_sha256,
                loaded.metadata.binary_sha256,
                loaded.metadata.total_descriptor_rows,
            )
            configurations = _run_grid(
                loaded,
                queries,
                exact_rankings=exact_rankings,
                corpus=corpus,
                trees=trees,
                checks=checks,
                neighbor_depth=neighbor_depth,
                k_values=k_values,
                artifact_directory=artifacts,
                known_sources=known_sources,
                strata=query_strata,
            )
            report = _base_report(
                mode="SYNTHETIC_KNOWN_SOURCE",
                loaded=loaded,
                trees=trees,
                checks=checks,
                neighbor_depth=neighbor_depth,
                k_values=k_values,
                queries=queries,
                extraction_profiler=extraction_profiler,
                exact_profiler=exact_profiler,
                configurations=configurations,
            )
            report["synthetic"] = {
                "corpus_size": corpus_size,
                "query_count": query_count,
                "seed": arguments.seed,
                "strata": list(strata_values),
                "metric_terminology": "GENUINE_KNOWN_SOURCE_RECALL",
                "exact_bf_known_source_recall": {
                    f"Recall@{k}": value
                    for k, value in recall_at_k(
                        (
                            (known_sources[query.crop.relative_path], ranking)
                            for query, ranking in zip(
                                queries, exact_rankings, strict=True
                            )
                        ),
                        k_values,
                    ).items()
                },
            }
            return report
        finally:
            if isinstance(loaded.descriptors, np.memmap):
                loaded.descriptors._mmap.close()


def run_bounded_real(
    arguments: argparse.Namespace, *, output: Path | None = None
) -> dict[str, object]:
    trees = _parse_positive_ints(arguments.trees, "trees")
    checks = _parse_positive_ints(arguments.checks, "checks")
    k_values = _parse_positive_ints(arguments.k_values, "K")
    _require_standard_k_values(k_values)
    neighbor_depth = _positive_int(arguments.neighbor_depth, "neighbor depth")
    query_max = _positive_int(arguments.query_max_descriptors, "query descriptor cap")
    bounded_output = output or _validate_output(arguments.output, private=True)
    assert bounded_output is not None
    paths, predictions_path = _validate_bounded_real_paths(arguments, bounded_output)
    predictions = _parse_historical_predictions(predictions_path)
    with tempfile.TemporaryDirectory(prefix="autocrop-flann-bounded-") as temporary:
        base = Path(temporary)
        loaded = load_index(paths)
        try:
            extraction_profiler = RetrievalProfiler()
            queries = _load_compact_queries(
                paths,
                loaded,
                query_max_descriptors=query_max,
                profiler=extraction_profiler,
            )
            _validate_prediction_artifact_coherence(loaded, queries, predictions)
            exact_rankings, exact_profiler = _exact_rankings(
                loaded, queries, neighbor_depth=neighbor_depth
            )
            corpus = DescriptorCorpusIdentity(
                loaded.metadata.corpus_identity_sha256,
                loaded.metadata.binary_sha256,
                loaded.metadata.total_descriptor_rows,
            )
            artifacts = base / "artifacts"
            artifacts.mkdir()
            configurations = _run_grid(
                loaded,
                queries,
                exact_rankings=exact_rankings,
                corpus=corpus,
                trees=trees,
                checks=checks,
                neighbor_depth=neighbor_depth,
                k_values=k_values,
                artifact_directory=artifacts,
                predictions=predictions,
            )
            report = _base_report(
                mode="BOUNDED_REAL_PREDICTION_CONSISTENCY",
                loaded=loaded,
                trees=trees,
                checks=checks,
                neighbor_depth=neighbor_depth,
                k_values=k_values,
                queries=queries,
                extraction_profiler=extraction_profiler,
                exact_profiler=exact_profiler,
                configurations=configurations,
            )
            report["bounded_real"] = {
                "terminology": "PREDICTION_CONSISTENCY_NOT_ACCURACY_OR_GROUND_TRUTH_RECALL",
                "prediction_report_sha256": hashlib.sha256(
                    predictions_path.read_bytes()
                ).hexdigest(),
                "prediction_decisions": dict(
                    sorted(Counter(item.decision for item in predictions).items())
                ),
            }
            return report
        finally:
            if isinstance(loaded.descriptors, np.memmap):
                loaded.descriptors._mmap.close()


def _print_bounded_real_summary(report: dict[str, object], stream: TextIO) -> None:
    configuration = report["configuration"]
    bounded = report["bounded_real"]
    assert isinstance(configuration, dict)
    assert isinstance(bounded, dict)
    decisions = bounded["prediction_decisions"]
    assert isinstance(decisions, dict)
    print(
        "FLANN_BOUNDED_REAL: "
        f"queries={configuration['query_count']} "
        f"configurations={len(report['evaluated_configurations'])} "
        f"predictions={sum(int(count) for count in decisions.values())} "
        "report=PRIVATE_OUTPUT_WRITTEN",
        file=stream,
    )


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
        if arguments.command == "synthetic":
            output = _validate_output(arguments.output)
            report = run_synthetic(arguments)
        else:
            output = _validate_output(arguments.output, private=True)
            report = run_bounded_real(arguments, output=output)
        if output is not None:
            write_manifest_atomic(output, report)
        if arguments.command == "synthetic":
            print(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                ),
                file=out,
            )
        else:
            _print_bounded_real_summary(report, out)
        return 0
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
        print(f"benchmark failure: {type(exc).__name__}", file=err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

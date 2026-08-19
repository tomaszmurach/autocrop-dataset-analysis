"""CLI orchestration for private compact-SIFT retrieval indexes and queries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from typing import BinaryIO, Sequence, TextIO

import numpy as np

from .audit import AuditItem, ReadStatus, RootRole, audit_root
from .candidate_retrieval import (
    DESCRIPTOR_DIMENSION,
    DESCRIPTOR_DTYPE,
    IndexMetadata,
    IndexParameters,
    IndexStatus,
    IndexValidationError,
    OriginalIndexRecord,
    QueryParameters,
    RetrievalQueryResult,
    build_index_manifest,
    build_retrieval_manifest,
    make_query_result,
    parse_json_bytes,
    select_spatially_balanced_descriptors,
    validate_index_manifest,
)
from .candidate_retrieval_profiling import RetrievalProfiler, profiling_stage
from .cli import ConfigurationError, OutputFailure, write_manifest_atomic
from .content_matching import (
    MatchingParameters,
    configure_opencv,
    ensure_sift_available,
    extract_features,
    unavailable_feature_image,
)


PRIVATE_JSON_SUFFIX = ".private.json"
PRIVATE_BINARY_SUFFIX = ".descriptors.private.f32"
HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class BuildPaths:
    originals: Path
    manifest: Path
    binary: Path
    profile_output: Path | None = None


@dataclass(frozen=True, slots=True)
class QueryPaths:
    index_manifest: Path
    index_binary: Path
    cropped: Path
    output: Path
    profile_output: Path | None = None


@dataclass(frozen=True, slots=True)
class LoadedIndex:
    raw_manifest_bytes: bytes
    metadata: IndexMetadata
    descriptors: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autocrop_analysis.candidate_retrieval_cli",
        description="Build or query an experimental private compact-SIFT retrieval index.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="Build a private original-corpus index.")
    build.add_argument("--originals", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--original-max-descriptors", type=int, default=128)
    build.add_argument("--profile-output", type=Path)

    query = commands.add_parser("query", help="Query a private index with crop images.")
    query.add_argument("--index", required=True, type=Path)
    query.add_argument("--cropped", required=True, type=Path)
    query.add_argument("--output", required=True, type=Path)
    query.add_argument("--k", type=int, default=50)
    query.add_argument("--query-max-descriptors", type=int, default=64)
    query.add_argument("--neighbor-depth", type=int, default=32)
    query.add_argument("--profile-output", type=Path)
    return parser


def descriptor_path_for_manifest(manifest: Path) -> Path:
    if not manifest.name.endswith(PRIVATE_JSON_SUFFIX):
        raise ConfigurationError("index filename must end with .private.json")
    stem = manifest.name[: -len(PRIVATE_JSON_SUFFIX)]
    if not stem:
        raise ConfigurationError("index filename must have a nonempty prefix")
    return manifest.with_name(stem + PRIVATE_BINARY_SUFFIX)


def validate_build_paths(
    originals: Path, output: Path, profile_output: Path | None = None
) -> BuildPaths:
    originals_path = _validate_root(originals, "originals")
    manifest = _validate_new_private_json(output, "output")
    binary = descriptor_path_for_manifest(manifest)
    if binary.exists():
        raise ConfigurationError("descriptor output must not already exist")
    if _contains(originals_path, manifest) or _contains(originals_path, binary):
        raise ConfigurationError("index outputs must be outside the originals root")
    profiling = (
        _validate_new_private_json(profile_output, "profile output")
        if profile_output is not None
        else None
    )
    if profiling is not None:
        if profiling == manifest:
            raise ConfigurationError("profile output must differ from index manifest")
        if _contains(originals_path, profiling):
            raise ConfigurationError("profile output must be outside the originals root")
    return BuildPaths(originals_path, manifest, binary, profiling)


def validate_query_paths(
    index: Path,
    cropped: Path,
    output: Path,
    profile_output: Path | None = None,
) -> QueryPaths:
    if not index.name.endswith(PRIVATE_JSON_SUFFIX):
        raise ConfigurationError("index filename must end with .private.json")
    try:
        index_manifest = index.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigurationError("index manifest must exist") from exc
    if not index_manifest.is_file() or index_manifest.is_symlink():
        raise ConfigurationError("index manifest must be a regular file")
    binary = descriptor_path_for_manifest(index_manifest)
    cropped_path = _validate_root(cropped, "cropped")
    output_path = _validate_new_private_json(output, "output")
    if output_path == index_manifest:
        raise ConfigurationError("query output must differ from index manifest")
    if _contains(cropped_path, output_path):
        raise ConfigurationError("query output must be outside the cropped root")
    profiling = (
        _validate_new_private_json(profile_output, "profile output")
        if profile_output is not None
        else None
    )
    if profiling is not None:
        if profiling in {index_manifest, output_path}:
            raise ConfigurationError("profile output must differ from query inputs and output")
        if _contains(cropped_path, profiling):
            raise ConfigurationError("profile output must be outside the cropped root")
    return QueryPaths(index_manifest, binary, cropped_path, output_path, profiling)


def _validate_root(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigurationError(f"{label} root must exist") from exc
    if not resolved.is_dir():
        raise ConfigurationError(f"{label} root must be a directory")
    try:
        with os.scandir(resolved):
            pass
    except OSError as exc:
        raise ConfigurationError(f"{label} root must be readable") from exc
    return resolved


def _validate_new_private_json(path: Path, label: str) -> Path:
    if not path.name.endswith(PRIVATE_JSON_SUFFIX):
        raise ConfigurationError(f"{label} filename must end with .private.json")
    if path.exists():
        raise ConfigurationError(f"{label} must not already exist")
    try:
        parent = path.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigurationError(f"{label} parent must exist") from exc
    if not parent.is_dir():
        raise ConfigurationError(f"{label} parent must be a directory")
    return parent / path.name


def _contains(parent: Path, child: Path) -> bool:
    return child == parent or child.is_relative_to(parent)


def hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while True:
            chunk = source.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _write_descriptor_block(
    output: BinaryIO, digest: "hashlib._Hash", descriptors: np.ndarray
) -> int:
    block = np.ascontiguousarray(descriptors, dtype=np.dtype(DESCRIPTOR_DTYPE))
    payload = block.tobytes(order="C")
    output.write(payload)
    digest.update(payload)
    return len(payload)


def _scan_issue_records(audit_result) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "root_role": issue.root_role.value,
            "relative_path": issue.relative_path,
            "category": issue.category.value,
        }
        for issue in audit_result.scan_issues
    )


def build_index(
    paths: BuildPaths,
    parameters: IndexParameters,
    *,
    profiler: RetrievalProfiler | None = None,
) -> dict[str, object]:
    if profiler is not None:
        profiler.snapshot_memory("before_build")
    with profiling_stage(profiler, "build.total") as total_timing:
        ensure_sift_available()
        matching_parameters = MatchingParameters(
            sift_nfeatures=parameters.sift_nfeatures,
            random_seed=parameters.random_seed,
        )
        configure_opencv(matching_parameters)
        with profiling_stage(profiler, "build.corpus_audit") as audit_timing:
            audit_result = audit_root(paths.originals, RootRole.ORIGINAL)
            candidates = tuple(item for item in audit_result.items if item.is_image_candidate)
            if profiler is not None:
                audit_timing.add_work(
                    audited_regular_files=len(audit_result.items),
                    original_image_candidates=len(candidates),
                    scan_issues=len(audit_result.scan_issues),
                )
        if profiler is not None:
            profiler.snapshot_memory("after_build_audit")
        records: list[OriginalIndexRecord] = []
        descriptor_offset = 0
        temporary_path: Path | None = None
        binary_created = False
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{paths.binary.name}.",
                suffix=".tmp",
                dir=paths.binary.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                binary_digest = hashlib.sha256()
                binary_size = 0
                for item_ordinal, item in enumerate(candidates):
                    record, compact = _build_original_record(
                        item,
                        paths.originals,
                        matching_parameters,
                        parameters,
                        descriptor_offset,
                        profiler=profiler,
                        item_ordinal=item_ordinal,
                    )
                    if compact.shape[0]:
                        with profiling_stage(
                            profiler,
                            "build.descriptor_write_and_hash",
                            item_ordinal=item_ordinal,
                        ) as write_timing:
                            written = _write_descriptor_block(
                                temporary, binary_digest, compact
                            )
                            binary_size += written
                            descriptor_offset += int(compact.shape[0])
                            if profiler is not None:
                                write_timing.add_work(
                                    selected_descriptor_rows=int(compact.shape[0]),
                                    descriptor_bytes=written,
                                )
                    records.append(record)
                with profiling_stage(profiler, "build.binary_flush_fsync"):
                    temporary.flush()
                    os.fsync(temporary.fileno())

            with profiling_stage(
                profiler, "build.manifest_and_corpus_identity"
            ) as manifest_timing:
                manifest = build_index_manifest(
                    parameters=parameters,
                    binary_filename=paths.binary.name,
                    binary_sha256=binary_digest.hexdigest(),
                    binary_byte_size=binary_size,
                    originals=records,
                    scan_issues=_scan_issue_records(audit_result),
                )
                if profiler is not None:
                    manifest_timing.add_work(
                        original_records=len(records),
                        descriptor_rows=descriptor_offset,
                        descriptor_bytes=binary_size,
                    )
            with profiling_stage(profiler, "build.publication"):
                if paths.binary.exists() or paths.manifest.exists():
                    raise FileExistsError
                os.link(temporary_path, paths.binary)
                binary_created = True
                try:
                    temporary_path.unlink()
                    temporary_path = None
                except OSError:
                    pass
                try:
                    write_manifest_atomic(paths.manifest, manifest)
                except Exception:
                    if binary_created:
                        try:
                            paths.binary.unlink()
                        except OSError:
                            pass
                    raise
            if profiler is not None:
                indexed = sum(record.status is IndexStatus.INDEXED for record in records)
                no_descriptors = sum(
                    record.status is IndexStatus.NO_DESCRIPTORS for record in records
                )
                unavailable = sum(
                    record.status is IndexStatus.UNAVAILABLE for record in records
                )
                total_timing.add_work(
                    original_image_candidates=len(candidates),
                    indexed_originals=indexed,
                    no_descriptor_originals=no_descriptors,
                    unavailable_originals=unavailable,
                    selected_descriptor_rows=descriptor_offset,
                    descriptor_bytes=binary_size,
                )
                profiler.snapshot_memory("after_build")
            return manifest
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass


def _build_original_record(
    item: AuditItem,
    root: Path,
    matching_parameters: MatchingParameters,
    parameters: IndexParameters,
    descriptor_offset: int,
    *,
    profiler: RetrievalProfiler | None = None,
    item_ordinal: int | None = None,
) -> tuple[OriginalIndexRecord, np.ndarray]:
    source_path = root / Path(item.relative_path)
    with profiling_stage(
        profiler, "build.first_encoded_hash", item_ordinal=item_ordinal
    ) as first_hash_timing:
        try:
            encoded_sha, hashed_size = hash_file(source_path)
        except OSError:
            encoded_sha = None
            hashed_size = item.size_bytes
        if profiler is not None and hashed_size is not None:
            first_hash_timing.add_work(encoded_bytes=hashed_size)
    if item.read_status is not ReadStatus.READABLE or encoded_sha is None:
        return (
            OriginalIndexRecord(
                item.reference,
                item.display_width,
                item.display_height,
                hashed_size,
                encoded_sha,
                IndexStatus.UNAVAILABLE,
                0,
                descriptor_offset,
                0,
            ),
            np.empty((0, DESCRIPTOR_DIMENSION), dtype=np.dtype(DESCRIPTOR_DTYPE)),
        )

    with profiling_stage(
        profiler, "build.feature_extraction", item_ordinal=item_ordinal
    ) as extraction_timing:
        feature = extract_features(
            source_path,
            item.reference,
            matching_parameters,
            retain_grayscale=False,
        )
        if profiler is not None:
            extraction_timing.add_work(
                display_pixels=max(0, feature.width * feature.height),
                extracted_descriptor_rows=len(feature.keypoints),
            )
    with profiling_stage(
        profiler, "build.second_stability_hash", item_ordinal=item_ordinal
    ) as second_hash_timing:
        try:
            verified_sha, verified_size = hash_file(source_path)
        except OSError:
            verified_sha, verified_size = None, hashed_size
        if profiler is not None and verified_size is not None:
            second_hash_timing.add_work(encoded_bytes=verified_size)
    if verified_sha != encoded_sha or verified_size != hashed_size:
        return (
            OriginalIndexRecord(
                item.reference,
                feature.width if feature.width > 0 else item.display_width,
                feature.height if feature.height > 0 else item.display_height,
                verified_size,
                verified_sha,
                IndexStatus.UNAVAILABLE,
                0,
                descriptor_offset,
                0,
            ),
            np.empty((0, DESCRIPTOR_DIMENSION), dtype=np.dtype(DESCRIPTOR_DTYPE)),
        )
    with profiling_stage(
        profiler, "build.compact_descriptor_selection", item_ordinal=item_ordinal
    ) as selection_timing:
        compact = select_spatially_balanced_descriptors(
            feature,
            maximum=parameters.original_max_descriptors,
            grid_rows=parameters.grid_rows,
            grid_columns=parameters.grid_columns,
        )
        if profiler is not None:
            selection_timing.add_work(
                extracted_descriptor_rows=len(feature.keypoints),
                selected_descriptor_rows=int(compact.shape[0]),
            )
    if compact.shape[0] == 0:
        status = (
            IndexStatus.NO_DESCRIPTORS
            if feature.diagnostic_reason == "NO_DESCRIPTORS"
            else IndexStatus.UNAVAILABLE
        )
    else:
        status = IndexStatus.INDEXED
    count = int(compact.shape[0]) if status is IndexStatus.INDEXED else 0
    if count == 0 and compact.shape[0]:
        compact = np.empty((0, DESCRIPTOR_DIMENSION), dtype=np.dtype(DESCRIPTOR_DTYPE))
    return (
        OriginalIndexRecord(
            item.reference,
            feature.width if feature.width > 0 else item.display_width,
            feature.height if feature.height > 0 else item.display_height,
            hashed_size,
            encoded_sha,
            status,
            count,
            descriptor_offset,
            count,
        ),
        compact,
    )


def load_index(
    paths: QueryPaths,
    *,
    profiler: RetrievalProfiler | None = None,
    load_context: str | None = None,
) -> LoadedIndex:
    if profiler is not None:
        profiler.snapshot_memory("before_index_load")
    with profiling_stage(
        profiler, "load.total", phase=load_context
    ) as total_timing:
        with profiling_stage(profiler, "load.manifest_read") as read_timing:
            try:
                raw_manifest = paths.index_manifest.read_bytes()
            except OSError as exc:
                raise IndexValidationError("INDEX_MANIFEST_READ_FAILED") from exc
            if profiler is not None:
                read_timing.add_work(manifest_bytes=len(raw_manifest))
        with profiling_stage(profiler, "load.manifest_parse_validation"):
            parsed = parse_json_bytes(raw_manifest)
            metadata = validate_index_manifest(
                parsed, expected_binary_filename=paths.index_binary.name
            )
        if (
            not paths.index_binary.exists()
            or not paths.index_binary.is_file()
            or paths.index_binary.is_symlink()
        ):
            raise IndexValidationError("BINARY_MISSING")
        with profiling_stage(
            profiler, "load.descriptor_binary_integrity_hash"
        ) as hash_timing:
            try:
                binary_sha, binary_size = hash_file(paths.index_binary)
            except OSError as exc:
                raise IndexValidationError("BINARY_READ_FAILED") from exc
            if profiler is not None:
                hash_timing.add_work(descriptor_binary_bytes=binary_size)
        if binary_size != metadata.binary_byte_size:
            raise IndexValidationError("BINARY_SIZE_MISMATCH")
        if binary_sha != metadata.binary_sha256:
            raise IndexValidationError("BINARY_HASH_MISMATCH")
        with profiling_stage(profiler, "load.memmap_creation"):
            if metadata.total_descriptor_rows == 0:
                descriptors = np.empty(
                    (0, DESCRIPTOR_DIMENSION), dtype=np.dtype(DESCRIPTOR_DTYPE)
                )
            else:
                try:
                    descriptors = np.memmap(
                        paths.index_binary,
                        mode="r",
                        dtype=np.dtype(DESCRIPTOR_DTYPE),
                        shape=(metadata.total_descriptor_rows, DESCRIPTOR_DIMENSION),
                    )
                except Exception as exc:
                    raise IndexValidationError("BINARY_SHAPE_MISMATCH") from exc
        if profiler is not None:
            total_timing.add_work(
                original_records=len(metadata.originals),
                indexed_descriptor_rows=metadata.total_descriptor_rows,
                descriptor_binary_bytes=metadata.binary_byte_size,
            )
            profiler.snapshot_memory("after_index_load")
        return LoadedIndex(raw_manifest, metadata, descriptors)


def query_index(
    paths: QueryPaths,
    loaded: LoadedIndex,
    parameters: QueryParameters,
    *,
    profiler: RetrievalProfiler | None = None,
) -> tuple[dict[str, object], tuple[RetrievalQueryResult, ...]]:
    if profiler is not None:
        profiler.snapshot_memory("before_query_batch")
    with profiling_stage(profiler, "query.batch_processing_total") as batch_timing:
        matching_parameters = MatchingParameters(
            sift_nfeatures=loaded.metadata.parameters.sift_nfeatures,
            random_seed=loaded.metadata.parameters.random_seed,
        )
        configure_opencv(matching_parameters)
        with profiling_stage(profiler, "query.crop_root_audit") as audit_timing:
            audit_result = audit_root(paths.cropped, RootRole.CROPPED)
            candidates = tuple(
                item for item in audit_result.items if item.is_image_candidate
            )
            if profiler is not None:
                audit_timing.add_work(
                    audited_regular_files=len(audit_result.items),
                    query_image_candidates=len(candidates),
                    scan_issues=len(audit_result.scan_issues),
                )
        results: list[RetrievalQueryResult] = []
        for item_ordinal, item in enumerate(candidates):
            phase = "first_query" if item_ordinal == 0 else "subsequent_query"
            with profiling_stage(
                profiler,
                "query.total",
                item_ordinal=item_ordinal,
                phase=phase,
            ) as query_timing:
                with profiling_stage(
                    profiler, "query.feature_extraction", item_ordinal=item_ordinal
                ) as extraction_timing:
                    if item.read_status is ReadStatus.READABLE:
                        feature = extract_features(
                            paths.cropped / Path(item.relative_path),
                            item.reference,
                            matching_parameters,
                            retain_grayscale=False,
                        )
                    else:
                        feature = unavailable_feature_image(
                            item.reference, "AUDIT_UNAVAILABLE"
                        )
                    if profiler is not None:
                        extraction_timing.add_work(
                            display_pixels=max(0, feature.width * feature.height),
                            extracted_descriptor_rows=len(feature.keypoints),
                        )
                with profiling_stage(
                    profiler,
                    "query.compact_descriptor_selection",
                    item_ordinal=item_ordinal,
                ) as selection_timing:
                    compact = select_spatially_balanced_descriptors(
                        feature,
                        maximum=parameters.query_max_descriptors,
                        grid_rows=loaded.metadata.parameters.grid_rows,
                        grid_columns=loaded.metadata.parameters.grid_columns,
                    )
                    if profiler is not None:
                        selection_timing.add_work(
                            extracted_descriptor_rows=len(feature.keypoints),
                            selected_query_descriptor_rows=int(compact.shape[0]),
                        )
                result = make_query_result(
                    crop=feature,
                    compact_descriptors=compact,
                    metadata=loaded.metadata,
                    descriptor_matrix=loaded.descriptors,
                    parameters=parameters,
                    profiler=profiler,
                    item_ordinal=item_ordinal,
                )
                results.append(result)
                if profiler is not None:
                    query_timing.add_work(
                        selected_query_descriptor_rows=int(compact.shape[0]),
                        indexed_descriptor_rows=loaded.metadata.total_descriptor_rows,
                        returned_candidates=result.returned_candidate_count,
                    )
        stable_results = tuple(results)
        with profiling_stage(
            profiler, "query.retrieval_manifest_construction"
        ) as manifest_timing:
            manifest = build_retrieval_manifest(
                index_manifest_sha256=hashlib.sha256(
                    loaded.raw_manifest_bytes
                ).hexdigest(),
                metadata=loaded.metadata,
                query_parameters=parameters,
                queries=stable_results,
                query_scan_issue_count=len(audit_result.scan_issues),
            )
            if profiler is not None:
                manifest_timing.add_work(queries=len(stable_results))
        if profiler is not None:
            batch_timing.add_work(
                queries=len(stable_results),
                selected_query_descriptor_rows=sum(
                    result.selected_descriptor_count for result in stable_results
                ),
                indexed_descriptor_rows=loaded.metadata.total_descriptor_rows,
            )
            profiler.snapshot_memory("after_query_batch")
        return manifest, stable_results


def _print_build_summary(manifest: dict[str, object], stream: TextIO) -> None:
    summary = manifest["summary"]
    assert isinstance(summary, dict)
    print(
        "CANDIDATE_INDEX: "
        f"originals={summary['supplied_original_image_candidates']} "
        f"indexed={summary['INDEXED']} "
        f"descriptors={summary['total_descriptor_rows']} "
        f"index_corpus_complete={str(summary['index_corpus_complete']).lower()}",
        file=stream,
    )


def _print_query_summary(manifest: dict[str, object], stream: TextIO) -> None:
    summary = manifest["summary"]
    assert isinstance(summary, dict)
    print(
        "CANDIDATE_RETRIEVAL: "
        f"queries={summary['queries']} "
        f"retrieved={summary['RETRIEVED']} "
        f"query_failures={summary['NO_QUERY_DESCRIPTORS'] + summary['QUERY_UNAVAILABLE']} "
        f"index_incomplete={summary['INDEX_INCOMPLETE']}",
        file=stream,
    )


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
        if arguments.command == "build":
            paths = validate_build_paths(
                arguments.originals,
                arguments.output,
                arguments.profile_output,
            )
            parameters = IndexParameters(
                original_max_descriptors=arguments.original_max_descriptors
            )
            profiler = RetrievalProfiler() if paths.profile_output is not None else None
            manifest = build_index(paths, parameters, profiler=profiler)
            if profiler is not None and paths.profile_output is not None:
                write_manifest_atomic(paths.profile_output, profiler.as_report())
            _print_build_summary(manifest, output_stream)
            return 0

        paths = validate_query_paths(
            arguments.index,
            arguments.cropped,
            arguments.output,
            arguments.profile_output,
        )
        query_parameters = QueryParameters(
            query_max_descriptors=arguments.query_max_descriptors,
            neighbor_depth=arguments.neighbor_depth,
            requested_k=arguments.k,
        )
        profiler = RetrievalProfiler() if paths.profile_output is not None else None
        ensure_sift_available()
        loaded = load_index(
            paths,
            profiler=profiler,
            load_context="process_fresh_load" if profiler is not None else None,
        )
        with profiling_stage(
            profiler, "query.shared_index_batch_total"
        ) as batch_timing:
            manifest, results = query_index(
                paths, loaded, query_parameters, profiler=profiler
            )
            with profiling_stage(profiler, "query.output_publication") as output_timing:
                write_manifest_atomic(paths.output, manifest)
                if profiler is not None:
                    output_timing.add_work(queries=len(results))
            if profiler is not None:
                batch_timing.add_work(queries=len(results))
        if profiler is not None and paths.profile_output is not None:
            profiler.snapshot_memory("after_query_output")
            write_manifest_atomic(paths.profile_output, profiler.as_report())
        _print_query_summary(manifest, output_stream)
        return 0
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=error_stream)
        return 2
    except IndexValidationError as exc:
        print(f"index input error: {exc.code}", file=error_stream)
        return 4
    except OutputFailure as exc:
        print(f"result output failure: {exc.error_type}", file=error_stream)
        return 3
    except RuntimeError as exc:
        if str(exc) == "SIFT_UNAVAILABLE":
            print("runtime dependency error: SIFT_UNAVAILABLE", file=error_stream)
            return 1
        print(f"unexpected internal failure: {type(exc).__name__}", file=error_stream)
        return 1
    except Exception as exc:
        print(f"unexpected internal failure: {type(exc).__name__}", file=error_stream)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

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


@dataclass(frozen=True, slots=True)
class QueryPaths:
    index_manifest: Path
    index_binary: Path
    cropped: Path
    output: Path


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

    query = commands.add_parser("query", help="Query a private index with crop images.")
    query.add_argument("--index", required=True, type=Path)
    query.add_argument("--cropped", required=True, type=Path)
    query.add_argument("--output", required=True, type=Path)
    query.add_argument("--k", type=int, default=50)
    query.add_argument("--query-max-descriptors", type=int, default=64)
    query.add_argument("--neighbor-depth", type=int, default=32)
    return parser


def descriptor_path_for_manifest(manifest: Path) -> Path:
    if not manifest.name.endswith(PRIVATE_JSON_SUFFIX):
        raise ConfigurationError("index filename must end with .private.json")
    stem = manifest.name[: -len(PRIVATE_JSON_SUFFIX)]
    if not stem:
        raise ConfigurationError("index filename must have a nonempty prefix")
    return manifest.with_name(stem + PRIVATE_BINARY_SUFFIX)


def validate_build_paths(originals: Path, output: Path) -> BuildPaths:
    originals_path = _validate_root(originals, "originals")
    manifest = _validate_new_private_json(output, "output")
    binary = descriptor_path_for_manifest(manifest)
    if binary.exists():
        raise ConfigurationError("descriptor output must not already exist")
    if _contains(originals_path, manifest) or _contains(originals_path, binary):
        raise ConfigurationError("index outputs must be outside the originals root")
    return BuildPaths(originals_path, manifest, binary)


def validate_query_paths(index: Path, cropped: Path, output: Path) -> QueryPaths:
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
    return QueryPaths(index_manifest, binary, cropped_path, output_path)


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


def build_index(paths: BuildPaths, parameters: IndexParameters) -> dict[str, object]:
    ensure_sift_available()
    matching_parameters = MatchingParameters(
        sift_nfeatures=parameters.sift_nfeatures,
        random_seed=parameters.random_seed,
    )
    configure_opencv(matching_parameters)
    audit_result = audit_root(paths.originals, RootRole.ORIGINAL)
    candidates = tuple(item for item in audit_result.items if item.is_image_candidate)
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
            for item in candidates:
                record, compact = _build_original_record(
                    item,
                    paths.originals,
                    matching_parameters,
                    parameters,
                    descriptor_offset,
                )
                if compact.shape[0]:
                    binary_size += _write_descriptor_block(temporary, binary_digest, compact)
                    descriptor_offset += int(compact.shape[0])
                records.append(record)
            temporary.flush()
            os.fsync(temporary.fileno())

        manifest = build_index_manifest(
            parameters=parameters,
            binary_filename=paths.binary.name,
            binary_sha256=binary_digest.hexdigest(),
            binary_byte_size=binary_size,
            originals=records,
            scan_issues=_scan_issue_records(audit_result),
        )
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
) -> tuple[OriginalIndexRecord, np.ndarray]:
    source_path = root / Path(item.relative_path)
    try:
        encoded_sha, hashed_size = hash_file(source_path)
    except OSError:
        encoded_sha = None
        hashed_size = item.size_bytes
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

    feature = extract_features(
        source_path,
        item.reference,
        matching_parameters,
        retain_grayscale=False,
    )
    try:
        verified_sha, verified_size = hash_file(source_path)
    except OSError:
        verified_sha, verified_size = None, hashed_size
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
    compact = select_spatially_balanced_descriptors(
        feature,
        maximum=parameters.original_max_descriptors,
        grid_rows=parameters.grid_rows,
        grid_columns=parameters.grid_columns,
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


def load_index(paths: QueryPaths) -> LoadedIndex:
    try:
        raw_manifest = paths.index_manifest.read_bytes()
    except OSError as exc:
        raise IndexValidationError("INDEX_MANIFEST_READ_FAILED") from exc
    parsed = parse_json_bytes(raw_manifest)
    metadata = validate_index_manifest(
        parsed, expected_binary_filename=paths.index_binary.name
    )
    if not paths.index_binary.exists() or not paths.index_binary.is_file() or paths.index_binary.is_symlink():
        raise IndexValidationError("BINARY_MISSING")
    try:
        binary_sha, binary_size = hash_file(paths.index_binary)
    except OSError as exc:
        raise IndexValidationError("BINARY_READ_FAILED") from exc
    if binary_size != metadata.binary_byte_size:
        raise IndexValidationError("BINARY_SIZE_MISMATCH")
    if binary_sha != metadata.binary_sha256:
        raise IndexValidationError("BINARY_HASH_MISMATCH")
    if metadata.total_descriptor_rows == 0:
        descriptors = np.empty((0, DESCRIPTOR_DIMENSION), dtype=np.dtype(DESCRIPTOR_DTYPE))
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
    return LoadedIndex(raw_manifest, metadata, descriptors)


def query_index(
    paths: QueryPaths,
    loaded: LoadedIndex,
    parameters: QueryParameters,
) -> tuple[dict[str, object], tuple[RetrievalQueryResult, ...]]:
    matching_parameters = MatchingParameters(
        sift_nfeatures=loaded.metadata.parameters.sift_nfeatures,
        random_seed=loaded.metadata.parameters.random_seed,
    )
    configure_opencv(matching_parameters)
    audit_result = audit_root(paths.cropped, RootRole.CROPPED)
    candidates = tuple(item for item in audit_result.items if item.is_image_candidate)
    results: list[RetrievalQueryResult] = []
    for item in candidates:
        if item.read_status is ReadStatus.READABLE:
            feature = extract_features(
                paths.cropped / Path(item.relative_path),
                item.reference,
                matching_parameters,
                retain_grayscale=False,
            )
        else:
            feature = unavailable_feature_image(item.reference, "AUDIT_UNAVAILABLE")
        compact = select_spatially_balanced_descriptors(
            feature,
            maximum=parameters.query_max_descriptors,
            grid_rows=loaded.metadata.parameters.grid_rows,
            grid_columns=loaded.metadata.parameters.grid_columns,
        )
        results.append(
            make_query_result(
                crop=feature,
                compact_descriptors=compact,
                metadata=loaded.metadata,
                descriptor_matrix=loaded.descriptors,
                parameters=parameters,
            )
        )
    stable_results = tuple(results)
    manifest = build_retrieval_manifest(
        index_manifest_sha256=hashlib.sha256(loaded.raw_manifest_bytes).hexdigest(),
        metadata=loaded.metadata,
        query_parameters=parameters,
        queries=stable_results,
        query_scan_issue_count=len(audit_result.scan_issues),
    )
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
            paths = validate_build_paths(arguments.originals, arguments.output)
            parameters = IndexParameters(
                original_max_descriptors=arguments.original_max_descriptors
            )
            manifest = build_index(paths, parameters)
            _print_build_summary(manifest, output_stream)
            return 0

        paths = validate_query_paths(arguments.index, arguments.cropped, arguments.output)
        query_parameters = QueryParameters(
            query_max_descriptors=arguments.query_max_descriptors,
            neighbor_depth=arguments.neighbor_depth,
            requested_k=arguments.k,
        )
        ensure_sift_available()
        loaded = load_index(paths)
        manifest, _ = query_index(paths, loaded, query_parameters)
        write_manifest_atomic(paths.output, manifest)
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

"""Command-line orchestration and private manifest serialization."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Sequence, TextIO

import PIL

from . import __version__
from .audit import (
    AuditItem,
    AuditResult,
    CollisionGroup,
    CollisionKeyType,
    RootRole,
    ScanIssue,
    ScanIssueCategory,
    SemanticReference,
    audit_datasets,
    build_role_summary,
    semantic_reference_sort_key,
)
from .pairing import PairDiscoveryResult, PairResult, discover_pairs


SCHEMA_VERSION = "1.0"
PRIVATE_OUTPUT_SUFFIX = ".private.json"


class ConfigurationError(ValueError):
    """Raised when explicit input/output paths violate the safety contract."""


class OutputFailure(RuntimeError):
    """Raised when a complete manifest cannot be finalized safely."""

    def __init__(self, error_type: str) -> None:
        super().__init__(error_type)
        self.error_type = error_type


@dataclass(frozen=True, slots=True)
class ValidatedPaths:
    originals: Path
    cropped: Path
    output: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autocrop_analysis",
        description="Audit two explicitly supplied image datasets without modifying them.",
    )
    parser.add_argument("--originals", required=True, type=Path)
    parser.add_argument("--cropped", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def validate_paths(originals: Path, cropped: Path, output: Path) -> ValidatedPaths:
    originals_path = _validate_root(originals, "originals")
    cropped_path = _validate_root(cropped, "cropped")

    if originals_path == cropped_path:
        raise ConfigurationError("input roots must be distinct")
    if _contains(originals_path, cropped_path) or _contains(
        cropped_path, originals_path
    ):
        raise ConfigurationError("input roots must not contain one another")

    if not output.name.endswith(PRIVATE_OUTPUT_SUFFIX):
        raise ConfigurationError("output filename must end with .private.json")
    if output.exists():
        raise ConfigurationError("output must not already exist")

    try:
        output_parent = output.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigurationError("output parent must exist") from exc
    if not output_parent.is_dir():
        raise ConfigurationError("output parent must be a directory")

    output_path = output_parent / output.name
    if _contains(originals_path, output_path) or _contains(cropped_path, output_path):
        raise ConfigurationError("output must be outside both input roots")

    return ValidatedPaths(originals_path, cropped_path, output_path)


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


def _contains(parent: Path, child: Path) -> bool:
    return child == parent or child.is_relative_to(parent)


def build_manifest(
    paths: ValidatedPaths,
    audit_result: AuditResult,
    pair_result: PairDiscoveryResult,
) -> dict[str, object]:
    registered_extensions = dict(audit_result.registered_extensions)
    role_summaries = {
        role.value: build_role_summary(
            (item for item in audit_result.items if item.root_role is role),
            registered_extensions,
        )
        for role in RootRole
    }
    collisions = tuple(
        sorted(
            (*audit_result.collisions, *pair_result.one_to_many_collisions),
            key=_collision_sort_key,
        )
    )
    scan_issue_counts = Counter(issue.category.value for issue in audit_result.scan_issues)

    summary = {
        "roles": role_summaries,
        "pairs": {
            "MATCHED": pair_result.matched_count,
            "reconstruction_ready_matches": pair_result.reconstruction_ready_matched_count,
            "UNMATCHED": pair_result.unmatched_count,
            "AMBIGUOUS": pair_result.ambiguous_count,
        },
        "collisions": _collision_summary(collisions),
        "scan_issues": {
            category.value: scan_issue_counts.get(category.value, 0)
            for category in ScanIssueCategory
        },
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "tool_version": __version__,
        "runtime": {
            "python_version": platform.python_version(),
            "pillow_version": PIL.__version__,
        },
        "roots": {
            RootRole.ORIGINAL.value: str(paths.originals),
            RootRole.CROPPED.value: str(paths.cropped),
        },
        "summary": summary,
        "items": [_serialize_item(item) for item in audit_result.items],
        "scan_issues": [_serialize_scan_issue(issue) for issue in audit_result.scan_issues],
        "collisions": [_serialize_collision(group) for group in collisions],
        "pairs": [_serialize_pair(pair) for pair in pair_result.pairs],
    }


def _serialize_reference(reference: SemanticReference) -> dict[str, str]:
    return {
        "root_role": reference.root_role.value,
        "relative_path": reference.relative_path,
    }


def _serialize_item(item: AuditItem) -> dict[str, object]:
    return {
        "root_role": item.root_role.value,
        "relative_path": item.relative_path,
        "size_bytes": item.size_bytes,
        "extension": item.extension,
        "is_image_candidate": item.is_image_candidate,
        "read_status": item.read_status.value,
        "metadata_status": item.metadata_status.value,
        "detected_format": item.detected_format,
        "encoded_width": item.encoded_width,
        "encoded_height": item.encoded_height,
        "exif_orientation": item.exif_orientation,
        "display_width": item.display_width,
        "display_height": item.display_height,
        "error_category": (
            item.error_category.value if item.error_category is not None else None
        ),
        "error_type": item.error_type,
    }


def _serialize_scan_issue(issue: ScanIssue) -> dict[str, object]:
    return {
        "root_role": issue.root_role.value,
        "relative_path": issue.relative_path,
        "category": issue.category.value,
        "error_type": issue.error_type,
    }


def _serialize_collision(group: CollisionGroup) -> dict[str, object]:
    return {
        "key_type": group.key_type.value,
        "normalized_key": group.normalized_key,
        "members": [_serialize_reference(member) for member in group.members],
    }


def _serialize_pair(pair: PairResult) -> dict[str, object]:
    return {
        "cropped": _serialize_reference(pair.cropped),
        "status": pair.status.value,
        "matched_original": (
            _serialize_reference(pair.matched_original)
            if pair.matched_original is not None
            else None
        ),
        "candidate_count": pair.candidate_count,
        "candidates": [_serialize_reference(candidate) for candidate in pair.candidates],
        "matching_rule": (
            pair.matching_rule.value if pair.matching_rule is not None else None
        ),
    }


def _collision_sort_key(
    group: CollisionGroup,
) -> tuple[str, str, tuple[tuple[int, str, str], ...]]:
    return (
        group.key_type.value,
        group.normalized_key,
        tuple(semantic_reference_sort_key(member) for member in group.members),
    )


def _collision_summary(collisions: Sequence[CollisionGroup]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for collision_key_type in CollisionKeyType:
        key_type = collision_key_type.value
        matching = [
            collision
            for collision in collisions
            if collision.key_type.value == key_type
        ]
        summary[key_type] = {
            "groups": len(matching),
            "items": sum(len(group.members) for group in matching),
        }
    return summary


def write_manifest_atomic(output: Path, manifest: dict[str, object]) -> None:
    """Publish a complete manifest atomically without replacing an existing path.

    A same-directory hard link creates the destination as one atomic filesystem
    operation and fails if that directory entry already exists. This provides
    no-clobber publication on Windows (for example, NTFS) and POSIX filesystems
    that support hard links. An unsupported hard-link operation fails safely as
    an output error; it never falls back to replacement semantics.
    """

    temporary_path: Path | None = None
    try:
        if output.exists():
            raise FileExistsError
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                manifest,
                temporary,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())

        if output.exists():
            raise FileExistsError
        os.link(temporary_path, output)
        try:
            temporary_path.unlink()
        except OSError:
            pass
        temporary_path = None
    except Exception as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise OutputFailure(type(exc).__name__) from exc


def print_summary(manifest: dict[str, object], stream: TextIO) -> None:
    summary = manifest["summary"]
    assert isinstance(summary, dict)
    roles = summary["roles"]
    assert isinstance(roles, dict)
    for role in RootRole:
        role_summary = roles[role.value]
        assert isinstance(role_summary, dict)
        print(
            f"{role.value}: "
            f"files={role_summary['total_regular_files']} "
            f"candidates={role_summary['image_candidates']} "
            f"non_image={role_summary['non_image_files']} "
            f"readable={role_summary['readable_candidates']} "
            f"unsupported={role_summary['unsupported_candidates']} "
            f"unreadable={role_summary['unreadable_candidates']} "
            f"filesystem_errors={role_summary['filesystem_error_candidates']}",
            file=stream,
        )
    pairs = summary["pairs"]
    assert isinstance(pairs, dict)
    print(
        "PAIRS: "
        f"matched={pairs['MATCHED']} "
        f"reconstruction_ready={pairs['reconstruction_ready_matches']} "
        f"unmatched={pairs['UNMATCHED']} "
        f"ambiguous={pairs['AMBIGUOUS']}",
        file=stream,
    )
    issue_counts = summary["scan_issues"]
    assert isinstance(issue_counts, dict)
    print(f"SCAN_ISSUES: total={sum(issue_counts.values())}", file=stream)


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
        paths = validate_paths(arguments.originals, arguments.cropped, arguments.output)
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=error_stream)
        return 2

    try:
        audit_result = audit_datasets(paths.originals, paths.cropped)
        pair_result = discover_pairs(audit_result.items)
        manifest = build_manifest(paths, audit_result, pair_result)
    except Exception as exc:
        print(f"unexpected internal failure: {type(exc).__name__}", file=error_stream)
        return 1

    try:
        write_manifest_atomic(paths.output, manifest)
    except OutputFailure as exc:
        print(f"manifest output failure: {exc.error_type}", file=error_stream)
        return 3

    print_summary(manifest, output_stream)
    return 0

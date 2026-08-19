"""CLI for deterministic manifest-to-manifest crop reconstruction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
from typing import Sequence, TextIO

from .cli import ConfigurationError, OutputFailure, write_manifest_atomic
from .crop_reconstruction import (
    ManifestValidationError,
    ReconstructionParameters,
    parse_provenance_bytes,
    reconstruct_manifest,
    validate_provenance_manifest,
)


PRIVATE_OUTPUT_SUFFIX = ".private.json"


@dataclass(frozen=True, slots=True)
class ValidatedReconstructionPaths:
    provenance: Path
    output: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autocrop_analysis.crop_reconstruction_cli",
        description="Reconstruct canonical crops from a private schema 1.1 provenance manifest.",
    )
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def validate_reconstruction_paths(
    provenance: Path, output: Path
) -> ValidatedReconstructionPaths:
    if not provenance.name.endswith(PRIVATE_OUTPUT_SUFFIX):
        raise ConfigurationError("provenance filename must end with .private.json")
    try:
        provenance_path = provenance.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigurationError("provenance input must exist") from exc
    if not provenance_path.is_file():
        raise ConfigurationError("provenance input must be a regular file")
    if not os.access(provenance_path, os.R_OK):
        raise ConfigurationError("provenance input must be readable")

    if not output.name.endswith(PRIVATE_OUTPUT_SUFFIX):
        raise ConfigurationError("output filename must end with .private.json")
    try:
        output_parent = output.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigurationError("output parent must exist") from exc
    if not output_parent.is_dir():
        raise ConfigurationError("output parent must be a directory")
    output_path = output_parent / output.name
    if provenance_path == output_path:
        raise ConfigurationError("provenance input and output must differ")
    if output_path.exists():
        raise ConfigurationError("output must not already exist")
    return ValidatedReconstructionPaths(provenance_path, output_path)


def print_summary(manifest: dict[str, object], stream: TextIO) -> None:
    summary = manifest["summary"]
    assert isinstance(summary, dict)
    print(
        "CROP_RECONSTRUCTION: "
        f"items={summary['items']} "
        f"reconstructed={summary['RECONSTRUCTED']} "
        f"not_reconstructed={summary['NOT_RECONSTRUCTED']}",
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
        paths = validate_reconstruction_paths(arguments.provenance, arguments.output)
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=error_stream)
        return 2

    try:
        raw_bytes = paths.provenance.read_bytes()
    except OSError as exc:
        print(f"input manifest error: READ_{type(exc).__name__}", file=error_stream)
        return 4

    try:
        parsed = parse_provenance_bytes(raw_bytes)
        parameters = ReconstructionParameters()
        provenance = validate_provenance_manifest(parsed, parameters)
        manifest = reconstruct_manifest(
            provenance,
            source_path=paths.provenance,
            source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            parameters=parameters,
        )
    except ManifestValidationError as exc:
        print(f"input manifest error: {exc.code}", file=error_stream)
        return 4
    except Exception as exc:
        print(f"unexpected internal failure: {type(exc).__name__}", file=error_stream)
        return 1

    try:
        write_manifest_atomic(paths.output, manifest)
    except OutputFailure as exc:
        print(f"result output failure: {exc.error_type}", file=error_stream)
        return 3

    print_summary(manifest, output_stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

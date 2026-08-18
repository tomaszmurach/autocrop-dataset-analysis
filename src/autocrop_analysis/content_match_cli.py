"""Explicit CLI for the isolated content-pairing feasibility experiment."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import argparse
from pathlib import Path
import platform
import sys
from typing import Sequence, TextIO

import cv2
import numpy as np
import PIL

from . import __version__
from .audit import (
    AuditItem,
    ReadStatus,
    RootRole,
    SemanticReference,
    audit_datasets,
    audit_item_sort_key,
)
from .cli import ConfigurationError, OutputFailure, ValidatedPaths, validate_paths, write_manifest_atomic
from .content_matching import (
    ALGORITHM_NAME,
    PROJECTED_CORNER_ORDER,
    CandidateEvidence,
    ContentDecision,
    CropMatchResult,
    FeatureImage,
    MatchingParameters,
    ProvenanceInterpretation,
    ensure_sift_available,
    extract_features,
    finalize_candidate_set_completeness,
    match_crops,
)


SCHEMA_VERSION = "1.1"


@dataclass(frozen=True, slots=True)
class PreparedInputs:
    originals: tuple[FeatureImage, ...]
    crops: tuple[FeatureImage, ...]
    supplied_original_candidates: int
    audit_readable_original_candidates: int
    audit_unreadable_original_candidates: int
    audit_unsupported_original_candidates: int
    audit_filesystem_error_original_candidates: int
    feature_extraction_unavailable_originals: int
    successfully_preprocessed_originals: int
    candidate_set_complete: bool
    audit_unavailable_original_candidates: int
    audit_unavailable_crop_candidates: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autocrop_analysis.content_match_cli",
        description="Run experimental exact-frame content provenance discovery.",
    )
    parser.add_argument("--originals", required=True, type=Path)
    parser.add_argument("--cropped", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def prepare_inputs(paths: ValidatedPaths, parameters: MatchingParameters) -> PreparedInputs:
    audit = audit_datasets(paths.originals, paths.cropped)
    original_candidates = tuple(
        item
        for item in audit.items
        if item.root_role is RootRole.ORIGINAL and item.is_image_candidate
    )
    readable = tuple(
        sorted(
            (
                item
                for item in audit.items
                if item.is_image_candidate and item.read_status is ReadStatus.READABLE
            ),
            key=audit_item_sort_key,
        )
    )
    originals = tuple(
        _extract_item(item, paths.originals, parameters, retain_grayscale=False)
        for item in readable
        if item.root_role is RootRole.ORIGINAL
    )
    crops = tuple(
        _extract_item(item, paths.cropped, parameters, retain_grayscale=True)
        for item in readable
        if item.root_role is RootRole.CROPPED
    )
    excluded_originals = sum(
        item.root_role is RootRole.ORIGINAL
        and item.is_image_candidate
        and item.read_status is not ReadStatus.READABLE
        for item in audit.items
    )
    excluded_crops = sum(
        item.root_role is RootRole.CROPPED
        and item.is_image_candidate
        and item.read_status is not ReadStatus.READABLE
        for item in audit.items
    )
    unavailable_features = sum(
        original.descriptors is None or original.diagnostic_reason is not None
        for original in originals
    )
    successfully_preprocessed = len(originals) - unavailable_features
    return PreparedInputs(
        originals=originals,
        crops=crops,
        supplied_original_candidates=len(original_candidates),
        audit_readable_original_candidates=sum(
            item.read_status is ReadStatus.READABLE for item in original_candidates
        ),
        audit_unreadable_original_candidates=sum(
            item.read_status is ReadStatus.UNREADABLE for item in original_candidates
        ),
        audit_unsupported_original_candidates=sum(
            item.read_status is ReadStatus.UNSUPPORTED for item in original_candidates
        ),
        audit_filesystem_error_original_candidates=sum(
            item.read_status is ReadStatus.FILESYSTEM_ERROR for item in original_candidates
        ),
        feature_extraction_unavailable_originals=unavailable_features,
        successfully_preprocessed_originals=successfully_preprocessed,
        candidate_set_complete=(
            len(original_candidates) == successfully_preprocessed
        ),
        audit_unavailable_original_candidates=excluded_originals,
        audit_unavailable_crop_candidates=excluded_crops,
    )


def _extract_item(
    item: AuditItem,
    root: Path,
    parameters: MatchingParameters,
    *,
    retain_grayscale: bool,
) -> FeatureImage:
    return extract_features(
        root / Path(item.relative_path),
        item.reference,
        parameters,
        retain_grayscale=retain_grayscale,
    )


def build_manifest(
    paths: ValidatedPaths,
    prepared: PreparedInputs,
    results: tuple[CropMatchResult, ...],
    parameters: MatchingParameters,
    *,
    candidate_set_complete: bool,
) -> dict[str, object]:
    crop_images = _index_feature_images(prepared.crops)
    original_images = _index_feature_images(prepared.originals)
    decisions = Counter(result.decision.value for result in results)
    lazy_unavailable_originals = {
        candidate.original
        for result in results
        for candidate in result.ranked_candidates
        if not candidate.evaluation_complete
        and candidate.diagnostic_reason.startswith("ORIGINAL_PHOTOMETRIC_DECODE_FAILED:")
    }
    observed_candidate_set_complete = prepared.candidate_set_complete and all(
        candidate.evaluation_complete
        for result in results
        for candidate in result.ranked_candidates
    )
    if candidate_set_complete is not observed_candidate_set_complete:
        raise ValueError("INCONSISTENT_CANDIDATE_SET_COMPLETENESS")
    if not candidate_set_complete and (
        decisions[ContentDecision.MATCHED.value] != 0
        or any(
            result.decision is not ContentDecision.AMBIGUOUS
            or result.provenance_interpretation
            is not ProvenanceInterpretation.AMBIGUOUS_PROVENANCE
            or result.diagnostic_reason != "INCOMPLETE_CANDIDATE_SET"
            for result in results
        )
    ):
        raise ValueError("INCONSISTENT_INCOMPLETE_CANDIDATE_SET")
    return {
        "schema_version": SCHEMA_VERSION,
        "tool_version": __version__,
        "runtime": {
            "python_version": platform.python_version(),
            "pillow_version": PIL.__version__,
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "sift_available": True,
        },
        "algorithm": {
            "name": ALGORITHM_NAME,
            "status": "EXPERIMENTAL_UNLABELED_PROVENANCE_DISCOVERY",
            "grayscale_convention": "uint8_0_255_after_exif_transpose",
            "geometric_model": "estimateAffinePartial2D_RANSAC_crop_to_original_display",
            "projected_corner_order": list(PROJECTED_CORNER_ORDER),
            "ranking_policy": "strong_then_plausible_geometry_then_photometric_then_geometry_then_semantic_reference",
        },
        "parameters": parameters.as_dict(),
        "roots": {
            RootRole.ORIGINAL.value: str(paths.originals),
            RootRole.CROPPED.value: str(paths.cropped),
        },
        "summary": {
            "originals_analyzed": len(prepared.originals),
            "crops_analyzed": len(prepared.crops),
            "candidate_comparisons": len(prepared.originals) * len(prepared.crops),
            "supplied_original_image_candidates": prepared.supplied_original_candidates,
            "audit_readable_original_candidates": prepared.audit_readable_original_candidates,
            "audit_unreadable_original_candidates": prepared.audit_unreadable_original_candidates,
            "audit_unsupported_original_candidates": prepared.audit_unsupported_original_candidates,
            "audit_filesystem_error_original_candidates": prepared.audit_filesystem_error_original_candidates,
            "feature_extraction_unavailable_originals": prepared.feature_extraction_unavailable_originals,
            "photometric_decode_unavailable_originals": len(lazy_unavailable_originals),
            "successfully_evaluated_originals": (
                prepared.successfully_preprocessed_originals
                - len(lazy_unavailable_originals)
            ),
            "candidate_set_complete": candidate_set_complete,
            "audit_unavailable_original_candidates": prepared.audit_unavailable_original_candidates,
            "audit_unavailable_crop_candidates": prepared.audit_unavailable_crop_candidates,
            "MATCHED": decisions[ContentDecision.MATCHED.value],
            "UNIQUE_STRONG_PROVENANCE": decisions[ContentDecision.MATCHED.value],
            "AMBIGUOUS": decisions[ContentDecision.AMBIGUOUS.value],
            "AMBIGUOUS_PROVENANCE": decisions[ContentDecision.AMBIGUOUS.value],
            "NO_MATCH": decisions[ContentDecision.NO_MATCH.value],
            "NO_VALID_PROVENANCE": decisions[ContentDecision.NO_MATCH.value],
        },
        "crops": [
            _serialize_crop_result(result, crop_images, original_images)
            for result in results
        ],
    }


def _index_feature_images(
    images: tuple[FeatureImage, ...],
) -> dict[SemanticReference, FeatureImage]:
    indexed: dict[SemanticReference, FeatureImage] = {}
    for image in images:
        if image.reference in indexed:
            raise ValueError("DUPLICATE_FEATURE_IMAGE_REFERENCE")
        indexed[image.reference] = image
    return indexed


def _require_feature_image(
    reference: SemanticReference,
    images: dict[SemanticReference, FeatureImage],
) -> FeatureImage:
    try:
        return images[reference]
    except KeyError as exc:
        raise ValueError("MISSING_FEATURE_IMAGE_REFERENCE") from exc


def _serialize_reference(
    reference: SemanticReference,
    image: FeatureImage,
) -> dict[str, str | int | None]:
    if image.reference != reference:
        raise ValueError("INCONSISTENT_FEATURE_IMAGE_REFERENCE")
    display_width, display_height = _display_dimensions(image)
    return {
        "root_role": reference.root_role.value,
        "relative_path": reference.relative_path,
        "display_width": display_width,
        "display_height": display_height,
    }


def _display_dimensions(image: FeatureImage) -> tuple[int | None, int | None]:
    if type(image.width) is not int or type(image.height) is not int:
        raise ValueError("INVALID_DISPLAY_DIMENSION_TYPE")
    if image.width > 0 and image.height > 0:
        return image.width, image.height
    if image.width == 0 and image.height == 0:
        return None, None
    raise ValueError("INCONSISTENT_DISPLAY_DIMENSIONS")


def _serialize_crop_result(
    result: CropMatchResult,
    crop_images: dict[SemanticReference, FeatureImage],
    original_images: dict[SemanticReference, FeatureImage],
) -> dict[str, object]:
    best = result.ranked_candidates[0] if result.ranked_candidates else None
    second = next(
        (
            candidate
            for candidate in result.ranked_candidates[1:]
            if candidate.transform_plausible
        ),
        None,
    )
    crop_image = _require_feature_image(result.crop, crop_images)
    return {
        "crop": _serialize_reference(result.crop, crop_image),
        "decision": result.decision.value,
        "provenance_interpretation": result.provenance_interpretation.value,
        "diagnostic_reason": result.diagnostic_reason,
        "best_vs_second_margins": _serialize_margins(best, second),
        "ranked_candidates": [
            _serialize_candidate(rank, candidate, original_images)
            for rank, candidate in enumerate(result.ranked_candidates, start=1)
        ],
    }


def _serialize_margins(
    best: CandidateEvidence | None,
    second: CandidateEvidence | None,
) -> dict[str, float | None]:
    if best is None or second is None:
        return {
            "luminance_correlation": None,
            "gradient_correlation": None,
            "normalized_residual": None,
        }
    return {
        "luminance_correlation": _difference(best.luminance_correlation, second.luminance_correlation),
        "gradient_correlation": _difference(best.gradient_correlation, second.gradient_correlation),
        "normalized_residual": _difference(second.normalized_residual, best.normalized_residual),
    }


def _difference(first: float | None, second: float | None) -> float | None:
    return None if first is None or second is None else first - second


def _serialize_candidate(
    rank: int,
    candidate: CandidateEvidence,
    original_images: dict[SemanticReference, FeatureImage],
) -> dict[str, object]:
    original_image = _require_feature_image(candidate.original, original_images)
    return {
        "rank": rank,
        "original": _serialize_reference(candidate.original, original_image),
        "descriptor_evidence": {
            "crop_keypoints": candidate.crop_keypoints,
            "original_keypoints": candidate.original_keypoints,
            "raw_knn_count": candidate.raw_knn_count,
            "ratio_passed_count": candidate.ratio_passed_count,
            "reverse_ratio_passed_count": candidate.reverse_ratio_passed_count,
            "mutual_match_count": candidate.mutual_match_count,
        },
        "geometric_evidence": {
            "transform_valid": candidate.transform_valid,
            "inlier_count": candidate.inlier_count,
            "inlier_ratio": candidate.inlier_ratio,
            "reprojection_rmse": candidate.reprojection_rmse,
            "reprojection_median": candidate.reprojection_median,
            "reprojection_p95": candidate.reprojection_p95,
            "bbox_coverage": candidate.bbox_coverage,
            "grid_coverage": candidate.grid_coverage,
        },
        "transform_evidence": {
            "plausible": candidate.transform_plausible,
            "scale": candidate.scale,
            "rotation_degrees": candidate.rotation_degrees,
            "translation_x": candidate.translation_x,
            "translation_y": candidate.translation_y,
            "projected_corners": [list(point) for point in candidate.projected_corners],
            "projected_inside_fraction": candidate.projected_inside_fraction,
        },
        "photometric_evidence": {
            "base_luminance_correlation": candidate.base_luminance_correlation,
            "base_gradient_correlation": candidate.base_gradient_correlation,
            "base_normalized_residual": candidate.base_normalized_residual,
            "luminance_correlation": candidate.luminance_correlation,
            "gradient_correlation": candidate.gradient_correlation,
            "normalized_residual": candidate.normalized_residual,
            "alignment_refined": candidate.alignment_refined,
            "refinement_shift_x": candidate.refinement_shift_x,
            "refinement_shift_y": candidate.refinement_shift_y,
        },
        "evaluation_complete": candidate.evaluation_complete,
        "strong_provenance": candidate.strong_provenance,
        "diagnostic_reason": candidate.diagnostic_reason,
    }


def print_summary(manifest: dict[str, object], stream: TextIO) -> None:
    summary = manifest["summary"]
    assert isinstance(summary, dict)
    print(
        "CONTENT_PROVENANCE: "
        f"originals={summary['originals_analyzed']} "
        f"crops={summary['crops_analyzed']} "
        f"comparisons={summary['candidate_comparisons']} "
        f"evaluated_originals={summary['successfully_evaluated_originals']}/"
        f"{summary['supplied_original_image_candidates']} "
        f"candidate_set_complete={str(summary['candidate_set_complete']).lower()} "
        f"matched_unique={summary['MATCHED']} "
        f"ambiguous={summary['AMBIGUOUS']} "
        f"no_match={summary['NO_MATCH']}",
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
        paths = validate_paths(arguments.originals, arguments.cropped, arguments.output)
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=error_stream)
        return 2

    parameters = MatchingParameters()
    try:
        ensure_sift_available()
        prepared = prepare_inputs(paths, parameters)
        initial_results = match_crops(
            prepared.crops,
            prepared.originals,
            parameters,
            candidate_set_complete=prepared.candidate_set_complete,
        )
        results, candidate_set_complete = finalize_candidate_set_completeness(
            initial_results,
            preprocessed_candidate_set_complete=prepared.candidate_set_complete,
        )
        manifest = build_manifest(
            paths,
            prepared,
            results,
            parameters,
            candidate_set_complete=candidate_set_complete,
        )
    except RuntimeError as exc:
        if str(exc) == "SIFT_UNAVAILABLE":
            print("runtime dependency error: SIFT_UNAVAILABLE", file=error_stream)
            return 1
        print(f"unexpected internal failure: {type(exc).__name__}", file=error_stream)
        return 1
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

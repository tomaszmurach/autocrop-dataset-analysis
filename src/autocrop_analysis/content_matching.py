"""Experimental exact-frame provenance matching using image content.

Images are decoded into EXIF-normalized, 8-bit grayscale arrays in the range
``[0, 255]``.  The subsystem is intentionally separate from deterministic
filename/stem pairing.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps

from .audit import SemanticReference, semantic_reference_sort_key


ALGORITHM_NAME = "SIFT_BF_L2_MUTUAL_SIMILARITY_RANSAC_ALIGNED_PHOTOMETRIC"
GRID_SIZE = 4
PROJECTED_CORNER_ORDER = ("TOP_LEFT", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT")


class ContentDecision(str, Enum):
    MATCHED = "MATCHED"
    AMBIGUOUS = "AMBIGUOUS"
    NO_MATCH = "NO_MATCH"


class ProvenanceInterpretation(str, Enum):
    UNIQUE_STRONG_PROVENANCE = "UNIQUE_STRONG_PROVENANCE"
    AMBIGUOUS_PROVENANCE = "AMBIGUOUS_PROVENANCE"
    NO_VALID_PROVENANCE = "NO_VALID_PROVENANCE"


@dataclass(frozen=True, slots=True)
class MatchingParameters:
    """Explicit provisional thresholds, calibrated only with synthetic data."""

    sift_nfeatures: int = 3_000
    ratio_threshold: float = 0.75
    min_mutual_matches: int = 6
    ransac_reprojection_threshold: float = 4.0
    ransac_max_iterations: int = 2_000
    ransac_confidence: float = 0.99
    ransac_refine_iterations: int = 10
    random_seed: int = 17_029
    min_inliers: int = 6
    min_inlier_ratio: float = 0.50
    max_reprojection_rmse: float = 4.0
    min_bbox_coverage: float = 0.10
    min_grid_coverage: float = 0.1875
    min_scale: float = 0.05
    max_scale: float = 25.0
    max_abs_rotation_degrees: float = 5.0
    min_projected_inside_fraction: float = 0.97
    alignment_refine_radius_pixels: int = 1
    min_luminance_correlation: float = 0.88
    min_gradient_correlation: float = 0.50
    max_normalized_residual: float = 0.26
    min_luminance_margin: float = 0.015
    min_gradient_margin: float = 0.015
    min_residual_margin: float = 0.010

    def as_dict(self) -> dict[str, int | float]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class FeatureImage:
    reference: SemanticReference
    source_path: Path | None
    width: int
    height: int
    grayscale: np.ndarray | None
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray | None
    diagnostic_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    crop: SemanticReference
    original: SemanticReference
    crop_keypoints: int
    original_keypoints: int
    raw_knn_count: int
    ratio_passed_count: int
    reverse_ratio_passed_count: int
    mutual_match_count: int
    transform_valid: bool
    inlier_count: int
    inlier_ratio: float | None
    reprojection_rmse: float | None
    reprojection_median: float | None
    reprojection_p95: float | None
    bbox_coverage: float | None
    grid_coverage: float | None
    scale: float | None
    rotation_degrees: float | None
    translation_x: float | None
    translation_y: float | None
    projected_corners: tuple[tuple[float, float], ...]
    projected_inside_fraction: float | None
    transform_plausible: bool
    base_luminance_correlation: float | None
    base_gradient_correlation: float | None
    base_normalized_residual: float | None
    luminance_correlation: float | None
    gradient_correlation: float | None
    normalized_residual: float | None
    alignment_refined: bool
    refinement_shift_x: int
    refinement_shift_y: int
    evaluation_complete: bool
    strong_provenance: bool
    diagnostic_reason: str


@dataclass(frozen=True, slots=True)
class CropMatchResult:
    crop: SemanticReference
    decision: ContentDecision
    provenance_interpretation: ProvenanceInterpretation
    diagnostic_reason: str
    ranked_candidates: tuple[CandidateEvidence, ...]


def configure_opencv(parameters: MatchingParameters) -> None:
    """Apply the reproducibility controls available through OpenCV."""

    if hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads(1)
    cv2.setRNGSeed(parameters.random_seed)


def ensure_sift_available() -> None:
    """Fail explicitly rather than substituting a different detector."""

    sift_factory = getattr(cv2, "SIFT_create", None)
    if sift_factory is None:
        raise RuntimeError("SIFT_UNAVAILABLE")
    try:
        sift_factory()
    except Exception as exc:  # pragma: no cover - depends on binary build
        raise RuntimeError("SIFT_UNAVAILABLE") from exc


def extract_features(
    path: Path,
    reference: SemanticReference,
    parameters: MatchingParameters,
    *,
    retain_grayscale: bool = True,
) -> FeatureImage:
    """Extract bounded features, optionally releasing decoded pixels afterward."""

    try:
        grayscale = _decode_grayscale(path)
    except Exception as exc:
        return FeatureImage(reference, path, 0, 0, None, (), None, type(exc).__name__)

    grayscale = np.ascontiguousarray(grayscale)
    grayscale.setflags(write=False)
    sift = cv2.SIFT_create(nfeatures=parameters.sift_nfeatures)
    try:
        keypoint_list, descriptors = sift.detectAndCompute(grayscale, None)
    except Exception as exc:
        return FeatureImage(
            reference,
            path,
            int(grayscale.shape[1]),
            int(grayscale.shape[0]),
            grayscale if retain_grayscale else None,
            (),
            None,
            type(exc).__name__,
        )

    keypoint_list = keypoint_list or []
    order = sorted(range(len(keypoint_list)), key=lambda index: _keypoint_sort_key(keypoint_list[index]))
    if parameters.sift_nfeatures > 0:
        order = order[: parameters.sift_nfeatures]
    keypoints = tuple(keypoint_list[index] for index in order)
    if descriptors is None or not order:
        stable_descriptors = None
        diagnostic_reason = "NO_DESCRIPTORS"
    else:
        stable_descriptors = np.ascontiguousarray(descriptors[order], dtype=np.float32)
        stable_descriptors.setflags(write=False)
        diagnostic_reason = None

    return FeatureImage(
        reference,
        path,
        int(grayscale.shape[1]),
        int(grayscale.shape[0]),
        grayscale if retain_grayscale else None,
        keypoints,
        stable_descriptors,
        diagnostic_reason,
    )


def unavailable_feature_image(
    reference: SemanticReference,
    diagnostic_reason: str,
) -> FeatureImage:
    return FeatureImage(reference, None, 0, 0, None, (), None, diagnostic_reason)


def compare_candidate(
    crop: FeatureImage,
    original: FeatureImage,
    parameters: MatchingParameters,
) -> CandidateEvidence:
    """Produce immutable descriptor, geometry, and photometric evidence."""

    base = {
        "crop": crop.reference,
        "original": original.reference,
        "crop_keypoints": len(crop.keypoints),
        "original_keypoints": len(original.keypoints),
    }
    if crop.grayscale is None or crop.descriptors is None:
        return _invalid_evidence(
            **base,
            diagnostic_reason=f"CROP_UNAVAILABLE:{crop.diagnostic_reason or 'NO_DESCRIPTORS'}",
        )
    if original.descriptors is None:
        return _invalid_evidence(
            **base,
            evaluation_complete=False,
            diagnostic_reason=f"ORIGINAL_UNAVAILABLE:{original.diagnostic_reason or 'NO_DESCRIPTORS'}",
        )

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    forward_knn = matcher.knnMatch(crop.descriptors, original.descriptors, k=2)
    reverse_knn = matcher.knnMatch(original.descriptors, crop.descriptors, k=2)
    forward = _ratio_matches(forward_knn, parameters.ratio_threshold)
    reverse = _ratio_matches(reverse_knn, parameters.ratio_threshold)
    reverse_pairs = {(match.queryIdx, match.trainIdx) for match in reverse}
    mutual = tuple(
        match for match in forward if (match.trainIdx, match.queryIdx) in reverse_pairs
    )

    descriptor_counts = {
        "raw_knn_count": sum(len(row) >= 2 for row in forward_knn),
        "ratio_passed_count": len(forward),
        "reverse_ratio_passed_count": len(reverse),
        "mutual_match_count": len(mutual),
    }
    if len(mutual) < parameters.min_mutual_matches:
        return _invalid_evidence(
            **base,
            **descriptor_counts,
            diagnostic_reason="INSUFFICIENT_MUTUAL_MATCHES",
        )

    crop_points = np.float32([crop.keypoints[match.queryIdx].pt for match in mutual])
    original_points = np.float32(
        [original.keypoints[match.trainIdx].pt for match in mutual]
    )
    cv2.setRNGSeed(parameters.random_seed)
    transform, inlier_mask = cv2.estimateAffinePartial2D(
        crop_points,
        original_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=parameters.ransac_reprojection_threshold,
        maxIters=parameters.ransac_max_iterations,
        confidence=parameters.ransac_confidence,
        refineIters=parameters.ransac_refine_iterations,
    )
    if transform is None or inlier_mask is None or not np.isfinite(transform).all():
        return _invalid_evidence(
            **base,
            **descriptor_counts,
            diagnostic_reason="TRANSFORM_ESTIMATION_FAILED",
        )

    transform = np.asarray(transform, dtype=np.float64)
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    inlier_count = int(mask.sum())
    inlier_ratio = inlier_count / len(mutual)
    projected_matches = cv2.transform(crop_points.reshape(-1, 1, 2), transform).reshape(-1, 2)
    errors = np.linalg.norm(projected_matches[mask] - original_points[mask], axis=1)
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors.size else None
    median = float(np.median(errors)) if errors.size else None
    p95 = float(np.percentile(errors, 95)) if errors.size else None
    inlier_crop_points = crop_points[mask]
    bbox_coverage, grid_coverage = _spatial_coverage(
        inlier_crop_points, crop.width, crop.height
    )

    a = float(transform[0, 0])
    b = float(transform[1, 0])
    scale = math.hypot(a, b)
    rotation_degrees = math.degrees(math.atan2(b, a))
    corners = np.float32(
        [[0.0, 0.0], [crop.width, 0.0], [crop.width, crop.height], [0.0, crop.height]]
    )
    projected = cv2.transform(corners.reshape(-1, 1, 2), transform).reshape(-1, 2)
    projected_corners = tuple((float(x), float(y)) for x, y in projected)
    inside_fraction = _projected_inside_fraction(projected, original.width, original.height)

    transform_plausible = (
        inlier_count >= parameters.min_inliers
        and inlier_ratio >= parameters.min_inlier_ratio
        and rmse is not None
        and rmse <= parameters.max_reprojection_rmse
        and bbox_coverage >= parameters.min_bbox_coverage
        and grid_coverage >= parameters.min_grid_coverage
        and parameters.min_scale <= scale <= parameters.max_scale
        and abs(rotation_degrees) <= parameters.max_abs_rotation_degrees
        and inside_fraction >= parameters.min_projected_inside_fraction
    )

    geometry = {
        "transform_valid": True,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "reprojection_rmse": rmse,
        "reprojection_median": median,
        "reprojection_p95": p95,
        "bbox_coverage": bbox_coverage,
        "grid_coverage": grid_coverage,
        "scale": scale,
        "rotation_degrees": rotation_degrees,
        "translation_x": float(transform[0, 2]),
        "translation_y": float(transform[1, 2]),
        "projected_corners": projected_corners,
        "projected_inside_fraction": inside_fraction,
        "transform_plausible": transform_plausible,
    }
    if not transform_plausible:
        return CandidateEvidence(
            **base,
            **descriptor_counts,
            **geometry,
            base_luminance_correlation=None,
            base_gradient_correlation=None,
            base_normalized_residual=None,
            luminance_correlation=None,
            gradient_correlation=None,
            normalized_residual=None,
            alignment_refined=False,
            refinement_shift_x=0,
            refinement_shift_y=0,
            evaluation_complete=True,
            strong_provenance=False,
            diagnostic_reason=_geometry_failure_reason(
                inlier_count,
                inlier_ratio,
                rmse,
                bbox_coverage,
                grid_coverage,
                scale,
                rotation_degrees,
                inside_fraction,
                parameters,
            ),
        )

    original_grayscale, decode_error = _grayscale_for_photometric_verification(original)
    if original_grayscale is None:
        return CandidateEvidence(
            **base,
            **descriptor_counts,
            **geometry,
            base_luminance_correlation=None,
            base_gradient_correlation=None,
            base_normalized_residual=None,
            luminance_correlation=None,
            gradient_correlation=None,
            normalized_residual=None,
            alignment_refined=False,
            refinement_shift_x=0,
            refinement_shift_y=0,
            evaluation_complete=False,
            strong_provenance=False,
            diagnostic_reason=f"ORIGINAL_PHOTOMETRIC_DECODE_FAILED:{decode_error}",
        )

    base_photo = _photometric_metrics(crop.grayscale, original_grayscale, transform)
    best_photo, shift_x, shift_y = _refine_alignment(
        crop.grayscale,
        original_grayscale,
        transform,
        parameters.alignment_refine_radius_pixels,
    )
    luma, gradient, residual = best_photo
    strong = (
        luma is not None
        and gradient is not None
        and residual is not None
        and luma >= parameters.min_luminance_correlation
        and gradient >= parameters.min_gradient_correlation
        and residual <= parameters.max_normalized_residual
    )
    return CandidateEvidence(
        **base,
        **descriptor_counts,
        **geometry,
        base_luminance_correlation=base_photo[0],
        base_gradient_correlation=base_photo[1],
        base_normalized_residual=base_photo[2],
        luminance_correlation=luma,
        gradient_correlation=gradient,
        normalized_residual=residual,
        alignment_refined=(shift_x != 0 or shift_y != 0),
        refinement_shift_x=shift_x,
        refinement_shift_y=shift_y,
        evaluation_complete=True,
        strong_provenance=strong,
        diagnostic_reason=("STRONG_PROVENANCE_EVIDENCE" if strong else _photometric_failure_reason(luma, gradient, residual, parameters)),
    )


def rank_candidates(
    candidates: Iterable[CandidateEvidence],
) -> tuple[CandidateEvidence, ...]:
    """Rank by an explicit evidence hierarchy, with semantic ties last."""

    return tuple(sorted(candidates, key=_candidate_sort_key))


def decide_candidates(
    crop: SemanticReference,
    ranked_candidates: Iterable[CandidateEvidence],
    parameters: MatchingParameters,
    *,
    candidate_set_complete: bool = True,
) -> CropMatchResult:
    """Apply conservative externally visible decision semantics."""

    ranked = rank_candidates(ranked_candidates)
    if not candidate_set_complete or any(
        not candidate.evaluation_complete for candidate in ranked
    ):
        return CropMatchResult(
            crop,
            ContentDecision.AMBIGUOUS,
            ProvenanceInterpretation.AMBIGUOUS_PROVENANCE,
            "INCOMPLETE_CANDIDATE_SET",
            ranked,
        )
    if not ranked or not ranked[0].strong_provenance:
        return CropMatchResult(
            crop,
            ContentDecision.NO_MATCH,
            ProvenanceInterpretation.NO_VALID_PROVENANCE,
            "NO_CANDIDATE_WITH_STRONG_PROVENANCE_EVIDENCE",
            ranked,
        )

    best = ranked[0]
    base_ranked = tuple(
        sorted(
            (candidate for candidate in ranked if candidate.transform_plausible),
            key=_base_photometric_sort_key,
        )
    )
    if base_ranked and base_ranked[0].original != best.original:
        return CropMatchResult(
            crop,
            ContentDecision.AMBIGUOUS,
            ProvenanceInterpretation.AMBIGUOUS_PROVENANCE,
            "ALIGNMENT_REFINEMENT_CHANGED_WINNER",
            ranked,
        )

    geometry_ranked = tuple(
        sorted(
            (candidate for candidate in ranked if candidate.transform_plausible),
            key=_geometry_sort_key,
        )
    )
    if geometry_ranked and geometry_ranked[0].original != best.original:
        return CropMatchResult(
            crop,
            ContentDecision.AMBIGUOUS,
            ProvenanceInterpretation.AMBIGUOUS_PROVENANCE,
            "GEOMETRIC_AND_PHOTOMETRIC_LEADERS_DISAGREE",
            ranked,
        )

    runner_up = next(
        (candidate for candidate in ranked[1:] if candidate.transform_plausible), None
    )
    if runner_up is not None and not _has_runner_up_separation(best, runner_up, parameters):
        return CropMatchResult(
            crop,
            ContentDecision.AMBIGUOUS,
            ProvenanceInterpretation.AMBIGUOUS_PROVENANCE,
            "INSUFFICIENT_BEST_VS_SECOND_SEPARATION",
            ranked,
        )

    return CropMatchResult(
        crop,
        ContentDecision.MATCHED,
        ProvenanceInterpretation.UNIQUE_STRONG_PROVENANCE,
        "UNIQUE_STRONG_PROVENANCE_EVIDENCE",
        ranked,
    )


def match_crop(
    crop: FeatureImage,
    originals: Iterable[FeatureImage],
    parameters: MatchingParameters,
    *,
    candidate_set_complete: bool = True,
) -> CropMatchResult:
    evidence = tuple(compare_candidate(crop, original, parameters) for original in originals)
    return decide_candidates(
        crop.reference,
        evidence,
        parameters,
        candidate_set_complete=candidate_set_complete,
    )


def match_crops(
    crops: Iterable[FeatureImage],
    originals: Iterable[FeatureImage],
    parameters: MatchingParameters,
    *,
    candidate_set_complete: bool = True,
) -> tuple[CropMatchResult, ...]:
    configure_opencv(parameters)
    stable_originals = tuple(
        sorted(originals, key=lambda item: semantic_reference_sort_key(item.reference))
    )
    stable_crops = tuple(
        sorted(crops, key=lambda item: semantic_reference_sort_key(item.reference))
    )
    return tuple(
        match_crop(
            crop,
            stable_originals,
            parameters,
            candidate_set_complete=candidate_set_complete,
        )
        for crop in stable_crops
    )


def finalize_candidate_set_completeness(
    results: Iterable[CropMatchResult],
    *,
    preprocessed_candidate_set_complete: bool,
) -> tuple[tuple[CropMatchResult, ...], bool]:
    """Apply run-wide abstention after all lazy candidate evaluations finish."""

    stable_results = tuple(results)
    candidate_set_complete = preprocessed_candidate_set_complete and all(
        candidate.evaluation_complete
        for result in stable_results
        for candidate in result.ranked_candidates
    )
    if candidate_set_complete:
        return stable_results, True

    finalized = tuple(
        CropMatchResult(
            crop=result.crop,
            decision=ContentDecision.AMBIGUOUS,
            provenance_interpretation=ProvenanceInterpretation.AMBIGUOUS_PROVENANCE,
            diagnostic_reason="INCOMPLETE_CANDIDATE_SET",
            ranked_candidates=result.ranked_candidates,
        )
        for result in stable_results
    )
    return finalized, False


def _keypoint_sort_key(keypoint: cv2.KeyPoint) -> tuple[float, ...]:
    return (
        round(float(keypoint.pt[1]), 6),
        round(float(keypoint.pt[0]), 6),
        round(float(keypoint.size), 6),
        round(float(keypoint.angle), 6),
        round(float(keypoint.response), 9),
        float(keypoint.octave),
        float(keypoint.class_id),
    )


def _ratio_matches(rows: Iterable[tuple[cv2.DMatch, ...]], threshold: float) -> tuple[cv2.DMatch, ...]:
    matches = [row[0] for row in rows if len(row) >= 2 and row[0].distance < threshold * row[1].distance]
    return tuple(sorted(matches, key=lambda match: (match.queryIdx, match.trainIdx, match.distance)))


def _spatial_coverage(points: np.ndarray, width: int, height: int) -> tuple[float, float]:
    if points.size == 0 or width <= 0 or height <= 0:
        return 0.0, 0.0
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    bbox_area = max(0.0, float(maximum[0] - minimum[0])) * max(0.0, float(maximum[1] - minimum[1]))
    bbox_coverage = min(1.0, bbox_area / (width * height))
    occupied: set[tuple[int, int]] = set()
    for x, y in points:
        column = min(GRID_SIZE - 1, max(0, int(float(x) * GRID_SIZE / width)))
        row = min(GRID_SIZE - 1, max(0, int(float(y) * GRID_SIZE / height)))
        occupied.add((column, row))
    return bbox_coverage, len(occupied) / (GRID_SIZE * GRID_SIZE)


def _projected_inside_fraction(points: np.ndarray, width: int, height: int) -> float:
    polygon = np.asarray(points, dtype=np.float32)
    area = abs(float(cv2.contourArea(polygon)))
    if area <= 0.0:
        return 0.0
    source = np.float32([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
    try:
        intersection_area, _ = cv2.intersectConvexConvex(polygon, source)
    except cv2.error:
        return 0.0
    return min(1.0, max(0.0, float(intersection_area) / area))


def _photometric_metrics(
    crop: np.ndarray,
    original: np.ndarray,
    transform: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    height, width = crop.shape
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    aligned = cv2.warpAffine(original, transform, (width, height), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    validity_source = np.full(original.shape, 255, dtype=np.uint8)
    valid = cv2.warpAffine(
        validity_source,
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    if int(valid.sum()) < max(16, int(width * height * 0.90)):
        return None, None, None

    crop_float = crop.astype(np.float32)
    aligned_float = aligned.astype(np.float32)
    luminance = _normalized_correlation(crop_float[valid], aligned_float[valid])
    residual = _normalized_residual(crop_float[valid], aligned_float[valid])
    crop_gradient = _gradient_magnitude(crop_float)
    aligned_gradient = _gradient_magnitude(aligned_float)
    gradient = _normalized_correlation(crop_gradient[valid], aligned_gradient[valid])
    return luminance, gradient, residual


def _decode_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as encoded:
        display = ImageOps.exif_transpose(encoded)
        grayscale = np.array(display.convert("L"), dtype=np.uint8, copy=True)
    grayscale = np.ascontiguousarray(grayscale)
    grayscale.setflags(write=False)
    return grayscale


def _grayscale_for_photometric_verification(
    image: FeatureImage,
) -> tuple[np.ndarray | None, str | None]:
    if image.grayscale is not None:
        return image.grayscale, None
    if image.source_path is None:
        return None, "SOURCE_PATH_UNAVAILABLE"
    try:
        return _decode_grayscale(image.source_path), None
    except Exception as exc:
        return None, type(exc).__name__


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    x_gradient = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    y_gradient = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(x_gradient, y_gradient)


def _normalized_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    first_centered = first.astype(np.float64) - float(np.mean(first))
    second_centered = second.astype(np.float64) - float(np.mean(second))
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator <= 1e-12:
        return None
    return float(np.clip(np.dot(first_centered, second_centered) / denominator, -1.0, 1.0))


def _normalized_residual(first: np.ndarray, second: np.ndarray) -> float | None:
    first_std = float(np.std(first))
    second_std = float(np.std(second))
    if first_std <= 1e-6 or second_std <= 1e-6:
        return None
    first_normalized = (first.astype(np.float64) - float(np.mean(first))) / first_std
    second_normalized = (second.astype(np.float64) - float(np.mean(second))) / second_std
    return float(np.sqrt(np.mean(np.square(first_normalized - second_normalized))) / 2.0)


def _refine_alignment(
    crop: np.ndarray,
    original: np.ndarray,
    transform: np.ndarray,
    radius: int,
) -> tuple[tuple[float | None, float | None, float | None], int, int]:
    choices: list[tuple[tuple[float, ...], tuple[float | None, float | None, float | None], int, int]] = []
    for shift_y in range(-radius, radius + 1):
        for shift_x in range(-radius, radius + 1):
            shifted = transform.copy()
            shifted[0, 2] += shift_x
            shifted[1, 2] += shift_y
            metrics = _photometric_metrics(crop, original, shifted)
            luma, gradient, residual = metrics
            key = (
                -(luma if luma is not None else -2.0),
                -(gradient if gradient is not None else -2.0),
                residual if residual is not None else 2.0,
                abs(shift_x) + abs(shift_y),
                shift_y,
                shift_x,
            )
            choices.append((key, metrics, shift_x, shift_y))
    _, metrics, shift_x, shift_y = min(choices, key=lambda choice: choice[0])
    return metrics, shift_x, shift_y


def _candidate_sort_key(candidate: CandidateEvidence) -> tuple[object, ...]:
    return (
        0 if candidate.strong_provenance else 1,
        0 if candidate.transform_plausible else 1,
        0 if candidate.transform_valid else 1,
        -(candidate.luminance_correlation if candidate.luminance_correlation is not None else -2.0),
        -(candidate.gradient_correlation if candidate.gradient_correlation is not None else -2.0),
        candidate.normalized_residual if candidate.normalized_residual is not None else 2.0,
        -candidate.inlier_count,
        -(candidate.inlier_ratio if candidate.inlier_ratio is not None else -1.0),
        -(candidate.bbox_coverage if candidate.bbox_coverage is not None else -1.0),
        candidate.reprojection_rmse if candidate.reprojection_rmse is not None else float("inf"),
        semantic_reference_sort_key(candidate.original),
    )


def _base_photometric_sort_key(candidate: CandidateEvidence) -> tuple[object, ...]:
    return (
        -(candidate.base_luminance_correlation if candidate.base_luminance_correlation is not None else -2.0),
        -(candidate.base_gradient_correlation if candidate.base_gradient_correlation is not None else -2.0),
        candidate.base_normalized_residual if candidate.base_normalized_residual is not None else 2.0,
        semantic_reference_sort_key(candidate.original),
    )


def _geometry_sort_key(candidate: CandidateEvidence) -> tuple[object, ...]:
    return (
        -candidate.inlier_count,
        -(candidate.inlier_ratio if candidate.inlier_ratio is not None else -1.0),
        -(candidate.bbox_coverage if candidate.bbox_coverage is not None else -1.0),
        -(candidate.grid_coverage if candidate.grid_coverage is not None else -1.0),
        candidate.reprojection_rmse if candidate.reprojection_rmse is not None else float("inf"),
        semantic_reference_sort_key(candidate.original),
    )


def _has_runner_up_separation(
    best: CandidateEvidence,
    second: CandidateEvidence,
    parameters: MatchingParameters,
) -> bool:
    if not second.transform_plausible:
        return True
    if any(
        value is None
        for value in (
            best.luminance_correlation,
            best.gradient_correlation,
            best.normalized_residual,
            second.luminance_correlation,
            second.gradient_correlation,
            second.normalized_residual,
        )
    ):
        return False
    assert best.luminance_correlation is not None
    assert best.gradient_correlation is not None
    assert best.normalized_residual is not None
    assert second.luminance_correlation is not None
    assert second.gradient_correlation is not None
    assert second.normalized_residual is not None
    return (
        best.luminance_correlation - second.luminance_correlation >= parameters.min_luminance_margin
        and best.gradient_correlation - second.gradient_correlation >= parameters.min_gradient_margin
        and second.normalized_residual - best.normalized_residual >= parameters.min_residual_margin
    )


def _invalid_evidence(
    *,
    crop: SemanticReference,
    original: SemanticReference,
    crop_keypoints: int,
    original_keypoints: int,
    diagnostic_reason: str,
    raw_knn_count: int = 0,
    ratio_passed_count: int = 0,
    reverse_ratio_passed_count: int = 0,
    mutual_match_count: int = 0,
    evaluation_complete: bool = True,
) -> CandidateEvidence:
    return CandidateEvidence(
        crop=crop,
        original=original,
        crop_keypoints=crop_keypoints,
        original_keypoints=original_keypoints,
        raw_knn_count=raw_knn_count,
        ratio_passed_count=ratio_passed_count,
        reverse_ratio_passed_count=reverse_ratio_passed_count,
        mutual_match_count=mutual_match_count,
        transform_valid=False,
        inlier_count=0,
        inlier_ratio=None,
        reprojection_rmse=None,
        reprojection_median=None,
        reprojection_p95=None,
        bbox_coverage=None,
        grid_coverage=None,
        scale=None,
        rotation_degrees=None,
        translation_x=None,
        translation_y=None,
        projected_corners=(),
        projected_inside_fraction=None,
        transform_plausible=False,
        base_luminance_correlation=None,
        base_gradient_correlation=None,
        base_normalized_residual=None,
        luminance_correlation=None,
        gradient_correlation=None,
        normalized_residual=None,
        alignment_refined=False,
        refinement_shift_x=0,
        refinement_shift_y=0,
        evaluation_complete=evaluation_complete,
        strong_provenance=False,
        diagnostic_reason=diagnostic_reason,
    )


def _geometry_failure_reason(
    inliers: int,
    inlier_ratio: float,
    rmse: float | None,
    bbox_coverage: float,
    grid_coverage: float,
    scale: float,
    rotation: float,
    inside: float,
    parameters: MatchingParameters,
) -> str:
    if inliers < parameters.min_inliers:
        return "INSUFFICIENT_RANSAC_INLIERS"
    if inlier_ratio < parameters.min_inlier_ratio:
        return "INSUFFICIENT_RANSAC_INLIER_RATIO"
    if rmse is None or rmse > parameters.max_reprojection_rmse:
        return "EXCESSIVE_REPROJECTION_ERROR"
    if bbox_coverage < parameters.min_bbox_coverage or grid_coverage < parameters.min_grid_coverage:
        return "INSUFFICIENT_INLIER_SPATIAL_COVERAGE"
    if not parameters.min_scale <= scale <= parameters.max_scale:
        return "IMPLAUSIBLE_SCALE"
    if abs(rotation) > parameters.max_abs_rotation_degrees:
        return "IMPLAUSIBLE_ROTATION"
    if inside < parameters.min_projected_inside_fraction:
        return "PROJECTED_CROP_OUTSIDE_SOURCE"
    return "IMPLAUSIBLE_CROP_TRANSFORM"


def _photometric_failure_reason(
    luminance: float | None,
    gradient: float | None,
    residual: float | None,
    parameters: MatchingParameters,
) -> str:
    if luminance is None or gradient is None or residual is None:
        return "PHOTOMETRIC_METRICS_UNAVAILABLE"
    if luminance < parameters.min_luminance_correlation:
        return "WEAK_LUMINANCE_CORRELATION"
    if gradient < parameters.min_gradient_correlation:
        return "WEAK_GRADIENT_CORRELATION"
    if residual > parameters.max_normalized_residual:
        return "EXCESSIVE_NORMALIZED_RESIDUAL"
    return "CONFLICTING_PHOTOMETRIC_EVIDENCE"

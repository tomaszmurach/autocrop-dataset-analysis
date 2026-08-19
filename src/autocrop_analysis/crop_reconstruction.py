"""Deterministic crop reconstruction from content-provenance schema 1.1."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields
from enum import Enum
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

from . import __version__


SUPPORTED_PROVENANCE_SCHEMA_VERSION = "1.1"
RECONSTRUCTION_SCHEMA_VERSION = "1.0"
EXPECTED_ALGORITHM_NAME = "SIFT_BF_L2_MUTUAL_SIMILARITY_RANSAC_ALIGNED_PHOTOMETRIC"
EXPECTED_ALGORITHM_STATUS = "EXPERIMENTAL_UNLABELED_PROVENANCE_DISCOVERY"
EXPECTED_GEOMETRIC_MODEL = "estimateAffinePartial2D_RANSAC_crop_to_original_display"
EXPECTED_PROJECTED_CORNER_ORDER = (
    "TOP_LEFT",
    "TOP_RIGHT",
    "BOTTOM_RIGHT",
    "BOTTOM_LEFT",
)
COORDINATE_SYSTEM = "EXIF_NORMALIZED_DISPLAY_PIXEL_BOUNDARIES"


class ReconstructionStatus(str, Enum):
    RECONSTRUCTED = "RECONSTRUCTED"
    NOT_RECONSTRUCTED = "NOT_RECONSTRUCTED"


class ReconstructionReason(str, Enum):
    UPSTREAM_PROVENANCE_NOT_MATCHED = "UPSTREAM_PROVENANCE_NOT_MATCHED"
    DEGENERATE_GEOMETRY = "DEGENERATE_GEOMETRY"
    INCONSISTENT_GEOMETRY = "INCONSISTENT_GEOMETRY"
    NON_AXIS_ALIGNED_GEOMETRY = "NON_AXIS_ALIGNED_GEOMETRY"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"


class ManifestValidationError(ValueError):
    """A stable, privacy-safe failure code for invalid provenance input."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ReconstructionParameters:
    max_abs_rotation_degrees: float = 1.0
    numeric_relative_tolerance: float = 1e-5
    numeric_absolute_tolerance: float = 1e-4
    transform_relative_tolerance: float = 1e-6
    transform_absolute_tolerance: float = 1e-4

    def as_dict(self) -> dict[str, float]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class ImageReference:
    root_role: str
    relative_path: str
    display_width: int | None
    display_height: int | None

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "root_role": self.root_role,
            "relative_path": self.relative_path,
            "display_width": self.display_width,
            "display_height": self.display_height,
        }

    @property
    def has_dimensions(self) -> bool:
        return self.display_width is not None and self.display_height is not None


@dataclass(frozen=True, slots=True)
class BaseTransform:
    scale: float
    rotation_degrees: float
    translation_x: float
    translation_y: float

    def as_dict(self) -> dict[str, float]:
        return {
            "scale": self.scale,
            "rotation_degrees": self.rotation_degrees,
            "translation_x": self.translation_x,
            "translation_y": self.translation_y,
        }


Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class ValidatedCandidate:
    rank: int
    original: ImageReference
    evaluation_complete: bool
    strong_provenance: bool
    transform_valid: bool
    transform_plausible: bool
    transform: BaseTransform | None
    projected_corners: tuple[Point, ...]
    projected_inside_fraction: float | None


@dataclass(frozen=True, slots=True)
class ValidatedCrop:
    crop: ImageReference
    decision: str
    provenance_interpretation: str
    ranked_candidates: tuple[ValidatedCandidate, ...]


@dataclass(frozen=True, slots=True)
class ValidatedProvenanceManifest:
    tool_version: str
    algorithm_name: str
    candidate_set_complete: bool
    crops: tuple[ValidatedCrop, ...]


@dataclass(frozen=True, slots=True)
class Rectangle:
    left: float
    top: float
    right: float
    bottom: float
    width: float
    height: float
    center_x: float
    center_y: float
    aspect_ratio: float

    def as_dict(self) -> dict[str, float]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "aspect_ratio": self.aspect_ratio,
        }


@dataclass(frozen=True, slots=True)
class GeometryDiagnostics:
    quadrilateral_area: float
    top_width: float
    bottom_width: float
    left_height: float
    right_height: float
    opposite_width_relative_error: float
    opposite_height_relative_error: float
    maximum_parallel_error: float
    maximum_perpendicular_error: float
    serialized_rotation_degrees: float
    measured_rotation_degrees: float
    rotation_difference_degrees: float
    transform_corner_max_error: float
    bounds_excursion: float
    inside_bounds: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class GeometryResult:
    status: ReconstructionStatus
    reason: ReconstructionReason | None
    rectangle: Rectangle | None
    diagnostics: GeometryDiagnostics


def parse_provenance_bytes(raw_bytes: bytes) -> Mapping[str, object]:
    """Decode and parse one already-read provenance byte sequence strictly."""

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestValidationError("INVALID_UTF8") from exc
    try:
        value = json.loads(
            text,
            parse_constant=_reject_non_finite_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except ManifestValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise ManifestValidationError("INVALID_JSON") from exc
    _reject_non_finite_json_values(value)
    if not isinstance(value, dict):
        raise ManifestValidationError("TOP_LEVEL_NOT_OBJECT")
    return value


def validate_provenance_manifest(
    manifest: Mapping[str, object],
    parameters: ReconstructionParameters | None = None,
) -> ValidatedProvenanceManifest:
    """Validate only the schema 1.1 contract needed for reconstruction."""

    active_parameters = parameters or ReconstructionParameters()
    if _required_string(manifest, "schema_version") != SUPPORTED_PROVENANCE_SCHEMA_VERSION:
        raise ManifestValidationError("UNSUPPORTED_SCHEMA_VERSION")
    tool_version = _required_string(manifest, "tool_version")
    algorithm = _required_mapping(manifest, "algorithm")
    algorithm_name = _required_string(algorithm, "name")
    if algorithm_name != EXPECTED_ALGORITHM_NAME:
        raise ManifestValidationError("UNSUPPORTED_ALGORITHM_NAME")
    if _required_string(algorithm, "status") != EXPECTED_ALGORITHM_STATUS:
        raise ManifestValidationError("UNSUPPORTED_ALGORITHM_STATUS")
    if _required_string(algorithm, "geometric_model") != EXPECTED_GEOMETRIC_MODEL:
        raise ManifestValidationError("UNSUPPORTED_GEOMETRIC_MODEL")
    corner_order = _required_list(algorithm, "projected_corner_order")
    if tuple(corner_order) != EXPECTED_PROJECTED_CORNER_ORDER:
        raise ManifestValidationError("UNSUPPORTED_PROJECTED_CORNER_ORDER")

    summary = _required_mapping(manifest, "summary")
    candidate_set_complete = _required_bool(summary, "candidate_set_complete")
    crops_data = _required_list(manifest, "crops")
    crops: list[ValidatedCrop] = []
    seen_crops: set[tuple[str, str]] = set()
    for crop_data in crops_data:
        crop = _parse_crop(_as_mapping(crop_data, "INVALID_CROP_RECORD"))
        crop_key = (crop.crop.root_role, crop.crop.relative_path)
        if crop_key in seen_crops:
            raise ManifestValidationError("DUPLICATE_CROP_REFERENCE")
        seen_crops.add(crop_key)
        crops.append(crop)

    _validate_summary(summary, crops, candidate_set_complete)
    if not candidate_set_complete and any(crop.decision == "MATCHED" for crop in crops):
        raise ManifestValidationError("MATCHED_WITH_INCOMPLETE_CANDIDATE_SET")
    if candidate_set_complete and any(
        not candidate.evaluation_complete
        for crop in crops
        for candidate in crop.ranked_candidates
    ):
        raise ManifestValidationError("COMPLETE_SET_WITH_INCOMPLETE_EVIDENCE")

    for crop in crops:
        if crop.decision == "MATCHED":
            _validate_matched_state(crop, candidate_set_complete, active_parameters)

    stable_crops = tuple(sorted(crops, key=lambda item: _reference_sort_key(item.crop)))
    return ValidatedProvenanceManifest(
        tool_version=tool_version,
        algorithm_name=algorithm_name,
        candidate_set_complete=candidate_set_complete,
        crops=stable_crops,
    )


def reconstruct_manifest(
    provenance: ValidatedProvenanceManifest,
    *,
    source_path: Path,
    source_sha256: str,
    parameters: ReconstructionParameters | None = None,
) -> dict[str, object]:
    """Build deterministic reconstruction data from validated provenance."""

    active_parameters = parameters or ReconstructionParameters()
    items = [_reconstruct_crop(crop, provenance, active_parameters) for crop in provenance.crops]
    status_counts = Counter(item["status"] for item in items)
    reason_counts = Counter(
        item["reason"] for item in items if item["reason"] is not None
    )
    return {
        "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
        "tool_version": __version__,
        "source_provenance_manifest": {
            "path": str(source_path),
            "sha256": source_sha256,
            "schema_version": SUPPORTED_PROVENANCE_SCHEMA_VERSION,
            "tool_version": provenance.tool_version,
            "algorithm_name": provenance.algorithm_name,
        },
        "coordinate_system": COORDINATE_SYSTEM,
        "rectangle_convention": {
            "authoritative_representation": "FLOAT",
            "interval": "HALF_OPEN",
            "axis_order": ["left", "top", "right", "bottom"],
        },
        "parameters": active_parameters.as_dict(),
        "summary": {
            "items": len(items),
            ReconstructionStatus.RECONSTRUCTED.value: status_counts[
                ReconstructionStatus.RECONSTRUCTED.value
            ],
            ReconstructionStatus.NOT_RECONSTRUCTED.value: status_counts[
                ReconstructionStatus.NOT_RECONSTRUCTED.value
            ],
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "items": items,
    }


def reconstruct_geometry(
    projected_corners: Sequence[Point],
    transform: BaseTransform,
    *,
    crop_width: int,
    crop_height: int,
    original_width: int,
    original_height: int,
    parameters: ReconstructionParameters | None = None,
) -> GeometryResult:
    """Validate projected geometry and derive its canonical axis-aligned crop."""

    active = parameters or ReconstructionParameters()
    corners = tuple((float(point[0]), float(point[1])) for point in projected_corners)
    if len(corners) != 4 or any(not math.isfinite(value) for point in corners for value in point):
        raise ValueError("projected_corners must contain four finite points")
    tl, tr, br, bl = corners
    top = _subtract(tr, tl)
    bottom = _subtract(br, bl)
    left = _subtract(bl, tl)
    right = _subtract(br, tr)
    polygon_edges = (top, right, _subtract(bl, br), _subtract(tl, bl))
    top_width = _length(top)
    bottom_width = _length(bottom)
    left_height = _length(left)
    right_height = _length(right)
    area = _quadrilateral_area(corners)
    center_x = sum(point[0] for point in corners) / 4.0
    center_y = sum(point[1] for point in corners) / 4.0
    width = (top_width + bottom_width) / 2.0
    height = (left_height + right_height) / 2.0
    rectangle = None
    if height > 0.0:
        rectangle = Rectangle(
            left=center_x - width / 2.0,
            top=center_y - height / 2.0,
            right=center_x + width / 2.0,
            bottom=center_y + height / 2.0,
            width=width,
            height=height,
            center_x=center_x,
            center_y=center_y,
            aspect_ratio=width / height,
        )

    opposite_width_error = _relative_difference(top_width, bottom_width)
    opposite_height_error = _relative_difference(left_height, right_height)
    parallel_errors = (
        _normalized_cross_error(top, bottom),
        _normalized_cross_error(left, right),
    )
    perpendicular_errors = (
        _normalized_dot_error(top, left),
        _normalized_dot_error(top, right),
        _normalized_dot_error(bottom, left),
        _normalized_dot_error(bottom, right),
    )
    measured_rotation = math.degrees(math.atan2(top[1], top[0])) if top_width > 0.0 else 0.0
    rotation_difference = abs(_angle_difference(measured_rotation, transform.rotation_degrees))
    expected_corners = project_corners(
        transform,
        crop_width=crop_width,
        crop_height=crop_height,
    )
    transform_corner_max_error = max(
        math.hypot(actual[0] - expected[0], actual[1] - expected[1])
        for actual, expected in zip(corners, expected_corners, strict=True)
    )
    bounds_excursion = _bounds_excursion(
        corners,
        rectangle,
        original_width=original_width,
        original_height=original_height,
    )
    projected_inside = all(
        -active.numeric_absolute_tolerance <= x <= original_width + active.numeric_absolute_tolerance
        and -active.numeric_absolute_tolerance <= y <= original_height + active.numeric_absolute_tolerance
        for x, y in corners
    )
    rectangle_inside = rectangle is not None and (
        0.0 <= rectangle.left < rectangle.right <= original_width
        and 0.0 <= rectangle.top < rectangle.bottom <= original_height
    )
    inside_bounds = projected_inside and rectangle_inside
    diagnostics = GeometryDiagnostics(
        quadrilateral_area=area,
        top_width=top_width,
        bottom_width=bottom_width,
        left_height=left_height,
        right_height=right_height,
        opposite_width_relative_error=opposite_width_error,
        opposite_height_relative_error=opposite_height_error,
        maximum_parallel_error=max(parallel_errors),
        maximum_perpendicular_error=max(perpendicular_errors),
        serialized_rotation_degrees=transform.rotation_degrees,
        measured_rotation_degrees=measured_rotation,
        rotation_difference_degrees=rotation_difference,
        transform_corner_max_error=transform_corner_max_error,
        bounds_excursion=bounds_excursion,
        inside_bounds=inside_bounds,
    )

    side_lengths = (top_width, bottom_width, left_height, right_height)
    if (
        any(length <= active.numeric_absolute_tolerance for length in side_lengths)
        or area <= active.numeric_absolute_tolerance**2
        or rectangle is None
        or rectangle.width <= 0.0
        or rectangle.height <= 0.0
    ):
        return GeometryResult(
            ReconstructionStatus.NOT_RECONSTRUCTED,
            ReconstructionReason.DEGENERATE_GEOMETRY,
            None,
            diagnostics,
        )

    winding = tuple(
        _normalized_cross_error(first, second, signed=True)
        for first, second in zip(polygon_edges, polygon_edges[1:] + polygon_edges[:1], strict=True)
    )
    consistent_winding = all(value > active.numeric_relative_tolerance for value in winding) or all(
        value < -active.numeric_relative_tolerance for value in winding
    )
    transform_consistent = all(
        math.isclose(
            actual_value,
            expected_value,
            rel_tol=active.transform_relative_tolerance,
            abs_tol=active.transform_absolute_tolerance,
        )
        for actual, expected in zip(corners, expected_corners, strict=True)
        for actual_value, expected_value in zip(actual, expected, strict=True)
    )
    expected_width = transform.scale * crop_width
    expected_height = transform.scale * crop_height
    size_consistent = math.isclose(
        width,
        expected_width,
        rel_tol=active.numeric_relative_tolerance,
        abs_tol=active.numeric_absolute_tolerance,
    ) and math.isclose(
        height,
        expected_height,
        rel_tol=active.numeric_relative_tolerance,
        abs_tol=active.numeric_absolute_tolerance,
    )
    rotation_tolerance = math.degrees(active.numeric_relative_tolerance)
    consistent = (
        consistent_winding
        and max(parallel_errors) <= active.numeric_relative_tolerance
        and max(perpendicular_errors) <= active.numeric_relative_tolerance
        and opposite_width_error <= active.numeric_relative_tolerance
        and opposite_height_error <= active.numeric_relative_tolerance
        and size_consistent
        and rotation_difference <= rotation_tolerance
        and transform_consistent
    )
    if not consistent:
        return GeometryResult(
            ReconstructionStatus.NOT_RECONSTRUCTED,
            ReconstructionReason.INCONSISTENT_GEOMETRY,
            None,
            diagnostics,
        )
    if abs(transform.rotation_degrees) > active.max_abs_rotation_degrees:
        return GeometryResult(
            ReconstructionStatus.NOT_RECONSTRUCTED,
            ReconstructionReason.NON_AXIS_ALIGNED_GEOMETRY,
            None,
            diagnostics,
        )
    if not inside_bounds:
        return GeometryResult(
            ReconstructionStatus.NOT_RECONSTRUCTED,
            ReconstructionReason.OUT_OF_BOUNDS,
            None,
            diagnostics,
        )
    return GeometryResult(ReconstructionStatus.RECONSTRUCTED, None, rectangle, diagnostics)


def project_corners(
    transform: BaseTransform,
    *,
    crop_width: int,
    crop_height: int,
) -> tuple[Point, Point, Point, Point]:
    """Apply OpenCV's estimateAffinePartial2D crop-to-original convention."""

    theta = math.radians(transform.rotation_degrees)
    a = transform.scale * math.cos(theta)
    b = transform.scale * math.sin(theta)

    def project(x: float, y: float) -> Point:
        return (
            a * x - b * y + transform.translation_x,
            b * x + a * y + transform.translation_y,
        )

    return (
        project(0.0, 0.0),
        project(float(crop_width), 0.0),
        project(float(crop_width), float(crop_height)),
        project(0.0, float(crop_height)),
    )


def _reconstruct_crop(
    crop: ValidatedCrop,
    provenance: ValidatedProvenanceManifest,
    parameters: ReconstructionParameters,
) -> dict[str, object]:
    base: dict[str, object] = {
        "crop": crop.crop.as_dict(),
        "upstream_decision": crop.decision,
        "upstream_provenance_interpretation": crop.provenance_interpretation,
    }
    if crop.decision != "MATCHED" or not provenance.candidate_set_complete:
        return {
            **base,
            "selected_original": None,
            "status": ReconstructionStatus.NOT_RECONSTRUCTED.value,
            "reason": ReconstructionReason.UPSTREAM_PROVENANCE_NOT_MATCHED.value,
            "projected_corners": None,
            "base_transform": None,
            "rectangle": None,
            "validation": None,
        }

    candidate = crop.ranked_candidates[0]
    assert candidate.transform is not None
    assert crop.crop.display_width is not None
    assert crop.crop.display_height is not None
    assert candidate.original.display_width is not None
    assert candidate.original.display_height is not None
    geometry = reconstruct_geometry(
        candidate.projected_corners,
        candidate.transform,
        crop_width=crop.crop.display_width,
        crop_height=crop.crop.display_height,
        original_width=candidate.original.display_width,
        original_height=candidate.original.display_height,
        parameters=parameters,
    )
    return {
        **base,
        "selected_original": candidate.original.as_dict(),
        "status": geometry.status.value,
        "reason": geometry.reason.value if geometry.reason is not None else None,
        "projected_corners": [list(point) for point in candidate.projected_corners],
        "base_transform": candidate.transform.as_dict(),
        "rectangle": geometry.rectangle.as_dict() if geometry.rectangle is not None else None,
        "validation": geometry.diagnostics.as_dict(),
    }


def _parse_crop(data: Mapping[str, object]) -> ValidatedCrop:
    crop_reference = _parse_reference(_required_mapping(data, "crop"), expected_role="CROPPED")
    decision = _required_string(data, "decision")
    interpretation = _required_string(data, "provenance_interpretation")
    expected_pairs = {
        "MATCHED": "UNIQUE_STRONG_PROVENANCE",
        "AMBIGUOUS": "AMBIGUOUS_PROVENANCE",
        "NO_MATCH": "NO_VALID_PROVENANCE",
    }
    if decision not in expected_pairs:
        raise ManifestValidationError("UNKNOWN_DECISION")
    if interpretation not in expected_pairs.values():
        raise ManifestValidationError("UNKNOWN_PROVENANCE_INTERPRETATION")
    if expected_pairs[decision] != interpretation:
        raise ManifestValidationError("INCONSISTENT_DECISION_INTERPRETATION")
    candidate_data = _required_list(data, "ranked_candidates")
    candidates = tuple(
        _parse_candidate(_as_mapping(value, "INVALID_CANDIDATE_RECORD"))
        for value in candidate_data
    )
    ranks = [candidate.rank for candidate in candidates]
    if ranks != list(range(1, len(candidates) + 1)):
        raise ManifestValidationError("INVALID_CANDIDATE_RANKS")
    original_keys = [
        (candidate.original.root_role, candidate.original.relative_path)
        for candidate in candidates
    ]
    if len(set(original_keys)) != len(original_keys):
        raise ManifestValidationError("DUPLICATE_CANDIDATE_REFERENCE")
    return ValidatedCrop(crop_reference, decision, interpretation, candidates)


def _parse_candidate(data: Mapping[str, object]) -> ValidatedCandidate:
    rank = _required_positive_int(data, "rank")
    original = _parse_reference(_required_mapping(data, "original"), expected_role="ORIGINAL")
    evaluation_complete = _required_bool(data, "evaluation_complete")
    strong_provenance = _required_bool(data, "strong_provenance")
    geometric = _required_mapping(data, "geometric_evidence")
    transform_valid = _required_bool(geometric, "transform_valid")
    transform_data = _required_mapping(data, "transform_evidence")
    plausible = _required_bool(transform_data, "plausible")
    projected = _required_list(transform_data, "projected_corners")
    transform: BaseTransform | None
    projected_corners: tuple[Point, ...]
    projected_inside_fraction: float | None
    if transform_valid:
        transform = BaseTransform(
            scale=_required_finite_number(transform_data, "scale"),
            rotation_degrees=_required_finite_number(transform_data, "rotation_degrees"),
            translation_x=_required_finite_number(transform_data, "translation_x"),
            translation_y=_required_finite_number(transform_data, "translation_y"),
        )
        if len(projected) != 4:
            raise ManifestValidationError("MALFORMED_PROJECTED_CORNERS")
        projected_corners = tuple(_parse_point(point) for point in projected)
        projected_inside_fraction = _required_finite_number(
            transform_data, "projected_inside_fraction"
        )
        if not 0.0 <= projected_inside_fraction <= 1.0:
            raise ManifestValidationError("INVALID_PROJECTED_INSIDE_FRACTION")
    else:
        transform = None
        projected_corners = ()
        projected_inside_fraction = None
        if plausible or projected:
            raise ManifestValidationError("CONTRADICTORY_INVALID_TRANSFORM")
        for key in (
            "scale",
            "rotation_degrees",
            "translation_x",
            "translation_y",
            "projected_inside_fraction",
        ):
            if key not in transform_data or transform_data[key] is not None:
                raise ManifestValidationError("CONTRADICTORY_INVALID_TRANSFORM")
    if strong_provenance and (not evaluation_complete or not transform_valid or not plausible):
        raise ManifestValidationError("CONTRADICTORY_STRONG_PROVENANCE")
    return ValidatedCandidate(
        rank=rank,
        original=original,
        evaluation_complete=evaluation_complete,
        strong_provenance=strong_provenance,
        transform_valid=transform_valid,
        transform_plausible=plausible,
        transform=transform,
        projected_corners=projected_corners,
        projected_inside_fraction=projected_inside_fraction,
    )


def _parse_reference(data: Mapping[str, object], *, expected_role: str) -> ImageReference:
    role = _required_string(data, "root_role")
    if role != expected_role:
        raise ManifestValidationError("INVALID_REFERENCE_ROLE")
    relative_path = _required_string(data, "relative_path")
    _validate_relative_path(relative_path)
    width = _dimension(data, "display_width")
    height = _dimension(data, "display_height")
    if (width is None) != (height is None):
        raise ManifestValidationError("PARTIAL_NULL_DIMENSIONS")
    return ImageReference(role, relative_path, width, height)


def _validate_matched_state(
    crop: ValidatedCrop,
    candidate_set_complete: bool,
    parameters: ReconstructionParameters,
) -> None:
    if not candidate_set_complete:
        raise ManifestValidationError("MATCHED_WITH_INCOMPLETE_CANDIDATE_SET")
    if not crop.ranked_candidates:
        raise ManifestValidationError("MATCHED_WITHOUT_RANK_ONE")
    candidate = crop.ranked_candidates[0]
    if not crop.crop.has_dimensions or not candidate.original.has_dimensions:
        raise ManifestValidationError("MATCHED_WITHOUT_DISPLAY_DIMENSIONS")
    if not (
        candidate.evaluation_complete
        and candidate.strong_provenance
        and candidate.transform_valid
        and candidate.transform_plausible
        and candidate.transform is not None
        and len(candidate.projected_corners) == 4
    ):
        raise ManifestValidationError("CONTRADICTORY_MATCHED_RANK_ONE")
    assert crop.crop.display_width is not None
    assert crop.crop.display_height is not None
    expected = project_corners(
        candidate.transform,
        crop_width=crop.crop.display_width,
        crop_height=crop.crop.display_height,
    )
    for actual_point, expected_point in zip(
        candidate.projected_corners, expected, strict=True
    ):
        for actual, expected_value in zip(actual_point, expected_point, strict=True):
            if not math.isclose(
                actual,
                expected_value,
                rel_tol=parameters.transform_relative_tolerance,
                abs_tol=parameters.transform_absolute_tolerance,
            ):
                raise ManifestValidationError("TRANSFORM_CORNER_CONTRADICTION")


def _validate_summary(
    summary: Mapping[str, object],
    crops: Sequence[ValidatedCrop],
    candidate_set_complete: bool,
) -> None:
    originals_analyzed = _required_nonnegative_int(summary, "originals_analyzed")
    crops_analyzed = _required_nonnegative_int(summary, "crops_analyzed")
    if crops_analyzed != len(crops):
        raise ManifestValidationError("INCONSISTENT_CROP_SUMMARY")
    if any(len(crop.ranked_candidates) != originals_analyzed for crop in crops):
        raise ManifestValidationError("INCONSISTENT_CANDIDATE_COUNT")
    expected_comparisons = originals_analyzed * crops_analyzed
    if (
        _required_nonnegative_int(summary, "candidate_comparisons")
        != expected_comparisons
    ):
        raise ManifestValidationError("INCONSISTENT_COMPARISON_SUMMARY")

    candidate_universe: frozenset[tuple[str, str]] | None = None
    for crop in crops:
        crop_universe = frozenset(
            (candidate.original.root_role, candidate.original.relative_path)
            for candidate in crop.ranked_candidates
        )
        if candidate_universe is None:
            candidate_universe = crop_universe
        elif crop_universe != candidate_universe:
            raise ManifestValidationError("INCONSISTENT_CANDIDATE_UNIVERSE")

    supplied = _required_nonnegative_int(
        summary, "supplied_original_image_candidates"
    )
    audit_readable = _required_nonnegative_int(
        summary, "audit_readable_original_candidates"
    )
    audit_unreadable = _required_nonnegative_int(
        summary, "audit_unreadable_original_candidates"
    )
    audit_unsupported = _required_nonnegative_int(
        summary, "audit_unsupported_original_candidates"
    )
    audit_filesystem_error = _required_nonnegative_int(
        summary, "audit_filesystem_error_original_candidates"
    )
    audit_unavailable = _required_nonnegative_int(
        summary, "audit_unavailable_original_candidates"
    )
    feature_unavailable = _required_nonnegative_int(
        summary, "feature_extraction_unavailable_originals"
    )
    photometric_unavailable = _required_nonnegative_int(
        summary, "photometric_decode_unavailable_originals"
    )
    successfully_evaluated = _required_nonnegative_int(
        summary, "successfully_evaluated_originals"
    )

    expected_audit_unavailable = (
        audit_unreadable + audit_unsupported + audit_filesystem_error
    )
    if audit_unavailable != expected_audit_unavailable:
        raise ManifestValidationError("INCONSISTENT_ORIGINAL_AVAILABILITY_SUMMARY")
    if supplied != audit_readable + audit_unavailable:
        raise ManifestValidationError("INCONSISTENT_ORIGINAL_AVAILABILITY_SUMMARY")
    if originals_analyzed != audit_readable:
        raise ManifestValidationError("INCONSISTENT_ORIGINAL_AVAILABILITY_SUMMARY")
    if feature_unavailable > originals_analyzed:
        raise ManifestValidationError("INCONSISTENT_ORIGINAL_EVALUATION_SUMMARY")
    successfully_preprocessed = originals_analyzed - feature_unavailable
    if photometric_unavailable > successfully_preprocessed:
        raise ManifestValidationError("INCONSISTENT_ORIGINAL_EVALUATION_SUMMARY")
    expected_successfully_evaluated = (
        successfully_preprocessed - photometric_unavailable
    )
    if successfully_evaluated != expected_successfully_evaluated:
        raise ManifestValidationError("INCONSISTENT_ORIGINAL_EVALUATION_SUMMARY")
    expected_candidate_set_complete = supplied == successfully_evaluated
    if candidate_set_complete is not expected_candidate_set_complete:
        raise ManifestValidationError("INCONSISTENT_CANDIDATE_SET_COMPLETENESS")

    decisions = Counter(crop.decision for crop in crops)
    interpretations = Counter(crop.provenance_interpretation for crop in crops)
    for key in ("MATCHED", "AMBIGUOUS", "NO_MATCH"):
        if _required_nonnegative_int(summary, key) != decisions[key]:
            raise ManifestValidationError("INCONSISTENT_DECISION_SUMMARY")
    for key in (
        "UNIQUE_STRONG_PROVENANCE",
        "AMBIGUOUS_PROVENANCE",
        "NO_VALID_PROVENANCE",
    ):
        if _required_nonnegative_int(summary, key) != interpretations[key]:
            raise ManifestValidationError("INCONSISTENT_INTERPRETATION_SUMMARY")


def _validate_relative_path(value: str) -> None:
    if not value or "\\" in value or "\x00" in value:
        raise ManifestValidationError("UNSAFE_RELATIVE_PATH")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ManifestValidationError("UNSAFE_RELATIVE_PATH")
    posix_path = PurePosixPath(value)
    if posix_path.is_absolute() or posix_path.as_posix() != value:
        raise ManifestValidationError("UNSAFE_RELATIVE_PATH")
    if PureWindowsPath(value).drive:
        raise ManifestValidationError("UNSAFE_RELATIVE_PATH")


def _parse_point(value: object) -> Point:
    if not isinstance(value, list) or len(value) != 2:
        raise ManifestValidationError("MALFORMED_PROJECTED_CORNERS")
    return (_finite_number(value[0]), _finite_number(value[1]))


def _dimension(data: Mapping[str, object], key: str) -> int | None:
    if key not in data:
        raise ManifestValidationError("MISSING_REQUIRED_FIELD")
    value = data[key]
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ManifestValidationError("INVALID_DISPLAY_DIMENSION")
    return value


def _required_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    if key not in data:
        raise ManifestValidationError("MISSING_REQUIRED_FIELD")
    return _as_mapping(data[key], "INVALID_FIELD_TYPE")


def _as_mapping(value: object, code: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ManifestValidationError(code)
    return value


def _required_list(data: Mapping[str, object], key: str) -> list[object]:
    if key not in data:
        raise ManifestValidationError("MISSING_REQUIRED_FIELD")
    value = data[key]
    if not isinstance(value, list):
        raise ManifestValidationError("INVALID_FIELD_TYPE")
    return value


def _required_string(data: Mapping[str, object], key: str) -> str:
    if key not in data:
        raise ManifestValidationError("MISSING_REQUIRED_FIELD")
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ManifestValidationError("INVALID_FIELD_TYPE")
    return value


def _required_bool(data: Mapping[str, object], key: str) -> bool:
    if key not in data:
        raise ManifestValidationError("MISSING_REQUIRED_FIELD")
    value = data[key]
    if type(value) is not bool:
        raise ManifestValidationError("INVALID_FIELD_TYPE")
    return value


def _required_positive_int(data: Mapping[str, object], key: str) -> int:
    value = _required_nonnegative_int(data, key)
    if value <= 0:
        raise ManifestValidationError("INVALID_POSITIVE_INTEGER")
    return value


def _required_nonnegative_int(data: Mapping[str, object], key: str) -> int:
    if key not in data:
        raise ManifestValidationError("MISSING_REQUIRED_FIELD")
    value = data[key]
    if type(value) is not int or value < 0:
        raise ManifestValidationError("INVALID_NONNEGATIVE_INTEGER")
    return value


def _required_finite_number(data: Mapping[str, object], key: str) -> float:
    if key not in data:
        raise ManifestValidationError("MISSING_REQUIRED_FIELD")
    return _finite_number(data[key])


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestValidationError("INVALID_NUMERIC_VALUE")
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ManifestValidationError("NON_FINITE_NUMERIC_VALUE") from exc
    if not math.isfinite(converted):
        raise ManifestValidationError("NON_FINITE_NUMERIC_VALUE")
    return converted


def _reject_non_finite_json_constant(value: str) -> None:
    raise ManifestValidationError("NON_FINITE_JSON_NUMBER")


def _reject_non_finite_json_values(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ManifestValidationError("NON_FINITE_JSON_NUMBER")
    if isinstance(value, dict):
        for nested in value.values():
            _reject_non_finite_json_values(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_non_finite_json_values(nested)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestValidationError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _reference_sort_key(reference: ImageReference) -> tuple[str, str, str]:
    return reference.root_role, reference.relative_path.casefold(), reference.relative_path


def _subtract(first: Point, second: Point) -> Point:
    return first[0] - second[0], first[1] - second[1]


def _length(vector: Point) -> float:
    return math.hypot(vector[0], vector[1])


def _cross(first: Point, second: Point) -> float:
    return first[0] * second[1] - first[1] * second[0]


def _dot(first: Point, second: Point) -> float:
    return first[0] * second[0] + first[1] * second[1]


def _normalized_cross_error(first: Point, second: Point, *, signed: bool = False) -> float:
    denominator = _length(first) * _length(second)
    if denominator == 0.0:
        return 0.0
    value = _cross(first, second) / denominator
    return value if signed else abs(value)


def _normalized_dot_error(first: Point, second: Point) -> float:
    denominator = _length(first) * _length(second)
    if denominator == 0.0:
        return 0.0
    return abs(_dot(first, second) / denominator)


def _relative_difference(first: float, second: float) -> float:
    denominator = max(first, second)
    if denominator == 0.0:
        return 0.0
    return abs(first - second) / denominator


def _quadrilateral_area(corners: Sequence[Point]) -> float:
    return abs(
        sum(
            point[0] * following[1] - following[0] * point[1]
            for point, following in zip(corners, corners[1:] + corners[:1], strict=True)
        )
    ) / 2.0


def _angle_difference(first: float, second: float) -> float:
    return (first - second + 180.0) % 360.0 - 180.0


def _bounds_excursion(
    corners: Sequence[Point],
    rectangle: Rectangle | None,
    *,
    original_width: int,
    original_height: int,
) -> float:
    excursions = [
        max(0.0, -x, x - original_width, -y, y - original_height)
        for x, y in corners
    ]
    if rectangle is not None:
        excursions.extend(
            (
                max(0.0, -rectangle.left),
                max(0.0, -rectangle.top),
                max(0.0, rectangle.right - original_width),
                max(0.0, rectangle.bottom - original_height),
            )
        )
    return max(excursions, default=0.0)

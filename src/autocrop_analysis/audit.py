"""Read-only, deterministic inventory and metadata inspection for image datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Iterable

from PIL import Image, UnidentifiedImageError


EXIF_ORIENTATION_TAG = 274
MOST_COMMON_DIMENSION_LIMIT = 10

KNOWN_IMAGE_EXTENSIONS = frozenset(
    {
        ".apng",
        ".arw",
        ".avif",
        ".bmp",
        ".cr2",
        ".cr3",
        ".dds",
        ".dng",
        ".gif",
        ".heic",
        ".heif",
        ".ico",
        ".jfif",
        ".jpe",
        ".jpeg",
        ".jpg",
        ".nef",
        ".orf",
        ".pbm",
        ".pcx",
        ".pgm",
        ".png",
        ".pnm",
        ".ppm",
        ".psd",
        ".raw",
        ".rw2",
        ".tga",
        ".tif",
        ".tiff",
        ".webp",
    }
)


class RootRole(str, Enum):
    ORIGINAL = "ORIGINAL"
    CROPPED = "CROPPED"


class ReadStatus(str, Enum):
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    READABLE = "READABLE"
    UNSUPPORTED = "UNSUPPORTED"
    UNREADABLE = "UNREADABLE"
    FILESYSTEM_ERROR = "FILESYSTEM_ERROR"


class MetadataStatus(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"


class ErrorCategory(str, Enum):
    FILESYSTEM_ACCESS = "FILESYSTEM_ACCESS"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    IMAGE_DECODE = "IMAGE_DECODE"
    METADATA_READ = "METADATA_READ"


class ScanIssueCategory(str, Enum):
    FILESYSTEM_ACCESS = "FILESYSTEM_ACCESS"
    REPARSE_POINT_SKIPPED = "REPARSE_POINT_SKIPPED"
    SPECIAL_FILE_SKIPPED = "SPECIAL_FILE_SKIPPED"


class CollisionKeyType(str, Enum):
    RELATIVE_PATH = "RELATIVE_PATH"
    FILENAME = "FILENAME"
    STEM = "STEM"
    PHYSICAL_FILE = "PHYSICAL_FILE"
    ORIGINAL_WITH_MULTIPLE_CROPS = "ORIGINAL_WITH_MULTIPLE_CROPS"


@dataclass(frozen=True, slots=True)
class SemanticReference:
    root_role: RootRole
    relative_path: str

    @property
    def filename(self) -> str:
        return PurePosixPath(self.relative_path).name

    @property
    def stem(self) -> str:
        return PurePosixPath(self.relative_path).stem


@dataclass(frozen=True, slots=True)
class AuditItem:
    root_role: RootRole
    relative_path: str
    size_bytes: int | None
    extension: str
    is_image_candidate: bool
    read_status: ReadStatus
    metadata_status: MetadataStatus
    detected_format: str | None = None
    encoded_width: int | None = None
    encoded_height: int | None = None
    exif_orientation: int | None = None
    display_width: int | None = None
    display_height: int | None = None
    error_category: ErrorCategory | None = None
    error_type: str | None = None

    @property
    def reference(self) -> SemanticReference:
        return SemanticReference(self.root_role, self.relative_path)

    @property
    def filename(self) -> str:
        return self.reference.filename

    @property
    def stem(self) -> str:
        return self.reference.stem


@dataclass(frozen=True, slots=True)
class ScanIssue:
    root_role: RootRole
    relative_path: str
    category: ScanIssueCategory
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class CollisionGroup:
    key_type: CollisionKeyType
    normalized_key: str
    members: tuple[SemanticReference, ...]


@dataclass(frozen=True, slots=True)
class AuditResult:
    items: tuple[AuditItem, ...]
    scan_issues: tuple[ScanIssue, ...]
    collisions: tuple[CollisionGroup, ...]
    registered_extensions: tuple[tuple[str, str], ...]


def semantic_reference_sort_key(reference: SemanticReference) -> tuple[int, str, str]:
    role_order = 0 if reference.root_role is RootRole.ORIGINAL else 1
    return role_order, reference.relative_path.casefold(), reference.relative_path


def audit_item_sort_key(item: AuditItem) -> tuple[int, str, str]:
    return semantic_reference_sort_key(item.reference)


def _entry_sort_key(entry: os.DirEntry[str]) -> tuple[str, str]:
    return entry.name.casefold(), entry.name


def pillow_registered_extensions() -> dict[str, str]:
    """Return Pillow's normalized extension-to-format registry."""

    Image.init()
    return {
        extension.casefold(): format_name
        for extension, format_name in Image.registered_extensions().items()
    }


def audit_datasets(originals_root: Path, cropped_root: Path) -> AuditResult:
    """Audit two already validated roots without modifying their contents."""

    registered_extensions = pillow_registered_extensions()
    candidate_extensions = KNOWN_IMAGE_EXTENSIONS | registered_extensions.keys()

    all_items: list[AuditItem] = []
    all_issues: list[ScanIssue] = []
    all_physical_members: dict[tuple[int, int], list[SemanticReference]] = defaultdict(list)

    for role, root in (
        (RootRole.ORIGINAL, originals_root),
        (RootRole.CROPPED, cropped_root),
    ):
        items, issues, physical_members = _scan_root(
            root,
            role,
            candidate_extensions,
            registered_extensions,
        )
        all_items.extend(items)
        all_issues.extend(issues)
        for physical_key, members in physical_members.items():
            all_physical_members[physical_key].extend(members)

    sorted_items = tuple(sorted(all_items, key=audit_item_sort_key))
    sorted_issues = tuple(sorted(all_issues, key=_scan_issue_sort_key))
    collisions = build_identity_collisions(sorted_items)
    physical_collisions = _build_physical_collisions(all_physical_members)

    return AuditResult(
        items=sorted_items,
        scan_issues=sorted_issues,
        collisions=tuple(sorted((*collisions, *physical_collisions), key=_collision_sort_key)),
        registered_extensions=tuple(sorted(registered_extensions.items())),
    )


def _scan_root(
    root: Path,
    role: RootRole,
    candidate_extensions: Iterable[str],
    registered_extensions: dict[str, str],
) -> tuple[
    list[AuditItem],
    list[ScanIssue],
    dict[tuple[int, int], list[SemanticReference]],
]:
    items: list[AuditItem] = []
    issues: list[ScanIssue] = []
    physical_members: dict[tuple[int, int], list[SemanticReference]] = defaultdict(list)
    candidate_extension_set = frozenset(candidate_extensions)

    def scan_directory(directory: Path, relative_directory: PurePosixPath) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=_entry_sort_key)
        except OSError as exc:
            issues.append(
                ScanIssue(
                    role,
                    _manifest_path(relative_directory),
                    ScanIssueCategory.FILESYSTEM_ACCESS,
                    type(exc).__name__,
                )
            )
            return

        for entry in entries:
            relative = relative_directory / entry.name
            relative_path = _manifest_path(relative)
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                issues.append(
                    ScanIssue(
                        role,
                        relative_path,
                        ScanIssueCategory.FILESYSTEM_ACCESS,
                        type(exc).__name__,
                    )
                )
                continue

            if _is_reparse_or_symlink(entry, entry_stat):
                issues.append(
                    ScanIssue(
                        role,
                        relative_path,
                        ScanIssueCategory.REPARSE_POINT_SKIPPED,
                    )
                )
                continue

            if stat.S_ISDIR(entry_stat.st_mode):
                scan_directory(Path(entry.path), relative)
                continue

            if not stat.S_ISREG(entry_stat.st_mode):
                issues.append(
                    ScanIssue(
                        role,
                        relative_path,
                        ScanIssueCategory.SPECIAL_FILE_SKIPPED,
                    )
                )
                continue

            extension = PurePosixPath(relative_path).suffix.casefold()
            is_candidate = extension in candidate_extension_set
            item = _inspect_file(
                Path(entry.path),
                role,
                relative_path,
                entry_stat.st_size,
                extension,
                is_candidate,
                registered_extensions,
            )
            items.append(item)

            physical_stat = entry_stat
            if not getattr(physical_stat, "st_ino", 0):
                try:
                    physical_stat = os.stat(entry.path, follow_symlinks=False)
                except OSError:
                    physical_stat = entry_stat
            inode = getattr(physical_stat, "st_ino", 0)
            device = getattr(physical_stat, "st_dev", 0)
            if inode:
                physical_members[(device, inode)].append(item.reference)

    scan_directory(root, PurePosixPath())
    return items, issues, physical_members


def _manifest_path(path: PurePosixPath) -> str:
    value = path.as_posix()
    return "" if value == "." else value


def _is_reparse_or_symlink(
    entry: os.DirEntry[str], entry_stat: os.stat_result
) -> bool:
    if entry.is_symlink():
        return True
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(entry_stat, "st_file_attributes", 0)
    if reparse_attribute and file_attributes & reparse_attribute:
        return True
    return os.path.ismount(entry.path)


def _inspect_file(
    path: Path,
    role: RootRole,
    relative_path: str,
    size_bytes: int,
    extension: str,
    is_candidate: bool,
    registered_extensions: dict[str, str],
) -> AuditItem:
    base = {
        "root_role": role,
        "relative_path": relative_path,
        "size_bytes": size_bytes,
        "extension": extension,
        "is_image_candidate": is_candidate,
    }
    if not is_candidate:
        return AuditItem(
            **base,
            read_status=ReadStatus.NOT_ATTEMPTED,
            metadata_status=MetadataStatus.NOT_APPLICABLE,
        )

    try:
        source = path.open("rb")
    except OSError as exc:
        return AuditItem(
            **base,
            read_status=ReadStatus.FILESYSTEM_ERROR,
            metadata_status=MetadataStatus.NOT_APPLICABLE,
            error_category=ErrorCategory.FILESYSTEM_ACCESS,
            error_type=type(exc).__name__,
        )

    try:
        with source:
            with Image.open(source) as verification_image:
                detected_format = verification_image.format
                encoded_width, encoded_height = verification_image.size
                try:
                    verification_image.verify()
                except Exception as exc:
                    return AuditItem(
                        **base,
                        read_status=ReadStatus.UNREADABLE,
                        metadata_status=MetadataStatus.PARTIAL,
                        detected_format=detected_format,
                        encoded_width=encoded_width,
                        encoded_height=encoded_height,
                        error_category=ErrorCategory.IMAGE_DECODE,
                        error_type=type(exc).__name__,
                    )

            metadata_status = MetadataStatus.COMPLETE
            error_category = None
            error_type = None
            try:
                source.seek(0)
                with Image.open(source) as metadata_image:
                    orientation_value = metadata_image.getexif().get(EXIF_ORIENTATION_TAG)
                if orientation_value is None:
                    exif_orientation = None
                    display_width, display_height = encoded_width, encoded_height
                elif isinstance(orientation_value, int) and 1 <= orientation_value <= 8:
                    exif_orientation = orientation_value
                    if orientation_value >= 5:
                        display_width, display_height = encoded_height, encoded_width
                    else:
                        display_width, display_height = encoded_width, encoded_height
                else:
                    exif_orientation = None
                    display_width = None
                    display_height = None
                    metadata_status = MetadataStatus.PARTIAL
                    error_category = ErrorCategory.METADATA_READ
                    error_type = "InvalidExifOrientation"
            except Exception as exc:
                exif_orientation = None
                display_width = None
                display_height = None
                metadata_status = MetadataStatus.PARTIAL
                error_category = ErrorCategory.METADATA_READ
                error_type = type(exc).__name__

            return AuditItem(
                **base,
                read_status=ReadStatus.READABLE,
                metadata_status=metadata_status,
                detected_format=detected_format,
                encoded_width=encoded_width,
                encoded_height=encoded_height,
                exif_orientation=exif_orientation,
                display_width=display_width,
                display_height=display_height,
                error_category=error_category,
                error_type=error_type,
            )
    except (FileNotFoundError, PermissionError) as exc:
        return AuditItem(
            **base,
            read_status=ReadStatus.FILESYSTEM_ERROR,
            metadata_status=MetadataStatus.NOT_APPLICABLE,
            error_category=ErrorCategory.FILESYSTEM_ACCESS,
            error_type=type(exc).__name__,
        )
    except UnidentifiedImageError as exc:
        supported_extension = extension in registered_extensions
        return AuditItem(
            **base,
            read_status=(
                ReadStatus.UNREADABLE if supported_extension else ReadStatus.UNSUPPORTED
            ),
            metadata_status=MetadataStatus.NOT_APPLICABLE,
            error_category=(
                ErrorCategory.IMAGE_DECODE
                if supported_extension
                else ErrorCategory.UNSUPPORTED_FORMAT
            ),
            error_type=type(exc).__name__,
        )
    except OSError as exc:
        supported_extension = extension in registered_extensions
        return AuditItem(
            **base,
            read_status=(
                ReadStatus.UNREADABLE if supported_extension else ReadStatus.UNSUPPORTED
            ),
            metadata_status=MetadataStatus.NOT_APPLICABLE,
            error_category=(
                ErrorCategory.IMAGE_DECODE
                if supported_extension
                else ErrorCategory.UNSUPPORTED_FORMAT
            ),
            error_type=type(exc).__name__,
        )
    except (SyntaxError, TypeError, ValueError, Image.DecompressionBombError) as exc:
        supported_extension = extension in registered_extensions
        return AuditItem(
            **base,
            read_status=(
                ReadStatus.UNREADABLE if supported_extension else ReadStatus.UNSUPPORTED
            ),
            metadata_status=MetadataStatus.NOT_APPLICABLE,
            error_category=(
                ErrorCategory.IMAGE_DECODE
                if supported_extension
                else ErrorCategory.UNSUPPORTED_FORMAT
            ),
            error_type=type(exc).__name__,
        )


def build_identity_collisions(items: Iterable[AuditItem]) -> tuple[CollisionGroup, ...]:
    groups: list[CollisionGroup] = []
    item_list = tuple(items)
    key_functions = (
        (CollisionKeyType.RELATIVE_PATH, lambda item: item.relative_path.casefold()),
        (CollisionKeyType.FILENAME, lambda item: item.filename.casefold()),
        (CollisionKeyType.STEM, lambda item: item.stem.casefold()),
    )

    for role in RootRole:
        role_items = (item for item in item_list if item.root_role is role)
        role_item_list = tuple(role_items)
        for key_type, key_function in key_functions:
            members_by_key: dict[str, list[SemanticReference]] = defaultdict(list)
            for item in role_item_list:
                members_by_key[key_function(item)].append(item.reference)
            for normalized_key, members in members_by_key.items():
                if len(members) > 1:
                    groups.append(
                        CollisionGroup(
                            key_type,
                            normalized_key,
                            tuple(sorted(members, key=semantic_reference_sort_key)),
                        )
                    )

    return tuple(sorted(groups, key=_collision_sort_key))


def _build_physical_collisions(
    physical_members: dict[tuple[int, int], list[SemanticReference]],
) -> tuple[CollisionGroup, ...]:
    member_groups = [
        tuple(sorted(members, key=semantic_reference_sort_key))
        for members in physical_members.values()
        if len(members) > 1
    ]
    member_groups.sort(key=lambda members: tuple(map(semantic_reference_sort_key, members)))
    return tuple(
        CollisionGroup(
            CollisionKeyType.PHYSICAL_FILE,
            f"physical-group-{index:04d}",
            members,
        )
        for index, members in enumerate(member_groups, start=1)
    )


def _scan_issue_sort_key(issue: ScanIssue) -> tuple[int, str, str, str]:
    role_order = 0 if issue.root_role is RootRole.ORIGINAL else 1
    return (
        role_order,
        issue.relative_path.casefold(),
        issue.relative_path,
        issue.category.value,
    )


def _collision_sort_key(
    collision: CollisionGroup,
) -> tuple[str, str, tuple[tuple[int, str, str], ...]]:
    return (
        collision.key_type.value,
        collision.normalized_key,
        tuple(semantic_reference_sort_key(member) for member in collision.members),
    )


def build_role_summary(
    items: Iterable[AuditItem], registered_extensions: dict[str, str]
) -> dict[str, object]:
    item_list = tuple(items)
    extension_counts = Counter(item.extension or "<none>" for item in item_list)
    format_counts = Counter(
        item.detected_format for item in item_list if item.detected_format is not None
    )
    encoded_dimensions = Counter(
        (item.encoded_width, item.encoded_height)
        for item in item_list
        if item.encoded_width is not None and item.encoded_height is not None
    )
    display_dimensions = Counter(
        (item.display_width, item.display_height)
        for item in item_list
        if item.display_width is not None and item.display_height is not None
    )
    exif_counts = {"absent": 0, **{str(value): 0 for value in range(1, 9)}}
    for item in item_list:
        if (
            item.read_status is not ReadStatus.READABLE
            or item.metadata_status is not MetadataStatus.COMPLETE
        ):
            continue
        key = "absent" if item.exif_orientation is None else str(item.exif_orientation)
        exif_counts[key] += 1

    return {
        "total_regular_files": len(item_list),
        "image_candidates": sum(item.is_image_candidate for item in item_list),
        "non_image_files": sum(not item.is_image_candidate for item in item_list),
        "readable_candidates": _status_count(item_list, ReadStatus.READABLE),
        "unsupported_candidates": _status_count(item_list, ReadStatus.UNSUPPORTED),
        "unreadable_candidates": _status_count(item_list, ReadStatus.UNREADABLE),
        "filesystem_error_candidates": _status_count(
            item_list, ReadStatus.FILESYSTEM_ERROR
        ),
        "partial_metadata_images": sum(
            item.metadata_status is MetadataStatus.PARTIAL for item in item_list
        ),
        "extension_distribution": dict(sorted(extension_counts.items())),
        "detected_format_distribution": dict(sorted(format_counts.items())),
        "encoded_orientation_counts": _orientation_counts(encoded_dimensions),
        "display_orientation_counts": _orientation_counts(display_dimensions),
        "exif_orientation_distribution": exif_counts,
        "unique_encoded_dimension_count": len(encoded_dimensions),
        "unique_display_dimension_count": len(display_dimensions),
        "most_common_encoded_dimensions": _most_common_dimensions(encoded_dimensions),
        "most_common_display_dimensions": _most_common_dimensions(display_dimensions),
        "extension_detected_format_mismatch_count": sum(
            _is_extension_format_mismatch(item, registered_extensions)
            for item in item_list
        ),
    }


def _status_count(items: Iterable[AuditItem], status: ReadStatus) -> int:
    return sum(item.read_status is status for item in items)


def _orientation_counts(
    dimensions: Counter[tuple[int | None, int | None]],
) -> dict[str, int]:
    counts = {"portrait": 0, "landscape": 0, "square": 0}
    for (width, height), count in dimensions.items():
        if width is None or height is None:
            continue
        if width < height:
            counts["portrait"] += count
        elif width > height:
            counts["landscape"] += count
        else:
            counts["square"] += count
    return counts


def _most_common_dimensions(
    dimensions: Counter[tuple[int | None, int | None]],
) -> list[dict[str, int]]:
    ordered = sorted(
        dimensions.items(),
        key=lambda entry: (-entry[1], entry[0][0] or 0, entry[0][1] or 0),
    )[:MOST_COMMON_DIMENSION_LIMIT]
    return [
        {"width": width, "height": height, "count": count}
        for (width, height), count in ordered
        if width is not None and height is not None
    ]


def _is_extension_format_mismatch(
    item: AuditItem, registered_extensions: dict[str, str]
) -> bool:
    if item.detected_format is None:
        return False
    expected_format = registered_extensions.get(item.extension)
    return expected_format is not None and expected_format != item.detected_format

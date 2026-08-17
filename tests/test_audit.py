"""Synthetic tests for deterministic, read-only dataset auditing."""

from __future__ import annotations

from pathlib import Path
import os
import tempfile
import unittest
from unittest import mock

from PIL import Image

from autocrop_analysis.audit import (
    AuditItem,
    CollisionKeyType,
    ErrorCategory,
    KNOWN_IMAGE_EXTENSIONS,
    MetadataStatus,
    ReadStatus,
    RootRole,
    ScanIssueCategory,
    audit_datasets,
    build_identity_collisions,
    build_role_summary,
    pillow_registered_extensions,
    _is_reparse_or_symlink,
)


def save_image(
    path: Path,
    *,
    size: tuple[int, int] = (12, 8),
    format_name: str | None = None,
    orientation: int | None = None,
) -> None:
    image = Image.new("RGB", size, color=(30, 60, 90))
    if orientation is None:
        image.save(path, format=format_name)
    else:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(path, format=format_name, exif=exif)


class AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.originals = base / "originals"
        self.cropped = base / "cropped"
        self.originals.mkdir()
        self.cropped.mkdir()

    def audit(self):
        return audit_datasets(self.originals, self.cropped)

    def test_empty_roots(self) -> None:
        result = self.audit()
        self.assertEqual(result.items, ())
        self.assertEqual(result.scan_issues, ())

    def test_valid_nested_hidden_and_case_normalized_files(self) -> None:
        nested = self.originals / "Nested"
        nested.mkdir()
        save_image(nested / "Photo.JPG", format_name="JPEG")
        save_image(self.cropped / ".hidden.PNG", format_name="PNG")

        result = self.audit()

        self.assertEqual(
            [item.relative_path for item in result.items],
            ["Nested/Photo.JPG", ".hidden.PNG"],
        )
        self.assertEqual([item.extension for item in result.items], [".jpg", ".png"])
        self.assertTrue(all(item.read_status is ReadStatus.READABLE for item in result.items))

    def test_non_image_and_unknown_extension_are_retained_without_decode(self) -> None:
        (self.originals / "notes.txt").write_text("synthetic", encoding="utf-8")
        (self.originals / "payload.odd").write_bytes(b"not an image")

        result = self.audit()
        original_items = [item for item in result.items if item.root_role is RootRole.ORIGINAL]

        self.assertEqual(len(original_items), 2)
        self.assertTrue(all(not item.is_image_candidate for item in original_items))
        self.assertTrue(
            all(item.read_status is ReadStatus.NOT_ATTEMPTED for item in original_items)
        )

    def test_unsupported_image_like_extension_is_visible(self) -> None:
        registered = pillow_registered_extensions()
        raw_extension = next(
            (
                extension
                for extension in (".cr3", ".dng", ".nef", ".arw", ".rw2")
                if extension not in registered
            ),
            None,
        )
        if raw_extension is None:
            self.skipTest("installed Pillow advertises every test RAW extension")
        path = self.originals / f"synthetic{raw_extension}"
        path.write_bytes(b"synthetic unsupported content")

        item = self.audit().items[0]

        self.assertIn(raw_extension, KNOWN_IMAGE_EXTENSIONS)
        self.assertTrue(item.is_image_candidate)
        self.assertIs(item.read_status, ReadStatus.UNSUPPORTED)
        self.assertIs(item.error_category, ErrorCategory.UNSUPPORTED_FORMAT)

    def test_malformed_supported_extension_is_unreadable(self) -> None:
        (self.originals / "broken.jpg").write_bytes(b"not a jpeg")

        item = self.audit().items[0]

        self.assertIs(item.read_status, ReadStatus.UNREADABLE)
        self.assertIs(item.error_category, ErrorCategory.IMAGE_DECODE)
        self.assertNotIn(str(self.originals), item.error_type or "")

    def test_detected_format_extension_mismatch(self) -> None:
        save_image(self.originals / "mismatch.jpg", format_name="PNG")
        result = self.audit()
        summary = build_role_summary(
            (item for item in result.items if item.root_role is RootRole.ORIGINAL),
            dict(result.registered_extensions),
        )

        self.assertEqual(result.items[0].detected_format, "PNG")
        self.assertEqual(summary["extension_detected_format_mismatch_count"], 1)

    def test_exif_absent_and_orientation_one_preserve_dimensions(self) -> None:
        save_image(self.originals / "absent.jpg", size=(12, 8), format_name="JPEG")
        save_image(
            self.originals / "one.jpg",
            size=(12, 8),
            format_name="JPEG",
            orientation=1,
        )

        items = {item.filename: item for item in self.audit().items}

        self.assertIsNone(items["absent.jpg"].exif_orientation)
        self.assertEqual(
            (items["absent.jpg"].display_width, items["absent.jpg"].display_height),
            (12, 8),
        )
        self.assertEqual(items["one.jpg"].exif_orientation, 1)
        self.assertEqual(
            (items["one.jpg"].display_width, items["one.jpg"].display_height),
            (12, 8),
        )

    def test_exif_orientation_six_swaps_display_dimensions(self) -> None:
        save_image(
            self.originals / "rotated.jpg",
            size=(12, 8),
            format_name="JPEG",
            orientation=6,
        )

        item = self.audit().items[0]

        self.assertEqual((item.encoded_width, item.encoded_height), (12, 8))
        self.assertEqual(item.exif_orientation, 6)
        self.assertEqual((item.display_width, item.display_height), (8, 12))

    def test_exif_orientations_five_through_eight_swap_dimensions(self) -> None:
        for orientation in range(5, 9):
            save_image(
                self.originals / f"orientation-{orientation}.png",
                size=(9, 4),
                format_name="PNG",
                orientation=orientation,
            )

        for item in self.audit().items:
            self.assertEqual((item.display_width, item.display_height), (4, 9))

    def test_invalid_exif_orientation_is_partial_without_display_guess(self) -> None:
        save_image(
            self.originals / "invalid.jpg",
            format_name="JPEG",
            orientation=9,
        )

        item = self.audit().items[0]

        self.assertIs(item.read_status, ReadStatus.READABLE)
        self.assertIs(item.metadata_status, MetadataStatus.PARTIAL)
        self.assertIs(item.error_category, ErrorCategory.METADATA_READ)
        self.assertIsNone(item.exif_orientation)
        self.assertIsNone(item.display_width)
        self.assertIsNone(item.display_height)

    def test_scan_does_not_change_source_bytes_or_directory_contents(self) -> None:
        source = self.originals / "source.png"
        save_image(source, format_name="PNG")
        before_bytes = source.read_bytes()
        before_entries = sorted(path.name for path in self.originals.iterdir())

        self.audit()

        self.assertEqual(source.read_bytes(), before_bytes)
        self.assertEqual(
            sorted(path.name for path in self.originals.iterdir()), before_entries
        )

    def test_ordering_and_repeated_inventory_are_deterministic(self) -> None:
        for name in ("z.png", "A.png", "middle.png"):
            save_image(self.originals / name, format_name="PNG")

        first = self.audit()
        second = self.audit()

        self.assertEqual(first, second)
        self.assertEqual(
            [item.relative_path for item in first.items],
            ["A.png", "middle.png", "z.png"],
        )

    def test_casefolded_filename_and_stem_collisions_are_reported(self) -> None:
        (self.originals / "one").mkdir()
        (self.originals / "two").mkdir()
        save_image(self.originals / "one" / "Same.JPG", format_name="JPEG")
        save_image(self.originals / "two" / "same.jpg", format_name="JPEG")

        collisions = self.audit().collisions
        key_types = {collision.key_type for collision in collisions}

        self.assertIn(CollisionKeyType.FILENAME, key_types)
        self.assertIn(CollisionKeyType.STEM, key_types)

    def test_casefolded_relative_path_collision_is_platform_independent(self) -> None:
        common = {
            "root_role": RootRole.ORIGINAL,
            "size_bytes": 1,
            "extension": ".jpg",
            "is_image_candidate": True,
            "read_status": ReadStatus.READABLE,
            "metadata_status": MetadataStatus.COMPLETE,
        }
        collisions = build_identity_collisions(
            [
                AuditItem(relative_path="Case/Photo.JPG", **common),
                AuditItem(relative_path="case/photo.jpg", **common),
            ]
        )

        self.assertTrue(
            any(
                collision.key_type is CollisionKeyType.RELATIVE_PATH
                for collision in collisions
            )
        )

    def test_symlink_is_skipped_without_traversal(self) -> None:
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        save_image(outside / "escaped.png", format_name="PNG")
        link = self.originals / "linked"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"directory symlink unavailable: {type(exc).__name__}")

        result = self.audit()

        self.assertFalse(any(item.filename == "escaped.png" for item in result.items))
        self.assertTrue(
            any(
                issue.category is ScanIssueCategory.REPARSE_POINT_SKIPPED
                for issue in result.scan_issues
            )
        )

    def test_hard_links_remain_separate_with_best_effort_diagnostic(self) -> None:
        first = self.originals / "first.png"
        second = self.originals / "second.png"
        save_image(first, format_name="PNG")
        try:
            os.link(first, second)
        except OSError as exc:
            self.skipTest(f"hard links unavailable: {type(exc).__name__}")

        result = self.audit()

        self.assertEqual(len(result.items), 2)
        physical = [
            group
            for group in result.collisions
            if group.key_type is CollisionKeyType.PHYSICAL_FILE
        ]
        if first.stat().st_ino:
            self.assertEqual(len(physical), 1)

    def test_file_disappearing_before_open_becomes_filesystem_finding(self) -> None:
        source = self.originals / "vanishing.png"
        save_image(source, format_name="PNG")
        original_open = Path.open

        def race_open(path: Path, *args, **kwargs):
            if path.name == "vanishing.png":
                raise FileNotFoundError
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", race_open):
            item = self.audit().items[0]

        self.assertIs(item.read_status, ReadStatus.FILESYSTEM_ERROR)
        self.assertIs(item.error_category, ErrorCategory.FILESYSTEM_ACCESS)
        self.assertEqual(item.error_type, "FileNotFoundError")

    def test_windows_reparse_attribute_is_recognized_without_following(self) -> None:
        reparse_attribute = getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if not reparse_attribute:
            self.skipTest("Windows reparse attributes unavailable")
        entry = mock.Mock()
        entry.is_symlink.return_value = False
        entry.path = str(self.originals / "synthetic-junction")
        entry_stat = mock.Mock(st_file_attributes=reparse_attribute)

        with mock.patch("autocrop_analysis.audit.os.path.ismount", return_value=False):
            self.assertTrue(_is_reparse_or_symlink(entry, entry_stat))


if __name__ == "__main__":
    unittest.main()

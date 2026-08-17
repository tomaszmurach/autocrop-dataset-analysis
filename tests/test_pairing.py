"""Tests for conservative, decoder-independent pair discovery."""

from __future__ import annotations

import unittest

from autocrop_analysis.audit import (
    AuditItem,
    CollisionKeyType,
    MetadataStatus,
    ReadStatus,
    RootRole,
)
from autocrop_analysis.pairing import MatchingRule, PairStatus, discover_pairs


def item(
    role: RootRole,
    relative_path: str,
    *,
    status: ReadStatus = ReadStatus.READABLE,
) -> AuditItem:
    extension = "." + relative_path.rsplit(".", 1)[1].casefold()
    dimensions = (10, 8) if status is ReadStatus.READABLE else (None, None)
    return AuditItem(
        root_role=role,
        relative_path=relative_path,
        size_bytes=100,
        extension=extension,
        is_image_candidate=True,
        read_status=status,
        metadata_status=(
            MetadataStatus.COMPLETE
            if status is ReadStatus.READABLE
            else MetadataStatus.NOT_APPLICABLE
        ),
        detected_format="JPEG" if status is ReadStatus.READABLE else None,
        encoded_width=dimensions[0],
        encoded_height=dimensions[1],
        display_width=dimensions[0],
        display_height=dimensions[1],
    )


class PairingTests(unittest.TestCase):
    def test_exact_unique_filename_including_extension(self) -> None:
        result = discover_pairs(
            [
                item(RootRole.ORIGINAL, "nested/photo.jpg"),
                item(RootRole.CROPPED, "elsewhere/photo.jpg"),
            ]
        )
        pair = result.pairs[0]
        self.assertIs(pair.status, PairStatus.MATCHED)
        self.assertIs(pair.matching_rule, MatchingRule.EXACT_FILENAME_UNIQUE)
        self.assertEqual(pair.matched_original.relative_path, "nested/photo.jpg")

    def test_filename_comparison_is_casefolded(self) -> None:
        pair = discover_pairs(
            [
                item(RootRole.ORIGINAL, "PHOTO.JPG"),
                item(RootRole.CROPPED, "photo.jpg"),
            ]
        ).pairs[0]
        self.assertIs(pair.status, PairStatus.MATCHED)

    def test_unique_stem_allows_extension_change(self) -> None:
        pair = discover_pairs(
            [
                item(RootRole.ORIGINAL, "photo.jpg"),
                item(RootRole.CROPPED, "photo.png"),
            ]
        ).pairs[0]
        self.assertIs(pair.status, PairStatus.MATCHED)
        self.assertIs(pair.matching_rule, MatchingRule.EXACT_STEM_UNIQUE)

    def test_unmatched_crop(self) -> None:
        pair = discover_pairs(
            [
                item(RootRole.ORIGINAL, "one.jpg"),
                item(RootRole.CROPPED, "two.jpg"),
            ]
        ).pairs[0]
        self.assertIs(pair.status, PairStatus.UNMATCHED)
        self.assertEqual(pair.candidate_count, 0)
        self.assertIsNone(pair.matching_rule)

    def test_duplicate_filename_is_ambiguous_and_stops(self) -> None:
        pair = discover_pairs(
            [
                item(RootRole.ORIGINAL, "a/photo.jpg"),
                item(RootRole.ORIGINAL, "b/photo.jpg"),
                item(RootRole.CROPPED, "a/photo.jpg"),
            ]
        ).pairs[0]
        self.assertIs(pair.status, PairStatus.AMBIGUOUS)
        self.assertIs(pair.matching_rule, MatchingRule.EXACT_FILENAME_UNIQUE)
        self.assertEqual(pair.candidate_count, 2)

    def test_duplicate_stem_is_ambiguous_at_fallback(self) -> None:
        pair = discover_pairs(
            [
                item(RootRole.ORIGINAL, "a/photo.jpg"),
                item(RootRole.ORIGINAL, "b/photo.tif"),
                item(RootRole.CROPPED, "photo.png"),
            ]
        ).pairs[0]
        self.assertIs(pair.status, PairStatus.AMBIGUOUS)
        self.assertIs(pair.matching_rule, MatchingRule.EXACT_STEM_UNIQUE)

    def test_exact_filename_precedes_a_broader_stem_collision(self) -> None:
        pair = discover_pairs(
            [
                item(RootRole.ORIGINAL, "a/photo.jpg"),
                item(RootRole.ORIGINAL, "b/photo.tif"),
                item(RootRole.CROPPED, "photo.jpg"),
            ]
        ).pairs[0]
        self.assertIs(pair.status, PairStatus.MATCHED)
        self.assertIs(pair.matching_rule, MatchingRule.EXACT_FILENAME_UNIQUE)
        self.assertEqual(pair.matched_original.relative_path, "a/photo.jpg")

    def test_duplicate_cropped_identities_remain_separate(self) -> None:
        result = discover_pairs(
            [
                item(RootRole.ORIGINAL, "photo.jpg"),
                item(RootRole.CROPPED, "a/photo.jpg"),
                item(RootRole.CROPPED, "b/PHOTO.JPG"),
            ]
        )
        self.assertEqual(len(result.pairs), 2)
        self.assertTrue(all(pair.status is PairStatus.MATCHED for pair in result.pairs))

    def test_one_original_with_multiple_crops_is_diagnosed_not_collapsed(self) -> None:
        result = discover_pairs(
            [
                item(RootRole.ORIGINAL, "photo.jpg"),
                item(RootRole.CROPPED, "a/photo.jpg"),
                item(RootRole.CROPPED, "b/photo.jpg"),
            ]
        )
        self.assertEqual(result.matched_count, 2)
        self.assertEqual(len(result.one_to_many_collisions), 1)
        self.assertIs(
            result.one_to_many_collisions[0].key_type,
            CollisionKeyType.ORIGINAL_WITH_MULTIPLE_CROPS,
        )

    def test_candidate_ordering_is_stable(self) -> None:
        pair = discover_pairs(
            [
                item(RootRole.ORIGINAL, "z/photo.jpg"),
                item(RootRole.ORIGINAL, "A/photo.jpg"),
                item(RootRole.CROPPED, "photo.jpg"),
            ]
        ).pairs[0]
        self.assertEqual(
            [candidate.relative_path for candidate in pair.candidates],
            ["A/photo.jpg", "z/photo.jpg"],
        )

    def test_unsupported_candidates_identity_match_but_are_not_ready(self) -> None:
        result = discover_pairs(
            [
                item(RootRole.ORIGINAL, "photo.dng", status=ReadStatus.UNSUPPORTED),
                item(RootRole.CROPPED, "photo.dng", status=ReadStatus.UNSUPPORTED),
            ]
        )
        self.assertEqual(result.matched_count, 1)
        self.assertEqual(result.reconstruction_ready_matched_count, 0)

    def test_only_readable_pair_with_core_dimensions_is_ready(self) -> None:
        ready = discover_pairs(
            [
                item(RootRole.ORIGINAL, "photo.jpg"),
                item(RootRole.CROPPED, "photo.jpg"),
            ]
        )
        not_ready = discover_pairs(
            [
                item(RootRole.ORIGINAL, "photo.jpg"),
                item(RootRole.CROPPED, "photo.jpg", status=ReadStatus.UNREADABLE),
            ]
        )
        self.assertEqual(ready.reconstruction_ready_matched_count, 1)
        self.assertEqual(not_ready.reconstruction_ready_matched_count, 0)

    def test_relative_directory_similarity_never_breaks_global_ambiguity(self) -> None:
        pair = discover_pairs(
            [
                item(RootRole.ORIGINAL, "same/photo.jpg"),
                item(RootRole.ORIGINAL, "other/photo.jpg"),
                item(RootRole.CROPPED, "same/photo.jpg"),
            ]
        ).pairs[0]
        self.assertIs(pair.status, PairStatus.AMBIGUOUS)


if __name__ == "__main__":
    unittest.main()

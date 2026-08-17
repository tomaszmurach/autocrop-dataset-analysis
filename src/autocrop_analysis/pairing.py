"""Conservative, deterministic pairing over audited semantic identities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable

from .audit import (
    AuditItem,
    CollisionGroup,
    CollisionKeyType,
    ReadStatus,
    RootRole,
    SemanticReference,
    semantic_reference_sort_key,
)


class PairStatus(str, Enum):
    MATCHED = "MATCHED"
    UNMATCHED = "UNMATCHED"
    AMBIGUOUS = "AMBIGUOUS"


class MatchingRule(str, Enum):
    EXACT_FILENAME_UNIQUE = "EXACT_FILENAME_UNIQUE"
    EXACT_STEM_UNIQUE = "EXACT_STEM_UNIQUE"


@dataclass(frozen=True, slots=True)
class PairResult:
    cropped: SemanticReference
    status: PairStatus
    matched_original: SemanticReference | None
    candidate_count: int
    candidates: tuple[SemanticReference, ...]
    matching_rule: MatchingRule | None


@dataclass(frozen=True, slots=True)
class PairDiscoveryResult:
    pairs: tuple[PairResult, ...]
    one_to_many_collisions: tuple[CollisionGroup, ...]
    matched_count: int
    reconstruction_ready_matched_count: int
    unmatched_count: int
    ambiguous_count: int


def discover_pairs(items: Iterable[AuditItem]) -> PairDiscoveryResult:
    """Pair every cropped candidate using filename, then unique-stem fallback."""

    item_list = tuple(items)
    original_items = tuple(
        sorted(
            (
                item
                for item in item_list
                if item.root_role is RootRole.ORIGINAL and item.is_image_candidate
            ),
            key=lambda item: semantic_reference_sort_key(item.reference),
        )
    )
    cropped_items = tuple(
        sorted(
            (
                item
                for item in item_list
                if item.root_role is RootRole.CROPPED and item.is_image_candidate
            ),
            key=lambda item: semantic_reference_sort_key(item.reference),
        )
    )

    originals_by_filename = _index_originals(
        original_items, lambda item: item.filename.casefold()
    )
    originals_by_stem = _index_originals(
        original_items, lambda item: item.stem.casefold()
    )
    items_by_reference = {item.reference: item for item in item_list}

    pairs = tuple(
        _pair_one(cropped, originals_by_filename, originals_by_stem)
        for cropped in cropped_items
    )
    one_to_many = _build_one_to_many_collisions(pairs)

    return PairDiscoveryResult(
        pairs=pairs,
        one_to_many_collisions=one_to_many,
        matched_count=sum(pair.status is PairStatus.MATCHED for pair in pairs),
        reconstruction_ready_matched_count=sum(
            _is_reconstruction_ready(pair, items_by_reference) for pair in pairs
        ),
        unmatched_count=sum(pair.status is PairStatus.UNMATCHED for pair in pairs),
        ambiguous_count=sum(pair.status is PairStatus.AMBIGUOUS for pair in pairs),
    )


def _index_originals(
    originals: Iterable[AuditItem], key_function: Callable[[AuditItem], str]
) -> dict[str, tuple[SemanticReference, ...]]:
    members: dict[str, list[SemanticReference]] = defaultdict(list)
    for original in originals:
        members[key_function(original)].append(original.reference)
    return {
        key: tuple(sorted(references, key=semantic_reference_sort_key))
        for key, references in members.items()
    }


def _pair_one(
    cropped: AuditItem,
    originals_by_filename: dict[str, tuple[SemanticReference, ...]],
    originals_by_stem: dict[str, tuple[SemanticReference, ...]],
) -> PairResult:
    filename_candidates = originals_by_filename.get(cropped.filename.casefold(), ())
    if filename_candidates:
        return _result_for_candidates(
            cropped.reference,
            filename_candidates,
            MatchingRule.EXACT_FILENAME_UNIQUE,
        )

    stem_candidates = originals_by_stem.get(cropped.stem.casefold(), ())
    if stem_candidates:
        return _result_for_candidates(
            cropped.reference,
            stem_candidates,
            MatchingRule.EXACT_STEM_UNIQUE,
        )

    return PairResult(
        cropped=cropped.reference,
        status=PairStatus.UNMATCHED,
        matched_original=None,
        candidate_count=0,
        candidates=(),
        matching_rule=None,
    )


def _result_for_candidates(
    cropped: SemanticReference,
    candidates: tuple[SemanticReference, ...],
    rule: MatchingRule,
) -> PairResult:
    is_unique = len(candidates) == 1
    return PairResult(
        cropped=cropped,
        status=PairStatus.MATCHED if is_unique else PairStatus.AMBIGUOUS,
        matched_original=candidates[0] if is_unique else None,
        candidate_count=len(candidates),
        candidates=candidates,
        matching_rule=rule,
    )


def _is_reconstruction_ready(
    pair: PairResult,
    items_by_reference: dict[SemanticReference, AuditItem],
) -> bool:
    if pair.status is not PairStatus.MATCHED or pair.matched_original is None:
        return False
    cropped = items_by_reference[pair.cropped]
    original = items_by_reference[pair.matched_original]
    return _has_reconstruction_metadata(cropped) and _has_reconstruction_metadata(original)


def _has_reconstruction_metadata(item: AuditItem) -> bool:
    return (
        item.read_status is ReadStatus.READABLE
        and item.encoded_width is not None
        and item.encoded_height is not None
        and item.display_width is not None
        and item.display_height is not None
    )


def _build_one_to_many_collisions(
    pairs: Iterable[PairResult],
) -> tuple[CollisionGroup, ...]:
    crops_by_original: dict[SemanticReference, list[SemanticReference]] = defaultdict(list)
    for pair in pairs:
        if pair.status is PairStatus.MATCHED and pair.matched_original is not None:
            crops_by_original[pair.matched_original].append(pair.cropped)

    groups = [
        CollisionGroup(
            CollisionKeyType.ORIGINAL_WITH_MULTIPLE_CROPS,
            f"{original.root_role.value}:{original.relative_path.casefold()}",
            tuple(sorted(crops, key=semantic_reference_sort_key)),
        )
        for original, crops in crops_by_original.items()
        if len(crops) > 1
    ]
    return tuple(
        sorted(
            groups,
            key=lambda group: (
                group.normalized_key,
                tuple(semantic_reference_sort_key(member) for member in group.members),
            ),
        )
    )

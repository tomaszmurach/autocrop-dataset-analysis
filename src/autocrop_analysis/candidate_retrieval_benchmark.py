"""Configurable deterministic synthetic benchmark for candidate retrieval."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .audit import RootRole, SemanticReference
from .candidate_retrieval import (
    IndexParameters,
    QueryParameters,
    recall_at_k,
)
from .candidate_retrieval_cli import (
    build_index,
    load_index,
    query_index,
    validate_build_paths,
    validate_query_paths,
)
from .candidate_retrieval_profiling import RetrievalProfiler


DEFAULT_STRATA = ("resize", "jpeg", "luminance", "blur_noise", "exif")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autocrop_analysis.candidate_retrieval_benchmark",
        description="Run a deterministic synthetic known-source retrieval benchmark.",
    )
    parser.add_argument("--corpus-size", type=int, default=32)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--original-max-descriptors", type=int, default=128)
    parser.add_argument("--query-max-descriptors", type=int, default=64)
    parser.add_argument("--neighbor-depth", type=int, default=32)
    parser.add_argument("--k-values", default="1,5,10,20,50,100")
    parser.add_argument("--seed", type=int, default=17_029)
    parser.add_argument("--strata", default=",".join(DEFAULT_STRATA))
    return parser


def _positive_int(value: int, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label}_MUST_BE_POSITIVE")
    return value


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(sorted({_positive_int(int(part), "K") for part in value.split(",")}))
    if not result:
        raise ValueError("K_VALUES_REQUIRED")
    return result


def _parse_strata(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result or any(item not in DEFAULT_STRATA for item in result):
        raise ValueError("INVALID_STRATA")
    return result


def _structured_image(seed: int, size: tuple[int, int]) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = rng.integers(10, 240, (size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(array, "RGB")
    draw = ImageDraw.Draw(image)
    for index in range(28):
        x = int(rng.integers(0, max(1, size[0] - 45)))
        y = int(rng.integers(0, max(1, size[1] - 35)))
        color = tuple(int(item) for item in rng.integers(0, 255, 3))
        draw.ellipse((x, y, x + 32, y + 24), outline=color, width=3)
        draw.line((x, y, x + 32, y + 24), fill=color, width=2)
    return image


def _save_query(
    source: Image.Image,
    path: Path,
    stratum: str,
    rng: np.random.Generator,
) -> None:
    width, height = source.size
    left = int(rng.integers(width // 8, width // 4))
    top = int(rng.integers(height // 8, height // 4))
    right = int(rng.integers(3 * width // 4, 7 * width // 8))
    bottom = int(rng.integers(3 * height // 4, 7 * height // 8))
    crop = source.crop((left, top, right, bottom))
    if stratum == "resize":
        crop.resize((max(80, crop.width * 3 // 4), max(80, crop.height * 3 // 4)), Image.Resampling.LANCZOS).save(path)
    elif stratum == "jpeg":
        crop.resize((max(80, crop.width * 2 // 3), max(80, crop.height * 2 // 3)), Image.Resampling.LANCZOS).save(path, format="JPEG", quality=58)
    elif stratum == "luminance":
        ImageEnhance.Contrast(ImageEnhance.Brightness(crop).enhance(1.08)).enhance(0.92).save(path)
    elif stratum == "blur_noise":
        blurred = np.asarray(crop.filter(ImageFilter.GaussianBlur(0.55)), dtype=np.int16)
        noise = rng.normal(0, 1.5, blurred.shape)
        Image.fromarray(np.clip(blurred + noise, 0, 255).astype(np.uint8), "RGB").save(path)
    else:
        exif = Image.Exif()
        exif[274] = 6
        crop.transpose(Image.Transpose.ROTATE_90).save(path, format="JPEG", quality=80, exif=exif)


def run_benchmark(arguments: argparse.Namespace) -> dict[str, object]:
    corpus_size = _positive_int(arguments.corpus_size, "CORPUS_SIZE")
    query_count = _positive_int(arguments.queries, "QUERIES")
    k_values = _parse_ints(arguments.k_values)
    strata = _parse_strata(arguments.strata)
    rng = np.random.default_rng(arguments.seed)

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        originals = base / "originals"
        crops = base / "crops"
        results = base / "results"
        originals.mkdir()
        crops.mkdir()
        results.mkdir()
        sources: list[Image.Image] = []
        for index in range(corpus_size):
            size = (360 + 20 * (index % 3), 280 + 20 * (index % 4))
            image = _structured_image(arguments.seed + index * 101, size)
            image.save(originals / f"source-{index:05d}.png")
            sources.append(image)

        truth: dict[str, tuple[SemanticReference, str]] = {}
        for query_number in range(query_count):
            source_index = query_number % corpus_size
            stratum = strata[query_number % len(strata)]
            suffix = ".jpg" if stratum in {"jpeg", "exif"} else ".png"
            filename = f"query-{query_number:05d}-{stratum}{suffix}"
            _save_query(sources[source_index], crops / filename, stratum, rng)
            truth[filename] = (
                SemanticReference(RootRole.ORIGINAL, f"source-{source_index:05d}.png"),
                stratum,
            )

        index_path = results / "benchmark-index.private.json"
        build_paths = validate_build_paths(originals, index_path)
        index_parameters = IndexParameters(
            original_max_descriptors=arguments.original_max_descriptors,
            random_seed=arguments.seed,
        )
        profiler = RetrievalProfiler()
        index_manifest = build_index(
            build_paths, index_parameters, profiler=profiler
        )

        query_paths = validate_query_paths(
            index_path, crops, results / "unused-retrieval.private.json"
        )
        loaded = load_index(
            query_paths,
            profiler=profiler,
            load_context="same_process_after_build",
        )
        query_parameters = QueryParameters(
            query_max_descriptors=arguments.query_max_descriptors,
            neighbor_depth=arguments.neighbor_depth,
            requested_k=max(k_values),
        )
        _, query_results = query_index(
            query_paths, loaded, query_parameters, profiler=profiler
        )
        cases = []
        query_times = [
            measurement.elapsed_ns / 1_000_000_000
            for measurement in profiler.measurements
            if measurement.stage == "query.total"
        ]
        ranks: list[int | None] = []
        returned_sizes: list[int] = []
        tie_extensions: list[int] = []
        by_stratum: dict[str, list[tuple[SemanticReference, tuple]]] = defaultdict(list)
        for result in query_results:
            source, stratum = truth[result.crop.relative_path]
            candidates = result.ranked_candidates
            cases.append((source, candidates))
            by_stratum[stratum].append((source, candidates))
            rank = next(
                (index for index, candidate in enumerate(candidates, start=1) if candidate.original == source),
                None,
            )
            ranks.append(rank)
            returned_sizes.append(result.returned_candidate_count)
            tie_extensions.append(result.tie_extension_count)

        recalls = recall_at_k(cases, k_values)
        per_stratum = {
            stratum: {f"Recall@{k}": value for k, value in recall_at_k(values, k_values).items()}
            for stratum, values in sorted(by_stratum.items())
        }
        binary_path = build_paths.binary
        exact_shortlist_comparisons = sum(returned_sizes)
        exhaustive_comparisons = corpus_size * query_count
        profiler.snapshot_memory("benchmark_final")
        report = {
            "benchmark_semantics": "SYNTHETIC_KNOWN_SOURCE_RETRIEVAL",
            "configuration": {
                "corpus_size": corpus_size,
                "queries": query_count,
                "original_max_descriptors": arguments.original_max_descriptors,
                "query_max_descriptors": arguments.query_max_descriptors,
                "neighbor_depth": arguments.neighbor_depth,
                "k_values": list(k_values),
                "seed": arguments.seed,
                "strata": list(strata),
            },
            "recall": {f"Recall@{k}": value for k, value in recalls.items()},
            "per_stratum_recall": per_stratum,
            "source_rank_distribution": {
                "misses": sum(rank is None for rank in ranks),
                "ranks": dict(sorted(Counter(str(rank) if rank is not None else "MISS" for rank in ranks).items())),
            },
            "performance": {
                "index_build_seconds": profiler.elapsed_ns("build.total")
                / 1_000_000_000,
                "index_load_seconds": profiler.elapsed_ns(
                    "load.total", phase="same_process_after_build"
                )
                / 1_000_000_000,
                "query_batch_processing_seconds": profiler.elapsed_ns(
                    "query.batch_processing_total"
                )
                / 1_000_000_000,
                "query_median_seconds": float(np.median(query_times)),
                "query_p95_seconds": float(np.percentile(query_times, 95)),
                "index_manifest_bytes": index_path.stat().st_size,
                "descriptor_binary_bytes": binary_path.stat().st_size,
                "persistent_index_bytes": index_path.stat().st_size + binary_path.stat().st_size,
            },
            "shortlists": {
                "mean_returned_candidates": float(np.mean(returned_sizes)),
                "maximum_returned_candidates": max(returned_sizes),
                "total_tie_extension": sum(tie_extensions),
            },
            "comparison_estimate": {
                "exhaustive_comparisons": exhaustive_comparisons,
                "retrieval_shortlist_comparisons_at_max_k": exact_shortlist_comparisons,
                "reduction_factor": (
                    exhaustive_comparisons / exact_shortlist_comparisons
                    if exact_shortlist_comparisons
                    else None
                ),
            },
            "index_corpus_complete": index_manifest["summary"]["index_corpus_complete"],
            "profiling": profiler.as_report(),
        }
        if isinstance(loaded.descriptors, np.memmap):
            loaded.descriptors._mmap.close()
        return report


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        report = run_benchmark(arguments)
    except Exception as exc:
        print(f"benchmark failure: {type(exc).__name__}")
        return 1
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

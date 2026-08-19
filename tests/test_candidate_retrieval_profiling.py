"""Focused tests for opt-in exact-BF baseline profiling."""

from __future__ import annotations

from unittest import mock
import unittest

import numpy as np

from autocrop_analysis.audit import RootRole, SemanticReference
from autocrop_analysis.candidate_retrieval import (
    IndexStatus,
    OriginalIndexRecord,
    QueryParameters,
    retrieve_candidates,
)
from autocrop_analysis.candidate_retrieval_profiling import (
    ProcessMemoryReading,
    RetrievalProfiler,
    profiling_stage,
    read_process_memory,
)
from autocrop_analysis.candidate_retrieval_benchmark import (
    build_parser as build_benchmark_parser,
    run_benchmark,
)


def unavailable_memory() -> ProcessMemoryReading:
    return ProcessMemoryReading("unavailable", None, None)


class RetrievalProfilerTests(unittest.TestCase):
    def test_empty_report_is_valid_and_has_no_measurements(self) -> None:
        profiler = RetrievalProfiler(memory_reader=unavailable_memory)
        report = profiler.as_report()
        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(report["stages"], [])
        self.assertEqual(report["per_item"], [])
        self.assertEqual(report["memory"]["snapshots"], [])

    def test_disabled_stage_does_not_read_clock_or_evaluate_work(self) -> None:
        with mock.patch(
            "autocrop_analysis.candidate_retrieval_profiling.perf_counter_ns"
        ) as clock:
            with profiling_stage(None, "disabled") as timing:
                timing.add_work(rows=1)
        clock.assert_not_called()

    def test_multiple_invocations_are_aggregated_with_work_counts(self) -> None:
        profiler = RetrievalProfiler(memory_reader=unavailable_memory)
        with profiler.measure("synthetic.stage") as first:
            first.add_work(rows=2)
        with profiler.measure("synthetic.stage") as second:
            second.add_work(rows=3, bytes=12)

        report = profiler.as_report()
        stage = report["stages"][0]
        self.assertEqual(stage["stage"], "synthetic.stage")
        self.assertEqual(stage["invocation_count"], 2)
        self.assertEqual(stage["work_counts"], {"bytes": 12, "rows": 5})
        self.assertGreaterEqual(stage["elapsed_ns_total"], 0)

    def test_per_item_and_first_subsequent_phases_remain_distinct(self) -> None:
        profiler = RetrievalProfiler(memory_reader=unavailable_memory)
        with profiler.measure("query.total", item_ordinal=0, phase="first_query"):
            pass
        with profiler.measure(
            "query.total", item_ordinal=1, phase="subsequent_query"
        ):
            pass

        report = profiler.as_report()
        self.assertEqual(
            {(item["item_ordinal"], item["phase"]) for item in report["per_item"]},
            {(0, "first_query"), (1, "subsequent_query")},
        )
        self.assertEqual(len(report["stages"]), 2)

    def test_memory_unavailable_is_explicit(self) -> None:
        profiler = RetrievalProfiler(memory_reader=unavailable_memory)
        measurement = profiler.snapshot_memory("before")
        self.assertEqual(measurement.provider, "unavailable")
        self.assertIsNone(measurement.current_rss_bytes)
        self.assertIsNone(measurement.peak_rss_bytes)
        snapshot = profiler.as_report()["memory"]["snapshots"][0]
        self.assertEqual(snapshot["provider"], "unavailable")
        self.assertIsNone(snapshot["current_rss_bytes"])

    def test_platform_memory_reader_is_truthful_or_explicitly_unavailable(self) -> None:
        reading = read_process_memory()
        self.assertTrue(reading.provider)
        if reading.current_rss_bytes is not None:
            self.assertGreater(reading.current_rss_bytes, 0)
        if reading.peak_rss_bytes is not None:
            self.assertGreater(reading.peak_rss_bytes, 0)

    def test_profiled_exact_retrieval_is_identical_and_records_work_units(self) -> None:
        rng = np.random.default_rng(123)
        source = rng.normal(50, 2, (6, 128)).astype(np.float32)
        distractor = rng.normal(200, 2, (6, 128)).astype(np.float32)
        matrix = np.vstack((distractor, source)).astype(np.float32)
        records = (
            OriginalIndexRecord(
                SemanticReference(RootRole.ORIGINAL, "distractor.png"),
                100,
                100,
                10,
                "a" * 64,
                IndexStatus.INDEXED,
                6,
                0,
                6,
            ),
            OriginalIndexRecord(
                SemanticReference(RootRole.ORIGINAL, "source.png"),
                100,
                100,
                10,
                "b" * 64,
                IndexStatus.INDEXED,
                6,
                6,
                6,
            ),
        )
        parameters = QueryParameters(neighbor_depth=2, requested_k=5)
        expected = retrieve_candidates(source, matrix, records, parameters)
        profiler = RetrievalProfiler(memory_reader=unavailable_memory)
        actual = retrieve_candidates(
            source,
            matrix,
            records,
            parameters,
            profiler=profiler,
            item_ordinal=0,
        )

        self.assertEqual(actual, expected)
        report = profiler.as_report()
        stages = {stage["stage"]: stage for stage in report["stages"]}
        self.assertEqual(
            stages["query.exact_bf_search"]["work_counts"]
            ["descriptor_distance_work_units"],
            source.shape[0] * matrix.shape[0],
        )
        self.assertIn("query.vote_aggregation_ranking", stages)

    def test_small_synthetic_benchmark_surfaces_build_load_and_batch_profile(self) -> None:
        arguments = build_benchmark_parser().parse_args(
            [
                "--corpus-size",
                "2",
                "--queries",
                "2",
                "--k-values",
                "1,2",
            ]
        )
        report = run_benchmark(arguments)
        profiling = report["profiling"]
        stages = profiling["stages"]
        stage_names = {stage["stage"] for stage in stages}
        self.assertIn("build.total", stage_names)
        self.assertIn("load.total", stage_names)
        self.assertIn("query.batch_processing_total", stage_names)
        self.assertIn("query.exact_bf_search", stage_names)
        load = next(stage for stage in stages if stage["stage"] == "load.total")
        self.assertEqual(load["phase"], "same_process_after_build")
        self.assertEqual(report["recall"]["Recall@2"], 1.0)


if __name__ == "__main__":
    unittest.main()

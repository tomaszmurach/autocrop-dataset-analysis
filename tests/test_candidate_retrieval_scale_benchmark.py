"""Focused tiny-pool tests for descriptor-only exact-BF scale profiling."""

from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from autocrop_analysis import candidate_retrieval_scale_benchmark as benchmark
from autocrop_analysis.candidate_retrieval import (
    DESCRIPTOR_DIMENSION,
    DESCRIPTOR_DTYPE,
    QueryParameters,
    retrieve_candidates,
)


class DescriptorGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def generate(self, name: str, *, seed: int = 123) -> Path:
        path = self.base / name
        benchmark.generate_descriptor_file(
            path,
            12 * benchmark.DESCRIPTORS_PER_ORIGINAL,
            seed,
            chunk_rows=128,
        )
        return path

    def test_generation_is_deterministic_bounded_and_float32(self) -> None:
        first = self.generate("first.f32")
        second = self.generate("second.f32")

        self.assertEqual(first.read_bytes(), second.read_bytes())
        expected_bytes = 12 * 128 * DESCRIPTOR_DIMENSION * 4
        self.assertEqual(first.stat().st_size, expected_bytes)
        matrix = np.memmap(
            first,
            mode="r",
            dtype=np.dtype(DESCRIPTOR_DTYPE),
            shape=(12 * 128, DESCRIPTOR_DIMENSION),
        )
        try:
            self.assertEqual(matrix.dtype, np.dtype("<f4"))
            self.assertTrue(matrix.flags.c_contiguous)
            self.assertFalse(matrix.flags.writeable)
            self.assertGreaterEqual(float(matrix.min()), 0.0)
            self.assertLessEqual(float(matrix.max()), 255.0)
        finally:
            matrix._mmap.close()

    def test_ownership_ranges_are_contiguous_and_complete(self) -> None:
        records = benchmark.build_original_records(12)

        self.assertEqual(len(records), 12)
        self.assertEqual(records[0].descriptor_offset, 0)
        self.assertTrue(all(item.descriptor_count == 128 for item in records))
        self.assertEqual(
            records[-1].descriptor_offset + records[-1].descriptor_count,
            12 * 128,
        )
        self.assertEqual(
            [item.descriptor_offset for item in records],
            [index * 128 for index in range(12)],
        )

    def test_known_source_and_memmap_ndarray_results_are_identical(self) -> None:
        path = self.generate("matrix.f32")
        rows = 12 * 128
        matrix = np.memmap(
            path,
            mode="r",
            dtype=np.dtype(DESCRIPTOR_DTYPE),
            shape=(rows, DESCRIPTOR_DIMENSION),
        )
        try:
            records = benchmark.build_original_records(12)
            query = benchmark.make_query_descriptors(
                matrix,
                source_owner=3,
                query_rows=64,
                seed=123,
                query_ordinal=2,
            )
            parameters = QueryParameters()
            mapped = retrieve_candidates(query, matrix, records, parameters)
            in_memory = retrieve_candidates(
                query, np.array(matrix, copy=True), records, parameters
            )
        finally:
            matrix._mmap.close()

        self.assertEqual(mapped, in_memory)
        self.assertEqual(mapped[0].original, records[3].reference)

    def test_worker_result_contains_required_stages_memory_and_diagnostics(self) -> None:
        path = self.generate("worker.f32")
        result = benchmark.execute_scale_worker(
            path,
            descriptor_rows=12 * 128,
            seed=123,
            warm_queries=2,
            include_sensitivity=False,
        )

        self.assertEqual(result["descriptor_rows"], 12 * 128)
        self.assertEqual(len(result["warm_queries"]), 2)
        self.assertTrue(result["correctness_guard"]["passed"])
        self.assertIn("exact_bf_search_ns", result["first_touch"])
        self.assertIn("vote_aggregation_ranking_ns", result["first_touch"])
        self.assertIn("shortlist_construction_ns", result["first_touch"])
        self.assertIn(
            "descriptor_distance_work_units",
            result["first_touch"]["work_counts"]["exact_bf_search"],
        )
        self.assertEqual(
            [item["label"] for item in result["memory"]["snapshots"]],
            [
                "worker_start",
                "after_read_only_memmap_open",
                "before_first_search",
                "after_first_search",
                "before_warm_series",
                "after_warm_series",
                "final_state",
            ],
        )
        diagnostics = result["descriptor_diagnostics"]
        self.assertTrue(diagnostics["is_memmap"])
        self.assertFalse(diagnostics["writeable"])
        self.assertEqual(diagnostics["shape"], [12 * 128, 128])

    def test_spawned_worker_completes_in_a_fresh_process(self) -> None:
        path = self.generate("spawn.f32")
        outcome = benchmark.run_scale_in_fresh_process(
            path,
            descriptor_rows=12 * 128,
            seed=123,
            warm_queries=1,
            include_sensitivity=False,
            timeout_seconds=30.0,
        )

        self.assertEqual(outcome.status, "completed")
        self.assertIsNotNone(outcome.result)
        assert outcome.result is not None
        self.assertTrue(outcome.result["correctness_guard"]["passed"])

    def test_missing_memmap_is_reported_as_mapping_failure(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "MEMMAP_OPEN_FAILED"):
            benchmark.execute_scale_worker(
                self.base / "missing.f32",
                descriptor_rows=12 * 128,
                seed=123,
                warm_queries=1,
                include_sensitivity=False,
            )

    def test_semantics_repeat_when_timing_and_memory_are_excluded(self) -> None:
        path = self.generate("repeat.f32")
        first = benchmark.execute_scale_worker(
            path,
            descriptor_rows=12 * 128,
            seed=123,
            warm_queries=2,
            include_sensitivity=False,
        )
        second = benchmark.execute_scale_worker(
            path,
            descriptor_rows=12 * 128,
            seed=123,
            warm_queries=2,
            include_sensitivity=False,
        )

        def semantic_projection(result: dict[str, object]) -> object:
            first_touch = result["first_touch"]
            warm = result["warm_queries"]
            return {
                "descriptor_rows": result["descriptor_rows"],
                "originals": result["synthetic_originals"],
                "diagnostics": result["descriptor_diagnostics"],
                "first_rank": first_touch["known_source_rank"],
                "first_work": first_touch["work_counts"],
                "warm_ranks": [item["known_source_rank"] for item in warm],
                "warm_work": [item["work_counts"] for item in warm],
            }

        self.assertEqual(semantic_projection(first), semantic_projection(second))


class ProgressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.worker_paths: list[Path] = []

    def fake_worker(self, descriptor_path: Path, **arguments) -> benchmark.WorkerOutcome:
        self.worker_paths.append(descriptor_path)
        return benchmark.WorkerOutcome(
            "completed",
            result={
                "descriptor_rows": arguments["descriptor_rows"],
                "synthetic_originals": arguments["descriptor_rows"] // 128,
            },
        )

    def configuration(self, maximum: int | None = None) -> benchmark.ScaleBenchmarkConfiguration:
        ladder = (12 * 128, 13 * 128, 14 * 128)
        return benchmark.ScaleBenchmarkConfiguration(
            scale_ladder=ladder,
            maximum_descriptor_rows=maximum or ladder[-1],
            warm_queries=2,
            seed=123,
            generation_chunk_rows=128,
        )

    def test_progression_runs_every_selected_scale_and_cleans_artifacts(self) -> None:
        report = benchmark.run_benchmark(
            self.configuration(), worker_runner=self.fake_worker
        )

        self.assertEqual(
            report["completion"]["completed_scales"],
            [12 * 128, 13 * 128, 14 * 128],
        )
        self.assertEqual(
            report["completion"]["stop_reason"], "COMPLETED_REQUESTED_SCALES"
        )
        self.assertEqual(len(self.worker_paths), 3)
        self.assertTrue(all(not path.exists() for path in self.worker_paths))
        self.assertTrue(
            all(
                item["temporary_descriptor_cleanup_complete"]
                for item in report["scales"]
            )
        )

    def test_cleanup_exception_is_structured_and_preserves_completed_scales(self) -> None:
        created_directories = []

        class CleanupFailsOnce:
            def __init__(self, **arguments) -> None:
                self.actual = tempfile.TemporaryDirectory(**arguments)
                self.name = self.actual.name
                self.cleanup_calls = 0

            def cleanup(self) -> None:
                self.cleanup_calls += 1
                if self.cleanup_calls == 1:
                    raise PermissionError("simulated cleanup failure")
                self.actual.cleanup()

        factory_calls = 0

        def directory_factory(**arguments):
            nonlocal factory_calls
            factory_calls += 1
            if factory_calls == 1:
                directory = tempfile.TemporaryDirectory(**arguments)
            else:
                directory = CleanupFailsOnce(**arguments)
            created_directories.append(directory)
            return directory

        report = benchmark.run_benchmark(
            self.configuration(),
            worker_runner=self.fake_worker,
            temporary_directory_factory=directory_factory,
        )

        self.assertEqual(report["completion"]["completed_scales"], [12 * 128])
        self.assertEqual(
            report["completion"]["stop_reason"],
            "TEMPORARY_DESCRIPTOR_CLEANUP_FAILED",
        )
        self.assertEqual(report["completion"]["next_unattempted_scale"], 13 * 128)
        failed_directory = created_directories[1]
        self.assertEqual(failed_directory.cleanup_calls, 2)
        self.assertFalse(Path(failed_directory.name).exists())

    def test_operator_maximum_stops_before_next_scale(self) -> None:
        report = benchmark.run_benchmark(
            self.configuration(13 * 128), worker_runner=self.fake_worker
        )

        self.assertEqual(report["completion"]["completed_scales"], [12 * 128, 13 * 128])
        self.assertEqual(report["completion"]["stop_reason"], "OPERATOR_MAXIMUM_REACHED")
        self.assertEqual(report["completion"]["next_unattempted_scale"], 14 * 128)

    def test_worker_failure_stops_and_preserves_completed_results(self) -> None:
        calls = 0

        def failing_worker(path: Path, **arguments) -> benchmark.WorkerOutcome:
            nonlocal calls
            calls += 1
            if calls == 2:
                return benchmark.WorkerOutcome(
                    "failed", stop_reason="WORKER_EXITED_UNEXPECTEDLY"
                )
            return self.fake_worker(path, **arguments)

        report = benchmark.run_benchmark(
            self.configuration(), worker_runner=failing_worker
        )

        self.assertEqual(report["completion"]["completed_scales"], [12 * 128])
        self.assertEqual(
            report["completion"]["stop_reason"], "WORKER_EXITED_UNEXPECTEDLY"
        )
        self.assertEqual(report["completion"]["next_unattempted_scale"], 13 * 128)

    def test_descriptor_creation_failure_is_reported(self) -> None:
        def failing_generator(*args, **kwargs):
            raise MemoryError

        report = benchmark.run_benchmark(
            self.configuration(),
            worker_runner=self.fake_worker,
            descriptor_generator=failing_generator,
        )

        self.assertEqual(report["completion"]["completed_scales"], [])
        self.assertEqual(
            report["completion"]["stop_reason"],
            "DESCRIPTOR_FILE_CREATION_FAILED",
        )

    def test_wall_time_budget_stop_uses_clock_without_sleeping(self) -> None:
        with mock.patch.object(
            benchmark,
            "run_correctness_guards",
            return_value={"passed": True, "checks": {}},
        ), mock.patch.object(
            benchmark, "perf_counter_ns", side_effect=[0, 2_000_000_000, 2_000_000_000]
        ):
            report = benchmark.run_benchmark(
                benchmark.ScaleBenchmarkConfiguration(
                    scale_ladder=(12 * 128,),
                    maximum_descriptor_rows=12 * 128,
                    warm_queries=2,
                    wall_time_seconds=1.0,
                ),
                worker_runner=self.fake_worker,
            )

        self.assertEqual(
            report["completion"]["stop_reason"], "WALL_TIME_BUDGET_EXHAUSTED"
        )
        self.assertEqual(report["completion"]["next_unattempted_scale"], 12 * 128)

    def test_resource_preflight_failure_is_graceful(self) -> None:
        with mock.patch.object(
            benchmark,
            "_resource_stop_reason",
            return_value=(
                "INSUFFICIENT_AVAILABLE_MEMORY",
                {
                    "available_physical_memory_bytes": 1,
                    "available_temporary_disk_bytes": 2,
                },
            ),
        ):
            report = benchmark.run_benchmark(
                self.configuration(), worker_runner=self.fake_worker
            )

        self.assertEqual(
            report["completion"]["stop_reason"],
            "INSUFFICIENT_AVAILABLE_MEMORY",
        )
        self.assertEqual(report["completion"]["completed_scales"], [])

    def test_correctness_guard_failure_stops_before_generation(self) -> None:
        with mock.patch.object(
            benchmark,
            "run_correctness_guards",
            return_value={"passed": False, "checks": {"synthetic": False}},
        ):
            report = benchmark.run_benchmark(
                self.configuration(), worker_runner=self.fake_worker
            )

        self.assertEqual(
            report["completion"]["stop_reason"], "CORRECTNESS_GUARD_FAILED"
        )
        self.assertEqual(report["completion"]["completed_scales"], [])
        self.assertEqual(self.worker_paths, [])

    def test_report_has_metadata_configuration_scales_and_completion(self) -> None:
        report = benchmark.run_benchmark(
            self.configuration(12 * 128), worker_runner=self.fake_worker
        )

        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(
            report["benchmark_semantics"],
            benchmark.SCALE_BENCHMARK_SEMANTICS,
        )
        self.assertIn("opencv_version", report["metadata"])
        self.assertEqual(
            report["metadata"]["opencv_configuration"][
                "requested_threads_per_worker"
            ],
            1,
        )
        self.assertEqual(report["configuration"]["descriptor_dimension"], 128)
        self.assertEqual(len(report["scales"]), 1)
        self.assertTrue(report["correctness_guard"]["passed"])
        self.assertNotIn("descriptor_path", json.dumps(report))


class ConfigurationAndOutputTests(unittest.TestCase):
    def test_invalid_configuration_values_are_rejected(self) -> None:
        invalid = (
            {"scale_ladder": ()},
            {"scale_ladder": (128, 128)},
            {"scale_ladder": (129,)},
            {"maximum_descriptor_rows": 0},
            {"warm_queries": 0},
            {"seed": -1},
            {"wall_time_seconds": 0.0},
            {"generation_chunk_rows": 0},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    benchmark.ScaleBenchmarkConfiguration(**values)

    def test_output_validation_is_json_no_clobber_and_existing_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            self.assertEqual(
                benchmark.validate_output_path(base / "report.json"),
                base.resolve() / "report.json",
            )
            with self.assertRaisesRegex(benchmark.ConfigurationError, "end with .json"):
                benchmark.validate_output_path(base / "report.txt")
            existing = base / "existing.json"
            existing.write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(benchmark.ConfigurationError, "must not already exist"):
                benchmark.validate_output_path(existing)
            with self.assertRaisesRegex(benchmark.ConfigurationError, "parent must exist"):
                benchmark.validate_output_path(base / "missing" / "report.json")

    def test_main_persists_report_without_descriptor_artifacts(self) -> None:
        report = {
            "schema_version": "1.0",
            "benchmark_semantics": benchmark.SCALE_BENCHMARK_SEMANTICS,
            "completion": {
                "last_completed_scale": 1,
                "stop_reason": "COMPLETED_REQUESTED_SCALES",
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / "report.json"
            stdout = StringIO()
            with mock.patch.object(benchmark, "run_benchmark", return_value=report):
                exit_code = benchmark.main(
                    ["--max-descriptor-rows", "10880", "--output", str(output)],
                    stdout=stdout,
                    stderr=StringIO(),
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
            self.assertEqual(json.loads(stdout.getvalue()), report)
            self.assertEqual([item.name for item in base.iterdir()], ["report.json"])

    def test_main_returns_zero_for_normal_terminal_reasons(self) -> None:
        for reason in (
            "COMPLETED_REQUESTED_SCALES",
            "OPERATOR_MAXIMUM_REACHED",
            "WALL_TIME_BUDGET_EXHAUSTED",
        ):
            with self.subTest(reason=reason):
                report = {
                    "completion": {
                        "last_completed_scale": 10_880,
                        "stop_reason": reason,
                    }
                }
                with mock.patch.object(
                    benchmark, "run_benchmark", return_value=report
                ):
                    exit_code = benchmark.main(
                        ["--max-descriptor-rows", "10880"],
                        stdout=StringIO(),
                        stderr=StringIO(),
                    )
                self.assertEqual(exit_code, 0)

    def test_main_partial_worker_failure_is_nonzero_and_report_is_preserved(self) -> None:
        report = {
            "schema_version": "1.0",
            "scales": [{"descriptor_rows": 10_880}],
            "completion": {
                "completed_scales": [10_880],
                "last_completed_scale": 10_880,
                "stop_reason": "WORKER_EXITED_UNEXPECTEDLY",
                "next_unattempted_scale": 65_408,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "partial.json"
            stdout = StringIO()
            with mock.patch.object(benchmark, "run_benchmark", return_value=report):
                exit_code = benchmark.main(
                    ["--max-descriptor-rows", "10880", "--output", str(output)],
                    stdout=stdout,
                    stderr=StringIO(),
                )

            self.assertEqual(exit_code, 1)
            self.assertEqual(json.loads(stdout.getvalue()), report)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)

    def test_main_cleanup_failure_is_nonzero(self) -> None:
        report = {
            "completion": {
                "last_completed_scale": 10_880,
                "stop_reason": "TEMPORARY_DESCRIPTOR_CLEANUP_FAILED",
            }
        }
        with mock.patch.object(benchmark, "run_benchmark", return_value=report):
            exit_code = benchmark.main(
                ["--max-descriptor-rows", "10880"],
                stdout=StringIO(),
                stderr=StringIO(),
            )

        self.assertEqual(exit_code, 1)

    def test_main_unknown_stop_reason_is_nonzero(self) -> None:
        report = {
            "completion": {
                "last_completed_scale": 10_880,
                "stop_reason": "UNRECOGNIZED_STOP_REASON",
            }
        }
        with mock.patch.object(benchmark, "run_benchmark", return_value=report):
            exit_code = benchmark.main(
                ["--max-descriptor-rows", "10880"],
                stdout=StringIO(),
                stderr=StringIO(),
            )

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()

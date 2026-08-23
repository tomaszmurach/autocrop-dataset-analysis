"""Bounded report and scale tests for the FLANN feasibility experiment."""

from __future__ import annotations

from argparse import Namespace
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock
import unittest

import numpy as np

from autocrop_analysis.audit import RootRole, SemanticReference
from autocrop_analysis.candidate_retrieval import RetrievalCandidate
from autocrop_analysis.candidate_retrieval_flann_benchmark import (
    CompactQuery,
    HistoricalPrediction,
    _shortlist_membership_signature,
    _source_ranking_signature,
    _validate_bounded_real_paths,
    _validate_output,
    _validate_prediction_artifact_coherence,
    _validate_predictions_path,
    _parse_historical_predictions,
    _prediction_consistency,
    build_parser,
    main as image_main,
    run_synthetic,
)
from autocrop_analysis.candidate_retrieval_flann_scale_benchmark import (
    FlannScaleConfiguration,
    _semantic_projection,
    execute_flann_scale_worker,
    run_benchmark,
    run_flann_in_fresh_process,
)
from autocrop_analysis.candidate_retrieval_scale_benchmark import (
    DESCRIPTORS_PER_ORIGINAL,
    WorkerOutcome,
    generate_descriptor_file,
)
from autocrop_analysis.cli import ConfigurationError


def original(name: str) -> SemanticReference:
    return SemanticReference(RootRole.ORIGINAL, name)


def crop(name: str) -> SemanticReference:
    return SemanticReference(RootRole.CROPPED, name)


def candidate(name: str, votes: int, distance: float) -> RetrievalCandidate:
    return RetrievalCandidate(original(name), votes, distance, distance)


class SyntheticReportTests(unittest.TestCase):
    def arguments(self, *extra: str) -> Namespace:
        return build_parser().parse_args(
            [
                "synthetic",
                "--corpus-size",
                "3",
                "--queries",
                "2",
                "--trees",
                "1,2",
                "--checks",
                "32,64",
                *extra,
            ]
        )

    def test_synthetic_report_contains_every_configuration_and_metric_namespace(self) -> None:
        report = run_synthetic(self.arguments())

        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(report["mode"], "SYNTHETIC_KNOWN_SOURCE")
        self.assertEqual(report["oracle"], "EXACT_BF_L2")
        self.assertEqual(len(report["evaluated_configurations"]), 4)
        self.assertEqual(
            {
                (item["parameters"]["trees"], item["parameters"]["checks"])
                for item in report["evaluated_configurations"]
            },
            {(1, 32), (1, 64), (2, 32), (2, 64)},
        )
        self.assertEqual(
            set(report["synthetic"]["exact_bf_known_source_recall"]),
            {"Recall@5", "Recall@10", "Recall@20", "Recall@50"},
        )
        first = report["evaluated_configurations"][0]
        self.assertIn("synthetic_known_source_recall", first["comparison"])
        self.assertIn("oracle_agreement_by_k", first["comparison"])
        self.assertIn(
            "exact_top_source_retention_rate",
            first["comparison"]["oracle_agreement_by_k"]["50"],
        )
        self.assertIn("individual_queries", first["comparison"])
        self.assertIn("same_index_source_rankings_stable", first["reproducibility"])
        for lifecycle in ("same_index", "save_reload", "same_process_rebuild"):
            for evidence in (
                "descriptor_rows",
                "descriptor_distances",
                "source_rankings",
                "shortlist_membership",
            ):
                self.assertIn(
                    f"{lifecycle}_{evidence}_stable", first["reproducibility"]
                )
        serialized = json.dumps(report)
        self.assertNotIn("artifact_path", serialized)
        self.assertNotIn("autocrop-flann-synthetic-", serialized)

    def test_main_persists_no_clobber_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            stdout = StringIO()
            exit_code = image_main(
                [
                    "synthetic",
                    "--corpus-size",
                    "2",
                    "--queries",
                    "1",
                    "--trees",
                    "1",
                    "--checks",
                    "32",
                    "--output",
                    str(output),
                ],
                stdout=stdout,
                stderr=StringIO(),
            )
            self.assertEqual(exit_code, 0)
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(persisted, json.loads(stdout.getvalue()))
            self.assertEqual(
                image_main(
                    [
                        "synthetic",
                        "--corpus-size",
                        "2",
                        "--queries",
                        "1",
                        "--trees",
                        "1",
                        "--checks",
                        "32",
                        "--output",
                        str(output),
                    ],
                    stdout=StringIO(),
                    stderr=StringIO(),
                ),
                2,
            )

    def test_required_safety_k_values_cannot_be_silently_omitted(self) -> None:
        stderr = StringIO()
        exit_code = image_main(
            [
                "synthetic",
                "--corpus-size",
                "2",
                "--queries",
                "1",
                "--trees",
                "1",
                "--checks",
                "32",
                "--k-values",
                "5,10,20",
            ],
            stdout=StringIO(),
            stderr=stderr,
        )
        self.assertEqual(exit_code, 2)
        self.assertIn("must include 5,10,20,50", stderr.getvalue())


class BoundedRealSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.cropped = self.base / "client-crops"
        self.results = self.base / "private-results"
        self.cropped.mkdir()
        self.results.mkdir()
        self.index = self.results / "candidate-index.private.json"
        self.index.write_text("{}", encoding="utf-8")
        self.predictions = self.results / "predictions.private.json"
        self.predictions.write_text('{"crops": []}', encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def argv(self, output: Path, predictions: Path | None = None) -> list[str]:
        return [
            "bounded-real",
            "--index",
            str(self.index),
            "--cropped",
            str(self.cropped),
            "--provenance-predictions",
            str(predictions or self.predictions),
            "--output",
            str(output),
        ]

    def arguments(self, output: Path, predictions: Path | None = None) -> Namespace:
        return build_parser().parse_args(self.argv(output, predictions))

    def test_bounded_real_requires_output(self) -> None:
        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "bounded-real",
                    "--index",
                    str(self.index),
                    "--cropped",
                    str(self.cropped),
                    "--provenance-predictions",
                    str(self.predictions),
                ]
            )
        self.assertIn("--output", stderr.getvalue())

    def test_private_output_contract_matches_gitignore_and_uses_actual_query_output(
        self,
    ) -> None:
        output = self.results / "bounded-flann.private.json"
        before = tuple(self.cropped.iterdir())
        validated_output = _validate_output(output, private=True)
        assert validated_output is not None
        paths, predictions = _validate_bounded_real_paths(
            self.arguments(output), validated_output
        )

        ignore_lines = (
            (Path(__file__).parents[1] / ".gitignore")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        self.assertIn("*.private.json", ignore_lines)
        self.assertTrue(paths.output.name.endswith(".private.json"))
        self.assertEqual(paths.output, validated_output)
        self.assertFalse(paths.output.is_relative_to(paths.cropped))
        self.assertEqual(predictions, self.predictions.resolve())
        self.assertEqual(tuple(self.cropped.iterdir()), before)

    def test_output_rejects_nonprivate_existing_missing_parent_and_cropped_paths(
        self,
    ) -> None:
        plain = self.results / "report.json"
        with self.assertRaisesRegex(ConfigurationError, "end with .private.json"):
            _validate_output(plain, private=True)

        existing = self.results / "existing.private.json"
        existing.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ConfigurationError, "must not already exist"):
            _validate_output(existing, private=True)

        missing_parent = self.base / "missing" / "report.private.json"
        with self.assertRaisesRegex(ConfigurationError, "parent must exist"):
            _validate_output(missing_parent, private=True)

        inside = self.cropped / "report.private.json"
        validated_inside = _validate_output(inside, private=True)
        assert validated_inside is not None
        with self.assertRaisesRegex(ConfigurationError, "outside the cropped root"):
            _validate_bounded_real_paths(self.arguments(inside), validated_inside)

    def test_prediction_input_requires_exact_private_suffix_regular_readable_file(
        self,
    ) -> None:
        wrong_suffix = self.results / "predictions.json"
        wrong_suffix.write_text('{"crops": []}', encoding="utf-8")
        with self.assertRaisesRegex(ConfigurationError, "end with .private.json"):
            _validate_predictions_path(wrong_suffix)

        directory = self.results / "directory.private.json"
        directory.mkdir()
        with self.assertRaisesRegex(ConfigurationError, "regular file"):
            _validate_predictions_path(directory)

        with mock.patch("pathlib.Path.open", side_effect=PermissionError):
            with self.assertRaisesRegex(ConfigurationError, "must be readable"):
                _validate_predictions_path(self.predictions)

    def test_success_stdout_is_aggregate_only_while_private_report_is_persisted(
        self,
    ) -> None:
        output = self.results / "bounded-output.private.json"
        private_crop = "private-client-crop.jpg"
        private_original = "private-client-original.png"
        report = {
            "configuration": {"query_count": 1},
            "evaluated_configurations": [
                {
                    "comparison": {
                        "individual_queries": [
                            {"crop": private_crop, "original": private_original}
                        ]
                    }
                }
            ],
            "bounded_real": {"prediction_decisions": {"MATCHED": 1}},
        }
        stdout = StringIO()
        with mock.patch(
            "autocrop_analysis.candidate_retrieval_flann_benchmark.run_bounded_real",
            return_value=report,
        ):
            exit_code = image_main(
                self.argv(output), stdout=stdout, stderr=StringIO()
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
        emitted = stdout.getvalue()
        self.assertIn("queries=1", emitted)
        self.assertNotIn(private_crop, emitted)
        self.assertNotIn(private_original, emitted)
        self.assertNotIn(str(output), emitted)
        self.assertNotIn("individual_queries", emitted)

    def test_unexpected_failure_does_not_emit_raw_private_path_message(self) -> None:
        output = self.results / "failure.private.json"
        secret = str(self.cropped / "private-client-crop.jpg")
        stderr = StringIO()
        with mock.patch(
            "autocrop_analysis.candidate_retrieval_flann_benchmark.run_bounded_real",
            side_effect=RuntimeError(secret),
        ):
            exit_code = image_main(
                self.argv(output), stdout=StringIO(), stderr=stderr
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("RuntimeError", stderr.getvalue())
        self.assertNotIn(secret, stderr.getvalue())
        self.assertFalse(output.exists())


class PredictionConsistencyTests(unittest.TestCase):
    def test_existing_provenance_report_shape_is_parsed_without_private_paths(self) -> None:
        report = {
            "crops": [
                {
                    "crop": {"relative_path": "matched.jpg"},
                    "decision": "MATCHED",
                    "ranked_candidates": [
                        {"original": {"relative_path": "source.png"}}
                    ],
                },
                {
                    "crop": {"relative_path": "ambiguous.jpg"},
                    "decision": "AMBIGUOUS",
                    "ranked_candidates": [
                        {"original": {"relative_path": "first.png"}},
                        {"original": {"relative_path": "second.png"}},
                    ],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            predictions = _parse_historical_predictions(path)
        self.assertEqual([item.decision for item in predictions], ["MATCHED", "AMBIGUOUS"])
        self.assertEqual(predictions[1].ranked_originals[1], original("second.png"))

    def test_duplicate_historical_prediction_crop_is_rejected(self) -> None:
        report = {
            "crops": [
                {
                    "crop": {"relative_path": "duplicate.jpg"},
                    "decision": "NO_MATCH",
                    "ranked_candidates": [],
                },
                {
                    "crop": {"relative_path": "duplicate.jpg"},
                    "decision": "MATCHED",
                    "ranked_candidates": [
                        {"original": {"relative_path": "source.png"}}
                    ],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.private.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "duplicate"):
                _parse_historical_predictions(path)

    def test_prediction_artifact_coherence_validates_crops_and_used_sources(
        self,
    ) -> None:
        loaded = SimpleNamespace(
            metadata=SimpleNamespace(
                originals=(
                    SimpleNamespace(reference=original("matched.png")),
                    SimpleNamespace(reference=original("first.png")),
                    SimpleNamespace(reference=original("second.png")),
                )
            )
        )
        queries = (
            CompactQuery(crop("matched.jpg"), np.empty((0, 128), np.float32), 0),
            CompactQuery(crop("ambiguous.jpg"), np.empty((0, 128), np.float32), 0),
        )
        valid = (
            HistoricalPrediction(
                crop("matched.jpg"), "MATCHED", (original("matched.png"),)
            ),
            HistoricalPrediction(
                crop("ambiguous.jpg"),
                "AMBIGUOUS",
                (original("first.png"), original("second.png")),
            ),
        )
        _validate_prediction_artifact_coherence(loaded, queries, valid)

        with self.assertRaisesRegex(ConfigurationError, "query corpus"):
            _validate_prediction_artifact_coherence(
                loaded,
                queries,
                (
                    HistoricalPrediction(
                        crop("missing.jpg"),
                        "MATCHED",
                        (original("matched.png"),),
                    ),
                ),
            )
        with self.assertRaisesRegex(ConfigurationError, "candidate-index corpus"):
            _validate_prediction_artifact_coherence(
                loaded,
                queries,
                (
                    HistoricalPrediction(
                        crop("ambiguous.jpg"),
                        "AMBIGUOUS",
                        (original("first.png"), original("missing.png")),
                    ),
                ),
            )
        with self.assertRaisesRegex(ConfigurationError, "duplicate"):
            _validate_prediction_artifact_coherence(
                loaded,
                queries,
                (valid[0], valid[0]),
            )

    def test_source_ranking_and_shortlist_signatures_are_independent(self) -> None:
        first = ((candidate("a.png", 2, 1.0), candidate("b.png", 1, 2.0)),)
        distance_changed = (
            (candidate("a.png", 2, 9.0), candidate("b.png", 1, 10.0)),
        )
        reordered = ((candidate("b.png", 2, 1.0), candidate("a.png", 1, 2.0)),)

        self.assertEqual(
            _source_ranking_signature(first),
            _source_ranking_signature(distance_changed),
        )
        self.assertNotEqual(
            _source_ranking_signature(first), _source_ranking_signature(reordered)
        )
        self.assertEqual(
            _shortlist_membership_signature(first, (2,)),
            _shortlist_membership_signature(reordered, (2,)),
        )

    def test_prediction_consistency_and_ambiguous_top_two_are_separate(self) -> None:
        queries = (
            CompactQuery(crop("matched.jpg"), np.empty((0, 128), np.float32), 0),
            CompactQuery(crop("ambiguous.jpg"), np.empty((0, 128), np.float32), 0),
        )
        exact = (
            (candidate("source.png", 2, 1.0), candidate("other.png", 1, 2.0)),
            (
                candidate("first.png", 2, 1.0),
                candidate("second.png", 1, 2.0),
            ),
        )
        flann = (
            (candidate("other.png", 2, 1.0), candidate("source.png", 1, 2.0)),
            (
                candidate("second.png", 2, 1.0),
                candidate("first.png", 1, 2.0),
            ),
        )
        predictions = (
            HistoricalPrediction(crop("matched.jpg"), "MATCHED", (original("source.png"),)),
            HistoricalPrediction(
                crop("ambiguous.jpg"),
                "AMBIGUOUS",
                (original("first.png"), original("second.png")),
            ),
        )
        report = _prediction_consistency(
            queries, exact, flann, predictions, (5, 10, 20, 50)
        )
        self.assertEqual(
            report["terminology"],
            "PREDICTION_CONSISTENCY_NOT_GROUND_TRUTH_ACCURACY",
        )
        self.assertEqual(report["matched_prediction_count"], 1)
        self.assertEqual(report["ambiguous_prediction_count"], 1)
        self.assertTrue(
            report["ambiguous_cases"][0]["prior_top_2_contained_at_k50"]["flann"]
        )
        self.assertEqual(report["matched_cases"][0]["rank_delta_vs_exact_bf"], 1)


class DescriptorScaleTests(unittest.TestCase):
    @staticmethod
    def fake_flann_result() -> dict[str, object]:
        return {
            "descriptor_rows": 10_880,
            "tree_count": 1,
            "configurations": [
                {
                    "parameters": {"trees": 1, "checks": 32, "neighbor_depth": 32},
                    "first_touch": {
                        "known_source_rank": 1,
                        "returned_candidates": 1,
                        "descriptor_neighbor_rows_sha256": "a" * 64,
                        "descriptor_neighbor_distances_sha256": "0" * 64,
                        "source_ranking_sha256": "b" * 64,
                        "shortlist_membership_sha256": "c" * 64,
                    },
                    "warm_queries": [
                        {
                            "known_source_rank": 1,
                            "descriptor_neighbor_rows_sha256": "d" * 64,
                            "descriptor_neighbor_distances_sha256": "1" * 64,
                            "source_ranking_sha256": "e" * 64,
                            "shortlist_membership_sha256": "f" * 64,
                        }
                    ],
                }
            ],
        }

    def test_scale_configuration_is_bounded_by_default_and_validates_grid(self) -> None:
        configuration = FlannScaleConfiguration()
        self.assertEqual(configuration.selected_scales, (10_880,))
        for values in (
            {"trees": ()},
            {"checks": (32, 32)},
            {"neighbor_depth": 0},
            {"fresh_process_repetitions": 0},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    FlannScaleConfiguration(**values)

    def test_worker_reports_lifecycle_timings_memory_and_cleanup(self) -> None:
        rows = 12 * DESCRIPTORS_PER_ORIGINAL
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "descriptors.f32"
            generate_descriptor_file(path, rows, 123, chunk_rows=128)
            from autocrop_analysis.candidate_retrieval_flann import hash_file

            descriptor_hash, _ = hash_file(path)
            result = execute_flann_scale_worker(
                path,
                descriptor_rows=rows,
                descriptor_sha256=descriptor_hash,
                seed=123,
                trees=1,
                checks=(32, 64),
                neighbor_depth=32,
                warm_queries=1,
            )
            self.assertEqual(result["descriptor_rows"], rows)
            self.assertEqual(len(result["configurations"]), 2)
            self.assertGreater(result["lifecycle"]["build_ns"], 0)
            self.assertGreater(result["lifecycle"]["artifact"]["artifact_bytes"], 0)
            self.assertTrue(result["temporary_artifact_cleanup_complete"])
            first = result["configurations"][0]["first_touch"]
            self.assertEqual(len(first["descriptor_neighbor_rows_sha256"]), 64)
            self.assertEqual(len(first["descriptor_neighbor_distances_sha256"]), 64)
            self.assertEqual(len(first["source_ranking_sha256"]), 64)
            self.assertEqual(len(first["shortlist_membership_sha256"]), 64)
            self.assertEqual(
                [item["label"] for item in result["memory"]["snapshots"]],
                [
                    "worker_start",
                    "after_read_only_memmap_open",
                    "before_flann_build",
                    "after_flann_build",
                    "after_flann_save",
                    "after_flann_load",
                    "final_state",
                ],
            )
            reproducibility = result["configurations"][0]["reproducibility"]
            for lifecycle in ("same_index", "save_reload"):
                for evidence in (
                    "descriptor_rows",
                    "descriptor_distances",
                    "source_ranking",
                    "shortlist_membership",
                ):
                    self.assertIn(
                        f"{lifecycle}_{evidence}_stable", reproducibility
                    )

    def test_fresh_process_rebuilds_produce_comparable_bounded_source_results(self) -> None:
        rows = 12 * DESCRIPTORS_PER_ORIGINAL
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "descriptors.f32"
            generate_descriptor_file(path, rows, 123, chunk_rows=128)
            from autocrop_analysis.candidate_retrieval_flann import hash_file

            descriptor_hash, _ = hash_file(path)
            results = []
            for _ in range(2):
                outcome = run_flann_in_fresh_process(
                    path,
                    descriptor_rows=rows,
                    descriptor_sha256=descriptor_hash,
                    seed=123,
                    trees=1,
                    checks=(32,),
                    neighbor_depth=32,
                    warm_queries=1,
                    timeout_seconds=30.0,
                )
                self.assertEqual(outcome.status, "completed")
                assert outcome.result is not None
                results.append(_semantic_projection(outcome.result))
            self.assertEqual(
                results[0]["configurations"][0]["parameters"],
                results[1]["configurations"][0]["parameters"],
            )
            self.assertIsNotNone(
                results[0]["configurations"][0]["first_known_source_rank"]
            )
            self.assertIsNotNone(
                results[1]["configurations"][0]["first_known_source_rank"]
            )

    def test_scale_cleanup_failure_is_structured_and_retried(self) -> None:
        class CleanupFailsOnce:
            def __init__(self, **arguments) -> None:
                self.actual = tempfile.TemporaryDirectory(**arguments)
                self.name = self.actual.name
                self.calls = 0

            def cleanup(self) -> None:
                self.calls += 1
                if self.calls == 1:
                    raise PermissionError("simulated cleanup failure")
                self.actual.cleanup()

        created = []

        def factory(**arguments):
            directory = CleanupFailsOnce(**arguments)
            created.append(directory)
            return directory

        def generator(path: Path, rows: int, seed: int, **arguments):
            path.write_bytes(b"bounded")
            return {"descriptor_rows": rows, "generation_seed": seed}

        exact = WorkerOutcome("completed", result={"descriptor_rows": 10_880})
        flann_result = self.fake_flann_result()
        with mock.patch(
            "autocrop_analysis.candidate_retrieval_flann_scale_benchmark._resource_stop_reason",
            return_value=(None, {}),
        ):
            report = run_benchmark(
                FlannScaleConfiguration(
                    trees=(1,),
                    checks=(32,),
                    warm_queries=1,
                    fresh_process_repetitions=1,
                ),
                exact_worker=lambda *args, **kwargs: exact,
                flann_worker=lambda *args, **kwargs: WorkerOutcome(
                    "completed", result=flann_result
                ),
                descriptor_generator=generator,
                temporary_directory_factory=factory,
            )
        self.assertEqual(
            report["completion"]["stop_reason"],
            "TEMPORARY_DESCRIPTOR_CLEANUP_FAILED",
        )
        self.assertEqual(report["completion"]["completed_scales"], [])
        self.assertEqual(created[0].calls, 2)
        self.assertFalse(Path(created[0].name).exists())

    def test_scale_report_schema_keeps_exact_and_flann_results_separate(self) -> None:
        def generator(path: Path, rows: int, seed: int, **arguments):
            path.write_bytes(b"bounded")
            return {"descriptor_rows": rows, "generation_seed": seed}

        with mock.patch(
            "autocrop_analysis.candidate_retrieval_flann_scale_benchmark._resource_stop_reason",
            return_value=(None, {"available_physical_memory_bytes": 1}),
        ):
            report = run_benchmark(
                FlannScaleConfiguration(
                    trees=(1,),
                    checks=(32,),
                    warm_queries=1,
                    fresh_process_repetitions=1,
                ),
                exact_worker=lambda *args, **kwargs: WorkerOutcome(
                    "completed", result={"descriptor_rows": 10_880}
                ),
                flann_worker=lambda *args, **kwargs: WorkerOutcome(
                    "completed", result=self.fake_flann_result()
                ),
                descriptor_generator=generator,
            )
        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(
            report["configuration"]["descriptor_scale_semantics"],
            "PERFORMANCE_EVIDENCE_NOT_KNOWN_SOURCE_RECALL",
        )
        self.assertEqual(
            report["completion"]["stop_reason"], "OPERATOR_MAXIMUM_REACHED"
        )
        scale = report["scales"][0]
        self.assertIn("exact_bf_reference", scale)
        self.assertIn("flann", scale)
        self.assertTrue(scale["temporary_descriptor_cleanup_complete"])
        self.assertNotIn("descriptor_path", json.dumps(report))
        self.assertEqual(
            set(scale["flann"][0]["reproducibility"]),
            {
                "fresh_process_descriptor_rows_stable",
                "fresh_process_descriptor_distances_stable",
                "fresh_process_source_rankings_stable",
                "fresh_process_shortlist_membership_stable",
            },
        )


if __name__ == "__main__":
    unittest.main()

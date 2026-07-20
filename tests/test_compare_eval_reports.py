from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARATOR = REPO_ROOT / "scripts" / "compare_eval_reports.py"


def make_report(case_results: dict[str, bool]) -> dict[str, Any]:
    passed = sum(case_results.values())
    failed = len(case_results) - passed
    total = len(case_results)
    return {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "knowledge_path": "/tmp/knowledge",
        "cases_file": "/tmp/cases.json",
        "chat_model": "fixture-chat-model",
        "embedding_model": "fixture-embedding-model",
        "summary": {
            "passed": passed,
            "failed": failed,
            "total": total,
            "pass_rate": passed / total if total else 0.0,
            "total_duration_seconds": 0.0,
            "failure_category_counts": {},
            "retrieval_hit_rate": 0.0,
            "citation_hit_rate": 0.0,
            "insufficient_context_accuracy": 0.0,
            "verification_status_accuracy": 0.0,
        },
        "cases": [
            {
                "id": case_id,
                "passed": case_passed,
                "question": f"Question for {case_id}",
                "answer_text": "Fixture answer",
                "expected_sources": [],
                "retrieved_sources": [],
                "verification_status": None,
                "expected_verification_status": None,
                "insufficient_context": None,
                "expected_insufficient_context": False,
                "cited_sources": [],
                "missing_required_terms": [],
                "exception_type": None,
                "exception_message": None,
                "issues": [],
                "failure_categories": [],
            }
            for case_id, case_passed in case_results.items()
        ],
    }


class CompareEvalReportsTests(unittest.TestCase):
    def run_comparator(
        self,
        baseline: dict[str, Any] | str,
        candidate: dict[str, Any] | str,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            baseline_path = temporary_path / "baseline.json"
            candidate_path = temporary_path / "candidate.json"
            self.write_fixture(baseline_path, baseline)
            self.write_fixture(candidate_path, candidate)

            return subprocess.run(
                [
                    sys.executable,
                    str(COMPARATOR),
                    str(baseline_path),
                    str(candidate_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

    def write_fixture(self, path: Path, report: dict[str, Any] | str) -> None:
        if isinstance(report, str):
            path.write_text(report, encoding="utf-8")
            return
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle)

    def test_identical_reports_exit_zero(self) -> None:
        report = make_report({"passing_case": True, "failing_case": False})

        result = self.run_comparator(report, report)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Baseline pass rate: 50.0%", result.stdout)
        self.assertIn("Candidate pass rate: 50.0%", result.stdout)
        self.assertIn("Unchanged passes: 1", result.stdout)
        self.assertIn("Unchanged failures: 1", result.stdout)

    def test_regression_exits_one_and_prints_case_id(self) -> None:
        baseline = make_report({"regressed_case": True})
        candidate = make_report({"regressed_case": False})

        result = self.run_comparator(baseline, candidate)

        self.assertEqual(result.returncode, 1)
        self.assertIn("Regressions (PASS -> FAIL): 1", result.stdout)
        self.assertIn("  - regressed_case", result.stdout)

    def test_improvement_exits_zero_and_prints_case_id(self) -> None:
        baseline = make_report({"improved_case": False})
        candidate = make_report({"improved_case": True})

        result = self.run_comparator(baseline, candidate)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Improvements (FAIL -> PASS): 1", result.stdout)
        self.assertIn("  - improved_case", result.stdout)

    def test_removed_case_exits_one_and_prints_case_id(self) -> None:
        baseline = make_report({"stable_case": True, "removed_case": True})
        candidate = make_report({"stable_case": True})

        result = self.run_comparator(baseline, candidate)

        self.assertEqual(result.returncode, 1)
        self.assertIn("Removed cases: 1", result.stdout)
        self.assertIn("  - removed_case", result.stdout)

    def test_added_case_without_regression_exits_zero(self) -> None:
        baseline = make_report({"stable_case": True})
        candidate = make_report({"stable_case": True, "added_case": False})

        result = self.run_comparator(baseline, candidate)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Added cases: 1", result.stdout)
        self.assertIn("  - added_case", result.stdout)
        self.assertIn("Regressions (PASS -> FAIL): 0", result.stdout)

    def test_malformed_or_invalid_report_exits_one_with_error(self) -> None:
        valid_report = make_report({"stable_case": True})
        invalid_reports = (
            ("malformed JSON", "{not valid JSON", "not valid JSON"),
            (
                "missing passed field",
                {
                    "summary": {
                        "passed": 0,
                        "failed": 1,
                        "total": 1,
                        "pass_rate": 0.0,
                    },
                    "cases": [{"id": "invalid_case"}],
                },
                ".passed must be a boolean",
            ),
        )

        for description, invalid_report, expected_error in invalid_reports:
            with self.subTest(description=description):
                result = self.run_comparator(valid_report, invalid_report)

                self.assertEqual(result.returncode, 1)
                self.assertIn("Error:", result.stderr)
                self.assertIn(expected_error, result.stderr)


if __name__ == "__main__":
    unittest.main()

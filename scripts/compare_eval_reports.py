#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class ReportValidationError(ValueError):
    pass


@dataclass(frozen=True)
class EvalReport:
    path: Path
    passed: int
    failed: int
    total: int
    pass_rate: float
    cases: dict[str, bool]


def require_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReportValidationError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def load_report(path: Path, label: str) -> EvalReport:
    prefix = f"{label} report {path}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as error:
        raise ReportValidationError(f"{prefix} does not exist") from error
    except OSError as error:
        raise ReportValidationError(f"could not read {prefix}: {error}") from error
    except UnicodeError as error:
        raise ReportValidationError(f"{prefix} is not valid UTF-8: {error}") from error
    except json.JSONDecodeError as error:
        raise ReportValidationError(
            f"{prefix} is not valid JSON at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error

    if not isinstance(data, dict):
        raise ReportValidationError(f"{prefix} root must be a JSON object")

    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise ReportValidationError(f"{prefix} field 'summary' must be an object")

    passed = require_nonnegative_int(summary.get("passed"), f"{prefix} summary.passed")
    failed = require_nonnegative_int(summary.get("failed"), f"{prefix} summary.failed")
    total = require_nonnegative_int(summary.get("total"), f"{prefix} summary.total")
    pass_rate_value = summary.get("pass_rate")
    if (
        isinstance(pass_rate_value, bool)
        or not isinstance(pass_rate_value, (int, float))
        or not 0.0 <= pass_rate_value <= 1.0
        or not math.isfinite(float(pass_rate_value))
    ):
        raise ReportValidationError(
            f"{prefix} summary.pass_rate must be a number between 0.0 and 1.0"
        )
    pass_rate = float(pass_rate_value)

    cases_data = data.get("cases")
    if not isinstance(cases_data, list):
        raise ReportValidationError(f"{prefix} field 'cases' must be an array")

    cases: dict[str, bool] = {}
    for index, case in enumerate(cases_data, start=1):
        case_prefix = f"{prefix} case #{index}"
        if not isinstance(case, dict):
            raise ReportValidationError(f"{case_prefix} must be an object")

        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ReportValidationError(
                f"{case_prefix}.id must be a non-empty string"
            )
        if case_id in cases:
            raise ReportValidationError(
                f"{prefix} contains duplicate case ID {case_id!r}"
            )

        case_passed = case.get("passed")
        if not isinstance(case_passed, bool):
            raise ReportValidationError(
                f"{case_prefix}.passed must be a boolean"
            )
        cases[case_id] = case_passed

    actual_passed = sum(cases.values())
    actual_failed = len(cases) - actual_passed
    if total != len(cases):
        raise ReportValidationError(
            f"{prefix} summary.total is {total}, but the report contains "
            f"{len(cases)} cases"
        )
    if passed != actual_passed or failed != actual_failed:
        raise ReportValidationError(
            f"{prefix} summary counts do not match case results "
            f"(summary={passed} passed/{failed} failed, "
            f"cases={actual_passed} passed/{actual_failed} failed)"
        )

    expected_pass_rate = passed / total if total else 0.0
    if not math.isclose(pass_rate, expected_pass_rate, rel_tol=0.0, abs_tol=1e-12):
        raise ReportValidationError(
            f"{prefix} summary.pass_rate is {pass_rate}, expected "
            f"{expected_pass_rate} from the case results"
        )

    return EvalReport(
        path=path,
        passed=passed,
        failed=failed,
        total=total,
        pass_rate=pass_rate,
        cases=cases,
    )


def print_case_ids(title: str, case_ids: list[str]) -> None:
    print(f"{title}: {len(case_ids)}")
    for case_id in case_ids:
        print(f"  - {case_id}")


def compare_reports(baseline: EvalReport, candidate: EvalReport) -> bool:
    baseline_ids = set(baseline.cases)
    candidate_ids = set(candidate.cases)
    shared_ids = baseline_ids & candidate_ids

    regressions = sorted(
        case_id
        for case_id in shared_ids
        if baseline.cases[case_id] and not candidate.cases[case_id]
    )
    improvements = sorted(
        case_id
        for case_id in shared_ids
        if not baseline.cases[case_id] and candidate.cases[case_id]
    )
    unchanged_passes = sum(
        baseline.cases[case_id] and candidate.cases[case_id]
        for case_id in shared_ids
    )
    unchanged_failures = sum(
        not baseline.cases[case_id] and not candidate.cases[case_id]
        for case_id in shared_ids
    )
    added = sorted(candidate_ids - baseline_ids)
    removed = sorted(baseline_ids - candidate_ids)
    pass_rate_delta = candidate.pass_rate - baseline.pass_rate

    print(
        f"Baseline pass rate: {baseline.pass_rate:.1%} "
        f"({baseline.passed}/{baseline.total})"
    )
    print(
        f"Candidate pass rate: {candidate.pass_rate:.1%} "
        f"({candidate.passed}/{candidate.total})"
    )
    print(f"Pass-rate delta: {pass_rate_delta * 100:+.1f} percentage points")
    print()
    print_case_ids("Regressions (PASS -> FAIL)", regressions)
    print_case_ids("Improvements (FAIL -> PASS)", improvements)
    print(f"Unchanged passes: {unchanged_passes}")
    print(f"Unchanged failures: {unchanged_failures}")
    print_case_ids("Added cases", added)
    print_case_ids("Removed cases", removed)

    return bool(regressions or removed)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two saved RAG evaluation reports by stable case ID."
    )
    parser.add_argument("baseline", type=Path, metavar="BASELINE")
    parser.add_argument("candidate", type=Path, metavar="CANDIDATE")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        baseline = load_report(args.baseline, "baseline")
        candidate = load_report(args.candidate, "candidate")
    except ReportValidationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    has_blocking_change = compare_reports(baseline, candidate)
    return 1 if has_blocking_change else 0


if __name__ == "__main__":
    sys.exit(main())

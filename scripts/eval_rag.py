#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tiny_rag_hybrid_smart_verified import RagConfig, run_verified_rag


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_PATH = REPO_ROOT / "docs"
DEFAULT_CASES_PATH = REPO_ROOT / "evals" / "rag_cases.json"


class BaselineReportError(ValueError):
    pass


@dataclass
class CaseDiagnostics:
    question: str
    expected_insufficient_context: bool
    expected_sources: list[str]
    expected_support_status: Any
    answer_text: str | None = None
    actual_insufficient_context: bool | None = None
    retrieved_sources: list[str] | None = None
    cited_sources: list[str] | None = None
    actual_support_status: str | None = None
    missing_required_terms: list[str] | None = None
    exception_type: str | None = None
    exception_message: str | None = None


def load_cases(cases_path: Path) -> list[dict[str, Any]]:
    with cases_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("Cases file must contain a JSON array.")

    validated_cases: list[dict[str, Any]] = []
    required_fields = {
        "id",
        "question",
        "expected_sources",
        "required_terms",
        "expect_insufficient_context",
        "expected_support_status",
    }

    for index, case in enumerate(data, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Case #{index} is not a JSON object.")

        missing_fields = sorted(required_fields - case.keys())
        if missing_fields:
            raise ValueError(
                f"Case #{index} is missing required fields: {', '.join(missing_fields)}"
            )

        validated_cases.append(case)

    return validated_cases


def normalize_source_name(value: str) -> str:
    return Path(value).name.lower()


def normalize_expected_support_statuses(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]

    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value

    raise ValueError(
        "expected_support_status must be a string, a list of strings, or null."
    )


def evaluate_case(
    case: dict[str, Any],
    knowledge_path: Path,
    config: RagConfig,
) -> tuple[bool, list[str], CaseDiagnostics]:
    diagnostics = CaseDiagnostics(
        question=case["question"],
        expected_insufficient_context=case["expect_insufficient_context"],
        expected_sources=[Path(value).name for value in case["expected_sources"]],
        expected_support_status=case["expected_support_status"],
    )

    result = run_verified_rag(
        path=knowledge_path,
        question=case["question"],
        config=config,
        show_progress=False,
    )

    issues: list[str] = []
    diagnostics.retrieved_sources = sorted(
        {ranked_chunk.chunk.source.name for ranked_chunk in result.ranked_chunks}
    )

    if not result.answer_data:
        issues.append("No answer data was returned.")
        return False, issues, diagnostics

    diagnostics.answer_text = result.answer_data["answer"]
    diagnostics.actual_insufficient_context = result.answer_data[
        "insufficient_context"
    ]
    diagnostics.missing_required_terms = []
    answer_text = diagnostics.answer_text.lower()
    retrieved_source_names = {
        ranked_chunk.chunk.source.name.lower() for ranked_chunk in result.ranked_chunks
    }

    cited_source_filenames: set[str] = set()
    for source_id in result.answer_data.get("cited_source_ids", []):
        source = result.sources.get(source_id)
        if source is not None:
            cited_source_filenames.add(source.file.name)
    diagnostics.cited_sources = sorted(cited_source_filenames)
    cited_source_names = {name.lower() for name in cited_source_filenames}

    if result.verification_data:
        diagnostics.actual_support_status = result.verification_data[
            "support_status"
        ]

    for expected_source in case["expected_sources"]:
        normalized_source = normalize_source_name(expected_source)
        if (
            normalized_source not in retrieved_source_names
            and normalized_source not in cited_source_names
        ):
            issues.append(
                "Expected source not retrieved or cited: "
                f"{expected_source} "
                f"(retrieved={sorted(retrieved_source_names)}, "
                f"cited={sorted(cited_source_names)})"
            )

    for required_term in case["required_terms"]:
        if required_term.lower() not in answer_text:
            diagnostics.missing_required_terms.append(required_term)
            issues.append(f"Required term missing from answer: {required_term}")

    if result.answer_data["insufficient_context"] != case["expect_insufficient_context"]:
        issues.append(
            "Unexpected insufficient_context value: "
            f"expected {case['expect_insufficient_context']}, "
            f"got {result.answer_data['insufficient_context']}"
        )

    expected_support_status = case["expected_support_status"]
    if expected_support_status is not None:
        if not result.verification_data:
            issues.append("No verification data was returned.")
        else:
            expected_statuses = normalize_expected_support_statuses(
                expected_support_status
            )
            actual_support_status = result.verification_data["support_status"]
            if actual_support_status not in expected_statuses:
                issues.append(
                    "Unexpected support status: "
                    f"expected one of {expected_statuses}, got {actual_support_status}"
                )

    return not issues, issues, diagnostics


def format_value(value: Any) -> str:
    if value is None:
        return "<unavailable>"
    return json.dumps(value, ensure_ascii=False)


def print_diagnostics(
    diagnostics: CaseDiagnostics,
    issues: list[str],
) -> None:
    print("  Details:")
    print(f"    Question: {diagnostics.question}")
    print(
        "    Answer: "
        f"{diagnostics.answer_text if diagnostics.answer_text is not None else '<unavailable>'}"
    )
    print(
        "    Insufficient context: "
        f"expected={format_value(diagnostics.expected_insufficient_context)}, "
        f"actual={format_value(diagnostics.actual_insufficient_context)}"
    )
    print(f"    Expected sources: {format_value(diagnostics.expected_sources)}")
    print(f"    Retrieved sources: {format_value(diagnostics.retrieved_sources)}")
    if diagnostics.cited_sources is not None:
        print(f"    Cited sources: {format_value(diagnostics.cited_sources)}")
    print(
        "    Verification status: "
        f"expected={format_value(diagnostics.expected_support_status)}, "
        f"actual={format_value(diagnostics.actual_support_status)}"
    )
    print(
        "    Missing required terms: "
        f"{format_value(diagnostics.missing_required_terms)}"
    )
    if diagnostics.exception_type is not None:
        print(
            "    Exception: "
            f"{diagnostics.exception_type}: {diagnostics.exception_message}"
        )
    if issues:
        print(f"    Issues: {format_value(issues)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small evaluation suite against the verified RAG pipeline."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_KNOWLEDGE_PATH,
        help=f"Knowledge path to search. Default: {DEFAULT_KNOWLEDGE_PATH}",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=f"JSON test cases file. Default: {DEFAULT_CASES_PATH}",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Print diagnostic details for passing cases as well as failures.",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        help="Compare this run with a previous JSON evaluation report.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        help="Write this run's evaluation report as JSON.",
    )
    return parser.parse_args()


def make_case_report(
    case_id: str,
    success: bool,
    diagnostics: CaseDiagnostics,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "passed": success,
        "verification_status": diagnostics.actual_support_status,
        "insufficient_context": diagnostics.actual_insufficient_context,
        "cited_sources": diagnostics.cited_sources,
    }


def make_report(
    passed: int,
    failed: int,
    duration_seconds: float,
    case_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    total = passed + failed
    return {
        "summary": {
            "passed": passed,
            "failed": failed,
            "total": total,
            "pass_rate": passed / total if total else 0.0,
            "total_duration_seconds": duration_seconds,
        },
        "cases": case_reports,
    }


def load_baseline_report(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except FileNotFoundError as error:
        raise BaselineReportError(f"baseline report does not exist: {path}") from error
    except OSError as error:
        raise BaselineReportError(f"could not read baseline report {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise BaselineReportError(
            f"baseline report is not valid JSON ({path}:{error.lineno}:{error.colno}): "
            f"{error.msg}"
        ) from error

    if not isinstance(report, dict):
        raise BaselineReportError("baseline report root must be a JSON object")
    summary = report.get("summary")
    cases = report.get("cases")
    if not isinstance(summary, dict):
        raise BaselineReportError("baseline report field 'summary' must be an object")
    for field in ("pass_rate", "total_duration_seconds"):
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BaselineReportError(
                f"baseline report summary.{field} must be a number"
            )
    if not 0.0 <= summary["pass_rate"] <= 1.0:
        raise BaselineReportError("baseline report summary.pass_rate must be between 0 and 1")
    if summary["total_duration_seconds"] < 0:
        raise BaselineReportError(
            "baseline report summary.total_duration_seconds cannot be negative"
        )
    if not isinstance(cases, list):
        raise BaselineReportError("baseline report field 'cases' must be an array")

    seen_ids: set[str] = set()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise BaselineReportError(f"baseline report case #{index} must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise BaselineReportError(
                f"baseline report case #{index}.id must be a non-empty string"
            )
        if case_id in seen_ids:
            raise BaselineReportError(f"baseline report has duplicate case id: {case_id}")
        seen_ids.add(case_id)
        if not isinstance(case.get("passed"), bool):
            raise BaselineReportError(
                f"baseline report case {case_id!r}.passed must be a boolean"
            )
        if "verification_status" not in case:
            raise BaselineReportError(
                f"baseline report case {case_id!r} is missing verification_status"
            )
        if not isinstance(case["verification_status"], (str, type(None))):
            raise BaselineReportError(
                f"baseline report case {case_id!r}.verification_status "
                "must be a string or null"
            )
        if "insufficient_context" not in case:
            raise BaselineReportError(
                f"baseline report case {case_id!r} is missing insufficient_context"
            )
        if not isinstance(case["insufficient_context"], (bool, type(None))):
            raise BaselineReportError(
                f"baseline report case {case_id!r}.insufficient_context "
                "must be a boolean or null"
            )
        cited_sources = case.get("cited_sources")
        if not (
            cited_sources is None
            or (
                isinstance(cited_sources, list)
                and all(isinstance(source, str) for source in cited_sources)
            )
        ):
            raise BaselineReportError(
                f"baseline report case {case_id!r}.cited_sources "
                "must be an array of strings or null"
            )
    return report


def changed_case_ids(
    previous_cases: dict[str, dict[str, Any]],
    current_cases: dict[str, dict[str, Any]],
    field: str,
) -> list[str]:
    def comparable(value: Any) -> Any:
        if field == "cited_sources" and isinstance(value, list):
            return sorted(value)
        return value

    return sorted(
        case_id
        for case_id in previous_cases.keys() & current_cases.keys()
        if comparable(previous_cases[case_id][field])
        != comparable(current_cases[case_id][field])
    )


def print_comparison(
    baseline: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    previous_summary = baseline["summary"]
    current_summary = current["summary"]
    previous_cases = {case["id"]: case for case in baseline["cases"]}
    current_cases = {case["id"]: case for case in current["cases"]}
    shared_ids = previous_cases.keys() & current_cases.keys()
    newly_passing = sorted(
        case_id
        for case_id in shared_ids
        if not previous_cases[case_id]["passed"] and current_cases[case_id]["passed"]
    )
    newly_failing = sorted(
        case_id
        for case_id in shared_ids
        if previous_cases[case_id]["passed"] and not current_cases[case_id]["passed"]
    )
    pass_rate_change = (
        current_summary["pass_rate"] - previous_summary["pass_rate"]
    )
    duration_change = (
        current_summary["total_duration_seconds"]
        - previous_summary["total_duration_seconds"]
    )

    print()
    print("Comparison with baseline:")
    print(f"  Previous pass rate: {previous_summary['pass_rate']:.1%}")
    print(f"  Current pass rate: {current_summary['pass_rate']:.1%}")
    print(f"  Pass-rate change: {pass_rate_change:+.1%}")
    print(
        "  Previous total duration: "
        f"{previous_summary['total_duration_seconds']:.3f}s"
    )
    print(f"  Current total duration: {current_summary['total_duration_seconds']:.3f}s")
    print(f"  Duration change: {duration_change:+.3f}s")
    print(f"  Newly passing cases: {format_value(newly_passing)}")
    print(f"  Newly failing cases: {format_value(newly_failing)}")
    print(
        "  Verification status changed: "
        f"{format_value(changed_case_ids(previous_cases, current_cases, 'verification_status'))}"
    )
    print(
        "  insufficient_context changed: "
        f"{format_value(changed_case_ids(previous_cases, current_cases, 'insufficient_context'))}"
    )
    print(
        "  Cited sources changed: "
        f"{format_value(changed_case_ids(previous_cases, current_cases, 'cited_sources'))}"
    )
    if newly_failing:
        print()
        print("REGRESSION: previously passing cases now fail")
        for case_id in newly_failing:
            print(f"  - {case_id}")
    return newly_failing


def main() -> None:
    args = parse_args()
    try:
        cases = load_cases(args.cases)
        baseline = (
            load_baseline_report(args.baseline_report)
            if args.baseline_report is not None
            else None
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
    config = RagConfig()

    passed = 0
    failed = 0
    case_reports: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    print(f"Knowledge path: {args.path}")
    print(f"Cases file: {args.cases}")
    print(f"Cases loaded: {len(cases)}")
    print()

    for case in cases:
        case_id = case["id"]
        diagnostics = CaseDiagnostics(
            question=case["question"],
            expected_insufficient_context=case["expect_insufficient_context"],
            expected_sources=[Path(value).name for value in case["expected_sources"]],
            expected_support_status=case["expected_support_status"],
        )
        try:
            success, issues, diagnostics = evaluate_case(case, args.path, config)
        except Exception as error:
            success = False
            issues = [f"Exception: {error}"]
            diagnostics.exception_type = type(error).__name__
            diagnostics.exception_message = str(error)

        if success:
            passed += 1
            print(f"PASS {case_id}")
            if args.show_details:
                print_diagnostics(diagnostics, issues)
        else:
            failed += 1
            print(f"FAIL {case_id}")
            print_diagnostics(diagnostics, issues)
        case_reports.append(make_case_report(case_id, success, diagnostics))

    duration_seconds = time.perf_counter() - started_at
    report = make_report(passed, failed, duration_seconds, case_reports)
    print()
    print(f"Summary: {passed} passed, {failed} failed, {len(cases)} total")

    if args.report_file is not None:
        try:
            args.report_file.parent.mkdir(parents=True, exist_ok=True)
            with args.report_file.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        except OSError as error:
            print(
                f"Error: could not write report {args.report_file}: {error}",
                file=sys.stderr,
            )
            sys.exit(2)

    regressions = print_comparison(baseline, report) if baseline is not None else []
    if failed or regressions:
        sys.exit(1)


if __name__ == "__main__":
    main()

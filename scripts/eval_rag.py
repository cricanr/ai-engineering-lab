#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tiny_rag_hybrid_smart_verified import RagConfig, run_verified_rag


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_PATH = REPO_ROOT / "docs"
DEFAULT_CASES_PATH = REPO_ROOT / "evals" / "rag_cases.json"


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    config = RagConfig()

    passed = 0
    failed = 0

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
            continue

        failed += 1
        print(f"FAIL {case_id}")
        print_diagnostics(diagnostics, issues)

    print()
    print(f"Summary: {passed} passed, {failed} failed, {len(cases)} total")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

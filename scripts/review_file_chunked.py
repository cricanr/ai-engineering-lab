#!/usr/bin/env python3
"""
Review a source file in chunks using a local Ollama model.

Why this exists:
- Small files can be sent to an LLM in one prompt.
- Large files should be split into chunks so the model does not lose context
  or exceed its context window.
- Each chunk is reviewed independently, then a final aggregation step merges
  the findings.

Run from repo root:
    uv run python scripts/review_file_chunked.py path/to/file.py

Optional:
    OLLAMA_MODEL=qwen3:14b-q4_K_M uv run python scripts/review_file_chunked.py path/to/file.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b-q4_K_M")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")


@dataclass(frozen=True)
class Chunk:
    index: int
    start_line: int
    end_line: int
    text: str


def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def chunk_by_lines(text: str, max_chars: int) -> list[Chunk]:
    lines = text.splitlines()
    chunks: list[Chunk] = []

    current_lines: list[str] = []
    current_start_line = 1
    current_char_count = 0

    for line_number, line in enumerate(lines, start=1):
        line_with_newline = line + "\n"

        if current_lines and current_char_count + len(line_with_newline) > max_chars:
            chunks.append(
                Chunk(
                    index=len(chunks) + 1,
                    start_line=current_start_line,
                    end_line=line_number - 1,
                    text="".join(current_lines),
                )
            )
            current_lines = []
            current_start_line = line_number
            current_char_count = 0

        current_lines.append(line_with_newline)
        current_char_count += len(line_with_newline)

    if current_lines:
        chunks.append(
            Chunk(
                index=len(chunks) + 1,
                start_line=current_start_line,
                end_line=len(lines),
                text="".join(current_lines),
            )
        )

    return chunks


def strip_thinking(text: str) -> str:
    """Remove common local-model thinking tags before JSON extraction."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def extract_first_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_thinking(text)

    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(cleaned)):
        char = cleaned[index]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                json_text = cleaned[start : index + 1]
                return json.loads(json_text)

    raise ValueError("Could not find a complete JSON object in model response.")


def call_ollama(prompt: str, model: str) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful senior software engineer. "
                    "Return only valid JSON. No markdown. No commentary outside JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not call Ollama at {OLLAMA_URL}. Is Ollama running?"
        ) from error

    try:
        return response_data["message"]["content"]
    except KeyError as error:
        raise RuntimeError(f"Unexpected Ollama response: {response_data}") from error


def review_chunk(path: Path, chunk: Chunk, total_chunks: int, model: str) -> dict[str, Any]:
    prompt = f"""
Review this source-code chunk.

File: {path}
Chunk: {chunk.index} of {total_chunks}
Line range: {chunk.start_line}-{chunk.end_line}

Focus on:
- correctness bugs
- edge cases
- error handling
- maintainability
- confusing design
- security only if clearly relevant

Return exactly this JSON shape:
{{
  "chunk": {chunk.index},
  "line_range": "{chunk.start_line}-{chunk.end_line}",
  "summary": "short summary of this chunk",
  "issues": [
    {{
      "severity": "low|medium|high",
      "line_hint": "line number or small range if possible",
      "title": "short issue title",
      "details": "what is wrong and why it matters",
      "suggestion": "concrete improvement"
    }}
  ],
  "questions": ["important unclear questions, if any"]
}}

Code chunk:
```text
{chunk.text}
```
""".strip()

    raw_response = call_ollama(prompt, model=model)
    return extract_first_json_object(raw_response)


def aggregate_reviews(path: Path, chunk_reviews: list[dict[str, Any]], model: str) -> dict[str, Any]:
    compact_reviews = json.dumps(chunk_reviews, indent=2)

    prompt = f"""
You reviewed a source file in chunks. Merge the chunk-level reviews into one concise final review.

File: {path}

Rules:
- Deduplicate repeated issues.
- Prefer high-impact issues.
- Keep line hints when available.
- Do not invent issues that are not supported by chunk reviews.

Return exactly this JSON shape:
{{
  "file": "{path}",
  "summary": "overall summary",
  "top_issues": [
    {{
      "severity": "low|medium|high",
      "line_hint": "line number or small range if possible",
      "title": "short issue title",
      "details": "what is wrong and why it matters",
      "suggestion": "concrete improvement"
    }}
  ],
  "recommended_next_steps": ["step 1", "step 2", "step 3"]
}}

Chunk reviews:
```json
{compact_reviews}
```
""".strip()

    raw_response = call_ollama(prompt, model=model)
    return extract_first_json_object(raw_response)


def print_final_review(review: dict[str, Any]) -> None:
    print("\n=== Chunked Code Review ===\n")
    print(f"File: {review.get('file', 'unknown')}\n")
    print(f"Summary: {review.get('summary', 'No summary returned.')}\n")

    issues = review.get("top_issues", [])
    if not issues:
        print("No major issues found.")
    else:
        print("Top issues:")
        for number, issue in enumerate(issues, start=1):
            severity = issue.get("severity", "unknown")
            line_hint = issue.get("line_hint", "unknown line")
            title = issue.get("title", "Untitled issue")
            details = issue.get("details", "")
            suggestion = issue.get("suggestion", "")

            print(f"\n{number}. [{severity}] {title}")
            print(f"   Line: {line_hint}")
            print(f"   Details: {details}")
            print(f"   Suggestion: {suggestion}")

    steps = review.get("recommended_next_steps", [])
    if steps:
        print("\nRecommended next steps:")
        for step in steps:
            print(f"- {step}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a source file in chunks using Ollama.")
    parser.add_argument("file", type=Path, help="Path to the source file to review.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="Approximate max characters per chunk. Default: 6000.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name. Default: {DEFAULT_MODEL}",
    )
    args = parser.parse_args()

    text = read_text_file(args.file)
    chunks = chunk_by_lines(text, max_chars=args.max_chars)

    if not chunks:
        print(f"File is empty: {args.file}")
        return

    print(f"Model: {args.model}")
    print(f"File: {args.file}")
    print(f"Chunks: {len(chunks)}")

    chunk_reviews: list[dict[str, Any]] = []
    for chunk in chunks:
        print(f"Reviewing chunk {chunk.index}/{len(chunks)} lines {chunk.start_line}-{chunk.end_line}...")
        try:
            chunk_reviews.append(review_chunk(args.file, chunk, len(chunks), model=args.model))
        except Exception as error:
            print(f"Failed to review chunk {chunk.index}: {error}", file=sys.stderr)
            raise

    print("Aggregating chunk reviews...")
    final_review = aggregate_reviews(args.file, chunk_reviews, model=args.model)
    print_final_review(final_review)


if __name__ == "__main__":
    main()

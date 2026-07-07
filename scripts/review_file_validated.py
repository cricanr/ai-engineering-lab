import json
import os
import sys
from typing import List

import requests
from pydantic import BaseModel, Field, ValidationError


DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b-q4_K_M")
OLLAMA_URL = "http://localhost:11434/api/generate"


class CodeIssue(BaseModel):
    severity: str = Field(description="One of: low, medium, high")
    line_hint: str = Field(description="Approximate line or area, e.g. 'function main' or 'line 12'")
    problem: str
    suggestion: str


class CodeReview(BaseModel):
    summary: str
    issues: List[CodeIssue]
    overall_risk: str = Field(description="One of: low, medium, high")


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def extract_json(text: str) -> str:
    """
    Best-effort extraction in case the model wraps JSON in explanations or markdown.
    """
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")

    return text[start : end + 1]


def build_prompt(code: str, previous_error: str | None = None) -> str:
    retry_note = ""
    if previous_error:
        retry_note = f"""
Your previous answer was invalid.

Validation error:
{previous_error}

Return corrected JSON only.
"""

    return f"""
You are a strict code review assistant.

Review the following Python code.

Return ONLY valid JSON.
Do not use markdown.
Do not add explanations outside JSON.

The JSON must match this shape exactly:

{{
  "summary": "short summary",
  "issues": [
    {{
      "severity": "low | medium | high",
      "line_hint": "approximate line or function",
      "problem": "what is wrong",
      "suggestion": "how to improve it"
    }}
  ],
  "overall_risk": "low | medium | high"
}}

{retry_note}

Code:
```python
{code}
```
"""


def review_code(code: str, retries: int = 2) -> CodeReview:
    last_error = None

    for attempt in range(retries + 1):
        prompt = build_prompt(code, previous_error=last_error)
        raw_response = call_ollama(prompt)

        try:
            json_text = extract_json(raw_response)
            data = json.loads(json_text)
            return CodeReview.model_validate(data)
        except (ValueError, json.JSONDecodeError, ValidationError) as error:
            last_error = str(error)

            if attempt == retries:
                print("Raw model response:")
                print(raw_response)
                raise RuntimeError(f"Could not get valid structured output: {error}") from error

    raise RuntimeError("Unexpected validation failure.")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/review_file_validated.py <path-to-python-file>")
        sys.exit(1)

    path = sys.argv[1]
    code = read_file(path)
    review = review_code(code)

    print("\nValidated code review:\n")
    print(f"Summary: {review.summary}")
    print(f"Overall risk: {review.overall_risk}")
    print("\nIssues:")

    if not review.issues:
        print("- No issues found.")
    else:
        for issue in review.issues:
            print(f"- [{issue.severity}] {issue.line_hint}")
            print(f"  Problem: {issue.problem}")
            print(f"  Suggestion: {issue.suggestion}")


if __name__ == "__main__":
    main()

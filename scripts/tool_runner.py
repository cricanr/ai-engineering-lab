import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import requests
from pydantic import BaseModel, Field, ValidationError


DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b-q4_K_M")
OLLAMA_URL = "http://localhost:11434/api/generate"
REPO_ROOT = Path.cwd().resolve()


class ToolCall(BaseModel):
    name: Literal["list_files", "show_file", "review_file"]
    args: dict[str, Any] = Field(default_factory=dict)


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
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")

    return text[start : end + 1]


def safe_repo_path(path: str) -> Path:
    target = (REPO_ROOT / path).resolve()

    if target != REPO_ROOT and REPO_ROOT not in target.parents:
        raise ValueError(f"Path is outside repo: {path}")

    return target


def list_files(directory: str = ".") -> str:
    root = safe_repo_path(directory)

    if not root.exists():
        return f"Directory does not exist: {directory}"

    if not root.is_dir():
        return f"Not a directory: {directory}"

    ignored_dirs = {".git", ".venv", "__pycache__", ".ruff_cache"}
    results: list[str] = []

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(REPO_ROOT)

        if any(part in ignored_dirs for part in relative.parts):
            continue

        if path.is_file():
            results.append(str(relative))

        if len(results) >= 80:
            results.append("... truncated ...")
            break

    if not results:
        return "No files found."

    return "\n".join(results)


def show_file(path: str) -> str:
    target = safe_repo_path(path)

    if not target.exists():
        return f"File does not exist: {path}"

    if not target.is_file():
        return f"Not a file: {path}"

    text = target.read_text(encoding="utf-8")

    if len(text) > 8000:
        text = text[:8000] + "\n\n... truncated ..."

    return text


def review_file(path: str) -> str:
    code = show_file(path)

    prompt = f"""
You are a senior software engineer.

Review this file briefly.
Focus on correctness, maintainability, and practical improvements.

File path:
{path}

File content starts below.

{code}

File content ends above.
"""

    return call_ollama(prompt)


def build_tool_choice_prompt(user_request: str) -> str:
    return f"""
You are a tool-selection assistant.

Choose exactly one tool to satisfy the user request.

Available tools:

1. list_files
Description: list files in the project.
Args:
{{"directory": "."}}

2. show_file
Description: show the contents of a file.
Args:
{{"path": "scripts/example.py"}}

3. review_file
Description: review a Python file and suggest improvements.
Args:
{{"path": "scripts/example.py"}}

Return ONLY valid JSON.
Do not use markdown.
Do not explain.

The JSON must match this shape:

{{
  "name": "list_files | show_file | review_file",
  "args": {{
    "directory": "optional directory",
    "path": "optional file path"
  }}
}}

User request:
{user_request}
"""


def choose_tool(user_request: str) -> ToolCall:
    prompt = build_tool_choice_prompt(user_request)
    raw_response = call_ollama(prompt)

    try:
        json_text = extract_json(raw_response)
        data = json.loads(json_text)
        return ToolCall.model_validate(data)
    except (ValueError, json.JSONDecodeError, ValidationError) as error:
        print("Raw model response:")
        print(raw_response)
        raise RuntimeError(f"Could not parse tool call: {error}") from error


def execute_tool(tool_call: ToolCall) -> str:
    if tool_call.name == "list_files":
        directory = tool_call.args.get("directory", ".")
        return list_files(directory)

    if tool_call.name == "show_file":
        path = tool_call.args.get("path")
        if not path:
            return "Missing required argument: path"
        return show_file(path)

    if tool_call.name == "review_file":
        path = tool_call.args.get("path")
        if not path:
            return "Missing required argument: path"
        return review_file(path)

    raise ValueError(f"Unknown tool: {tool_call.name}")


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: uv run python scripts/tool_runner.py "review scripts/ask_ollama.py"')
        sys.exit(1)

    user_request = " ".join(sys.argv[1:])

    tool_call = choose_tool(user_request)
    result = execute_tool(tool_call)

    print("\nSelected tool:")
    print(tool_call.model_dump_json(indent=2))

    print("\nTool result:\n")
    print(result)


if __name__ == "__main__":
    main()

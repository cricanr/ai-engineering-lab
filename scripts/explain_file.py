#!/usr/bin/env python3

import argparse
import ast
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from textwrap import dedent


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b-q4_K_M")
MAX_CHARS = 20_000


def format_ollama_error(error: urllib.error.HTTPError, model: str) -> str:
    body = error.read().decode("utf-8", errors="replace").strip()

    try:
        parsed = json.loads(body) if body else {}
    except json.JSONDecodeError:
        parsed = {}

    detail = parsed.get("error") or body or error.reason

    if error.code == 404 and "model" in str(detail).lower() and "not found" in str(detail).lower():
        return (
            f"Ollama model '{model}' is not available. {detail}. "
            f"Pull it first or set OLLAMA_MODEL to an installed model."
        )

    return f"Ollama request failed with HTTP {error.code}: {detail}"


def ask_ollama(prompt: str, model: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        },
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body)
            return parsed.get("response", "")
    except urllib.error.HTTPError as error:
        raise RuntimeError(format_ollama_error(error, model)) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_URL}. Is Ollama running?"
        ) from error


def read_source_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    content = path.read_text(encoding="utf-8")

    if len(content) > MAX_CHARS:
        content = content[:MAX_CHARS] + "\n\n# [TRUNCATED: file was too large]\n"

    return content


def build_prompt(file_path: Path, content: str) -> str:
    return dedent(f"""
        You are a senior backend engineer reviewing a source file.

        Explain the file for another experienced engineer.

        File path:
        {file_path}

        Source code:
        ```text
        {content}
        ```

        Return your answer using this structure:

        1. Purpose
        Explain what this file does in 2-4 sentences.

        2. Main flow
        Describe the execution flow step by step.

        3. Important functions/classes
        List the important functions/classes and what each one does.

        4. External dependencies
        Mention libraries, APIs, files, services, or environment assumptions.

        5. Risks and limitations
        Mention possible bugs, edge cases, missing error handling, or scaling issues.

        6. Suggested tests
        Suggest concrete tests that would give confidence this file works.

        Keep the explanation practical and avoid generic AI hype.
    """).strip()


def summarize_docstring(value: str | None) -> str:
    if not value:
        return "No module docstring is present, so the purpose must be inferred from the code structure."

    first_line = value.strip().splitlines()[0].strip()
    return first_line or "The module has a docstring, but it does not add much context."


def summarize_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    doc = ast.get_docstring(node)
    detail = doc.strip().splitlines()[0] if doc else "No docstring."
    kind = "async function" if isinstance(node, ast.AsyncFunctionDef) else "function"
    return f"- `{node.name}`: top-level {kind}. {detail}"


def summarize_class(node: ast.ClassDef) -> str:
    doc = ast.get_docstring(node)
    detail = doc.strip().splitlines()[0] if doc else "No docstring."
    methods = [
        child.name for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    method_text = ", ".join(f"`{name}`" for name in methods) if methods else "no methods"
    return f"- `{node.name}`: class. {detail} Methods: {method_text}."


def collect_imports(tree: ast.AST) -> list[str]:
    modules: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)

    seen: set[str] = set()
    ordered: list[str] = []
    for module in modules:
        root = module.split(".")[0]
        if root not in seen:
            seen.add(root)
            ordered.append(root)
    return ordered


def infer_risks(content: str, imports: list[str], has_main_guard: bool) -> list[str]:
    risks: list[str] = []

    if "urllib.request" in content or "requests" in imports:
        risks.append("It depends on external network access, so failures around timeouts, service availability, and bad responses need test coverage.")
    if "read_text(" in content:
        risks.append("It reads source as UTF-8 text only, so non-UTF-8 files will fail.")
    if "MAX_CHARS" in content:
        risks.append("Large files are truncated before analysis, which can hide important logic later in the file.")
    if not has_main_guard:
        risks.append("There is no obvious CLI entrypoint guard, so importing the module may have side effects depending on future edits.")

    if not risks:
        risks.append("The code looks straightforward, but edge cases around invalid input paths and unusual file contents should still be tested.")

    return risks


def build_offline_explanation(file_path: Path, content: str) -> str:
    if file_path.suffix != ".py":
        return dedent(f"""
            1. Purpose
            Offline fallback is available, but detailed static analysis is currently only implemented for Python files. `{file_path}` was treated as plain text.

            2. Main flow
            The file was read successfully, but no language-aware explanation was generated.

            3. Important functions/classes
            Not available in offline mode for this file type.

            4. External dependencies
            None detected by the generic fallback.

            5. Risks and limitations
            The explanation is limited because Ollama was unavailable and this file type does not have a parser-backed fallback.

            6. Suggested tests
            Start Ollama and rerun the command for a richer explanation, or extend the script with a parser for this file type.
        """).strip()

    try:
        tree = ast.parse(content)
    except SyntaxError as error:
        return dedent(f"""
            1. Purpose
            Offline fallback could not parse `{file_path}` as Python because it contains a syntax error at line {error.lineno}.

            2. Main flow
            No execution-flow summary was generated because parsing failed.

            3. Important functions/classes
            Not available due to the parse error.

            4. External dependencies
            Not available due to the parse error.

            5. Risks and limitations
            The file may be incomplete, invalid for the current Python version, or truncated.

            6. Suggested tests
            Fix the syntax issue first, then rerun the command or use the Ollama-backed mode for a richer explanation.
        """).strip()

    module_doc = ast.get_docstring(tree)
    imports = collect_imports(tree)
    top_level_functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]
    has_main_guard = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in tree.body
    )

    main_flow = [
        f"The module imports: {', '.join(f'`{name}`' for name in imports)}." if imports else "The module has no imports.",
        f"It defines {len(top_level_functions)} top-level function(s) and {len(classes)} class(es).",
        "Execution starts in a `main()` function protected by `if __name__ == \"__main__\":`." if has_main_guard else "There is no obvious CLI entrypoint guard at module level.",
    ]

    important_items = [summarize_function(node) for node in top_level_functions]
    important_items.extend(summarize_class(node) for node in classes)
    if not important_items:
        important_items.append("- No top-level functions or classes were found.")

    dependencies = imports[:] if imports else ["No external imports were detected."]
    if "localhost:11434" in content:
        dependencies.append("Local Ollama server at `http://localhost:11434`.")

    risks = infer_risks(content, imports, has_main_guard)

    tests = [
        "Add a happy-path test against a small Python file and assert that all six sections are present.",
        "Test a missing file path and confirm the CLI exits with a clear error.",
        "Test a non-UTF-8 or oversized file to verify current failure or truncation behavior is explicit.",
    ]
    if "urllib.request" in content:
        tests.append("Mock the Ollama HTTP call and cover success, HTTP error, and connection failure paths.")

    dependency_lines = "\n".join(f"- {item}" for item in dependencies)
    risk_lines = "\n".join(f"- {item}" for item in risks)
    test_lines = "\n".join(f"- {item}" for item in tests)

    return (
        "1. Purpose\n"
        f"{summarize_docstring(module_doc)}\n\n"
        "2. Main flow\n"
        f"{' '.join(main_flow)}\n\n"
        "3. Important functions/classes\n"
        f"{chr(10).join(important_items)}\n\n"
        "4. External dependencies\n"
        f"{dependency_lines}\n\n"
        "5. Risks and limitations\n"
        f"{risk_lines}\n\n"
        "6. Suggested tests\n"
        f"{test_lines}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain a source file using a local Ollama model."
    )
    parser.add_argument(
        "file",
        help="Path to the source file to explain.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip Ollama and use static offline analysis.",
    )
    parser.add_argument(
        "--require-ollama",
        action="store_true",
        help="Fail instead of falling back to offline analysis when Ollama is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_path = Path(args.file)

    try:
        content = read_source_file(file_path)
        if args.offline:
            answer = build_offline_explanation(file_path, content)
        else:
            prompt = build_prompt(file_path, content)
            try:
                answer = ask_ollama(prompt, args.model)
            except RuntimeError as error:
                if args.require_ollama or "Could not connect to Ollama" not in str(error):
                    raise

                print(
                    "Warning: Ollama is unavailable, using offline static analysis instead.",
                    file=sys.stderr,
                )
                answer = build_offline_explanation(file_path, content)
        print(answer.strip())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

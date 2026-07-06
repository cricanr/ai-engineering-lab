#!/usr/bin/env python3

import json
import os
import urllib.request
import urllib.error
from textwrap import dedent


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b-q4_K_M")


def format_ollama_error(error: urllib.error.HTTPError) -> str:
    body = error.read().decode("utf-8", errors="replace").strip()

    try:
        parsed = json.loads(body) if body else {}
    except json.JSONDecodeError:
        parsed = {}

    detail = parsed.get("error") or body or error.reason

    if error.code == 404 and "model" in str(detail).lower() and "not found" in str(detail).lower():
        return (
            f"Ollama model '{MODEL}' is not available. {detail}. "
            f"Pull it first or set OLLAMA_MODEL to an installed model."
        )

    return f"Ollama request failed with HTTP {error.code}: {detail}"


def ask_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
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
        raise RuntimeError(format_ollama_error(error)) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_URL}. Is Ollama running?"
        ) from error


TASK = """
Explain what RAG is and why it is useful for a senior backend engineer.
"""


PROMPTS = {
    "1_vague": TASK,

    "2_role_based": dedent(f"""
        You are a Staff Backend Engineer explaining AI engineering concepts
        to another experienced backend/platform engineer.

        Task:
        {TASK}
    """),

    "3_structured": dedent(f"""
        You are a Staff Backend Engineer.

        Explain RAG for a senior backend engineer.

        Use this structure:
        1. One-sentence definition
        2. Simple architecture flow
        3. Why it is useful
        4. Common failure modes
        5. One practical example
    """),

    "4_practical": dedent("""
        Explain RAG as if we are going to implement it in Python with:
        - local markdown files
        - embeddings
        - vector search
        - local Qwen through Ollama

        Avoid hype. Focus on practical engineering tradeoffs.
    """),

    "5_critical": dedent("""
        Explain RAG for a senior backend engineer.

        Include:
        - what problem it solves
        - where it often fails
        - why retrieval quality matters more than prompt wording
        - how to test whether it works
        - what you would log in production
    """),
}


def main() -> None:
    for name, prompt in PROMPTS.items():
        print("\n" + "=" * 80)
        print(f"PROMPT: {name}")
        print("=" * 80)
        print(prompt.strip())

        print("\n--- MODEL ANSWER ---\n")
        answer = ask_ollama(prompt)
        print(answer.strip())


if __name__ == "__main__":
    main()
    

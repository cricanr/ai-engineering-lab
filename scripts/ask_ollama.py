#!/usr/bin/env python3
"""Minimal dependency-free client for the local Ollama server."""

import json
import sys
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:14b-q4_K_M"


def generate(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read().decode("utf-8")

    result = json.loads(body)
    return result["response"]


def main() -> None:
    prompt = " ".join(sys.argv[1:]).strip() or "Say hello in one sentence."

    try:
        print(generate(prompt))
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach Ollama at {OLLAMA_URL}: {exc}") from exc


if __name__ == "__main__":
    main()

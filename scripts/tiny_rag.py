#!/usr/bin/env python3
"""
Tiny local RAG over plain-text / markdown files.

RAG = Retrieval-Augmented Generation:
1. Load documents from a folder.
2. Split them into chunks.
3. Search for the chunks most relevant to a question.
4. Send only those chunks to the local model.
5. Ask the model to answer using only the provided context.

Run from repo root:
    uv run python scripts/tiny_rag.py docs "What is this project about?"

Example:
    mkdir -p docs
    echo "# AI Engineering Lab\nThis repo contains local LLM exercises." > docs/intro.md
    uv run python scripts/tiny_rag.py docs "What does this repo contain?"

Optional:
    OLLAMA_MODEL=qwen3:14b-q4_K_M uv run python scripts/tiny_rag.py docs "Your question"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b-q4_K_M")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".toml"}


@dataclass(frozen=True)
class DocumentChunk:
    source: Path
    index: int
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class ScoredChunk:
    chunk: DocumentChunk
    score: float


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def find_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    if root.is_file():
        return [root]

    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            ignored_parts = {".git", ".venv", "__pycache__", "node_modules"}
            if ignored_parts.intersection(path.parts):
                continue
            files.append(path)

    return sorted(files)


def chunk_text(source: Path, text: str, max_chars: int) -> list[DocumentChunk]:
    lines = text.splitlines()
    chunks: list[DocumentChunk] = []

    current_lines: list[str] = []
    current_start_line = 1
    current_char_count = 0

    for line_number, line in enumerate(lines, start=1):
        line_with_newline = line + "\n"

        if current_lines and current_char_count + len(line_with_newline) > max_chars:
            chunks.append(
                DocumentChunk(
                    source=source,
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
            DocumentChunk(
                source=source,
                index=len(chunks) + 1,
                start_line=current_start_line,
                end_line=len(lines),
                text="".join(current_lines),
            )
        )

    return chunks


def load_chunks(root: Path, max_chars: int) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []

    for file_path in find_files(root):
        text = read_text_file(file_path)
        file_chunks = chunk_text(file_path, text, max_chars=max_chars)
        chunks.extend(file_chunks)

    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def term_frequency(tokens: list[str]) -> Counter[str]:
    return Counter(tokens)


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    shared_terms = set(left) & set(right)
    numerator = sum(left[term] * right[term] for term in shared_terms)

    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return numerator / (left_norm * right_norm)


def retrieve_relevant_chunks(
    question: str,
    chunks: list[DocumentChunk],
    top_k: int,
) -> list[ScoredChunk]:
    question_vector = term_frequency(tokenize(question))
    scored: list[ScoredChunk] = []

    for chunk in chunks:
        chunk_vector = term_frequency(tokenize(chunk.text))
        score = cosine_similarity(question_vector, chunk_vector)
        if score > 0:
            scored.append(ScoredChunk(chunk=chunk, score=score))

    return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


def call_ollama(prompt: str, model: str) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You answer questions using only the provided context. "
                    "If the context is insufficient, say what is missing. "
                    "Be concise and practical."
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


def build_prompt(question: str, scored_chunks: list[ScoredChunk]) -> str:
    context_blocks: list[str] = []

    for item in scored_chunks:
        chunk = item.chunk
        context_blocks.append(
            f"""Source: {chunk.source}
Chunk: {chunk.index}
Lines: {chunk.start_line}-{chunk.end_line}
Score: {item.score:.4f}

{chunk.text}
"""
        )

    context = "\n---\n".join(context_blocks)

    return f"""
Answer the question using only the context below.

Question:
{question}

Context:
{context}

Instructions:
- If the answer is present, answer directly.
- Mention the source file and line range you used.
- If the context does not contain enough information, say that clearly.
""".strip()


def print_retrieved_chunks(scored_chunks: list[ScoredChunk]) -> None:
    print("\nRetrieved chunks:")
    for number, item in enumerate(scored_chunks, start=1):
        chunk = item.chunk
        print(
            f"{number}. score={item.score:.4f} "
            f"{chunk.source} lines {chunk.start_line}-{chunk.end_line}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny local RAG over text files.")
    parser.add_argument("path", type=Path, help="Folder or file to search.")
    parser.add_argument("question", help="Question to answer.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2500,
        help="Approximate max characters per chunk. Default: 2500.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name. Default: {DEFAULT_MODEL}",
    )
    args = parser.parse_args()

    chunks = load_chunks(args.path, max_chars=args.max_chars)
    if not chunks:
        print(f"No supported text files found under: {args.path}")
        return

    scored_chunks = retrieve_relevant_chunks(
        question=args.question,
        chunks=chunks,
        top_k=args.top_k,
    )

    if not scored_chunks:
        print("No relevant chunks found.")
        print("Try a question with words that appear in the documents.")
        return

    print(f"Model: {args.model}")
    print(f"Loaded chunks: {len(chunks)}")
    print_retrieved_chunks(scored_chunks)

    prompt = build_prompt(args.question, scored_chunks)
    answer = call_ollama(prompt, model=args.model)

    print("\n=== Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()

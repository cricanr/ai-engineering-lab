#!/usr/bin/env python3
"""
Tiny local RAG with hybrid retrieval.

This upgrades the previous RAG scripts:

Session 8:
    keyword retrieval only

Session 9:
    embedding retrieval only

Session 10:
    hybrid retrieval:
        keyword retrieval + embedding retrieval -> fused ranking -> Qwen answer

Why hybrid?
- Keyword search is good for exact names, IDs, model names, function names, errors.
- Embedding search is good for meaning and paraphrases.
- Real RAG/search systems often combine both.

Run from repo root:
    uv run python scripts/tiny_rag_hybrid.py docs "What is RAG?"

Try exact-token questions:
    uv run python scripts/tiny_rag_hybrid.py docs "Where is qwen3:14b-q4_K_M mentioned?"

Try semantic questions:
    uv run python scripts/tiny_rag_hybrid.py docs "How do we avoid sending too much text to the model?"

Embedding model must exist in your Docker Ollama:
    docker exec -it ollama ollama pull embeddinggemma
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CHAT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b-q4_K_M")
DEFAULT_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "embeddinggemma")

OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
OLLAMA_EMBED_URL = os.environ.get("OLLAMA_EMBED_URL", "http://localhost:11434/api/embed")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".toml"}
IGNORED_PARTS = {".git", ".venv", "__pycache__", "node_modules", ".rag_cache"}


@dataclass(frozen=True)
class DocumentChunk:
    source: Path
    index: int
    start_line: int
    end_line: int
    text: str

    @property
    def cache_key(self) -> str:
        stable_content = (
            f"{self.source.as_posix()}|{self.index}|"
            f"{self.start_line}|{self.end_line}|{self.text}"
        )
        return hashlib.sha256(stable_content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RankedChunk:
    chunk: DocumentChunk
    keyword_score: float
    embedding_score: float
    keyword_rank: int | None
    embedding_rank: int | None
    hybrid_score: float


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def find_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    if root.is_file():
        if root.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {root.suffix}")
        return [root]

    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if IGNORED_PARTS.intersection(path.parts):
            continue

        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
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
        chunks.extend(chunk_text(file_path, text, max_chars=max_chars))

    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_:\-.]+", text.lower())


def term_frequency(tokens: list[str]) -> Counter[str]:
    return Counter(tokens)


def sparse_cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    shared_terms = set(left) & set(right)
    numerator = sum(left[term] * right[term] for term in shared_terms)

    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return numerator / (left_norm * right_norm)


def keyword_rank(question: str, chunks: list[DocumentChunk]) -> list[tuple[DocumentChunk, float]]:
    question_vector = term_frequency(tokenize(question))
    scored: list[tuple[DocumentChunk, float]] = []

    for chunk in chunks:
        chunk_vector = term_frequency(tokenize(chunk.text))
        score = sparse_cosine_similarity(question_vector, chunk_vector)
        scored.append((chunk, score))

    return sorted(scored, key=lambda item: item[1], reverse=True)


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)


def cache_path(root: Path, embed_model: str) -> Path:
    base = root if root.is_dir() else root.parent
    return base / ".rag_cache" / f"embeddings_{sanitize_model_name(embed_model)}.json"


def load_embedding_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        return {}

    cache: dict[str, list[float]] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, list):
            cache[key] = [float(number) for number in value]

    return cache


def save_embedding_cache(path: Path, cache: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache), encoding="utf-8")


def call_ollama_embed(texts: list[str], model: str) -> list[list[float]]:
    if not texts:
        return []

    payload = {
        "model": model,
        "input": texts,
    }

    request = urllib.request.Request(
        OLLAMA_EMBED_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama embedding request failed for model '{model}'. "
            f"Try: docker exec -it ollama ollama pull {model}\n"
            f"Server response: {message}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not call Ollama at {OLLAMA_EMBED_URL}. Is Ollama running?"
        ) from error

    embeddings = response_data.get("embeddings")
    if not isinstance(embeddings, list):
        raise RuntimeError(f"Unexpected Ollama embed response: {response_data}")

    return [[float(value) for value in embedding] for embedding in embeddings]


def embed_chunks(
    root: Path,
    chunks: list[DocumentChunk],
    embed_model: str,
    batch_size: int,
    refresh_cache: bool,
) -> dict[str, list[float]]:
    path = cache_path(root, embed_model)

    if refresh_cache:
        cache: dict[str, list[float]] = {}
    else:
        cache = load_embedding_cache(path)

    missing_chunks = [chunk for chunk in chunks if chunk.cache_key not in cache]

    if missing_chunks:
        print(f"Embedding missing chunks: {len(missing_chunks)}")
    else:
        print("All chunk embeddings loaded from cache.")

    for start in range(0, len(missing_chunks), batch_size):
        batch = missing_chunks[start : start + batch_size]
        print(f"Embedding batch {start // batch_size + 1}...")
        embeddings = call_ollama_embed([chunk.text for chunk in batch], model=embed_model)

        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} embeddings, received {len(embeddings)}."
            )

        for chunk, embedding in zip(batch, embeddings, strict=True):
            cache[chunk.cache_key] = embedding

        save_embedding_cache(path, cache)

    return cache


def dense_cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Vector dimensions differ: {len(left)} != {len(right)}")

    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return numerator / (left_norm * right_norm)


def embedding_rank(
    question: str,
    chunks: list[DocumentChunk],
    chunk_embeddings: dict[str, list[float]],
    embed_model: str,
) -> list[tuple[DocumentChunk, float]]:
    question_embedding = call_ollama_embed([question], model=embed_model)[0]

    scored: list[tuple[DocumentChunk, float]] = []
    for chunk in chunks:
        embedding = chunk_embeddings.get(chunk.cache_key)
        if embedding is None:
            continue

        score = dense_cosine_similarity(question_embedding, embedding)
        scored.append((chunk, score))

    return sorted(scored, key=lambda item: item[1], reverse=True)


def rank_lookup(ranking: list[tuple[DocumentChunk, float]]) -> dict[str, tuple[int, float]]:
    result: dict[str, tuple[int, float]] = {}

    for rank, (chunk, score) in enumerate(ranking, start=1):
        result[chunk.cache_key] = (rank, score)

    return result


def reciprocal_rank(rank: int | None, rank_constant: int) -> float:
    if rank is None:
        return 0.0

    return 1.0 / (rank_constant + rank)


def hybrid_rank(
    chunks: list[DocumentChunk],
    keyword_ranking: list[tuple[DocumentChunk, float]],
    embedding_ranking: list[tuple[DocumentChunk, float]],
    keyword_weight: float,
    embedding_weight: float,
    rank_constant: int,
    top_k: int,
) -> list[RankedChunk]:
    keyword_lookup = rank_lookup(keyword_ranking)
    embedding_lookup = rank_lookup(embedding_ranking)

    ranked: list[RankedChunk] = []

    for chunk in chunks:
        keyword_info = keyword_lookup.get(chunk.cache_key)
        embedding_info = embedding_lookup.get(chunk.cache_key)

        keyword_rank_value = keyword_info[0] if keyword_info else None
        embedding_rank_value = embedding_info[0] if embedding_info else None

        keyword_score = keyword_info[1] if keyword_info else 0.0
        embedding_score = embedding_info[1] if embedding_info else 0.0

        hybrid_score = (
            keyword_weight * reciprocal_rank(keyword_rank_value, rank_constant)
            + embedding_weight * reciprocal_rank(embedding_rank_value, rank_constant)
        )

        ranked.append(
            RankedChunk(
                chunk=chunk,
                keyword_score=keyword_score,
                embedding_score=embedding_score,
                keyword_rank=keyword_rank_value,
                embedding_rank=embedding_rank_value,
                hybrid_score=hybrid_score,
            )
        )

    return sorted(ranked, key=lambda item: item.hybrid_score, reverse=True)[:top_k]


def call_ollama_chat(
    prompt: str,
    model: str,
    *,
    json_mode: bool = False,
    think: bool | None = None,
) -> str:
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

    if json_mode:
        payload["format"] = "json"

    if think is not None:
        payload["think"] = think

    request = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not call Ollama at {OLLAMA_CHAT_URL}. Is Ollama running?"
        ) from error

    try:
        return response_data["message"]["content"]
    except KeyError as error:
        raise RuntimeError(f"Unexpected Ollama chat response: {response_data}") from error


def build_prompt(question: str, ranked_chunks: list[RankedChunk]) -> str:
    context_blocks: list[str] = []

    for item in ranked_chunks:
        chunk = item.chunk
        context_blocks.append(
            f"""Source: {chunk.source}
Chunk: {chunk.index}
Lines: {chunk.start_line}-{chunk.end_line}
Keyword rank: {item.keyword_rank}
Embedding rank: {item.embedding_rank}
Hybrid score: {item.hybrid_score:.6f}

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


def print_ranked_chunks(ranked_chunks: list[RankedChunk]) -> None:
    print("\nHybrid retrieved chunks:")
    for number, item in enumerate(ranked_chunks, start=1):
        chunk = item.chunk
        keyword_rank_text = item.keyword_rank if item.keyword_rank is not None else "-"
        embedding_rank_text = item.embedding_rank if item.embedding_rank is not None else "-"
        print(
            f"{number}. hybrid={item.hybrid_score:.6f} "
            f"keyword_rank={keyword_rank_text} "
            f"embedding_rank={embedding_rank_text} "
            f"keyword_score={item.keyword_score:.4f} "
            f"embedding_score={item.embedding_score:.4f} "
            f"{chunk.source} lines {chunk.start_line}-{chunk.end_line}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny local RAG with hybrid retrieval.")
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
        "--chat-model",
        default=DEFAULT_CHAT_MODEL,
        help=f"Ollama chat model name. Default: {DEFAULT_CHAT_MODEL}",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help=f"Ollama embedding model name. Default: {DEFAULT_EMBED_MODEL}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="How many chunks to embed per Ollama request. Default: 8.",
    )
    parser.add_argument(
        "--keyword-weight",
        type=float,
        default=1.0,
        help="Weight for keyword ranking. Default: 1.0.",
    )
    parser.add_argument(
        "--embedding-weight",
        type=float,
        default=1.0,
        help="Weight for embedding ranking. Default: 1.0.",
    )
    parser.add_argument(
        "--rank-constant",
        type=int,
        default=60,
        help="RRF rank constant. Higher means ranks decay slower. Default: 60.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore old cached embeddings and rebuild the cache.",
    )
    args = parser.parse_args()

    chunks = load_chunks(args.path, max_chars=args.max_chars)
    if not chunks:
        print(f"No supported text files found under: {args.path}")
        return

    print(f"Chat model: {args.chat_model}")
    print(f"Embedding model: {args.embed_model}")
    print(f"Loaded chunks: {len(chunks)}")

    try:
        print("Running keyword retrieval...")
        keyword_ranking = keyword_rank(args.question, chunks)

        print("Preparing embeddings...")
        chunk_embeddings = embed_chunks(
            root=args.path,
            chunks=chunks,
            embed_model=args.embed_model,
            batch_size=args.batch_size,
            refresh_cache=args.refresh_cache,
        )

        print("Running embedding retrieval...")
        embedding_ranking = embedding_rank(
            question=args.question,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            embed_model=args.embed_model,
        )

        ranked_chunks = hybrid_rank(
            chunks=chunks,
            keyword_ranking=keyword_ranking,
            embedding_ranking=embedding_ranking,
            keyword_weight=args.keyword_weight,
            embedding_weight=args.embedding_weight,
            rank_constant=args.rank_constant,
            top_k=args.top_k,
        )

        if not ranked_chunks:
            print("No chunks were retrieved.")
            return

        print_ranked_chunks(ranked_chunks)

        prompt = build_prompt(args.question, ranked_chunks)
        answer = call_ollama_chat(prompt, model=args.chat_model)

        print("\n=== Answer ===\n")
        print(answer)

    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

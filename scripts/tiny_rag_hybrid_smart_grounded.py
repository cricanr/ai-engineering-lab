#!/usr/bin/env python3
"""
Tiny local RAG with hybrid retrieval, smart chunking, and grounded citations.

This script builds on:
- scripts/smart_chunking.py
- scripts/tiny_rag_hybrid.py
- scripts/tiny_rag_hybrid_smart.py

New in this session:
- Retrieved chunks are labeled as S1, S2, S3...
- Qwen must answer in JSON.
- Qwen must cite source IDs instead of inventing file paths.
- Python maps source IDs back to real file paths and line ranges.
- If context is insufficient, Qwen should say so clearly.

Run from repo root:
    uv run python scripts/tiny_rag_hybrid_smart_grounded.py docs "What is RAG?"

Try codebase questions:
    uv run python scripts/tiny_rag_hybrid_smart_grounded.py scripts "How does hybrid_rank work?"
    uv run python scripts/tiny_rag_hybrid_smart_grounded.py scripts "Where is the Ollama embedding API called?"

Requirements:
- scripts/smart_chunking.py exists
- scripts/tiny_rag_hybrid.py exists
- embedding model exists in Docker Ollama:
    docker exec -it ollama ollama pull embeddinggemma
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smart_chunking import find_files, smart_chunk_file
from tiny_rag_hybrid import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DocumentChunk,
    RankedChunk,
    call_ollama_chat,
    embed_chunks,
    embedding_rank,
    hybrid_rank,
    keyword_rank,
    print_ranked_chunks,
)


@dataclass(frozen=True)
class GroundedSource:
    source_id: str
    file: Path
    line_range: str
    kind: str
    title: str


def smart_text_for_retrieval(kind: str, title: str, text: str) -> str:
    """
    Add structure metadata into the text that gets embedded/searched.

    This helps retrieval find a chunk by title or type.

    Example:
        Kind: python-symbol
        Title: def hybrid_rank

        <actual code>
    """

    return f"Kind: {kind}\nTitle: {title}\n\n{text}"


def load_smart_chunks(root: Path, max_chars: int) -> list[DocumentChunk]:
    """
    Load files and split them using scripts/smart_chunking.py.

    Then convert smart_chunking.Chunk into tiny_rag_hybrid.DocumentChunk so
    we can reuse the existing hybrid retrieval code.
    """

    document_chunks: list[DocumentChunk] = []

    for file_path in find_files(root):
        smart_chunks = smart_chunk_file(file_path, max_chars=max_chars)

        for index, smart_chunk in enumerate(smart_chunks, start=1):
            text = smart_text_for_retrieval(
                kind=smart_chunk.kind,
                title=smart_chunk.title,
                text=smart_chunk.text,
            )

            document_chunks.append(
                DocumentChunk(
                    source=smart_chunk.source,
                    index=index,
                    start_line=smart_chunk.start_line,
                    end_line=smart_chunk.end_line,
                    text=text,
                )
            )

    return document_chunks


def parse_kind_and_title(chunk: DocumentChunk) -> tuple[str, str]:
    lines = chunk.text.splitlines()
    kind = "unknown"
    title = "unknown"

    if lines and lines[0].startswith("Kind: "):
        kind = lines[0].replace("Kind: ", "", 1)

    if len(lines) > 1 and lines[1].startswith("Title: "):
        title = lines[1].replace("Title: ", "", 1)

    return kind, title


def make_grounded_sources(ranked_chunks: list[RankedChunk]) -> dict[str, GroundedSource]:
    sources: dict[str, GroundedSource] = {}

    for index, ranked_chunk in enumerate(ranked_chunks, start=1):
        chunk = ranked_chunk.chunk
        source_id = f"S{index}"
        kind, title = parse_kind_and_title(chunk)

        sources[source_id] = GroundedSource(
            source_id=source_id,
            file=chunk.source,
            line_range=f"{chunk.start_line}-{chunk.end_line}",
            kind=kind,
            title=title,
        )

    return sources


def build_grounded_prompt(
    question: str,
    ranked_chunks: list[RankedChunk],
    sources: dict[str, GroundedSource],
) -> str:
    context_blocks: list[str] = []

    for index, ranked_chunk in enumerate(ranked_chunks, start=1):
        source_id = f"S{index}"
        source = sources[source_id]
        chunk = ranked_chunk.chunk

        context_blocks.append(
            f"""Source ID: {source_id}
File: {source.file}
Lines: {source.line_range}
Kind: {source.kind}
Title: {source.title}
Keyword rank: {ranked_chunk.keyword_rank}
Embedding rank: {ranked_chunk.embedding_rank}
Hybrid score: {ranked_chunk.hybrid_score:.6f}

{chunk.text}
"""
        )

    context = "\n---\n".join(context_blocks)

    return f"""
You are answering a question using retrieved context chunks.

Question:
{question}

Retrieved context:
{context}

Rules:
- Use only the retrieved context.
- Do not invent facts outside the context.
- Cite source IDs like S1, S2, S3.
- Only cite source IDs that appear in the retrieved context.
- If the context is insufficient, set "insufficient_context" to true and explain what is missing.
- Return only valid JSON. No markdown. No text outside JSON.

Return exactly this JSON shape:
{{
  "answer": "direct answer, or explanation that context is insufficient",
  "insufficient_context": false,
  "confidence": "low|medium|high",
  "cited_source_ids": ["S1"],
  "missing_information": []
}}
""".strip()


def strip_thinking(text: str) -> str:
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
                return json.loads(cleaned[start : index + 1])

    raise ValueError("Could not find a complete JSON object in model response.")


def normalize_model_answer(data: dict[str, Any]) -> dict[str, Any]:
    answer = data.get("answer")
    insufficient_context = data.get("insufficient_context")
    confidence = data.get("confidence")
    cited_source_ids = data.get("cited_source_ids")
    missing_information = data.get("missing_information")

    if not isinstance(answer, str):
        answer = "The model did not return a valid answer string."

    if not isinstance(insufficient_context, bool):
        insufficient_context = False

    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"

    if not isinstance(cited_source_ids, list):
        cited_source_ids = []

    cited_source_ids = [item for item in cited_source_ids if isinstance(item, str)]

    if not isinstance(missing_information, list):
        missing_information = []

    missing_information = [
        item for item in missing_information if isinstance(item, str)
    ]

    return {
        "answer": answer,
        "insufficient_context": insufficient_context,
        "confidence": confidence,
        "cited_source_ids": cited_source_ids,
        "missing_information": missing_information,
    }


def filter_valid_source_ids(
    cited_source_ids: list[str],
    sources: dict[str, GroundedSource],
) -> list[str]:
    return [source_id for source_id in cited_source_ids if source_id in sources]


def print_grounded_answer(
    model_data: dict[str, Any],
    sources: dict[str, GroundedSource],
) -> None:
    valid_source_ids = filter_valid_source_ids(
        cited_source_ids=model_data["cited_source_ids"],
        sources=sources,
    )

    print("\n=== Grounded Answer ===\n")
    print(model_data["answer"])

    print("\nInsufficient context:")
    print("yes" if model_data["insufficient_context"] else "no")

    print("\nConfidence:")
    print(model_data["confidence"])

    if model_data["missing_information"]:
        print("\nMissing information:")
        for item in model_data["missing_information"]:
            print(f"- {item}")

    if valid_source_ids:
        print("\nSources:")
        for source_id in valid_source_ids:
            source = sources[source_id]
            print(
                f"- {source.source_id}: {source.file} "
                f"lines {source.line_range} "
                f"({source.kind}, {source.title})"
            )
    else:
        print("\nSources:")
        print("- No valid source IDs cited by the model.")


def print_smart_chunk_summary(chunks: list[DocumentChunk]) -> None:
    print("\nSmart chunks:")
    for number, chunk in enumerate(chunks, start=1):
        kind, title = parse_kind_and_title(chunk)

        print(
            f"{number}. {chunk.source} "
            f"lines {chunk.start_line}-{chunk.end_line} "
            f"kind={kind} title={title}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tiny local RAG with hybrid retrieval, smart chunking, and grounding."
    )
    parser.add_argument("path", type=Path, help="Folder or file to search.")
    parser.add_argument("question", help="Question to answer.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2500,
        help="Maximum characters per smart chunk before fallback splitting.",
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
    parser.add_argument(
        "--show-smart-chunks",
        action="store_true",
        help="Print all smart chunks before retrieval.",
    )
    parser.add_argument(
        "--show-raw-json",
        action="store_true",
        help="Print the raw parsed JSON returned by the model.",
    )
    args = parser.parse_args()

    chunks = load_smart_chunks(args.path, max_chars=args.max_chars)
    if not chunks:
        print(f"No supported text files found under: {args.path}")
        return

    print(f"Chat model: {args.chat_model}")
    print(f"Embedding model: {args.embed_model}")
    print(f"Loaded smart chunks: {len(chunks)}")

    if args.show_smart_chunks:
        print_smart_chunk_summary(chunks)

    try:
        print("Running keyword retrieval over smart chunks...")
        keyword_ranking = keyword_rank(args.question, chunks)

        print("Preparing smart chunk embeddings...")
        chunk_embeddings = embed_chunks(
            root=args.path,
            chunks=chunks,
            embed_model=args.embed_model,
            batch_size=args.batch_size,
            refresh_cache=args.refresh_cache,
        )

        print("Running embedding retrieval over smart chunks...")
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

        sources = make_grounded_sources(ranked_chunks)
        prompt = build_grounded_prompt(args.question, ranked_chunks, sources)

        raw_answer = call_ollama_chat(prompt, model=args.chat_model)
        parsed_answer = extract_first_json_object(raw_answer)
        model_data = normalize_model_answer(parsed_answer)

        if args.show_raw_json:
            print("\n=== Raw parsed JSON ===\n")
            print(json.dumps(model_data, indent=2))

        print_grounded_answer(model_data, sources)

    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

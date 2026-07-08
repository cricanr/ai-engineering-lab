#!/usr/bin/env python3
"""
Tiny local RAG with hybrid retrieval + smart chunking.

This script keeps scripts/tiny_rag_hybrid.py as a reference and changes only
the chunking strategy.

Old hybrid RAG:
    file -> size chunks -> keyword retrieval + embedding retrieval -> Qwen

Smart hybrid RAG:
    Markdown -> heading chunks
    Python   -> top-level function/class chunks
    Other    -> size chunks

Then:
    smart chunks -> keyword retrieval + embedding retrieval -> hybrid ranking -> Qwen

Run from repo root:
    uv run python scripts/tiny_rag_hybrid_smart.py docs "What is RAG?"

Try codebase questions:
    uv run python scripts/tiny_rag_hybrid_smart.py scripts "How does hybrid_rank work?"
    uv run python scripts/tiny_rag_hybrid_smart.py scripts "Where is the Ollama embedding API called?"

Requirements:
- scripts/smart_chunking.py exists
- scripts/tiny_rag_hybrid.py exists
- embedding model exists in Docker Ollama:
    docker exec -it ollama ollama pull embeddinggemma
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from smart_chunking import find_files, smart_chunk_file
from tiny_rag_hybrid import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DocumentChunk,
    build_prompt,
    call_ollama_chat,
    embed_chunks,
    embedding_rank,
    hybrid_rank,
    keyword_rank,
    print_ranked_chunks,
)


def smart_text_for_retrieval(
    kind: str,
    title: str,
    text: str,
) -> str:
    """
    Add structure metadata into the text that gets embedded/searched.

    Why:
    - The original chunk text is useful.
    - But the chunk title/kind is also useful context.
    - For example, a Python chunk titled "def hybrid_rank" should be searchable
      by the function name even if the function body does not repeat that name often.

    This means retrieval sees:

        Kind: python-symbol
        Title: def hybrid_rank

        <actual code>
    """

    return f"Kind: {kind}\nTitle: {title}\n\n{text}"


def load_smart_chunks(root: Path, max_chars: int) -> list[DocumentChunk]:
    """
    Load files and split them using scripts/smart_chunking.py.

    Then convert smart_chunking.Chunk into tiny_rag_hybrid.DocumentChunk so we
    can reuse the existing hybrid retrieval code without copying it.
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


def print_smart_chunk_summary(chunks: list[DocumentChunk]) -> None:
    print("\nSmart chunks:")
    for number, chunk in enumerate(chunks, start=1):
        first_lines = chunk.text.splitlines()
        kind = first_lines[0].replace("Kind: ", "") if first_lines else "unknown"
        title = first_lines[1].replace("Title: ", "") if len(first_lines) > 1 else "unknown"

        print(
            f"{number}. {chunk.source} "
            f"lines {chunk.start_line}-{chunk.end_line} "
            f"kind={kind} title={title}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tiny local RAG with hybrid retrieval and smart chunking."
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

        prompt = build_prompt(args.question, ranked_chunks)
        answer = call_ollama_chat(prompt, model=args.chat_model)

        print("\n=== Answer ===\n")
        print(answer)

    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate smart hybrid retrieval without running answer generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tiny_rag_hybrid import (
    DEFAULT_EMBED_MODEL,
    RankedChunk,
    embed_chunks,
    embedding_rank,
    hybrid_rank,
    keyword_rank,
)
from tiny_rag_hybrid_smart_verified import load_smart_chunks


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_PATH = REPO_ROOT / "docs"
DEFAULT_CASES_PATH = REPO_ROOT / "evals" / "rag_cases.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate smart hybrid retrieval without calling a chat model."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_KNOWLEDGE_PATH,
        help=f"Knowledge path to search. Default: {DEFAULT_KNOWLEDGE_PATH}",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=f"JSON evaluation cases. Default: {DEFAULT_CASES_PATH}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of hybrid retrieval results to evaluate. Default: 4.",
    )
    parser.add_argument(
        "--max-chunks-per-source",
        type=int,
        help="Optionally cap selected chunks from each source file.",
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Cases file must contain a JSON array.")

    cases: list[dict[str, Any]] = []
    for index, case in enumerate(data, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Case #{index} must be a JSON object.")
        for field in ("id", "question", "expected_sources"):
            if field not in case:
                raise ValueError(f"Case #{index} is missing required field: {field}")
        if not isinstance(case["id"], str) or not isinstance(case["question"], str):
            raise ValueError(f"Case #{index} id and question must be strings.")
        if not isinstance(case["expected_sources"], list) or not all(
            isinstance(source, str) for source in case["expected_sources"]
        ):
            raise ValueError(f"Case #{index} expected_sources must be an array of strings.")
        if case["expected_sources"]:
            cases.append(case)
    return cases


def normalized_filename(value: str | Path) -> str:
    return Path(value).name.lower()


def diversify_ranked_chunks(
    ranked_chunks: list[RankedChunk], top_k: int, max_chunks_per_source: int
) -> list[RankedChunk]:
    selected: list[RankedChunk] = []
    source_counts: dict[Path, int] = {}
    for ranked_chunk in ranked_chunks:
        source = ranked_chunk.chunk.source
        if source_counts.get(source, 0) >= max_chunks_per_source:
            continue
        selected.append(ranked_chunk)
        source_counts[source] = source_counts.get(source, 0) + 1
        if len(selected) == top_k:
            break
    return selected


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        print("Error: --top-k must be at least 1.", file=sys.stderr)
        sys.exit(2)
    if args.max_chunks_per_source is not None and args.max_chunks_per_source < 1:
        print("Error: --max-chunks-per-source must be at least 1.", file=sys.stderr)
        sys.exit(2)

    try:
        cases = load_cases(args.cases)
        chunks = load_smart_chunks(args.path, max_chars=2500)
        if not chunks:
            raise ValueError(f"No supported content found under: {args.path}")
        chunk_embeddings = embed_chunks(
            root=args.path,
            chunks=chunks,
            embed_model=DEFAULT_EMBED_MODEL,
            batch_size=8,
            refresh_cache=False,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)

    hits = 0
    reciprocal_rank_sum = 0.0
    print(f"Knowledge path: {args.path}")
    print(f"Cases file: {args.cases}")
    print(f"Retrieval cases: {len(cases)}")
    print(f"Top-k: {args.top_k}")
    if args.max_chunks_per_source is not None:
        print(f"Max chunks per source: {args.max_chunks_per_source}")
    print()

    for case in cases:
        expected_sources = {
            normalized_filename(source) for source in case["expected_sources"]
        }
        print(f"Case: {case['id']}")
        print(f"  Expected sources: {case['expected_sources']}")
        try:
            keyword_ranking = keyword_rank(case["question"], chunks)
            embedding_ranking = embedding_rank(
                question=case["question"],
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                embed_model=DEFAULT_EMBED_MODEL,
            )
            ranked_chunks = hybrid_rank(
                chunks=chunks,
                keyword_ranking=keyword_ranking,
                embedding_ranking=embedding_ranking,
                keyword_weight=1.0,
                embedding_weight=1.0,
                rank_constant=60,
                top_k=(
                    len(chunks)
                    if args.max_chunks_per_source is not None
                    else args.top_k
                ),
            )
            raw_ranked_sources = [
                item.chunk.source.name for item in ranked_chunks[: args.top_k]
            ]
            if args.max_chunks_per_source is not None:
                final_chunks = diversify_ranked_chunks(
                    ranked_chunks,
                    top_k=args.top_k,
                    max_chunks_per_source=args.max_chunks_per_source,
                )
            else:
                final_chunks = ranked_chunks
            ranked_sources = [item.chunk.source.name for item in final_chunks]
            first_expected_rank = next(
                (
                    rank
                    for rank, source in enumerate(ranked_sources, start=1)
                    if normalized_filename(source) in expected_sources
                ),
                None,
            )
            hit_at_k = first_expected_rank is not None
            reciprocal_rank = 1.0 / first_expected_rank if first_expected_rank else 0.0
        except Exception as error:
            raw_ranked_sources = []
            ranked_sources = []
            hit_at_k = False
            reciprocal_rank = 0.0
            print(f"  Error: {type(error).__name__}: {error}")

        hits += int(hit_at_k)
        reciprocal_rank_sum += reciprocal_rank
        print("  Raw ranked sources:")
        for rank, source in enumerate(raw_ranked_sources, start=1):
            print(f"  {rank}. {source}")
        if not raw_ranked_sources:
            print("    <none>")
        if args.max_chunks_per_source is not None:
            print("  Diversified ranked sources:")
        else:
            print("  Ranked sources:")
        for rank, source in enumerate(ranked_sources, start=1):
            print(f"  {rank}. {source}")
        if not ranked_sources:
            print("    <none>")
        unique_source_count = len(
            {normalized_filename(source) for source in ranked_sources}
        )
        print(f"  Unique source files: {unique_source_count}")
        print(f"  Hit@{args.top_k}: {str(hit_at_k).lower()}")
        print(f"  Reciprocal rank: {reciprocal_rank:.4f}")

    total = len(cases)
    retrieval_hit_rate = hits / total if total else 0.0
    mean_reciprocal_rank = reciprocal_rank_sum / total if total else 0.0
    print()
    print("Summary:")
    print(f"  Retrieval hit rate: {retrieval_hit_rate:.1%}")
    print(f"  Mean reciprocal rank: {mean_reciprocal_rank:.4f}")

    if hits != total:
        sys.exit(1)


if __name__ == "__main__":
    main()

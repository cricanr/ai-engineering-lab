#!/usr/bin/env python3
"""Compare hybrid retrieval with local chat-model reranking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from eval_retrieval import load_cases, normalized_filename
from tiny_rag_hybrid import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    RankedChunk,
    call_ollama_chat,
    embed_chunks,
    embedding_rank,
    hybrid_rank,
    keyword_rank,
)
from tiny_rag_hybrid_smart_verified import extract_first_json_object, load_smart_chunks


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_PATH = REPO_ROOT / "docs"
DEFAULT_CASES_PATH = REPO_ROOT / "evals" / "rag_cases.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare hybrid retrieval with local model reranking."
    )
    parser.add_argument("--path", type=Path, default=DEFAULT_KNOWLEDGE_PATH)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--candidate-k", type=int, default=8, help="Candidate count. Default: 8."
    )
    parser.add_argument("--top-k", type=int, default=4, help="Final count. Default: 4.")
    return parser.parse_args()


def build_reranking_prompt(question: str, candidates: list[RankedChunk]) -> str:
    blocks: list[str] = []
    for index, ranked_chunk in enumerate(candidates, start=1):
        chunk = ranked_chunk.chunk
        blocks.append(
            f"C{index}\n"
            f"Source: {chunk.source.name}\n"
            f"Lines: {chunk.start_line}-{chunk.end_line}\n"
            f"Content:\n{chunk.text}"
        )
    candidate_text = "\n\n---\n\n".join(blocks)
    return f"""
Rank the candidate chunks by relevance to the question.

Question:
{question}

Candidates:
{candidate_text}

Return only JSON with every candidate ID ordered from most to least relevant:
{{"candidate_ids": ["C1", "C2"]}}
""".strip()


def validated_candidate_order(data: dict[str, Any], count: int) -> list[int]:
    valid_ids = {f"C{index}": index - 1 for index in range(1, count + 1)}
    raw_ids = data.get("candidate_ids", [])
    if not isinstance(raw_ids, list):
        raw_ids = []

    ordered_indexes: list[int] = []
    seen: set[int] = set()
    for candidate_id in raw_ids:
        if not isinstance(candidate_id, str):
            continue
        index = valid_ids.get(candidate_id)
        if index is not None and index not in seen:
            ordered_indexes.append(index)
            seen.add(index)
    ordered_indexes.extend(index for index in range(count) if index not in seen)
    return ordered_indexes


def reciprocal_rank(
    ranked_chunks: list[RankedChunk], expected_sources: set[str]
) -> float:
    first_rank = next(
        (
            rank
            for rank, item in enumerate(ranked_chunks, start=1)
            if normalized_filename(item.chunk.source) in expected_sources
        ),
        None,
    )
    return 1.0 / first_rank if first_rank is not None else 0.0


def comparison_label(original: float, reranked: float) -> str:
    if reranked > original:
        return "improved"
    if reranked < original:
        return "worsened"
    return "kept the ranking unchanged"


def main() -> None:
    args = parse_args()
    if args.candidate_k < 1 or args.top_k < 1:
        print("Error: --candidate-k and --top-k must be at least 1.", file=sys.stderr)
        sys.exit(2)
    if args.top_k > args.candidate_k:
        print("Error: --top-k cannot exceed --candidate-k.", file=sys.stderr)
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

    original_hits = 0
    reranked_hits = 0
    original_rr_sum = 0.0
    reranked_rr_sum = 0.0

    print(f"Chat model: {DEFAULT_CHAT_MODEL}")
    print(f"Embedding model: {DEFAULT_EMBED_MODEL}")
    print(f"Candidate-k: {args.candidate_k}")
    print(f"Top-k: {args.top_k}")
    print()

    for case in cases:
        expected_sources = {
            normalized_filename(source) for source in case["expected_sources"]
        }
        try:
            candidates = hybrid_rank(
                chunks=chunks,
                keyword_ranking=keyword_rank(case["question"], chunks),
                embedding_ranking=embedding_rank(
                    question=case["question"],
                    chunks=chunks,
                    chunk_embeddings=chunk_embeddings,
                    embed_model=DEFAULT_EMBED_MODEL,
                ),
                keyword_weight=1.0,
                embedding_weight=1.0,
                rank_constant=60,
                top_k=args.candidate_k,
            )
            raw_response = call_ollama_chat(
                build_reranking_prompt(case["question"], candidates),
                model=DEFAULT_CHAT_MODEL,
                json_mode=True,
                think=False,
            )
            parsed_response = extract_first_json_object(raw_response)
            order = validated_candidate_order(parsed_response, len(candidates))
            original_results = candidates[: args.top_k]
            reranked_results = [candidates[index] for index in order[: args.top_k]]
            original_rr = reciprocal_rank(original_results, expected_sources)
            reranked_rr = reciprocal_rank(reranked_results, expected_sources)
        except Exception as error:
            print(f"Case: {case['id']}")
            print(f"  Error: {type(error).__name__}: {error}")
            print()
            continue

        original_hits += int(original_rr > 0.0)
        reranked_hits += int(reranked_rr > 0.0)
        original_rr_sum += original_rr
        reranked_rr_sum += reranked_rr
        print(f"Case: {case['id']}")
        print(
            "  Original hybrid-ranked filenames: "
            f"{[item.chunk.source.name for item in original_results]}"
        )
        print(
            f"  Reranked filenames: {[item.chunk.source.name for item in reranked_results]}"
        )
        print(f"  Original reciprocal rank: {original_rr:.4f}")
        print(f"  Reranked reciprocal rank: {reranked_rr:.4f}")
        print(f"  Result: {comparison_label(original_rr, reranked_rr)}")
        print()

    total = len(cases)
    print("Summary:")
    print(f"  Original retrieval hit rate: {original_hits / total if total else 0.0:.1%}")
    print(f"  Reranked retrieval hit rate: {reranked_hits / total if total else 0.0:.1%}")
    print(f"  Original mean reciprocal rank: {original_rr_sum / total if total else 0.0:.4f}")
    print(f"  Reranked mean reciprocal rank: {reranked_rr_sum / total if total else 0.0:.4f}")


if __name__ == "__main__":
    main()

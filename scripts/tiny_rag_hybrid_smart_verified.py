#!/usr/bin/env python3
"""
Tiny local RAG with smart chunking, hybrid retrieval, grounded citations,
and answer verification.

This builds on:
- scripts/smart_chunking.py
- scripts/tiny_rag_hybrid.py
- scripts/tiny_rag_hybrid_smart_grounded.py

New in this session:
- First pass: answer the question using retrieved context.
- Second pass: verify whether the answer is actually supported by the cited chunks.
- The verifier returns structured JSON:
    supported | partially_supported | unsupported

Run from repo root:
    uv run python scripts/tiny_rag_hybrid_smart_verified.py docs "What is RAG?"

Try codebase questions:
    uv run python scripts/tiny_rag_hybrid_smart_verified.py scripts "How does hybrid_rank work?"

Try insufficient-context behavior:
    uv run python scripts/tiny_rag_hybrid_smart_verified.py docs "What is the capital of Japan?"
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
    text: str


@dataclass(frozen=True)
class RagConfig:
    top_k: int = 4
    max_chars: int = 2500
    chat_model: str = DEFAULT_CHAT_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    batch_size: int = 8
    keyword_weight: float = 1.0
    embedding_weight: float = 1.0
    rank_constant: int = 60
    refresh_cache: bool = False
    show_smart_chunks: bool = False
    show_raw_json: bool = False
    skip_verification: bool = False


@dataclass(frozen=True)
class VerifiedRagResult:
    chunks: list[DocumentChunk]
    ranked_chunks: list[RankedChunk]
    sources: dict[str, GroundedSource]
    answer_data: dict[str, Any] | None
    verification_data: dict[str, Any] | None


def smart_text_for_retrieval(kind: str, title: str, text: str) -> str:
    return f"Kind: {kind}\nTitle: {title}\n\n{text}"


def load_smart_chunks(root: Path, max_chars: int) -> list[DocumentChunk]:
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
            text=chunk.text,
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

        context_blocks.append(
            f"""Source ID: {source_id}
File: {source.file}
Lines: {source.line_range}
Kind: {source.kind}
Title: {source.title}
Keyword rank: {ranked_chunk.keyword_rank}
Embedding rank: {ranked_chunk.embedding_rank}
Hybrid score: {ranked_chunk.hybrid_score:.6f}

{source.text}
"""
        )

    context = "\n---\n".join(context_blocks)

    return f"""
You are a context-only answering system.

Question:
{question}

Retrieved context:
{context}

Rules:
- Use only the retrieved context.
- Do not use training data or prior knowledge.
- Do not guess.
- Do not invent facts outside the context.
- Cite source IDs like S1, S2, S3.
- Only cite source IDs that appear in the retrieved context.
- If the context is insufficient, set "insufficient_context" to true and explain what is missing.
- Every factual claim in the answer must be supported by at least one cited source ID.
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


def normalize_answer_json(data: dict[str, Any]) -> dict[str, Any]:
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


def build_verification_prompt(
    question: str,
    answer_data: dict[str, Any],
    sources: dict[str, GroundedSource],
) -> str:
    valid_source_ids = filter_valid_source_ids(
        cited_source_ids=answer_data["cited_source_ids"],
        sources=sources,
    )

    cited_context_blocks: list[str] = []
    for source_id in valid_source_ids:
        source = sources[source_id]
        cited_context_blocks.append(
            f"""Source ID: {source.source_id}
File: {source.file}
Lines: {source.line_range}
Kind: {source.kind}
Title: {source.title}

{source.text}
"""
        )

    cited_context = "\n---\n".join(cited_context_blocks)

    if not cited_context:
        cited_context = "No valid cited context was provided."

    return f"""
You are a strict answer verifier.

Your job:
Check whether the answer is supported by the cited context only.

Question:
{question}

Answer to verify:
{answer_data["answer"]}

Insufficient context flag from answer:
{answer_data["insufficient_context"]}

Cited context:
{cited_context}

Rules:
- Use only the cited context.
- Do not use training data or prior knowledge.
- Do not verify whether the answer is globally true.
- Verify only whether the answer is supported by the cited context.
- If the answer contains factual claims not supported by cited context, list them.
- If the answer correctly says context is insufficient, that can be "supported".
- Return only valid JSON. No markdown. No text outside JSON.

Return exactly this JSON shape:
{{
  "support_status": "supported|partially_supported|unsupported",
  "verdict": "short human-readable verdict",
  "unsupported_claims": [],
  "source_ids_checked": ["S1"],
  "recommended_fix": "empty string if no fix needed"
}}
""".strip()


def normalize_verification_json(data: dict[str, Any]) -> dict[str, Any]:
    support_status = data.get("support_status")
    verdict = data.get("verdict")
    unsupported_claims = data.get("unsupported_claims")
    source_ids_checked = data.get("source_ids_checked")
    recommended_fix = data.get("recommended_fix")

    if support_status not in {"supported", "partially_supported", "unsupported"}:
        support_status = "unsupported"

    if not isinstance(verdict, str):
        verdict = "The verifier did not return a valid verdict."

    if not isinstance(unsupported_claims, list):
        unsupported_claims = []

    unsupported_claims = [
        item for item in unsupported_claims if isinstance(item, str)
    ]

    if not isinstance(source_ids_checked, list):
        source_ids_checked = []

    source_ids_checked = [
        item for item in source_ids_checked if isinstance(item, str)
    ]

    if not isinstance(recommended_fix, str):
        recommended_fix = ""

    return {
        "support_status": support_status,
        "verdict": verdict,
        "unsupported_claims": unsupported_claims,
        "source_ids_checked": source_ids_checked,
        "recommended_fix": recommended_fix,
    }


def print_grounded_answer(
    answer_data: dict[str, Any],
    sources: dict[str, GroundedSource],
) -> None:
    valid_source_ids = filter_valid_source_ids(
        cited_source_ids=answer_data["cited_source_ids"],
        sources=sources,
    )

    print("\n=== Grounded Answer ===\n")
    print(answer_data["answer"])

    print("\nInsufficient context:")
    print("yes" if answer_data["insufficient_context"] else "no")

    print("\nConfidence:")
    print(answer_data["confidence"])

    if answer_data["missing_information"]:
        print("\nMissing information:")
        for item in answer_data["missing_information"]:
            print(f"- {item}")

    print("\nSources:")
    if valid_source_ids:
        for source_id in valid_source_ids:
            source = sources[source_id]
            print(
                f"- {source.source_id}: {source.file} "
                f"lines {source.line_range} "
                f"({source.kind}, {source.title})"
            )
    else:
        print("- No valid source IDs cited by the model.")


def print_verification_result(
    verification_data: dict[str, Any],
    sources: dict[str, GroundedSource],
) -> None:
    checked_ids = filter_valid_source_ids(
        cited_source_ids=verification_data["source_ids_checked"],
        sources=sources,
    )

    print("\n=== Verification ===\n")
    print(f"Support status: {verification_data['support_status']}")
    print(f"Verdict: {verification_data['verdict']}")

    if verification_data["unsupported_claims"]:
        print("\nUnsupported claims:")
        for claim in verification_data["unsupported_claims"]:
            print(f"- {claim}")

    if verification_data["recommended_fix"]:
        print("\nRecommended fix:")
        print(verification_data["recommended_fix"])

    print("\nSources checked:")
    if checked_ids:
        for source_id in checked_ids:
            source = sources[source_id]
            print(f"- {source.source_id}: {source.file} lines {source.line_range}")
    else:
        print("- No valid source IDs checked.")


def print_smart_chunk_summary(chunks: list[DocumentChunk]) -> None:
    print("\nSmart chunks:")
    for number, chunk in enumerate(chunks, start=1):
        kind, title = parse_kind_and_title(chunk)

        print(
            f"{number}. {chunk.source} "
            f"lines {chunk.start_line}-{chunk.end_line} "
            f"kind={kind} title={title}"
        )


def run_verified_rag(
    path: Path,
    question: str,
    config: RagConfig,
    *,
    show_progress: bool = False,
) -> VerifiedRagResult:
    chunks = load_smart_chunks(path, max_chars=config.max_chars)
    if not chunks:
        return VerifiedRagResult(
            chunks=[],
            ranked_chunks=[],
            sources={},
            answer_data=None,
            verification_data=None,
        )

    if show_progress:
        print(f"Chat/verifier model: {config.chat_model}")
        print(f"Embedding model: {config.embed_model}")
        print(f"Loaded smart chunks: {len(chunks)}")

        if config.show_smart_chunks:
            print_smart_chunk_summary(chunks)

        print("Running keyword retrieval over smart chunks...")

    keyword_ranking = keyword_rank(question, chunks)

    if show_progress:
        print("Preparing smart chunk embeddings...")

    chunk_embeddings = embed_chunks(
        root=path,
        chunks=chunks,
        embed_model=config.embed_model,
        batch_size=config.batch_size,
        refresh_cache=config.refresh_cache,
    )

    if show_progress:
        print("Running embedding retrieval over smart chunks...")

    embedding_ranking = embedding_rank(
        question=question,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        embed_model=config.embed_model,
    )

    ranked_chunks = hybrid_rank(
        chunks=chunks,
        keyword_ranking=keyword_ranking,
        embedding_ranking=embedding_ranking,
        keyword_weight=config.keyword_weight,
        embedding_weight=config.embedding_weight,
        rank_constant=config.rank_constant,
        top_k=config.top_k,
    )

    if not ranked_chunks:
        return VerifiedRagResult(
            chunks=chunks,
            ranked_chunks=[],
            sources={},
            answer_data=None,
            verification_data=None,
        )

    sources = make_grounded_sources(ranked_chunks)
    answer_prompt = build_grounded_prompt(question, ranked_chunks, sources)

    raw_answer = call_ollama_chat(answer_prompt, model=config.chat_model)
    parsed_answer = extract_first_json_object(raw_answer)
    answer_data = normalize_answer_json(parsed_answer)

    verification_data: dict[str, Any] | None = None
    if not config.skip_verification:
        verification_prompt = build_verification_prompt(
            question=question,
            answer_data=answer_data,
            sources=sources,
        )
        raw_verification = call_ollama_chat(verification_prompt, model=config.chat_model)
        parsed_verification = extract_first_json_object(raw_verification)
        verification_data = normalize_verification_json(parsed_verification)

    return VerifiedRagResult(
        chunks=chunks,
        ranked_chunks=ranked_chunks,
        sources=sources,
        answer_data=answer_data,
        verification_data=verification_data,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tiny local RAG with hybrid retrieval, smart chunking, grounding, "
            "and verification."
        )
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
        help="Print the raw parsed answer and verification JSON.",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Only generate the grounded answer; skip verifier pass.",
    )
    args = parser.parse_args()

    try:
        config = RagConfig(
            top_k=args.top_k,
            max_chars=args.max_chars,
            chat_model=args.chat_model,
            embed_model=args.embed_model,
            batch_size=args.batch_size,
            keyword_weight=args.keyword_weight,
            embedding_weight=args.embedding_weight,
            rank_constant=args.rank_constant,
            refresh_cache=args.refresh_cache,
            show_smart_chunks=args.show_smart_chunks,
            show_raw_json=args.show_raw_json,
            skip_verification=args.skip_verification,
        )
        result = run_verified_rag(
            path=args.path,
            question=args.question,
            config=config,
            show_progress=True,
        )

        if not result.chunks:
            print(f"No supported text files found under: {args.path}")
            return

        if not result.ranked_chunks:
            print("No chunks were retrieved.")
            return

        print_ranked_chunks(result.ranked_chunks)

        if config.show_raw_json and result.answer_data is not None:
            print("\n=== Raw parsed answer JSON ===\n")
            print(json.dumps(result.answer_data, indent=2))

        if result.answer_data is not None:
            print_grounded_answer(result.answer_data, result.sources)

        if config.skip_verification:
            return

        if config.show_raw_json and result.verification_data is not None:
            print("\n=== Raw parsed verification JSON ===\n")
            print(json.dumps(result.verification_data, indent=2))

        if result.verification_data is not None:
            print_verification_result(result.verification_data, result.sources)

    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

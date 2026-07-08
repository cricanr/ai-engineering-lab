#!/usr/bin/env python3
"""
Smart chunking demo for the AI engineering lab.

Examples:
    uv run python scripts/smart_chunking.py docs
    uv run python scripts/smart_chunking.py scripts/tiny_rag_hybrid.py
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml"}
IGNORED_FOLDERS = {".git", ".venv", "__pycache__", "node_modules", ".rag_cache"}
MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")


@dataclass(frozen=True)
class Chunk:
    source: Path
    kind: str
    title: str
    start_line: int
    end_line: int
    text: str


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def find_files(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {path.suffix}")
        return [path]

    files: list[Path] = []

    def walk(folder: Path) -> None:
        for child in sorted(folder.iterdir()):
            if child.name in IGNORED_FOLDERS:
                continue
            if child.is_dir():
                walk(child)
                continue
            if child.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(child)

    walk(path)
    return files


def chunk_by_size(source: Path, text: str, max_chars: int, kind: str) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if not text:
        return []

    lines = text.splitlines(keepends=True)
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_start_line = 1
    current_char_count = 0

    def flush(end_line: int) -> None:
        nonlocal current_lines, current_start_line, current_char_count
        if not current_lines:
            return
        chunks.append(
            Chunk(
                source=source,
                kind=kind,
                title=f"{kind} chunk {len(chunks) + 1}",
                start_line=current_start_line,
                end_line=end_line,
                text="".join(current_lines),
            )
        )
        current_lines = []
        current_char_count = 0

    for line_number, line in enumerate(lines, start=1):
        if current_lines and current_char_count + len(line) > max_chars:
            flush(line_number - 1)
            current_start_line = line_number

        if len(line) <= max_chars:
            if not current_lines:
                current_start_line = line_number
            current_lines.append(line)
            current_char_count += len(line)
            continue

        if current_lines:
            flush(line_number - 1)

        current_start_line = line_number
        for start in range(0, len(line), max_chars):
            piece = line[start : start + max_chars]
            chunks.append(
                Chunk(
                    source=source,
                    kind=kind,
                    title=f"{kind} chunk {len(chunks) + 1}",
                    start_line=line_number,
                    end_line=line_number,
                    text=piece,
                )
            )

    if current_lines:
        flush(len(lines))

    return chunks


def split_if_large(chunk: Chunk, max_chars: int) -> list[Chunk]:
    if len(chunk.text) <= max_chars:
        return [chunk]

    parts = chunk_by_size(chunk.source, chunk.text, max_chars, chunk.kind)
    if len(parts) == 1:
        part = parts[0]
        return [
            Chunk(
                source=chunk.source,
                kind=chunk.kind,
                title=chunk.title,
                start_line=chunk.start_line + part.start_line - 1,
                end_line=chunk.start_line + part.end_line - 1,
                text=part.text,
            )
        ]

    split_chunks: list[Chunk] = []
    for index, part in enumerate(parts, start=1):
        split_chunks.append(
            Chunk(
                source=chunk.source,
                kind=chunk.kind,
                title=f"{chunk.title} (part {index})",
                start_line=chunk.start_line + part.start_line - 1,
                end_line=chunk.start_line + part.end_line - 1,
                text=part.text,
            )
        )

    return split_chunks


def chunk_markdown(source: Path, text: str, max_chars: int) -> list[Chunk]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    raw_chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_start_line = 1
    current_title = "document start"

    def flush(end_line: int) -> None:
        if not current_lines:
            return
        raw_chunks.append(
            Chunk(
                source=source,
                kind="markdown",
                title=current_title,
                start_line=current_start_line,
                end_line=end_line,
                text="".join(current_lines),
            )
        )

    for line_number, line in enumerate(lines, start=1):
        match = MARKDOWN_HEADING_RE.match(line)
        if match and current_lines:
            flush(line_number - 1)
            current_lines = []
            current_start_line = line_number
        if match:
            current_title = match.group(2)
        current_lines.append(line)

    if current_lines:
        flush(len(lines))

    chunks: list[Chunk] = []
    for chunk in raw_chunks:
        chunks.extend(split_if_large(chunk, max_chars))
    return chunks


def node_title(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    if isinstance(node, ast.AsyncFunctionDef):
        return f"async def {node.name}"
    if isinstance(node, ast.FunctionDef):
        return f"def {node.name}"
    return type(node).__name__


def chunk_python(source: Path, text: str, max_chars: int) -> list[Chunk]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_by_size(source, text, max_chars, "python-fallback")

    symbols = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not symbols:
        return chunk_by_size(source, text, max_chars, "python-size")

    def symbol_start_line(node: ast.AST) -> int:
        decorator_lines = [
            decorator.lineno
            for decorator in getattr(node, "decorator_list", [])
            if hasattr(decorator, "lineno")
        ]
        return min([getattr(node, "lineno", 1), *decorator_lines])

    def symbol_end_line(node: ast.AST) -> int:
        end_line = getattr(node, "end_lineno", None)
        if isinstance(end_line, int):
            return end_line
        return getattr(node, "lineno", 1)

    chunks: list[Chunk] = []
    cursor = 1

    def add_chunk(kind: str, title: str, start_line: int, end_line: int) -> None:
        if end_line < start_line:
            return
        chunk_text = "".join(lines[start_line - 1 : end_line])
        if not chunk_text.strip():
            return
        chunks.append(
            Chunk(
                source=source,
                kind=kind,
                title=title,
                start_line=start_line,
                end_line=end_line,
                text=chunk_text,
            )
        )

    first_symbol_start = symbol_start_line(symbols[0])
    if first_symbol_start > 1:
        add_chunk("python-preamble", "module preamble/imports", 1, first_symbol_start - 1)
        cursor = first_symbol_start

    for node in symbols:
        start_line = symbol_start_line(node)
        end_line = symbol_end_line(node)

        if cursor < start_line:
            add_chunk("python-module", "module code", cursor, start_line - 1)

        add_chunk("python-symbol", node_title(node), start_line, end_line)
        cursor = end_line + 1

    if cursor <= len(lines):
        add_chunk("python-module", "module code", cursor, len(lines))

    result: list[Chunk] = []
    for chunk in chunks:
        result.extend(split_if_large(chunk, max_chars))
    return result


def smart_chunk_file(path: Path, max_chars: int) -> list[Chunk]:
    text = read_text(path)
    suffix = path.suffix.lower()

    if suffix == ".md":
        return chunk_markdown(path, text, max_chars)
    if suffix == ".py":
        return chunk_python(path, text, max_chars)
    return chunk_by_size(path, text, max_chars, "text-size")


def preview(text: str, limit: int = 120) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart chunking demo.")
    parser.add_argument("path", type=Path, help="File or folder to chunk.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2500,
        help="Maximum characters per chunk.",
    )
    args = parser.parse_args()
    if args.max_chars <= 0:
        parser.error("--max-chars must be greater than 0")

    files = find_files(args.path)
    chunks: list[Chunk] = []

    for file_path in files:
        chunks.extend(smart_chunk_file(file_path, args.max_chars))

    print(f"Files: {len(files)}")
    print(f"Chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {index}")
        print(f"source: {chunk.source}")
        print(f"kind: {chunk.kind}")
        print(f"title: {chunk.title}")
        print(f"lines: {chunk.start_line}-{chunk.end_line}")
        print(f"chars: {len(chunk.text)}")
        print(f"preview: {preview(chunk.text)}")


if __name__ == "__main__":
    main()

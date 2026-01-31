"""Docling + HybridChunker parsing logic."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

ALLOWED_EXTENSIONS = {".pdf", ".csv", ".json", ".docx", ".txt"}
DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP = 200


@dataclass(frozen=True)
class ParsedChunk:
    text: str
    metadata: dict[str, Any]


def parse_documents(paths: list[str]) -> list[str]:
    chunks = parse_documents_with_metadata(paths)
    return [chunk.text for chunk in chunks]


def parse_documents_with_metadata(paths: list[str]) -> list[ParsedChunk]:
    validate_paths(paths)
    all_chunks: list[ParsedChunk] = []
    for path in paths:
        all_chunks.extend(_parse_single_path(path))
    return all_chunks


def validate_paths(paths: list[str]) -> None:
    if not paths:
        raise ValueError("At least one document path is required.")

    for path in paths:
        if path is None or not str(path).strip():
            raise ValueError("Document path cannot be empty.")
        if not os.path.exists(path):
            raise ValueError(f"Document path does not exist: {path}")
        if not os.path.isfile(path):
            raise ValueError(f"Document path is not a file: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}' for {path}.")


@dataclass(frozen=True)
class PageInfo:
    """Tracks page boundaries and headings from Docling parsing."""

    page_number: int
    char_start: int
    char_end: int
    heading: str


def _parse_single_path(path: str) -> list[ParsedChunk]:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".json", ".csv"}:
        text = _read_text_file(path, ext)
        page_info: list[PageInfo] = []
    else:
        text, page_info = _read_docling_text_with_metadata(path)

    document_title = _infer_document_title(path)
    paragraphs = _split_paragraphs(text)
    chunks = _hybrid_chunk_paragraphs(
        paragraphs, max_chars=DEFAULT_MAX_CHARS, overlap=DEFAULT_OVERLAP
    )

    total = len(chunks)
    parsed: list[ParsedChunk] = []
    char_offset = 0
    for idx, chunk_text in enumerate(chunks, start=1):
        heading = _get_heading_for_chunk(char_offset, page_info)
        page_number = _get_page_for_chunk(char_offset, page_info)
        metadata = {
            "source_path": path,
            "document_title": document_title,
            "heading": heading,
            "page_number": page_number,
            "chunk_index": idx,
            "chunk_count": total,
            "context_summary": _contextualize(chunk_text, document_title),
        }
        parsed.append(ParsedChunk(text=chunk_text, metadata=metadata))
        char_offset += len(chunk_text)
    return parsed


def _read_docling_text_with_metadata(path: str) -> tuple[str, list[PageInfo]]:
    """Read document with Docling and extract page/heading metadata.

    Returns:
        Tuple of (full_text, list of PageInfo with page boundaries and headings)
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ValueError(
            "Docling is required to parse PDF/DOCX files. Install docling and retry."
        ) from exc

    converter = DocumentConverter()
    result = converter.convert(path)
    text = _extract_docling_text(result)
    if not text.strip():
        raise ValueError(f"Docling returned empty text for {path}.")

    page_info = _extract_page_info(result, text)
    return text, page_info


def _extract_page_info(result: Any, full_text: str) -> list[PageInfo]:
    """Extract page boundaries and headings from Docling result."""
    page_info: list[PageInfo] = []

    doc = None
    for attr in ("document", "doc"):
        if hasattr(result, attr):
            doc = getattr(result, attr)
            break

    if doc is None:
        return [PageInfo(page_number=1, char_start=0, char_end=len(full_text), heading="")]

    pages = getattr(doc, "pages", None)
    if pages and hasattr(pages, "__iter__"):
        char_offset = 0
        for page_num, page in enumerate(pages, start=1):
            page_text = ""
            if hasattr(page, "text"):
                page_text = page.text
            elif hasattr(page, "export_to_text"):
                page_text = page.export_to_text()

            heading = _extract_heading_from_page(page)
            page_info.append(
                PageInfo(
                    page_number=page_num,
                    char_start=char_offset,
                    char_end=char_offset + len(page_text),
                    heading=heading,
                )
            )
            char_offset += len(page_text)
        return page_info

    items = getattr(doc, "items", None) or getattr(doc, "elements", None)
    if items and hasattr(items, "__iter__"):
        current_page = 1
        current_heading = ""
        char_offset = 0

        for item in items:
            item_page = getattr(item, "page", None) or getattr(item, "page_number", current_page)
            if isinstance(item_page, int) and item_page != current_page:
                page_info.append(
                    PageInfo(
                        page_number=current_page,
                        char_start=page_info[-1].char_end if page_info else 0,
                        char_end=char_offset,
                        heading=current_heading,
                    )
                )
                current_page = item_page
                current_heading = ""

            item_type = getattr(item, "type", "") or getattr(item, "label", "")
            if "heading" in str(item_type).lower() or "title" in str(item_type).lower():
                item_text = getattr(item, "text", "") or str(item)
                if item_text:
                    current_heading = item_text[:100]

            item_text = getattr(item, "text", "")
            if item_text:
                char_offset += len(item_text)

        if current_page:
            page_info.append(
                PageInfo(
                    page_number=current_page,
                    char_start=page_info[-1].char_end if page_info else 0,
                    char_end=char_offset,
                    heading=current_heading,
                )
            )
        return page_info

    return [PageInfo(page_number=1, char_start=0, char_end=len(full_text), heading="")]


def _extract_heading_from_page(page: Any) -> str:
    """Extract the first heading from a page object."""
    items = getattr(page, "items", None) or getattr(page, "elements", None)
    if not items:
        return ""

    for item in items:
        item_type = getattr(item, "type", "") or getattr(item, "label", "")
        if "heading" in str(item_type).lower() or "title" in str(item_type).lower():
            text = getattr(item, "text", "") or str(item)
            if text:
                return text[:100]
    return ""


def _get_heading_for_chunk(char_offset: int, page_info: list[PageInfo]) -> str:
    """Get the heading for a chunk based on its character offset."""
    if not page_info:
        return ""

    for info in page_info:
        if info.char_start <= char_offset < info.char_end:
            return info.heading

    return page_info[-1].heading if page_info else ""


def _get_page_for_chunk(char_offset: int, page_info: list[PageInfo]) -> int:
    """Get the page number for a chunk based on its character offset."""
    if not page_info:
        return 1

    for info in page_info:
        if info.char_start <= char_offset < info.char_end:
            return info.page_number

    return page_info[-1].page_number if page_info else 1


def _read_text_file(path: str, ext: str) -> str:
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)

    if ext == ".csv":
        frame = pd.read_csv(path)
        return frame.to_csv(index=False)

    raise ValueError(f"Unsupported text file type '{ext}'.")


def _read_docling_text(path: str) -> str:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ValueError(
            "Docling is required to parse PDF/DOCX files. Install docling and retry."
        ) from exc

    converter = DocumentConverter()
    result = converter.convert(path)
    text = _extract_docling_text(result)
    if not text.strip():
        raise ValueError(f"Docling returned empty text for {path}.")
    return text


def _extract_docling_text(result: Any) -> str:
    for attr in ("document", "doc", "document_text"):
        if hasattr(result, attr):
            candidate = getattr(result, attr)
            if isinstance(candidate, str):
                return candidate
            if hasattr(candidate, "export_to_text"):
                return candidate.export_to_text()
            if hasattr(candidate, "text"):
                return candidate.text

    if hasattr(result, "export_to_text"):
        return result.export_to_text()

    raise ValueError("Docling conversion did not expose text output.")


def _infer_document_title(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name or base


def _split_paragraphs(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if not line:
            if buffer:
                paragraphs.append(" ".join(buffer).strip())
                buffer = []
            continue
        buffer.append(line)
    if buffer:
        paragraphs.append(" ".join(buffer).strip())
    return [p for p in paragraphs if p]


def _hybrid_chunk_paragraphs(
    paragraphs: Iterable[str],
    *,
    max_chars: int,
    overlap: int,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if not paragraph:
            continue

        if current_len + len(paragraph) + 1 > max_chars and current:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)
            current = _tail_overlap(chunk, overlap)
            current_len = len(" ".join(current))

        current.append(paragraph)
        current_len += len(paragraph) + 1

    final_chunk = " ".join(current).strip()
    if final_chunk:
        chunks.append(final_chunk)

    return chunks


def _tail_overlap(chunk: str, overlap: int) -> list[str]:
    if overlap <= 0 or not chunk:
        return []
    tail = chunk[-overlap:]
    return [tail.strip()] if tail.strip() else []


def _contextualize(chunk_text: str, document_title: str) -> str:
    snippet = chunk_text.strip().replace("\n", " ")
    if len(snippet) > 160:
        snippet = snippet[:160].rstrip() + "..."
    return f"{document_title}: {snippet}" if document_title else snippet

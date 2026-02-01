"""Document parsing with Docling + HybridChunker."""

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import pandas as pd

ALLOWED_EXTENSIONS = {".pdf", ".csv", ".json", ".docx", ".txt", ".md"}
CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
    ".csv": "text/csv",
}


@dataclass(frozen=True)
class ParsedChunk:
    text: str
    metadata: dict[str, Any]


def parse_file(path: str) -> list[ParsedChunk]:
    """Parse a single file into chunks with metadata."""
    if not os.path.isfile(path):
        raise ValueError(f"File does not exist: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'.")

    if ext in {".pdf", ".docx"}:
        return _parse_with_docling(path, source_path=path)

    if ext in {".txt", ".json", ".csv"}:
        return _parse_text_with_docling(path, ext)

    return _parse_text_with_docling(path, ext)


def get_content_type(path: str) -> str:
    """Get MIME content type for a file."""
    ext = os.path.splitext(path)[1].lower()
    return CONTENT_TYPES.get(ext, "text/plain")


def _parse_with_docling(path: str, *, source_path: str) -> list[ParsedChunk]:
    """Parse a document using Docling's HybridChunker."""
    try:
        from docling.chunking import HybridChunker
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ValueError("Docling required. Install with: pip install docling") from exc

    converter = DocumentConverter()
    result = converter.convert(path)
    doc = result.document

    from app.core.config import get_settings

    settings = get_settings()
    chunker = HybridChunker(
        tokenizer=settings.docling_tokenizer,
        max_tokens=512,
        merge_peers=True,
    )

    parsed_chunks = []
    doc_title = os.path.basename(source_path)

    for i, chunk in enumerate(chunker.chunk(doc)):
        # Extract page number from chunk metadata
        page_no = None
        if chunk.meta.doc_items:
            prov = getattr(chunk.meta.doc_items[0], "prov", [])
            if prov:
                page_no = prov[0].page_no

        metadata = {
            "source_path": source_path,
            "document_title": doc_title,
            "heading": chunk.meta.headings[0] if chunk.meta.headings else "",
            "page_number": page_no or 1,
            "chunk_index": i + 1,
            "context_summary": chunker.contextualize(chunk),  # Native hierarchical context
        }
        parsed_chunks.append(ParsedChunk(text=chunk.text, metadata=metadata))

    # Fallback: if doc is too simple for chunker, use full markdown export
    if not parsed_chunks:
        markdown = doc.export_to_markdown()
        if not markdown.strip():
            raise ValueError(f"No content extracted from {path}.")
        parsed_chunks.append(
            ParsedChunk(
                text=markdown,
                metadata={
                    "source_path": source_path,
                    "document_title": doc_title,
                    "heading": "",
                    "page_number": 1,
                    "chunk_index": 1,
                    "context_summary": f"{doc_title} [full document]",
                },
            )
        )

    return parsed_chunks


def _parse_text_with_docling(path: str, ext: str) -> list[ParsedChunk]:
    """Parse TXT/JSON/CSV by normalizing to text and running Docling."""
    text = _read_text_payload(path, ext)
    if not text.strip():
        raise ValueError(f"File is empty: {path}")

    temp_path = _write_temp_text(text)
    try:
        return _parse_with_docling(temp_path, source_path=path)
    finally:
        _safe_remove(temp_path)


def _read_text_payload(path: str, ext: str) -> str:
    """Load and normalize text payloads for TXT/JSON/CSV."""
    if ext == ".txt":
        with open(path, encoding="utf-8") as f:
            return f.read()
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            return json.dumps(json.load(f), ensure_ascii=False, indent=2)
    if ext == ".csv":
        return pd.read_csv(path).to_csv(index=False)
    raise ValueError(f"Unsupported text file type '{ext}'.")


def _write_temp_text(text: str) -> str:
    """Write normalized text to a temp file for Docling input.

    Uses .md extension since Docling supports markdown but not plain text.
    """
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
    with open(handle.name, "w", encoding="utf-8") as f:
        f.write(text)
    return handle.name


def _safe_remove(path: str) -> None:
    """Best-effort removal of temp files."""
    try:
        os.remove(path)
    except OSError:
        pass


def _parse_text_file(path: str, ext: str) -> list[ParsedChunk]:
    """Parse text files (TXT, JSON, CSV) with simple chunking."""
    text = _read_text_payload(path, ext)

    if not text.strip():
        raise ValueError(f"File is empty: {path}")

    # Simple paragraph-based chunking
    chunks = _chunk_text(text, max_chars=2000)
    doc_title = os.path.basename(path)

    return [
        ParsedChunk(
            text=chunk_text,
            metadata={
                "source_path": path,
                "document_title": doc_title,
                "heading": "",
                "page_number": 1,
                "chunk_index": i + 1,
                # Use actual text as context (with doc title prefix for multi-chunk docs)
                "context_summary": (
                    f"{doc_title}: {chunk_text}"
                    if len(chunks) > 1
                    else chunk_text
                ),
            },
        )
        for i, chunk_text in enumerate(chunks)
    ]


def _chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    """Split text with overlap to ensure semantic continuity."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars.")

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += max_chars - overlap
    return chunks

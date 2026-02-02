"""Document parsing with Docling + HybridChunker."""

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)

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

    if ext in {".pdf", ".docx", ".md"}:
        return _parse_with_docling(path, source_path=path)

    if ext in {".txt", ".json", ".csv"}:
        return _parse_text_with_docling(path, ext)

    raise ValueError(f"Unsupported file type '{ext}'.")


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

    logger.debug("Converting document: %s", path)
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
            "context_summary": chunker.contextualize(chunk=chunk),  # Native hierarchical context
        }
        parsed_chunks.append(ParsedChunk(text=chunk.text, metadata=metadata))

    # Fallback: if doc is too simple for chunker, use full markdown export
    if not parsed_chunks:
        logger.debug("No chunks from chunker, using full markdown export")
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

    logger.debug("Parsed %d chunks from %s", len(parsed_chunks), source_path)
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

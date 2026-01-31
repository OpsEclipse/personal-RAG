import hashlib
import json
import os
import uuid
from collections import deque
from typing import Any

from app.models.ingest import IngestJobRecord
from app.services.embedder import embed_texts_batched
from app.services.file_storage import cleanup_job_files
from app.services.parser import parse_documents_with_metadata
from app.services.vectordb import upsert_vectors

ALLOWED_EXTENSIONS = {".pdf", ".csv", ".json", ".docx", ".txt"}
JOB_STORE: dict[str, IngestJobRecord] = {}
JOB_QUEUE: deque[str] = deque()


def validate_ingest_request(filename: str | None, namespace: str, index: str) -> None:
    if filename is None or not filename.strip():
        raise ValueError("Filename is required.")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'.")

    if not namespace.strip():
        raise ValueError("Namespace is required.")

    if not index.strip():
        raise ValueError("Index is required.")


def validate_job_id(job_id: str) -> None:
    try:
        uuid.UUID(job_id)
    except ValueError as exc:
        raise ValueError("Invalid job id.") from exc


def parse_metadata_json(metadata_json: str | None) -> dict[str, Any] | None:
    if metadata_json is None:
        return None

    if not metadata_json.strip():
        return None

    try:
        parsed = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise ValueError("metadata_json must be valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise ValueError("metadata_json must be a JSON object.")

    return parsed


def add_file_to_queue(
    *,
    job_id: str,
    filename: str,
    content_type: str | None,
    file_path: str,
    namespace: str,
    index: str,
    routing_hint: str | None,
    metadata: dict[str, Any] | None,
) -> str:
    record_metadata: dict[str, Any] = {"namespace": namespace, "index": index}
    if routing_hint:
        record_metadata["routing_hint"] = routing_hint
    if metadata:
        record_metadata["metadata"] = metadata

    record = IngestJobRecord(
        job_id=job_id,
        filename=filename,
        content_type=content_type,
        file_path=file_path,
        status="queued",
        metadata=record_metadata or None,
    )
    JOB_STORE[job_id] = record
    JOB_QUEUE.append(job_id)
    return job_id


def get_job_record(job_id: str) -> IngestJobRecord | None:
    return JOB_STORE.get(job_id)


def _infer_content_type(file_path: str) -> str:
    """Infer content type from file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    content_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
        ".json": "application/json",
        ".csv": "text/csv",
    }
    return content_types.get(ext, "text")


def process_job(job_id: str) -> None:
    """Process an ingestion job: parse → embed → upsert to Pinecone."""
    record = JOB_STORE.get(job_id)
    if record is None:
        return

    record.status = "processing"

    try:
        JOB_QUEUE.remove(job_id)
    except ValueError:
        pass

    try:
        if not record.file_path:
            raise ValueError("No file path available for processing.")

        # 1. Parse document into chunks
        chunks = parse_documents_with_metadata([record.file_path])
        if not chunks:
            raise ValueError("No chunks extracted from document.")

        # 2. Generate embeddings (batched)
        texts = [c.text for c in chunks]
        embeddings = embed_texts_batched(texts)

        # 3. Get user-provided metadata and index info
        record_meta = record.metadata or {}
        user_meta = record_meta.get("metadata", {})
        index_name = record_meta.get("index", "")
        namespace = record_meta.get("namespace", "")

        if not index_name:
            raise ValueError("No index name specified.")

        content_type = _infer_content_type(record.file_path)

        # 4. Prepare vectors with target schema
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vector_id = hashlib.md5(chunk.text.encode()).hexdigest()
            vectors.append(
                {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk.text,
                        "doc_title": chunk.metadata.get("document_title", ""),
                        "heading": chunk.metadata.get("heading", ""),
                        "source_url": user_meta.get("source_url", ""),
                        "page_number": chunk.metadata.get("page_number", 1),
                        "content_type": content_type,
                        "context_summary": chunk.metadata.get("context_summary", ""),
                    },
                }
            )

        # 5. Upsert to Pinecone
        upsert_vectors(index_name, namespace, vectors)

        record.chunks_processed = len(chunks)
        record.status = "completed"
        cleanup_job_files(job_id)

    except Exception as exc:
        record.status = "failed"
        record.error = str(exc)

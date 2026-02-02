import asyncio
import hashlib
import json
import threading
import uuid
from typing import Any

from openai import RateLimitError
from pinecone.exceptions import PineconeException

from app.core.logging import get_logger
from app.core.namespaces import Namespace, is_valid_namespace
from app.models.ingest import IngestJobRecord, RoutingMode
from app.services.embedder import embed_texts_batched
from app.services.file_storage import cleanup_job_files
from app.services.namespace_router import (
    classify_chunks_individually,
    classify_document,
    did_last_call_use_fallback,
)
from app.services.parser import ALLOWED_EXTENSIONS, ParsedChunk, get_content_type, parse_file
from app.services.vectordb import upsert_vectors

logger = get_logger(__name__)

JOB_STORE: dict[str, IngestJobRecord] = {}
JOB_QUEUE: list[str] = []
JOB_STORE_LOCK = threading.Lock()

# Semaphore to limit concurrent job processing to 1
_PROCESSING_SEMAPHORE = asyncio.Semaphore(1)


def validate_ingest_request(
    filename: str | None,
    namespace: str | None,
    index: str,
    routing_mode: RoutingMode,
) -> None:
    if not filename or not filename.strip():
        raise ValueError("Filename is required.")

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if f".{ext}" not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '.{ext}'.")

    normalized_namespace = namespace.strip() if namespace else ""
    # Namespace is required only for manual mode
    if routing_mode == RoutingMode.MANUAL:
        if not normalized_namespace:
            raise ValueError("Namespace is required when routing_mode is 'manual'.")
        if not is_valid_namespace(normalized_namespace):
            valid = ", ".join([ns.value for ns in Namespace])
            raise ValueError(f"Invalid namespace '{normalized_namespace}'. Must be one of: {valid}")
    elif normalized_namespace:
        raise ValueError("Namespace is only allowed when routing_mode is 'manual'.")

    if not index.strip():
        raise ValueError("Index is required.")


def validate_job_id(job_id: str) -> None:
    try:
        uuid.UUID(job_id)
    except ValueError as exc:
        raise ValueError("Invalid job id.") from exc


def parse_metadata_json(metadata_json: str | None) -> dict[str, Any] | None:
    if not metadata_json or not metadata_json.strip():
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
    namespace: str | None,
    index: str,
    routing_mode: RoutingMode,
    metadata: dict[str, Any] | None,
) -> str:
    record_metadata: dict[str, Any] = {"index": index, "routing_mode": routing_mode.value}
    if namespace:
        record_metadata["namespace"] = namespace
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
    with JOB_STORE_LOCK:
        JOB_STORE[job_id] = record
        JOB_QUEUE.append(job_id)
    return job_id


def get_job_record(job_id: str) -> IngestJobRecord | None:
    with JOB_STORE_LOCK:
        return JOB_STORE.get(job_id)


def _update_status(job_id: str, status: str) -> None:
    """Update job status with thread-safe locking."""
    with JOB_STORE_LOCK:
        record = JOB_STORE.get(job_id)
        if record is not None:
            record.status = status  # type: ignore[assignment]


def process_job(job_id: str) -> None:
    """Process an ingestion job: chunk → route → embed → upsert."""
    logger.info("Starting job %s", job_id)

    with JOB_STORE_LOCK:
        record = JOB_STORE.get(job_id)
        if record is None:
            logger.error("Job %s not found in store", job_id)
            return

    try:
        if not record.file_path:
            raise ValueError("No file path available for processing.")

        record_meta = record.metadata or {}
        index_name = record_meta.get("index", "")
        manual_namespace = record_meta.get("namespace", "")
        routing_mode = RoutingMode(record_meta.get("routing_mode", "auto"))
        user_meta = record_meta.get("metadata", {})

        if not index_name:
            raise ValueError("No index name specified.")

        logger.info(
            "Job %s: file=%s, index=%s, routing_mode=%s",
            job_id,
            record.filename,
            index_name,
            routing_mode.value,
        )

        # 1. Parse file with DocLing + HybridChunker
        _update_status(job_id, "chunking")
        logger.info("Job %s: parsing file", job_id)
        try:
            chunks = parse_file(record.file_path)
            content_type = get_content_type(record.file_path)
            logger.info("Job %s: parsed %d chunks", job_id, len(chunks))
        except ValueError as e:
            logger.error("Job %s: parsing failed - %s", job_id, e)
            raise

        # 2. Route to namespace(s) based on routing mode
        _update_status(job_id, "routing")
        logger.info("Job %s: routing chunks (mode=%s)", job_id, routing_mode.value)
        routing_fallback_used = False
        if routing_mode == RoutingMode.MANUAL:
            # Use provided namespace for all chunks
            chunk_namespaces = [manual_namespace] * len(chunks)
        elif routing_mode == RoutingMode.PER_CHUNK:
            # LLM classifies each chunk individually
            chunk_namespaces = [ns.value for ns in classify_chunks_individually(chunks)]
            routing_fallback_used = did_last_call_use_fallback()
        else:  # AUTO (default) - document-level classification
            doc_namespace = classify_document(chunks)
            routing_fallback_used = did_last_call_use_fallback()
            chunk_namespaces = [doc_namespace.value] * len(chunks)

        if routing_fallback_used:
            logger.warning("Job %s: routing used fallback namespace", job_id)
            with JOB_STORE_LOCK:
                record.routing_fallback_used = True

        unique_namespaces = set(chunk_namespaces)
        logger.info("Job %s: routed to namespaces %s", job_id, unique_namespaces)

        # 3. Embed CONTEXTUALIZED text (not raw chunk text)
        _update_status(job_id, "embedding")
        logger.info("Job %s: embedding %d chunks", job_id, len(chunks))
        contextualized_texts = [
            chunk.metadata.get("context_summary", chunk.text) for chunk in chunks
        ]
        try:
            embeddings = embed_texts_batched(contextualized_texts)
        except RateLimitError as e:
            logger.error("Job %s: embedding rate limited - %s", job_id, e)
            raise
        except ValueError as e:
            logger.error("Job %s: embedding failed - %s", job_id, e)
            raise

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Embedding count mismatch: expected "
                f"{len(chunks)} vectors, got {len(embeddings)}."
            )
        logger.info("Job %s: embedded %d chunks", job_id, len(embeddings))

        # 4. Build vectors and group by namespace
        _update_status(job_id, "upserting")
        vectors_by_namespace: dict[str, list[dict]] = {}
        for chunk, embedding, namespace in zip(chunks, embeddings, chunk_namespaces):
            vector = {
                "id": hashlib.md5(chunk.text.encode()).hexdigest(),
                "values": embedding,
                "metadata": {
                    "text": chunk.text,
                    "contextualized_text": chunk.metadata.get("context_summary", ""),
                    "doc_title": chunk.metadata.get("document_title", ""),
                    "heading": chunk.metadata.get("heading", ""),
                    "source_url": user_meta.get("source_url", ""),
                    "page_number": chunk.metadata.get("page_number", 1),
                    "content_type": content_type,
                    "chunk_index": chunk.metadata.get("chunk_index", 1),
                },
            }
            if namespace not in vectors_by_namespace:
                vectors_by_namespace[namespace] = []
            vectors_by_namespace[namespace].append(vector)

        # 5. Upsert to Pinecone (grouped by namespace)
        logger.info("Job %s: upserting to %d namespaces", job_id, len(vectors_by_namespace))
        try:
            for namespace, vectors in vectors_by_namespace.items():
                upsert_vectors(index_name, namespace, vectors)
        except PineconeException as e:
            logger.error("Job %s: Pinecone upsert failed - %s", job_id, e)
            raise
        except ConnectionError as e:
            logger.error("Job %s: connection error during upsert - %s", job_id, e)
            raise

        record.chunks_processed = len(chunks)
        _update_status(job_id, "completed")
        cleanup_job_files(job_id)
        logger.info("Job %s: completed successfully, %d chunks processed", job_id, len(chunks))

    except Exception as exc:
        # Unwrap tenacity RetryError to get the actual cause
        actual_error = exc
        if hasattr(exc, "last_attempt") and exc.last_attempt is not None:
            try:
                actual_error = exc.last_attempt.result()
            except Exception as inner:
                actual_error = inner

        error_msg = f"{type(actual_error).__name__}: {actual_error}"
        logger.error("Job %s: failed - %s", job_id, error_msg)

        with JOB_STORE_LOCK:
            record.status = "failed"
            record.error = error_msg


async def process_job_with_limit(job_id: str) -> None:
    """Process job with concurrency limit of 1.

    Uses a semaphore to ensure only one job processes at a time,
    preventing resource exhaustion from parallel document processing.
    """
    logger.info("Job %s: waiting for processing slot", job_id)
    async with _PROCESSING_SEMAPHORE:
        logger.info("Job %s: acquired processing slot", job_id)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, process_job, job_id)

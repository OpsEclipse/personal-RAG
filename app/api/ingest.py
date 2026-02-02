import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status

from app.core.config import get_settings, reset_settings
from app.models.ingest import IngestAccepted, IngestJobRecord, IngestJobSummary, RoutingMode
from app.services.embedder import reset_client as reset_embedder_client
from app.services.file_storage import save_uploaded_file
from app.services.ingest_queue import (
    add_file_to_queue,
    get_job_record,
    parse_metadata_json,
    process_job_with_limit,
    validate_ingest_request,
    validate_job_id,
)

router = APIRouter(prefix="/v1", tags=["ingestion"])


@router.post("/debug/reset")
def reset_cached_clients() -> dict:
    """Reset all cached clients and settings. Use after changing API keys."""
    reset_settings()
    reset_embedder_client()
    return {"status": "ok", "message": "All cached clients and settings cleared"}


@router.post("/debug/test-openai")
def test_openai_connection() -> dict:
    """Test OpenAI API connection with a minimal embedding request."""
    from openai import OpenAI

    settings = get_settings()
    # Mask API key for display
    key_preview = settings.openai_api_key[:8] + "..." if settings.openai_api_key else "NOT SET"

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.embeddings.create(model="text-embedding-3-small", input=["test"])
        return {
            "status": "ok",
            "api_key_preview": key_preview,
            "embedding_dimensions": len(resp.data[0].embedding),
        }
    except Exception as e:
        return {
            "status": "error",
            "api_key_preview": key_preview,
            "error_type": type(e).__name__,
            "error_message": str(e),
        }


@router.post("/ingest", response_model=IngestAccepted, status_code=status.HTTP_202_ACCEPTED)
async def ingest_documents(
    background_tasks: BackgroundTasks,
    file: list[UploadFile] = File(...),
    index: str | None = Form(None),
    namespace: str | None = Form(None),
    routing_mode: RoutingMode = Form(RoutingMode.AUTO),
    metadata_json: str | None = Form(None),
) -> IngestAccepted:
    """Ingest one or more documents into the RAG system.

    Args:
        file: One or more document files (PDF, DOCX, TXT, JSON, CSV)
        index: Pinecone index name
        namespace: Target namespace (required for manual mode; must be omitted otherwise)
        routing_mode: How to determine namespace:
            - auto (default): LLM classifies at document level
            - manual: Use provided namespace, no LLM
            - per_chunk: LLM classifies each chunk individually
        metadata_json: Optional JSON string with custom metadata
    """
    try:
        if not file:
            raise ValueError("At least one file is required.")
        namespace = namespace.strip() if namespace else None
        if namespace == "":
            namespace = None
        if not index or not index.strip():
            index = get_settings().pinecone_index
        index = index.strip()
        for upload in file:
            filename = upload.filename or ""
            validate_ingest_request(filename, namespace, index, routing_mode)
        metadata = parse_metadata_json(metadata_json)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    jobs: list[IngestJobSummary] = []
    for upload in file:
        filename = upload.filename or ""
        job_id = str(uuid.uuid4())
        file_path = save_uploaded_file(job_id, upload)

        add_file_to_queue(
            job_id=job_id,
            filename=filename,
            content_type=upload.content_type,
            file_path=file_path,
            namespace=namespace,
            index=index,
            routing_mode=routing_mode,
            metadata=metadata,
        )
        background_tasks.add_task(process_job_with_limit, job_id)
        jobs.append(IngestJobSummary(job_id=job_id, filename=filename))

    return IngestAccepted(jobs=jobs, received_count=len(jobs))


@router.get("/ingest/{job_id}", response_model=IngestJobRecord)
def get_ingest_status(job_id: str) -> IngestJobRecord:
    try:
        validate_job_id(job_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    record = get_job_record(job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )
    return record

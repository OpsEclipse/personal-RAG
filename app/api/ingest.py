import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status

from app.models.ingest import IngestAccepted, IngestJobRecord
from app.services.file_storage import save_uploaded_file
from app.services.ingest_queue import (
    add_file_to_queue,
    get_job_record,
    parse_metadata_json,
    process_job,
    validate_ingest_request,
    validate_job_id,
)

router = APIRouter(prefix="/v1", tags=["ingestion"])

@router.post("/ingest", response_model=IngestAccepted, status_code=status.HTTP_202_ACCEPTED)
async def ingest_documents(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    namespace: str = Form(...),
    index: str = Form(...),
    metadata_json: str | None = Form(None),
    routing_hint: str | None = Form(None),
) -> IngestAccepted:
    filename = file.filename or ""
    try:
        validate_ingest_request(filename, namespace, index)
        metadata = parse_metadata_json(metadata_json)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    job_id = str(uuid.uuid4())
    file_path = save_uploaded_file(job_id, file)

    add_file_to_queue(
        job_id=job_id,
        filename=filename,
        content_type=file.content_type,
        file_path=file_path,
        namespace=namespace,
        index=index,
        routing_hint=routing_hint,
        metadata=metadata,
    )
    background_tasks.add_task(process_job, job_id)

    return IngestAccepted(job_id=job_id, received_count=1)


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

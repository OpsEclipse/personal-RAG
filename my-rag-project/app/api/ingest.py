from fastapi import APIRouter, status

from app.models.ingest import IngestAccepted, IngestRequest

router = APIRouter(prefix="/v1", tags=["ingestion"])


@router.post("/ingest", response_model=IngestAccepted, status_code=status.HTTP_202_ACCEPTED)
def ingest_documents(payload: IngestRequest) -> IngestAccepted:
    # TODO: enqueue document processing job
    return IngestAccepted(
        job_id="00000000-0000-0000-0000-000000000000",
        received_count=len(payload.sources),
    )

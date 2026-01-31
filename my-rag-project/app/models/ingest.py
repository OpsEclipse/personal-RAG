from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source_type: Literal["url", "file_path"]
    sources: list[str] = Field(min_length=1)
    namespace: str
    index: str
    metadata: dict[str, Any] | None = None
    routing_hint: str | None = None


class IngestAccepted(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"
    received_count: int


class IngestJobRecord(BaseModel):
    job_id: str
    filename: str
    content_type: str | None = None
    status: Literal["queued", "processing", "completed", "failed"] = "queued"
    metadata: dict[str, Any] | None = None
    file_path: str | None = None
    error: str | None = None
    chunks_processed: int | None = None

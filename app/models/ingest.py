from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class RoutingMode(str, Enum):
    """Namespace routing mode options."""

    AUTO = "auto"  # LLM decides at document level (default)
    MANUAL = "manual"  # Use provided namespace, no LLM
    PER_CHUNK = "per_chunk"  # LLM analyzes each chunk individually


class IngestRequest(BaseModel):
    source_type: Literal["url", "file_path"]
    sources: list[str] = Field(min_length=1)
    namespace: str | None = None  # Optional when using auto/per_chunk routing
    index: str
    metadata: dict[str, Any] | None = None
    routing_mode: RoutingMode = RoutingMode.AUTO


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

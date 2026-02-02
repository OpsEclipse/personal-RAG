from fastapi import FastAPI

from app.api.ingest import router as ingest_router
from app.core.config import get_settings
from app.core.logging import setup_logging

app = FastAPI(title="RAG Service")

app.include_router(ingest_router)


@app.on_event("startup")
def _startup() -> None:
    # Configure logging
    setup_logging(level="INFO")
    # Fail fast if required env vars are missing.
    get_settings()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

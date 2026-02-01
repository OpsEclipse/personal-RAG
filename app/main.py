from fastapi import FastAPI

from app.api.ingest import router as ingest_router
from app.core.config import get_settings

app = FastAPI(title="RAG Service")

app.include_router(ingest_router)


@app.on_event("startup")
def _load_settings() -> None:
    # Fail fast if required env vars are missing.
    get_settings()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

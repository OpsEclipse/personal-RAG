from fastapi import FastAPI

from app.api.ingest import router as ingest_router

app = FastAPI(title="RAG Service")

app.include_router(ingest_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

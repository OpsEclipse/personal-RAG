"""File storage utilities for uploaded documents."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import UploadFile

UPLOADS_DIR = Path(tempfile.gettempdir()) / "rag-uploads"


def save_uploaded_file(job_id: str, file: UploadFile) -> str:
    """Save uploaded file to /tmp/rag-uploads/{job_id}/{filename}.

    Args:
        job_id: Unique job identifier
        file: FastAPI UploadFile object

    Returns:
        Absolute path to the saved file
    """
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or "upload"
    file_path = job_dir / filename

    with open(file_path, "wb") as dest:
        content = file.file.read()
        dest.write(content)

    file.file.seek(0)

    return str(file_path)


def cleanup_job_files(job_id: str) -> None:
    """Remove job's upload directory.

    Call on successful processing only; retain files on failure for debugging.

    Args:
        job_id: Unique job identifier
    """
    job_dir = UPLOADS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)

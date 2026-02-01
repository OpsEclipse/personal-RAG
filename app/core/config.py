"""Runtime configuration and environment loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_SETTINGS: "Settings | None" = None


def reset_settings() -> None:
    """Reset cached settings. Call after changing environment variables."""
    global _SETTINGS
    _SETTINGS = None


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openrouter_api_key: str
    pinecone_api_key: str
    pinecone_index: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "meta-llama/llama-3.2-3b-instruct:free"
    pinecone_host: str | None = None
    docling_tokenizer: str = "gpt2"


def load_env_file(path: str, *, override: bool = False) -> None:
    """Load key=value pairs from a .env-style file into os.environ.

    Keeps existing env values unless override=True.
    """
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not override and key in os.environ:
            continue
        os.environ[key] = value


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_settings() -> Settings:
    """Load settings from my.env and environment variables."""
    global _SETTINGS
    if _SETTINGS is not None:
        return _SETTINGS

    env_file = os.getenv("MY_ENV_FILE", "my.env")
    load_env_file(env_file)

    _SETTINGS = Settings(
        openai_api_key=_required_env("OPENAI_API_KEY"),
        openrouter_api_key=_required_env("OPENROUTER_API_KEY"),
        openrouter_base_url=(
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
            or "https://openrouter.ai/api/v1"
        ),
        openrouter_model=(
            os.getenv(
                "OPENROUTER_MODEL",
                "meta-llama/llama-3.2-3b-instruct:free",
            ).strip()
            or "meta-llama/llama-3.2-3b-instruct:free"
        ),
        pinecone_api_key=_required_env("PINECONE_API_KEY"),
        pinecone_index=_required_env("PINECONE_INDEX"),
        pinecone_host=os.getenv("PINECONE_HOST", "").strip() or None,
        docling_tokenizer=os.getenv("DOCLING_TOKENIZER", "gpt2").strip() or "gpt2",
    )
    return _SETTINGS


def get_pinecone_host(index_name: str) -> str:
    """Resolve Pinecone host for an index."""
    settings = get_settings()
    env_key = f"PINECONE_HOST_{index_name.upper()}"
    per_index = os.getenv(env_key, "").strip()
    host = per_index or settings.pinecone_host
    if not host:
        raise ValueError(
            "Missing Pinecone host. Set PINECONE_HOST or "
            f"{env_key} for index '{index_name}'."
        )
    return host

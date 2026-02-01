"""OpenAI embedding calls with batching and retry logic."""

import time

from openai import OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.config import get_settings

_client: OpenAI | None = None

# Batch size for embedding requests (OpenAI allows up to 2048 texts per request)
EMBEDDING_BATCH_SIZE = 20


def reset_client() -> None:
    """Reset the cached OpenAI client. Call this after changing API keys."""
    global _client
    _client = None

# Delay between batches to avoid rate limits (seconds)
INTER_BATCH_DELAY = 1.0


def _get_client() -> OpenAI:
    """Lazy initialization of OpenAI client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(10),
    wait=wait_exponential_jitter(initial=5, max=120, jitter=5),
    reraise=True,
)
def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Embed a batch of texts with retry logic for rate limits.

    Args:
        texts: List of texts to embed
        model: OpenAI embedding model name

    Returns:
        List of embedding vectors

    Raises:
        RateLimitError: If rate limit is exceeded after all retries
        ValueError: If texts are invalid
    """
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("All texts must be non-empty strings for embedding.")

    client = _get_client()
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


def embed_texts_batched(
    texts: list[str], model: str = "text-embedding-3-small"
) -> list[list[float]]:
    """Embed texts in batches with delays to avoid rate limits.

    Args:
        texts: List of texts to embed
        model: OpenAI embedding model name

    Returns:
        List of embedding vectors in same order as input
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    num_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = embed_texts(batch, model=model)
        all_embeddings.extend(batch_embeddings)

        # Add delay between batches to respect rate limits (skip after last batch)
        batch_num = i // EMBEDDING_BATCH_SIZE + 1
        if batch_num < num_batches:
            time.sleep(INTER_BATCH_DELAY)

    return all_embeddings

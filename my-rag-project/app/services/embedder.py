"""OpenAI embedding calls with batching and retry logic."""

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

_client: OpenAI | None = None

EMBEDDING_BATCH_SIZE = 100


def _get_client() -> OpenAI:
    """Lazy initialization of OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Embed a batch of texts with retry logic.

    Args:
        texts: List of texts to embed
        model: OpenAI embedding model name

    Returns:
        List of embedding vectors
    """
    client = _get_client()
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


def embed_texts_batched(
    texts: list[str], model: str = "text-embedding-3-small"
) -> list[list[float]]:
    """Embed texts in batches of EMBEDDING_BATCH_SIZE.

    Args:
        texts: List of texts to embed
        model: OpenAI embedding model name

    Returns:
        List of embedding vectors in same order as input
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = embed_texts(batch, model=model)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings

"""Pinecone upsert/search logic."""

from pinecone.exceptions import PineconeException
from pinecone.grpc import PineconeGRPC as Pinecone
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.config import get_pinecone_host, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

UPSERT_BATCH_SIZE = 100


def get_index(index_name: str):
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    host = get_pinecone_host(index_name)
    return pc.Index(host=host)


@retry(
    retry=retry_if_exception_type((PineconeException, ConnectionError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=1, max=30),
    reraise=True,
)
def upsert_vectors(index_name: str, namespace: str, vectors: list[dict]) -> int:
    """Upsert vectors to Pinecone in batches with retry logic.

    Args:
        index_name: Name of the Pinecone index
        namespace: Namespace within the index
        vectors: List of vector dicts with keys: id, values, metadata

    Returns:
        Total count of upserted vectors

    Raises:
        PineconeException: If upsert fails after all retries
        ConnectionError: If connection fails after all retries
    """
    if not vectors:
        return 0

    logger.info(
        "Upserting %d vectors to index '%s' namespace '%s'",
        len(vectors),
        index_name,
        namespace,
    )

    index = get_index(index_name)
    total_upserted = 0

    for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[i : i + UPSERT_BATCH_SIZE]
        upsert_data = [
            {"id": v["id"], "values": v["values"], "metadata": v.get("metadata", {})}
            for v in batch
        ]
        index.upsert(vectors=upsert_data, namespace=namespace)
        total_upserted += len(batch)
        logger.debug(
            "Upserted batch %d/%d (%d vectors)",
            i // UPSERT_BATCH_SIZE + 1,
            (len(vectors) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE,
            len(batch),
        )

    logger.info("Successfully upserted %d vectors", total_upserted)
    return total_upserted

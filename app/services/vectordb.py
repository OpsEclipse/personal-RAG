"""Pinecone upsert/search logic."""

from pinecone.grpc import PineconeGRPC as Pinecone

from app.core.config import get_pinecone_host, get_settings

UPSERT_BATCH_SIZE = 100


def get_index(index_name: str):
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    host = get_pinecone_host(index_name)
    return pc.Index(host=host)


def upsert_vectors(index_name: str, namespace: str, vectors: list[dict]) -> int:
    """Upsert vectors to Pinecone in batches.

    Args:
        index_name: Name of the Pinecone index
        namespace: Namespace within the index
        vectors: List of vector dicts with keys: id, values, metadata

    Returns:
        Total count of upserted vectors
    """
    if not vectors:
        return 0

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

    return total_upserted

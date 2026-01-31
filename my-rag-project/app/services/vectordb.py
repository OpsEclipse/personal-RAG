"""Pinecone upsert/search logic."""

from pinecone import Pinecone

UPSERT_BATCH_SIZE = 100


def get_index(index_name: str):
    pc = Pinecone()
    return pc.Index(index_name)


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

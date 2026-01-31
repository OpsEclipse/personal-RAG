"""Pinecone upsert/search logic placeholder."""

from pinecone import Pinecone


def get_index(index_name: str):
    pc = Pinecone()
    return pc.Index(index_name)

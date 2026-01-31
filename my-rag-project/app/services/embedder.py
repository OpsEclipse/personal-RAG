"""OpenAI embedding calls placeholder."""

from openai import OpenAI


client = OpenAI()


def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    # TODO: implement batching + retries
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]

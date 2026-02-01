"""LLM-based namespace routing for chunk classification."""

import httpx

from app.core.config import get_settings
from app.core.namespaces import Namespace, get_namespace_prompt
from app.services.parser import ParsedChunk

_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    """Lazy initialization of OpenRouter HTTP client."""
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.openrouter_api_key.strip():
            raise ValueError("Missing OpenRouter API key.")
        _client = httpx.Client(
            base_url=settings.openrouter_base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Accept": "application/json",
            },
            timeout=20.0,
        )
    return _client


def _build_classification_prompt(text: str, headings: list[str]) -> str:
    """Build the prompt for namespace classification."""
    heading_section = ""
    if headings:
        heading_section = f"\n\nDocument headings:\n- " + "\n- ".join(headings)

    namespace_list = ", ".join(ns.value for ns in Namespace)

    return f"""You are a classifier for a personal portfolio RAG system. Your task is to determine which namespace best fits the given content.

{get_namespace_prompt()}

Analyze the following content and respond with ONLY the namespace name (one of: {namespace_list}). No explanation, just the single word.

Content:{heading_section}

{text[:2000]}"""


def classify_document(chunks: list[ParsedChunk]) -> Namespace:
    """Classify an entire document based on first chunk and all headings.

    Args:
        chunks: List of parsed chunks from the document

    Returns:
        The determined namespace for the entire document
    """
    if not chunks:
        return Namespace.PROJECTS  # Default fallback

    # Gather all unique headings from the document
    headings = []
    for chunk in chunks:
        heading = chunk.metadata.get("heading", "")
        if heading and heading not in headings:
            headings.append(heading)

    # Use first chunk's contextualized text if available
    first_chunk_text = _get_context_text(chunks[0])

    return _call_llm_for_classification(first_chunk_text, headings)


def classify_chunk(chunk: ParsedChunk) -> Namespace:
    """Classify a single chunk individually.

    Args:
        chunk: A single parsed chunk

    Returns:
        The determined namespace for this chunk
    """
    heading = chunk.metadata.get("heading", "")
    headings = [heading] if heading else []

    return _call_llm_for_classification(_get_context_text(chunk), headings)


def classify_chunks_individually(chunks: list[ParsedChunk]) -> list[Namespace]:
    """Classify each chunk individually (per-chunk mode).

    Args:
        chunks: List of parsed chunks

    Returns:
        List of namespaces, one per chunk
    """
    return [classify_chunk(chunk) for chunk in chunks]


def _call_llm_for_classification(text: str, headings: list[str]) -> Namespace:
    """Call the LLM to classify content into a namespace.

    Args:
        text: The text content to classify
        headings: List of relevant headings

    Returns:
        The determined namespace
    """
    if not text.strip() and not headings:
        return Namespace.PROJECTS

    settings = get_settings()
    model = _normalize_model(settings.openrouter_model)
    prompt = _build_classification_prompt(text, headings)
    if not model or not prompt.strip():
        return Namespace.PROJECTS

    try:
        client = _get_client()
    except ValueError:
        return Namespace.PROJECTS
    try:
        response = client.post(
            "/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "stream": False,
            },
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError, TypeError):
        return Namespace.PROJECTS

    result_raw = _extract_message_content(payload)
    first_line = result_raw.splitlines()[0] if result_raw else ""
    first_token = first_line.split(maxsplit=1)[0] if first_line else ""
    result = first_token.strip("`'\".,:;()[]{}")

    # Map response to namespace, with fallback
    namespace_map = {ns.value: ns for ns in Namespace}
    return namespace_map.get(result, Namespace.PROJECTS)


def _get_context_text(chunk: ParsedChunk) -> str:
    """Prefer contextualized text when available for routing."""
    return chunk.metadata.get("context_summary") or chunk.text


def _normalize_model(model: str) -> str:
    """Ensure we use an OpenRouter free-tier model variant."""
    normalized = model.strip()
    if not normalized:
        return ""
    if ":free" not in normalized:
        normalized = f"{normalized}:free"
    return normalized


def _extract_message_content(payload: dict) -> str:
    """Safely extract the assistant message content from OpenRouter payload."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    return str(content).strip().lower()

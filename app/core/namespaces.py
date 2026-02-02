"""Namespace definitions for personal portfolio RAG."""

from enum import Enum


class Namespace(str, Enum):
    """Valid namespaces for the personal portfolio."""

    PERSONAL_LIFE = "personal_life"
    PROFESSIONAL_LIFE = "professional_life"
    ABOUT_RAG = "about_rag"


NAMESPACE_DESCRIPTIONS: dict[Namespace, str] = {
    Namespace.PERSONAL_LIFE: (
        "Personal life, hobbies, interests, values, beliefs, personal stories, "
        "travel, lifestyle, relationships, and non-professional aspects of life."
    ),
    Namespace.PROFESSIONAL_LIFE: (
        "Work experience, projects, internships, jobs, professional roles, career history, "
        "responsibilities, achievements at companies, and employment-related content."
    ),
    Namespace.ABOUT_RAG: (
        "System documentation about this RAG project, architecture notes, "
        "pipeline explanations, data flow details, and operational runbooks."
    ),
}


def get_namespace_prompt() -> str:
    """Generate a prompt section describing all namespaces for LLM classification."""
    lines = ["Available namespaces and their descriptions:"]
    for ns in Namespace:
        lines.append(f"- {ns.value}: {NAMESPACE_DESCRIPTIONS[ns]}")
    return "\n".join(lines)


def is_valid_namespace(value: str) -> bool:
    """Check if a string is a valid namespace."""
    return value in {ns.value for ns in Namespace}

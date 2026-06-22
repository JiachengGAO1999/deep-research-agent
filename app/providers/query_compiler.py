"""Compile provider-neutral search intent text for individual APIs."""

from __future__ import annotations

import re


_QUESTION_PREFIX = re.compile(
    r"^(to what extent|how does|how do|how|what|which|why|does|do|is|are|can|could)\s+",
    re.IGNORECASE,
)


def compile_query(provider: str, query: str) -> str:
    """Return a conservative query accepted by the target provider."""
    cleaned = re.sub(r"[?！？，,;；:：(){}\[\]]", " ", query or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = _QUESTION_PREFIX.sub("", cleaned)
    if provider == "arxiv":
        tokens = [
            token
            for token in cleaned.split()
            if token.upper() not in {"AND", "OR", "NOT"}
        ]
        return " ".join(tokens[:12])
    return cleaned[:500]

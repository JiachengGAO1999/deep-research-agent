"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings from environment variables."""

    # LLM
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL_FAST: str = os.getenv("LLM_MODEL_FAST", "deepseek-chat")
    LLM_MODEL_STRONG: str = os.getenv("LLM_MODEL_STRONG", "")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "120"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))

    # Token limits per role
    LLM_FAST_MAX_TOKENS: int = int(os.getenv("LLM_FAST_MAX_TOKENS", "1024"))
    LLM_STRONG_MAX_TOKENS: int = int(os.getenv("LLM_STRONG_MAX_TOKENS", "4096"))

    # Thinking mode (vLLM chat_template_kwargs)
    LLM_FAST_ENABLE_THINKING: bool = os.getenv(
        "LLM_FAST_ENABLE_THINKING", "false"
    ).lower() in ("1", "true", "yes")
    LLM_STRONG_ENABLE_THINKING: bool = os.getenv(
        "LLM_STRONG_ENABLE_THINKING", "true"
    ).lower() in ("1", "true", "yes")

    # OpenAlex
    OPENALEX_API_KEY: str = os.getenv("OPENALEX_API_KEY", "")
    OPENALEX_MAILTO: str = os.getenv("OPENALEX_MAILTO", "")
    OPENALEX_BASE_URL: str = os.getenv("OPENALEX_BASE_URL", "https://api.openalex.org")

    # Semantic Scholar
    SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    SEMANTIC_SCHOLAR_BASE_URL: str = os.getenv(
        "SEMANTIC_SCHOLAR_BASE_URL", "https://api.semanticscholar.org/graph/v1"
    )

    # arXiv
    ARXIV_BASE_URL: str = os.getenv("ARXIV_BASE_URL", "https://export.arxiv.org/api")

    # Crossref
    CROSSREF_BASE_URL: str = os.getenv("CROSSREF_BASE_URL", "https://api.crossref.org")

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite+aiosqlite:///./storage/app.db"
    )

    # Search limits
    MAX_SEARCH_ROUNDS: int = int(os.getenv("MAX_SEARCH_ROUNDS", "3"))
    MAX_PAPERS_PER_SOURCE: int = int(os.getenv("MAX_PAPERS_PER_SOURCE", "20"))
    MAX_SELECTED_PAPERS: int = int(os.getenv("MAX_SELECTED_PAPERS", "20"))

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # HTTP client
    HTTP_TIMEOUT: int = int(os.getenv("HTTP_TIMEOUT", "30"))
    HTTP_MAX_RETRIES: int = int(os.getenv("HTTP_MAX_RETRIES", "3"))

    # Mock mode (no API keys required)
    MOCK_MODE: bool = os.getenv("MOCK_MODE", "").lower() in ("1", "true", "yes")

    @property
    def has_llm_key(self) -> bool:
        return bool(self.LLM_API_KEY) and self.LLM_API_KEY != "your-api-key-here"

    @property
    def model_fast(self) -> str:
        return self.LLM_MODEL_FAST

    @property
    def model_strong(self) -> str:
        return self.LLM_MODEL_STRONG or self.LLM_MODEL_FAST


@lru_cache()
def get_settings() -> Settings:
    return Settings()

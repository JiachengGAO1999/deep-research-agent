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
    LLM_INPUT_COST_PER_1M: float = float(
        os.getenv("LLM_INPUT_COST_PER_1M", "0")
    )
    LLM_OUTPUT_COST_PER_1M: float = float(
        os.getenv("LLM_OUTPUT_COST_PER_1M", "0")
    )

    # Token limits per role
    LLM_FAST_MAX_TOKENS: int = int(os.getenv("LLM_FAST_MAX_TOKENS", "1024"))
    LLM_STRONG_MAX_TOKENS: int = int(os.getenv("LLM_STRONG_MAX_TOKENS", "4096"))

    # Thinking mode (vLLM chat_template_kwargs — only sent when explicitly set)
    LLM_FAST_ENABLE_THINKING: bool | None = None
    LLM_STRONG_ENABLE_THINKING: bool | None = None

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

    # Per-provider per-query result limits (avoid oversampling; rely on query diversity)
    MAX_RESULTS_PER_QUERY_OPENALEX: int = int(os.getenv("MAX_RESULTS_PER_QUERY_OPENALEX", "20"))
    MAX_RESULTS_PER_QUERY_S2: int = int(os.getenv("MAX_RESULTS_PER_QUERY_S2", "10"))
    MAX_RESULTS_PER_QUERY_ARXIV: int = int(os.getenv("MAX_RESULTS_PER_QUERY_ARXIV", "20"))

    # Candidate pool sizing
    MAX_CANDIDATES_AFTER_DEDUP: int = int(os.getenv("MAX_CANDIDATES_AFTER_DEDUP", "100"))
    MAX_CANDIDATES_FOR_RERANK: int = int(os.getenv("MAX_CANDIDATES_FOR_RERANK", "40"))
    MAX_CANDIDATES_FOR_LLM: int = int(os.getenv("MAX_CANDIDATES_FOR_LLM", "15"))
    CANDIDATE_POOL_MULTIPLIER: int = int(os.getenv("CANDIDATE_POOL_MULTIPLIER", "5"))

    # Backward-compatible fallback
    MAX_PAPERS_PER_SOURCE: int = int(os.getenv("MAX_PAPERS_PER_SOURCE", "20"))
    MAX_SELECTED_PAPERS: int = int(os.getenv("MAX_SELECTED_PAPERS", "20"))

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # HTTP client
    HTTP_TIMEOUT: int = int(os.getenv("HTTP_TIMEOUT", "30"))
    HTTP_MAX_RETRIES: int = int(os.getenv("HTTP_MAX_RETRIES", "3"))

    # PDF storage and parsing
    PDF_CACHE_DIR: str = os.getenv("PDF_CACHE_DIR", "./storage/cache/pdf")
    PDF_CACHE_TTL_DAYS: int = int(os.getenv("PDF_CACHE_TTL_DAYS", "7"))
    DELETE_PDF_AFTER_PARSE: bool = os.getenv(
        "DELETE_PDF_AFTER_PARSE", "false"
    ).lower() in ("1", "true", "yes")
    PERSIST_PARSED_CHUNKS: bool = os.getenv(
        "PERSIST_PARSED_CHUNKS", "true"
    ).lower() in ("1", "true", "yes")
    PDF_MAX_SIZE_MB: int = int(os.getenv("PDF_MAX_SIZE_MB", "50"))
    PDF_DOWNLOAD_TIMEOUT: int = int(os.getenv("PDF_DOWNLOAD_TIMEOUT", "60"))
    ENABLE_FULL_TEXT: bool = os.getenv(
        "ENABLE_FULL_TEXT", "false"
    ).lower() in ("1", "true", "yes")
    PDF_PARSER_BACKEND: str = os.getenv("PDF_PARSER_BACKEND", "docling")

    # Evidence engine
    # Direct workflow callers retain an abstract-safe default. Product API
    # profiles select PaperQA2/Hybrid explicitly.
    EVIDENCE_BACKEND: str = os.getenv("EVIDENCE_BACKEND", "abstract")
    EVIDENCE_TOP_K: int = int(os.getenv("EVIDENCE_TOP_K", "8"))
    EVIDENCE_MAX_PER_PAPER: int = int(os.getenv("EVIDENCE_MAX_PER_PAPER", "3"))
    HYBRID_DENSE_MODEL: str = os.getenv("HYBRID_DENSE_MODEL", "BAAI/bge-m3")
    HYBRID_RERANK_MODEL: str = os.getenv(
        "HYBRID_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"
    )
    HYBRID_DEVICE: str = os.getenv("HYBRID_DEVICE", "cpu")
    HYBRID_ENABLE_RERANKER: bool = os.getenv(
        "HYBRID_ENABLE_RERANKER", "true"
    ).lower() in ("1", "true", "yes")
    HYBRID_RRF_K: int = int(os.getenv("HYBRID_RRF_K", "60"))
    HYBRID_CANDIDATE_MULTIPLIER: int = int(
        os.getenv("HYBRID_CANDIDATE_MULTIPLIER", "5")
    )
    PAPERQA_INDEX_DIR: str = os.getenv(
        "PAPERQA_INDEX_DIR", "./storage/paperqa"
    )
    PAPERQA_SETTINGS: str = os.getenv("PAPERQA_SETTINGS", "high_quality")
    PAPERQA_LLM: str = os.getenv("PAPERQA_LLM", "")
    PAPERQA_SUMMARY_LLM: str = os.getenv("PAPERQA_SUMMARY_LLM", "")
    PAPERQA_EMBEDDING: str = os.getenv(
        "PAPERQA_EMBEDDING", "text-embedding-3-small"
    )

    # Product profiles and cost guardrails
    DEFAULT_RETRIEVAL_PROFILE: str = os.getenv(
        "DEFAULT_RETRIEVAL_PROFILE", "quality"
    )
    QUALITY_EVIDENCE_BACKEND: str = os.getenv(
        "QUALITY_EVIDENCE_BACKEND", "hybrid"
    )
    DEFAULT_MAX_COST_USD: float = float(os.getenv("DEFAULT_MAX_COST_USD", "0"))

    # Report quality gates
    MIN_CORE_CLAIM_SUPPORT: int = int(os.getenv("MIN_CORE_CLAIM_SUPPORT", "1"))
    MAX_UNSUPPORTED_IMPORTANT_CLAIMS: int = int(
        os.getenv("MAX_UNSUPPORTED_IMPORTANT_CLAIMS", "0")
    )
    REPORT_GENERATION_MODE: str = os.getenv(
        "REPORT_GENERATION_MODE", "strict"
    )

    # OA resolution
    UNPAYWALL_EMAIL: str = os.getenv("UNPAYWALL_EMAIL", "")

    # Tavily Search (Quick Research mode)
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    TAVILY_SEARCH_DEPTH: str = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")
    TAVILY_MAX_RESULTS_PER_QUERY: int = int(
        os.getenv("TAVILY_MAX_RESULTS_PER_QUERY", "8")
    )
    TAVILY_EXTRACT_DEPTH: str = os.getenv("TAVILY_EXTRACT_DEPTH", "advanced")
    TAVILY_MAX_SOURCES: int = int(os.getenv("TAVILY_MAX_SOURCES", "20"))
    TAVILY_TIMEOUT: int = int(os.getenv("TAVILY_TIMEOUT", "30"))

    # Quick Research mode limits
    QUICK_MAX_SEARCH_ROUNDS: int = int(os.getenv("QUICK_MAX_SEARCH_ROUNDS", "2"))
    QUICK_MAX_QUERIES_PER_ROUND: int = int(
        os.getenv("QUICK_MAX_QUERIES_PER_ROUND", "6")
    )
    QUICK_MIN_HIGH_QUALITY_SOURCES: int = int(
        os.getenv("QUICK_MIN_HIGH_QUALITY_SOURCES", "5")
    )

    # Mock mode (no API keys required)
    MOCK_MODE: bool = os.getenv("MOCK_MODE", "").lower() in ("1", "true", "yes")

    @property
    def has_tavily_key(self) -> bool:
        return bool(self.TAVILY_API_KEY) and self.TAVILY_API_KEY not in (
            "",
            "your-tavily-key-here",
        )

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

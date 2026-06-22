"""Programmatic verification for direct quotations."""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel

from app.models.evidence import VerificationStatus


class QuoteVerificationResult(BaseModel):
    status: VerificationStatus
    reason: str
    normalized_quote: str


def normalize_quote(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def verify_quote(quote: str, source_text: str) -> QuoteVerificationResult:
    normalized_quote = normalize_quote(quote)
    normalized_source = normalize_quote(source_text)
    if not normalized_quote:
        return QuoteVerificationResult(
            status=VerificationStatus.FAILED,
            reason="empty quote",
            normalized_quote=normalized_quote,
        )
    if normalized_quote in normalized_source:
        return QuoteVerificationResult(
            status=VerificationStatus.VERIFIED,
            reason="normalized exact substring match",
            normalized_quote=normalized_quote,
        )
    return QuoteVerificationResult(
        status=VerificationStatus.FAILED,
        reason="quote is not present in the source passage",
        normalized_quote=normalized_quote,
    )

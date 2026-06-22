"""Deterministic EvidenceCard validation — no LLM involved."""

from __future__ import annotations

import re
from typing import List, Optional

from app.models.document import DocumentChunk
from app.models.evidence import ExtractedEvidence, VerificationStatus

VALID_EVIDENCE_TYPES = {
    "empirical_result", "method_description", "theoretical_claim",
    "limitation", "benchmark_result", "research_question", "background", "other",
}
VALID_PAPER_ROLES = {
    "direct_evidence", "mitigation_method", "benchmark",
    "conceptual_framework", "personalization_study",
    "safety_study", "adjacent_application", "unknown",
}
VALID_DIRECTIONS = {"increase", "decrease", "positive", "negative", None}
MIN_QUOTE_LENGTH = 20


def _extract_numbers(text: str) -> set[str]:
    """Extract percentage and decimal numbers from text."""
    nums = set()
    for m in re.finditer(r"(\d+(?:\.\d+)?\s*%)", text):
        nums.add(m.group(1).replace(" ", ""))
    return nums


def validate_evidence_card(
    evidence: ExtractedEvidence,
    chunk: Optional[DocumentChunk] = None,
) -> list[str]:
    """Deterministic checks on a single EvidenceCard. Returns list of failure reasons.

    An empty list means the card passed all deterministic checks.
    """
    failures: list[str] = []

    # 1. Required fields
    if not evidence.evidence_id:
        failures.append("missing evidence_id")
    if not evidence.paper_id:
        failures.append("missing paper_id")

    # 2. exact_quote must exist and be non-trivial
    quote = (evidence.evidence_quote or "").strip()
    if not quote:
        failures.append("evidence_quote is empty")
    elif len(quote) < MIN_QUOTE_LENGTH:
        failures.append(f"evidence_quote too short ({len(quote)} < {MIN_QUOTE_LENGTH})")
    elif re.match(r"^[\w\s\-:,;.!?'\"()]+$", quote) and len(quote.split()) <= 3:
        # Looks like a title, not a substantive quote
        failures.append("evidence_quote appears to be a title, not a substantive quote")

    # 3. Quote must exist in chunk text (if chunk provided)
    if chunk and quote:
        chunk_text = (chunk.text or "").strip()
        if quote not in chunk_text:
            # Try normalized
            from app.services.quote_verification import normalize_quote
            nq = normalize_quote(quote)
            nc = normalize_quote(chunk_text)
            if nq not in nc:
                failures.append(
                    f"exact_quote not found in chunk {evidence.chunk_id}"
                )

    # 4. paper_id / chunk_id consistency
    if chunk and evidence.chunk_id:
        if evidence.paper_id != chunk.paper_id:
            failures.append(
                f"paper_id mismatch: evidence={evidence.paper_id} chunk={chunk.paper_id}"
            )
    if evidence.chunk_id and chunk and evidence.chunk_id != chunk.chunk_id:
        failures.append(
            f"chunk_id mismatch: evidence={evidence.chunk_id} provided={chunk.chunk_id}"
        )

    # 5. Page / section consistency with chunk
    if chunk and evidence.page_start is not None:
        if chunk.page_start is not None and evidence.page_start != chunk.page_start:
            failures.append(
                f"page_start mismatch: evidence={evidence.page_start} chunk={chunk.page_start}"
            )
    if chunk and evidence.section_title and chunk.section_title:
        if evidence.section_title.strip().lower() != chunk.section_title.strip().lower():
            failures.append("section_title does not match source chunk")

    # 6. Numbers in value/metric must appear in exact_quote
    if evidence.value and quote:
        ev_nums = _extract_numbers(evidence.value)
        quote_nums = _extract_numbers(quote)
        for n in ev_nums:
            if n not in quote_nums:
                failures.append(f"value '{n}' not found in exact_quote")

    if evidence.metric and quote:
        # Metric name (e.g., "accuracy") should appear near the number in the quote
        metric_lower = evidence.metric.lower().strip()
        if metric_lower and metric_lower not in quote.lower():
            failures.append(f"metric '{evidence.metric}' not found in exact_quote")

    # 7. Direction validation
    if evidence.direction is not None:
        if evidence.direction not in VALID_DIRECTIONS:
            failures.append(f"invalid direction '{evidence.direction}'")
        elif evidence.direction in ("increase", "positive") and quote:
            dir_words = {"increase", "improve", "higher", "better", "gain", "boost",
                         "enhance", "positive", "more", "greater"}
            if not any(w in quote.lower() for w in dir_words):
                failures.append(
                    f"direction={evidence.direction} but no supporting word in quote"
                )
        elif evidence.direction in ("decrease", "negative") and quote:
            dir_words = {"decrease", "reduce", "lower", "worse", "drop", "decline",
                         "negative", "less", "fewer"}
            if not any(w in quote.lower() for w in dir_words):
                failures.append(
                    f"direction={evidence.direction} but no supporting word in quote"
                )

    # 8. Enum validation
    if evidence.evidence_type and hasattr(evidence.evidence_type, 'value'):
        pass  # Enum already validated by Pydantic
    if evidence.paper_role not in VALID_PAPER_ROLES:
        failures.append(f"invalid paper_role '{evidence.paper_role}'")

    # 9. is_inference consistency
    if evidence.is_inference and evidence.evidence_level == "direct_quote":
        failures.append("is_inference=True but evidence_level=direct_quote")

    # 10. normalized_statement must not be stronger than exact_quote
    if evidence.normalized_statement and quote:
        ns_len = len(evidence.normalized_statement.strip())
        q_len = len(quote.strip())
        if ns_len > q_len * 1.5 and "significantly" in evidence.normalized_statement.lower():
            failures.append("normalized_statement may be stronger than source quote")

    return failures


def validate_all_evidence(
    evidence_list: List[ExtractedEvidence],
    chunk_map: dict[str, DocumentChunk],
) -> tuple[List[ExtractedEvidence], dict]:
    """Validate all evidence cards. Returns (validated_list, stats).

    Failed cards are marked with verification_status=FAILED and kept
    in the list so downstream can inspect them. Only VERIFIED cards
    should enter claim construction.
    """
    stats = {"total": len(evidence_list), "passed": 0, "failed": 0, "reasons": []}
    validated: List[ExtractedEvidence] = []

    for ev in evidence_list:
        chunk = chunk_map.get(ev.chunk_id or "")
        failures = validate_evidence_card(ev, chunk)

        if failures:
            ev.verification_status = VerificationStatus.FAILED
            ev.verification_reason = "; ".join(failures)
            stats["failed"] += 1
            stats["reasons"].append(
                {"evidence_id": ev.evidence_id, "failures": failures}
            )
        else:
            ev.verification_status = VerificationStatus.VERIFIED
            stats["passed"] += 1

        validated.append(ev)

    return validated, stats

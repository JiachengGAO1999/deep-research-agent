"""Claim construction and task-level evidence quality gates."""

from __future__ import annotations

from collections import defaultdict
import re

from app.models.evidence import (
    EvidenceQualitySummary,
    EvidenceType,
    ExtractedEvidence,
    ResearchClaim,
    VerificationStatus,
)


def build_claims(evidence: list[ExtractedEvidence]) -> list[ResearchClaim]:
    """Build conservative claims only from evidence that passed verification."""
    claims: list[ResearchClaim] = []
    seen: set[tuple[str, str]] = set()
    for item in evidence:
        if item.verification_status != VerificationStatus.VERIFIED:
            continue
        section = (item.section_title or "").casefold()
        if any(
            excluded in section
            for excluded in ("references", "bibliography", "acknowledg")
        ):
            continue
        finding = next((x.strip() for x in item.key_findings if x.strip()), "")
        if not finding:
            finding = (item.evidence_quote or "").strip()
        if not finding:
            continue
        lines = [line.strip() for line in finding.splitlines() if line.strip()]
        if len(lines) >= 2:
            first_line = re.sub(r"\s+", " ", lines[0]).strip()
            looks_like_heading = (
                len(first_line) <= 100
                and (
                    first_line.upper() == first_line
                    or bool(re.match(r"^\d+(?:\.\d+)*\s+\S+", first_line))
                    or first_line.casefold()
                    in {
                        "abstract",
                        "introduction",
                        "conclusion",
                        "discussion",
                        "results",
                        "methods",
                    }
                )
            )
            if looks_like_heading:
                lines = lines[1:]
        finding = " ".join(lines).strip()
        if not finding:
            continue
        first_sentence = re.split(
            r"(?<=[.!?。！？])\s+(?=[A-Z\u4e00-\u9fff])",
            finding,
            maxsplit=1,
        )[0]
        if len(first_sentence) >= 20:
            finding = first_sentence
        if len(finding.split()) < 3:
            continue
        if re.match(r"^(figure|table|references?\b)", finding, re.I):
            continue
        if len(finding) > 500:
            finding = finding[:497].rstrip() + "..."
        dedup_key = (item.paper_id, finding.casefold())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        claims.append(
            ResearchClaim(
                claim_text=finding,
                claim_type="source_finding",
                importance="core",
                evidence_ids=[item.evidence_id],
                paper_ids=[item.paper_id],
                support_status="supported",
                confidence=item.confidence,
                section_id=item.section_title,
            )
        )
    return claims


def evaluate_evidence_quality(
    evidence: list[ExtractedEvidence],
    claims: list[ResearchClaim],
    max_unsupported_important_claims: int = 0,
) -> EvidenceQualitySummary:
    verified = [
        item
        for item in evidence
        if item.verification_status == VerificationStatus.VERIFIED
    ]
    direct = [
        item for item in evidence if item.evidence_type == EvidenceType.DIRECT_QUOTE
    ]
    verified_direct = [
        item
        for item in direct
        if item.verification_status == VerificationStatus.VERIFIED
    ]
    abstract = [
        item for item in evidence if item.evidence_type == EvidenceType.ABSTRACT
    ]
    supported = [
        claim
        for claim in claims
        if claim.support_status == "supported"
        and claim.validation_status == "validated"
    ]
    unsupported_important = [
        claim
        for claim in claims
        if claim.importance == "core"
        and (
            claim.support_status != "supported"
            or claim.validation_status != "validated"
        )
    ]
    quote_rate = len(verified_direct) / len(direct) if direct else 1.0
    completeness = len(supported) / len(claims) if claims else 0.0
    issues: list[str] = []
    if not evidence:
        issues.append("no evidence")
    if not claims:
        issues.append("no supported claims")
    if quote_rate < 1.0:
        issues.append("one or more direct quotes failed source verification")
    if len(unsupported_important) > max_unsupported_important_claims:
        issues.append("unsupported important claim threshold exceeded")
    return EvidenceQualitySummary(
        evidence_count=len(evidence),
        verified_evidence_count=len(verified),
        direct_quote_count=len(direct),
        abstract_evidence_count=len(abstract),
        claim_count=len(claims),
        supported_claim_count=len(supported),
        unsupported_important_claim_count=len(unsupported_important),
        quote_verification_rate=quote_rate,
        citation_completeness=completeness,
        passed=not issues,
        issues=issues,
    )


def claims_by_paper(claims: list[ResearchClaim]) -> dict[str, list[ResearchClaim]]:
    grouped: dict[str, list[ResearchClaim]] = defaultdict(list)
    for claim in claims:
        for paper_id in claim.paper_ids:
            grouped[paper_id].append(claim)
    return dict(grouped)

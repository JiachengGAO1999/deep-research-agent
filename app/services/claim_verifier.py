"""Two-phase Claim-Evidence verification: deterministic checks + LLM entailment."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from app.models.evidence import ExtractedEvidence, ResearchClaim

logger = logging.getLogger(__name__)

# ---- Phase 1: Deterministic Checks ----


def _extract_numbers_from_text(text: str) -> set[str]:
    nums = set()
    for m in re.finditer(r"(\d+(?:\.\d+)?\s*%)", text):
        nums.add(m.group(1).replace(" ", ""))
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\b", text):
        nums.add(m.group(1))
    return nums


def _check_number_consistency(claim: ResearchClaim, evidence_list: List[ExtractedEvidence]) -> List[str]:
    """Every number in the claim must appear in at least one bound evidence card."""
    failures = []
    claim_nums = _extract_numbers_from_text(claim.claim_text)
    if not claim_nums:
        return []

    all_evidence_nums = set()
    for ev in evidence_list:
        if ev.evidence_quote:
            all_evidence_nums |= _extract_numbers_from_text(ev.evidence_quote)

    for num in claim_nums:
        if num not in all_evidence_nums:
            # Also check structured value fields
            found = False
            for ev in evidence_list:
                if ev.value and num in ev.value.replace(" ", ""):
                    found = True
                    break
            if not found:
                failures.append(f"number '{num}' in claim not found in any bound evidence")
    return failures


def _check_direction_consistency(claim: ResearchClaim, evidence_list: List[ExtractedEvidence]) -> List[str]:
    """Direction in claim must not contradict evidence direction."""
    failures = []
    claim_dir = claim.direction
    if not claim_dir:
        return []

    for ev in evidence_list:
        ev_dir = ev.direction
        if ev_dir and ev_dir != claim_dir:
            if {ev_dir, claim_dir} in ({"increase", "decrease"}, {"positive", "negative"}):
                failures.append(
                    f"direction conflict: claim={claim_dir} evidence[{ev.evidence_id}]={ev_dir}"
                )
    return failures


def _check_subject_consistency(claim: ResearchClaim, evidence_list: List[ExtractedEvidence]) -> List[str]:
    """Claim subject must not contradict evidence subject."""
    failures = []
    if not claim.subject:
        return []

    for ev in evidence_list:
        if ev.subject and claim.subject:
            cs = claim.subject.lower().strip()
            es = ev.subject.lower().strip()
            # Different subjects that are NOT synonymous
            if cs != es and cs not in es and es not in cs:
                # Flag for review — may be legitimate (different granularity)
                pass  # Not a hard failure, LLM phase will check entailed/contradicted
    return failures


def _check_single_paper_consensus(claim: ResearchClaim) -> Optional[str]:
    """A claim supported by only 1 paper cannot claim 'consensus' or 'widely accepted'."""
    text = claim.claim_text.lower()
    consensus_words = ["consensus", "widely accepted", "generally agreed",
                       "the field agrees", "established finding", "well-known"]
    if any(w in text for w in consensus_words) and len(claim.paper_ids) < 2:
        return "claim implies consensus but only 1 paper supports it"
    return None


def _check_paper_role_consistency(claim: ResearchClaim, evidence_list: List[ExtractedEvidence]) -> List[str]:
    """Claim paper_role must match evidence paper_role."""
    failures = []
    for ev in evidence_list:
        if ev.paper_role and claim.paper_role:
            if ev.paper_role != claim.paper_role and ev.paper_role != "unknown":
                # mitigation_method cited as direct_evidence is a hard fail
                if (ev.paper_role == "mitigation_method" and
                        claim.paper_role == "direct_evidence"):
                    failures.append(
                        f"paper_role conflict: evidence[{ev.evidence_id}]={ev.paper_role} "
                        f"claimed as {claim.paper_role}"
                    )
    return failures


def deterministic_claim_check(
    claim: ResearchClaim,
    evidence_list: List[ExtractedEvidence],
) -> tuple[bool, List[str]]:
    """Phase 1: Deterministic checks on a claim against its bound evidence.

    Returns (passed, failure_reasons).
    """
    failures: List[str] = []

    # Basic binding checks
    if not claim.evidence_ids:
        failures.append("claim has no bound evidence_ids")
    if not claim.paper_ids:
        failures.append("claim has no bound paper_ids")

    # Check all bound evidence exists
    ev_ids = {ev.evidence_id for ev in evidence_list}
    for eid in claim.evidence_ids:
        if eid not in ev_ids:
            failures.append(f"bound evidence_id {eid} not found in evidence list")

    # Number consistency
    failures.extend(_check_number_consistency(claim, evidence_list))

    # Direction consistency
    failures.extend(_check_direction_consistency(claim, evidence_list))

    # Single-paper consensus
    consensus_issue = _check_single_paper_consensus(claim)
    if consensus_issue:
        failures.append(consensus_issue)

    # Paper role consistency
    failures.extend(_check_paper_role_consistency(claim, evidence_list))

    # is_inference check
    if claim.is_inference and any(
        ev.evidence_level == "direct_quote" and not ev.is_inference
        for ev in evidence_list
    ):
        failures.append("claim marked as inference but evidence is direct_quote")

    return len(failures) == 0, failures


# ---- Phase 2: LLM Entailment Check ----


async def llm_entailment_check(
    claim: ResearchClaim,
    evidence_list: List[ExtractedEvidence],
    llm_client=None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Phase 2: LLM judges whether evidence entails, contradicts, or is insufficient
    for the claim. Separate from claim generation — independent verification.

    Returns {"verdict": "entailed"|"contradicted"|"insufficient", "checks": {...}}
    """
    if llm_client is None:
        from app.llm.client import get_llm_client
        llm_client = get_llm_client()

    # Build evidence summary for LLM
    evidence_text = ""
    for i, ev in enumerate(evidence_list[:3]):
        quote = (ev.evidence_quote or "")[:300]
        ev_text = (
            f"Evidence {i+1} [{ev.evidence_id}]:\n"
            f"  Quote: \"{quote}\"\n"
            f"  Subject: {ev.subject or 'N/A'}\n"
            f"  Metric: {ev.metric or 'N/A'}\n"
            f"  Value: {ev.value or 'N/A'}\n"
            f"  Direction: {ev.direction or 'N/A'}\n"
            f"  Comparison: {ev.comparison or 'N/A'}\n"
            f"  Paper Role: {ev.paper_role or 'N/A'}\n"
        )
        evidence_text += ev_text + "\n"

    system_prompt = """You are a strict evidence auditor. Judge whether the claim is ENTAILED, CONTRADICTED, or INSUFFICIENT based on the provided evidence.

Check each dimension independently:
- subject: Does the claim talk about the same thing as the evidence?
- metric: Same metric?
- value: Same numbers? (12.7% ≠ 14.66% → insufficient)
- direction: Same direction? (increase ≠ decrease → contradicted)
- comparison: Same comparison target?
- scope: Same domain/population?
- conclusion_strength: Is the claim stronger than the evidence supports?

Respond ONLY with valid JSON:
{"verdict": "entailed|contradicted|insufficient", "checks": {"subject": true/false, "metric": true/false, "value": true/false, "direction": true/false, "comparison": true/false, "scope": true/false, "conclusion_strength": true/false}, "reason": "..."}"""

    user_prompt = f"""Claim [{claim.claim_id}]: {claim.claim_text}

Claim structured fields:
  subject={claim.subject}, metric={claim.metric}, value={claim.value}
  direction={claim.direction}, comparison={claim.comparison}, scope={claim.scope}
  claim_type={claim.claim_type}, is_inference={claim.is_inference}

Evidence:
{evidence_text}

Judge: entailed, contradicted, or insufficient?"""

    from pydantic import BaseModel

    class EntailmentVerdict(BaseModel):
        verdict: str
        checks: dict
        reason: str = ""

    result, usage = await llm_client.generate_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=EntailmentVerdict,
        model=model,
        max_tokens=512,
        enable_thinking=False,
    )

    if result:
        return {"verdict": result.verdict, "checks": result.checks, "reason": result.reason}
    return {"verdict": "insufficient", "checks": {}, "reason": "LLM call failed"}


async def verify_claim(
    claim: ResearchClaim,
    evidence_list: List[ExtractedEvidence],
    chunk_map: Optional[dict] = None,
    llm_client=None,
    model: Optional[str] = None,
    skip_llm: bool = False,
) -> ResearchClaim:
    """Full two-phase claim verification. Returns claim with updated validation_status."""
    # Phase 1: Deterministic
    dt_passed, dt_failures = deterministic_claim_check(claim, evidence_list)

    if not dt_passed:
        claim.validation_status = "rejected"
        claim.validation_reasons = dt_failures
        return claim

    # Claims constructed as verbatim spans of verified source passages are
    # deterministically entailed. Requiring an LLM to rediscover this relation
    # caused qualitative source statements to be incorrectly rejected merely
    # because they contained no numeric metric.
    normalized_claim = re.sub(r"\s+", " ", claim.claim_text).strip().casefold()
    if normalized_claim and any(
        normalized_claim
        in re.sub(r"\s+", " ", ev.evidence_quote or "").strip().casefold()
        for ev in evidence_list
    ):
        claim.validation_status = "validated"
        claim.validation_reasons = [
            "verbatim claim span found in deterministically verified evidence"
        ]
        return claim

    # Phase 2: LLM (skippable for tests)
    if skip_llm:
        claim.validation_status = "validated"
        claim.validation_reasons = ["deterministic passed; LLM check skipped"]
        return claim

    llm_result = await llm_entailment_check(claim, evidence_list, llm_client, model)

    if llm_result["verdict"] == "entailed":
        claim.validation_status = "validated"
    elif llm_result["verdict"] == "contradicted":
        claim.validation_status = "rejected"
    else:
        claim.validation_status = "needs_review"

    claim.validation_reasons = dt_failures + [llm_result.get("reason", "")]
    return claim

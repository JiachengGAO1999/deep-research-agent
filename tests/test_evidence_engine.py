import pytest

from app.models.evidence import (
    EvidenceType,
    VerificationStatus,
)
from app.models.paper import Paper
from app.services.claim_evidence import build_claims, evaluate_evidence_quality
from app.services.evidence_engine.abstract import AbstractEvidenceEngine
from app.services.evidence_engine.factory import get_evidence_engine
from app.services.quote_verification import normalize_quote, verify_quote


def test_quote_verifier_accepts_normalized_exact_match():
    source = "The model's reasoning accuracy degrades across multiple turns."
    result = verify_quote(
        "reasoning   accuracy degrades across multiple turns",
        source,
    )
    assert result.status == VerificationStatus.VERIFIED


def test_quote_verifier_rejects_fabricated_quote():
    result = verify_quote("Accuracy improves by 50%.", "Accuracy declined.")
    assert result.status == VerificationStatus.FAILED


@pytest.mark.asyncio
async def test_abstract_engine_returns_verifiable_evidence():
    papers = [
        Paper(
            internal_id="p1",
            title="Dialogue Reasoning",
            abstract="Reasoning accuracy degrades across multiple dialogue turns.",
        ),
        Paper(
            internal_id="p2",
            title="Image Segmentation",
            abstract="A method for segmenting medical images.",
        ),
    ]
    engine = AbstractEvidenceEngine()
    result = await engine.ingest(papers)
    assert result.ingested == 2

    passages = await engine.retrieve(
        question="How does dialogue history affect reasoning?",
        sub_question="reasoning degradation across dialogue turns",
        paper_ids=["p1", "p2"],
        limit=1,
    )
    assert passages
    assert passages[0].paper_id == "p1"

    evidence = await engine.extract("reasoning degradation", passages)
    assert evidence[0].verification_status == VerificationStatus.VERIFIED
    assert evidence[0].evidence_type == EvidenceType.ABSTRACT


@pytest.mark.asyncio
async def test_paperqa_backend_is_explicitly_optional():
    engine = get_evidence_engine("paperqa")
    if not await engine.is_available():
        result = await engine.ingest([Paper(internal_id="p1", title="Test")])
        assert result.ingested == 0
        assert result.warnings


@pytest.mark.asyncio
async def test_paperqa_context_is_mapped_to_raw_source_passage():
    from types import SimpleNamespace

    from app.core.config import Settings
    from app.services.evidence_engine.paperqa import PaperQAEvidenceEngine

    text_record = SimpleNamespace(
        text="Raw source text from the PDF.",
        name="pages 4-5",
        doc=SimpleNamespace(docname="p1"),
    )
    session = SimpleNamespace(
        contexts=[
            SimpleNamespace(
                id="ctx1",
                text=text_record,
                context="LLM-generated summary that must not become a quote.",
                score=9,
            )
        ]
    )

    class FakeDocs:
        async def aget_evidence(self, query, settings=None):
            return session

    engine = PaperQAEvidenceEngine(settings=Settings())
    engine._docs = FakeDocs()
    engine._paper_by_docname = {"p1": "p1"}
    engine.is_available = lambda: _true()
    passages = await engine.retrieve("question", "sub-question", ["p1"], 3)
    assert passages[0].text == "Raw source text from the PDF."
    assert "summary" not in passages[0].text
    assert passages[0].page_start == 4


async def _true():
    return True


def test_claims_only_use_verified_evidence():
    from app.models.evidence import ExtractedEvidence

    verified = ExtractedEvidence(
        paper_id="p1",
        key_findings=["A supported finding."],
        evidence_quote="A supported finding.",
        evidence_type="direct_quote",
        verification_status="verified",
        confidence=0.9,
    )
    failed = ExtractedEvidence(
        paper_id="p2",
        key_findings=["A fabricated finding."],
        evidence_quote="A fabricated finding.",
        evidence_type="direct_quote",
        verification_status="failed",
    )
    claims = build_claims([verified, failed])
    assert len(claims) == 1
    assert claims[0].paper_ids == ["p1"]

    quality = evaluate_evidence_quality([verified, failed], claims)
    assert not quality.passed
    assert quality.quote_verification_rate == 0.5

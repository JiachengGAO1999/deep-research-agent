from app.evaluation.runner import score
import json
from pathlib import Path


def test_evaluation_scores_recall_and_concept_coverage():
    dataset = [
        {
            "case_id": "c1",
            "question": "Question",
            "domain": "test",
            "expected_concepts": ["retrieval", "citation"],
            "gold_papers": [
                {"title": "Grounded RAG", "doi": "10.1/test"}
            ],
        }
    ]
    artifact = {
        "results": [
            {
                "case_id": "c1",
                "selected_papers": [
                    {"title": "Grounded RAG", "doi": "10.1/test"}
                ],
                "report": "Retrieval improves citation quality.",
                "citation_validation": {"is_valid": True},
                "evidence_quality": {"verified_evidence_count": 2},
            }
        ]
    }
    summary = score(dataset, artifact)
    assert summary.mean_paper_recall_at_k == 1.0
    assert summary.mean_concept_coverage == 1.0
    assert summary.citation_integrity_rate == 1.0


def test_external_gold_annotations_are_merged():
    dataset = [
        {
            "case_id": "c1",
            "question": "Question",
            "domain": "test",
            "gold_papers": [],
        }
    ]
    artifact = {
        "results": [
            {
                "case_id": "c1",
                "selected_papers": [{"title": "Gold", "doi": "10.1/gold"}],
                "report": "",
                "citation_validation": {"is_valid": True},
            }
        ]
    }
    gold = {
        "cases": {
            "c1": [{"title": "Gold", "doi": "10.1/gold", "role": "anchor"}]
        }
    }
    summary = score(dataset, artifact, gold)
    assert summary.mean_paper_recall_at_k == 1.0


def test_all_seed_cases_have_multiple_gold_anchors():
    import os
    root = Path(__file__).parent.parent
    questions = json.loads((root / "evals/questions.json").read_text())
    gold = json.loads((root / "evals/gold_annotations.json").read_text())
    assert set(gold["cases"]) == {case["case_id"] for case in questions}
    assert all(len(items) >= 2 for items in gold["cases"].values())


def test_layered_passage_and_claim_metrics():
    dataset = [{
        "case_id": "layered",
        "question": "Question",
        "domain": "test",
        "gold_papers": [{"paper_id": "p1", "title": "Gold Paper"}],
        "gold_passages": [{
            "paper_id": "p1",
            "text": "hybrid retrieval improves scientific evidence recall",
            "relevance": 3,
        }],
        "gold_claims": [{
            "text": "hybrid retrieval improves scientific evidence recall"
        }],
    }]
    artifact = {"results": [{
        "case_id": "layered",
        "discovery_candidates": [{"internal_id": "p1", "title": "Gold Paper"}],
        "selected_papers": [{"internal_id": "p1", "title": "Gold Paper"}],
        "retrieved_passages": [{
            "paper_id": "p1",
            "text": "Results show hybrid retrieval improves scientific evidence recall.",
        }],
        "claims": [{
            "claim_text": "Hybrid retrieval improves scientific evidence recall.",
            "support_status": "supported",
            "validation_status": "validated",
        }],
        "evidence_quality": {"evidence_count": 1, "verified_evidence_count": 1},
        "citation_validation": {"is_valid": True},
        "report": "",
    }]}
    summary = score(dataset, artifact)
    case = summary.cases[0]
    assert case.discovery_recall_at_50 == 1.0
    assert case.passage_recall_at_10 == 1.0
    assert case.passage_ndcg_at_10 == 1.0
    assert case.claim_entailment_precision == 1.0

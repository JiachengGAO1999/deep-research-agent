from app.evaluation.runner import score


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

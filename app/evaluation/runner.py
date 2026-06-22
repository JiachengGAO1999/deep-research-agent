"""Score a saved baseline artifact against the fixed evaluation dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.models import CaseScore, EvaluationCase, EvaluationSummary
from app.models.paper import normalize_title


def _paper_recall(case: EvaluationCase, result: dict) -> float | None:
    if not case.gold_papers:
        return None
    returned = result.get("selected_papers", [])
    returned_dois = {
        (paper.get("doi") or "").lower().strip() for paper in returned
    }
    returned_titles = {
        normalize_title(paper.get("title") or "") for paper in returned
    }
    hits = 0
    for gold in case.gold_papers:
        if gold.doi and gold.doi.lower().strip() in returned_dois:
            hits += 1
        elif normalize_title(gold.title) in returned_titles:
            hits += 1
    return hits / len(case.gold_papers)


def score(dataset: list[dict], artifact: dict) -> EvaluationSummary:
    cases = {
        item.case_id: item
        for item in (EvaluationCase.model_validate(row) for row in dataset)
    }
    scores = []
    for result in artifact.get("results", []):
        case = cases[result["case_id"]]
        report = (result.get("report") or "").casefold()
        covered = sum(
            1 for concept in case.expected_concepts if concept.casefold() in report
        )
        concept_coverage = (
            covered / len(case.expected_concepts)
            if case.expected_concepts
            else 0.0
        )
        citation = result.get("citation_validation") or {}
        quality = result.get("evidence_quality") or {}
        scores.append(
            CaseScore(
                case_id=case.case_id,
                paper_recall_at_k=_paper_recall(case, result),
                concept_coverage=concept_coverage,
                citation_integrity=bool(citation.get("is_valid")),
                selected_paper_count=len(result.get("selected_papers", [])),
                verified_evidence_count=quality.get(
                    "verified_evidence_count", 0
                ),
            )
        )
    recalls = [
        item.paper_recall_at_k
        for item in scores
        if item.paper_recall_at_k is not None
    ]
    return EvaluationSummary(
        cases=scores,
        mean_paper_recall_at_k=(
            sum(recalls) / len(recalls) if recalls else None
        ),
        mean_concept_coverage=(
            sum(item.concept_coverage for item in scores) / len(scores)
            if scores
            else 0.0
        ),
        citation_integrity_rate=(
            sum(item.citation_integrity for item in scores) / len(scores)
            if scores
            else 0.0
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evals/questions.json")
    parser.add_argument("--artifact", default="evals/baselines/latest.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    summary = score(dataset, artifact)
    rendered = summary.model_dump_json(indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

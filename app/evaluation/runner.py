"""Score a saved baseline artifact against the fixed evaluation dataset."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from app.evaluation.models import CaseScore, EvaluationCase, EvaluationSummary
from app.models.paper import normalize_title


def _paper_recall(
    case: EvaluationCase, result: dict, field: str = "selected_papers"
) -> float | None:
    if not case.gold_papers:
        return None
    returned = result.get(field, [])
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


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", (text or "").casefold()))


def _passage_scores(case: EvaluationCase, result: dict) -> tuple[float | None, float | None]:
    if not case.gold_passages:
        return None, None
    returned = result.get("retrieved_passages", [])[:10]
    gains: list[int] = []
    matched_gold: set[int] = set()
    for passage in returned:
        candidate = _tokens(passage.get("text", ""))
        best_index = None
        best_overlap = 0.0
        for index, gold in enumerate(case.gold_passages):
            gold_tokens = _tokens(gold.text)
            overlap = len(candidate & gold_tokens) / max(len(gold_tokens), 1)
            same_paper = (
                not gold.paper_id or passage.get("paper_id") == gold.paper_id
            )
            if same_paper and overlap > best_overlap:
                best_index, best_overlap = index, overlap
        if best_index is not None and best_overlap >= 0.5:
            matched_gold.add(best_index)
            gains.append(case.gold_passages[best_index].relevance)
        else:
            gains.append(0)
    recall = len(matched_gold) / len(case.gold_passages)
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal = sorted(
        (gold.relevance for gold in case.gold_passages), reverse=True
    )[:10]
    idcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal))
    return recall, dcg / idcg if idcg else 0.0


def _claim_precision(case: EvaluationCase, result: dict) -> float | None:
    validated = [
        claim for claim in result.get("claims", [])
        if claim.get("validation_status") == "validated"
    ]
    if not validated:
        return 0.0 if case.gold_claims else None
    if not case.gold_claims:
        return sum(
            claim.get("support_status") == "supported" for claim in validated
        ) / len(validated)
    correct = 0
    for claim in validated:
        candidate = _tokens(claim.get("claim_text", ""))
        if any(
            len(candidate & _tokens(gold.text)) / max(len(_tokens(gold.text)), 1)
            >= 0.5
            for gold in case.gold_claims
        ):
            correct += 1
    return correct / len(validated)


def score(
    dataset: list[dict],
    artifact: dict,
    gold_annotations: dict | None = None,
) -> EvaluationSummary:
    gold_cases = (gold_annotations or {}).get("cases", {})
    resolved_dataset = []
    for row in dataset:
        row = row.copy()
        if not row.get("gold_papers") and row["case_id"] in gold_cases:
            annotation = gold_cases[row["case_id"]]
            paper_annotations = (
                annotation.get("gold_papers", [])
                if isinstance(annotation, dict)
                else annotation
            )
            row["gold_papers"] = [
                {"title": item["title"], "doi": item.get("doi")}
                for item in paper_annotations
            ]
            if isinstance(annotation, dict):
                row["gold_passages"] = annotation.get("gold_passages", [])
                row["gold_claims"] = annotation.get("gold_claims", [])
        resolved_dataset.append(row)
    cases = {
        item.case_id: item
        for item in (
            EvaluationCase.model_validate(row) for row in resolved_dataset
        )
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
        passage_recall, passage_ndcg = _passage_scores(case, result)
        claims = result.get("claims", [])
        unsupported = [
            claim for claim in claims
            if claim.get("support_status") != "supported"
            or claim.get("validation_status") != "validated"
        ]
        evidence_count = int(quality.get("evidence_count", 0) or 0)
        verified_count = int(quality.get("verified_evidence_count", 0) or 0)
        metrics = result.get("metrics") or {}
        scores.append(
            CaseScore(
                case_id=case.case_id,
                discovery_recall_at_50=_paper_recall(
                    case, result, "discovery_candidates"
                ),
                selected_paper_recall_at_k=_paper_recall(case, result),
                paper_recall_at_k=_paper_recall(case, result),
                passage_recall_at_10=passage_recall,
                passage_ndcg_at_10=passage_ndcg,
                evidence_card_validity=(
                    verified_count / evidence_count if evidence_count else 0.0
                ),
                claim_entailment_precision=_claim_precision(case, result),
                concept_coverage=concept_coverage,
                citation_integrity=bool(citation.get("is_valid")),
                unsupported_claim_rate=(
                    len(unsupported) / len(claims) if claims else 0.0
                ),
                selected_paper_count=len(result.get("selected_papers", [])),
                verified_evidence_count=verified_count,
                elapsed_seconds=result.get("elapsed_seconds"),
                estimated_cost_usd=float(metrics.get("estimated_cost_usd", 0) or 0),
            )
        )
    recalls = [
        item.paper_recall_at_k
        for item in scores
        if item.paper_recall_at_k is not None
    ]
    def mean_optional(values):
        resolved = [value for value in values if value is not None]
        return sum(resolved) / len(resolved) if resolved else None

    summary = EvaluationSummary(
        cases=scores,
        mean_discovery_recall_at_50=mean_optional(
            [item.discovery_recall_at_50 for item in scores]
        ),
        mean_selected_paper_recall_at_k=mean_optional(
            [item.selected_paper_recall_at_k for item in scores]
        ),
        mean_paper_recall_at_k=(
            sum(recalls) / len(recalls) if recalls else None
        ),
        mean_passage_recall_at_10=mean_optional(
            [item.passage_recall_at_10 for item in scores]
        ),
        mean_passage_ndcg_at_10=mean_optional(
            [item.passage_ndcg_at_10 for item in scores]
        ),
        mean_claim_entailment_precision=mean_optional(
            [item.claim_entailment_precision for item in scores]
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
        unsupported_claim_rate=(
            sum(item.unsupported_claim_rate for item in scores) / len(scores)
            if scores else 0.0
        ),
        total_estimated_cost_usd=sum(
            item.estimated_cost_usd for item in scores
        ),
    )
    passage_gate = (
        summary.mean_passage_recall_at_10 is None
        or summary.mean_passage_recall_at_10 >= 0.75
    )
    claim_gate = (
        summary.mean_claim_entailment_precision is None
        or summary.mean_claim_entailment_precision >= 0.90
    )
    summary.gates_passed = bool(
        (summary.mean_discovery_recall_at_50 or 0) >= 0.80
        and passage_gate
        and claim_gate
        and summary.citation_integrity_rate == 1.0
        and summary.unsupported_claim_rate == 0.0
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evals/questions.json")
    parser.add_argument("--artifact", default="evals/baselines/latest.json")
    parser.add_argument("--output")
    parser.add_argument("--gold", default="evals/gold_annotations.json")
    args = parser.parse_args()
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    gold = (
        json.loads(Path(args.gold).read_text(encoding="utf-8"))
        if args.gold and Path(args.gold).exists()
        else None
    )
    summary = score(dataset, artifact, gold)
    rendered = summary.model_dump_json(indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

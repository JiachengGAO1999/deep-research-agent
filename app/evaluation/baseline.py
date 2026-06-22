"""Run the workflow over a fixed dataset and save a compact baseline artifact."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path


def _load_cases(path: Path, limit: int | None):
    from app.evaluation.models import EvaluationCase

    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [EvaluationCase.model_validate(item) for item in raw]
    return cases[:limit] if limit else cases


async def _run(args) -> None:
    if args.mock:
        os.environ["MOCK_MODE"] = "1"
        os.environ["LLM_API_KEY"] = ""
    os.environ["EVIDENCE_BACKEND"] = args.backend
    os.environ["ENABLE_FULL_TEXT"] = (
        "true" if args.full_text or args.backend != "abstract" else "false"
    )

    from app.core.config import get_settings
    from app.models.task import TaskMetrics
    from app.workflow.graph import run_research

    get_settings.cache_clear()
    cases = _load_cases(Path(args.dataset), args.limit)
    results = []
    for case in cases:
        started = time.perf_counter()
        state = {
            "task_id": f"baseline_{case.case_id}",
            "original_question": case.question,
            "year_from": case.year_from,
            "year_to": case.year_to,
            "max_rounds": get_settings().MAX_SEARCH_ROUNDS,
            "status": "pending",
            "current_round": 0,
            "queries": [],
            "normalized_papers": [],
            "selected_papers": [],
            "evidence": [],
            "metrics": TaskMetrics(),
            "supplementary_rounds_done": 0,
            "previous_round_paper_ids": [],
            "new_papers_this_round": 0,
        }
        final = await run_research(state)
        quality = final.get("evidence_quality")
        results.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "status": final.get("status"),
                "selected_papers": [
                    {
                        "internal_id": paper.internal_id,
                        "title": paper.title,
                        "doi": paper.doi,
                    }
                    for paper in final.get("selected_papers", [])
                ],
                "discovery_candidates": [
                    {
                        "internal_id": paper.internal_id,
                        "title": paper.title,
                        "doi": paper.doi,
                    }
                    for paper in final.get("discovery_candidates", [])[:50]
                ],
                "retrieved_passages": [
                    item.model_dump(mode="json")
                    for item in final.get("retrieved_passages", [])[:10]
                ],
                "evidence": [
                    {
                        "evidence_id": item.evidence_id,
                        "paper_id": item.paper_id,
                        "passage_id": item.passage_id,
                        "sub_question_id": item.sub_question_id,
                        "evidence_type": item.evidence_type.value,
                        "verification_status": item.verification_status.value,
                        "page_start": item.page_start,
                        "page_end": item.page_end,
                        "evidence_quote": (item.evidence_quote or "")[:240],
                    }
                    for item in final.get("evidence", [])
                ],
                "claims": [
                    item.model_dump(mode="json")
                    for item in final.get("claims", [])
                ],
                "evidence_quality": (
                    quality.model_dump(mode="json") if quality else None
                ),
                "citation_validation": (
                    final["citation_validation"].model_dump(mode="json")
                    if final.get("citation_validation")
                    else None
                ),
                "report": final.get("report"),
                "metrics": final.get("metrics", TaskMetrics()).model_dump(mode="json"),
                "elapsed_seconds": time.perf_counter() - started,
                "warnings": final.get("warnings", []),
                "errors": final.get("errors", []),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "mock": args.mock,
                "evidence_backend": get_settings().EVIDENCE_BACKEND,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(results)} baseline cases to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evals/questions.json")
    parser.add_argument("--output", default="evals/baselines/latest.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("abstract", "fts", "hybrid", "paperqa"),
        default="abstract",
    )
    parser.add_argument("--full-text", action="store_true")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

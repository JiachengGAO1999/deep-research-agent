"""Choose a production retriever from scored PaperQA2 and Hybrid artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def choose_backend(
    hybrid: dict,
    paperqa: dict,
    quality_tolerance: float = 0.02,
) -> dict:
    candidates = {"hybrid": hybrid, "paperqa": paperqa}

    def quality(item: dict) -> float:
        passage = item.get("mean_passage_recall_at_10")
        ndcg = item.get("mean_passage_ndcg_at_10")
        if passage is None or ndcg is None:
            raise ValueError(
                "Both backends require passage gold and non-null Recall@10/nDCG@10"
            )
        return (float(passage) + float(ndcg)) / 2

    quality_scores = {name: quality(item) for name, item in candidates.items()}
    best_quality = max(quality_scores.values())
    quality_band = [
        name
        for name, value in quality_scores.items()
        if best_quality - value <= quality_tolerance
    ]
    winner = min(
        quality_band,
        key=lambda name: (
            float(candidates[name].get("total_estimated_cost_usd", 0) or 0),
            sum(
                float(case.get("elapsed_seconds", 0) or 0)
                for case in candidates[name].get("cases", [])
            ),
        ),
    )
    return {
        "winner": winner,
        "quality_tolerance": quality_tolerance,
        "quality_scores": quality_scores,
        "eligible_within_tolerance": quality_band,
        "winner_gates_passed": bool(candidates[winner].get("gates_passed")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", required=True)
    parser.add_argument("--paperqa", required=True)
    parser.add_argument("--output")
    parser.add_argument("--quality-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    result = choose_backend(
        json.loads(Path(args.hybrid).read_text(encoding="utf-8")),
        json.loads(Path(args.paperqa).read_text(encoding="utf-8")),
        args.quality_tolerance,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

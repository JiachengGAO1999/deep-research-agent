import pytest

from app.evaluation.race import choose_backend


def test_race_prefers_cheaper_backend_within_two_percent():
    hybrid = {
        "mean_passage_recall_at_10": 0.80,
        "mean_passage_ndcg_at_10": 0.80,
        "total_estimated_cost_usd": 1,
        "cases": [{"elapsed_seconds": 10}],
        "gates_passed": True,
    }
    paperqa = {
        "mean_passage_recall_at_10": 0.81,
        "mean_passage_ndcg_at_10": 0.81,
        "total_estimated_cost_usd": 3,
        "cases": [{"elapsed_seconds": 8}],
        "gates_passed": True,
    }
    assert choose_backend(hybrid, paperqa)["winner"] == "hybrid"


def test_race_refuses_to_choose_without_passage_gold():
    with pytest.raises(ValueError):
        choose_backend(
            {"mean_passage_recall_at_10": None, "mean_passage_ndcg_at_10": None},
            {"mean_passage_recall_at_10": None, "mean_passage_ndcg_at_10": None},
        )

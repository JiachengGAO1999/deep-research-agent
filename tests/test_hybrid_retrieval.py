from app.services.evidence_engine.hybrid import reciprocal_rank_fusion


def test_hybrid_rrf_rewards_cross_channel_agreement():
    scores = reciprocal_rank_fusion(
        [["lexical_only", "shared"], ["shared", "dense_only"]],
        k=60,
    )
    assert scores["shared"] > scores["lexical_only"]
    assert scores["shared"] > scores["dense_only"]

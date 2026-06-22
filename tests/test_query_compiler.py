from app.providers.query_compiler import compile_query


def test_openalex_query_removes_question_punctuation():
    compiled = compile_query(
        "openalex",
        "How does varying dialogue history affect reasoning reliability in LLMs?",
    )
    assert "?" not in compiled
    assert not compiled.lower().startswith("how does")


def test_arxiv_query_removes_boolean_tokens_and_limits_terms():
    compiled = compile_query(
        "arxiv",
        "What methods improve reasoning OR reliability AND context in language models?",
    )
    # arXiv fallback joins terms with AND (standard arXiv syntax)
    assert " OR " not in f" {compiled} "
    # Terms should be limited (~12 terms joined by AND)
    assert len(compiled.split(" AND ")) <= 12

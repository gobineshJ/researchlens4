"""
Tests pipeline.run_pipeline's orchestration logic -- specifically query
decomposition and self-correction retry -- with every external boundary
(Groq, ChromaDB/embeddings, PDF export) mocked out. This isolates the
WIRING (does the right thing get called, in the right order, with the
right data flowing between stages) from the real model behavior already
covered by test_query_expansion.py, test_reranker.py, and
test_hybrid_search.py.
"""

from unittest.mock import MagicMock, patch

from app import pipeline


def _chunks_for(source_name: str, n: int, topic: str) -> list[dict]:
    return [
        {"text": f"{source_name} chunk {i} about {topic}", "source_label": f"{source_name}, page {i}",
         "source_type": "file", "source_name": source_name}
        for i in range(n)
    ]


def _run(all_chunks, intent, instructions="Discuss the topic", **overrides):
    """Runs the pipeline with everything mocked except the logic under
    test. `overrides` lets individual tests replace specific mocks."""
    fake_collection = MagicMock()

    def fake_retrieve_candidates(collection, queries, k_per_query, source_name=None):
        pool = [c for c in all_chunks if c.get("source_name") == source_name] if source_name else all_chunks
        return [{**c, "distance": 0.1 * i} for i, c in enumerate(pool[:k_per_query])]

    defaults = dict(
        doc_parser_parse_document=[],
        chunk_all=all_chunks,
        new_session_collection=(fake_collection, "sess1"),
        classify_intent=intent,
        decompose_query=[instructions],
        expand_query=lambda instr, *a, **k: [instr],
        rerank_with_fallback=lambda q, cands, top_k: cands[:top_k],
        generate_report={"report_text": "Report [1]", "citations_used": [all_chunks[0]["source_label"]]},
        build_pdf="/tmp/fake.pdf",
    )
    defaults.update(overrides)

    with patch("app.doc_parser.parse_document", return_value=defaults["doc_parser_parse_document"]), \
         patch("app.chunker.chunk_all", return_value=defaults["chunk_all"]), \
         patch("app.vectorstore.new_session_collection", return_value=defaults["new_session_collection"]), \
         patch("app.vectorstore.add_chunks", return_value=None), \
         patch("app.vectorstore.retrieve_candidates", side_effect=fake_retrieve_candidates), \
         patch("app.groq_client.classify_intent", return_value=defaults["classify_intent"]), \
         patch("app.groq_client.decompose_query", return_value=defaults["decompose_query"]), \
         patch("app.groq_client.expand_query", side_effect=defaults["expand_query"]), \
         patch("app.reranker.rerank_with_fallback", side_effect=defaults["rerank_with_fallback"]), \
         patch("app.groq_client.generate_report", side_effect=defaults["generate_report"] if callable(defaults["generate_report"]) else lambda *a, **k: defaults["generate_report"]), \
         patch("app.pdf_export.build_pdf", return_value=defaults["build_pdf"]):
        return pipeline.run_pipeline(file_paths=[], seed_urls=[], instructions=instructions)


# ---------- Query decomposition wiring ----------

def test_single_question_is_not_decomposed():
    chunks = _chunks_for("doc.pdf", 10, "neural networks")
    result = _run(chunks, intent="targeted_qa", decompose_query=["what dataset did they use?"])
    decomposition_check = next(c for c in result["evaluation"]["checks"] if c["name"] == "query_decomposition")
    assert "single focused question" in decomposition_check["detail"]


def test_compound_question_is_decomposed_and_recorded():
    chunks = _chunks_for("doc.pdf", 10, "neural networks")
    result = _run(
        chunks, intent="analysis",
        decompose_query=["what dataset did they use?", "what are the main limitations?"],
    )
    decomposition_check = next(c for c in result["evaluation"]["checks"] if c["name"] == "query_decomposition")
    assert "decomposed into 2" in decomposition_check["detail"]


def test_summary_intent_skips_decomposition_entirely():
    """"summary" retrieval is positional, not query-driven -- decompose_query
    should never even be called for it."""
    chunks = _chunks_for("doc.pdf", 10, "neural networks")
    with patch("app.groq_client.decompose_query") as mock_decompose:
        _run(chunks, intent="summary")
        mock_decompose.assert_not_called()


# ---------- Self-correction retry ----------

def test_retry_not_triggered_when_first_attempt_is_well_grounded():
    chunks = _chunks_for("doc.pdf", 10, "neural networks")
    result = _run(
        chunks, intent="targeted_qa",
        generate_report={
            "report_text": "Neural networks chunk 0 about neural networks [1].",
            "citations_used": ["doc.pdf, page 0"],
        },
    )
    checks = result["evaluation"]["checks"]
    assert not any(c["name"] == "self_correction" for c in checks)


def test_retry_triggered_and_adopted_when_it_improves_groundedness():
    chunks = _chunks_for("doc.pdf", 10, "neural networks")
    call_count = {"n": 0}

    def generate_report_side_effect(retrieved, instructions):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First attempt: poorly grounded (report text shares no real
            # vocabulary with the retrieved chunks).
            return {"report_text": "Completely unrelated content about gardening [1].",
                     "citations_used": [retrieved[0]["source_label"]]}
        # Retry: well grounded.
        return {"report_text": f"{retrieved[0]['text']} [1].", "citations_used": [retrieved[0]["source_label"]]}

    result = _run(chunks, intent="targeted_qa", generate_report=generate_report_side_effect)

    checks = result["evaluation"]["checks"]
    self_correction = next(c for c in checks if c["name"] == "self_correction")
    assert "using the retried result" in self_correction["detail"]
    assert call_count["n"] == 2
    # The FINAL report should be the retry's (well-grounded) output.
    assert "gardening" not in result["report_text"]


def test_retry_keeps_original_when_retry_does_not_improve():
    chunks = _chunks_for("doc.pdf", 10, "neural networks")

    def generate_report_side_effect(retrieved, instructions):
        # Both attempts poorly grounded -- retry shouldn't help here.
        return {"report_text": "Completely unrelated content about gardening [1].",
                 "citations_used": [retrieved[0]["source_label"]]}

    result = _run(chunks, intent="targeted_qa", generate_report=generate_report_side_effect)

    checks = result["evaluation"]["checks"]
    self_correction = next(c for c in checks if c["name"] == "self_correction")
    assert "keeping the original result" in self_correction["detail"]
    assert "gardening" in result["report_text"]


def test_retry_never_triggered_for_summary_intent():
    """Positional sampling doesn't depend on the search query, so
    reformulating queries and retrying wouldn't change anything -- the
    pipeline should never even compute a retry for "summary"."""
    chunks = _chunks_for("doc.pdf", 10, "neural networks")
    result = _run(
        chunks, intent="summary",
        generate_report={"report_text": "Completely unrelated content about gardening [1].",
                          "citations_used": ["doc.pdf, page 0"]},
    )
    checks = result["evaluation"]["checks"]
    assert not any(c["name"] == "self_correction" for c in checks)


def test_retry_reformulates_queries_avoiding_the_originals():
    chunks = _chunks_for("doc.pdf", 10, "neural networks")
    seen_avoid_args = []

    def expand_query_side_effect(instr, *args, **kwargs):
        seen_avoid_args.append(kwargs.get("avoid"))
        return [instr]

    def generate_report_side_effect(retrieved, instructions):
        return {"report_text": "Completely unrelated content about gardening [1].",
                 "citations_used": [retrieved[0]["source_label"]]}

    _run(
        chunks, intent="targeted_qa",
        expand_query=expand_query_side_effect,
        generate_report=generate_report_side_effect,
    )
    # First call (initial attempt) should have no avoid list; the retry's
    # call should pass one.
    assert seen_avoid_args[0] is None
    assert any(a for a in seen_avoid_args[1:])

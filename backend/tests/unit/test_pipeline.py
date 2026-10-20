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


def _chunks_for(source_name: str, n: int, topic: str, chunk_chars: int = 0) -> list[dict]:
    return [
        {"text": f"{source_name} chunk {i} about {topic}" + " filler" * (chunk_chars // 7),
         "source_label": f"{source_name}, page {i}",
         "source_type": "file", "source_name": source_name, "position": i}
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
        plan_follow_up=lambda *a, **k: {"complete": True, "missing": "", "queries": []},
        revise_claims=lambda flagged, chunks, **k: [None] * len(flagged),
        judge_claims=lambda claims, chunks, **k: [
            {"claim": c["text"], "sources": c["sources"], "verdict": "supported", "reason": ""} for c in claims
        ],
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
         patch("app.pdf_export.build_pdf", return_value=defaults["build_pdf"]),          patch("app.groq_client.plan_follow_up", side_effect=defaults["plan_follow_up"]),          patch("app.groq_client.judge_claims", side_effect=defaults["judge_claims"]),          patch("app.groq_client.revise_claims", side_effect=defaults["revise_claims"]):
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


# ---------- Summary: positional sampling sized to the budget ----------

def _check(result, name):
    return next((c for c in result["evaluation"]["checks"] if c["name"] == name), None)


def _summary_run(chunks, **overrides):
    return _run(chunks, intent="summary", instructions="Summarize these documents", **overrides)


def _reported_run_chunks():
    """The shape of the real run whose span_coverage failed three times:
    three papers of 36/33/65 chunks, each chunk ~630 budget tokens, far
    more than one 5000-token generation request can hold."""
    return (_chunks_for("s41591-021-01506-3.pdf", 36, "x", chunk_chars=2500)
            + _chunks_for("s41591-022-01811-5 (1).pdf", 33, "y", chunk_chars=2500)
            + _chunks_for("Machine_Learning_ML_in_Medicine.pdf", 65, "z", chunk_chars=2500))


def test_reported_run_now_spans_every_document_end_to_end():
    """Regression for the reported failure: spans of 40%, 22% and 41%, because
    the budget trim dropped chunks by relevance rank instead of position."""
    result = _summary_run(_reported_run_chunks())
    spans = [c for c in result["evaluation"]["checks"] if c["name"] == "span_coverage"]
    assert len(spans) == 3
    for check in spans:
        assert check["passed"], check["detail"]
        assert check["value"] == 1.0, check["detail"]


def test_reported_run_survives_even_when_the_model_ranks_opening_chunks_last():
    """Worst case for the old code: a reranker that puts every document's
    opening chunks at the bottom. The summary path no longer reranks at all,
    so this must not matter."""
    def opening_chunks_last(queries, candidates, top_k):
        return sorted(candidates, key=lambda c: c.get("position", 0), reverse=True)[:top_k]

    result = _summary_run(_reported_run_chunks(), rerank_with_fallback=opening_chunks_last)
    assert all(c["passed"] for c in result["evaluation"]["checks"] if c["name"] == "span_coverage")


def test_summary_stays_within_the_generation_budget():
    from app import chunker
    from app.config import GROQ_MAX_CONTEXT_TOKENS
    result = _summary_run(_reported_run_chunks())
    sent = sum(chunker.estimate_tokens(c["text"]) for c in result["retrieved_chunks"])
    assert sent <= GROQ_MAX_CONTEXT_TOKENS


def test_summary_sample_is_sent_in_document_order():
    """Document order is the right order for a summary prompt -- and it is
    what the budget trim's endpoint-first ordering depends on not being
    shuffled by relevance."""
    result = _summary_run(_reported_run_chunks())
    by_source = {}
    for c in result["retrieved_chunks"]:
        by_source.setdefault(c["source_name"], []).append(c["position"])
    for positions in by_source.values():
        assert positions == sorted(positions)


def test_budget_fit_explains_the_trimmed_excerpts():
    result = _summary_run(_reported_run_chunks())
    check = _check(result, "budget_fit")
    assert check["passed"]
    assert check["value"] > 3 * 2  # far more positions than whole chunks would allow
    assert "trimmed to" in check["detail"]


def test_budget_fit_fails_loudly_when_excerpts_must_be_too_thin():
    """Every document still gets both ends, but with enough documents each
    excerpt drops below the useful minimum -- that must be reported."""
    chunks = []
    for i in range(15):
        chunks += _chunks_for(f"doc{i}.pdf", 20, "x", chunk_chars=4000)
    check = _check(_summary_run(chunks), "budget_fit")
    assert not check["passed"]
    assert "below the" in check["detail"]


def test_summary_path_does_not_rerank():
    """Reranking a positional sample can only reorder it, and that reordering
    is what let the budget trim drop a document's opening chunks."""
    chunks = _chunks_for("doc.pdf", 8, "neural networks")
    calls = []

    def recording_rerank(queries, candidates, top_k):
        calls.append(queries)
        return candidates[:top_k]

    _summary_run(chunks, rerank_with_fallback=recording_rerank)
    assert calls == []


def test_summary_confidence_explains_there_is_no_relevance_score():
    check = _check(_summary_run(_chunks_for("doc.pdf", 8, "x")), "confidence")
    assert check["passed"]
    assert "selected by position" in check["detail"]


# ---------- Summary: back matter is never sampled ----------

_REFERENCES = " ".join(f"{n}. Smith, J. et al. A study. Nature 1, 2 (2020)." for n in range(1, 60))
_CAPTION = "Extended Data Fig. 3 | Calibration plots for the validation cohort."


def _paper(name: str, body: int, references: int, captions: int) -> list[dict]:
    """Main text, then a reference list, then figure captions -- the order
    Nature-style papers arrive in, and why "the last chunk" is not a conclusion."""
    texts = ([f"{name} body {i} " + "finding " * 600 for i in range(body)]
             + [_REFERENCES] * references
             + [_CAPTION + " filler" * 50] * captions)
    return [{"text": t, "source_label": f"{name}, page {i}", "source_type": "file",
             "source_name": name, "position": i} for i, t in enumerate(texts)]


def test_summary_never_samples_references_or_figure_captions():
    chunks = _paper("a.pdf", 20, 4, 6) + _paper("b.pdf", 15, 3, 5)
    result = _summary_run(chunks)
    from app import chunker
    assert result["retrieved_chunks"]
    assert all(chunker.back_matter_kind(c["text"]) is None for c in result["retrieved_chunks"])


def test_summary_last_sampled_chunk_is_the_last_content_chunk():
    chunks = _paper("a.pdf", 20, 4, 6)
    positions = [c["position"] for c in _summary_run(chunks)["retrieved_chunks"]]
    assert min(positions) == 0
    assert max(positions) == 19  # end of the body, not the captions at 29


def test_span_coverage_is_measured_against_content_not_back_matter():
    """A selection that reaches the last content chunk covers the whole
    document -- the excluded back matter must not count as missed text."""
    result = _summary_run(_paper("a.pdf", 20, 4, 6))
    span = _check(result, "span_coverage")
    assert span["passed"] and span["value"] == 1.0
    assert "content" in span["detail"]


def test_back_matter_check_says_what_was_excluded_and_why():
    check = _check(_summary_run(_paper("a.pdf", 20, 4, 6)), "back_matter")
    assert check["passed"]  # informational
    assert check["value"] == 10
    assert "references" in check["detail"] and "figure caption" in check["detail"]


def test_back_matter_check_reports_when_nothing_was_excluded():
    check = _check(_summary_run(_chunks_for("doc.pdf", 8, "x")), "back_matter")
    assert check["value"] == 0
    assert "sampling from every chunk" in check["detail"]


def test_targeted_qa_does_not_filter_back_matter():
    """Back matter is only excluded for summaries -- a question can be about
    the references or a figure, and retrieval should be able to find it."""
    result = _run(_paper("a.pdf", 5, 2, 2), intent="targeted_qa")
    assert _check(result, "back_matter") is None


def test_extra_positions_go_to_the_longest_document():
    """With four documents the 20 excerpt slots run out, so allocation matters."""
    chunks = _chunks_for("long.pdf", 60, "x", chunk_chars=2500)
    for name in ("short.pdf", "short2.pdf", "short3.pdf"):
        chunks += _chunks_for(name, 10, "y", chunk_chars=2500)
    counts = {}
    for c in _summary_run(chunks)["retrieved_chunks"]:
        counts[c["source_name"]] = counts.get(c["source_name"], 0) + 1
    assert counts["long.pdf"] > counts["short.pdf"]


# ---------- Test isolation ----------

def test_pipeline_runs_in_tests_never_touch_the_real_run_history():
    """Regression: tests used to append to the real evaluation_log.jsonl."""
    from pathlib import Path
    from app import config, evaluation
    real = Path(config.EVALUATION_LOG_PATH).resolve()
    before = real.read_bytes() if real.exists() else None

    _run(_chunks_for("doc.pdf", 3, "x"), intent="targeted_qa")

    assert (real.read_bytes() if real.exists() else None) == before
    isolated = Path(evaluation.EVALUATION_LOG_PATH)
    assert isolated.resolve() != real and isolated.exists()

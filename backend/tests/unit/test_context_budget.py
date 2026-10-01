"""
Tests for groq_client.fit_chunks_to_budget -- keeps the generation request
under Groq's per-request size limit (the cause of "413 Payload Too Large").
"""

from app import groq_client


def _chunk(source, text):
    return {"source_name": source, "source_label": source, "text": text}


def test_returns_all_chunks_when_under_budget():
    chunks = [_chunk("a.pdf", "x" * 40), _chunk("b.pdf", "y" * 40)]
    assert groq_client.fit_chunks_to_budget(chunks, max_tokens=1000) == chunks


def test_trims_over_budget_but_keeps_every_source():
    # Each chunk ~101 estimated tokens; budget fits 4 of the 6.
    chunks = [_chunk("a.pdf", "a" * 400) for _ in range(4)] + [_chunk("b.pdf", "b" * 400) for _ in range(2)]
    result = groq_client.fit_chunks_to_budget(chunks, max_tokens=410)

    assert len(result) == 4
    assert {c["source_name"] for c in result} == {"a.pdf", "b.pdf"}
    assert sum(groq_client._estimate_tokens(c["text"]) for c in result) <= 410


def test_preserves_original_order():
    chunks = [_chunk("a.pdf", "a1" * 100), _chunk("b.pdf", "b1" * 100), _chunk("a.pdf", "a2" * 100)]
    result = groq_client.fit_chunks_to_budget(chunks, max_tokens=1000)
    assert [c["text"][:2] for c in result] == ["a1", "b1", "a2"]


def test_single_oversized_chunk_is_truncated_not_dropped():
    result = groq_client.fit_chunks_to_budget([_chunk("a.pdf", "z" * 10_000)], max_tokens=100)
    assert len(result) == 1
    assert len(result[0]["text"]) <= 400


def test_empty_input_returns_empty_list():
    assert groq_client.fit_chunks_to_budget([], max_tokens=100) == []


# ---------- preserve_span (the "summary" intent path) ----------

def _positioned(source: str, position: int, chars: int = 400) -> dict:
    return {"source_name": source, "source_label": source,
            "text": f"{source}-{position}-" + "x" * chars, "position": position}


def _spans(result):
    """{source: (min position, max position)} among the kept chunks."""
    by_source = {}
    for c in result:
        by_source.setdefault(c["source_name"], []).append(c["position"])
    return {name: (min(p), max(p)) for name, p in by_source.items()}


def _summary_run():
    """Reproduces the real failure: 3 documents, 5 positionally sampled
    chunks each, handed to the trim in CROSS-ENCODER RELEVANCE order --
    here worst case, with each document's opening chunks ranked last."""
    sampled = {"a.pdf": [0, 7, 14, 21, 28], "b.pdf": [0, 6, 13, 19, 26], "c.pdf": [0, 13, 26, 39, 52]}
    return [
        _positioned(source, position)
        for source, positions in sampled.items()
        for position in reversed(positions)
    ], sampled


def test_preserve_span_keeps_each_documents_first_and_last_chunk():
    chunks, sampled = _summary_run()
    # ~104 tokens per chunk; room for about 8 of the 15, as in the real run.
    result = groq_client.fit_chunks_to_budget(chunks, max_tokens=850, preserve_span=True)

    assert len(result) < len(chunks)  # the trim really did bite
    spans = _spans(result)
    for source, positions in sampled.items():
        assert spans[source] == (min(positions), max(positions)), source


def test_without_preserve_span_relevance_order_collapses_the_span():
    """The bug this fixes: trimming by relevance rank is position-blind, so
    a document's opening chunks can be the ones dropped."""
    chunks, sampled = _summary_run()
    result = groq_client.fit_chunks_to_budget(chunks, max_tokens=850)

    spans = _spans(result)
    collapsed = [s for s, p in sampled.items() if spans[s] != (min(p), max(p))]
    assert collapsed, "expected the default ordering to lose at least one endpoint"


def test_preserve_span_still_returns_chunks_in_input_order():
    chunks, _ = _summary_run()
    result = groq_client.fit_chunks_to_budget(chunks, max_tokens=850, preserve_span=True)
    assert result == [c for c in chunks if c in result]


def test_preserve_span_stays_within_budget():
    chunks, _ = _summary_run()
    result = groq_client.fit_chunks_to_budget(chunks, max_tokens=850, preserve_span=True)
    assert sum(groq_client._estimate_tokens(c["text"]) for c in result) <= 850


def test_preserve_span_still_covers_every_source_without_position_fields():
    """Chunks lacking "position" have no span to preserve, but the flag must
    still not cost a source its representation or blow the budget."""
    chunks = [_chunk("a.pdf", "a" * 400) for _ in range(4)] + [_chunk("b.pdf", "b" * 400) for _ in range(2)]
    result = groq_client.fit_chunks_to_budget(chunks, max_tokens=410, preserve_span=True)
    assert {c["source_name"] for c in result} == {"a.pdf", "b.pdf"}
    assert sum(groq_client._estimate_tokens(c["text"]) for c in result) <= 410


def test_preserve_span_completes_one_document_before_starting_another():
    """When the budget cannot give every document both endpoints, finishing
    some documents' spans beats leaving every document with a lone chunk and
    no span at all -- but one round-robin pass still runs first, so no
    document loses representation entirely."""
    chunks = [_positioned(source, position, chars=1000)
              for source in ("a.pdf", "b.pdf", "c.pdf")
              for position in (0, 30)]
    # ~254 tokens each; room for 4 of the 6.
    result = groq_client.fit_chunks_to_budget(chunks, max_tokens=1020, preserve_span=True)

    assert len(result) == 4
    assert {c["source_name"] for c in result} == {"a.pdf", "b.pdf", "c.pdf"}
    spans = _spans(result)
    full = [s for s, (lo, hi) in spans.items() if (lo, hi) == (0, 30)]
    assert len(full) == 1  # one complete span, rather than three spanless singles


def test_preserve_span_orders_a_two_chunk_source_by_position():
    """The endpoint-first ordering has to cover the 2-chunk case too -- that is
    the minimum the summary excerpt plan gives any document."""
    chunks = [_positioned("a.pdf", 30), _positioned("a.pdf", 0)]  # relevance order
    result = groq_client.fit_chunks_to_budget(chunks, max_tokens=105, preserve_span=True)
    assert [c["position"] for c in result] == [0]

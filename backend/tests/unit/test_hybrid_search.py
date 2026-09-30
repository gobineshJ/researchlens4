"""
Tests BM25 keyword search and RRF fusion. Both are pure functions with
no network/model dependency, so these run fast and deterministically.
"""

from app import hybrid_search


def _chunk(text: str) -> dict:
    return {"text": text, "source_label": "x", "source_type": "file", "source_name": "doc.pdf"}


# ---------- BM25 search ----------

def test_search_finds_exact_keyword_match():
    # More than 2 chunks, and the shared term appearing in only ONE of
    # them -- with just 2 equally-split documents, BM25's idf formula
    # (log((N - freq + 0.5) / (freq + 0.5))) degenerates to exactly zero
    # when a term appears in precisely half the corpus, which isn't a
    # realistic scenario and would make every score 0 regardless of
    # whether the search logic works.
    chunks = [
        _chunk("the mitochondria is the powerhouse of the cell"),
        _chunk("stock markets fluctuate based on investor sentiment"),
        _chunk("photosynthesis converts sunlight into chemical energy"),
        _chunk("interest rates influence consumer borrowing behavior"),
    ]
    index, indexed_chunks = hybrid_search.build_index(chunks)
    results = hybrid_search.search(index, indexed_chunks, "mitochondria powerhouse", top_k=3)
    assert results
    assert results[0]["text"] == chunks[0]["text"]
    assert results[0]["bm25_score"] > 0


def test_search_excludes_zero_overlap_chunks():
    chunks = [
        _chunk("the mitochondria is the powerhouse of the cell"),
        _chunk("stock markets fluctuate based on investor sentiment"),
        _chunk("completely unrelated content about gardening tools"),
        _chunk("interest rates influence consumer borrowing behavior"),
    ]
    index, indexed_chunks = hybrid_search.build_index(chunks)
    results = hybrid_search.search(index, indexed_chunks, "mitochondria", top_k=5)
    assert len(results) == 1
    assert results[0]["text"] == chunks[0]["text"]


def test_build_index_handles_empty_chunk_list():
    index, chunks = hybrid_search.build_index([])
    assert index is None
    assert chunks == []
    assert hybrid_search.search(index, chunks, "anything", top_k=5) == []


def test_search_returns_at_most_top_k():
    # A query term that appears in EVERY chunk has zero (or negative,
    # once BM25's epsilon floor kicks in) idf -- it doesn't distinguish
    # anything, correctly. So this mixes in filler chunks that DON'T
    # contain the shared terms, giving them a real, positive idf and
    # something genuine to rank by.
    matching = [
        _chunk(f"shared keyword topic appears here with extra padding word{i}")
        for i in range(5)
    ]
    filler = [_chunk(f"totally different filler content number {i}") for i in range(7)]
    index, indexed_chunks = hybrid_search.build_index(matching + filler)
    results = hybrid_search.search(index, indexed_chunks, "shared keyword topic", top_k=3)
    assert len(results) == 3


# ---------- RRF fusion ----------

def test_fuse_rrf_ranks_chunk_first_in_both_lists_highest():
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    vector_ranked = [a, b, c]   # a is rank 1
    bm25_ranked = [a, c, b]     # a is rank 1 here too
    fused = hybrid_search.fuse_rrf(vector_ranked, bm25_ranked, k_rrf=60)
    assert fused[0]["text"] == "a"


def test_fuse_rrf_gives_credit_for_appearing_in_only_one_list():
    a, b = _chunk("a"), _chunk("b")
    vector_ranked = [a]
    bm25_ranked = [b]
    fused = hybrid_search.fuse_rrf(vector_ranked, bm25_ranked, k_rrf=60)
    texts = {c["text"] for c in fused}
    assert texts == {"a", "b"}
    # Both ranked #1 in their own list -- should score identically
    scores = {c["text"]: c["rrf_score"] for c in fused}
    assert scores["a"] == scores["b"]


def test_fuse_rrf_outranks_single_list_appearance_with_double_appearance():
    a, b = _chunk("a"), _chunk("b")
    vector_ranked = [a, b]   # a rank 1, b rank 2
    bm25_ranked = [b, a]     # b rank 1, a rank 2
    fused = hybrid_search.fuse_rrf(vector_ranked, bm25_ranked, k_rrf=60)
    # a and b each appear once at rank 1 and once at rank 2 -- should tie
    scores = {c["text"]: c["rrf_score"] for c in fused}
    assert scores["a"] == scores["b"]

    # Now give "a" a rank-1 appearance in a THIRD list-like scenario:
    # appearing in both lists at rank 1 should beat appearing in only one.
    only_in_one = hybrid_search.fuse_rrf([a], [], k_rrf=60)
    in_both = hybrid_search.fuse_rrf([a], [a], k_rrf=60)
    assert in_both[0]["rrf_score"] > only_in_one[0]["rrf_score"]


def test_fuse_rrf_handles_empty_lists():
    assert hybrid_search.fuse_rrf([], [], k_rrf=60) == []

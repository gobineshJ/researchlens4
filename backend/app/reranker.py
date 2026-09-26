"""
Reranks a pool of candidate chunks against a query using a cross-encoder
model. Vector search (ChromaDB + sentence-transformers) is fast but only
approximates relevance via embedding distance; a cross-encoder looks at
the query and each candidate TOGETHER and scores relevance directly,
which is slower (so it only runs on a small oversampled pool, not the
whole collection) but more accurate.

Lazy-loads the model the same way vectorstore.py does for the embedder --
first call downloads it from Hugging Face and caches it; later calls
reuse the cached instance.
"""

from sentence_transformers import CrossEncoder

from .config import RERANKER_MODEL_NAME, RERANK_SNIPPET_CHARS

_reranker = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def rerank(queries: str | list[str], candidates: list[dict], top_k: int) -> list[dict]:
    """
    Scores each candidate against EVERY query in `queries` and keeps
    each candidate's BEST (highest) score, then returns the top_k by
    that score. Accepts a single string for backward compatibility.

    Why multiple queries: a cross-encoder scores how well a passage
    answers a SPECIFIC query -- a vague one-word instruction like
    "summary" has no topic for the model to match against, so every
    candidate scores deeply negative regardless of true relevance
    (confirmed: -7.160 average on a demonstrably well-grounded real
    run). groq_client.expand_query() already turns vague instructions
    into several specific search-style queries -- passing ALL of them
    here, and keeping each candidate's best match across all of them,
    means a chunk only needs to be genuinely relevant to ONE reasonable
    phrasing of what the user meant, not to their exact literal wording.

    Adds "rerank_score" (the best score found) and "rerank_query" (which
    query produced it, for debugging/transparency) to each candidate.
    Scoring uses only the first RERANK_SNIPPET_CHARS of each candidate's
    text, not the full chunk -- keeps inputs closer to the short-passage
    length ms-marco-MiniLM-L-6-v2 was actually trained on. The FULL text
    is still what gets returned and later sent to Groq.
    """
    if not candidates:
        return []
    if isinstance(queries, str):
        queries = [queries]

    model = _get_reranker()
    snippets = [c["text"][:RERANK_SNIPPET_CHARS] for c in candidates]

    # best_score[i] / best_query[i] track candidate i's best result so far
    best_score = [float("-inf")] * len(candidates)
    best_query = [None] * len(candidates)

    for query in queries:
        pairs = [(query, snippet) for snippet in snippets]
        scores = model.predict(pairs)
        for i, score in enumerate(scores):
            if float(score) > best_score[i]:
                best_score[i] = float(score)
                best_query[i] = query

    scored = [
        {**candidate, "rerank_score": best_score[i], "rerank_query": best_query[i]}
        for i, candidate in enumerate(candidates)
    ]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]


def rerank_with_fallback(queries: str | list[str], candidates: list[dict], top_k: int) -> list[dict]:
    """
    Same as rerank(), but never lets a reranker failure (model can't
    download, out of memory, etc.) break the whole pipeline. Falls back
    to the candidates' existing embedding "distance" (lower = better,
    from the vector search that built this candidate pool) instead --
    degraded quality, not a crash. Chunks from the fallback path have no
    "rerank_score" field, which evaluation.py's confidence check uses to
    tell the two paths apart.
    """
    if not candidates:
        return []
    try:
        return rerank(queries, candidates, top_k)
    except Exception:
        ranked_by_distance = sorted(candidates, key=lambda c: c.get("distance", float("inf")))
        return ranked_by_distance[:top_k]

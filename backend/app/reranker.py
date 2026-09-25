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


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Scores each candidate against `query` and returns the top_k, each
    with a "rerank_score" field added (higher = more relevant). Existing
    fields (text, source_label, source_name, distance, etc.) are kept
    unchanged -- this only adds a field and reorders/truncates the list.

    Scoring uses only the first RERANK_SNIPPET_CHARS of each candidate's
    text, not the full chunk -- ms-marco-MiniLM-L-6-v2 was trained on
    short passages, and scoring full 650-token chunks pushes scores
    systematically negative regardless of true relevance (see config.py).
    The FULL text is still what gets returned and later sent to Groq --
    only the score itself is based on the snippet.

    If candidates is empty, or top_k >= len(candidates), still runs the
    scoring (so rerank_score is always present on returned chunks) but
    obviously doesn't need to drop anything.
    """
    if not candidates:
        return []

    model = _get_reranker()
    pairs = [(query, c["text"][:RERANK_SNIPPET_CHARS]) for c in candidates]
    scores = model.predict(pairs)

    scored = [
        {**candidate, "rerank_score": float(score)}
        for candidate, score in zip(candidates, scores)
    ]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]


def rerank_with_fallback(query: str, candidates: list[dict], top_k: int) -> list[dict]:
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
        return rerank(query, candidates, top_k)
    except Exception:
        ranked_by_distance = sorted(candidates, key=lambda c: c.get("distance", float("inf")))
        return ranked_by_distance[:top_k]

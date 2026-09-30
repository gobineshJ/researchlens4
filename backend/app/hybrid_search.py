"""
BM25 keyword search, plus Reciprocal Rank Fusion (RRF) to combine it
with the existing dense vector search in vectorstore.py.

Why this exists alongside vector search: embeddings are good at
semantic similarity but can miss exact-term matches (a specific
acronym, dataset name, or method name that appears verbatim in the
text but isn't semantically "close" to how the query phrases it).
BM25 is the opposite -- great at exact terms, blind to paraphrase.
Combining their RANKINGS (not raw scores, which live on incomparable
scales) via RRF gets the benefit of both without having to tune a
weighting between two different scoring systems.

Pure Python (rank_bm25), no network access or model download needed --
unlike the embedder and reranker, this has zero cold-start cost.
"""

import re

from rank_bm25 import BM25Okapi

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "this", "that", "these", "those",
    "it", "its", "as", "by", "at", "from", "which", "their", "has", "have",
    "can", "also", "such", "into", "than", "then", "not", "no", "so",
}


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) > 1 and w not in _STOPWORDS]


def build_index(chunks: list[dict]) -> tuple[BM25Okapi | None, list[dict]]:
    """
    Builds a BM25 index over `chunks` (one source document's chunks,
    typically). Returns (index, chunks) so callers keep the exact chunk
    objects the index was built from, for mapping hits back later.

    Returns (None, []) for an empty chunk list -- rank_bm25 can't build
    an index over nothing, and there's nothing to search anyway.
    """
    if not chunks:
        return None, []
    tokenized = [_tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized), chunks


def search(index: BM25Okapi | None, chunks: list[dict], query: str, top_k: int) -> list[dict]:
    """
    Returns the top_k chunks by BM25 score against `query`, each with a
    "bm25_score" field added. Chunks with zero score (no keyword overlap
    with the query at all) are excluded -- a zero-overlap "match" isn't
    a real hit, and including it would let BM25 drag in noise that
    RRF fusion would otherwise correctly ignore.
    """
    if index is None or not chunks:
        return []
    scores = index.get_scores(_tokenize(query))
    scored = [
        {**chunk, "bm25_score": float(score)}
        for chunk, score in zip(chunks, scores)
        if score > 0
    ]
    scored.sort(key=lambda c: c["bm25_score"], reverse=True)
    return scored[:top_k]


def fuse_rrf(vector_ranked: list[dict], bm25_ranked: list[dict], k_rrf: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion: combines two independently-RANKED lists of
    chunks into one fused ranking, using only each chunk's POSITION in
    each list (not the raw vector-distance or BM25 scores, which live on
    incomparable scales and can't be combined directly).

        score(chunk) = sum over lists containing chunk of 1 / (k_rrf + rank)

    A chunk ranked highly in both lists scores highest; a chunk that
    only appears in one list still contributes, just less. Chunks are
    matched by their `text` field (a chunk's text is unique within one
    source document, which is the scope this is called at).

    Returns chunks sorted by fused "rrf_score" descending, each carrying
    whichever copy of its metadata was seen first (vector or BM25 --
    both describe the same underlying chunk, so it doesn't matter which).
    """
    scores: dict[str, float] = {}
    chunk_by_text: dict[str, dict] = {}

    for ranked_list in (vector_ranked, bm25_ranked):
        for rank, chunk in enumerate(ranked_list, start=1):
            text = chunk["text"]
            scores[text] = scores.get(text, 0.0) + 1.0 / (k_rrf + rank)
            chunk_by_text.setdefault(text, chunk)

    fused = [{**chunk_by_text[text], "rrf_score": score} for text, score in scores.items()]
    fused.sort(key=lambda c: c["rrf_score"], reverse=True)
    return fused

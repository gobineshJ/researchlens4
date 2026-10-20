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

import re

from sentence_transformers import CrossEncoder

from .config import RERANKER_MODEL_NAME, RERANK_SNIPPET_CHARS, RERANK_STRIP_BOILERPLATE

_reranker = None

# Text that sits at the top of real academic-PDF chunks without being
# content. Chunk text is whitespace-joined (no line breaks survive), so
# these match patterns, not lines.
_BOILERPLATE = [
    # Running headers and footers
    re.compile(r"arXiv:\s?\d{4}\.\d{4,5}(?:v\d+)?(?:\s*\[[\w.\-]+\])?(?:\s+\d{1,2}\s+[A-Z][a-z]{2,8}\.?\s+\d{4})?"),
    re.compile(r"(?:Published|Under review) as a (?:conference|workshop) paper at [A-Z][\w-]*(?: \d{4})?", re.I),
    re.compile(r"\bPreprint\.(?: Under review\.)?"),
    re.compile(r"\bPage \d+ of \d+\b", re.I),
    re.compile(r"(?:©|\(c\)|Copyright)\s?\d{4}[^.]{0,120}\.", re.I),
    re.compile(r"\bVol\.\s?\d+(?:,?\s?No\.\s?\d+)?,?"),
    re.compile(r"\bpp\.\s?\d+\s?[-–]\s?\d+"),
    # Identifiers
    re.compile(r"(?:https?://)?(?:dx\.)?doi\.org/\S+|\bdoi:\s?\S+", re.I),
    re.compile(r"https?://\S+"),
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    # A figure/table caption sentence. Only where a caption can start -- the
    # text's start or right after a sentence/number -- so an in-text "as
    # shown in Figure 3." is left alone.
    re.compile(r"(?:^|(?<=[.!?)\]\d] ))(?:Extended Data |Supplementary )?(?:Figure|Fig\.|Table|Tab\.)"
               r"\s?[A-Z]?\d+[a-z]?\s?[:.|]\s[^.]*\."),
    # Rows of table numbers: four or more numeric tokens in a row
    re.compile(r"(?:[-+±]?\d+(?:[.,]\d+)?%?\s+){3,}[-+±]?\d+(?:[.,]\d+)?%?"),
    # A bare page number left at the very start
    re.compile(r"^\s*\d{1,4}\s+(?=[A-Z])"),
]


def _scoring_snippet(text: str) -> str:
    """
    The part of a chunk the cross-encoder actually scores: its first
    RERANK_SNIPPET_CHARS characters, after page headers, captions,
    identifiers and table-number rows are stripped -- otherwise those are
    what fills the snippet on a real paper, and the score says how well
    "arXiv:2101.00001v2 [cs.CL] 3 Jan 2021 Table 2: ..." matches the query.

    Falls back to the raw text when stripping would leave almost nothing
    (a chunk that genuinely is mostly a table), so every candidate still
    gets a meaningful snippet.
    """
    if not RERANK_STRIP_BOILERPLATE:
        return text[:RERANK_SNIPPET_CHARS]
    cleaned = text
    for pattern in _BOILERPLATE:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) < min(100, len(text) // 4):
        return text[:RERANK_SNIPPET_CHARS]
    return cleaned[:RERANK_SNIPPET_CHARS]


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
    text (after boilerplate is stripped -- see _scoring_snippet), not the
    full chunk -- keeps inputs closer to the short-passage length
    ms-marco-MiniLM-L-6-v2 was actually trained on. The FULL text is still
    what gets returned and later sent to Groq.
    """
    if not candidates:
        return []
    if isinstance(queries, str):
        queries = [queries]

    model = _get_reranker()
    snippets = [_scoring_snippet(c["text"]) for c in candidates]

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

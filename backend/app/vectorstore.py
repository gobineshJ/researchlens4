"""
Wraps ChromaDB with sentence-transformers embeddings. One collection
per app run is enough for this project's scope -- each /process call
creates a fresh session collection so results from different sessions
never mix.
"""

import uuid
import chromadb
from sentence_transformers import SentenceTransformer

from .config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME

_embedder = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def get_client() -> chromadb.Client:
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def new_session_collection(session_id: str | None = None):
    """
    Creates a fresh, uniquely named collection for one processing session.
    Explicitly uses cosine distance (hnsw:space) rather than ChromaDB's
    default squared-L2 -- cosine has a standard, interpretable 0-2 range
    (0 = identical, 1 = unrelated, 2 = opposite) that evaluation.py's
    retrieval-confidence threshold is calibrated against. Leaving this
    unset silently falls back to L2, whose scale doesn't match that
    threshold at all.
    """
    session_id = session_id or uuid.uuid4().hex[:12]
    client = get_client()
    collection_name = f"session_{session_id}"
    # Clear any stale collection with the same name (shouldn't normally happen)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    return client.create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    ), session_id


def add_chunks(collection, chunks: list[dict]) -> None:
    """
    chunks: [{"text": ..., "source_label": ..., "source_type": ..., "source_name": ...}, ...]
    Embeds each chunk's text and stores it with its metadata.
    """
    if not chunks:
        return

    embedder = _get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts).tolist()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"source_label": c["source_label"], "source_type": c["source_type"],
         "source_name": c["source_name"], "position": c.get("position", 0),
         "content_type": c.get("content_type", "text")}
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )



def retrieve_candidates(collection, queries: list[str], k_per_query: int,
                         source_name: str | None = None) -> list[dict]:
    """
    Retrieves k_per_query candidates for EACH query in `queries`
    (optionally restricted to one source_name), then dedupes by chunk id
    across all of them. This is the query-expansion-aware building block:
    call it once per source with several query variants to build an
    oversampled candidate pool, then hand that pool to reranker.rerank()
    to pick the best k for that source.

    With a single query (the original instructions, no expansion) and
    source_name set, this is equivalent to a simple per-document top-k
    lookup, just returning more candidates than needed so reranking has
    something to work with.
    """
    embedder = _get_embedder()
    where = {"source_name": source_name} if source_name else None

    seen_ids: set[str] = set()
    all_hits: list[dict] = []
    for query in queries:
        query_embedding = embedder.encode([query]).tolist()
        kwargs = {"query_embeddings": query_embedding, "n_results": k_per_query}
        if where:
            kwargs["where"] = where
        results = collection.query(**kwargs)
        for hit in _parse_query_results(results):
            if hit["chunk_id"] not in seen_ids:
                seen_ids.add(hit["chunk_id"])
                all_hits.append(hit)
    return all_hits


def _parse_query_results(results: dict) -> list[dict]:
    hits = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        hits.append({
            "chunk_id": chunk_id,
            "text": text,
            "source_label": meta["source_label"],
            "source_type": meta["source_type"],
            "source_name": meta["source_name"],
            "position": meta.get("position"),
            "content_type": meta.get("content_type", "text"),
            "distance": distance,
        })
    return hits

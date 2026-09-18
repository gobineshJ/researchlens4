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
    """Creates a fresh, uniquely named collection for one processing session."""
    session_id = session_id or uuid.uuid4().hex[:12]
    client = get_client()
    collection_name = f"session_{session_id}"
    # Clear any stale collection with the same name (shouldn't normally happen)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    return client.create_collection(collection_name), session_id


def add_chunks(collection, chunks: list[dict]) -> None:
    """
    chunks: [{"text": ..., "source_label": ..., "source_type": ...}, ...]
    Embeds each chunk's text and stores it with its metadata.
    """
    if not chunks:
        return

    embedder = _get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts).tolist()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"source_label": c["source_label"], "source_type": c["source_type"]}
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )


def retrieve_top_k(collection, query: str, k: int) -> list[dict]:
    """
    Returns the k most relevant chunks to `query`:
        [{"text": ..., "source_label": ..., "source_type": ..., "distance": ...}, ...]
    """
    embedder = _get_embedder()
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
    )

    hits = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, distance in zip(documents, metadatas, distances):
        hits.append({
            "text": text,
            "source_label": meta["source_label"],
            "source_type": meta["source_type"],
            "distance": distance,
        })
    return hits

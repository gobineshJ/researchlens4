"""
Splits text records (from doc_parser or scraper) into overlapping
chunks sized in tokens, per the brief's 500-800 token / 10-20% overlap
spec. Uses a simple whitespace-word approximation for "tokens" rather
than a real tokenizer -- close enough for chunk sizing, and keeps this
module dependency-free.

Every input record must have "text" plus enough fields to build a
human-readable citation label -- this module adds a "source_label" and
carries the rest through untouched.
"""

from .config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_RATIO


def _label_for(record: dict) -> str:
    if record["source_type"] == "file":
        return f"{record['filename']}, {record['location']}"
    return f"{record['title']}, {record['url']}"


def _source_name_for(record: dict) -> str:
    """
    Identifies which DOCUMENT a chunk came from, as opposed to source_label
    which identifies a specific page/section within it. Needed so retrieval
    can guarantee every uploaded document gets representation, rather than
    treating all chunks from all documents as one undifferentiated pool.
    """
    if record["source_type"] == "file":
        return record["filename"]
    return record["url"]


def chunk_record(record: dict,
                  chunk_size: int = CHUNK_SIZE_TOKENS,
                  overlap_ratio: float = CHUNK_OVERLAP_RATIO) -> list[dict]:
    """
    Splits one record's text into word-bounded chunks. Returns a list of:
        {"text": ..., "source_label": ..., "source_type": ..., "source_name": ...}
    """
    words = record["text"].split()
    if not words:
        return []

    step = max(1, int(chunk_size * (1 - overlap_ratio)))
    label = _label_for(record)
    source_name = _source_name_for(record)

    chunks = []
    start = 0
    while start < len(words):
        piece = words[start:start + chunk_size]
        chunks.append({
            "text": " ".join(piece),
            "source_label": label,
            "source_type": record["source_type"],
            "source_name": source_name,
        })
        if start + chunk_size >= len(words):
            break
        start += step

    return chunks


def chunk_all(records: list[dict]) -> list[dict]:
    """Chunk every record in a list and flatten the result."""
    all_chunks = []
    for record in records:
        all_chunks.extend(chunk_record(record))
    return all_chunks


def sample_representative(chunks: list[dict], k: int) -> list[dict]:
    """
    Picks k chunks EVENLY SPACED across the list, by position -- not by
    similarity to any query. This is for "summary"-intent requests: a
    generic instruction like "summarize this document" has no specific
    topic for similarity search to match against, so the top-5-most-
    similar chunks would just reflect whatever happens to embed closest
    to the word "summarize" -- not a representative sample of the whole
    document, which is what the user actually wants covered.

    `chunks` is assumed to already be in the document's original reading
    order (chunker.chunk_record produces chunks that way).

    Returns all of `chunks` unchanged if there are k or fewer of them --
    nothing to sample down.
    """
    if len(chunks) <= k:
        return list(chunks)
    if k <= 0:
        return []

    step = len(chunks) / k
    indices = sorted({int(i * step) for i in range(k)})
    if len(indices) < k:
        # Rounding collisions (two computed positions landed on the same
        # index) can leave us short of k -- top up with the next unused
        # indices so the caller reliably gets k chunks back.
        remaining = [i for i in range(len(chunks)) if i not in indices]
        indices = sorted(indices + remaining[: k - len(indices)])
    return [chunks[i] for i in indices]

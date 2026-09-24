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

"""Tests chunk sizing, overlap, and source-label generation."""

from app import chunker


def _make_record(word_count: int, source_type="file") -> dict:
    text = " ".join(f"word{i}" for i in range(word_count))
    if source_type == "file":
        return {"source_type": "file", "filename": "doc.pdf", "location": "page 1", "text": text}
    return {"source_type": "web", "title": "Some Page", "url": "https://example.com", "text": text}


def test_short_text_produces_one_chunk():
    record = _make_record(50)
    chunks = chunker.chunk_record(record)
    assert len(chunks) == 1
    assert chunks[0]["source_label"] == "doc.pdf, page 1"


def test_long_text_produces_multiple_overlapping_chunks():
    record = _make_record(2000)  # well over CHUNK_SIZE_TOKENS (650)
    chunks = chunker.chunk_record(record)
    assert len(chunks) > 1
    # Every chunk except the last should be at (or very near) the configured size
    for c in chunks[:-1]:
        assert len(c["text"].split()) <= chunker.CHUNK_SIZE_TOKENS


def test_consecutive_chunks_actually_overlap():
    """
    Verifies REAL overlap between produced chunks, not just that the
    configured ratio implies a smaller step size (the config only proves
    intent, not that overlap was actually applied to real output).
    """
    record = _make_record(1500)
    chunks = chunker.chunk_record(record)
    assert len(chunks) >= 2
    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()
    shared = set(first_words) & set(second_words)
    assert len(shared) > 0, "consecutive chunks should share some words due to overlap"


def test_empty_text_produces_no_chunks():
    record = _make_record(0)
    assert chunker.chunk_record(record) == []


def test_web_record_label_uses_title_and_url():
    record = _make_record(20, source_type="web")
    chunks = chunker.chunk_record(record)
    assert chunks[0]["source_label"] == "Some Page, https://example.com"
    assert chunks[0]["source_type"] == "web"


def test_chunk_all_flattens_multiple_records():
    records = [_make_record(50), _make_record(50)]
    chunks = chunker.chunk_all(records)
    assert len(chunks) == 2

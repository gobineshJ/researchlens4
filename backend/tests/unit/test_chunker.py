"""Tests chunk sizing, overlap, and source-label generation."""

import pytest

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
    record = _make_record(2000)  # well over CHUNK_SIZE_WORDS (650)
    chunks = chunker.chunk_record(record)
    assert len(chunks) > 1
    # Every chunk except the last should be at (or very near) the configured size
    for c in chunks[:-1]:
        assert len(c["text"].split()) <= chunker.CHUNK_SIZE_WORDS


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


# ---------- sample_representative (positional sampling for "summary" intent) ----------

def _numbered_chunks(n: int) -> list[dict]:
    """Chunks tagged with their own index in `text`, so tests can check
    WHICH ones got picked, not just how many."""
    return [{"text": f"chunk-{i}", "source_label": "x", "source_type": "file"} for i in range(n)]


def test_sample_representative_returns_all_when_fewer_than_k():
    chunks = _numbered_chunks(3)
    result = chunker.sample_representative(chunks, 5)
    assert result == chunks


def test_sample_representative_spans_evenly_across_document():
    chunks = _numbered_chunks(100)
    result = chunker.sample_representative(chunks, 5)
    indices = [int(c["text"].split("-")[1]) for c in result]
    assert indices == [0, 25, 50, 74, 99]


@pytest.mark.parametrize("n,k", [(100, 5), (33, 5), (65, 5), (36, 3), (10, 2), (7, 6)])
def test_sample_representative_always_includes_both_endpoints(n, k):
    """The old int(i * n/k) spacing topped out around 0.8n, so a document's
    conclusions were structurally unreachable and span coverage could never
    exceed ~80%."""
    result = chunker.sample_representative(_numbered_chunks(n), k)
    indices = [int(c["text"].split("-")[1]) for c in result]
    assert indices[0] == 0
    assert indices[-1] == n - 1


def test_sample_representative_returns_exactly_k_chunks():
    chunks = _numbered_chunks(100)
    result = chunker.sample_representative(chunks, 10)
    assert len(result) == 10
    # No duplicates -- every picked chunk should be a distinct position
    assert len({c["text"] for c in result}) == 10


def test_sample_representative_handles_k_zero():
    chunks = _numbered_chunks(10)
    assert chunker.sample_representative(chunks, 0) == []


# ---------- position numbering ----------

def test_chunk_all_numbers_positions_per_source():
    records = [_make_record(2000), _make_record(2000)]  # both "doc.pdf"
    chunks = chunker.chunk_all(records)
    assert [c["position"] for c in chunks] == list(range(len(chunks)))


def test_chunk_all_numbers_positions_independently_per_source():
    """Positions must count within a document, not across the whole run --
    they are what span coverage measures a selection against."""
    file_record = _make_record(2000)
    web_record = _make_record(2000, source_type="web")
    chunks = chunker.chunk_all([file_record, web_record, _make_record(2000)])

    by_source = {}
    for c in chunks:
        by_source.setdefault(c["source_name"], []).append(c["position"])
    assert len(by_source) == 2
    for positions in by_source.values():
        assert positions == list(range(len(positions)))


def test_chunk_all_position_matches_index_in_its_source_list():
    """pipeline builds chunks_by_source by iterating chunk_all's output in
    order, and validate_span_coverage treats position as an index into that
    list -- so the two must agree exactly."""
    chunks = chunker.chunk_all([_make_record(3000), _make_record(3000, source_type="web")])
    by_source = {}
    for c in chunks:
        by_source.setdefault(c["source_name"], []).append(c)
    for source_chunks in by_source.values():
        assert [c["position"] for c in source_chunks] == list(range(len(source_chunks)))


# ---------- helpers ----------

def _sources(counts: dict[str, int], chunk_chars: int = 400) -> dict[str, list[dict]]:
    return {
        name: [{"text": "x" * chunk_chars, "source_name": name, "position": i} for i in range(n)]
        for name, n in counts.items()
    }


# ---------- back matter detection ----------
# Strings below are taken from the three real papers this was tuned on.

@pytest.mark.parametrize("text,kind", [
    ("Articles Nature Medicine Extended Data Fig. 10 | Calibration Plots for the MGB data "
     "and the new independent dataset, CDH, used for model validation.", "figure caption"),
    ("Mathematics 2021, 9, 2970 51 of 52 122. Gupta, S.; Pawar, S.; Ramrakhiyani, N. "
     "Semi-supervised recurrent neural network. 123. Kumar, R.L.; Khan, F. Recurrent model. "
     "124. Tu, J.; Chen, H. Evolutionary biogeography. 125. Lee, S.-W.; Ali, S. Survey.",
     "references"),
    ("Medical Computing and Analysis, Department of Radiology, Massachusetts General Hospital, "
     "Boston, MA, USA. Department of Medicine, University of Cambridge, UK. School of Medicine, "
     "Stanford University, USA. Institute of Health, College of Engineering.", "affiliations"),
    ("Articles Nature Medicine Acknowledgements The views expressed in this study are those "
     "of the authors.", "acknowledgements"),
    ("P.R., E.H., G.L., J.T., K.K., A.Q., A.F., C.C., D.B., I.Y. designed the study.",
     "acknowledgements"),
    ("adap- tation, distribution and reproduction in any medium or format, as long as you give "
     "appropriate credit. Creative Commons license.", "license"),
])
def test_back_matter_kinds(text, kind):
    assert chunker.back_matter_kind(text) == kind


@pytest.mark.parametrize("text", [
    # main text that merely REFERS to an extended data figure
    "(Figs. 3a,b and 4c-h, Extended Data Fig. 6c and Supplementary Fig. 1), the model "
    "performed consistently across sites.",
    # a Methods section naming participating hospitals (5.7 per 100 words, real case)
    "Articles Nature Medicine Methods Ethics approval. All procedures were conducted at "
    "Massachusetts General Hospital and the University of Cambridge under approved protocols. "
    + "The cohort included adult patients admitted with suspected infection. " * 6,
    # discussion running into the first citations: kept as content
    "which limits the generalizability of the results to other settings and populations. "
    + "Future work should validate the model prospectively. " * 8 + "References 1. James, S. L.",
])
def test_main_text_is_not_back_matter(text):
    assert chunker.back_matter_kind(text) is None


def test_content_chunks_keeps_positions_and_order():
    chunks = [{"text": "body text " * 50, "position": 0},
              {"text": "Extended Data Fig. 1 | caption", "position": 1},
              {"text": "more body " * 50, "position": 2}]
    assert [c["position"] for c in chunker.content_chunks(chunks)] == [0, 2]


def test_content_chunks_never_returns_nothing():
    """A document that is all back matter still gets summarized from something."""
    chunks = [{"text": "Extended Data Fig. 1 | caption", "position": 0}]
    assert chunker.content_chunks(chunks) == chunks


# ---------- plan_summary_excerpts ----------

def _excerpt_total(excerpts):
    return sum(chunker.estimate_tokens(c["text"]) for chunks in excerpts.values() for c in chunks)


def test_excerpts_fit_the_budget():
    sources = _sources({"a.pdf": 36, "b.pdf": 33, "c.pdf": 54}, chunk_chars=3800)  # ~950 tokens each
    excerpts, cap = chunker.plan_summary_excerpts(sources, 8, max_tokens=5000, min_excerpt_tokens=250)
    assert _excerpt_total(excerpts) <= 5000


def test_excerpts_trade_depth_for_positions():
    """Whole ~950-token chunks would allow 5 in a 5000-token request -- under
    two per paper. The plan takes many more positions, each shorter."""
    sources = _sources({"a.pdf": 36, "b.pdf": 33, "c.pdf": 54}, chunk_chars=3800)
    excerpts, cap = chunker.plan_summary_excerpts(sources, 8, max_tokens=5000, min_excerpt_tokens=250)
    assert sum(len(v) for v in excerpts.values()) == 5000 // 250
    assert cap >= 250


def test_excerpts_span_each_document_end_to_end():
    sources = _sources({"a.pdf": 36, "b.pdf": 33}, chunk_chars=3800)
    excerpts, _ = chunker.plan_summary_excerpts(sources, 8, max_tokens=5000, min_excerpt_tokens=250)
    for name, chunks in excerpts.items():
        positions = [c["position"] for c in chunks]
        assert positions[0] == 0 and positions[-1] == len(sources[name]) - 1


def test_extra_positions_go_to_the_longest_document():
    sources = _sources({"long.pdf": 60, "short.pdf": 10}, chunk_chars=3800)
    excerpts, _ = chunker.plan_summary_excerpts(sources, 8, max_tokens=3000, min_excerpt_tokens=250)
    assert len(excerpts["long.pdf"]) > len(excerpts["short.pdf"])


def test_excerpts_respect_the_per_document_cap_and_length():
    sources = _sources({"tiny.pdf": 3, "long.pdf": 50}, chunk_chars=400)
    excerpts, _ = chunker.plan_summary_excerpts(sources, 8, max_tokens=100_000, min_excerpt_tokens=250)
    assert len(excerpts["tiny.pdf"]) == 3
    assert len(excerpts["long.pdf"]) == 8


def test_small_documents_are_sent_whole():
    sources = _sources({"a.pdf": 4}, chunk_chars=400)
    excerpts, _ = chunker.plan_summary_excerpts(sources, 8, max_tokens=5000, min_excerpt_tokens=250)
    assert [c["text"] for c in excerpts["a.pdf"]] == [c["text"] for c in sources["a.pdf"]]


def test_short_chunks_stay_whole_and_long_ones_get_the_room_they_leave():
    """Water-filling: a 50-token chunk keeps all 50, not an equal share."""
    short = {"text": "x " * 100, "source_name": "a.pdf", "position": 0}   # ~51 tokens
    long_ = {"text": "y " * 4000, "source_name": "a.pdf", "position": 1}  # ~2001 tokens
    excerpts, cap = chunker.plan_summary_excerpts({"a.pdf": [short, long_]}, 8,
                                                  max_tokens=1000, min_excerpt_tokens=250)
    texts = [c["text"] for c in excerpts["a.pdf"]]
    assert texts[0] == short["text"]
    assert chunker.estimate_tokens(texts[1]) > 900


def test_excerpts_keep_labels_and_positions_for_citations():
    sources = {"a.pdf": [{"text": "word " * 2000, "source_name": "a.pdf",
                          "source_label": f"a.pdf, page {i}", "position": i} for i in range(10)]}
    excerpts, _ = chunker.plan_summary_excerpts(sources, 4, max_tokens=1000, min_excerpt_tokens=250)
    for c in excerpts["a.pdf"]:
        assert c["source_label"] == f"a.pdf, page {c['position']}"
    assert all(len(c["text"]) < len("word " * 2000) for c in excerpts["a.pdf"])


def test_every_document_gets_both_ends_even_when_excerpts_must_be_thin():
    sources = _sources({f"d{i}.pdf": 20 for i in range(15)}, chunk_chars=3800)
    excerpts, cap = chunker.plan_summary_excerpts(sources, 8, max_tokens=5000, min_excerpt_tokens=250)
    assert all(len(v) == 2 for v in excerpts.values())
    assert cap < 250  # reported by the pipeline as a failing budget_fit
    assert _excerpt_total(excerpts) <= 5000


def test_excerpt_ends_at_a_sentence_when_it_can():
    text = "First sentence here. " * 20 + "Trailing words without an end"
    cut = chunker._excerpt(text, 200)
    assert cut.endswith(".") and len(cut) <= 200


def test_excerpt_falls_back_to_a_word_boundary():
    cut = chunker._excerpt("word " * 100, 53)
    assert cut == ("word " * 10).strip()

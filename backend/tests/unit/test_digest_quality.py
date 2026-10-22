"""
The pieces behind better paper summaries: section-anchored sampling,
citations numbered by their own markers, the pipeline options the digest
uses (extra records, forced intent, no PDF), the cross-paper synthesis
call, and PDF markup.
"""

import json

import pymupdf
import pytest

from app import chunker, groq_client, pdf_export
from tests.unit.test_pipeline import _chunks_for, _run


def _doc(texts: list[str]) -> list[dict]:
    return [{"text": t, "source_label": f"p.pdf, page {i}", "source_type": "file", "source_name": "p.pdf",
             "position": i} for i, t in enumerate(texts)]


# --- section anchors ---

def test_section_anchors_find_numbered_and_unnumbered_headings():
    chunks = _doc([
        "Title Abstract We propose X. 1 Introduction Retrieval is hard.",
        "2 Related Work Prior results show that retrieval helps.",       # prose "results": not a heading
        "3 Method We embed. 4 Experiments We evaluate on NQ.",
        "Table 2 shows gains. 5 Results and Discussion Our model wins.",
        "6 Conclusion We presented our method. Limitations Our study uses one dataset.",
    ])
    anchors = chunker.section_anchors(chunks)
    assert anchors["results"][0] == 2
    assert chunks[2]["text"][anchors["results"][1]:].startswith("4 Experiments")
    assert anchors["conclusion"][0] == 4
    assert anchors["limitations"][0] == 4
    assert chunks[4]["text"][anchors["limitations"][1]:].startswith("Limitations Our")


@pytest.mark.parametrize("text", ["our results show gains", "limitations of this work are", "Table 3: Results of",
                                  "In the Evaluation section we"])
def test_prose_is_not_a_heading(text):
    assert chunker.section_anchors(_doc([text])) == {}


def test_sample_for_summary_anchors_sections_and_starts_at_the_heading():
    texts = [f"Body text number {i} goes here." for i in range(20)]
    texts[13] = "earlier paragraph ends. 5 Results Our method gains 4 points."
    texts[17] = "wrap up. Limitations We only test English."
    chunks = _doc(texts)
    picked = chunker.sample_for_summary(chunks, 5)
    assert [c["position"] for c in picked] == sorted(c["position"] for c in picked)
    positions = [c["position"] for c in picked]
    assert {0, 13, 17, 19} <= set(positions) and len(picked) == 5
    by_position = {c["position"]: c for c in picked}
    assert by_position[13]["text"] == "5 Results Our method gains 4 points."
    assert chunks[13]["text"].startswith("earlier")  # the original is untouched


def test_sample_for_summary_without_headings_equals_even_spacing():
    chunks = _doc([f"Plain text {i}." for i in range(30)])
    for k in (1, 2, 5, 8):
        assert chunker.sample_for_summary(chunks, k) == chunker.sample_representative(chunks, k)


def test_sample_for_summary_respects_k_when_anchors_outnumber_it():
    texts = [f"Body {i}." for i in range(10)]
    texts[4], texts[6], texts[8] = "5 Results We win.", "Limitations Few.", "7 Conclusion Done."
    picked = chunker.sample_for_summary(_doc(texts), 3)
    # Endpoints first, then results (highest priority).
    assert [c["position"] for c in picked] == [0, 4, 9]


def test_sample_for_summary_can_be_turned_off(monkeypatch):
    texts = [f"Body {i}." for i in range(10)]
    texts[4] = "5 Results We win."
    monkeypatch.setattr(chunker, "SUMMARY_SECTION_AWARE", False)
    chunks = _doc(texts)
    assert chunker.sample_for_summary(chunks, 3) == chunker.sample_representative(chunks, 3)


# --- numbered citations ---

def test_numbered_citations_keep_marker_numbers():
    chunks = [{"source_label": f"src{i}"} for i in range(1, 7)]
    # [3] never cited, [5] cited before [4]: printing 1, 2, 3... would misnumber both.
    report = "A [1]. B [2]. C [5]. D [4]. E [6]. F [1, 2]."
    assert groq_client.numbered_citations(report, chunks) == [
        {"n": 1, "label": "src1"}, {"n": 2, "label": "src2"}, {"n": 4, "label": "src4"},
        {"n": 5, "label": "src5"}, {"n": 6, "label": "src6"},
    ]
    assert groq_client.numbered_citations("No markers.", chunks) == []


# --- pipeline options ---

def test_pipeline_returns_numbered_citations():
    chunks = _chunks_for("doc.pdf", 6, "neural networks")
    result = _run(chunks, intent="summary", generate_report={"report_text": "X [2]. Y [1].",
                                                              "citations_used": []})
    assert [c["n"] for c in result["citations"]] == [1, 2]


def test_forced_intent_skips_classification():
    chunks = _chunks_for("doc.pdf", 6, "neural networks")

    def must_not_classify(*a, **k):
        raise AssertionError("intent was given")
    result = _run(chunks, intent=must_not_classify, run_kwargs={"intent": "summary"},
                  classify_intent=None)
    check = next(c for c in result["evaluation"]["checks"] if c["name"] == "intent")
    assert "Set by the caller to 'summary'" in check["detail"]


def test_extra_records_come_before_parsed_files():
    chunks = _chunks_for("doc.pdf", 6, "neural networks")
    seen_records = []

    def capture(records):
        seen_records.extend(records)
        return chunks

    abstract = {"source_type": "file", "filename": "doc.pdf", "location": "arXiv abstract", "text": "We show X."}
    _run(chunks, intent="summary", chunk_all=capture, doc_parser_parse_document=[
        {"source_type": "file", "filename": "doc.pdf", "location": "page 1", "text": "Page one."}],
        run_kwargs={"extra_records": [abstract]})
    assert [r["location"] for r in seen_records] == ["arXiv abstract"]  # parse_document isn't called with no files


def test_export_pdf_false_builds_no_pdf():
    chunks = _chunks_for("doc.pdf", 6, "neural networks")
    result = _run(chunks, intent="summary", run_kwargs={"export_pdf": False})
    assert result["pdf_path"] is None and result["pdf_filename"] is None
    assert result["report_text"]


# --- synthesis ---

def test_synthesize_digest_parses_and_strips_paper_citations(monkeypatch):
    sent = {}

    def fake_call(model, system_prompt, user_prompt, **kwargs):
        sent["user"] = user_prompt
        return json.dumps({"themes": "Both do RAG [P1] [P2].", "read_in_full": 2, "why": "Has numbers."})

    monkeypatch.setattr(groq_client, "_call_groq", fake_call)
    out = groq_client.synthesize_digest("RAG", [{"title": "A", "summary": "TL;DR: a [1]. More [2, 3]."},
                                                {"title": "B", "summary": "TL;DR: b [4]."}])
    assert out == {"themes": "Both do RAG [P1] [P2].", "read_in_full": 2, "why": "Has numbers."}
    assert "[P1] A" in sent["user"] and "[P2] B" in sent["user"]
    assert "[1]" not in sent["user"] and "[2, 3]" not in sent["user"]


@pytest.mark.parametrize("answer", [{"themes": "T", "read_in_full": 9, "why": "w"},
                                    {"themes": "T", "read_in_full": "2", "why": "w"},
                                    {"themes": "T"}])
def test_synthesize_digest_ignores_invalid_pick(monkeypatch, answer):
    monkeypatch.setattr(groq_client, "_call_groq", lambda *a, **k: json.dumps(answer))
    out = groq_client.synthesize_digest("RAG", [{"title": "A", "summary": "a"}, {"title": "B", "summary": "b"}])
    assert out == {"themes": "T", "read_in_full": None, "why": ""}


def test_synthesize_digest_raises_without_themes(monkeypatch):
    monkeypatch.setattr(groq_client, "_call_groq", lambda *a, **k: json.dumps({"themes": ""}))
    with pytest.raises(ValueError):
        groq_client.synthesize_digest("RAG", [{"title": "A", "summary": "a"}])


# --- PDF ---

def test_markup_renders_markdown_and_escapes():
    assert pdf_export._markup("**Bold** and *it* with a<b & c") == \
        "<b>Bold</b> and <i>it</i> with a&lt;b &amp; c"
    assert pdf_export._markup("2 * 3 * 4") == "2 * 3 * 4"
    assert pdf_export._markup("line one\nline two") == "line one<br/>line two"


def test_sanitize_replaces_every_unrenderable_character():
    out = pdf_export._sanitize_for_pdf("x ≈ y → z, α ≤ 1, café, 中")
    assert all(ord(ch) <= 0xFF for ch in out)
    assert out == "x ~ y -> z, ? <= 1, café, ?"


def _pdf_text(path) -> str:
    with pymupdf.open(path) as doc:
        return "".join(page.get_text() for page in doc)


def test_build_pdf_prints_sources_with_their_own_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_export, "OUTPUT_DIR", str(tmp_path))
    path = pdf_export.build_pdf("**Finding** A [1]. B [4].",
                                [{"n": 1, "label": "a.pdf, page 1"}, {"n": 4, "label": "a.pdf, page 9"}], "r.pdf")
    text = _pdf_text(path)
    assert "[4] a.pdf, page 9" in text and "[2] a.pdf, page 9" not in text
    assert "**" not in text


def test_build_digest_pdf(tmp_path):
    path = pdf_export.build_digest_pdf(
        str(tmp_path / "d.pdf"), "rag -- 2026-10-21", "2 new paper(s)",
        overview=[("Themes", "Both study retrieval [P1] [P2]."), ("At a glance", "**[P1] A**\nTL;DR a")],
        papers=[{"heading": "[P1] A", "meta": "X. Author", "text": "**TL;DR:** a [1].\n\nProblem: p [2].",
                 "citations": [{"n": 1, "label": "A.pdf, arXiv abstract"}, {"n": 2, "label": "A.pdf, page 3"}]}],
    )
    text = _pdf_text(path)
    for expected in ("rag -- 2026-10-21", "Themes", "[P1] A", "TL;DR:", "[2] A.pdf, page 3"):
        assert expected in text

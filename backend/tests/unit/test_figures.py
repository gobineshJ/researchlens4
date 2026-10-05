"""
Tests figure captioning: doc_parser finding figures in a real generated
PDF (one vector chart, one raster image, two near-misses that must NOT
count), turning them into citable records, and degrading to caption-only
records when the vision call fails. The vision model itself is mocked --
these test detection and wiring, not the model.
"""

import json

import pymupdf

from app import chunker, doc_parser, evaluation, groq_client


def _fake_describer(monkeypatch, result="A bar chart of accuracy per dataset."):
    """Replaces the vision call; returns the list of (png, caption) it was called with."""
    calls = []

    def fake(png, caption):
        calls.append((png, caption))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(groq_client, "describe_figure", fake)
    return calls


def _figures(records):
    return [r for r in records if r.get("content_type", "").startswith("figure")]


# ---------- Detection ----------

def test_finds_vector_and_raster_figures_and_skips_near_misses(figure_pdf, monkeypatch):
    calls = _fake_describer(monkeypatch)
    records = doc_parser.parse_document(figure_pdf)

    # "Figure 1 shows ..." in body text is not a caption, and "Figure 3." has
    # nothing graphical above it -- only the two real figures come back.
    assert [r["location"] for r in _figures(records)] == ["page 1, Figure 1", "page 2, Figure 2"]
    assert [caption for _, caption in calls] == [
        "Figure 1: Accuracy of the model across five datasets.",
        "Fig. 2 | Example input image from the dataset.",
    ]


def test_figure_is_rendered_from_its_region_not_the_whole_page(figure_pdf, monkeypatch):
    calls = _fake_describer(monkeypatch)
    doc_parser.parse_document(figure_pdf)

    png, _ = calls[0]
    assert png.startswith(b"\x89PNG")
    rendered = pymupdf.Pixmap(png)
    # The chart spans ~320 x 90 points; at 150 dpi a full page would be
    # ~1275 x 1650 pixels.
    assert rendered.width < 900 and rendered.height < 400


def test_figure_region_includes_axis_labels_but_not_the_paragraph(figure_pdf):
    doc = pymupdf.open(figure_pdf)
    page = doc[0]
    blocks = [(pymupdf.Rect(b[:4]), " ".join(b[4].split())) for b in page.get_text("blocks") if b[6] == 0]
    caption = next(rect for rect, text in blocks if text.startswith("Figure 1:"))
    region = doc_parser._figure_region(page, caption, blocks, doc_parser._graphic_rects(page))
    doc.close()

    assert region is not None
    assert region.x0 < 65          # the "Acc" y-axis title at x=60
    assert region.y1 > 345         # the D1..D5 tick labels under the axis
    assert region.y0 > 140         # but not the paragraph above the chart
    assert region.y1 <= caption.y0 + 2  # nor the caption itself


# ---------- Records ----------

def test_figure_records_follow_their_own_page(figure_pdf, monkeypatch):
    _fake_describer(monkeypatch)
    locations = [r["location"] for r in doc_parser.parse_document(figure_pdf)]
    assert locations == ["page 1", "page 1, Figure 1", "page 2", "page 2, Figure 2"]


def test_described_figure_record_has_caption_description_and_metadata(figure_pdf, monkeypatch):
    _fake_describer(monkeypatch)
    figure = _figures(doc_parser.parse_document(figure_pdf))[0]

    assert figure["source_type"] == "file"
    assert figure["filename"] == "figures.pdf"
    assert figure["content_type"] == "figure_description"
    assert "Caption: Figure 1: Accuracy of the model across five datasets." in figure["text"]
    assert "A bar chart of accuracy per dataset." in figure["text"]


def test_failed_description_keeps_caption_only_record(figure_pdf, monkeypatch):
    _fake_describer(monkeypatch, result=RuntimeError("429 rate limited"))
    figures = _figures(doc_parser.parse_document(figure_pdf))

    assert [f["content_type"] for f in figures] == ["figure_caption", "figure_caption"]
    assert figures[0]["text"] == (
        "Figure 1 (page 1). Caption: Figure 1: Accuracy of the model across five datasets."
    )


def test_missing_api_key_stops_calling_groq_for_the_rest_of_the_document(figure_pdf, monkeypatch):
    calls = _fake_describer(monkeypatch, result=groq_client.GroqNotConfigured("no key"))
    figures = _figures(doc_parser.parse_document(figure_pdf))

    assert len(calls) == 1
    assert [f["content_type"] for f in figures] == ["figure_caption", "figure_caption"]


def test_per_document_request_limit(figure_pdf, monkeypatch):
    monkeypatch.setattr(doc_parser, "FIGURE_MAX_PER_DOCUMENT", 1)
    calls = _fake_describer(monkeypatch)
    figures = _figures(doc_parser.parse_document(figure_pdf))

    assert len(calls) == 1
    assert [f["content_type"] for f in figures] == ["figure_description", "figure_caption"]


def test_captioning_disabled_gives_original_text_only_records(figure_pdf, monkeypatch):
    monkeypatch.setattr(doc_parser, "FIGURE_CAPTIONING_ENABLED", False)
    calls = _fake_describer(monkeypatch)
    records = doc_parser.parse_document(figure_pdf)

    assert calls == []
    assert [r["location"] for r in records] == ["page 1", "page 2"]


def test_pdf_without_figures_makes_no_vision_calls(sample_pdf, monkeypatch):
    calls = _fake_describer(monkeypatch)
    doc_parser.parse_document(sample_pdf)
    assert calls == []


# ---------- Cache ----------

def test_descriptions_are_cached_between_runs(figure_pdf, monkeypatch):
    calls = _fake_describer(monkeypatch)
    first = doc_parser.parse_document(figure_pdf)
    second = doc_parser.parse_document(figure_pdf)

    assert len(calls) == 2  # two figures, described once each
    assert _figures(first) == _figures(second)


def test_failures_are_not_cached(figure_pdf, monkeypatch):
    _fake_describer(monkeypatch, result=RuntimeError("timeout"))
    doc_parser.parse_document(figure_pdf)
    calls = _fake_describer(monkeypatch)
    doc_parser.parse_document(figure_pdf)
    assert len(calls) == 2


def test_corrupt_cache_file_is_ignored(figure_pdf, monkeypatch, tmp_path):
    (tmp_path / "figure_cache.json").write_text("{not json", encoding="utf-8")
    calls = _fake_describer(monkeypatch)
    doc_parser.parse_document(figure_pdf)

    assert len(calls) == 2
    assert len(json.loads((tmp_path / "figure_cache.json").read_text(encoding="utf-8"))) == 2


# ---------- Downstream ----------

def test_chunker_labels_figures_and_carries_content_type(figure_pdf, monkeypatch):
    _fake_describer(monkeypatch)
    chunks = chunker.chunk_all(doc_parser.parse_document(figure_pdf))

    by_label = {c["source_label"]: c for c in chunks}
    assert by_label["figures.pdf, page 1, Figure 1"]["content_type"] == "figure_description"
    assert by_label["figures.pdf, page 1"]["content_type"] == "text"
    assert [c["position"] for c in chunks] == [0, 1, 2, 3]


def test_extraction_check_reports_undescribed_figures(figure_pdf, monkeypatch):
    monkeypatch.setattr(doc_parser, "FIGURE_MAX_PER_DOCUMENT", 1)
    _fake_describer(monkeypatch)
    report = evaluation.EvaluationReport()
    evaluation.validate_extraction(doc_parser.parse_document(figure_pdf), report)

    check = next(c for c in report.checks if c["name"] == "figure_descriptions")
    assert check["passed"] is False
    assert check["value"] == 50.0
    assert "1 of 2 figure(s)" in check["detail"]


def test_extraction_check_absent_without_figures(sample_pdf):
    report = evaluation.EvaluationReport()
    evaluation.validate_extraction(doc_parser.parse_document(sample_pdf), report)
    assert all(c["name"] != "figure_descriptions" for c in report.checks)


def test_generation_check_flags_citations_of_generated_descriptions():
    retrieved = [
        {"text": "Accuracy rose to 91%.", "source_label": "p.pdf, page 3", "content_type": "text"},
        {"text": "A bar chart ...", "source_label": "p.pdf, page 7, Figure 3",
         "content_type": "figure_description"},
    ]
    report = evaluation.EvaluationReport()
    evaluation.validate_generation("Accuracy rose [1] per the chart [2].",
                                   ["p.pdf, page 3", "p.pdf, page 7, Figure 3"], retrieved, report)

    check = next(c for c in report.checks if c["name"] == "figure_citations")
    assert check["value"] == 1
    assert "p.pdf, page 7, Figure 3" in check["detail"]


# ---------- Vision request ----------

def test_describe_figure_sends_image_and_caption(monkeypatch):
    sent = {}

    def fake_post(payload, timeout=30):
        sent["payload"], sent["timeout"] = payload, timeout
        return "  A line plot.  "

    monkeypatch.setattr(groq_client, "_post_chat", fake_post)
    assert groq_client.describe_figure(b"\x89PNGfake", "Figure 2: Loss curves.") == "A line plot."

    payload = sent["payload"]
    assert payload["model"] == groq_client.GROQ_VISION_MODEL
    text_part, image_part = payload["messages"][0]["content"]
    assert "Figure 2: Loss curves." in text_part["text"]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert sent["timeout"] == 60

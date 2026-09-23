"""
Tests doc_parser against real generated PDF/DOCX/TXT files, checking
both the extracted text AND the metadata (filename/location) that
citations depend on.
"""

from app import doc_parser


def test_pdf_extracts_one_record_per_page(sample_pdf):
    records = doc_parser.parse_document(sample_pdf)
    assert len(records) == 2
    assert records[0]["location"] == "page 1"
    assert records[1]["location"] == "page 2"
    assert "machine learning" in records[0]["text"]
    assert "privacy" in records[1]["text"]


def test_pdf_record_has_required_metadata(sample_pdf):
    records = doc_parser.parse_document(sample_pdf)
    for r in records:
        assert r["source_type"] == "file"
        assert r["filename"] == "sample.pdf"
        assert r["location"]
        assert r["text"].strip()


def test_docx_splits_by_heading(sample_docx):
    records = doc_parser.parse_document(sample_docx)
    assert len(records) == 2
    locations = [r["location"] for r in records]
    assert "Introduction" in locations
    assert "Methodology" in locations


def test_txt_returns_single_record(sample_txt):
    records = doc_parser.parse_document(sample_txt)
    assert len(records) == 1
    assert records[0]["location"] == "full text"
    assert "neural networks" in records[0]["text"]


def test_empty_pdf_returns_no_records(empty_pdf):
    """
    Scenario 2 from the earlier test-case list: a scanned/image-only PDF
    with no extractable text should return an empty list, not crash and
    not fabricate content.
    """
    records = doc_parser.parse_document(empty_pdf)
    assert records == []


def test_unsupported_extension_returns_empty(tmp_path):
    fake = tmp_path / "data.xlsx"
    fake.write_text("not real xlsx content")
    assert doc_parser.parse_document(str(fake)) == []

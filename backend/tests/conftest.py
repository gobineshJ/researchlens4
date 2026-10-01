"""
Shared fixtures. Tests build their own sample files rather than relying
on files you happen to have lying around -- that's what makes this
suite runnable on any machine, including CI, with no setup.
"""

import sys
from pathlib import Path

import pytest
import pymupdf
import docx

# Make `from app...` imports work when pytest is run from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _isolated_run_history(tmp_path, monkeypatch):
    """
    Every run_pipeline call appends to the run history. Without this, every
    test run -- unit tests included -- wrote into the REAL
    evaluation_log.jsonl that the UI's "Run history" panel reads, burying
    real runs under hundreds of "Discuss the topic" entries.
    """
    from app import evaluation
    monkeypatch.setattr(evaluation, "EVALUATION_LOG_PATH", str(tmp_path / "evaluation_log.jsonl"))


@pytest.fixture
def sample_pdf(tmp_path) -> str:
    """A 2-page PDF with distinct, known text on each page."""
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Page one discusses machine learning in healthcare.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Page two discusses data privacy concerns in medical AI.")
    path = tmp_path / "sample.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def sample_docx(tmp_path) -> str:
    """A DOCX with two headed sections, for testing section-based splitting."""
    d = docx.Document()
    d.add_heading("Introduction", level=1)
    d.add_paragraph("This introduces the topic of machine learning in medicine.")
    d.add_heading("Methodology", level=1)
    d.add_paragraph("This describes the research methodology used in the study.")
    path = tmp_path / "sample.docx"
    d.save(str(path))
    return str(path)


@pytest.fixture
def sample_txt(tmp_path) -> str:
    path = tmp_path / "sample.txt"
    path.write_text("A short plain-text research note about neural networks.")
    return str(path)


@pytest.fixture
def empty_pdf(tmp_path) -> str:
    """A PDF with a page but no text at all -- simulates a scanned/image-only PDF."""
    doc = pymupdf.open()
    doc.new_page()  # blank page, no insert_text call
    path = tmp_path / "empty.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)

"""
Parses uploaded research documents into a common structure:

    {
        "source_type": "file",
        "filename": "paper.pdf",
        "location": "page 3",       # page / section / sheet, whatever applies
        "text": "...",
    }

PDF, DOCX, TXT are covered. XLSX/CSV are supplementary per the brief
and stubbed for now.
"""

from pathlib import Path
import pymupdf as fitz
import docx


def parse_pdf(file_path: str) -> list[dict]:
    """One record per page, so citations can point to a specific page."""
    records = []
    doc = fitz.open(file_path)
    filename = Path(file_path).name
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if not text:
            continue
        records.append({
            "source_type": "file",
            "filename": filename,
            "location": f"page {page_number}",
            "text": text,
        })
    doc.close()
    return records


def parse_docx(file_path: str) -> list[dict]:
    """Group by heading ('section') since DOCX has no fixed page numbers."""
    filename = Path(file_path).name
    document = docx.Document(file_path)

    records = []
    current_section = "Document start"
    current_text = []

    def flush():
        joined = "\n".join(current_text).strip()
        if joined:
            records.append({
                "source_type": "file",
                "filename": filename,
                "location": current_section,
                "text": joined,
            })

    for para in document.paragraphs:
        style = (para.style.name or "").lower()
        if style.startswith("heading"):
            flush()
            current_section = para.text.strip() or current_section
            current_text = []
        else:
            if para.text.strip():
                current_text.append(para.text)

    flush()
    return records


def parse_txt(file_path: str) -> list[dict]:
    filename = Path(file_path).name
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [{
        "source_type": "file",
        "filename": filename,
        "location": "full text",
        "text": text,
    }]


def parse_document(file_path: str) -> list[dict]:
    """Dispatch by extension. Returns [] for unsupported types."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    if suffix == ".docx":
        return parse_docx(file_path)
    if suffix == ".txt":
        return parse_txt(file_path)
    return []

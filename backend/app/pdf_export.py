"""
Turns the Groq-generated report into a PDF using ReportLab. Keeps
formatting simple (title, body paragraphs, a citations list at the
end) -- the brief asks for citation-aware output, not fancy layout.
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm

from .config import OUTPUT_DIR

# ReportLab's built-in fonts (Helvetica, Times-Roman) only cover the
# Latin-1 character range. Academic PDFs (this one included) often use
# Unicode punctuation just outside that range -- typographic hyphens
# (U+2010/2011/2012), en/em dashes, and curly quotes -- which have no
# glyph in those fonts and render as a black "missing character" box.
# Normalizing to plain ASCII equivalents before layout avoids that.
_UNICODE_TO_ASCII = {
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "--",  # em dash
    "\u2018": "'", "\u2019": "'",  # curly single quotes
    "\u201c": '"', "\u201d": '"',  # curly double quotes
    "\u2026": "...",  # ellipsis
}


def _sanitize_for_pdf(text: str) -> str:
    for unicode_char, ascii_equivalent in _UNICODE_TO_ASCII.items():
        text = text.replace(unicode_char, ascii_equivalent)
    return text


def build_pdf(report_text: str, citations_used: list[str], filename: str) -> str:
    """
    Writes a PDF to OUTPUT_DIR/filename and returns the full path.
    report_text: the generated report body (may contain [1], [2] markers)
    citations_used: list of human-readable source labels, in citation order
    """
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(OUTPUT_DIR) / filename)

    report_text = _sanitize_for_pdf(report_text)
    citations_used = [_sanitize_for_pdf(label) for label in citations_used]

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("ResearchLens Report", styles["Title"]))
    story.append(Spacer(1, 0.5 * cm))

    for paragraph in report_text.split("\n\n"):
        cleaned = paragraph.strip()
        if cleaned:
            story.append(Paragraph(cleaned, styles["BodyText"]))
            story.append(Spacer(1, 0.3 * cm))

    if citations_used:
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("Sources cited", styles["Heading2"]))
        for i, label in enumerate(citations_used, start=1):
            story.append(Paragraph(f"[{i}] {label}", styles["BodyText"]))

    doc.build(story)
    return output_path

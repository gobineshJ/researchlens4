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


def build_pdf(report_text: str, citations_used: list[str], filename: str) -> str:
    """
    Writes a PDF to OUTPUT_DIR/filename and returns the full path.
    report_text: the generated report body (may contain [1], [2] markers)
    citations_used: list of human-readable source labels, in citation order
    """
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(OUTPUT_DIR) / filename)

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

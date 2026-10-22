"""
Turns the Groq-generated report into a PDF using ReportLab. Keeps
formatting simple (title, body paragraphs, a citations list at the
end) -- the brief asks for citation-aware output, not fancy layout.

build_digest_pdf lays out the daily digest the same way: an overview,
then one section per paper with its own numbered sources.
"""

import html
import re
import unicodedata
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
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    "‒": "-",  # figure dash
    "–": "-",  # en dash
    "—": "--",  # em dash
    "‘": "'", "’": "'",  # curly single quotes
    "“": '"', "”": '"',  # curly double quotes
    "…": "...",  # ellipsis
    # Common in model-written summaries of ML papers:
    "−": "-",  # minus sign
    "≈": "~", "∼": "~",  # approximately, tilde operator
    "→": "->", "←": "<-", "↔": "<->",  # arrows
    "≤": "<=", "≥": ">=", "≠": "!=",
    "•": "-",  # bullet
    " ": " ", " ": " ", " ": " ", " ": " ",  # narrow / thin spaces
    "​": "",  # zero-width space
}


def _sanitize_for_pdf(text: str) -> str:
    for unicode_char, ascii_equivalent in _UNICODE_TO_ASCII.items():
        text = text.replace(unicode_char, ascii_equivalent)
    # Anything else outside Latin-1 would still be a black box: use its
    # ASCII decomposition when it has one (e.g. accented letters beyond
    # Latin-1), otherwise a visible "?" rather than a silent box.
    if any(ord(ch) > 0xFF for ch in text):
        text = "".join(
            ch if ord(ch) <= 0xFF
            else (unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode() or "?")
            for ch in text
        )
    return text


_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")
_HEADING = re.compile(r"^#{1,6}\s+(.*)$")


def _markup(text: str) -> str:
    """Sanitized text as ReportLab paragraph markup: &, <, > escaped (a raw
    "<" in a report otherwise breaks the layout), **bold** and *italic* from
    the model's Markdown rendered instead of printed as asterisks, line
    breaks kept."""
    text = html.escape(_sanitize_for_pdf(text), quote=False)
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _ITALIC.sub(r"<i>\1</i>", text)
    return text.replace("\n", "<br/>")


def _body_paragraphs(text: str, styles) -> list:
    story = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        heading = _HEADING.match(block)
        if heading and "\n" not in block:
            story.append(Paragraph(_markup(heading.group(1)), styles["Heading4"]))
        else:
            story.append(Paragraph(_markup(block), styles["BodyText"]))
            story.append(Spacer(1, 0.25 * cm))
    return story


def _sources(citations: list[dict], styles, heading: str = "Sources cited") -> list:
    """citations: [{"n": 4, "label": ...}] -- printed with their own numbers,
    so "[4]" in the text is "[4]" here (see groq_client.numbered_citations)."""
    if not citations:
        return []
    story = [Spacer(1, 0.2 * cm), Paragraph(heading, styles["Heading4"])]
    for c in citations:
        story.append(Paragraph(_markup(f"[{c['n']}] {c['label']}"), styles["BodyText"]))
    return story


def _document(output_path: str) -> SimpleDocTemplate:
    return SimpleDocTemplate(output_path, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm,
                             leftMargin=2 * cm, rightMargin=2 * cm)


def build_pdf(report_text: str, citations: list[dict], filename: str) -> str:
    """
    Writes a PDF to OUTPUT_DIR/filename and returns the full path.
    report_text: the generated report body (may contain [1], [2] markers)
    citations: [{"n": ..., "label": ...}] from groq_client.numbered_citations
    """
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(OUTPUT_DIR) / filename)
    styles = getSampleStyleSheet()
    story = [Paragraph("ResearchLens Report", styles["Title"]), Spacer(1, 0.5 * cm)]
    story += _body_paragraphs(report_text, styles)
    story += _sources(citations, styles)
    _document(output_path).build(story)
    return output_path


def build_digest_pdf(output_path: str, title: str, subtitle: str,
                     overview: list[tuple[str, str]], papers: list[dict]) -> str:
    """
    The daily digest as one PDF.

      overview: [(heading, text), ...] -- e.g. "Worth reading in full",
                "Themes", "At a glance"; text may use **bold** and newlines
      papers:   [{"heading", "meta", "text", "citations"}, ...] -- one
                section each, with that paper's own numbered sources
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = [Paragraph(_markup(title), styles["Title"]),
             Paragraph(_markup(subtitle), styles["Italic"]), Spacer(1, 0.4 * cm)]
    for heading, text in overview:
        story.append(Paragraph(_markup(heading), styles["Heading2"]))
        story += _body_paragraphs(text, styles)
    for paper in papers:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(_markup(paper["heading"]), styles["Heading2"]))
        if paper.get("meta"):
            story.append(Paragraph(_markup(paper["meta"]), styles["Italic"]))
            story.append(Spacer(1, 0.2 * cm))
        story += _body_paragraphs(paper["text"], styles)
        story += _sources(paper.get("citations", []), styles, heading="Sources")
    _document(output_path).build(story)
    return output_path

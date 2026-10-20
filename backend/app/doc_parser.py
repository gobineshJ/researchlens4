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

PDF figures additionally become records of their own, located as
"page 7, Figure 3", whose text is the paper's caption plus a description
of the rendered figure from a vision model. To the rest of the pipeline
a figure is then just another text record. These carry one extra field:

    "content_type": "figure_description"   # caption + model-generated description
                    "figure_caption"       # caption only (figure couldn't be described)

Every other record is plain text and has no "content_type".
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
import pymupdf as fitz
import docx

from . import groq_client, progress
from .config import (
    FIGURE_CACHE_PATH, FIGURE_CAPTIONING_ENABLED, FIGURE_MAX_PER_DOCUMENT,
    FIGURE_RENDER_DPI, GROQ_VISION_MODEL,
)

logger = logging.getLogger(__name__)


def parse_pdf(file_path: str) -> list[dict]:
    """
    One record per page, so citations can point to a specific page, with
    each page's figure records directly after it. Keeping figures in
    reading order matters: chunker.chunk_all numbers chunks by record
    order, and span coverage / summary sampling read those positions.
    """
    records = []
    doc = fitz.open(file_path)
    filename = Path(file_path).name
    describer = _FigureDescriber() if FIGURE_CAPTIONING_ENABLED else None
    seen_figures: set[str] = set()
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
        if describer:
            records.extend(_figure_records(page, filename, page_number, seen_figures, describer))
    doc.close()
    if describer:
        describer.save_cache()
    return records


# ---------- Figures ----------
#
# Figures are found from their CAPTIONS, not their images: a caption gives
# the figure its number (which is what makes "page 7, Figure 3" a useful
# citation) and tells us where to look, while images alone are mostly
# logos, journal banners and icons. The figure is then RENDERED rather than
# extracted, because plots exported from matplotlib or TikZ are vector
# drawings with no embedded image for page.get_images() to return.

_FIGURE_CAPTION = re.compile(r"^(?:Fig\.?|Figure)\s*(\d+)\s*[:.|–—-]", re.IGNORECASE)
# "Figure 3: ...", "Fig. 3 | ...", "FIGURE 3. ..." -- the punctuation after
# the number is what separates a caption from body text that merely starts
# a sentence with a reference ("Figure 3 shows ...").
# "Extended Data Fig. 3 |" does not match on purpose: supplementary figures
# are back matter (see chunker.back_matter_kind).

_BODY_TEXT_MIN_WORDS = 25
# A text block this long above a caption is a paragraph, which marks where
# the figure region stops. Shorter blocks are axis labels, legends and panel
# letters, and belong to the figure.
_MIN_GRAPHIC_SIDE = 10      # points; thinner drawings are rules and underlines
_MIN_FIGURE_SIDE = 40       # points; a smaller region is an icon, not a figure
_MAX_FIGURE_HEIGHT_RATIO = 0.75
_LABEL_MARGIN = 30          # points around the graphics to look for labels in --
                            # y-axis titles routinely sit 25+ points off the axis
_PADDING = 4                # points of whitespace kept around the render


def _x_overlap(a: fitz.Rect, b: fitz.Rect) -> bool:
    return min(a.x1, b.x1) > max(a.x0, b.x0)


def _graphic_rects(page) -> list[fitz.Rect]:
    """Bounding boxes of everything drawn on the page that isn't text:
    raster images, plus vector drawings grouped into clusters."""
    rects = [fitz.Rect(info["bbox"]) for info in page.get_image_info()]
    try:
        rects += [fitz.Rect(r) for r in page.cluster_drawings()]
    except Exception:
        pass  # malformed drawing commands -- raster images alone still work
    clipped = [r & page.rect for r in rects]
    return [r for r in clipped if min(r.width, r.height) >= _MIN_GRAPHIC_SIDE]


def _figure_region(page, caption: fitz.Rect, text_blocks: list[tuple], graphics: list[fitz.Rect]):
    """
    The area of a figure whose caption sits at `caption`: the graphics
    directly above the caption, in the same column, below the nearest
    paragraph or earlier caption -- plus the short text blocks around them
    (axis labels, legends). Returns None when nothing graphical is there,
    which usually means the "caption" was a false match.
    """
    top = max(page.rect.y0, caption.y0 - page.rect.height * _MAX_FIGURE_HEIGHT_RATIO)
    for rect, text in text_blocks:
        if rect.y1 > caption.y0 + 1 or not _x_overlap(rect, caption):
            continue
        if len(text.split()) >= _BODY_TEXT_MIN_WORDS or _FIGURE_CAPTION.match(text):
            top = max(top, rect.y1)

    in_band = [g for g in graphics
               if g.y0 >= top - 2 and g.y1 <= caption.y0 + 2 and _x_overlap(g, caption)]
    if not in_band:
        return None

    region = fitz.Rect(in_band[0])
    for g in in_band[1:]:
        region |= g
    search = fitz.Rect(region.x0 - _LABEL_MARGIN, region.y0 - _LABEL_MARGIN,
                       region.x1 + _LABEL_MARGIN, region.y1 + _LABEL_MARGIN)
    for rect, text in text_blocks:
        if (rect.intersects(search) and rect.y0 >= top - 2 and rect.y1 <= caption.y0 + 2
                and len(text.split()) < _BODY_TEXT_MIN_WORDS):
            region |= rect

    region = fitz.Rect(region.x0 - _PADDING, region.y0 - _PADDING,
                       region.x1 + _PADDING, region.y1 + _PADDING) & page.rect
    if min(region.width, region.height) < _MIN_FIGURE_SIDE:
        return None
    return region


def _figure_records(page, filename: str, page_number: int, seen_figures: set[str],
                    describer: "_FigureDescriber") -> list[dict]:
    """
    One record per captioned figure on this page. A figure that can't be
    rendered is skipped; one that can't be DESCRIBED keeps its caption, so
    "page 7, Figure 3" is still retrievable and citable. Nothing here is
    allowed to fail the page's text extraction.
    """
    try:
        text_blocks = [(fitz.Rect(b[:4]), " ".join(b[4].split()))
                       for b in page.get_text("blocks") if b[6] == 0]
        captions = [(rect, text, match.group(1)) for rect, text in text_blocks
                    if (match := _FIGURE_CAPTION.match(text))]
        if not captions:
            return []
        graphics = _graphic_rects(page)
    except Exception:
        return []

    records = []
    for caption_rect, caption_text, number in captions:
        if number in seen_figures:
            continue  # first caption wins; a repeat is a text reference that happened to match
        try:
            region = _figure_region(page, caption_rect, text_blocks, graphics)
            if region is None:
                continue
            png = page.get_pixmap(clip=region, dpi=FIGURE_RENDER_DPI).tobytes("png")
        except Exception:
            continue
        seen_figures.add(number)

        progress.detail(f"Describing Figure {number} of {filename} (page {page_number})")
        description = describer.describe(png, caption_text)
        heading = f"Figure {number} (page {page_number}). Caption: {caption_text}"
        records.append({
            "source_type": "file",
            "filename": filename,
            "location": f"page {page_number}, Figure {number}",
            "text": (f"{heading}\nDescription (generated from the figure image): {description}"
                     if description else heading),
            "content_type": "figure_description" if description else "figure_caption",
        })
    return records


class _FigureDescriber:
    """
    Wraps groq_client.describe_figure for one document: a disk cache keyed
    by image hash, a cap of FIGURE_MAX_PER_DOCUMENT live requests, and
    graceful failure -- describe() returns None instead of raising, and
    stops calling Groq for the rest of the document once it learns there is
    no API key.
    """

    def __init__(self, cache_path: str | None = None):
        self.cache_path = Path(cache_path or FIGURE_CACHE_PATH)
        self.requests_made = 0
        self.available = True
        self._dirty = False
        try:
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.cache = {}

    @staticmethod
    def _key(png: bytes) -> str:
        # The model and prompt are part of the key: a description is only
        # reusable if it came from the same model asked the same question.
        prefix = f"{GROQ_VISION_MODEL}|{groq_client.FIGURE_PROMPT_VERSION}|".encode()
        return hashlib.sha256(prefix + png).hexdigest()

    def describe(self, png: bytes, caption: str) -> str | None:
        key = self._key(png)
        if key in self.cache:
            return self.cache[key]
        if not self.available or self.requests_made >= FIGURE_MAX_PER_DOCUMENT:
            return None
        self.requests_made += 1
        try:
            description = groq_client.describe_figure(png, caption)
        except groq_client.GroqNotConfigured:
            self.available = False
            return None
        except Exception as exc:
            # rate limit, timeout, bad response -- caption-only for this one
            logger.warning("Figure not described (%s): %s", caption[:60], str(exc)[:300])
            return None
        if not description:
            return None
        self.cache[key] = description
        self._dirty = True
        return description

    def save_cache(self) -> None:
        """
        Several worker processes can each be describing figures at once, so
        this merges with whatever is on disk NOW (another process may have
        saved since this one loaded) and writes via a temp file + rename,
        so a concurrent reader never sees a half-written file.
        """
        if not self._dirty:
            return
        try:
            try:
                on_disk = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                on_disk = {}
            merged = {**on_disk, **self.cache}
            tmp_path = self.cache_path.with_name(f"{self.cache_path.name}.{os.getpid()}.tmp")
            tmp_path.write_text(json.dumps(merged), encoding="utf-8")
            os.replace(tmp_path, self.cache_path)
        except OSError:
            pass  # a cache that can't be written only costs a re-description next run


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

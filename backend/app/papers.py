"""
arXiv as a paper source, for scheduled digests (an n8n workflow, cron, an
MCP tool). Two jobs, matching how a digest runs:

1. search_arxiv()        -- the API's Atom feed -> plain dicts: id, title,
                            abstract, authors, categories, dates, links.
                            Metadata only, cheap, called by the API.
2. download_arxiv_pdfs() -- fetches the chosen papers' PDFs into a job's
                            upload dir, called by the Celery worker, so the
                            pipeline then reads them like any uploaded file.

Which papers a digest has already covered is not this module's business --
db.seen_papers tracks that, and main.py filters with it.

arXiv's API terms ask for at most one request every three seconds; every
request here (searches and PDFs alike) goes through _get, which keeps that
pace within a process. https://info.arxiv.org/help/api/tou.html
"""

import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from . import progress
from .config import (
    ARXIV_API_URL,
    ARXIV_MAX_PDF_BYTES,
    ARXIV_PDF_URL,
    ARXIV_REQUEST_INTERVAL_SECONDS,
    ARXIV_TIMEOUT_SECONDS,
    USER_AGENT,
)

logger = logging.getLogger(__name__)

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

# New-style ids (2410.01234, 2410.01234v2) and pre-2007 ones (hep-th/9901001,
# math.GT/0309136v1). Anything else is refused before it reaches a URL.
_ARXIV_ID = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")
_VERSION_SUFFIX = re.compile(r"v\d+$")
# Category names as arXiv writes them: cs.CL, stat.ML, hep-th, q-bio.NC ...
_CATEGORY = re.compile(r"^[a-z][a-z\-]*(?:\.[A-Za-z\-]+)?$")

_request_lock = threading.Lock()
_last_request_at = 0.0


class ArxivError(Exception):
    """arXiv was unreachable, or answered with an error."""


def is_valid_arxiv_id(arxiv_id: str) -> bool:
    return bool(_ARXIV_ID.match(arxiv_id))


def base_id(arxiv_id: str) -> str:
    """'2410.01234v2' -> '2410.01234'. Seen-tracking uses this, so a new
    version of a paper already covered doesn't count as a new paper."""
    return _VERSION_SUFFIX.sub("", arxiv_id)


def _get(url: str, **kwargs) -> requests.Response:
    """requests.get, no sooner than ARXIV_REQUEST_INTERVAL_SECONDS after the
    previous arXiv request from this process."""
    global _last_request_at
    with _request_lock:
        wait = ARXIV_REQUEST_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        try:
            return requests.get(url, headers={"User-Agent": USER_AGENT},
                                timeout=ARXIV_TIMEOUT_SECONDS, **kwargs)
        finally:
            _last_request_at = time.monotonic()


# --- search ---

def build_query(keywords: list[str], categories: list[str], days: int | None,
                now: datetime | None = None) -> str:
    """
    arXiv search_query syntax from friendlier parts:
      keywords   -> each one a phrase searched in all fields, OR'd together
      categories -> cat:X, OR'd together
      days       -> submittedDate range ending now
    The three groups are AND'ed. At least keywords or categories is needed --
    arXiv has no "everything from the last N days" query.
    """
    groups = []
    phrases = [" ".join(k.replace('"', " ").split()) for k in keywords]
    phrases = [p for p in phrases if p]
    if phrases:
        groups.append(" OR ".join(f'all:"{p}"' for p in phrases))
    cats = [c.strip() for c in categories if c.strip()]
    for cat in cats:
        if not _CATEGORY.match(cat):
            raise ValueError(f"Not an arXiv category: {cat!r} (expected something like 'cs.CL').")
    if cats:
        groups.append(" OR ".join(f"cat:{c}" for c in cats))
    if not groups:
        raise ValueError("Give at least one keyword or category.")
    if days:
        now = now or datetime.now(timezone.utc)
        start = now - timedelta(days=days)
        groups.append(f"submittedDate:[{start:%Y%m%d%H%M} TO {now:%Y%m%d%H%M}]")
    if len(groups) == 1:
        return groups[0]
    return " AND ".join(f"({g})" for g in groups)


def _text(element: ET.Element | None) -> str:
    return " ".join(element.text.split()) if element is not None and element.text else ""


def parse_feed(xml_text: str) -> tuple[int, list[dict]]:
    """arXiv's Atom response -> (total results matching, papers on this page)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ArxivError(f"arXiv returned a response that isn't valid XML ({exc}).")

    total = int(_text(root.find("opensearch:totalResults", _NS)) or 0)
    papers = []
    for entry in root.findall("atom:entry", _NS):
        entry_id = _text(entry.find("atom:id", _NS))
        # A malformed query comes back as HTTP 200 with a single entry
        # describing the error, not as an HTTP error.
        if "/api/errors" in entry_id:
            raise ArxivError(f"arXiv rejected the query: {_text(entry.find('atom:summary', _NS))}")

        arxiv_id = entry_id.rsplit("/abs/", 1)[-1]
        links = {link.get("title") or link.get("rel"): link.get("href")
                 for link in entry.findall("atom:link", _NS)}
        primary = entry.find("arxiv:primary_category", _NS)
        papers.append({
            "arxiv_id": arxiv_id,
            "id": base_id(arxiv_id),
            "title": _text(entry.find("atom:title", _NS)),
            "abstract": _text(entry.find("atom:summary", _NS)),
            "authors": [_text(a.find("atom:name", _NS)) for a in entry.findall("atom:author", _NS)],
            "primary_category": primary.get("term") if primary is not None else "",
            "categories": [c.get("term") for c in entry.findall("atom:category", _NS)],
            "published": _text(entry.find("atom:published", _NS)),
            "updated": _text(entry.find("atom:updated", _NS)),
            "abs_url": links.get("alternate", f"https://arxiv.org/abs/{arxiv_id}"),
            "pdf_url": links.get("pdf", f"https://arxiv.org/pdf/{arxiv_id}"),
        })
    return total, papers


def search_arxiv(query: str, max_results: int = 25) -> tuple[int, list[dict]]:
    """Newest submissions first. Returns (total matching, papers)."""
    try:
        response = _get(ARXIV_API_URL, params={
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
    except requests.RequestException as exc:
        raise ArxivError(f"Could not reach arXiv ({type(exc).__name__}).")
    if response.status_code != 200:
        raise ArxivError(f"arXiv answered HTTP {response.status_code}.")
    return parse_feed(response.text)


# --- PDFs ---

def _pdf_filename(arxiv_id: str) -> str:
    # Old-style ids contain a slash; the filename becomes the source's name
    # in the report, so keep it recognisable rather than hashing it.
    return f"arXiv {arxiv_id.replace('/', '_')}.pdf"


def download_arxiv_pdf(arxiv_id: str, dest_dir: Path) -> Path:
    if not is_valid_arxiv_id(arxiv_id):
        raise ValueError(f"Not an arXiv id: {arxiv_id!r}")
    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    try:
        response = _get(url, stream=True)
    except requests.RequestException as exc:
        raise ArxivError(f"Could not download {arxiv_id} ({type(exc).__name__}).")
    with response:
        if response.status_code != 200:
            raise ArxivError(f"Could not download {arxiv_id} (HTTP {response.status_code}).")
        if "pdf" not in response.headers.get("Content-Type", ""):
            # Withdrawn papers, and the odd one with no PDF, answer with HTML.
            raise ArxivError(f"arXiv has no PDF for {arxiv_id}.")
        dest = dest_dir / _pdf_filename(arxiv_id)
        size = 0
        with dest.open("wb") as f:
            for block in response.iter_content(chunk_size=64 * 1024):
                size += len(block)
                if size > ARXIV_MAX_PDF_BYTES:
                    f.close()
                    dest.unlink(missing_ok=True)
                    raise ArxivError(f"{arxiv_id}'s PDF is over {ARXIV_MAX_PDF_BYTES // (1024 * 1024)} MB.")
                f.write(block)
    return dest


def download_arxiv_pdfs(arxiv_ids: list[str], dest_dir: str) -> tuple[list[str], list[str], list[str]]:
    """
    Downloads each paper's PDF into dest_dir. One paper failing doesn't stop
    the rest. Returns (file paths, ids downloaded, "id: reason" for each one
    that failed); raises ArxivError only if none could be downloaded.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    progress.stage("downloading")
    paths, downloaded, failures = [], [], []
    for i, arxiv_id in enumerate(arxiv_ids, start=1):
        progress.detail(f"Downloading arXiv {arxiv_id} ({i}/{len(arxiv_ids)})")
        try:
            paths.append(str(download_arxiv_pdf(arxiv_id, dest).resolve()))
            downloaded.append(arxiv_id)
        except (ArxivError, ValueError) as exc:
            logger.warning("Skipping arXiv %s: %s", arxiv_id, exc)
            failures.append(f"{arxiv_id}: {exc}")
    if not paths:
        raise ArxivError("None of the papers could be downloaded -- " + "; ".join(failures))
    return paths, downloaded, failures

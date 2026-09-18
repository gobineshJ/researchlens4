"""
Two jobs, matching the brief's two-step design:

1. discover()   -- cheap crawl: title + snippet only, used to build the
                    candidate list the Groq filter judges.
2. scrape_page() -- full extraction of ONE page's main text, called only
                    after the filter has said a page is worth reading.

Never scrape everything found by discover() -- only what the filter selects.
"""

from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

from .config import (
    CRAWL_MAX_DEPTH,
    CRAWL_MAX_PAGES,
    CRAWL_SAME_DOMAIN_ONLY,
    REQUEST_TIMEOUT,
    USER_AGENT,
)


def _same_domain(url_a: str, url_b: str) -> bool:
    return urlparse(url_a).netloc == urlparse(url_b).netloc


def _fetch(url: str) -> requests.Response | None:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 200 and "text/html" in response.headers.get("Content-Type", ""):
            return response
    except requests.RequestException:
        pass
    return None


def _extract_title_and_snippet(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    first_paragraph = soup.find("p")
    snippet = first_paragraph.get_text(strip=True)[:200] if first_paragraph else ""
    return title, snippet


def _extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for tag in soup.find_all("a", href=True):
        absolute = urljoin(base_url, tag["href"])
        if absolute.startswith("http"):
            links.append(absolute)
    return links


def discover(seed_url: str,
             max_depth: int = CRAWL_MAX_DEPTH,
             max_pages: int = CRAWL_MAX_PAGES,
             same_domain_only: bool = CRAWL_SAME_DOMAIN_ONLY) -> list[dict]:
    """
    Bounded breadth-first crawl from one seed URL. Cheap: title + snippet
    only, no full-page extraction. Returns deduped candidates:
        [{"url": ..., "title": ..., "snippet": ..., "depth": ...}, ...]
    """
    visited: set[str] = set()
    candidates: list[dict] = []
    queue: list[tuple[str, int]] = [(seed_url, 0)]

    while queue and len(candidates) < max_pages:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        response = _fetch(url)
        if response is None:
            continue

        title, snippet = _extract_title_and_snippet(response.text)
        candidates.append({
            "url": url,
            "title": title,
            "snippet": snippet,
            "depth": depth,
        })

        if depth < max_depth:
            for link in _extract_links(response.text, url):
                if link in visited:
                    continue
                if same_domain_only and not _same_domain(link, seed_url):
                    continue
                queue.append((link, depth + 1))

    return candidates


def scrape_page(url: str, title: str = "") -> dict | None:
    """
    Full extraction of one page's main readable text, run only on pages
    that passed the Groq relevance filter. Strips script/style/nav tags
    before pulling text, since those aren't real content.

    Returns a record shaped like doc_parser's output, so downstream
    chunking code doesn't need to know if a chunk came from a file or
    the web:
        {"source_type": "web", "url": ..., "title": ..., "text": ...}
    """
    response = _fetch(url)
    if response is None:
        return None

    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ").split())
    if not text:
        return None

    page_title = title or (soup.title.get_text(strip=True) if soup.title else url)
    return {
        "source_type": "web",
        "url": url,
        "title": page_title,
        "text": text,
    }

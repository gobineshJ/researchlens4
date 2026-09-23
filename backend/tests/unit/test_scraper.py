"""
Tests the pure HTML-parsing pieces of scraper.py with canned HTML strings --
no live network requests, so these run anywhere instantly.
"""

from app import scraper


def test_extract_title_and_snippet():
    html = """<html><head><title>Test Page</title></head>
    <body><p>This is the first paragraph on the page.</p></body></html>"""
    title, snippet = scraper._extract_title_and_snippet(html)
    assert title == "Test Page"
    assert "first paragraph" in snippet


def test_extract_title_missing():
    html = "<html><body><p>No title here.</p></body></html>"
    title, snippet = scraper._extract_title_and_snippet(html)
    assert title == ""
    assert "No title" in snippet


def test_extract_links_resolves_relative_urls():
    html = """<html><body>
    <a href="/about">About</a>
    <a href="https://external.com/page">External</a>
    <a href="mailto:test@example.com">Email</a>
    </body></html>"""
    links = scraper._extract_links(html, "https://example.com/start")
    assert "https://example.com/about" in links
    assert "https://external.com/page" in links
    # mailto: links aren't http(s), shouldn't be included
    assert not any("mailto" in link for link in links)


def test_same_domain_check():
    assert scraper._same_domain("https://example.com/a", "https://example.com/b")
    assert not scraper._same_domain("https://example.com/a", "https://other.com/b")


def test_scrape_page_strips_script_and_nav_tags(monkeypatch):
    """
    scrape_page should strip script/nav/footer content -- verified here
    by mocking the HTTP fetch so no live network call is needed.
    """
    html = """<html><head><title>Article</title></head><body>
    <nav>Navigation links here</nav>
    <script>console.log('should not appear')</script>
    <p>This is the real article content.</p>
    <footer>Footer text here</footer>
    </body></html>"""

    class FakeResponse:
        text = html

    monkeypatch.setattr(scraper, "_fetch", lambda url: FakeResponse())
    result = scraper.scrape_page("https://example.com/article")

    assert result is not None
    assert "real article content" in result["text"]
    assert "Navigation links" not in result["text"]
    assert "should not appear" not in result["text"]
    assert "Footer text" not in result["text"]
    assert result["source_type"] == "web"


def test_scrape_page_returns_none_on_fetch_failure(monkeypatch):
    monkeypatch.setattr(scraper, "_fetch", lambda url: None)
    assert scraper.scrape_page("https://example.com/unreachable") is None

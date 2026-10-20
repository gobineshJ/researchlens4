"""
The crawler's robots.txt handling: every fetch (discovery and full
scrape) is checked against the site's rules first, with RFC 9309's
conventions for a missing or unreachable robots.txt. requests.get is
faked, so no network is needed.
"""

import pytest
import requests

from app import scraper

ROBOTS = """
User-agent: *
Disallow: /private/
"""


class _Response:
    def __init__(self, status_code=200, text="", content_type="text/html"):
        self.status_code = status_code
        self.text = text
        self.headers = {"Content-Type": content_type}


@pytest.fixture
def fake_web(monkeypatch):
    """Serves robots.txt from `robots` (a response or an exception) and a
    small HTML page for everything else; records every URL requested."""
    state = {"robots": _Response(200, ROBOTS, "text/plain"), "requested": []}

    def fake_get(url, headers=None, timeout=None):
        state["requested"].append(url)
        assert headers["User-Agent"] == scraper.USER_AGENT
        if url.endswith("/robots.txt"):
            if isinstance(state["robots"], Exception):
                raise state["robots"]
            return state["robots"]
        return _Response(200, "<html><head><title>T</title></head><body><p>Hi</p></body></html>")

    monkeypatch.setattr(scraper.requests, "get", fake_get)
    monkeypatch.setattr(scraper.time, "sleep", lambda s: state.setdefault("sleeps", []).append(s))
    return state


def test_disallowed_path_is_never_fetched(fake_web):
    assert scraper._fetch("https://example.com/private/page") is None
    assert "https://example.com/private/page" not in fake_web["requested"]


def test_rules_for_our_own_user_agent_replace_the_wildcard_group(fake_web):
    # RFC 9309: a group naming our agent is the ONLY group that applies to us.
    fake_web["robots"] = _Response(200, ROBOTS + "\nUser-agent: ResearchLens-Bot\nDisallow: /no-bots/\n",
                                   "text/plain")
    assert not scraper.robots_allows("https://example.com/no-bots/x")
    assert scraper.robots_allows("https://example.com/private/x")


def test_allowed_path_is_fetched(fake_web):
    assert scraper._fetch("https://example.com/public") is not None


def test_robots_txt_is_fetched_once_per_site(fake_web):
    scraper._fetch("https://example.com/a")
    scraper._fetch("https://example.com/b")
    assert fake_web["requested"].count("https://example.com/robots.txt") == 1


@pytest.mark.parametrize("robots,allowed", [
    (_Response(404), True),                       # no robots.txt -> no rules
    (_Response(403), False),                      # access denied -> stay out
    (_Response(503), False),                      # server error -> stay out
    (requests.ConnectionError("down"), False),    # unreachable -> stay out
])
def test_missing_or_unreachable_robots_txt(fake_web, robots, allowed):
    fake_web["robots"] = robots
    assert scraper.robots_allows("https://example.com/page") is allowed


def test_discover_skips_disallowed_links(fake_web, monkeypatch):
    pages = {
        "https://example.com/": '<a href="/private/secret">x</a><a href="/open">y</a>',
        "https://example.com/open": "<p>open page</p>",
    }

    def fake_get(url, headers=None, timeout=None):
        fake_web["requested"].append(url)
        if url.endswith("/robots.txt"):
            return _Response(200, ROBOTS, "text/plain")
        return _Response(200, f"<html><body>{pages.get(url, '')}</body></html>")

    monkeypatch.setattr(scraper.requests, "get", fake_get)
    found = [c["url"] for c in scraper.discover("https://example.com/", max_depth=1)]
    assert found == ["https://example.com/", "https://example.com/open"]
    assert "https://example.com/private/secret" not in fake_web["requested"]


def test_crawl_delay_is_waited_out_between_requests(fake_web, monkeypatch):
    fake_web["robots"] = _Response(200, "User-agent: *\nCrawl-delay: 3\n", "text/plain")
    clock = [100.0]
    monkeypatch.setattr(scraper.time, "monotonic", lambda: clock[0])
    scraper._fetch("https://example.com/a")
    clock[0] += 1
    scraper._fetch("https://example.com/b")
    assert fake_web["sleeps"] == [pytest.approx(2.0)]


def test_crawl_delay_beyond_the_cap_means_skip(fake_web):
    fake_web["robots"] = _Response(
        200, f"User-agent: *\nCrawl-delay: {scraper.ROBOTS_MAX_CRAWL_DELAY_SECONDS + 50}\n", "text/plain"
    )
    assert scraper._fetch("https://example.com/a") is None


def test_can_be_turned_off(fake_web, monkeypatch):
    monkeypatch.setattr(scraper, "CRAWL_RESPECT_ROBOTS_TXT", False)
    assert scraper._fetch("https://example.com/private/page") is not None
    assert "https://example.com/robots.txt" not in fake_web["requested"]

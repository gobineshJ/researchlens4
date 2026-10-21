"""
arXiv digests: building search queries, parsing arXiv's Atom feed, the
/papers routes, PDF downloads, and the worker marking papers seen only
once their report succeeded. Nothing here talks to the real arXiv.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config, db, papers
from app.auth import create_api_key, hash_password
from app.main import app

client = TestClient(app)


def _feed(*entries: str, total: int | None = None) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>{len(entries) if total is None else total}</opensearch:totalResults>
  {''.join(entries)}
</feed>"""


def _entry(arxiv_id="2410.01234v2", title="A  Paper\n  About RAG", category="cs.CL") -> str:
    return f"""<entry>
    <id>http://arxiv.org/abs/{arxiv_id}</id>
    <published>2026-10-20T17:59:59Z</published>
    <updated>2026-10-20T17:59:59Z</updated>
    <title>{title}</title>
    <summary>  We study retrieval.
      It helps.  </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <link href="http://arxiv.org/abs/{arxiv_id}" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/{arxiv_id}" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="{category}" scheme="http://arxiv.org/schemas/atom"/>
    <category term="{category}" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
  </entry>"""


@pytest.fixture
def key_headers():
    db.create_user("alice", hash_password("correct-horse"))
    return {"X-API-Key": create_api_key("alice", "n8n").api_key}


@pytest.fixture(autouse=True)
def _no_arxiv_pacing(monkeypatch):
    monkeypatch.setattr(papers, "ARXIV_REQUEST_INTERVAL_SECONDS", 0)


# --- ids ---

@pytest.mark.parametrize("arxiv_id", ["2410.01234", "2410.01234v3", "0704.0001", "hep-th/9901001",
                                      "math.GT/0309136v1"])
def test_valid_ids(arxiv_id):
    assert papers.is_valid_arxiv_id(arxiv_id)


@pytest.mark.parametrize("arxiv_id", ["", "2410", "../../etc/passwd", "2410.01234/../x",
                                      "https://evil.example/2410.01234", "2410.01234v", "abc"])
def test_invalid_ids(arxiv_id):
    assert not papers.is_valid_arxiv_id(arxiv_id)


def test_base_id_drops_version():
    assert papers.base_id("2410.01234v2") == "2410.01234"
    assert papers.base_id("hep-th/9901001v1") == "hep-th/9901001"
    assert papers.base_id("2410.01234") == "2410.01234"


# --- query building ---

def test_query_combines_keywords_categories_and_dates():
    now = datetime(2026, 10, 21, 6, 30, tzinfo=timezone.utc)
    query = papers.build_query(["retrieval augmented generation", "RAG"], ["cs.CL", "cs.IR"], 2, now=now)
    assert query == (
        '(all:"retrieval augmented generation" OR all:"RAG") AND (cat:cs.CL OR cat:cs.IR) '
        "AND (submittedDate:[202610190630 TO 202610210630])"
    )


def test_query_with_one_group_has_no_parentheses():
    assert papers.build_query([], ["cs.CL"], None) == "cat:cs.CL"


def test_query_strips_quotes_and_blank_keywords():
    assert papers.build_query(['  "large  language" models ', "  "], [], None) == 'all:"large language models"'


def test_query_needs_keywords_or_categories():
    with pytest.raises(ValueError):
        papers.build_query(["  "], [], 3)


def test_query_rejects_malformed_category():
    with pytest.raises(ValueError):
        papers.build_query([], ["cs.CL) OR (all:x"], None)


# --- feed parsing ---

def test_parse_feed():
    total, found = papers.parse_feed(_feed(_entry(), total=42))
    assert total == 42
    assert found == [{
        "arxiv_id": "2410.01234v2",
        "id": "2410.01234",
        "title": "A Paper About RAG",
        "abstract": "We study retrieval. It helps.",
        "authors": ["Ada Lovelace", "Alan Turing"],
        "primary_category": "cs.CL",
        "categories": ["cs.CL", "cs.AI"],
        "published": "2026-10-20T17:59:59Z",
        "updated": "2026-10-20T17:59:59Z",
        "abs_url": "http://arxiv.org/abs/2410.01234v2",
        "pdf_url": "http://arxiv.org/pdf/2410.01234v2",
    }]


def test_parse_feed_raises_on_arxiv_error_entry():
    error = """<entry><id>http://arxiv.org/api/errors#incorrect_id_format_for_x</id>
               <title>Error</title><summary>incorrect id format for x</summary></entry>"""
    with pytest.raises(papers.ArxivError, match="incorrect id format"):
        papers.parse_feed(_feed(error))


def test_parse_feed_raises_on_non_xml():
    with pytest.raises(papers.ArxivError):
        papers.parse_feed("<html>Service unavailable")


# --- /papers/arxiv/search ---

def test_search_requires_auth():
    assert client.post("/papers/arxiv/search", json={"categories": ["cs.CL"]}).status_code == 401


def test_search_returns_papers_and_ids(monkeypatch, key_headers):
    queries = []

    def fake_search(query, max_results):
        queries.append((query, max_results))
        return 2, papers.parse_feed(_feed(_entry("2410.00001v1"), _entry("2410.00002v1")))[1]

    monkeypatch.setattr(papers, "search_arxiv", fake_search)
    response = client.post("/papers/arxiv/search", headers=key_headers,
                           json={"categories": ["cs.CL"], "days": None, "max_results": 10})
    assert response.status_code == 200
    body = response.json()
    assert queries == [("cat:cs.CL", 10)]
    assert body["count"] == 2 and body["skipped_seen"] == 0
    assert body["arxiv_ids"] == ["2410.00001v1", "2410.00002v1"]


def test_search_skips_papers_the_feed_has_seen(monkeypatch, key_headers):
    monkeypatch.setattr(papers, "search_arxiv", lambda query, max_results: (
        2, papers.parse_feed(_feed(_entry("2410.00001v2"), _entry("2410.00002v1")))[1]))
    # Seen as v1 -- a new version of a covered paper still counts as covered.
    db.mark_papers_seen("alice", "daily-nlp", "arxiv", ["2410.00001"])

    body = client.post("/papers/arxiv/search", headers=key_headers,
                       json={"categories": ["cs.CL"], "feed": "daily-nlp"}).json()
    assert body["skipped_seen"] == 1
    assert [p["id"] for p in body["papers"]] == ["2410.00002"]

    # Other feeds have their own history...
    other = client.post("/papers/arxiv/search", headers=key_headers,
                        json={"categories": ["cs.CL"], "feed": "weekly"}).json()
    assert other["count"] == 2
    # ...and only_new=false ignores it.
    everything = client.post("/papers/arxiv/search", headers=key_headers,
                             json={"categories": ["cs.CL"], "feed": "daily-nlp", "only_new": False}).json()
    assert everything["count"] == 2


def test_search_caps_suggested_ids_at_the_per_run_limit(monkeypatch, key_headers):
    entries = [_entry(f"2410.{i:05d}v1") for i in range(config.ARXIV_MAX_PAPERS_PER_RUN + 3)]
    monkeypatch.setattr(papers, "search_arxiv", lambda query, max_results: (
        len(entries), papers.parse_feed(_feed(*entries))[1]))
    body = client.post("/papers/arxiv/search", headers=key_headers, json={"keywords": ["rag"]}).json()
    assert body["count"] == len(entries)
    assert len(body["arxiv_ids"]) == config.ARXIV_MAX_PAPERS_PER_RUN


def test_search_validation_errors(key_headers):
    assert client.post("/papers/arxiv/search", headers=key_headers, json={}).status_code == 422
    assert client.post("/papers/arxiv/search", headers=key_headers,
                       json={"categories": ["bad cat"]}).status_code == 422
    assert client.post("/papers/arxiv/search", headers=key_headers,
                       json={"categories": ["cs.CL"], "feed": "no spaces"}).status_code == 422
    assert client.post("/papers/arxiv/search", headers=key_headers,
                       json={"categories": ["cs.CL"], "max_results": 1000}).status_code == 422


def test_search_reports_arxiv_failure_as_502(monkeypatch, key_headers):
    def down(query, max_results):
        raise papers.ArxivError("Could not reach arXiv (ConnectionError).")
    monkeypatch.setattr(papers, "search_arxiv", down)
    response = client.post("/papers/arxiv/search", headers=key_headers, json={"categories": ["cs.CL"]})
    assert response.status_code == 502
    assert "Could not reach arXiv" in response.json()["detail"]


# --- /papers/arxiv/process ---

@pytest.fixture
def captured_jobs(monkeypatch):
    from app import main as main_module
    calls = []
    monkeypatch.setattr(main_module.run_pipeline_task, "apply_async",
                        lambda args, task_id, kwargs=None:
                        calls.append({"args": args, "kwargs": kwargs, "task_id": task_id}))
    return calls


def test_process_enqueues_ids_and_feed(key_headers, captured_jobs):
    response = client.post("/papers/arxiv/process", headers=key_headers,
                           json={"arxiv_ids": ["2410.00001v1", " 2410.00002 ", "2410.00001v1"],
                                 "feed": "daily-nlp"})
    assert response.status_code == 202
    body = response.json()
    assert body["arxiv_ids"] == ["2410.00001v1", "2410.00002"]

    [call] = captured_jobs
    job_dir, file_paths, seed_urls, instructions, username = call["args"]
    assert (file_paths, seed_urls, username) == ([], [], "alice")
    assert instructions == config.ARXIV_DEFAULT_INSTRUCTIONS
    assert Path(job_dir).is_dir()
    assert call["kwargs"] == {"arxiv_ids": ["2410.00001v1", "2410.00002"], "feed": "daily-nlp"}
    assert db.get_job_owner(body["job_id"]) == "alice"


def test_process_keeps_custom_instructions(key_headers, captured_jobs):
    client.post("/papers/arxiv/process", headers=key_headers,
                json={"arxiv_ids": ["2410.00001"], "instructions": "Compare their evaluation setups."})
    assert captured_jobs[0]["args"][3] == "Compare their evaluation setups."


@pytest.mark.parametrize("ids", [[], ["not-an-id"], ["../../etc/passwd"], ["   "]])
def test_process_rejects_bad_ids(key_headers, captured_jobs, ids):
    response = client.post("/papers/arxiv/process", headers=key_headers, json={"arxiv_ids": ids})
    assert response.status_code == 422
    assert captured_jobs == []


def test_process_rejects_too_many_papers(key_headers, captured_jobs):
    ids = [f"2410.{i:05d}" for i in range(config.ARXIV_MAX_PAPERS_PER_RUN + 1)]
    assert client.post("/papers/arxiv/process", headers=key_headers,
                       json={"arxiv_ids": ids}).status_code == 422


def test_process_cleans_up_when_queue_is_down(monkeypatch, key_headers):
    from app import main as main_module

    def broken(args, task_id, kwargs=None):
        raise ConnectionError("redis down")
    monkeypatch.setattr(main_module.run_pipeline_task, "apply_async", broken)
    monkeypatch.setattr(config, "CELERY_TASK_ALWAYS_EAGER", False)

    response = client.post("/papers/arxiv/process", headers=key_headers, json={"arxiv_ids": ["2410.00001"]})
    assert response.status_code == 503
    assert not any(Path(config.UPLOAD_DIR).glob("*"))


# --- /papers/seen ---

def test_mark_seen(key_headers):
    body = client.post("/papers/seen", headers=key_headers,
                       json={"paper_ids": ["2410.00001v3", "2410.00002"], "feed": "daily-nlp"}).json()
    assert body == {"feed": "daily-nlp", "marked": 2, "already_seen": 0}
    again = client.post("/papers/seen", headers=key_headers,
                        json={"paper_ids": ["2410.00001"], "feed": "daily-nlp"}).json()
    assert again == {"feed": "daily-nlp", "marked": 0, "already_seen": 1}
    assert db.seen_paper_ids("alice", "daily-nlp", "arxiv", ["2410.00001", "2410.00002"]) == {
        "2410.00001", "2410.00002"}


# --- PDF downloads ---

class _FakeResponse:
    def __init__(self, status=200, content_type="application/pdf", body=b"%PDF-1.4 fake"):
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        self._body = body

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_uses_export_host_and_saves_pdf(monkeypatch, tmp_path):
    urls = []

    def fake_get(url, **kwargs):
        urls.append(url)
        return _FakeResponse()
    monkeypatch.setattr(papers.requests, "get", fake_get)

    path = papers.download_arxiv_pdf("hep-th/9901001v1", tmp_path)
    assert urls == ["https://export.arxiv.org/pdf/hep-th/9901001v1"]
    assert path == tmp_path / "arXiv hep-th_9901001v1.pdf"
    assert path.read_bytes() == b"%PDF-1.4 fake"


def test_download_refuses_html_and_oversized(monkeypatch, tmp_path):
    monkeypatch.setattr(papers.requests, "get", lambda url, **kw: _FakeResponse(content_type="text/html"))
    with pytest.raises(papers.ArxivError, match="no PDF"):
        papers.download_arxiv_pdf("2410.00001", tmp_path)

    monkeypatch.setattr(papers, "ARXIV_MAX_PDF_BYTES", 10)
    monkeypatch.setattr(papers.requests, "get", lambda url, **kw: _FakeResponse(body=b"x" * 100))
    with pytest.raises(papers.ArxivError, match="over"):
        papers.download_arxiv_pdf("2410.00001", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_many_skips_failures_but_not_all(monkeypatch, tmp_path):
    def fake_get(url, **kwargs):
        return _FakeResponse(status=404) if "00002" in url else _FakeResponse()
    monkeypatch.setattr(papers.requests, "get", fake_get)

    paths, downloaded, failures = papers.download_arxiv_pdfs(["2410.00001", "2410.00002"], str(tmp_path))
    assert downloaded == ["2410.00001"] and len(paths) == 1
    assert failures == ["2410.00002: Could not download 2410.00002 (HTTP 404)."]

    monkeypatch.setattr(papers.requests, "get", lambda url, **kw: _FakeResponse(status=404))
    with pytest.raises(papers.ArxivError, match="None of the papers"):
        papers.download_arxiv_pdfs(["2410.00003"], str(tmp_path))


# --- the worker task ---

def _run_task(monkeypatch, tmp_path, pipeline):
    from app import pipeline as pipeline_module
    from app.celery_app import run_pipeline_task

    monkeypatch.setattr(papers, "download_arxiv_pdfs", lambda ids, job_dir: (
        [str(tmp_path / f"{i}.pdf") for i in ids[:1]], ids[:1], [f"{i}: HTTP 404" for i in ids[1:]]))
    monkeypatch.setattr(pipeline_module, "run_pipeline", pipeline)
    db.create_user("alice", hash_password("correct-horse"))

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    run_pipeline_task.push_request(id="job-1")
    try:
        return run_pipeline_task.run(str(job_dir), [], [], "summarize", "alice",
                                     arxiv_ids=["2410.00001v2", "2410.00002"], feed="daily-nlp")
    finally:
        run_pipeline_task.pop_request()


def test_task_feeds_pdfs_to_pipeline_then_marks_them_seen(monkeypatch, tmp_path):
    received = {}

    def fake_pipeline(file_paths, seed_urls, instructions, username=None, run_id=None):
        received.update(file_paths=file_paths, run_id=run_id)
        return {"report": "ok"}

    result = _run_task(monkeypatch, tmp_path, fake_pipeline)
    assert received == {"file_paths": [str(tmp_path / "2410.00001v2.pdf")], "run_id": "job-1"}
    assert result["arxiv"] == {"included": ["2410.00001v2"], "failed": ["2410.00002: HTTP 404"]}
    # Only the paper that made it into the report is marked, without its version.
    assert db.seen_paper_ids("alice", "daily-nlp", "arxiv", ["2410.00001", "2410.00002"]) == {"2410.00001"}


def test_failed_run_marks_nothing_seen(monkeypatch, tmp_path):
    def failing_pipeline(*args, **kwargs):
        raise RuntimeError("Groq is down")

    with pytest.raises(RuntimeError):
        _run_task(monkeypatch, tmp_path, failing_pipeline)
    assert db.seen_paper_ids("alice", "daily-nlp", "arxiv", ["2410.00001"]) == set()
    assert not (tmp_path / "job").exists()

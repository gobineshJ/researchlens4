"""
The scheduled digest (app.daily_digest): config parsing, and whole runs with
arXiv, the pipeline and SMTP faked -- which papers get processed, what gets
written, what gets marked seen, and when an email goes out.
"""

import json
from pathlib import Path

import pytest

from app import daily_digest, papers
from app import pipeline as pipeline_module


def _write_config(tmp_path, body: str) -> Path:
    path = tmp_path / "digest.toml"
    path.write_text(body, encoding="utf-8")
    return path


TWO_FEEDS = """
[defaults]
days = 2
max_papers = 2

[[feed]]
name = "rag"
keywords = ["retrieval augmented generation"]

[[feed]]
name = "agents"
categories = ["cs.AI"]
max_papers = 1
instructions = "Compare them."
"""


def _paper(arxiv_id: str, title: str = "") -> dict:
    return {"arxiv_id": arxiv_id, "id": papers.base_id(arxiv_id), "title": title or f"Paper {arxiv_id}",
            "abstract": "", "authors": ["A. Author", "B. Author"], "primary_category": "cs.CL",
            "categories": ["cs.CL"], "published": "", "updated": "",
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}", "pdf_url": ""}


# --- config ---

def test_config_merges_defaults(tmp_path):
    cfg = daily_digest.load_config(_write_config(tmp_path, TWO_FEEDS))
    rag, agents = cfg.feeds
    assert (rag.name, rag.days, rag.max_papers) == ("rag", 2, 2)
    assert rag.instructions == daily_digest.config.ARXIV_DEFAULT_INSTRUCTIONS
    assert (agents.max_papers, agents.instructions) == (1, "Compare them.")
    assert cfg.email_when_nothing_new is False


@pytest.mark.parametrize("body,message", [
    ("", "no \\[\\[feed\\]\\]"),
    ('[[feed]]\nname = "x"\nkeywords = ["a"]\ntypo = 1', "Unknown setting"),
    ('[[feed]]\nname = "has space"\nkeywords = ["a"]', "Feed name"),
    ('[[feed]]\nname = "x"\nkeywords = ["a"]\nmax_papers = 50', "max_papers"),
    ('[[feed]]\nname = "x"', "keyword or category"),
    ('[[feed]]\nname = "x"\ncategories = ["not a cat"]', "Not an arXiv category"),
    ('[[feed]]\nname = "x"\nkeywords = ["a"]\n[[feed]]\nname = "x"\nkeywords = ["b"]', "unique"),
])
def test_config_errors(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        daily_digest.load_config(_write_config(tmp_path, body))


def test_repo_digest_config_is_valid():
    daily_digest.load_config(Path(__file__).resolve().parents[3] / "digest.toml")


# --- whole runs ---

@pytest.fixture
def fakes(monkeypatch, tmp_path):
    """arXiv returns `search_results[feed query]`; the pipeline writes a fake
    PDF; emails are captured instead of sent."""
    state = {"search": {}, "pipeline_calls": [], "emails": [], "fail_pipeline_for": set()}

    def fake_search(query, max_results):
        for key, found in state["search"].items():
            if key in query:
                return len(found), found
        return 0, []

    def fake_download(ids, dest):
        paths = []
        for i in ids:
            path = Path(dest) / f"arXiv {i}.pdf"
            path.write_bytes(b"%PDF")
            paths.append(str(path))
        return paths, list(ids), []

    def fake_pipeline(file_paths, seed_urls, instructions, username=None, run_id=None):
        state["pipeline_calls"].append({"files": [Path(p).name for p in file_paths],
                                        "instructions": instructions, "run_id": run_id})
        if any(name in run_id for name in state["fail_pipeline_for"]):
            raise RuntimeError("Groq is down")
        pdf = tmp_path / f"report_{len(state['pipeline_calls'])}.pdf"
        pdf.write_bytes(b"%PDF-report")
        return {"report_text": "The papers propose things [1].", "citations_used": ["arXiv x.pdf, page 1"],
                "pdf_path": str(pdf)}

    monkeypatch.setattr(papers, "search_arxiv", fake_search)
    monkeypatch.setattr(papers, "download_arxiv_pdfs", fake_download)
    monkeypatch.setattr(pipeline_module, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(daily_digest, "send_email", lambda msg: state["emails"].append(msg))
    monkeypatch.setenv("SMTP_USER", "me@example.com")
    monkeypatch.delenv("DIGEST_EMAIL_TO", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    return state


def _run(tmp_path, *extra, config=TWO_FEEDS) -> tuple[int, Path]:
    state_dir = tmp_path / "digests"
    code = daily_digest.main(["--config", str(_write_config(tmp_path, config)),
                              "--state-dir", str(state_dir), *extra])
    return code, state_dir


def _body(email) -> str:
    return email.get_body(("plain",)).get_content()


def _seen(state_dir) -> dict:
    return json.loads((state_dir / "seen.json").read_text())


def test_full_run_writes_reports_marks_seen_and_emails(tmp_path, fakes):
    fakes["search"] = {
        "retrieval augmented generation": [_paper("2610.00001v2"), _paper("2610.00002v1"), _paper("2610.00003v1")],
        "cat:cs.AI": [_paper("2610.00009v1")],
    }
    code, state_dir = _run(tmp_path)
    assert code == 0

    # max_papers caps each feed; each feed gets its own instructions.
    rag_call, agents_call = fakes["pipeline_calls"]
    assert rag_call["files"] == ["arXiv 2610.00001v2.pdf", "arXiv 2610.00002v1.pdf"]
    assert agents_call["instructions"] == "Compare them."

    reports = sorted(p.name for p in (state_dir / "reports").iterdir())
    day = reports[0][:10]
    assert reports == [f"{day}-agents.md", f"{day}-agents.pdf", f"{day}-rag.md", f"{day}-rag.pdf"]
    markdown = (state_dir / "reports" / f"{day}-rag.md").read_text()
    assert "Paper 2610.00001v2" in markdown and "The papers propose things [1]." in markdown
    assert "1. arXiv x.pdf, page 1" in markdown

    assert _seen(state_dir) == {"rag": {"2610.00001": day, "2610.00002": day}, "agents": {"2610.00009": day}}

    [email] = fakes["emails"]
    assert email["Subject"] == f"ResearchLens digest {day}: 3 new papers"
    assert email["To"] == "me@example.com"
    assert sorted(a.get_filename() for a in email.iter_attachments()) == [f"{day}-agents.pdf", f"{day}-rag.pdf"]


def test_second_run_skips_covered_papers(tmp_path, fakes):
    fakes["search"] = {"retrieval augmented generation": [_paper("2610.00001v1"), _paper("2610.00002v1")]}
    _run(tmp_path)
    fakes["pipeline_calls"].clear()
    fakes["emails"].clear()

    # Same papers again (one as a new version) plus one new one.
    fakes["search"] = {"retrieval augmented generation": [_paper("2610.00001v2"), _paper("2610.00002v1"),
                                                          _paper("2610.00004v1")]}
    code, _ = _run(tmp_path)
    assert code == 0
    assert [c["files"] for c in fakes["pipeline_calls"]] == [["arXiv 2610.00004v1.pdf"]]


def test_nothing_new_sends_no_email_and_writes_nothing(tmp_path, fakes):
    code, state_dir = _run(tmp_path)
    assert code == 0
    assert fakes["pipeline_calls"] == [] and fakes["emails"] == []
    assert not (state_dir / "reports").exists()


def test_nothing_new_can_still_email(tmp_path, fakes):
    _run(tmp_path, config="email_when_nothing_new = true\n" + TWO_FEEDS)
    [email] = fakes["emails"]
    assert "0 new papers" in email["Subject"]
    assert "rag: nothing new" in _body(email)


def test_failed_feed_doesnt_stop_others_and_isnt_marked(tmp_path, fakes):
    fakes["search"] = {"retrieval augmented generation": [_paper("2610.00001v1")],
                       "cat:cs.AI": [_paper("2610.00009v1")]}
    fakes["fail_pipeline_for"] = {"digest-rag-"}
    code, state_dir = _run(tmp_path)

    assert code == 1
    seen = _seen(state_dir)
    assert seen["rag"] == {} and list(seen["agents"]) == ["2610.00009"]
    [email] = fakes["emails"]
    assert "rag: FAILED" in _body(email) and "Groq is down" in _body(email)


def test_email_failure_keeps_progress_but_fails_run(tmp_path, fakes, monkeypatch):
    fakes["search"] = {"retrieval augmented generation": [_paper("2610.00001v1")]}

    def broken(msg):
        raise OSError("SMTP auth failed")
    monkeypatch.setattr(daily_digest, "send_email", broken)

    code, state_dir = _run(tmp_path)
    assert code == 1
    assert list(_seen(state_dir)["rag"]) == ["2610.00001"]


def test_no_smtp_user_skips_email(tmp_path, fakes, monkeypatch):
    monkeypatch.delenv("SMTP_USER")
    fakes["search"] = {"retrieval augmented generation": [_paper("2610.00001v1")]}
    code, _ = _run(tmp_path)
    assert code == 0 and fakes["emails"] == []


def test_dry_run_changes_nothing(tmp_path, fakes, capsys):
    fakes["search"] = {"retrieval augmented generation": [_paper("2610.00001v1", "A RAG paper")]}
    code, state_dir = _run(tmp_path, "--dry-run")
    assert code == 0
    assert "would process 2610.00001v1: A RAG paper" in capsys.readouterr().out
    assert fakes["pipeline_calls"] == [] and fakes["emails"] == []
    assert not state_dir.exists()


def test_email_links_to_report_on_github(tmp_path, fakes, monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/researchlens")
    monkeypatch.setenv("DIGEST_EMAIL_TO", "inbox@example.com")
    fakes["search"] = {"retrieval augmented generation": [_paper("2610.00001v1")]}
    _run(tmp_path)
    [email] = fakes["emails"]
    assert email["To"] == "inbox@example.com"
    assert "https://github.com/someone/researchlens/blob/main/digests/reports/" in _body(email)

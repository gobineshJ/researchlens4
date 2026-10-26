"""
The scheduled digest (app.daily_digest): config parsing, choosing papers by
relevance, and whole runs with arXiv, the pipeline, the synthesis call and
SMTP faked -- which papers get summarised, what gets written, what gets
marked seen, and when an email goes out.
"""

import json
from pathlib import Path

import pytest

from app import daily_digest, groq_client, papers
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


def _paper(arxiv_id: str, title: str = "", abstract: str = "") -> dict:
    return {"arxiv_id": arxiv_id, "id": papers.base_id(arxiv_id), "title": title or f"Paper {arxiv_id}",
            "abstract": abstract or f"Abstract of {arxiv_id}.", "authors": ["A. Author", "B. Author"],
            "primary_category": "cs.CL", "categories": ["cs.CL"], "published": "", "updated": "",
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}", "pdf_url": ""}


def _body(email) -> str:
    return email.get_body(("plain",)).get_content()


def _seen(state_dir) -> dict:
    return json.loads((state_dir / "seen.json").read_text())


# --- config ---

def test_config_merges_defaults(tmp_path):
    cfg = daily_digest.load_config(_write_config(tmp_path, TWO_FEEDS))
    rag, agents = cfg.feeds
    assert (rag.name, rag.days, rag.max_papers) == ("rag", 2, 2)
    assert rag.instructions == daily_digest.config.DIGEST_PAPER_INSTRUCTIONS
    assert rag.min_relevance == daily_digest.config.DIGEST_MIN_RELEVANCE
    assert (agents.max_papers, agents.instructions) == (1, "Compare them.")
    assert cfg.email_when_nothing_new is False


def test_topic_text_prefers_topic_then_keywords():
    assert daily_digest.Feed(name="x", keywords=["a", " b "], topic="  RAG evaluation ").topic_text == "RAG evaluation"
    assert daily_digest.Feed(name="x", keywords=["a", " b "]).topic_text == "a; b"
    assert daily_digest.Feed(name="x", categories=["cs.AI"]).topic_text == ""


@pytest.mark.parametrize("body,message", [
    ("", "no \\[\\[feed\\]\\]"),
    ('[[feed]]\nname = "x"\nkeywords = ["a"]\ntypo = 1', "Unknown setting"),
    ('[[feed]]\nname = "has space"\nkeywords = ["a"]', "Feed name"),
    ('[[feed]]\nname = "x"\nkeywords = ["a"]\nmax_papers = 50', "max_papers"),
    ('[[feed]]\nname = "x"\nkeywords = ["a"]\nmin_relevance = 2.0', "min_relevance"),
    ('[[feed]]\nname = "x"', "keyword or category"),
    ('[[feed]]\nname = "x"\ncategories = ["not a cat"]', "Not an arXiv category"),
    ('[[feed]]\nname = "x"\nkeywords = ["a"]\n[[feed]]\nname = "x"\nkeywords = ["b"]', "unique"),
])
def test_config_errors(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        daily_digest.load_config(_write_config(tmp_path, body))


def test_repo_digest_config_is_valid():
    daily_digest.load_config(Path(__file__).resolve().parents[3] / "digest.toml")


# --- choosing papers ---

def test_choose_papers_ranks_by_relevance_and_drops_weak_matches(monkeypatch):
    scores = {"a": 0.31, "b": 0.62, "c": 0.12, "d": 0.45}
    monkeypatch.setattr(daily_digest, "relevance_scores",
                        lambda topic, cands: [scores[p["arxiv_id"]] for p in cands])
    feed = daily_digest.Feed(name="rag", keywords=["rag"], max_papers=2, min_relevance=0.2)
    chosen = daily_digest.choose_papers(feed, [_paper(i) for i in "abcd"])
    assert [(s, p["arxiv_id"]) for s, p in chosen] == [(0.62, "b"), (0.45, "d")]

    feed.max_papers = 5
    assert [p["arxiv_id"] for _, p in daily_digest.choose_papers(feed, [_paper(i) for i in "abcd"])] == \
        ["b", "d", "a"]


def test_categories_only_feed_keeps_newest_first_without_ranking(monkeypatch):
    def no_ranking(topic, cands):
        raise AssertionError("nothing to rank against")
    monkeypatch.setattr(daily_digest, "relevance_scores", no_ranking)
    feed = daily_digest.Feed(name="ai", categories=["cs.AI"], max_papers=2)
    assert daily_digest.choose_papers(feed, [_paper(i) for i in "xyz"]) == [(None, _paper("x")), (None, _paper("y"))]


def test_relevance_scores_are_cosine_similarities(monkeypatch):
    from app import vectorstore
    monkeypatch.setattr(vectorstore, "embed", lambda texts: [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    assert daily_digest.relevance_scores("topic", [_paper("a"), _paper("b")]) == pytest.approx([0.6, 0.0])


# --- per-paper helpers ---

@pytest.mark.parametrize("text,expected", [
    ("TL;DR: X beats Y by 3% [1].\n\n**Problem** ...", ("X beats Y by 3% [1].", "**Problem** ...")),
    ("**TL;DR:** A benchmark [2].**\nrest", ("A benchmark [2].", "rest")),
    ("\nTLDR - short one\nrest", ("short one", "rest")),
    ("Problem: no tldr here", ("", "Problem: no tldr here")),
])
def test_split_tldr(text, expected):
    assert daily_digest.split_tldr(text) == expected


def test_abstract_record_joins_the_papers_pdf_as_one_document():
    record = daily_digest.abstract_record(_paper("2610.1", "A Title", "We show X."), "arXiv 2610.1.pdf")
    assert record["filename"] == "arXiv 2610.1.pdf" and record["location"] == "arXiv abstract"
    assert "A Title" in record["text"] and "We show X." in record["text"] and "A. Author" in record["text"]


# --- whole runs ---

@pytest.fixture
def fakes(monkeypatch, tmp_path):
    """arXiv returns state["search"][a substring of the query]; relevance is
    the order papers were listed in; the pipeline summarises each PDF;
    synthesis and email are captured."""
    state = {"search": {}, "pipeline_calls": [], "emails": [], "synth_calls": [],
             "fail_summary_for": set(), "fail_synthesis": False}

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

    def fake_pipeline(file_paths, seed_urls, instructions, username=None, run_id=None, *,
                      extra_records=None, intent=None, export_pdf=True, mind_map=True):
        [pdf] = file_paths
        state["pipeline_calls"].append({"file": Path(pdf).name, "instructions": instructions, "run_id": run_id,
                                        "intent": intent, "export_pdf": export_pdf, "mind_map": mind_map,
                                        "extra": [r["location"] for r in extra_records or []]})
        if any(i in pdf for i in state["fail_summary_for"]):
            raise RuntimeError("Groq is down")
        name = Path(pdf).stem
        return {"report_text": f"TL;DR: {name} does something [1].\n\nProblem: it [1]. Results: 3% [3].",
                "citations": [{"n": 1, "label": f"{name}.pdf, arXiv abstract"},
                              {"n": 3, "label": f"{name}.pdf, page 7"}]}

    def fake_synthesis(topic, summaries):
        state["synth_calls"].append({"topic": topic, "titles": [s["title"] for s in summaries]})
        if state["fail_synthesis"]:
            raise ValueError("bad JSON")
        return {"themes": "Both papers study retrieval [P1] [P2].", "read_in_full": 2, "why": "It has numbers."}

    monkeypatch.setattr(papers, "search_arxiv", fake_search)
    monkeypatch.setattr(papers, "download_arxiv_pdfs", fake_download)
    monkeypatch.setattr(pipeline_module, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(groq_client, "synthesize_digest", fake_synthesis)
    monkeypatch.setattr(groq_client, "judge_claims", lambda claims, chunks, **k: [
        {"claim": c["text"], "sources": c["sources"], "verdict": "supported", "reason": ""} for c in claims])
    monkeypatch.setattr(daily_digest, "relevance_scores",
                        lambda topic, cands: [0.9 - 0.1 * i for i in range(len(cands))])
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


def test_full_run_summarises_each_paper_then_synthesises(tmp_path, fakes):
    fakes["search"] = {
        "retrieval augmented generation": [_paper("2610.00001v2"), _paper("2610.00002v1"), _paper("2610.00003v1")],
        "cat:cs.AI": [_paper("2610.00009v1")],
    }
    code, state_dir = _run(tmp_path)
    assert code == 0

    # One pipeline run per paper, in summary mode, with its abstract, no PDF of its own.
    calls = fakes["pipeline_calls"]
    assert [c["file"] for c in calls] == ["arXiv 2610.00001v2.pdf", "arXiv 2610.00002v1.pdf",
                                          "arXiv 2610.00009v1.pdf"]
    assert all(c["intent"] == "summary" and c["export_pdf"] is False and c["mind_map"] is False
               and c["extra"] == ["arXiv abstract"]
               for c in calls)
    assert calls[0]["instructions"] == daily_digest.config.DIGEST_PAPER_INSTRUCTIONS
    assert calls[2]["instructions"] == "Compare them."
    # Synthesis only for the feed with more than one paper.
    assert fakes["synth_calls"] == [{"topic": "retrieval augmented generation",
                                     "titles": ["Paper 2610.00001v2", "Paper 2610.00002v1"]}]

    reports = sorted(p.name for p in (state_dir / "reports").iterdir())
    day = reports[0][:10]
    assert reports == [f"{day}-agents.md", f"{day}-agents.pdf", f"{day}-rag.md", f"{day}-rag.pdf"]
    markdown = (state_dir / "reports" / f"{day}-rag.md").read_text(encoding="utf-8")
    assert "## Worth reading in full" in markdown and "**[P2] Paper 2610.00002v1** -- It has numbers." in markdown
    assert "Both papers study retrieval [P1] [P2]." in markdown
    assert "*1 theme sentence(s) checked against the paper summaries.*" in markdown
    assert "**TL;DR:** arXiv 2610.00001v2 does something [1]." in markdown
    # Sources keep the report's own numbers -- [3], not a renumbered [2].
    assert "- [3] arXiv 2610.00001v2.pdf, page 7" in markdown
    assert (state_dir / "reports" / f"{day}-rag.pdf").read_bytes().startswith(b"%PDF")

    assert _seen(state_dir) == {"rag": {"2610.00001": day, "2610.00002": day}, "agents": {"2610.00009": day}}

    [email] = fakes["emails"]
    assert email["Subject"] == f"ResearchLens digest {day}: 3 new papers"
    body = _body(email)
    assert "Worth reading in full: Paper 2610.00002v1" in body
    assert "arXiv 2610.00001v2 does something [1]." in body
    assert sorted(a.get_filename() for a in email.iter_attachments()) == [f"{day}-agents.pdf", f"{day}-rag.pdf"]


def test_low_relevance_papers_are_skipped_and_not_marked(tmp_path, fakes, monkeypatch):
    monkeypatch.setattr(daily_digest, "relevance_scores", lambda topic, cands: [0.5, 0.05])
    fakes["search"] = {"retrieval augmented generation": [_paper("2610.00001v1"), _paper("2610.00002v1")]}
    _run(tmp_path)
    assert [c["file"] for c in fakes["pipeline_calls"]] == ["arXiv 2610.00001v1.pdf"]
    assert fakes["synth_calls"] == []   # one paper: nothing to synthesise


def test_second_run_skips_covered_papers(tmp_path, fakes):
    fakes["search"] = {"retrieval augmented generation": [_paper("2610.00001v1"), _paper("2610.00002v1")]}
    _run(tmp_path)
    fakes["pipeline_calls"].clear()

    # Same papers again (one as a new version) plus one new one.
    fakes["search"] = {"retrieval augmented generation": [_paper("2610.00001v2"), _paper("2610.00002v1"),
                                                          _paper("2610.00004v1")]}
    assert _run(tmp_path)[0] == 0
    assert [c["file"] for c in fakes["pipeline_calls"]] == ["arXiv 2610.00004v1.pdf"]


def test_one_paper_failing_doesnt_sink_its_feed(tmp_path, fakes):
    fakes["search"] = {"retrieval augmented generation": [_paper("2610.00001v1"), _paper("2610.00002v1")]}
    fakes["fail_summary_for"] = {"2610.00001"}
    code, state_dir = _run(tmp_path)

    assert code == 0
    assert list(_seen(state_dir)["rag"]) == ["2610.00002"]
    markdown = next((state_dir / "reports").glob("*-rag.md")).read_text(encoding="utf-8")
    assert "## Not included" in markdown and "2610.00001v1: RuntimeError: Groq is down" in markdown


def test_failed_synthesis_still_writes_the_report(tmp_path, fakes):
    fakes["search"] = {"retrieval augmented generation": [_paper("2610.00001v1"), _paper("2610.00002v1")]}
    fakes["fail_synthesis"] = True
    code, state_dir = _run(tmp_path)
    assert code == 0
    markdown = next((state_dir / "reports").glob("*-rag.md")).read_text(encoding="utf-8")
    assert "## Themes" not in markdown and "## At a glance" in markdown
    assert len(_seen(state_dir)["rag"]) == 2


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


def test_feed_where_every_paper_fails_fails_alone(tmp_path, fakes):
    fakes["search"] = {"retrieval augmented generation": [_paper("2610.00001v1")],
                       "cat:cs.AI": [_paper("2610.00009v1")]}
    fakes["fail_summary_for"] = {"2610.00001"}
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
    assert "would process 2610.00001v1 (relevance 0.90): A RAG paper" in capsys.readouterr().out
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


# --- fact-checking the themes ---

def _summary(i: int, body: str) -> daily_digest.PaperSummary:
    return daily_digest.PaperSummary(_paper(f"2610.0000{i}v1", f"Paper {i}"), 0.5, f"tldr {i}", body, [])


def test_check_themes_fixes_a_wrong_number_against_the_summaries(monkeypatch):
    judged = {}

    def fake_judge(claims, sources, **kwargs):
        judged["claims"] = [(c["text"], c["sources"]) for c in claims]
        judged["sources"] = [s["text"] for s in sources]
        return [{"claim": c["text"], "sources": c["sources"],
                 "verdict": "contradicted" if "3B" in c["text"] else "supported",
                 "reason": "the summary says Gemma-3-4B"} for c in claims]

    def fake_revise(flagged, sources, **kwargs):
        return [{"action": "rewrite", "text": "Gemma-3-4B gains 27.5 points."} for _ in flagged]

    monkeypatch.setattr(groq_client, "judge_claims", fake_judge)
    monkeypatch.setattr(groq_client, "revise_claims", fake_revise)
    themes = "Both papers scale their retrieval corpora [P1, P2]. A 3B model gains 27.5 points [P2]."
    checked, note = daily_digest.check_themes(themes, [_summary(1, "MMR helps."), _summary(2, "Gemma-3-4B +27.5.")])

    assert judged["claims"] == [("Both papers scale their retrieval corpora.", [1, 2]), ("A 3B model gains 27.5 points.", [2])]
    assert "Gemma-3-4B +27.5." in judged["sources"][1]
    assert checked == "Both papers scale their retrieval corpora [P1, P2]. Gemma-3-4B gains 27.5 points [P2]."
    assert note == "2 theme sentence(s) checked against the paper summaries; 1 corrected, 0 removed."


def test_check_themes_without_paper_citations_is_left_alone(monkeypatch):
    def must_not_judge(*a, **k):
        raise AssertionError("nothing to check")
    monkeypatch.setattr(groq_client, "judge_claims", must_not_judge)
    assert daily_digest.check_themes("No citations here.", [_summary(1, "x")]) == \
        ("No citations here.", "Themes cite no papers, so they could not be checked.")


def test_theme_check_failure_keeps_themes_and_says_so(tmp_path, fakes, monkeypatch):
    def broken(*a, **k):
        raise TimeoutError("Groq slow")
    monkeypatch.setattr(groq_client, "judge_claims", broken)
    fakes["search"] = {"retrieval augmented generation": [_paper("2610.00001v1"), _paper("2610.00002v1")]}
    code, state_dir = _run(tmp_path)
    assert code == 0
    markdown = next((state_dir / "reports").glob("*-rag.md")).read_text(encoding="utf-8")
    assert "Both papers study retrieval [P1] [P2]." in markdown
    assert "*Themes could not be fact-checked (TimeoutError).*" in markdown


@pytest.mark.parametrize("text,expected", [
    ("Both study RAG. [P1] sweeps 192 configs. [P2] scales corpora [P2].",
     "Both study RAG. P1 sweeps 192 configs [P1]. P2 scales corpora [P2]."),
    ("[P1, P2] differ in focus.\n\n[P3] is alone", "P1, P2 differ in focus [P1, P2].\n\nP3 is alone [P3]"),
    ("Untouched sentence [P1].", "Untouched sentence [P1]."),
])
def test_leading_paper_markers_move_to_sentence_end(text, expected):
    assert daily_digest.markers_to_sentence_ends(text) == expected


def test_check_themes_judges_a_leading_marker_against_its_own_paper(monkeypatch):
    seen = {}

    def fake_judge(claims, sources, **kwargs):
        seen["claims"] = [(c["text"], c["sources"]) for c in claims]
        return [{"claim": c["text"], "sources": c["sources"], "verdict": "supported", "reason": ""}
                for c in claims]

    monkeypatch.setattr(groq_client, "judge_claims", fake_judge)
    daily_digest.check_themes("Both papers study retrieval at scale [P1, P2]. [P2] focuses on biomedical corpora.",
                              [_summary(1, "a"), _summary(2, "b")])
    assert seen["claims"][1] == ("P2 focuses on biomedical corpora.", [2])

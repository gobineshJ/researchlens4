"""
The mind map: the final report's cited claims grouped into themes and
linked to the documents they cite. Groq is faked throughout.
"""

import json

import pytest

from app import evaluation, groq_client, mindmap
from tests.unit.test_pipeline import _chunks_for, _run

RETRIEVED = [
    {"text": "Attention is all you need.", "source_label": "a.pdf, page 1", "source_name": "a.pdf",
     "source_type": "file"},
    {"text": "Training costs grow quickly.", "source_label": "a.pdf, page 4", "source_name": "a.pdf",
     "source_type": "file"},
    {"text": "A blog on scaling.", "source_label": "Scaling, https://example.com/blog/scaling/",
     "source_name": "https://example.com/blog/scaling/", "source_type": "web"},
]
REPORT = ("Transformers rely entirely on attention layers [1]. Training them costs a lot of compute [2, 3]. "
          "Scaling laws predict loss from model size [3].")


def _fake_group(themes):
    return lambda instructions, claims, *a, **k: themes


def _nodes(graph, kind):
    return [n for n in graph["nodes"] if n["type"] == kind]


# ---------- the grouping call ----------

def test_group_claims_keeps_valid_unique_members_only(monkeypatch):
    def fake_call(model, system_prompt, user_prompt, json_mode=True, temperature=0.2, **options):
        return json.dumps({"themes": [
            {"label": "  Architecture ", "summary": "How it works.", "sentences": [1, 1, 9, "x"]},
            {"label": "Cost", "summary": "", "sentences": [1, 2]},   # 1 already placed
            {"label": "Empty", "summary": "", "sentences": [42]},   # nothing valid: dropped
            {"label": "", "summary": "", "sentences": [3]},         # no label: dropped
        ]})

    monkeypatch.setattr(groq_client, "_call_groq", fake_call)
    themes = groq_client.group_claims("q", ["a", "b", "c"])
    assert themes == [
        {"label": "Architecture", "summary": "How it works.", "claims": [0]},
        {"label": "Cost", "summary": "", "claims": [1]},
    ]


def test_group_claims_caps_the_number_of_themes(monkeypatch):
    monkeypatch.setattr(groq_client, "_call_groq", lambda *a, **k: json.dumps({"themes": [
        {"label": f"T{i}", "summary": "", "sentences": [i + 1]} for i in range(5)
    ]}))
    assert [t["label"] for t in groq_client.group_claims("q", list("abcde"), max_themes=2)] == ["T0", "T1"]


def test_group_claims_raises_when_nothing_is_usable(monkeypatch):
    monkeypatch.setattr(groq_client, "_call_groq", lambda *a, **k: json.dumps({"themes": "nope"}))
    with pytest.raises(ValueError):
        groq_client.group_claims("q", ["a"])


# ---------- building the graph ----------

def test_graph_links_question_themes_claims_and_documents(monkeypatch):
    monkeypatch.setattr(groq_client, "group_claims", _fake_group([
        {"label": "Architecture", "summary": "s", "claims": [0]},
        {"label": "Cost", "summary": "s", "claims": [1, 2]},
    ]))
    verdicts = [{"claim": "Training them costs a lot of compute.", "sources": [2, 3],
                 "verdict": "unsupported", "reason": "Overstated."}]
    report = evaluation.EvaluationReport()
    graph = mindmap.build_mind_map(REPORT, RETRIEVED, "How do transformers work?", verdicts, report)

    assert graph["themes_from"] == "llm"
    assert _nodes(graph, "question") == [{"id": "q", "type": "question", "label": "How do transformers work?"}]
    assert [t["label"] for t in _nodes(graph, "theme")] == ["Architecture", "Cost"]

    claims = {c["id"]: c for c in _nodes(graph, "claim")}
    assert claims["c1"]["citations"] == [2, 3]
    assert claims["c1"]["verdict"] == "unsupported" and claims["c1"]["reason"] == "Overstated."
    assert claims["c0"]["verdict"] == "unjudged"  # the claim check didn't sample it

    # One node per DOCUMENT, not per excerpt; web pages shown without the scheme.
    sources = {s["name"]: s for s in _nodes(graph, "source")}
    assert set(sources) == {"a.pdf", "https://example.com/blog/scaling/"}
    assert sources["https://example.com/blog/scaling/"]["label"] == "example.com/blog/scaling"

    cites = [(e["source"], e["target"], e["excerpts"]) for e in graph["edges"] if e["type"] == "cites"]
    a, web = sources["a.pdf"]["id"], sources["https://example.com/blog/scaling/"]["id"]
    assert cites == [("c0", a, [1]), ("c1", a, [2]), ("c1", web, [3]), ("c2", web, [3])]
    assert {(e["source"], e["target"]) for e in graph["edges"] if e["type"] == "contains"} == {
        ("q", "t0"), ("q", "t1"), ("t0", "c0"), ("t1", "c1"), ("t1", "c2"),
    }
    assert [(c["name"], c["passed"]) for c in report.checks] == [("themes", True)]


def test_claims_the_grouping_left_out_go_under_other(monkeypatch):
    monkeypatch.setattr(groq_client, "group_claims", _fake_group([
        {"label": "Architecture", "summary": "s", "claims": [0]},
    ]))
    report = evaluation.EvaluationReport()
    graph = mindmap.build_mind_map(REPORT, RETRIEVED, "q", [], report)
    themes = _nodes(graph, "theme")
    assert [(t["label"], t["claim_count"]) for t in themes] == [("Architecture", 1), (mindmap.OTHER_THEME, 2)]
    assert "2 the grouping left out" in report.checks[0]["detail"]


def test_failed_grouping_falls_back_to_one_theme_per_document(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("Groq is down")

    monkeypatch.setattr(groq_client, "group_claims", broken)
    report = evaluation.EvaluationReport()
    graph = mindmap.build_mind_map(REPORT, RETRIEVED, "q", [], report)

    assert graph["themes_from"] == "sources"
    # Claim 1 cites [2, 3]: it goes under its FIRST cited document.
    assert [(t["label"], t["claim_count"]) for t in _nodes(graph, "theme")] == [
        ("a.pdf", 2), ("example.com/blog/scaling", 1),
    ]
    assert report.checks[0]["passed"] is False
    assert "Groq is down" in report.errors[0]


def test_report_without_citations_has_no_map(monkeypatch):
    monkeypatch.setattr(groq_client, "group_claims", _fake_group([]))
    assert mindmap.build_mind_map("Nothing cited here at all.", RETRIEVED, "q", [],
                                  evaluation.EvaluationReport()) is None


def test_long_questions_are_shortened_for_the_root_node(monkeypatch):
    monkeypatch.setattr(groq_client, "group_claims", _fake_group([
        {"label": "All", "summary": "", "claims": [0, 1, 2]},
    ]))
    graph = mindmap.build_mind_map(REPORT, RETRIEVED, "word " * 100, [], evaluation.EvaluationReport())
    label = _nodes(graph, "question")[0]["label"]
    assert len(label) == mindmap.QUESTION_LABEL_CHARS and label.endswith("…")


# ---------- in the pipeline ----------

REPORT_TWO_CLAIMS = "Neural networks learn representations well [1]. They need large datasets to train [2]."


def test_pipeline_returns_the_map():
    chunks = _chunks_for("doc.pdf", 4, "neural networks")
    result = _run(chunks, intent="targeted_qa",
                  generate_report={"report_text": REPORT_TWO_CLAIMS, "citations_used": []})
    graph = result["mind_map"]
    assert graph["themes_from"] == "llm"
    assert len(_nodes(graph, "claim")) == 2
    assert all(c["verdict"] == "supported" for c in _nodes(graph, "claim"))
    assert "mind_map" in result["evaluation"]["stage_durations_seconds"]


def test_pipeline_can_skip_the_map():
    chunks = _chunks_for("doc.pdf", 4, "neural networks")

    def must_not_run(*a, **k):
        raise AssertionError("group_claims called with mind_map=False")

    result = _run(chunks, intent="targeted_qa", run_kwargs={"mind_map": False}, group_claims=must_not_run,
                  generate_report={"report_text": REPORT_TWO_CLAIMS, "citations_used": []})
    assert result["mind_map"] is None


def test_a_failed_map_never_fails_the_run(monkeypatch):
    chunks = _chunks_for("doc.pdf", 4, "neural networks")

    def broken(*a, **k):
        raise RuntimeError("layout bug")

    monkeypatch.setattr(mindmap, "build_mind_map", broken)
    result = _run(chunks, intent="targeted_qa",
                  generate_report={"report_text": REPORT_TWO_CLAIMS, "citations_used": []})
    assert result["mind_map"] is None
    assert result["report_text"] == REPORT_TWO_CLAIMS
    assert any("layout bug" in e for e in result["evaluation"]["errors"])

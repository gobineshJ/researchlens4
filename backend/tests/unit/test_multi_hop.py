"""
Iterative multi-hop retrieval: after each retrieval pass, the hop planner
(groq_client.plan_follow_up, faked here) reads the evidence and may ask
for a follow-up lookup of something that evidence names. Embeddings and
the cross-encoder are faked; BM25 runs for real, which is what lets a
follow-up query naming a specific method find the one chunk about it.
"""

import json

import pytest

from app import groq_client, pipeline
from tests.unit.test_pipeline import _chunks_for, _run

QUESTION = "Which dataset did the method that beat the 2019 baseline use?"


@pytest.fixture(autouse=True)
def _no_self_correction_retry(monkeypatch):
    """The mocked report is poorly grounded, which would trigger the
    self-correction retry -- and with it a second round of hop planning."""
    monkeypatch.setattr(pipeline, "MAX_RETRIEVAL_ATTEMPTS", 1)


def _corpus():
    chunks = _chunks_for("paper.pdf", 20, "general background")
    chunks[2]["text"] = "MethodX beat the 2019 baseline by four points on every benchmark."
    chunks[19]["text"] = "MethodX was trained on the ZetaSet corpus of clinical notes."
    return chunks


def _keyword_rerank(queries, candidates, top_k):
    """Stands in for the cross-encoder: ranks by how many query words a
    candidate contains."""
    words = {w.lower() for q in queries for w in q.split()}
    return sorted(candidates, key=lambda c: -sum(w in c["text"].lower() for w in words))[:top_k]


def test_follow_up_hop_retrieves_the_bridged_fact():
    plans = []

    def planner(instructions, evidence, tried):
        plans.append({"evidence": [c["text"] for c in evidence], "tried": list(tried)})
        if len(plans) == 1:
            return {"complete": False, "missing": "the dataset MethodX used",
                    "queries": ["MethodX trained corpus"]}
        return {"complete": True, "missing": "", "queries": []}

    result = _run(_corpus(), intent="targeted_qa", instructions=QUESTION,
                  plan_follow_up=planner, rerank_with_fallback=_keyword_rerank)

    texts = [c["text"] for c in result["retrieved_chunks"]]
    assert "MethodX was trained on the ZetaSet corpus of clinical notes." in texts
    # Hop 1 found the bridge (the method's name) but not the dataset...
    assert any("beat the 2019 baseline" in t for t in plans[0]["evidence"])
    assert not any("ZetaSet" in t for t in plans[0]["evidence"])
    # ...and the planner's second look includes what hop 2 found.
    assert any("ZetaSet" in t for t in plans[1]["evidence"])
    assert plans[1]["tried"] == [QUESTION, "MethodX trained corpus"]

    check = next(c for c in result["evaluation"]["checks"] if c["name"] == "multi_hop")
    assert check["value"] == 2
    assert "the dataset MethodX used" in check["detail"]


def test_hop_loop_stops_when_a_hop_finds_nothing_new():
    calls = []

    def planner(instructions, evidence, tried):
        calls.append(1)
        return {"complete": False, "missing": "more", "queries": [f"query {len(calls)}"]}

    result = _run(_chunks_for("doc.pdf", 4, "x"), intent="targeted_qa", plan_follow_up=planner)
    assert len(calls) == 1  # 4 chunks, all found by hop 1 -- hop 2 adds nothing
    check = next(c for c in result["evaluation"]["checks"] if c["name"] == "multi_hop")
    assert check["value"] == 2 and "0 new chunk(s)" in check["detail"]


def test_hops_are_capped(monkeypatch):
    monkeypatch.setattr(pipeline, "MULTI_HOP_MAX_EXTRA_HOPS", 1)
    calls = []

    def planner(instructions, evidence, tried):
        calls.append(1)
        return {"complete": False, "missing": "", "queries": ["MethodX trained corpus"]}

    _run(_corpus(), intent="analysis", instructions=QUESTION,
         plan_follow_up=planner, rerank_with_fallback=_keyword_rerank)
    assert len(calls) == 1


def test_single_hop_when_the_evidence_suffices():
    result = _run(_chunks_for("doc.pdf", 10, "x"), intent="targeted_qa")
    check = next(c for c in result["evaluation"]["checks"] if c["name"] == "multi_hop")
    assert check["value"] == 1 and check["detail"].startswith("Single hop")


def test_summary_intent_never_plans_hops():
    calls = []
    _run(_chunks_for("doc.pdf", 10, "x"), intent="summary",
         plan_follow_up=lambda *a: calls.append(1) or {"complete": True, "missing": "", "queries": []})
    assert calls == []


def test_self_correction_retry_hops_too(monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_RETRIEVAL_ATTEMPTS", 2)
    seen_tried = []

    def planner(instructions, evidence, tried):
        seen_tried.append(list(tried))
        return {"complete": True, "missing": "", "queries": []}

    _run(_chunks_for("doc.pdf", 10, "x"), intent="targeted_qa", plan_follow_up=planner,
         expand_query=lambda q, avoid=None, **k: ["reworded " + q] if avoid else [q])
    # Once for the first attempt, once for the retry's reformulated queries.
    assert seen_tried == [["Discuss the topic"], ["reworded Discuss the topic"]]


def test_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(pipeline, "MULTI_HOP_ENABLED", False)
    calls = []
    result = _run(_chunks_for("doc.pdf", 10, "x"), intent="targeted_qa",
                  plan_follow_up=lambda *a: calls.append(1))
    assert calls == []
    assert not any(c["name"] == "multi_hop" for c in result["evaluation"]["checks"])


def test_interleave_keeps_each_hops_best_chunks_near_the_front():
    a = [{"text": f"a{i}", "source_name": "A"} for i in range(3)]
    b = [{"text": "b0", "source_name": "B"}]
    later = [{"text": "a-hop", "source_name": "A"}, {"text": "b-hop", "source_name": "B"}]
    merged = pipeline._interleave_by_source(a + b, later)
    assert [c["text"] for c in merged] == ["a0", "a-hop", "a1", "a2", "b0", "b-hop"]


# ---------- the planner's own parsing (Groq faked) ----------

def _fake_planner_reply(monkeypatch, reply):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "test-key")
    sent = {}

    def fake_call(model, system_prompt, user_prompt, json_mode=True, temperature=0.2):
        sent["prompt"] = user_prompt
        if isinstance(reply, Exception):
            raise reply
        return json.dumps(reply)

    monkeypatch.setattr(groq_client, "_call_groq", fake_call)
    return sent


EVIDENCE = [{"text": "MethodX beat the baseline.", "source_label": "paper.pdf, page 2"}]


def test_planner_returns_new_queries(monkeypatch):
    sent = _fake_planner_reply(monkeypatch, {"complete": False, "missing": "dataset",
                                             "queries": ["MethodX dataset", "what did they test"]})
    plan = groq_client.plan_follow_up("q", EVIDENCE, ["What did they test"])
    assert plan == {"complete": False, "missing": "dataset", "queries": ["MethodX dataset"]}
    assert "paper.pdf, page 2" in sent["prompt"] and "- What did they test" in sent["prompt"]


def test_planner_with_only_repeated_queries_is_complete(monkeypatch):
    _fake_planner_reply(monkeypatch, {"complete": False, "queries": ["q"]})
    assert groq_client.plan_follow_up("q", EVIDENCE, ["q"])["complete"] is True


def test_planner_failure_ends_the_loop(monkeypatch):
    _fake_planner_reply(monkeypatch, RuntimeError("Groq down"))
    assert groq_client.plan_follow_up("q", EVIDENCE, [])["complete"] is True


def test_planner_without_key_is_complete(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "")
    assert groq_client.plan_follow_up("q", EVIDENCE, []) == {"complete": True, "missing": "", "queries": []}


def test_planner_evidence_is_trimmed_to_its_budget(monkeypatch):
    sent = _fake_planner_reply(monkeypatch, {"complete": True})
    evidence = [{"text": "word " * 4000, "source_label": f"s{i}"} for i in range(4)]
    groq_client.plan_follow_up("q", evidence, [], max_tokens=1000)
    assert len(sent["prompt"]) < 5000

"""
The hard golden set is internally consistent: every evidence quote is in
its document, required facts are in the reference answers, multi-hop
questions really span documents, and the generated look-alike registry
never uses the names the questions depend on (or it would make them
ambiguous) -- and is exactly what its seeded generator produces.
"""

import importlib.util
import json

import pytest

from app import knowledge_graph
from eval import metrics

QUESTIONS = metrics.load_jsonl(metrics.HARD_DIR / "questions.jsonl")
DOCS = {p.name: p.read_text(encoding="utf-8") for p in (metrics.HARD_DIR / "docs").glob("*.txt")}


def test_ids_unique_and_every_type_covered():
    assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS)
    assert {q["type"] for q in QUESTIONS} == {"targeted_qa", "multi_hop", "analysis", "compound",
                                             "summary", "unanswerable"}


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q["id"])
def test_evidence_quotes_are_in_their_documents(question):
    for doc in question.get("docs", []):
        assert doc in DOCS
    for item in question["evidence"]:
        assert metrics._squash(item["quote"]) in metrics._squash(DOCS[item["doc"]]), item["quote"]


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q["id"])
def test_required_facts_are_in_the_reference_answer(question):
    assert metrics.fact_recall(question["required_facts"], question["reference_answer"]) in (None, 1.0)


def test_multi_hop_questions_span_documents_and_never_name_the_bridge():
    for q in [q for q in QUESTIONS if q["type"] == "multi_hop"]:
        assert len({e["doc"] for e in q["evidence"]}) >= 2, q["id"]
    # The Norrby chains must be found through "Kade and colleagues", not by name.
    for q in [q for q in QUESTIONS if q["id"].startswith("h-mh-norrby") or q["id"] == "h-mh-director"]:
        assert "KestrelCell" not in q["question"], q["id"]


def test_registry_never_uses_the_protected_names():
    registry = DOCS["sodium_cell_registry.txt"]
    for name in ("KestrelCell", "VX-9", "Norrby", "Kade", "Halvik"):
        assert name not in registry, name
    assert "SodaCore " not in registry and "SodaCore." not in registry


def test_registry_matches_its_generator(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("make_registry", metrics.HARD_DIR / "make_registry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUT", tmp_path / "registry.txt")
    module.main()
    assert (tmp_path / "registry.txt").read_text(encoding="utf-8") == DOCS["sodium_cell_registry.txt"]


def test_agent_prompt_contains_no_eval_set_names():
    """The agent's instructions must not leak hints about the questions it
    is scored on: no entity from either golden set may appear in them."""
    from app import agent
    prompt = agent.SYSTEM_PROMPT + agent.GRAPH_HINT + json.dumps(agent.SEARCH_TOOLS + agent.GRAPH_TOOLS + [agent.FINISH_TOOL])
    names = set()
    for golden in (metrics.GOLDEN_DIR, metrics.HARD_DIR):
        for doc in (golden / "docs").glob("*.txt"):
            names |= {n for n, _ in knowledge_graph.extract_entities(doc.read_text(encoding="utf-8"))}
    leaked = sorted(n for n in names if len(n) > 3 and n in prompt)
    assert leaked == []

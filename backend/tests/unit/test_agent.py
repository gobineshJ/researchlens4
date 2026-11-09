"""
The retrieval agent's loop, driven by a SCRIPTED fake model: each test
lists the tool calls the "model" makes, so the loop -- tool dispatch,
evidence ledger, limits, repeat detection, context condensing, fallback --
is tested exactly, with no Groq call. Search runs for real over BM25
(vector search and the cross-encoder are faked).
"""

import json

import pytest

from app import agent, knowledge_graph, pipeline
from tests.unit.test_pipeline import _run

# Documents are numbered alphabetically: 1 = paper.txt, 2 = registry.txt, 3 = varn.txt.
CHUNKS = [
    {"text": "KestrelCell is a sodium-sulfur cell. All cells used VX-9, supplied by Varn Materials.",
     "source_name": "paper.txt", "source_label": "paper.txt, part 1", "source_type": "file", "position": 0},
    {"text": "The median KestrelCell reached 2,400 cycles at the Oskarsund Battery Test Centre.",
     "source_name": "paper.txt", "source_label": "paper.txt, part 2", "source_type": "file", "position": 1},
    {"text": "VX-9 is produced only at the Brekke plant. VX-7 was recalled in 2024.",
     "source_name": "varn.txt", "source_label": "varn.txt, part 1", "source_type": "file", "position": 0},
    {"text": "KestrelPack-3 reached 1,900 cycles with VX-11.",
     "source_name": "registry.txt", "source_label": "registry.txt, part 1", "source_type": "file", "position": 0},
]


@pytest.fixture(autouse=True)
def _no_models(monkeypatch):
    """Vector search: nothing (BM25 does the work). Reranker: keep order."""
    monkeypatch.setattr(agent.vectorstore, "retrieve_candidates", lambda *a, **k: [])
    monkeypatch.setattr(agent.reranker, "rerank_with_fallback", lambda q, cands, top_k: cands[:top_k])


def _call(name, call_id="c", **args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class Script:
    """A fake model: replies with the next scripted message, records payloads."""

    def __init__(self, *steps, tokens=500):
        self.steps, self.payloads, self.tokens = list(steps), [], tokens

    def __call__(self, payload):
        self.payloads.append(payload)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        message = {"role": "assistant", "content": None, "tool_calls": step} if isinstance(step, list) \
            else {"role": "assistant", "content": step}
        return {"choices": [{"message": message}], "usage": {"total_tokens": self.tokens}}


def _graph():
    return knowledge_graph.KnowledgeGraph.build(
        [{"cid": f"{['paper.txt', 'registry.txt', 'varn.txt'].index(c['source_name']) + 1}.{c['position']}",
          "source_name": c["source_name"], "text": c["text"]} for c in CHUNKS])


# ---------- the loop ----------

def test_search_then_finish_returns_exactly_the_chosen_passages():
    model = Script([_call("search_documents", query="VX-9 Brekke plant")],
                   [_call("finish", passage_ids=["3.0", "1.0"], reason="supplier and plant")])
    result = agent.run_agent("Where is KestrelCell's electrolyte made?", CHUNKS, None, call_model=model)
    assert [c["source_label"] for c in result.passages] == ["varn.txt, part 1", "paper.txt, part 1"]
    assert result.steps == 2 and result.tokens == 1000
    assert [s["tool"] for s in result.trace] == ["search_documents", "finish"]
    assert result.stop_reason.startswith("finish: supplier and plant")
    # The model was shown the passages it then chose.
    tool_reply = next(m for m in model.payloads[1]["messages"] if m["role"] == "tool")
    assert "[3.0]" in tool_reply["content"] and "[1.0]" in tool_reply["content"]


def test_ids_the_model_was_never_shown_are_ignored():
    model = Script([_call("read_passage", passage_id="1.1")],
                   [_call("finish", passage_ids=["9.9", "3.0", "1.1"])])
    result = agent.run_agent("q", CHUNKS, None, call_model=model)
    assert [c["source_label"] for c in result.passages] == ["paper.txt, part 2"]


def test_graph_tools_are_offered_only_with_a_graph():
    without = Script([_call("finish", passage_ids=[])])
    agent.run_agent("q", CHUNKS, None, call_model=without)
    assert {t["function"]["name"] for t in without.payloads[0]["tools"]} == {"search_documents", "read_passage", "finish"}

    with_graph = Script([_call("graph_path", from_entity="KestrelCell", to_entity="Brekke")],
                        [_call("finish", passage_ids=["2.0"])])
    result = agent.run_agent("q", CHUNKS, None, graph=_graph(), call_model=with_graph)
    assert {"graph_lookup", "graph_path"} <= {t["function"]["name"] for t in with_graph.payloads[0]["tools"]}
    path_reply = [m for m in with_graph.payloads[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert "KestrelCell -> VX-9" in path_reply and "Brekke" in path_reply
    assert result.trace[0]["result"].startswith("2 passage(s)") or "passage" in result.trace[0]["result"]


def test_step_limit_stops_the_loop_and_keeps_what_was_found(monkeypatch):
    monkeypatch.setattr(agent, "AGENT_MAX_STEPS", 2)
    model = Script([_call("search_documents", query="KestrelCell cycles")],
                   [_call("search_documents", query="VX-9")])
    result = agent.run_agent("q", CHUNKS, None, call_model=model)
    # 2 steps + the forced finish, which fails here (the script ran out),
    # so the evidence falls back to what it saw.
    assert result.stop_reason.startswith("step limit (2)") and len(model.payloads) == 3
    assert result.passages


def test_token_limit_stops_the_loop(monkeypatch):
    monkeypatch.setattr(agent, "AGENT_MAX_TOKENS", 900)
    model = Script([_call("search_documents", query="a")], [_call("search_documents", query="b")], tokens=600)
    result = agent.run_agent("q", CHUNKS, None, call_model=model)
    assert result.stop_reason.startswith("token limit") and result.steps == 2


def test_a_repeated_call_is_not_run_again():
    calls = []
    model = Script([_call("search_documents", query="VX-9")], [_call("search_documents", query="VX-9")],
                   [_call("finish", passage_ids=["2.0"])])
    session_search = agent._Session.search_documents

    def counting(self, query, source=None):
        calls.append(query)
        return session_search(self, query, source)

    agent._Session.search_documents = counting
    try:
        agent.run_agent("q", CHUNKS, None, call_model=model)
    finally:
        agent._Session.search_documents = session_search
    assert calls == ["VX-9"]
    second = [m for m in model.payloads[2]["messages"] if m["role"] == "tool"][-1]["content"]
    assert second.startswith("You already made this exact call")


def test_old_tool_results_are_condensed_to_their_ids(monkeypatch):
    monkeypatch.setattr(agent, "KEEP_FULL_RESULTS", 1)
    model = Script([_call("read_passage", passage_id="1.0")], [_call("read_passage", passage_id="2.0")],
                   [_call("finish", passage_ids=["1.0"])])
    agent.run_agent("q", CHUNKS, None, call_model=model)
    tool_messages = [m["content"] for m in model.payloads[2]["messages"] if m["role"] == "tool"]
    assert tool_messages[0] == "(earlier result, condensed) passages shown: 1.0"
    assert tool_messages[1].startswith("[2.0]")


def test_bad_tools_and_arguments_are_reported_back_not_raised():
    model = Script([_call("delete_everything"), _call("read_passage", passage_id="nope", call_id="d")],
                   [_call("finish", passage_ids=[])])
    result = agent.run_agent("q", CHUNKS, None, call_model=model)
    replies = [m["content"] for m in model.payloads[1]["messages"] if m["role"] == "tool"]
    assert replies[0].startswith("Unknown tool 'delete_everything'.") and "No passage 'nope'" in replies[1]
    assert result.passages == []


def test_model_failure_on_the_first_call_raises_later_ones_end_the_loop():
    with pytest.raises(RuntimeError):
        agent.run_agent("q", CHUNKS, None, call_model=Script(RuntimeError("429")))
    result = agent.run_agent("q", CHUNKS, None,
                             call_model=Script([_call("read_passage", passage_id="1.0")], RuntimeError("429")))
    assert result.stop_reason.startswith("model call failed") and len(result.passages) == 1


def test_a_plain_text_reply_ends_the_loop():
    result = agent.run_agent("q", CHUNKS, None, call_model=Script([_call("read_passage", passage_id="1.1")], "Done."))
    assert result.stop_reason.startswith("model answered without calling finish")
    assert len(result.passages) == 1


def test_steps_are_published_for_the_live_trace():
    from app import progress
    steps = []
    with progress.listening(lambda *a: None, on_step=steps.append):
        agent.run_agent("q", CHUNKS, None, call_model=Script([_call("read_passage", passage_id="1.0")],
                                                             [_call("finish", passage_ids=["1.0"])]))
    assert [s["tool"] for s in steps] == ["read_passage", "finish"]
    assert steps[0]["result"] == "1 passage(s): 1.0"


# ---------- pipeline wiring ----------

def _pipeline_chunks():
    return [{**c, "source_label": c["source_label"]} for c in CHUNKS]


def test_agent_mode_replaces_the_fixed_retrieval_chain(monkeypatch):
    monkeypatch.setattr(pipeline, "AGENT_MODE", "on")
    seen = {}

    def fake_agent(question, chunks, collection, graph):
        seen["graph"] = graph
        return agent.AgentResult(passages=[chunks[2]], trace=[{"step": 1, "tool": "finish", "args": {},
                                                               "result": "x", "tokens": 10}],
                                 stop_reason="finish: ok", steps=1, tokens=10)

    monkeypatch.setattr(agent, "run_agent", fake_agent)
    decompose_calls = []
    result = _run(_pipeline_chunks(), intent="targeted_qa",
                  decompose_query=lambda *a: decompose_calls.append(1) or ["q"])
    checks = {c["name"]: c for c in result["evaluation"]["checks"]}
    assert "agent" in checks and checks["agent"]["passed"]
    assert "knowledge_graph" in checks and seen["graph"] is not None
    assert "query_decomposition" not in checks and decompose_calls == []
    assert result["retrieved_chunks"][0]["source_label"] == "varn.txt, part 1"
    assert result["agent_trace"][0]["tool"] == "finish"


def test_graph_tools_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(pipeline, "AGENT_MODE", "on")
    monkeypatch.setattr(pipeline, "AGENT_GRAPH_TOOLS", False)
    seen = {}
    monkeypatch.setattr(agent, "run_agent", lambda q, chunks, c, graph: seen.setdefault("graph", graph) or
                        agent.AgentResult(passages=[chunks[0]], steps=1))
    result = _run(_pipeline_chunks(), intent="targeted_qa")
    assert seen["graph"] is None
    assert not any(c["name"] == "knowledge_graph" for c in result["evaluation"]["checks"])


def test_a_failing_agent_falls_back_to_the_fixed_pipeline(monkeypatch):
    monkeypatch.setattr(pipeline, "AGENT_MODE", "on")

    def broken(*a, **k):
        raise RuntimeError("Groq down")

    monkeypatch.setattr(agent, "run_agent", broken)
    result = _run(_pipeline_chunks(), intent="targeted_qa")
    names = [c["name"] for c in result["evaluation"]["checks"]]
    assert "query_decomposition" in names  # the fixed chain ran
    assert any("Agent could not run" in e for e in result["evaluation"]["errors"])
    assert result["agent_trace"] is None


def test_summaries_never_use_the_agent(monkeypatch):
    monkeypatch.setattr(pipeline, "AGENT_MODE", "on")
    monkeypatch.setattr(agent, "run_agent", lambda *a, **k: pytest.fail("agent used for a summary"))
    _run(_pipeline_chunks(), intent="summary")


def test_near_duplicate_searches_are_pushed_back():
    model = Script([_call("search_documents", query="KestrelCell cycle life")],
                   [_call("search_documents", query="cycle life of KestrelCell")],
                   [_call("finish", passage_ids=["1.1"])])
    agent.run_agent("q", CHUNKS, None, call_model=model)
    second = [m for m in model.payloads[2]["messages"] if m["role"] == "tool"][-1]["content"]
    assert second.startswith("This search is nearly the same as one you already ran")


def test_searches_that_find_nothing_new_say_so():
    model = Script([_call("search_documents", query="Brekke plant")],
                   [_call("search_documents", query="recalled plant")],   # different words, same passage
                   [_call("finish", passage_ids=["3.0"])])
    agent.run_agent("q", CHUNKS, None, call_model=model)
    second = [m for m in model.payloads[2]["messages"] if m["role"] == "tool"][-1]["content"]
    assert "No new passages" in second


def test_trace_lists_passage_ids_not_numbers_from_the_text():
    chunks = [{**CHUNKS[0], "text": "The battery stores 2.0 MWh and lost 1.82 MWh capacity."}]
    model = Script([_call("read_passage", passage_id="1.0")], [_call("finish", passage_ids=["1.0"])])
    result = agent.run_agent("q", chunks, None, call_model=model)
    assert result.trace[0]["result"] == "1 passage(s): 1.0"


def test_with_a_graph_search_results_list_the_names_each_passage_mentions():
    with_graph = Script([_call("search_documents", query="KestrelCell sodium")], [_call("finish", passage_ids=["1.0"])])
    agent.run_agent("q", CHUNKS, None, graph=_graph(), call_model=with_graph)
    reply = [m for m in with_graph.payloads[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert "Mentions:" in reply and "Varn Materials" in reply

    without = Script([_call("search_documents", query="KestrelCell sodium")], [_call("finish", passage_ids=["1.0"])])
    agent.run_agent("q", CHUNKS, None, call_model=without)
    assert "Mentions:" not in [m for m in without.payloads[1]["messages"] if m["role"] == "tool"][0]["content"]


def test_at_the_step_limit_the_agent_is_made_to_choose_its_evidence(monkeypatch):
    monkeypatch.setattr(agent, "AGENT_MAX_STEPS", 2)
    model = Script([_call("search_documents", query="KestrelCell cycles")],
                   [_call("search_documents", query="VX-9 Brekke")],
                   [_call("finish", passage_ids=["3.0"], reason="forced")])
    result = agent.run_agent("q", CHUNKS, None, call_model=model)
    forced = model.payloads[2]
    assert forced["tool_choice"] == {"type": "function", "function": {"name": "finish"}}
    assert [t["function"]["name"] for t in forced["tools"]] == ["finish"]
    # Only what it chose -- not everything it saw.
    assert [c["source_label"] for c in result.passages] == ["varn.txt, part 1"]
    assert "forced" in result.trace[-1]["result"]


def test_tool_results_count_down_the_steps_left(monkeypatch):
    monkeypatch.setattr(agent, "AGENT_MAX_STEPS", 3)
    model = Script([_call("read_passage", passage_id="1.0")], [_call("read_passage", passage_id="1.1")],
                   [_call("finish", passage_ids=["1.0"])])
    agent.run_agent("q", CHUNKS, None, call_model=model)
    replies = [m["content"] for m in model.payloads[2]["messages"] if m["role"] == "tool"]
    assert replies[0].endswith("[2 step(s) left.]")
    assert replies[1].endswith("[Last step: call finish now with the passages you have.]")


def test_tool_calls_beyond_three_still_get_an_answer():
    many = [_call("read_passage", call_id=f"c{i}", passage_id="1.0") for i in range(5)]
    model = Script(many, [_call("finish", passage_ids=["1.0"])])
    agent.run_agent("q", CHUNKS, None, call_model=model)
    answered = {m["tool_call_id"] for m in model.payloads[1]["messages"] if m["role"] == "tool"}
    assert answered == {"c0", "c1", "c2", "c3", "c4"}


def test_a_failed_forced_finish_is_recorded(monkeypatch):
    monkeypatch.setattr(agent, "AGENT_MAX_STEPS", 1)
    model = Script([_call("read_passage", passage_id="1.0")], RuntimeError("400 bad request"))
    result = agent.run_agent("q", CHUNKS, None, call_model=model)
    assert "forced finish failed: 400 bad request" in result.stop_reason


def test_a_rejected_tool_call_gets_another_try():
    rejected = RuntimeError("400 -- Groq said: tool call validation failed: parameter source")
    model = Script([_call("read_passage", passage_id="1.0")], rejected,
                   [_call("finish", passage_ids=["1.0"])])
    result = agent.run_agent("q", CHUNKS, None, call_model=model)
    assert result.stop_reason.startswith("finish")
    assert "arguments did not match" in model.payloads[2]["messages"][-1]["content"]


def test_rejected_tool_calls_are_retried_at_most_twice():
    rejected = RuntimeError("tool call validation failed")
    model = Script([_call("read_passage", passage_id="1.0")], rejected, rejected, rejected,
                   [_call("finish", passage_ids=["1.0"])])
    result = agent.run_agent("q", CHUNKS, None, call_model=model)
    assert result.stop_reason.startswith("model call failed")

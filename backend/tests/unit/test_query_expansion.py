"""
Tests expand_query's fallback behavior specifically -- the one thing
the retrieval stage actually depends on: this must NEVER raise and
must ALWAYS return the original instructions as the first element,
even when Groq is unreachable or misconfigured.
"""

from app import groq_client


def test_expand_query_falls_back_without_api_key(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "")
    result = groq_client.expand_query("focus on research gaps")
    assert result == ["focus on research gaps"]


def test_expand_query_falls_back_on_api_failure(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")

    def broken_call(*args, **kwargs):
        raise RuntimeError("simulated network failure")
    monkeypatch.setattr(groq_client, "_call_groq", broken_call)

    result = groq_client.expand_query("focus on research gaps")
    # Original instructions must survive even when the expansion call itself fails
    assert result[0] == "focus on research gaps"
    assert len(result) == 1


def test_expand_query_returns_original_plus_variants_on_success(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")

    def fake_call(*args, **kwargs):
        return '{"queries": ["limitations of current methods", "future research directions", "data quality issues"]}'
    monkeypatch.setattr(groq_client, "_call_groq", fake_call)

    result = groq_client.expand_query("focus on research gaps", num_variants=3)
    assert result[0] == "focus on research gaps"
    assert len(result) == 4  # original + 3 variants
    assert "limitations of current methods" in result


def test_expand_query_handles_malformed_json_gracefully(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")
    monkeypatch.setattr(groq_client, "_call_groq", lambda *a, **k: "not valid json at all")

    result = groq_client.expand_query("focus on research gaps")
    assert result == ["focus on research gaps"]


def test_expand_query_calls_groq_with_zero_temperature(monkeypatch):
    """
    The actual determinism fix: query expansion feeds retrieval directly,
    so sampling here (the old default temperature=0.2) meant identical
    instructions could produce different expanded queries -- and
    therefore different retrieval/rerank results -- run to run.
    """
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")
    captured_kwargs = {}

    def capturing_call(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return '{"queries": ["a", "b", "c"]}'
    monkeypatch.setattr(groq_client, "_call_groq", capturing_call)

    groq_client.expand_query("focus on research gaps")
    assert captured_kwargs.get("temperature") == 0


def test_expand_query_includes_avoid_list_in_prompt(monkeypatch):
    """Self-correction retry passes previously-tried queries via `avoid` --
    confirm they actually reach the prompt, not just get silently ignored."""
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")
    captured = {}

    def capturing_call(model, system_prompt, user_prompt, **kwargs):
        captured["system_prompt"] = system_prompt
        return '{"queries": ["a", "b", "c"]}'
    monkeypatch.setattr(groq_client, "_call_groq", capturing_call)

    groq_client.expand_query("focus on research gaps", avoid=["limitations of current methods"])
    assert "limitations of current methods" in captured["system_prompt"]


def test_expand_query_without_avoid_omits_avoid_clause(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")
    captured = {}

    def capturing_call(model, system_prompt, user_prompt, **kwargs):
        captured["system_prompt"] = system_prompt
        return '{"queries": ["a", "b", "c"]}'
    monkeypatch.setattr(groq_client, "_call_groq", capturing_call)

    groq_client.expand_query("focus on research gaps")
    assert "already tried" not in captured["system_prompt"]


# ---------- decompose_query (query decomposition / compositional multi-hop) ----------

def test_decompose_query_falls_back_without_api_key(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "")
    assert groq_client.decompose_query("what dataset did they use?") == ["what dataset did they use?"]


def test_decompose_query_falls_back_on_api_failure(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")
    monkeypatch.setattr(groq_client, "_call_groq", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert groq_client.decompose_query("what dataset did they use?") == ["what dataset did they use?"]


def test_decompose_query_returns_single_entry_for_focused_question(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")
    monkeypatch.setattr(
        groq_client, "_call_groq",
        lambda *a, **k: '{"sub_questions": ["what dataset did they use?"]}',
    )
    result = groq_client.decompose_query("what dataset did they use?")
    assert result == ["what dataset did they use?"]


def test_decompose_query_splits_compound_instruction(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")
    monkeypatch.setattr(
        groq_client, "_call_groq",
        lambda *a, **k: '{"sub_questions": ["what dataset did they use?", "what are the main limitations?"]}',
    )
    result = groq_client.decompose_query("what dataset did they use, and what are the main limitations?")
    assert len(result) == 2
    assert "what dataset did they use?" in result
    assert "what are the main limitations?" in result


def test_decompose_query_caps_at_three_sub_questions(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")
    monkeypatch.setattr(
        groq_client, "_call_groq",
        lambda *a, **k: '{"sub_questions": ["a", "b", "c", "d", "e"]}',
    )
    result = groq_client.decompose_query("a compound question with many parts")
    assert len(result) == 3


def test_decompose_query_falls_back_on_empty_response(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")
    monkeypatch.setattr(groq_client, "_call_groq", lambda *a, **k: '{"sub_questions": []}')
    assert groq_client.decompose_query("what dataset did they use?") == ["what dataset did they use?"]


def test_decompose_query_calls_groq_with_zero_temperature(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")
    captured_kwargs = {}

    def capturing_call(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return '{"sub_questions": ["x"]}'
    monkeypatch.setattr(groq_client, "_call_groq", capturing_call)

    groq_client.decompose_query("what dataset did they use?")
    assert captured_kwargs.get("temperature") == 0


def test_classify_intent_falls_back_without_api_key(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "")
    assert groq_client.classify_intent("summarize this document") == "targeted_qa"


def test_classify_intent_falls_back_on_api_failure(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")

    def broken_call(*args, **kwargs):
        raise RuntimeError("simulated network failure")
    monkeypatch.setattr(groq_client, "_call_groq", broken_call)

    assert groq_client.classify_intent("summarize this document") == "targeted_qa"


def test_classify_intent_returns_summary_when_groq_says_so(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")
    monkeypatch.setattr(groq_client, "_call_groq", lambda *a, **k: '{"intent": "summary"}')
    assert groq_client.classify_intent("summarize this paper") == "summary"


def test_classify_intent_returns_analysis_when_groq_says_so(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")
    monkeypatch.setattr(groq_client, "_call_groq", lambda *a, **k: '{"intent": "analysis"}')
    assert groq_client.classify_intent("discuss the limitations of this approach") == "analysis"


def test_classify_intent_falls_back_on_unrecognized_label(monkeypatch):
    """A model that returns something outside the three known labels
    should fall back to the safe default, not propagate garbage."""
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")
    monkeypatch.setattr(groq_client, "_call_groq", lambda *a, **k: '{"intent": "something_else"}')
    assert groq_client.classify_intent("what is this about?") == "targeted_qa"


def test_classify_intent_falls_back_on_malformed_json(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")
    monkeypatch.setattr(groq_client, "_call_groq", lambda *a, **k: "not valid json")
    assert groq_client.classify_intent("what dataset did they use?") == "targeted_qa"


def test_classify_intent_calls_groq_with_zero_temperature(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")
    captured_kwargs = {}

    def capturing_call(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return '{"intent": "targeted_qa"}'
    monkeypatch.setattr(groq_client, "_call_groq", capturing_call)

    groq_client.classify_intent("what dataset did they use?")
    assert captured_kwargs.get("temperature") == 0


def test_filter_relevant_pages_calls_groq_with_zero_temperature(monkeypatch):
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake_key_for_test")
    captured_kwargs = {}

    def capturing_call(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return '{"decisions": []}'
    monkeypatch.setattr(groq_client, "_call_groq", capturing_call)

    groq_client.filter_relevant_pages(
        [{"url": "https://x.com", "title": "T", "snippet": "S", "depth": 0}],
        topic="test", instructions="test",
    )
    assert captured_kwargs.get("temperature") == 0

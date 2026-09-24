"""
Tests expand_query's fallback behavior specifically -- the one thing
the retrieval stage actually depends on: this must NEVER raise and
must ALWAYS return the original instructions as the first element,
even when Groq is unreachable or misconfigured.
"""

import os
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

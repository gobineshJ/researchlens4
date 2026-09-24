"""
Tests reranker.py's logic without needing to actually download the
cross-encoder model -- the model itself is mocked, so these test the
sorting/truncation/fallback behavior, which is what the pipeline
actually depends on being correct.
"""

import pytest
from app import reranker


class _FakeModel:
    """Mimics CrossEncoder.predict: returns a score per (query, text) pair."""
    def __init__(self, score_map):
        self.score_map = score_map  # {text: score}

    def predict(self, pairs):
        return [self.score_map.get(text, 0.0) for _, text in pairs]


def test_rerank_orders_by_score_descending(monkeypatch):
    candidates = [
        {"text": "low relevance chunk", "source_label": "a"},
        {"text": "high relevance chunk", "source_label": "b"},
        {"text": "medium relevance chunk", "source_label": "c"},
    ]
    fake = _FakeModel({"low relevance chunk": -2.0, "high relevance chunk": 5.0, "medium relevance chunk": 1.0})
    monkeypatch.setattr(reranker, "_get_reranker", lambda: fake)

    result = reranker.rerank("some query", candidates, top_k=3)
    assert [r["source_label"] for r in result] == ["b", "c", "a"]
    assert result[0]["rerank_score"] == 5.0


def test_rerank_truncates_to_top_k(monkeypatch):
    candidates = [{"text": f"chunk{i}", "source_label": str(i)} for i in range(5)]
    fake = _FakeModel({f"chunk{i}": float(i) for i in range(5)})
    monkeypatch.setattr(reranker, "_get_reranker", lambda: fake)

    result = reranker.rerank("query", candidates, top_k=2)
    assert len(result) == 2
    # Highest scores (chunk4=4.0, chunk3=3.0) should be kept
    assert {r["source_label"] for r in result} == {"3", "4"}


def test_rerank_preserves_existing_fields(monkeypatch):
    candidates = [{"text": "x", "source_label": "a", "source_name": "doc.pdf", "distance": 0.4}]
    fake = _FakeModel({"x": 1.0})
    monkeypatch.setattr(reranker, "_get_reranker", lambda: fake)

    result = reranker.rerank("query", candidates, top_k=1)
    assert result[0]["source_name"] == "doc.pdf"
    assert result[0]["distance"] == 0.4
    assert result[0]["rerank_score"] == 1.0


def test_rerank_with_fallback_on_empty_candidates():
    assert reranker.rerank_with_fallback("query", [], top_k=5) == []


def test_rerank_with_fallback_degrades_to_distance_sort_on_model_failure(monkeypatch):
    """
    Simulates the reranker model failing to load (e.g. no network) --
    should fall back to sorting by embedding distance instead of
    crashing the whole pipeline.
    """
    def broken_reranker():
        raise RuntimeError("simulated: could not reach huggingface.co")
    monkeypatch.setattr(reranker, "_get_reranker", broken_reranker)

    candidates = [
        {"text": "a", "source_label": "a", "distance": 0.9},
        {"text": "b", "source_label": "b", "distance": 0.2},
        {"text": "c", "source_label": "c", "distance": 0.5},
    ]
    result = reranker.rerank_with_fallback("query", candidates, top_k=2)
    assert len(result) == 2
    # Should keep the two closest by distance (b=0.2, c=0.5), not a=0.9
    assert {r["source_label"] for r in result} == {"b", "c"}
    # Fallback path should NOT add a rerank_score -- evaluation.py uses
    # its absence to know reranking degraded for this run.
    assert all("rerank_score" not in r for r in result)


def test_rerank_with_fallback_uses_real_rerank_when_model_works(monkeypatch):
    candidates = [{"text": "x", "source_label": "a", "distance": 0.5}]
    fake = _FakeModel({"x": 3.0})
    monkeypatch.setattr(reranker, "_get_reranker", lambda: fake)

    result = reranker.rerank_with_fallback("query", candidates, top_k=1)
    assert result[0]["rerank_score"] == 3.0

"""
Tests the FastAPI app's routes directly via TestClient. Only the parts
that don't need a real Groq key or live embeddings -- validation logic,
routing, error shapes -- /process's actual pipeline execution is what
tests/integration/ covers with real credentials.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_process_rejects_empty_input():
    response = client.post("/process", data={"seed_urls": "", "instructions": "summarize"})
    assert response.status_code == 400
    assert "Provide at least one file" in response.json()["detail"]


def test_download_404_for_missing_report():
    response = client.get("/download/does_not_exist.pdf")
    assert response.status_code == 404


def test_evaluation_history_returns_expected_shape(monkeypatch):
    """
    Monkeypatches the underlying read so this test is isolated from
    whatever real evaluation_log.jsonl might exist on disk from actual
    usage -- otherwise this test's result would depend on unrelated
    prior runs, which is exactly the kind of flakiness a test suite
    should never have.
    """
    from app import main as main_module
    fake_runs = [{"run_id": "abc", "summary": "5/5 checks passed", "checks": []}]
    monkeypatch.setattr(main_module.evaluation, "read_run_history", lambda limit=50: fake_runs)

    response = client.get("/evaluation-history")
    assert response.status_code == 200
    assert response.json() == {"runs": fake_runs}


def test_evaluation_history_respects_limit_param(monkeypatch):
    from app import main as main_module
    captured = {}
    def fake_read(limit=50):
        captured["limit"] = limit
        return []
    monkeypatch.setattr(main_module.evaluation, "read_run_history", fake_read)

    client.get("/evaluation-history?limit=5")
    assert captured["limit"] == 5

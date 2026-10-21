"""
Tests the FastAPI app's routes directly via TestClient. Only the parts
that don't need a real Groq key, live embeddings or a running Celery
worker -- validation logic, routing, auth, error shapes. Enqueuing is
checked by capturing the apply_async call instead of running the task.
"""

import pytest
from fastapi.testclient import TestClient

from app import db
from app.auth import create_access_token, hash_password
from app.main import app

client = TestClient(app)


def _make_user(username: str) -> dict:
    db.create_user(username, hash_password("correct-horse"))
    return {"Authorization": f"Bearer {create_access_token(username)}"}


@pytest.fixture
def auth_headers():
    return _make_user("alice")


@pytest.fixture
def captured_jobs(monkeypatch):
    """Replaces enqueuing with a recorder, so /process never needs Redis."""
    from app import main as main_module
    calls = []
    monkeypatch.setattr(main_module.run_pipeline_task, "apply_async",
                        lambda args, task_id, kwargs=None:
                        calls.append({"args": args, "kwargs": kwargs, "task_id": task_id}))
    return calls


class _FakeResult:
    def __init__(self, state, result=None):
        self.state = state
        self.result = result
        self.info = result  # Celery exposes the same value as .info


def _fake_async_result(monkeypatch, state, result=None):
    from app import main as main_module
    monkeypatch.setattr(main_module, "AsyncResult", lambda job_id, app: _FakeResult(state, result))


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("method,path", [
    ("post", "/process"),
    ("get", "/jobs/some-id"),
    ("get", "/download/report.pdf"),
    ("get", "/evaluation-history"),
    ("get", "/auth/me"),
])
def test_protected_routes_reject_missing_token(method, path):
    response = getattr(client, method)(path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_protected_routes_reject_garbage_token():
    response = client.get("/evaluation-history", headers={"Authorization": "Bearer not-a-jwt"})
    assert response.status_code == 401


def test_process_rejects_empty_input(auth_headers):
    response = client.post("/process", data={"seed_urls": "", "instructions": "summarize"},
                           headers=auth_headers)
    assert response.status_code == 400
    assert "Provide at least one file" in response.json()["detail"]


def test_process_enqueues_job_and_records_owner(auth_headers, captured_jobs):
    response = client.post(
        "/process",
        data={"seed_urls": "https://a.example\n\n https://b.example ", "instructions": "summarize"},
        files=[("files", ("paper.txt", b"some text", "text/plain"))],
        headers=auth_headers,
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert response.json()["status"] == "queued"

    assert len(captured_jobs) == 1
    job_dir, file_paths, urls, instructions, username = captured_jobs[0]["args"]
    assert captured_jobs[0]["task_id"] == job_id
    assert urls == ["https://a.example", "https://b.example"]
    assert instructions == "summarize"
    assert username == "alice"
    assert open(file_paths[0], "rb").read() == b"some text"
    assert db.get_job_owner(job_id) == "alice"


def test_process_strips_directory_from_upload_filename(auth_headers, captured_jobs):
    from pathlib import Path
    client.post("/process", files=[("files", ("../../evil.txt", b"x", "text/plain"))],
                headers=auth_headers)
    job_dir, file_paths, *_ = captured_jobs[0]["args"]
    assert Path(file_paths[0]).parent == Path(job_dir)
    assert Path(file_paths[0]).name == "evil.txt"


def test_job_status_hidden_from_other_users(auth_headers, captured_jobs, monkeypatch):
    _fake_async_result(monkeypatch, "PENDING")
    job_id = client.post("/process", data={"seed_urls": "https://a.example"},
                         headers=auth_headers).json()["job_id"]

    assert client.get(f"/jobs/{job_id}", headers=auth_headers).status_code == 200
    bob = _make_user("bob")
    assert client.get(f"/jobs/{job_id}", headers=bob).status_code == 404


def test_job_status_done_returns_result_and_unlocks_download(auth_headers, captured_jobs, monkeypatch, tmp_path):
    from app import config
    job_id = client.post("/process", data={"seed_urls": "https://a.example"},
                         headers=auth_headers).json()["job_id"]

    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    (tmp_path / "report_abc.pdf").write_bytes(b"%PDF-1.4 fake")
    assert client.get("/download/report_abc.pdf", headers=auth_headers).status_code == 404

    _fake_async_result(monkeypatch, "SUCCESS", {"report_text": "hi", "pdf_filename": "report_abc.pdf"})
    body = client.get(f"/jobs/{job_id}", headers=auth_headers).json()
    assert body["status"] == "done"
    assert body["result"]["report_text"] == "hi"

    response = client.get("/download/report_abc.pdf", headers=auth_headers)
    assert response.status_code == 200
    assert response.content == b"%PDF-1.4 fake"
    # Another user still can't fetch it, even knowing the filename.
    assert client.get("/download/report_abc.pdf", headers=_make_user("bob")).status_code == 404


def test_job_status_error_surfaces_exception(auth_headers, captured_jobs, monkeypatch):
    job_id = client.post("/process", data={"seed_urls": "https://a.example"},
                         headers=auth_headers).json()["job_id"]
    _fake_async_result(monkeypatch, "FAILURE", ValueError("bad url"))
    body = client.get(f"/jobs/{job_id}", headers=auth_headers).json()
    assert body == {"job_id": job_id, "status": "error", "error": "ValueError: bad url"}


def test_download_404_for_missing_report(auth_headers):
    response = client.get("/download/does_not_exist.pdf", headers=auth_headers)
    assert response.status_code == 404


def test_evaluation_history_returns_expected_shape(monkeypatch, auth_headers):
    """
    Monkeypatches the underlying read so this test is isolated from
    whatever real evaluation_log.jsonl might exist on disk from actual
    usage -- otherwise this test's result would depend on unrelated
    prior runs, which is exactly the kind of flakiness a test suite
    should never have.
    """
    from app import main as main_module
    fake_runs = [{"run_id": "abc", "summary": "5/5 checks passed", "checks": []}]
    monkeypatch.setattr(main_module.evaluation, "read_run_history", lambda limit=50, username=None: fake_runs)

    response = client.get("/evaluation-history", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == {"runs": fake_runs}


def test_evaluation_history_respects_limit_param(monkeypatch, auth_headers):
    from app import main as main_module
    captured = {}
    def fake_read(limit=50, username=None):
        captured["limit"] = limit
        return []
    monkeypatch.setattr(main_module.evaluation, "read_run_history", fake_read)

    client.get("/evaluation-history?limit=5", headers=auth_headers)
    assert captured["limit"] == 5


def test_job_status_running_includes_progress(auth_headers, captured_jobs, monkeypatch):
    job_id = client.post("/process", data={"seed_urls": "https://a.example"},
                         headers=auth_headers).json()["job_id"]
    fake = _FakeResult("PROGRESS", {"stage": "extraction", "detail": "Describing Figure 3 of a.pdf (page 7)"})
    from app import main as main_module
    monkeypatch.setattr(main_module, "AsyncResult", lambda job_id, app: fake)
    body = client.get(f"/jobs/{job_id}", headers=auth_headers).json()
    assert body["status"] == "running"
    assert body["progress"] == {"stage": "extraction", "detail": "Describing Figure 3 of a.pdf (page 7)",
                                "draft": ""}


def test_evaluation_history_only_shows_the_callers_runs(auth_headers):
    """The real log, not a monkeypatched read: runs logged for one user
    are invisible to another, and legacy entries without a user to anyone."""
    from app import evaluation
    for user in ("alice", "bob", None):
        evaluation.append_run_to_log(evaluation.EvaluationReport(), run_id=f"run-{user}",
                                     instructions=f"{user}'s private topic", username=user)

    alice_runs = client.get("/evaluation-history", headers=auth_headers).json()["runs"]
    assert [r["run_id"] for r in alice_runs] == ["run-alice"]

    bob_runs = client.get("/evaluation-history", headers=_make_user("bob")).json()["runs"]
    assert [r["run_id"] for r in bob_runs] == ["run-bob"]

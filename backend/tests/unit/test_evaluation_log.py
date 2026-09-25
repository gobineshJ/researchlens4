"""Tests append_run_to_log — the mechanism for real before/after comparisons."""

import json
from app import evaluation


def test_append_run_to_log_writes_valid_json_line(tmp_path):
    log_path = tmp_path / "evaluation_log.jsonl"
    report = evaluation.EvaluationReport()
    report.add_check("extraction", "completeness", True, "all good")
    report.add_check("retrieval", "confidence", False, "too weak")

    evaluation.append_run_to_log(report, run_id="test-run-1", log_path=str(log_path))

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["run_id"] == "test-run-1"
    assert entry["summary"] == "1/2 checks passed"
    assert len(entry["checks"]) == 2
    assert "timestamp" in entry


def test_append_run_to_log_appends_not_overwrites(tmp_path):
    log_path = tmp_path / "evaluation_log.jsonl"
    report = evaluation.EvaluationReport()
    report.add_check("extraction", "completeness", True, "ok")

    evaluation.append_run_to_log(report, run_id="run-1", log_path=str(log_path))
    evaluation.append_run_to_log(report, run_id="run-2", log_path=str(log_path))

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    run_ids = [json.loads(line)["run_id"] for line in lines]
    assert run_ids == ["run-1", "run-2"]


def test_append_run_to_log_creates_parent_directory(tmp_path):
    log_path = tmp_path / "nested" / "dir" / "evaluation_log.jsonl"
    report = evaluation.EvaluationReport()
    evaluation.append_run_to_log(report, log_path=str(log_path))
    assert log_path.exists()


def test_append_run_to_log_never_raises_on_bad_path():
    """
    An unwritable path (e.g. permissions issue) should never break the
    actual pipeline response -- logging failures are swallowed silently.
    """
    report = evaluation.EvaluationReport()
    # A path under a location that can't exist as a directory (a file
    # component in the middle of the path) -- forces a real write failure.
    bad_path = "/dev/null/impossible/path.jsonl"
    evaluation.append_run_to_log(report, log_path=bad_path)  # should not raise

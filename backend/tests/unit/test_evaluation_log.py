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


def test_append_run_to_log_includes_numeric_value_and_instructions(tmp_path):
    log_path = tmp_path / "log.jsonl"
    report = evaluation.EvaluationReport()
    report.add_check("retrieval", "confidence", False, "average was low", value=-5.97)

    evaluation.append_run_to_log(report, run_id="r1", instructions="summary", log_path=str(log_path))

    entry = json.loads(log_path.read_text().strip())
    assert entry["instructions"] == "summary"
    assert entry["checks"][0]["value"] == -5.97


def test_read_run_history_returns_newest_first(tmp_path):
    log_path = tmp_path / "log.jsonl"
    report = evaluation.EvaluationReport()
    evaluation.append_run_to_log(report, run_id="first", log_path=str(log_path))
    evaluation.append_run_to_log(report, run_id="second", log_path=str(log_path))

    history = evaluation.read_run_history(log_path=str(log_path))
    assert [h["run_id"] for h in history] == ["second", "first"]


def test_read_run_history_respects_limit(tmp_path):
    log_path = tmp_path / "log.jsonl"
    report = evaluation.EvaluationReport()
    for i in range(5):
        evaluation.append_run_to_log(report, run_id=f"run-{i}", log_path=str(log_path))

    history = evaluation.read_run_history(log_path=str(log_path), limit=2)
    assert len(history) == 2
    assert history[0]["run_id"] == "run-4"  # newest first


def test_read_run_history_returns_empty_list_when_no_log_exists(tmp_path):
    missing_path = tmp_path / "does_not_exist.jsonl"
    assert evaluation.read_run_history(log_path=str(missing_path)) == []


def test_read_run_history_skips_corrupted_lines(tmp_path):
    log_path = tmp_path / "log.jsonl"
    log_path.write_text('{"run_id": "good", "checks": []}\nnot valid json\n')
    history = evaluation.read_run_history(log_path=str(log_path))
    assert len(history) == 1
    assert history[0]["run_id"] == "good"


def test_read_run_history_returns_empty_list_when_log_missing(tmp_path):
    missing = tmp_path / "never_written.jsonl"
    assert evaluation.read_run_history(log_path=str(missing)) == []


def test_read_run_history_returns_newest_first(tmp_path):
    log_path = tmp_path / "log.jsonl"
    r1 = evaluation.EvaluationReport()
    evaluation.append_run_to_log(r1, run_id="first", log_path=str(log_path))
    r2 = evaluation.EvaluationReport()
    evaluation.append_run_to_log(r2, run_id="second", log_path=str(log_path))

    history = evaluation.read_run_history(log_path=str(log_path))
    assert [h["run_id"] for h in history] == ["second", "first"]


def test_read_run_history_respects_limit(tmp_path):
    log_path = tmp_path / "log.jsonl"
    for i in range(5):
        r = evaluation.EvaluationReport()
        evaluation.append_run_to_log(r, run_id=f"run-{i}", log_path=str(log_path))

    history = evaluation.read_run_history(log_path=str(log_path), limit=2)
    assert len(history) == 2
    assert history[0]["run_id"] == "run-4"  # newest


def test_read_run_history_skips_corrupted_lines(tmp_path):
    log_path = tmp_path / "log.jsonl"
    r = evaluation.EvaluationReport()
    evaluation.append_run_to_log(r, run_id="good-run", log_path=str(log_path))
    with open(log_path, "a") as f:
        f.write("this is not valid json\n")

    history = evaluation.read_run_history(log_path=str(log_path))
    assert len(history) == 1
    assert history[0]["run_id"] == "good-run"

"""
Regression tests for bugs the first full evaluation run exposed:

  * the streamed report was decoded as ISO-8859-1 (the event stream
    declares no charset), garbling every non-ASCII character;
  * gpt-oss sometimes cites as 【2】 / 【2†L10-L12】 / ［2］, which nothing
    downstream recognised;
  * the answer grader read citation markers as wrong numbers;
  * questions whose Groq calls hit the daily limit were averaged in.
"""

import json

import pytest

from app import groq_client
from eval import metrics, run_eval


# ---------- streaming ----------

class _StreamResponse:
    def __init__(self, lines: list[bytes]):
        self._lines = lines

    def iter_lines(self, decode_unicode=False):
        assert not decode_unicode, "must decode UTF-8 itself, not trust requests' charset guess"
        return iter(self._lines)


def _sse(content: str) -> bytes:
    return ("data: " + json.dumps({"choices": [{"delta": {"content": content}}]}, ensure_ascii=False)).encode("utf-8")


def test_stream_is_decoded_as_utf8():
    seen = []
    response = _StreamResponse([
        _sse("ClinBERT‑L beat it by 4.2 "),
        b"",
        b'data: {"choices": [{"delta": {"reasoning": "thinking..."}}]}',
        _sse("F1 【2】."),
        b"data: [DONE]",
        _sse("never read"),
    ])
    text = groq_client._read_stream(response, seen.append)
    assert text == "ClinBERT‑L beat it by 4.2 F1 【2】."
    assert seen[-1] == text


# ---------- citation normalisation ----------

@pytest.mark.parametrize("raw,expected", [
    ("by 4.2 F1 【2】.", "by 4.2 F1 [2]."),
    ("see 【1, 3】 and 【2†L10-L12】", "see [1, 3] and [2]"),
    ("full-width ［4］", "full-width [4]"),
    ("already fine [5] and [2-4]", "already fine [5] and [2-4]"),
])
def test_foreign_citation_brackets_become_ascii(raw, expected):
    assert groq_client.normalize_citations(raw) == expected


def test_generate_report_normalises_citations_before_reading_them(monkeypatch):
    monkeypatch.setattr(groq_client, "_post_chat",
                        lambda payload, timeout=30, on_text=None: "TerraNotes was used 【2】.")
    chunks = [{"text": "a", "source_label": "one"}, {"text": "b", "source_label": "two"}]
    result = groq_client.generate_report(chunks, "what corpus?")
    assert result == {"report_text": "TerraNotes was used [2].", "citations_used": ["two"]}


# ---------- answer grading ----------

def test_grader_never_sees_citation_markers(monkeypatch):
    sent = {}

    def fake_call(model, system_prompt, user_prompt, **kwargs):
        sent["prompt"] = user_prompt
        return json.dumps({"grade": "correct", "reason": "ok"})

    monkeypatch.setattr(groq_client, "_call_groq", fake_call)
    score, _ = run_eval.judge_answer("By how much?", "4.2 F1 points.", "It won by 4.2 F1 points [2]. Also [1, 3].")
    assert score == 1.0
    assert "[2]" not in sent["prompt"] and "4.2 F1 points. Also." in sent["prompt"]


# ---------- incomplete questions ----------

def _fake_pipeline_with(errors_by_question):
    def fake(file_paths, seed_urls, instructions, username=None, run_id=None):
        return {"retrieved_chunks": [], "report_text": "An answer.",
                "evaluation": {"checks": [], "errors": errors_by_question.get(run_id, [])}}
    return fake


QUESTIONS = metrics.load_jsonl(metrics.GOLDEN_DIR / "questions.jsonl")[:4]


def test_run_stops_at_the_daily_limit_and_marks_the_rest_not_run(monkeypatch):
    from app import pipeline
    limit = ["[entailment] Entailment check could not run: 429 Client Error: Too Many Requests"]
    monkeypatch.setattr(pipeline, "run_pipeline", _fake_pipeline_with({f"eval-{QUESTIONS[1]['id']}": limit}))
    rows = run_eval.run_questions(QUESTIONS, metrics.GOLDEN_DIR / "docs", grade=False)
    assert len(rows) == 4
    assert "incomplete" not in rows[0]
    assert "429" in rows[1]["incomplete"]
    assert all(r["incomplete"].startswith("not run") for r in rows[2:])


def test_incomplete_questions_stay_out_of_the_averages(tmp_path, capsys):
    rows = [{"id": "a", "type": "qa", "fact_recall": 1.0},
            {"id": "b", "type": "qa", "fact_recall": 0.0, "incomplete": "429"}]
    summary = run_eval.save_results(rows, {}, False, tmp_path / "r.json")
    assert summary["ALL"]["n"] == 1 and summary["ALL"]["fact_recall"] == 1.0
    assert "--retry-incomplete" in capsys.readouterr().out


def test_compare_warns_when_runs_averaged_different_questions(tmp_path, capsys):
    def save(name, questions):
        path = tmp_path / name
        path.write_text(json.dumps({"overrides": {}, "summary": {"ALL": {"n": 1}}, "questions": questions}))
        return str(path)

    full = [{"id": "a"}, {"id": "b"}]
    partial = [{"id": "a"}, {"id": "b", "incomplete": "429"}]
    run_eval.compare(save("a.json", full), save("b.json", partial))
    assert "WARNING: the runs averaged different questions (2 vs 1 complete)" in capsys.readouterr().out


def test_retry_incomplete_reruns_only_those_questions_with_the_same_settings(monkeypatch, tmp_path):
    from app import config, pipeline
    previous = tmp_path / "20261020-110320_no-multihop.json"
    previous.write_text(json.dumps({
        "overrides": {"MULTI_HOP_ENABLED": False}, "judge_answers": False, "summary": {},
        "questions": [{"id": QUESTIONS[0]["id"], "type": QUESTIONS[0]["type"], "fact_recall": 1.0},
                      {"id": QUESTIONS[1]["id"], "type": QUESTIONS[1]["type"], "incomplete": "429"}],
    }))
    ran = []

    def fake(file_paths, seed_urls, instructions, username=None, run_id=None):
        ran.append((run_id, pipeline.MULTI_HOP_ENABLED))
        return {"retrieved_chunks": [], "report_text": "x", "evaluation": {"checks": [], "errors": []}}

    monkeypatch.setattr(pipeline, "run_pipeline", fake)
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path / "out")
    monkeypatch.setattr(pipeline, "MULTI_HOP_ENABLED", True)
    # main() applies the file's overrides to every module; register each for restore.
    monkeypatch.setattr(config, "MULTI_HOP_ENABLED", config.MULTI_HOP_ENABLED)
    run_eval.main(["--retry-incomplete", str(previous)])

    assert ran == [(f"eval-{QUESTIONS[1]['id']}", False)]
    (merged_path,) = (tmp_path / "out").glob("*_no-multihop-retry.json")
    merged = json.loads(merged_path.read_text())
    assert [r["id"] for r in merged["questions"]] == [QUESTIONS[0]["id"], QUESTIONS[1]["id"]]
    assert "incomplete" not in merged["questions"][1]
    assert merged["overrides"] == {"MULTI_HOP_ENABLED": False}

"""
The entailment check: cited sentences are pulled out of the report, an
LLM judge (faked here) says whether each one is supported by the excerpt
it cites, and the check fails on a low support rate or any contradiction.
"""

import json

import pytest

from app import evaluation, groq_client


# ---------- claim extraction ----------

def _strip(claims):
    return [{"text": c["text"], "sources": c["sources"]} for c in claims]


def test_claims_keep_their_markers_including_ones_after_the_full_stop():
    report = ("The model reaches 91% accuracy on MIMIC [1]. It was trained on 40k notes. [2] "
              "Results generalise across sites [1, 3].")
    claims = evaluation.extract_cited_claims(report, num_sources=3)
    # The marker after "notes." belongs to that sentence's span.
    assert report[slice(*claims[1]["span"])] == "It was trained on 40k notes. [2]"
    assert _strip(claims) == [
        {"text": "The model reaches 91% accuracy on MIMIC.", "sources": [1]},
        {"text": "It was trained on 40k notes.", "sources": [2]},
        {"text": "Results generalise across sites.", "sources": [1, 3]},
    ]


def test_ranges_markdown_and_out_of_range_markers():
    report = ("## Findings\n- Three studies agree on the effect [2-4].\n"
              "* A ninth source is cited here as well [9].\nUncited sentence with no marker at all.")
    claims = evaluation.extract_cited_claims(report, num_sources=4)
    assert _strip(claims) == [{"text": "Three studies agree on the effect.", "sources": [2, 3, 4]}]


def test_short_fragments_and_duplicates_are_skipped():
    report = "Yes [1]. The effect is large and robust [1]. The effect is large and robust [2]."
    claims = evaluation.extract_cited_claims(report, num_sources=2)
    assert [c["text"] for c in claims] == ["The effect is large and robust."]


def test_long_reports_are_sampled_evenly():
    report = " ".join(f"Claim number {i} says something specific [1]." for i in range(50))
    claims = evaluation.extract_cited_claims(report, num_sources=1, max_claims=5)
    assert [c["text"].split()[2] for c in claims] == ["0", "10", "20", "30", "40"]


# ---------- the check itself ----------

def _verdicts(*verdicts):
    return [{"claim": f"claim {i}", "sources": [1], "verdict": v, "reason": f"why {i}"}
            for i, v in enumerate(verdicts)]


def _check(report):
    return next(c for c in report.checks if c["name"] == "entailment")


def test_mostly_supported_passes():
    report = evaluation.EvaluationReport()
    verdicts = _verdicts("supported", "supported", "supported", "unsupported")
    evaluation.validate_entailment([{}] * 4, verdicts, report)
    check = _check(report)
    assert check["passed"] and check["value"] == 75.0
    assert "UNSUPPORTED: \"claim 3\" [1] -- why 3" in check["detail"]


def test_low_support_rate_fails():
    report = evaluation.EvaluationReport()
    evaluation.validate_entailment([{}] * 3, _verdicts("supported", "unsupported", "unsupported"), report)
    assert not _check(report)["passed"]


def test_any_contradiction_fails_even_at_a_high_rate():
    report = evaluation.EvaluationReport()
    verdicts = _verdicts(*(["supported"] * 9 + ["contradicted"]))
    evaluation.validate_entailment([{}] * 10, verdicts, report)
    check = _check(report)
    assert not check["passed"]
    assert check["detail"].index("CONTRADICTED") < len(check["detail"])


def test_unjudged_claims_count_toward_neither_side():
    report = evaluation.EvaluationReport()
    evaluation.validate_entailment([{}] * 3, _verdicts("supported", "unjudged", "unjudged"), report)
    check = _check(report)
    assert check["passed"] and check["value"] == 100.0
    assert "2 not judged" in check["detail"]


def test_report_without_cited_claims_fails():
    report = evaluation.EvaluationReport()
    evaluation.validate_entailment([], [], report)
    assert not _check(report)["passed"]


# ---------- the judge (Groq faked) ----------

def _chunks(*token_sizes):
    return [{"text": f"excerpt{i} " + "word " * (4 * size // 5), "source_label": f"s{i}"}
            for i, size in enumerate(token_sizes)]


@pytest.fixture
def fake_judge(monkeypatch):
    """Records each request and answers with verdicts from `answers`."""
    calls, answers = [], []

    def fake_call(model, system_prompt, user_prompt, json_mode=True, temperature=0.2, **options):
        calls.append(user_prompt)
        return json.dumps(answers.pop(0))

    monkeypatch.setattr(groq_client, "_call_groq", fake_call)
    return calls, answers


def test_judge_maps_verdicts_back_to_claims(fake_judge):
    calls, answers = fake_judge
    answers.append({"verdicts": [{"claim": 2, "verdict": "Contradicted", "reason": "says 4%"},
                                 {"claim": 1, "verdict": "supported"},
                                 {"claim": 7, "verdict": "supported"}]})
    claims = [{"text": "A holds.", "sources": [1]}, {"text": "B is 40%.", "sources": [2]}]
    result = groq_client.judge_claims(claims, _chunks(100, 100))
    assert [r["verdict"] for r in result] == ["supported", "contradicted"]
    assert result[1]["reason"] == "says 4%"
    assert len(calls) == 1
    assert "(cites [2]) B is 40%." in calls[0] and "[1] excerpt0" in calls[0]


def test_judge_marks_skipped_claims_unjudged(fake_judge):
    _, answers = fake_judge
    answers.append({"verdicts": [{"claim": 1, "verdict": "maybe"}]})
    result = groq_client.judge_claims([{"text": "A holds.", "sources": [1]}], _chunks(100))
    assert result[0]["verdict"] == "unjudged"


def test_claims_are_batched_to_fit_the_excerpt_budget(fake_judge):
    calls, answers = fake_judge
    answers.extend([{"verdicts": [{"claim": 1, "verdict": "supported"},
                                  {"claim": 2, "verdict": "supported"}]},
                    {"verdicts": [{"claim": 1, "verdict": "unsupported"}]}])
    claims = [{"text": "A.", "sources": [1]}, {"text": "A again.", "sources": [1]},
              {"text": "B.", "sources": [2]}]
    result = groq_client.judge_claims(claims, _chunks(600, 600), max_tokens=1000)
    assert len(calls) == 2
    assert "excerpt1" not in calls[0] and "excerpt0" not in calls[1]
    assert [r["verdict"] for r in result] == ["supported", "supported", "unsupported"]


def test_judge_errors_propagate(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("Groq down")
    monkeypatch.setattr(groq_client, "_call_groq", boom)
    with pytest.raises(RuntimeError):
        groq_client.judge_claims([{"text": "A holds.", "sources": [1]}], _chunks(100))


# ---------- pipeline wiring ----------

def test_pipeline_judges_claims_against_the_chunks_the_markers_index():
    from tests.unit.test_pipeline import _chunks_for, _run
    chunks = _chunks_for("doc.pdf", 4, "transformers")
    seen = {}

    def judge(claims, retrieved, **k):
        seen["claims"], seen["retrieved"] = claims, retrieved
        return [{"claim": c["text"], "sources": c["sources"], "verdict": "contradicted", "reason": "no"}
                for c in claims]

    result = _run(chunks, intent="targeted_qa", judge_claims=judge,
                  generate_report={"report_text": "Transformers use attention heavily [2].",
                                   "citations_used": [chunks[1]["source_label"]]})
    assert _strip(seen["claims"]) == [{"text": "Transformers use attention heavily.", "sources": [2]}]
    assert seen["retrieved"] == result["retrieved_chunks"]
    check = next(c for c in result["evaluation"]["checks"] if c["name"] == "entailment")
    assert not check["passed"]
    assert result["claim_verdicts"][0]["verdict"] == "contradicted"


def test_pipeline_records_an_error_when_the_judge_cannot_run():
    from tests.unit.test_pipeline import _chunks_for, _run

    def judge(*a, **k):
        raise RuntimeError("daily limit reached")

    chunks = _chunks_for("doc.pdf", 4, "transformers")
    result = _run(chunks, intent="targeted_qa", judge_claims=judge,
                  generate_report={"report_text": "Transformers use attention heavily [1].",
                                   "citations_used": [chunks[0]["source_label"]]})
    assert not any(c["name"] == "entailment" for c in result["evaluation"]["checks"])
    assert result["evaluation"]["error_count"] == 1
    assert "daily limit reached" in result["evaluation"]["errors"][0]
    assert result["report_text"]  # the report itself still comes back


def test_abbreviations_do_not_split_a_claim():
    # Generated reports often put a narrow no-break space inside "et al."
    report = "The review by Rahmani et\u202fal. was published in 2021 [1]. Results differ, e.g. on MIMIC [2]."
    claims = evaluation.extract_cited_claims(report, num_sources=2)
    assert [c["text"] for c in claims] == [
        "The review by Rahmani et al. was published in 2021.",  # whitespace normalised
        "Results differ, e.g. on MIMIC.",
    ]


def test_judge_uses_low_reasoning_effort_and_an_explicit_budget(monkeypatch):
    sent = {}

    def fake_call(model, system_prompt, user_prompt, json_mode=True, temperature=0.2, **options):
        sent.update(options)
        return json.dumps({"verdicts": [{"claim": 1, "verdict": "supported"}]})

    monkeypatch.setattr(groq_client, "_call_groq", fake_call)
    groq_client.judge_claims([{"text": "A holds.", "sources": [1]}], _chunks(100))
    assert sent == {"reasoning_effort": "low",
                    "max_completion_tokens": groq_client.ENTAILMENT_MAX_COMPLETION_TOKENS}


def test_json_generation_failure_splits_the_batch(monkeypatch):
    """Groq's 400 json_validate_failed (reasoning ran out the completion
    budget) on a batch: each half is judged on its own instead."""
    calls = []

    def fake_call(model, system_prompt, user_prompt, json_mode=True, temperature=0.2, **options):
        n_claims = user_prompt.count("(cites")
        calls.append(n_claims)
        if n_claims > 1:
            raise groq_client.GroqJSONGenerationFailed("json_validate_failed")
        return json.dumps({"verdicts": [{"claim": 1, "verdict": "supported"}]})

    monkeypatch.setattr(groq_client, "_call_groq", fake_call)
    claims = [{"text": f"Claim {i} holds.", "sources": [1]} for i in range(3)]
    result = groq_client.judge_claims(claims, _chunks(100))
    assert [r["verdict"] for r in result] == ["supported"] * 3
    assert calls == [3, 1, 2, 1, 1]


def test_single_claim_that_cannot_be_judged_stays_unjudged(monkeypatch):
    def fail(*a, **k):
        raise groq_client.GroqJSONGenerationFailed("json_validate_failed")
    monkeypatch.setattr(groq_client, "_call_groq", fail)
    result = groq_client.judge_claims([{"text": "A holds.", "sources": [1]}], _chunks(100))
    assert result[0]["verdict"] == "unjudged"

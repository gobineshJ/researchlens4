"""
Targeted revision: sentences the entailment judge flags are rewritten
against their own cited excerpts (or removed), spliced back into the
report in place, and judged again. Groq is faked throughout.
"""

import json

from app import evaluation, groq_client, pipeline
from tests.unit.test_pipeline import _chunks_for, _run

REPORT = ("Transformers rely on attention [1]. They were invented in 1990 and solved vision [2]. "
          "Some say they are magic. [3] Training is costly [1].")


# ---------- splicing ----------

def _claims():
    return evaluation.extract_cited_claims(REPORT, num_sources=3)


def test_rewrite_keeps_the_original_markers():
    claims = _claims()
    revised = evaluation.apply_revisions(REPORT, [
        {"span": claims[1]["span"], "action": "rewrite", "text": "They were introduced for translation [9]."},
    ])
    assert revised == ("Transformers rely on attention [1]. They were introduced for translation [2]. "
                       "Some say they are magic. [3] Training is costly [1].")


def test_remove_drops_the_sentence_and_a_marker_placed_after_it():
    claims = _claims()
    revised = evaluation.apply_revisions(REPORT, [{"span": claims[2]["span"], "action": "remove"}])
    assert revised == ("Transformers rely on attention [1]. They were invented in 1990 and solved vision [2]. "
                       "Training is costly [1].")


def test_several_revisions_apply_without_shifting_each_other():
    claims = _claims()
    revised = evaluation.apply_revisions(REPORT, [
        {"span": claims[2]["span"], "action": "remove"},
        {"span": claims[1]["span"], "action": "rewrite", "text": "They came later"},
    ])
    assert revised == "Transformers rely on attention [1]. They came later [2] Training is costly [1]."


# ---------- the reviser (Groq faked) ----------

def test_revise_claims_parses_rewrites_and_removals(monkeypatch):
    sent = {}

    def fake_call(model, system_prompt, user_prompt, json_mode=True, temperature=0.2, **options):
        sent.update(prompt=user_prompt, options=options)
        return json.dumps({"revisions": [
            {"sentence": 2, "action": "remove"},
            {"sentence": 1, "action": "rewrite", "text": "  Fixed sentence.  "},
            {"sentence": 9, "action": "rewrite", "text": "out of range"},
        ]})

    monkeypatch.setattr(groq_client, "_call_groq", fake_call)
    chunks = [{"text": "Excerpt one."}, {"text": "Excerpt two."}]
    result = groq_client.revise_claims(
        [{"claim": "A.", "sources": [1], "reason": "wrong year"},
         {"claim": "B.", "sources": [2], "reason": ""},
         {"claim": "C.", "sources": [1], "reason": ""}],
        chunks,
    )
    assert result == [{"action": "rewrite", "text": "Fixed sentence."}, {"action": "remove"}, None]
    assert "Problem: wrong year" in sent["prompt"] and "[2] Excerpt two." in sent["prompt"]
    assert sent["options"]["reasoning_effort"] == "low"


# ---------- pipeline wiring ----------

def _judge_from(table):
    """A fake judge: verdict per claim text (default supported)."""
    def judge(claims, chunks, **k):
        return [{"claim": c["text"], "sources": c["sources"],
                 "verdict": table.get(c["text"], "supported"), "reason": "r"} for c in claims]
    return judge


def test_pipeline_rewrites_flagged_claims_and_rejudges_them(monkeypatch):
    chunks = _chunks_for("doc.pdf", 4, "transformers")
    calls = {}

    def reviser(flagged, retrieved):
        calls["flagged"] = flagged
        return [{"action": "rewrite", "text": "Transformers were proposed for translation."}]

    report = "Transformers use attention heavily [1]. Transformers were invented in 1990 for vision [2]."
    result = _run(chunks, intent="targeted_qa",
                  generate_report={"report_text": report, "citations_used": []},
                  revise_claims=reviser,
                  judge_claims=_judge_from({"Transformers were invented in 1990 for vision.": "unsupported"}))

    assert [f["claim"] for f in calls["flagged"]] == ["Transformers were invented in 1990 for vision."]
    assert result["report_text"] == ("Transformers use attention heavily [1]. "
                                     "Transformers were proposed for translation [2].")
    revised = result["claim_verdicts"][1]
    assert revised["verdict"] == "supported" and revised["original_verdict"] == "unsupported"
    assert revised["original"] == "Transformers were invented in 1990 for vision."
    checks = {c["name"]: c for c in result["evaluation"]["checks"]}
    assert checks["entailment"]["passed"] and checks["entailment"]["value"] == 100.0
    assert "1 rewritten" in checks["claim_revision"]["detail"]


def test_removed_claims_drop_out_of_the_report_and_citations(monkeypatch):
    chunks = _chunks_for("doc.pdf", 4, "transformers")
    report = "Transformers use attention heavily [1]. Transformers are conscious beings now [2]."
    result = _run(chunks, intent="targeted_qa",
                  revise_claims=lambda flagged, retrieved: [{"action": "remove"}],
                  generate_report={"report_text": report, "citations_used": []},
                  judge_claims=_judge_from({"Transformers are conscious beings now.": "contradicted"}))
    assert result["report_text"] == "Transformers use attention heavily [1]."
    assert result["citations_used"] == [chunks[0]["source_label"]]
    assert result["claim_verdicts"][1]["verdict"] == "removed"


def test_a_failed_revision_keeps_the_report_and_records_an_error(monkeypatch):
    chunks = _chunks_for("doc.pdf", 4, "transformers")

    def broken(flagged, retrieved):
        raise RuntimeError("Groq down")

    report = "Transformers use attention heavily [1]. Transformers are conscious beings now [2]."
    result = _run(chunks, intent="targeted_qa", revise_claims=broken,
                  generate_report={"report_text": report, "citations_used": []},
                  judge_claims=_judge_from({"Transformers are conscious beings now.": "contradicted"}))
    assert result["report_text"] == report
    assert "Groq down" in result["evaluation"]["errors"][0]
    assert not next(c for c in result["evaluation"]["checks"] if c["name"] == "entailment")["passed"]


def test_no_revision_call_when_nothing_is_flagged_or_it_is_disabled(monkeypatch):
    calls = []
    chunks = _chunks_for("doc.pdf", 4, "transformers")
    report = {"report_text": "Transformers use attention heavily [1].", "citations_used": []}
    _run(chunks, intent="targeted_qa", generate_report=report, judge_claims=_judge_from({}),
         revise_claims=lambda *a: calls.append(1))
    monkeypatch.setattr(pipeline, "CLAIM_REVISION_ENABLED", False)
    _run(chunks, intent="targeted_qa", generate_report=report, revise_claims=lambda *a: calls.append(1),
         judge_claims=_judge_from({"Transformers use attention heavily.": "unsupported"}))
    assert calls == []

"""
The evaluation harness itself: the golden set is internally consistent
(every evidence quote really is in its document), the metrics compute
what they claim, and the runner records them -- with the pipeline faked,
so no network.
"""

import json

import pytest

from eval import metrics, run_eval

QUESTIONS = metrics.load_jsonl(metrics.GOLDEN_DIR / "questions.jsonl")
DOCS = {p.name: p.read_text(encoding="utf-8") for p in (metrics.GOLDEN_DIR / "docs").glob("*.txt")}


# ---------- the golden set ----------

def test_question_ids_are_unique_and_types_known():
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    assert {q["type"] for q in QUESTIONS} <= {"targeted_qa", "multi_hop", "analysis", "compound",
                                             "summary", "unanswerable"}


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q["id"])
def test_every_evidence_quote_is_in_its_document(question):
    for doc in question.get("docs", []):
        assert doc in DOCS
    for item in question["evidence"]:
        assert metrics._squash(item["quote"]) in metrics._squash(DOCS[item["doc"]]), item["quote"]


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q["id"])
def test_required_facts_appear_in_the_reference_answer(question):
    assert metrics.fact_recall(question["required_facts"], question["reference_answer"]) in (None, 1.0)


def test_multi_hop_questions_need_more_than_one_document():
    for q in QUESTIONS:
        if q["type"] == "multi_hop":
            assert len({e["doc"] for e in q["evidence"]}) >= 2, q["id"]


def test_judge_labels_are_well_formed():
    items = metrics.load_jsonl(metrics.DATA_DIR / "judge_labels.jsonl")
    assert len(items) >= 30
    assert {i["label"] for i in items} == set(metrics.LABELS)
    assert len({i["id"] for i in items}) == len(items)


# ---------- metrics ----------

RETRIEVED = [{"text": "Unrelated   chunk."}, {"text": "HelixNet was  pre-trained and fine-tuned on the TerraNotes corpus."}]


def test_evidence_recall_and_mrr_ignore_whitespace_and_case():
    evidence = [{"doc": "d", "quote": "helixnet was pre-trained and fine-tuned"},
                {"doc": "d", "quote": "not retrieved anywhere"}]
    assert metrics.evidence_hits(evidence, RETRIEVED) == [2, None]
    assert metrics.evidence_recall(evidence, RETRIEVED) == 0.5
    assert metrics.evidence_mrr(evidence, RETRIEVED) == 0.5
    assert metrics.evidence_recall([], RETRIEVED) is None


def test_fact_recall_tolerates_punctuation_variants():
    report = "Licensed CC BY‑NC 4.0; notes over 16 000 tokens."
    assert metrics.fact_recall(["CC BY-NC 4.0", "16,000", "Kenya"], report) == pytest.approx(2 / 3)
    assert metrics.fact_recall([], report) is None


def test_aggregate_groups_by_type_and_skips_missing_values():
    rows = [{"type": "qa", "fact_recall": 1.0, "hops": None},
            {"type": "qa", "fact_recall": 0.0, "hops": 2},
            {"type": "multi_hop", "fact_recall": 0.5}]
    summary = metrics.aggregate(rows)
    assert summary["ALL"]["n"] == 3 and summary["ALL"]["fact_recall"] == 0.5
    assert summary["qa"]["hops"] == 2 and summary["multi_hop"]["evidence_recall"] is None


def test_cohens_kappa():
    assert metrics.cohens_kappa([("a", "a"), ("b", "b")]) == 1.0
    # Agreement no better than chance -> 0.
    assert metrics.cohens_kappa([("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]) == 0.0


def test_agreement_report_binary_view():
    items = [{"label": "supported"}, {"label": "unsupported"}, {"label": "contradicted"}, {"label": "supported"}]
    report = metrics.agreement_report(items, ["supported", "contradicted", "contradicted", "unsupported"])
    assert report["accuracy_binary"] == 0.75        # only item 4 disagrees on supported-vs-flagged
    assert report["accuracy_3class"] == 0.5
    assert report["flag_recall"] == 1.0             # both bad claims caught
    assert report["flag_precision"] == pytest.approx(2 / 3)
    assert report["confusion"]["unsupported"]["contradicted"] == 1


# ---------- runner (pipeline faked) ----------

def test_run_question_scores_a_pipeline_result(monkeypatch, tmp_path):
    from app import pipeline
    seen = {}

    def fake_pipeline(file_paths, seed_urls, instructions, username=None, run_id=None):
        seen.update(files=file_paths, username=username)
        return {
            "retrieved_chunks": [{"text": "HelixNet was pre-trained and fine-tuned on the TerraNotes corpus"}],
            "report_text": "HelixNet was trained on TerraNotes [1].",
            "evaluation": {"checks": [
                {"name": "intent", "detail": "Classified as 'targeted_qa' -- hybrid", "value": None},
                {"name": "entailment", "detail": "", "value": 100.0},
                {"name": "multi_hop", "detail": "", "value": 1},
            ], "errors": []},
        }

    monkeypatch.setattr(pipeline, "run_pipeline", fake_pipeline)
    question = next(q for q in QUESTIONS if q["id"] == "qa-corpus")
    row = run_eval.run_question(question, metrics.GOLDEN_DIR / "docs", run_eval._GroqCallCounter(), grade=False)
    assert row["intent"] == "targeted_qa"
    assert (row["evidence_recall"], row["evidence_mrr"], row["fact_recall"]) == (1.0, 1.0, 1.0)
    assert (row["entailment"], row["hops"], row["groq_calls"]) == (100.0, 1, 0)
    assert len(seen["files"]) == len(DOCS) and seen["username"] == "eval"


def test_summary_question_uses_only_its_own_docs(monkeypatch):
    from app import pipeline
    seen = {}
    monkeypatch.setattr(pipeline, "run_pipeline",
                        lambda files, *a, **k: seen.setdefault("files", files) and
                        {"retrieved_chunks": [], "report_text": "", "evaluation": {"checks": []}})
    question = next(q for q in QUESTIONS if q["id"] == "sum-datasheet")
    run_eval.run_question(question, metrics.GOLDEN_DIR / "docs", run_eval._GroqCallCounter(), grade=False)
    assert [f.split("\\")[-1].split("/")[-1] for f in seen["files"]] == ["terranotes_datasheet.txt"]


def test_a_failing_question_is_recorded_not_raised(monkeypatch):
    from app import pipeline

    def boom(*a, **k):
        raise RuntimeError("Groq down")

    monkeypatch.setattr(pipeline, "run_pipeline", boom)
    row = run_eval.run_question(QUESTIONS[0], metrics.GOLDEN_DIR / "docs", run_eval._GroqCallCounter(), grade=False)
    assert row["error"] == "RuntimeError: Groq down"


def test_overrides_reach_every_module_holding_a_copy(monkeypatch):
    from app import config, pipeline
    monkeypatch.setattr(config, "MULTI_HOP_ENABLED", True)
    monkeypatch.setattr(pipeline, "MULTI_HOP_ENABLED", True)
    applied = run_eval.apply_overrides(["MULTI_HOP_ENABLED=false", "TOP_K_CHUNKS=7"])
    try:
        assert applied == {"MULTI_HOP_ENABLED": False, "TOP_K_CHUNKS": 7}
        assert pipeline.MULTI_HOP_ENABLED is False and config.TOP_K_CHUNKS == 7
    finally:
        run_eval.apply_overrides(["TOP_K_CHUNKS=5"])
    with pytest.raises(SystemExit):
        run_eval.apply_overrides(["NOT_A_SETTING=1"])


def test_compare_prints_deltas(tmp_path, capsys):
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps({"overrides": {}, "summary": {"ALL": {"n": 1, "fact_recall": value}}}))
        return str(path)

    run_eval.compare(save("a.json", 0.5), save("b.json", 0.75))
    assert "| ALL | fact_recall | 0.50 | 0.75 | +0.25 |" in capsys.readouterr().out

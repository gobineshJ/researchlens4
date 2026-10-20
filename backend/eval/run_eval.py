"""
End-to-end evaluation: runs the REAL pipeline (real Groq, real embeddings)
on the golden set in eval/data/golden/ and scores every answer.

    python -m eval.run_eval                          # every question
    python -m eval.run_eval --only mh- qa-corpus     # ids starting with these
    python -m eval.run_eval --set MULTI_HOP_ENABLED=false --tag no-multihop
    python -m eval.run_eval --judge-answers          # + LLM answer grading
    python -m eval.run_eval --compare eval/results/A.json eval/results/B.json

Per question it records:
  evidence_recall  share of the gold evidence passages that reached the
                   chunks the report was written from (retrieval)
  evidence_mrr     1 / rank of the first chunk holding gold evidence
  fact_recall      share of the required facts the report states
  answer_score     LLM grade vs. the reference answer: 1 / 0.5 / 0
                   (only with --judge-answers; costs one call per question)
  entailment       the pipeline's own entailment check value (%)
  groundedness     the pipeline's own groundedness check value (%)
  hops             retrieval hops taken (multi-hop check)
  groq_calls       Groq requests the question cost
  seconds          wall-clock time

Results go to eval/results/<timestamp>_<tag>.json and a table is printed.
Cost: every question is a full pipeline run -- roughly 6-10 Groq calls,
one of them to the large generation model. On the free tier expect
about a minute per question, and the whole set uses a noticeable part of
the daily token allowance; use --only for quick checks.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from . import metrics

RESULTS_DIR = Path(__file__).parent / "results"


def _parse_value(text: str):
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def apply_overrides(assignments: list[str]) -> dict:
    """
    --set NAME=VALUE. Pipeline modules import settings by name
    (`from .config import X`), so a setting has to be replaced in every
    module that holds its own copy, not just in config.
    """
    from app import (chunker, config, doc_parser, evaluation, groq_client, hybrid_search,
                     pipeline, reranker, scraper, vectorstore)
    modules = [config, pipeline, groq_client, evaluation, reranker, scraper, chunker,
               doc_parser, hybrid_search, vectorstore]
    applied = {}
    for assignment in assignments:
        name, _, raw = assignment.partition("=")
        if not hasattr(config, name):
            raise SystemExit(f"--set {name}: no such setting in app/config.py")
        value = _parse_value(raw)
        for module in modules:
            if hasattr(module, name):
                setattr(module, name, value)
        applied[name] = value
    return applied


class _GroqCallCounter:
    """Counts requests by wrapping groq_client._post_chat."""

    def __init__(self):
        from app import groq_client
        self.count = 0
        original = groq_client._post_chat

        def counted(*args, **kwargs):
            self.count += 1
            return original(*args, **kwargs)

        groq_client._post_chat = counted


ANSWER_JUDGE_PROMPT = (
    "You grade an answer produced by a research assistant against a reference "
    "answer written by a person who read the source documents. Grade only "
    "whether the answer conveys the reference answer's facts correctly; extra "
    "correct detail is fine, wording does not matter. If the reference says the "
    "documents don't contain the answer, a correct answer says so instead of "
    "guessing. Respond ONLY with JSON: "
    '{"grade": "correct" | "partial" | "incorrect", "reason": "one sentence"}.'
)
_GRADES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def judge_answer(question: str, reference: str, answer: str) -> tuple[float | None, str]:
    from app import groq_client
    from app.config import GROQ_JUDGE_MODEL
    raw = groq_client._call_groq(
        GROQ_JUDGE_MODEL, ANSWER_JUDGE_PROMPT,
        f"Question: {question}\n\nReference answer: {reference}\n\nAnswer to grade:\n{answer}",
        json_mode=True, temperature=0, reasoning_effort="low", max_completion_tokens=1500,
    )
    parsed = json.loads(raw)
    return _GRADES.get(str(parsed.get("grade", "")).lower()), str(parsed.get("reason", ""))


def run_question(question: dict, docs_dir: Path, counter: _GroqCallCounter, grade: bool) -> dict:
    from app.pipeline import run_pipeline
    doc_names = question.get("docs") or sorted(p.name for p in docs_dir.glob("*.txt"))
    calls_before, started = counter.count, time.perf_counter()
    row = {"id": question["id"], "type": question["type"], "question": question["question"]}
    try:
        result = run_pipeline([str(docs_dir / name) for name in doc_names], [], question["question"],
                              username="eval", run_id=f"eval-{question['id']}")
    except Exception as exc:
        row.update(error=f"{type(exc).__name__}: {exc}", seconds=round(time.perf_counter() - started, 1),
                   groq_calls=counter.count - calls_before)
        return row

    retrieved, report = result["retrieved_chunks"], result["report_text"]
    evaluation = result["evaluation"]
    intent = next((c["detail"].split("'")[1] for c in evaluation["checks"] if c["name"] == "intent"), None)
    row.update(
        intent=intent,
        evidence_recall=metrics.evidence_recall(question["evidence"], retrieved),
        evidence_mrr=metrics.evidence_mrr(question["evidence"], retrieved),
        fact_recall=metrics.fact_recall(question["required_facts"], report),
        entailment=metrics.check_value(evaluation, "entailment"),
        groundedness=metrics.check_value(evaluation, "groundedness"),
        hops=metrics.check_value(evaluation, "multi_hop"),
        report_text=report,
        errors=evaluation.get("errors", []),
    )
    if grade:
        try:
            row["answer_score"], row["answer_reason"] = judge_answer(
                question["question"], question["reference_answer"], report)
        except Exception as exc:
            row["answer_reason"] = f"grading failed: {exc}"
    row.update(groq_calls=counter.count - calls_before, seconds=round(time.perf_counter() - started, 1))
    return row


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def print_table(summary: dict) -> None:
    cols = ["n"] + metrics.METRICS
    print("| group | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for group, values in summary.items():
        print(f"| {group} | " + " | ".join(_fmt(values.get(c)) for c in cols) + " |")


def compare(path_a: str, path_b: str) -> None:
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (path_a, path_b))
    print(f"A = {Path(path_a).name}  {a.get('overrides') or ''}")
    print(f"B = {Path(path_b).name}  {b.get('overrides') or ''}\n")
    print("| group | metric | A | B | B − A |")
    print("|---|---|---|---|---|")
    for group in a["summary"]:
        for metric in metrics.METRICS:
            va, vb = a["summary"][group].get(metric), b["summary"].get(group, {}).get(metric)
            if va is None and vb is None:
                continue
            delta = vb - va if va is not None and vb is not None else None
            print(f"| {group} | {metric} | {_fmt(va)} | {_fmt(vb)} | "
                  f"{('+' if delta and delta > 0 else '') + _fmt(delta) if delta is not None else '—'} |")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="*", default=[], help="question id prefixes to run")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="NAME=VALUE",
                        help="override app/config.py settings for this run")
    parser.add_argument("--tag", default="", help="label for the results file")
    parser.add_argument("--judge-answers", action="store_true", help="grade answers with an LLM")
    parser.add_argument("--questions", default=str(metrics.GOLDEN_DIR / "questions.jsonl"))
    parser.add_argument("--docs", default=str(metrics.GOLDEN_DIR / "docs"))
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    args = parser.parse_args(argv)

    if args.compare:
        compare(*args.compare)
        return

    from app import config, evaluation
    if not config.GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is not set in backend/.env -- the evaluation runs the real pipeline.")
    overrides = apply_overrides(args.overrides)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{stamp}{'_' + args.tag if args.tag else ''}.json"
    # Eval runs go to their own log, never into the app's Run history.
    evaluation.EVALUATION_LOG_PATH = str(out_path.with_suffix(".log.jsonl"))

    questions = [q for q in metrics.load_jsonl(args.questions)
                 if not args.only or any(q["id"].startswith(p) for p in args.only)]
    counter = _GroqCallCounter()
    rows = []
    for n, question in enumerate(questions, 1):
        print(f"[{n}/{len(questions)}] {question['id']}: {question['question']}", flush=True)
        row = run_question(question, Path(args.docs), counter, args.judge_answers)
        rows.append(row)
        print("    " + ("ERROR " + row["error"] if "error" in row else
                        "  ".join(f"{m}={_fmt(row.get(m))}" for m in metrics.METRICS if row.get(m) is not None)),
              flush=True)

    summary = metrics.aggregate([r for r in rows if "error" not in r])
    out_path.write_text(json.dumps({"overrides": overrides, "judge_answers": args.judge_answers,
                                    "summary": summary, "questions": rows}, indent=2), encoding="utf-8")
    print()
    print_table(summary)
    failed = [r["id"] for r in rows if "error" in r]
    if failed:
        print(f"\n{len(failed)} question(s) failed to run: {', '.join(failed)}")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

"""
Scoring for the evaluation harness. Pure functions, no network -- so the
numbers the harness reports are themselves unit-tested
(tests/unit/test_eval_metrics.py).
"""

import json
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
GOLDEN_DIR = DATA_DIR / "golden"


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _squash(text: str) -> str:
    """Whitespace-insensitive, case-insensitive form for substring checks
    (chunking re-joins text on single spaces)."""
    return " ".join((text or "").split()).lower()


def _alnum(text: str) -> str:
    """Only letters and digits: "CC BY‑NC 4.0" and "CC BY-NC 4.0" match,
    "16,000" and "16 000" match."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def evidence_hits(evidence: list[dict], retrieved: list[dict]) -> list[int | None]:
    """For each evidence quote, the 1-based rank of the first retrieved
    chunk containing it, or None if no retrieved chunk does."""
    hits = []
    for item in evidence:
        quote = _squash(item["quote"])
        rank = next((i + 1 for i, chunk in enumerate(retrieved) if quote in _squash(chunk.get("text", ""))), None)
        hits.append(rank)
    return hits


def evidence_recall(evidence: list[dict], retrieved: list[dict]) -> float | None:
    """Share of the evidence passages that made it into the chunks the
    report was written from. None when a question has no evidence list
    (summaries, unanswerable questions)."""
    if not evidence:
        return None
    hits = evidence_hits(evidence, retrieved)
    return sum(h is not None for h in hits) / len(hits)


def evidence_mrr(evidence: list[dict], retrieved: list[dict]) -> float | None:
    """Reciprocal rank of the first retrieved chunk holding ANY evidence."""
    if not evidence:
        return None
    ranks = [h for h in evidence_hits(evidence, retrieved) if h is not None]
    return 1 / min(ranks) if ranks else 0.0


def fact_recall(required_facts: list[str], report_text: str) -> float | None:
    """Share of the facts a correct answer must state that the report states."""
    if not required_facts:
        return None
    report = _alnum(report_text)
    return sum(_alnum(fact) in report for fact in required_facts) / len(required_facts)


def check_value(evaluation: dict, name: str) -> float | None:
    """The numeric value of one evaluation-panel check, if it ran."""
    for check in evaluation.get("checks", []):
        if check["name"] == name:
            return check.get("value")
    return None


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


METRICS = ["evidence_recall", "evidence_mrr", "fact_recall", "answer_score",
           "entailment", "groundedness", "hops", "groq_calls", "seconds"]


def aggregate(rows: list[dict]) -> dict:
    """Mean of every metric, overall and per question type."""
    groups: dict[str, list[dict]] = {"ALL": rows}
    for row in rows:
        groups.setdefault(row["type"], []).append(row)
    return {
        group: {"n": len(members), **{m: mean(r.get(m) for r in members) for m in METRICS}}
        for group, members in groups.items()
    }


# ---------- judge agreement ----------

LABELS = ("supported", "unsupported", "contradicted")


def cohens_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Agreement beyond chance between two labelers (here: human vs judge)."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[k] * right[k] for k in set(left) | set(right)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def agreement_report(items: list[dict], verdicts: list[str]) -> dict:
    """
    items: [{"label": human label, ...}], verdicts: the judge's verdict for
    each (or "unjudged"). The binary view -- supported vs. flagged
    (unsupported or contradicted) -- is the one that matters in practice:
    the check's job is to catch claims that shouldn't stand.
    """
    judged = [(item["label"], v) for item, v in zip(items, verdicts) if v in LABELS]
    binary = [("supported" if h == "supported" else "flagged",
               "supported" if j == "supported" else "flagged") for h, j in judged]
    true_flags = sum(1 for h, j in binary if h == "flagged" and j == "flagged")
    judge_flags = sum(1 for _, j in binary if j == "flagged")
    human_flags = sum(1 for h, _ in binary if h == "flagged")
    confusion = {h: {j: sum(1 for a, b in judged if a == h and b == j) for j in LABELS} for h in LABELS}
    return {
        "n": len(items),
        "judged": len(judged),
        "accuracy_3class": mean(float(h == j) for h, j in judged),
        "kappa_3class": cohens_kappa(judged),
        "accuracy_binary": mean(float(h == j) for h, j in binary),
        "kappa_binary": cohens_kappa(binary),
        # Of the claims the judge flagged, how many really were bad...
        "flag_precision": true_flags / judge_flags if judge_flags else None,
        # ...and of the claims that really were bad, how many it caught.
        "flag_recall": true_flags / human_flags if human_flags else None,
        "confusion": confusion,
    }

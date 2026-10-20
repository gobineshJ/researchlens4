"""
How far can the entailment judge be trusted? Runs the REAL judge
(groq_client.judge_claims, same model and settings as the pipeline) on
claims a person has labelled, and reports how often the two agree.

    python -m eval.judge_agreement                         # eval/data/judge_labels.jsonl
    python -m eval.judge_agreement --labels my_labels.jsonl

Each line of a labels file: {"id", "excerpt", "claim", "label"} where
label is "supported", "unsupported" or "contradicted" -- judged ONLY
against that excerpt, the same question the judge is asked.

Headline numbers are binary (supported vs. flagged), since that is what
the entailment check acts on: flag_recall = how many bad claims the judge
catches, flag_precision = how many of its flags are right, and Cohen's
kappa = agreement beyond chance (above ~0.6 is usually called substantial,
above ~0.8 almost perfect). Every disagreement is listed, so you can see
whether the judge or the label is wrong.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from . import metrics

RESULTS_DIR = Path(__file__).parent / "results"


def judge_items(items: list[dict]) -> list[dict]:
    """One judge verdict per labelled item. Each item's excerpt becomes its
    own numbered source, so judge_claims can pack many items per request."""
    from app import groq_client
    chunks = [{"text": item["excerpt"], "source_label": item["id"]} for item in items]
    claims = [{"text": item["claim"], "sources": [i + 1]} for i, item in enumerate(items)]
    return groq_client.judge_claims(claims, chunks)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--labels", default=str(metrics.DATA_DIR / "judge_labels.jsonl"))
    args = parser.parse_args(argv)

    from app import config
    if not config.GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is not set in backend/.env -- this runs the real judge.")
    items = [i for i in metrics.load_jsonl(args.labels) if i.get("label") in metrics.LABELS]
    if not items:
        raise SystemExit(f"No labelled items in {args.labels} (label must be one of {metrics.LABELS}).")

    verdicts = judge_items(items)
    report = metrics.agreement_report(items, [v["verdict"] for v in verdicts])

    print(f"{report['judged']} of {report['n']} claims judged ({config.GROQ_JUDGE_MODEL})\n")
    for key in ("accuracy_binary", "kappa_binary", "flag_recall", "flag_precision",
                "accuracy_3class", "kappa_3class"):
        value = report[key]
        print(f"  {key:16} {value:.2f}" if value is not None else f"  {key:16} —")
    print("\nConfusion (rows = human label, columns = judge):")
    print(f"  {'':14}" + "".join(f"{label:>14}" for label in metrics.LABELS))
    for human in metrics.LABELS:
        print(f"  {human:14}" + "".join(f"{report['confusion'][human][j]:>14}" for j in metrics.LABELS))

    disagreements = [(item, v) for item, v in zip(items, verdicts) if v["verdict"] != item["label"]]
    if disagreements:
        print(f"\nDisagreements ({len(disagreements)}):")
        for item, v in disagreements:
            print(f"  {item['id']} [{item.get('kind', '')}] human={item['label']} judge={v['verdict']}")
            print(f"     claim:  {item['claim']}")
            print(f"     judge:  {v['reason']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"judge_agreement_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"labels": args.labels, "model": config.GROQ_JUDGE_MODEL, "report": report,
                               "items": [{**item, "judge": v["verdict"], "judge_reason": v["reason"]}
                                         for item, v in zip(items, verdicts)]}, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

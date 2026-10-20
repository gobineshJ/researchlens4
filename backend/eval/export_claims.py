"""
Turns claims from YOUR real runs into a file you can label, so judge
agreement can be measured on real reports, not just the starter set.

    python -m eval.export_claims                 # recent finished jobs
    python -m eval.export_claims --limit 5 --out eval/data/my_labels.jsonl

Reads finished jobs' results from the Celery result backend (Redis keeps
them for 24 hours), and writes one line per judged claim with the
excerpt(s) it cited and "label": null. The judge's verdict is
deliberately NOT included -- label each claim yourself first, then run
    python -m eval.judge_agreement --labels <that file>
"""

import argparse
import json
import sqlite3
from pathlib import Path

from . import metrics


def claims_from_result(job_id: str, result: dict) -> list[dict]:
    retrieved = result.get("retrieved_chunks") or []
    rows = []
    for n, verdict in enumerate(result.get("claim_verdicts") or []):
        if verdict.get("verdict") == "removed" or not verdict.get("claim"):
            continue
        excerpts = [retrieved[i - 1]["text"] for i in verdict["sources"] if 1 <= i <= len(retrieved)]
        if not excerpts:
            continue
        rows.append({"id": f"{job_id[:8]}-{n + 1}", "excerpt": "\n\n".join(excerpts),
                     "claim": verdict["claim"], "label": None})
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=10, help="most recent jobs to read")
    parser.add_argument("--out", default=str(metrics.DATA_DIR / "my_labels.jsonl"))
    args = parser.parse_args(argv)

    from celery.result import AsyncResult
    from app import config
    from app.celery_app import celery_app

    with sqlite3.connect(config.AUTH_DB_PATH) as conn:
        job_ids = [r[0] for r in conn.execute(
            "SELECT job_id FROM jobs ORDER BY created_at DESC LIMIT ?", (args.limit,))]
    rows = []
    for job_id in job_ids:
        result = AsyncResult(job_id, app=celery_app)
        if result.state == "SUCCESS" and isinstance(result.result, dict):
            rows.extend(claims_from_result(job_id, result.result))
    if not rows:
        raise SystemExit("No judged claims found in recent jobs (results are kept for 24 hours).")
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} claims to {args.out}. Set each \"label\" to supported / unsupported / "
          f"contradicted, judging the claim against its excerpt only, then run:\n"
          f"    python -m eval.judge_agreement --labels {args.out}")


if __name__ == "__main__":
    main()

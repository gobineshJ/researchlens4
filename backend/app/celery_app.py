"""
Celery app + the one task it runs: the full pipeline. /process used to
run it inside the HTTP request, which held the connection open for the
minute or two a report takes; now it enqueues this task and the
frontend polls /jobs/{job_id} for the result.

Start a worker (from backend/, with Redis running):

    celery -A app.celery_app worker --loglevel=info

The pool is chosen for the platform below: on Linux/macOS/WSL a prefork
pool runs CELERY_WORKER_CONCURRENCY pipeline runs at once, each in its own
process; on native Windows, where prefork doesn't work, the solo pool runs
one at a time. --pool / --concurrency on the command line still override.
Several processes share one Groq key, so groq_client shares its
rate-limit pacing state through Redis (GROQ_RATE_LIMIT_REDIS_URL).
"""

import shutil
import sys

from celery import Celery

from . import config, db, papers, progress

celery_app = Celery("researchlens")

celery_app.conf.update(
    broker_url=config.CELERY_BROKER_URL,
    # Eager mode runs the task in-process, so it needs no broker -- but its
    # result still has to be stored somewhere /jobs/{id} can read it back.
    # An in-memory backend works there because the API is the only process.
    result_backend="cache+memory://" if config.CELERY_TASK_ALWAYS_EAGER else config.CELERY_RESULT_BACKEND,
    task_always_eager=config.CELERY_TASK_ALWAYS_EAGER,
    task_store_eager_result=True,
    task_track_started=True,     # lets /jobs report "running" instead of only "queued"
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=60 * 60 * 24,
    worker_prefetch_multiplier=1,  # one long pipeline run at a time per worker process
    task_acks_late=True,
    worker_pool="solo" if sys.platform == "win32" else "prefork",
    worker_concurrency=1 if sys.platform == "win32" else config.CELERY_WORKER_CONCURRENCY,
)

# If Redis is down, give up within a few seconds so /process can answer
# "is Redis running?" -- Celery's defaults keep retrying for ~2 minutes,
# leaving the browser hanging on "Processing..." the whole time.
_QUICK_RETRY = {"max_retries": 2, "interval_start": 0, "interval_step": 0.5, "interval_max": 1}
celery_app.conf.update(
    task_publish_retry_policy=_QUICK_RETRY,
    broker_transport_options={"max_retries": 2, "socket_connect_timeout": 3},
    result_backend_transport_options={"retry_policy": _QUICK_RETRY, "socket_connect_timeout": 3},
)


@celery_app.task(name="researchlens.run_pipeline", bind=True)
def run_pipeline_task(self, job_dir: str, file_paths: list[str], seed_urls: list[str], instructions: str,
                      username: str | None = None, arxiv_ids: list[str] | None = None,
                      feed: str | None = None) -> dict:
    # `username`, `arxiv_ids` and `feed` default to None so jobs queued
    # before they were added still run.
    # arxiv_ids: papers to download into job_dir before the pipeline starts
    # (/papers/arxiv/process). Once the report is done they are marked seen
    # in `feed` -- only then, so a failed run's papers are offered again on
    # the next digest instead of silently skipped.
    # Imported here rather than at module top so the API process (which
    # only enqueues) doesn't load the embedding and reranker models.
    from .pipeline import run_pipeline

    # A custom "PROGRESS" state whose meta /jobs/{id} hands to the frontend
    # while the run is still going: the current stage, a detail line, and
    # the report so far once generation is streaming it.
    meta = {"stage": "", "detail": "", "draft": ""}

    def publish(stage: str, detail: str) -> None:
        meta.update(stage=stage, detail=detail)
        self.update_state(state="PROGRESS", meta=dict(meta))

    def publish_draft(text: str) -> None:
        meta["draft"] = text
        self.update_state(state="PROGRESS", meta=dict(meta))

    try:
        with progress.listening(publish, on_draft=publish_draft):
            downloaded: list[str] = []
            failures: list[str] = []
            if arxiv_ids:
                pdf_paths, downloaded, failures = papers.download_arxiv_pdfs(arxiv_ids, job_dir)
                file_paths = list(file_paths) + pdf_paths
            # run_id = the job id, so a Run history entry can be matched to its job.
            result = run_pipeline(file_paths, seed_urls, instructions,
                                  username=username, run_id=self.request.id)
        if arxiv_ids:
            result["arxiv"] = {"included": downloaded, "failed": failures}
            if feed and username:
                db.mark_papers_seen(username, feed, "arxiv",
                                    [papers.base_id(i) for i in downloaded], job_id=self.request.id)
        return result
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

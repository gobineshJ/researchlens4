"""
Celery app + the one task it runs: the full pipeline. /process used to
run it inside the HTTP request, which held the connection open for the
minute or two a report takes; now it enqueues this task and the
frontend polls /jobs/{job_id} for the result.

Start a worker (from backend/, with Redis running):

    celery -A app.celery_app worker --loglevel=info --pool=solo

--pool=solo is needed on Windows, where Celery's default prefork pool
doesn't work. On Linux/macOS you can drop it.
"""

import shutil

from celery import Celery

from . import config, progress

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
def run_pipeline_task(self, job_dir: str, file_paths: list[str], seed_urls: list[str], instructions: str) -> dict:
    # Imported here rather than at module top so the API process (which
    # only enqueues) doesn't load the embedding and reranker models.
    from .pipeline import run_pipeline

    def publish(stage: str, detail: str) -> None:
        # A custom "PROGRESS" state whose meta /jobs/{id} hands to the
        # frontend while the run is still going.
        self.update_state(state="PROGRESS", meta={"stage": stage, "detail": detail})

    try:
        with progress.listening(publish):
            return run_pipeline(file_paths, seed_urls, instructions)
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

"""
FastAPI backend for ResearchLens.

  /health                 public liveness check the frontend can ping
  /auth/*                 login (OAuth2 password flow -> JWT), register, me
  /process                enqueues a pipeline run on Celery -> {job_id}
  /jobs/{job_id}          poll a run's status; holds the result when done
  /download/{filename}    the PDF report of one of YOUR runs
  /evaluation-history     YOUR runs from the run log

Everything except /health and /auth/token + /auth/register needs an
"Authorization: Bearer <token>" header -- the routes are mounted on a
router whose dependency is auth.get_current_user.

CORS only admits the origins in CORS_ORIGINS, with credentials, since the
refresh token travels in a cookie. The frontend normally reaches the API
through a same-origin /api proxy and needs no CORS at all.
"""

from pathlib import Path
import shutil
import uuid

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import auth, config, db, evaluation
from .auth import User, get_current_user
from .celery_app import celery_app, run_pipeline_task

app = FastAPI(title="ResearchLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(auth.router)

protected = APIRouter(dependencies=[Depends(get_current_user)])


@protected.post("/process", status_code=202)
def process(
    files: list[UploadFile] = File(default=[]),
    seed_urls: str = Form(default=""),
    instructions: str = Form(default=""),
    user: User = Depends(get_current_user),
):
    if not files and not seed_urls.strip():
        raise HTTPException(status_code=400, detail="Provide at least one file or one seed URL.")

    # Files go in a per-job directory the worker can still read after this
    # request returns; the task deletes it once the pipeline is done.
    job_id = str(uuid.uuid4())
    job_dir = Path(config.UPLOAD_DIR) / job_id
    job_dir.mkdir(parents=True)
    saved_paths = []
    try:
        for upload in files:
            # .name drops any directory part, so a filename like
            # "../../app/main.py" can't write outside job_dir.
            dest = job_dir / Path(upload.filename or "upload").name
            with dest.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
            saved_paths.append(str(dest.resolve()))

        urls = [u.strip() for u in seed_urls.splitlines() if u.strip()]

        try:
            run_pipeline_task.apply_async(
                args=[str(job_dir.resolve()), saved_paths, urls, instructions, user.username],
                task_id=job_id,
            )
        except Exception as exc:
            if config.CELERY_TASK_ALWAYS_EAGER:
                raise
            # Almost always "Redis isn't running" -- say so, instead of a
            # bare 500 with a page of redis/kombu traceback.
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Could not queue the job ({type(exc).__name__}). Is Redis running at "
                    f"{config.CELERY_BROKER_URL}? To run without Redis, set "
                    "CELERY_TASK_ALWAYS_EAGER=true in backend/.env and restart the API."
                ),
            )
        db.record_job(job_id, user.username)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    return {"job_id": job_id, "status": "queued"}


_STATUS_NAMES = {
    "PENDING": "queued",
    "RECEIVED": "queued",
    "STARTED": "running",
    "PROGRESS": "running",
    "RETRY": "running",
    "SUCCESS": "done",
    "FAILURE": "error",
    "REVOKED": "error",
}


@protected.get("/jobs/{job_id}")
def job_status(job_id: str, user: User = Depends(get_current_user)):
    # Someone else's job gets the same 404 as a nonexistent one, so job
    # ids can't be probed to find out whether they exist.
    if db.get_job_owner(job_id) != user.username:
        raise HTTPException(status_code=404, detail="Job not found.")

    result = AsyncResult(job_id, app=celery_app)
    response = {"job_id": job_id, "status": _STATUS_NAMES.get(result.state, "queued")}

    if result.state == "PROGRESS" and isinstance(result.info, dict):
        response["progress"] = {"stage": result.info.get("stage", ""),
                                "detail": result.info.get("detail", ""),
                                "draft": result.info.get("draft", "")}
    elif result.state == "SUCCESS":
        data = result.result
        if data.get("pdf_filename"):
            db.set_job_pdf(job_id, data["pdf_filename"])
        response["result"] = data
    elif result.state == "FAILURE":
        exc = result.result
        # Same "Type: message" shape /process used to return directly,
        # so failures (bad URLs, missing Groq key, ...) stay debuggable.
        response["error"] = f"{type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else str(exc)
    return response


@protected.get("/download/{filename}")
def download(filename: str, user: User = Depends(get_current_user)):
    if not db.user_owns_pdf(user.username, filename):
        raise HTTPException(status_code=404, detail="Report not found.")
    path = Path(config.OUTPUT_DIR) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(str(path), media_type="application/pdf", filename=filename)


@protected.get("/evaluation-history")
def evaluation_history(limit: int = 50, user: User = Depends(get_current_user)):
    """
    Every pipeline run is logged automatically (see pipeline.py's call
    to evaluation.append_run_to_log). This just reads that log back out
    for the frontend's comparison view -- newest first, and only the
    calling user's own runs (instructions and check details describe
    their documents, the same as their jobs and reports do).
    """
    return {"runs": evaluation.read_run_history(limit=limit, username=user.username)}


app.include_router(protected)

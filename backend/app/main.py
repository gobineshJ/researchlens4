"""
FastAPI backend for ResearchLens.

  /health                 public liveness check the frontend can ping
  /auth/*                 login (OAuth2 password flow -> JWT), register, me
  /process                enqueues a pipeline run on Celery -> {job_id}
  /jobs/{job_id}          poll a run's status; holds the result when done
  /download/{filename}    the PDF report of one of YOUR runs
  /evaluation-history     YOUR runs from the run log
  /papers/arxiv/search    newest arXiv papers for a topic, minus those a feed has covered
  /papers/arxiv/process   enqueues a run over arXiv papers (downloaded by the worker)
  /papers/seen            marks papers as covered by a feed by hand

Everything except /health and /auth/token + /auth/register needs an
"Authorization: Bearer <token>" header or an "X-API-Key" header -- the
routes are mounted on a router whose dependency is auth.get_current_user.

A daily digest (n8n, cron, ...) is: search with only_new -> process the
ids it returned -> poll /jobs/{id} -> /download the PDF. The worker marks
the papers seen when the run succeeds, so tomorrow's search skips them.

CORS only admits the origins in CORS_ORIGINS, with credentials, since the
refresh token travels in a cookie. The frontend normally reaches the API
through a same-origin /api proxy and needs no CORS at all.
"""

from pathlib import Path
import shutil
import uuid
from typing import Literal

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import auth, config, db, evaluation, papers
from .auth import User, get_current_user
from .celery_app import celery_app, run_pipeline_task

app = FastAPI(title="ResearchLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type", config.API_KEY_HEADER_NAME],
)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(auth.router)

protected = APIRouter(dependencies=[Depends(get_current_user)])


def _enqueue(job_id: str, args: list, kwargs: dict | None = None) -> None:
    try:
        run_pipeline_task.apply_async(args=args, kwargs=kwargs, task_id=job_id)
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

        _enqueue(job_id, [str(job_dir.resolve()), saved_paths, urls, instructions, user.username])
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


# --- arXiv digests ---

_FEED_PATTERN = r"^[A-Za-z0-9_.-]+$"


class ArxivSearchRequest(BaseModel):
    keywords: list[str] = Field(default=[], max_length=20,
                                description="Phrases searched in all fields, OR'd together.")
    categories: list[str] = Field(default=[], max_length=20,
                                  description="arXiv categories such as cs.CL, OR'd together.")
    days: int | None = Field(default=3, ge=1, le=60,
                             description="Only papers submitted in the last N days (null = any time).")
    max_results: int = Field(default=25, ge=1, le=100)
    only_new: bool = Field(default=True, description="Leave out papers this feed has already covered.")
    feed: str = Field(default="default", min_length=1, max_length=64, pattern=_FEED_PATTERN,
                      description="Names one digest's history of covered papers, e.g. 'daily-nlp'.")


class ArxivProcessRequest(BaseModel):
    arxiv_ids: list[str] = Field(min_length=1, max_length=config.ARXIV_MAX_PAPERS_PER_RUN)
    instructions: str = Field(default="", max_length=4000,
                              description="What the report should do; empty = a per-paper summary.")
    feed: str | None = Field(default="default", min_length=1, max_length=64, pattern=_FEED_PATTERN,
                             description="Feed to mark the papers seen in once the report succeeds; "
                                         "null = don't mark them.")


class MarkSeenRequest(BaseModel):
    paper_ids: list[str] = Field(min_length=1, max_length=500)
    source: Literal["arxiv"] = "arxiv"
    feed: str = Field(default="default", min_length=1, max_length=64, pattern=_FEED_PATTERN)


def _validated_arxiv_ids(ids: list[str]) -> list[str]:
    cleaned = list(dict.fromkeys(i.strip() for i in ids if i.strip()))
    bad = [i for i in cleaned if not papers.is_valid_arxiv_id(i)]
    if bad:
        raise HTTPException(status_code=422, detail=f"Not arXiv ids: {', '.join(bad[:5])}")
    if not cleaned:
        raise HTTPException(status_code=422, detail="No arXiv ids given.")
    return cleaned


@protected.post("/papers/arxiv/search")
def arxiv_search(body: ArxivSearchRequest, user: User = Depends(get_current_user)):
    try:
        query = papers.build_query(body.keywords, body.categories, body.days)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        total, found = papers.search_arxiv(query, max_results=body.max_results)
    except papers.ArxivError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    skipped = 0
    if body.only_new:
        seen = db.seen_paper_ids(user.username, body.feed, "arxiv", [p["id"] for p in found])
        skipped = sum(p["id"] in seen for p in found)
        found = [p for p in found if p["id"] not in seen]
    return {
        "query": query,
        "feed": body.feed,
        "total_matching": total,
        "skipped_seen": skipped,
        "count": len(found),
        "papers": found,
        # Ready to pass straight to /papers/arxiv/process, which takes at
        # most ARXIV_MAX_PAPERS_PER_RUN -- newest first.
        "arxiv_ids": [p["arxiv_id"] for p in found][:config.ARXIV_MAX_PAPERS_PER_RUN],
    }


@protected.post("/papers/arxiv/process", status_code=202)
def arxiv_process(body: ArxivProcessRequest, user: User = Depends(get_current_user)):
    arxiv_ids = _validated_arxiv_ids(body.arxiv_ids)
    instructions = body.instructions.strip() or config.ARXIV_DEFAULT_INSTRUCTIONS

    # Empty for now; the worker downloads the PDFs into it, and the task
    # deletes it once the pipeline is done, same as an upload's.
    job_id = str(uuid.uuid4())
    job_dir = Path(config.UPLOAD_DIR) / job_id
    job_dir.mkdir(parents=True)
    try:
        _enqueue(job_id, [str(job_dir.resolve()), [], [], instructions, user.username],
                 kwargs={"arxiv_ids": arxiv_ids, "feed": body.feed})
        db.record_job(job_id, user.username)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    return {"job_id": job_id, "status": "queued", "arxiv_ids": arxiv_ids, "feed": body.feed}


@protected.post("/papers/seen")
def mark_seen(body: MarkSeenRequest, user: User = Depends(get_current_user)):
    """For papers covered some other way, or ones a digest should never pick."""
    ids = [papers.base_id(i) for i in _validated_arxiv_ids(body.paper_ids)]
    added = db.mark_papers_seen(user.username, body.feed, body.source, ids)
    return {"feed": body.feed, "marked": added, "already_seen": len(ids) - added}


app.include_router(protected)

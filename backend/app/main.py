"""
FastAPI backend for ResearchLens. One main endpoint (/process) runs the
whole pipeline; /health is a quick check the frontend can ping first.

CORS is wide open (allow_origins=["*"]) since this is a local training
project, not a production deployment -- tighten this before deploying
anywhere real.
"""

from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .pipeline import run_pipeline
from .groq_client import GroqNotConfigured
from . import evaluation

app = FastAPI(title="ResearchLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process")
async def process(
    files: list[UploadFile] = File(default=[]),
    seed_urls: str = Form(default=""),
    instructions: str = Form(default=""),
):
    if not files and not seed_urls.strip():
        raise HTTPException(status_code=400, detail="Provide at least one file or one seed URL.")

    tmp_dir = tempfile.mkdtemp(prefix="researchlens_")
    saved_paths = []
    try:
        for upload in files:
            dest = Path(tmp_dir) / upload.filename
            with dest.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
            saved_paths.append(str(dest))

        urls = [u.strip() for u in seed_urls.splitlines() if u.strip()]

        try:
            result = run_pipeline(saved_paths, urls, instructions)
        except GroqNotConfigured as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            # Surface the real error to the frontend instead of a bare
            # "Internal Server Error" -- makes debugging failures (bad
            # URLs, network hiccups, etc.) far faster.
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/download/{filename}")
def download(filename: str):
    from .config import OUTPUT_DIR
    path = Path(OUTPUT_DIR) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(str(path), media_type="application/pdf", filename=filename)


@app.get("/evaluation-history")
def evaluation_history(limit: int = 50):
    """
    Every /process run is logged automatically (see pipeline.py's call
    to evaluation.append_run_to_log). This just reads that log back out
    for the frontend's comparison view -- newest first.
    """
    return {"runs": evaluation.read_run_history(limit=limit)}

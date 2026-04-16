"""
main.py – FastAPI backend for the Saykin-Style AI Video Editor
Run locally:   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import uuid
import json
import logging
import asyncio
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Load .env if present (local dev)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR.mkdir(exist_ok=True)

# ── In-memory job store  {job_id: {status, progress, message, output_path, strategy}}
JOBS: dict = {}

# ── API Key (set in .env or environment variable)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ──────────────────────────────────────────────
app = FastAPI(title="Saykin AI Video Editor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "api_key_set": bool(OPENAI_API_KEY)}


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    profession: str = Form(default=""),
    api_key: str = Form(default=""),
):
    """
    Accept a video upload and kick off background processing.
    Returns job_id immediately.
    """
    # Resolve API key: form field overrides env var
    effective_key = api_key.strip() or OPENAI_API_KEY
    if not effective_key:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key is required. Paste it in the field on the page.",
        )

    # Validate file type
    allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    # Save upload
    job_id = str(uuid.uuid4())
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / f"input{suffix}"
    with open(input_path, "wb") as f_out:
        while chunk := await file.read(1024 * 1024):
            f_out.write(chunk)

    # Init job record
    JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Queued...",
        "output_path": None,
        "strategy": None,
        "error": None,
    }

    # Run processing in a background thread (FastAPI doesn't block)
    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, str(input_path), str(job_dir), effective_key, profession),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


def _progress_cb(job_id: str):
    def cb(pct: int, msg: str):
        if job_id in JOBS:
            JOBS[job_id]["progress"] = pct
            JOBS[job_id]["message"] = msg
    return cb


def _run_pipeline(
    job_id: str,
    input_path: str,
    job_dir: str,
    api_key: str,
    profession: str,
):
    """Background thread that runs the full video processing pipeline."""
    try:
        JOBS[job_id]["status"] = "processing"

        from video_processor import process_video
        from content_strategist import generate_strategy

        progress_cb = _progress_cb(job_id)

        # ── Video processing
        result = process_video(
            input_path=input_path,
            job_dir=job_dir,
            api_key=api_key,
            progress_cb=progress_cb,
        )

        progress_cb(96, "Generating content strategy...")

        # ── Content strategy
        strategy = generate_strategy(
            transcript=result["transcript"],
            profession=profession,
            api_key=api_key,
        )

        JOBS[job_id]["output_path"] = result["output_path"]
        JOBS[job_id]["strategy"] = strategy
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["message"] = "Done! Your video is ready."

    except Exception as exc:
        logger.exception("Pipeline failed for job %s", job_id)
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(exc)
        JOBS[job_id]["message"] = f"Error: {exc}"


@app.get("/status/{job_id}")
async def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "error": job.get("error"),
        "strategy": job.get("strategy") if job["status"] == "done" else None,
    }


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Video not ready yet")

    output_path = job["output_path"]
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"saykin_edit_{job_id[:8]}.mp4",
    )

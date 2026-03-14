"""
api/app.py — FastAPI application for the Smart Scrapper Agent service.

Endpoints:
    GET  /                          Redirect to interactive API docs.
    GET  /health                    Liveness check and server stats.
    POST /research                  Start a new background research job.
    GET  /research/{job_id}         Poll job status and retrieve result.
    GET  /research/{job_id}/report  Download the raw Markdown report file.
    DELETE /research/{job_id}       Remove a completed job from the store.
    GET  /jobs                      List all jobs (filterable by status).
    POST /scrape                    Scrape a single URL synchronously.
"""

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse

from agent.core import ResearchAgent
from tools.scraper import scrape_page

from .models import (
    HealthResponse,
    JobListResponse,
    JobResponse,
    JobStatus,
    ResearchRequest,
    ScrapeRequest,
    ScrapeResponse,
)

load_dotenv()

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Scrapper Agent API",
    description=(
        "An AI-powered research service that autonomously searches the web, "
        "scrapes relevant pages, summarises them with Claude, and returns a "
        "structured Markdown report.\n\n"
        "**ReAct loop:** Plan → Act → Observe → Iterate → Report"
    ),
    version="1.0.0",
    contact={"name": "Smart Scrapper Agent"},
    license_info={"name": "MIT"},
)

API_VERSION = "1.0.0"

# ── In-memory job store ───────────────────────────────────────────────────────
# Dict keyed by job_id. Values are plain dicts that mirror JobResponse fields.
# A threading.Lock ensures safe concurrent reads/writes from background threads.

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_job_or_404(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


def _job_to_response(job: dict) -> JobResponse:
    return JobResponse(**job)


# ── Background task ───────────────────────────────────────────────────────────

def _run_research(job_id: str, request: ResearchRequest) -> None:
    """Execute the ResearchAgent in a background thread and update job state."""

    # Mark as running
    with _jobs_lock:
        _jobs[job_id]["status"] = JobStatus.running

    try:
        agent = ResearchAgent(
            model=request.model,
            max_iterations=request.max_iterations,
            max_sources=request.max_sources,
            token_budget=request.token_budget,
        )
        report_path = agent.run(question=request.question)

        # Read the report content so callers can get it inline
        report_markdown = None
        if report_path and Path(report_path).exists():
            report_markdown = Path(report_path).read_text(encoding="utf-8")

        with _jobs_lock:
            _jobs[job_id].update(
                status=JobStatus.complete,
                completed_at=_now_iso(),
                report_path=report_path,
                report_markdown=report_markdown,
                sources_scraped=agent.total_scraped,
                tokens_used=agent.tokens_used,
            )

    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id].update(
                status=JobStatus.failed,
                completed_at=_now_iso(),
                error=str(exc),
            )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Redirect root to the interactive Swagger UI docs."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
def health():
    """Returns service status and current job queue statistics."""
    with _jobs_lock:
        total = len(_jobs)
        running = sum(1 for j in _jobs.values() if j["status"] == JobStatus.running)
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        jobs_total=total,
        jobs_running=running,
    )


@app.post(
    "/research",
    response_model=JobResponse,
    status_code=202,
    summary="Start a research job",
    tags=["Research"],
)
def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Start an asynchronous research job.

    The agent will:
    1. **Plan** — Claude generates focused search queries.
    2. **Act** — Search DuckDuckGo, scrape top results, summarise each page.
    3. **Observe** — Claude scores coverage; decides if more searches are needed.
    4. **Iterate** — Refines queries and repeats if coverage is insufficient.
    5. **Report** — Compiles all summaries into a Markdown report saved to `output/`.

    Returns a `job_id` immediately. Poll `GET /research/{job_id}` to check progress.
    """
    job_id = str(uuid.uuid4())
    created_at = _now_iso()

    job: dict = {
        "job_id": job_id,
        "status": JobStatus.queued,
        "question": request.question,
        "created_at": created_at,
        "completed_at": None,
        "report_path": None,
        "report_markdown": None,
        "sources_scraped": None,
        "tokens_used": None,
        "token_budget": request.token_budget,
        "error": None,
    }

    with _jobs_lock:
        _jobs[job_id] = job

    background_tasks.add_task(_run_research, job_id, request)

    return _job_to_response(job)


@app.get(
    "/research/{job_id}",
    response_model=JobResponse,
    summary="Get job status and result",
    tags=["Research"],
)
def get_research(job_id: str):
    """
    Poll a research job by its `job_id`.

    - **queued** — job is waiting to start.
    - **running** — agent is actively searching and scraping.
    - **complete** — `report_markdown` contains the full report.
    - **failed** — `error` contains the failure reason.
    """
    job = _get_job_or_404(job_id)
    return _job_to_response(job)


@app.get(
    "/research/{job_id}/report",
    summary="Download the Markdown report file",
    tags=["Research"],
    response_class=FileResponse,
)
def download_report(job_id: str):
    """
    Download the generated Markdown report as a `.md` file.
    Only available when job status is `complete`.
    """
    job = _get_job_or_404(job_id)

    if job["status"] != JobStatus.complete:
        raise HTTPException(
            status_code=409,
            detail=f"Report not ready. Current status: {job['status']}",
        )

    report_path = job.get("report_path")
    if not report_path or not Path(report_path).exists():
        raise HTTPException(status_code=404, detail="Report file not found on disk.")

    return FileResponse(
        path=report_path,
        media_type="text/markdown",
        filename=Path(report_path).name,
    )


@app.delete(
    "/research/{job_id}",
    status_code=204,
    summary="Delete a job",
    tags=["Research"],
)
def delete_job(job_id: str):
    """
    Remove a job from the in-memory store.
    Returns 409 if the job is still running.
    """
    job = _get_job_or_404(job_id)

    if job["status"] == JobStatus.running:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a running job. Wait for it to complete or restart the server.",
        )

    with _jobs_lock:
        _jobs.pop(job_id, None)


@app.get(
    "/jobs",
    response_model=JobListResponse,
    summary="List all jobs",
    tags=["Research"],
)
def list_jobs(
    status: str | None = Query(
        None,
        description="Filter by status: queued | running | complete | failed",
    ),
):
    """
    Return all jobs in the store, optionally filtered by status.
    Jobs are ordered newest-first.
    """
    with _jobs_lock:
        all_jobs = list(_jobs.values())

    if status:
        try:
            filter_status = JobStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{status}'. Choose from: queued, running, complete, failed",
            )
        all_jobs = [j for j in all_jobs if j["status"] == filter_status]

    all_jobs.sort(key=lambda j: j["created_at"], reverse=True)

    return JobListResponse(
        total=len(all_jobs),
        jobs=[_job_to_response(j) for j in all_jobs],
    )


@app.post(
    "/scrape",
    response_model=ScrapeResponse,
    summary="Scrape a single URL",
    tags=["Scrape"],
)
def scrape(request: ScrapeRequest):
    """
    Fetch and extract the main text content from a single URL **synchronously**.

    Retries up to `max_retries` times with exponential backoff on transient
    errors (timeouts, 5xx, network failures). Returns immediately with
    `success: false` on non-retryable errors (4xx).
    """
    result = scrape_page(
        url=request.url,
        timeout=request.timeout,
        max_retries=request.max_retries,
    )
    return ScrapeResponse(
        url=result["url"],
        title=result["title"],
        text=result["text"],
        char_count=len(result["text"]),
        success=result["success"],
        error=result["error"],
        attempts=result["attempts"],
    )

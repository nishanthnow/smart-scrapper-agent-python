"""
api/models.py — Pydantic request and response models for the Smart Scrapper API.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


# ── Enums ─────────────────────────────────────────────────────────────────────


class JobStatus(str, Enum):
    queued   = "queued"
    running  = "running"
    complete = "complete"
    failed   = "failed"


# ── Requests ──────────────────────────────────────────────────────────────────


class ResearchRequest(BaseModel):
    """Payload for starting a new research job."""

    question: str = Field(
        ...,
        min_length=5,
        description="The research question to investigate.",
        examples=["What are the latest breakthroughs in solid-state batteries?"],
    )
    max_sources: int = Field(
        9, ge=1, le=30,
        description="Hard cap on total pages scraped across all iterations.",
    )
    max_iterations: int = Field(
        3, ge=1, le=5,
        description="Maximum Plan→Act→Observe loops before writing the report.",
    )
    token_budget: int = Field(
        200_000, ge=10_000,
        description="Maximum total tokens (input + output) across all Claude calls.",
    )
    model: str = Field(
        "claude-opus-4-6",
        description="Anthropic model ID to use for all reasoning steps.",
    )


class ScrapeRequest(BaseModel):
    """Payload for scraping a single URL."""

    url: str = Field(
        ...,
        description="The full URL to fetch and extract text from.",
        examples=["https://en.wikipedia.org/wiki/Solid-state_battery"],
    )
    timeout: int = Field(
        20, ge=5, le=60,
        description="Per-attempt HTTP timeout in seconds.",
    )
    max_retries: int = Field(
        3, ge=1, le=5,
        description="Maximum retry attempts on transient errors.",
    )


# ── Responses ─────────────────────────────────────────────────────────────────


class JobResponse(BaseModel):
    """Status and result of a research job."""

    job_id: str             = Field(..., description="Unique job identifier.")
    status: JobStatus       = Field(..., description="Current job status.")
    question: str           = Field(..., description="The original research question.")
    created_at: str         = Field(..., description="ISO-8601 timestamp when the job was created.")
    completed_at: Optional[str]  = Field(None, description="ISO-8601 timestamp when the job finished.")
    report_path: Optional[str]   = Field(None, description="Absolute path to the saved Markdown report.")
    report_markdown: Optional[str] = Field(None, description="Full Markdown content of the report.")
    sources_scraped: Optional[int] = Field(None, description="Number of pages successfully scraped.")
    tokens_used: Optional[int]     = Field(None, description="Total tokens consumed across all Claude calls.")
    token_budget: Optional[int]    = Field(None, description="Configured token budget for the job.")
    error: Optional[str]           = Field(None, description="Error message if the job failed.")


class ScrapeResponse(BaseModel):
    """Result of a single-URL scrape."""

    url: str      = Field(..., description="The requested URL.")
    title: str    = Field(..., description="Page title extracted from <title> tag.")
    text: str     = Field(..., description="Cleaned main text content of the page.")
    char_count: int = Field(..., description="Character count of the extracted text.")
    success: bool = Field(..., description="True if the page was fetched and parsed successfully.")
    error: str    = Field(..., description="Error message if success is False.")
    attempts: int = Field(..., description="Number of HTTP attempts made (including retries).")


class HealthResponse(BaseModel):
    """API liveness and stats."""

    status: str       = Field(..., description="Always 'ok' when the service is running.")
    version: str      = Field(..., description="API version string.")
    jobs_total: int   = Field(..., description="Total jobs submitted since server start.")
    jobs_running: int = Field(..., description="Jobs currently executing in the background.")


class JobListResponse(BaseModel):
    """Paginated list of jobs."""

    total: int                  = Field(..., description="Total number of jobs in the store.")
    jobs: list[JobResponse]     = Field(..., description="List of job summaries.")

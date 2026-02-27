"""
scraper.py — Fetches a URL and extracts the main readable text content.

Features:
  - Retries up to max_retries times with exponential backoff on transient errors.
  - Distinguishes retryable errors (timeouts, 5xx, 408/429) from fatal ones (4xx).
  - Logs each attempt and wait time clearly via rich.
  - Always returns a well-formed dict so callers can check success=False and move on.
"""

import time

import httpx
from bs4 import BeautifulSoup
from rich.console import Console

console = Console()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Tags that contain navigation, ads, or boilerplate — stripped before extraction
NOISE_TAGS = [
    "script", "style", "noscript", "iframe", "svg",
    "nav", "footer", "header", "aside", "form",
    "button", "input", "select", "textarea",
]

# HTTP status codes worth retrying (server/gateway errors + rate-limit + request-timeout)
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

# Base wait in seconds — actual wait = BACKOFF_BASE ** (attempt - 1)
# Attempt 1 → 1 s, attempt 2 → 2 s, attempt 3 → 4 s
BACKOFF_BASE: float = 2.0


def scrape_page(url: str, timeout: int = 20, max_retries: int = 3) -> dict:
    """Fetch a URL and extract its main text content, retrying on transient errors.

    Args:
        url:         The full URL to scrape.
        timeout:     Per-attempt request timeout in seconds (default 20).
        max_retries: Maximum number of attempts before giving up (default 3).

    Returns:
        A dict with keys:
            - url      (str):  The requested URL.
            - title    (str):  Page title, or empty string if not found.
            - text     (str):  Cleaned main text content.
            - success  (bool): True if content was fetched successfully.
            - error    (str):  Error message if success is False, else empty string.
            - attempts (int):  Total number of attempts made.
    """
    result: dict = {
        "url": url,
        "title": "",
        "text": "",
        "success": False,
        "error": "",
        "attempts": 0,
    }

    for attempt in range(1, max_retries + 1):
        result["attempts"] = attempt

        try:
            with httpx.Client(
                headers=HEADERS,
                follow_redirects=True,
                timeout=timeout,
            ) as client:
                response = client.get(url)
                response.raise_for_status()

            # ── Parse ────────────────────────────────────────────────────
            soup = BeautifulSoup(response.text, "lxml")

            title_tag = soup.find("title")
            result["title"] = title_tag.get_text(strip=True) if title_tag else ""

            for tag in soup(NOISE_TAGS):
                tag.decompose()

            content_tag = (
                soup.find("main")
                or soup.find("article")
                or soup.find("body")
            )

            if content_tag:
                lines = [
                    line.strip()
                    for line in content_tag.get_text(separator="\n").splitlines()
                ]
                result["text"] = "\n".join(line for line in lines if line)
            else:
                result["text"] = ""

            result["success"] = True
            result["error"] = ""
            return result  # success — no need to retry

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            msg = f"HTTP {status}"
            if status in RETRYABLE_STATUS and attempt < max_retries:
                wait = BACKOFF_BASE ** (attempt - 1)
                console.print(
                    f"  [yellow]⟳ {msg} — retry {attempt}/{max_retries} "
                    f"in {wait:.0f}s[/yellow]"
                )
                time.sleep(wait)
            else:
                full_msg = f"HTTP {status} error for {url}"
                console.print(f"  [red]✗ {full_msg}[/red]")
                result["error"] = full_msg
                return result  # non-retryable 4xx or exhausted retries

        except httpx.TimeoutException:
            msg = f"Timed out after {timeout}s"
            if attempt < max_retries:
                wait = BACKOFF_BASE ** (attempt - 1)
                console.print(
                    f"  [yellow]⟳ {msg} — retry {attempt}/{max_retries} "
                    f"in {wait:.0f}s[/yellow]"
                )
                time.sleep(wait)
            else:
                full_msg = f"Request timed out after {timeout}s (tried {max_retries}x): {url}"
                console.print(f"  [red]✗ {full_msg}[/red]")
                result["error"] = full_msg

        except httpx.RequestError as e:
            msg = f"Network error: {e}"
            if attempt < max_retries:
                wait = BACKOFF_BASE ** (attempt - 1)
                console.print(
                    f"  [yellow]⟳ {msg} — retry {attempt}/{max_retries} "
                    f"in {wait:.0f}s[/yellow]"
                )
                time.sleep(wait)
            else:
                full_msg = f"Network error for {url}: {e}"
                console.print(f"  [red]✗ {full_msg}[/red]")
                result["error"] = full_msg

        except Exception as e:
            # Non-retryable unexpected error — fail immediately
            full_msg = f"Unexpected error scraping {url}: {e}"
            console.print(f"  [red]✗ {full_msg}[/red]")
            result["error"] = full_msg
            return result

    return result

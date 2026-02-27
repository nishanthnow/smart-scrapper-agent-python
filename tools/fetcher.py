"""
fetcher.py — HTTP fetching with httpx (sync), fallback to requests.
"""

import httpx
from rich.console import Console

console = Console()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_page(url: str, timeout: int = 20) -> str | None:
    """Fetch a URL and return the raw HTML string, or None on failure."""
    try:
        with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.HTTPStatusError as e:
        console.print(f"[red]HTTP error {e.response.status_code} for {url}[/red]")
    except httpx.RequestError as e:
        console.print(f"[red]Request error: {e}[/red]")
    return None

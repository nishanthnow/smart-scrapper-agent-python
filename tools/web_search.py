"""
web_search.py — Searches DuckDuckGo's HTML interface and returns relevant URLs.
No API key required.
"""

import httpx
from bs4 import BeautifulSoup
from rich.console import Console

console = Console()

DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo for a query and return a list of results.

    Each result is a dict with keys:
        - title (str): page title
        - url   (str): result URL
        - snippet (str): short description text

    Args:
        query:       The search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        A list of result dicts, or an empty list on failure.
    """
    try:
        with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=15) as client:
            response = client.post(DUCKDUCKGO_URL, data={"q": query, "b": ""})
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        results = []

        for result in soup.select(".result"):
            title_tag = result.select_one(".result__title a")
            snippet_tag = result.select_one(".result__snippet")

            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            href = title_tag.get("href", "")
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

            # DuckDuckGo HTML wraps URLs — extract the real URL from the uddg param
            if "uddg=" in href:
                from urllib.parse import urlparse, parse_qs, unquote
                parsed = urlparse(href)
                uddg = parse_qs(parsed.query).get("uddg", [None])[0]
                if uddg:
                    href = unquote(uddg)

            if href.startswith("http"):
                results.append({"title": title, "url": href, "snippet": snippet})

            if len(results) >= max_results:
                break

        return results

    except httpx.HTTPStatusError as e:
        console.print(f"[red]HTTP error during search: {e.response.status_code}[/red]")
    except httpx.RequestError as e:
        console.print(f"[red]Network error during search: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Unexpected error during search: {e}[/red]")

    return []

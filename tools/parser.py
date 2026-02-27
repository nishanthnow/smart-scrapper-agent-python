"""
parser.py — HTML parsing with BeautifulSoup and Markdown conversion.
"""

from bs4 import BeautifulSoup
from markdownify import markdownify


def parse_html(html: str) -> BeautifulSoup:
    """Parse raw HTML and strip unwanted tags."""
    soup = BeautifulSoup(html, "lxml")

    # Remove noise elements
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "head"]):
        tag.decompose()

    return soup


def to_markdown(html: str) -> str:
    """Convert HTML string to clean Markdown."""
    return markdownify(html, heading_style="ATX", strip=["a"]).strip()

"""
ScraperAgent — orchestrates fetching, parsing, and LLM-based extraction.
"""

import os
import json
from datetime import datetime
from pathlib import Path

import anthropic
from rich.console import Console
from rich.spinner import Spinner
from rich.live import Live

from tools.fetcher import fetch_page
from tools.parser import parse_html, to_markdown

console = Console()
OUTPUT_DIR = Path("output")


class ScraperAgent:
    def __init__(self, model: str = "claude-opus-4-6"):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        OUTPUT_DIR.mkdir(exist_ok=True)

    def run(self, url: str, task: str) -> str:
        # Step 1: Fetch page
        with Live(Spinner("dots", text="Fetching page..."), console=console, transient=True):
            html = fetch_page(url)

        if not html:
            return "[red]Failed to fetch the page.[/red]"

        # Step 2: Parse + convert to markdown
        with Live(Spinner("dots", text="Parsing content..."), console=console, transient=True):
            soup = parse_html(html)
            markdown_content = to_markdown(str(soup))

        # Step 3: Ask Claude to extract based on task
        with Live(Spinner("dots", text="Running AI extraction..."), console=console, transient=True):
            result = self._extract_with_claude(url, task, markdown_content)

        # Step 4: Save output
        self._save_output(url, task, result)
        return result

    def _extract_with_claude(self, url: str, task: str, content: str) -> str:
        prompt = f"""You are a precise web data extraction assistant.

URL: {url}
Task: {task}

Page content (in Markdown):
---
{content[:12000]}
---

Extract the requested information clearly and concisely. Format your response in Markdown."""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _save_output(self, url: str, task: str, result: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = OUTPUT_DIR / f"result_{timestamp}.md"
        filename.write_text(
            f"# Scrape Result\n\n**URL:** {url}\n**Task:** {task}\n\n---\n\n{result}",
            encoding="utf-8",
        )
        console.print(f"[dim]Saved → {filename}[/dim]")
        return filename

"""
agent/core.py — ResearchAgent: a ReAct-style autonomous research agent.

Loop:
  PLAN    → Claude generates focused search queries for the question.
  ACT     → web_search + scrape_page + summarize for each query.
  OBSERVE → Claude evaluates coverage; decides if more searching is needed.
  ITERATE → Refine queries and repeat if coverage is insufficient.
  OUTPUT  → write_report compiles all summaries into a Markdown report.
"""

import os
import json

import anthropic
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from tools.web_search import web_search
from tools.scraper import scrape_page
from tools.summarizer import summarize
from tools.report_writer import write_report

console = Console()


class ResearchAgent:
    """
    Autonomous research agent that implements a ReAct (Reason + Act) loop.

    Given a natural-language research question, the agent:
      1. Plans  — Claude decides how many searches and what queries to use.
      2. Acts   — Searches DuckDuckGo, scrapes top results, summarizes each page.
      3. Observes — Claude scores coverage and flags gaps.
      4. Iterates — Refines queries and repeats until coverage is sufficient.
      5. Outputs — Writes a structured Markdown report to output/.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        max_iterations: int = 3,
        results_per_query: int = 3,
        queries_per_plan: int = 3,
        max_sources: int = 9,
        token_budget: int = 200_000,
    ):
        """
        Args:
            model:             Anthropic model ID used for all reasoning steps.
            max_iterations:    Hard cap on Plan→Act→Observe loops (default 3).
            results_per_query: Number of search results to scrape per query (default 3).
            queries_per_plan:  Number of queries Claude generates per planning step (default 3).
            max_sources:       Hard cap on total pages scraped across all iterations (default 9).
            token_budget:      Maximum total tokens (input + output) across all Claude calls
                               in a single run. Agent stops making new calls once this is
                               exceeded (default 200,000).
        """
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_iterations = max_iterations
        self.results_per_query = results_per_query
        self.queries_per_plan = queries_per_plan
        self.max_sources = max_sources
        self.token_budget = token_budget
        # Counters reset at the start of each run()
        self.total_scraped: int = 0
        self.tokens_used: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, question: str) -> str:
        """
        Execute the full ReAct research loop.

        Args:
            question: The research question to investigate.

        Returns:
            Absolute path to the saved Markdown report, or empty string on failure.
        """
        console.print()
        console.print(
            Panel.fit(
                f"[bold white]{question}[/bold white]",
                title="[bold cyan]  Research Agent[/bold cyan]",
                subtitle="[dim]ReAct Loop[/dim]",
                border_style="cyan",
                padding=(1, 4),
            )
        )

        all_summaries: list[dict] = []
        scraped_urls: set[str] = set()
        iteration = 0
        self.total_scraped = 0   # reset for this run
        self.tokens_used = 0     # reset for this run

        # ── PLAN ────────────────────────────────────────────────────────
        queries = self._plan(question)

        # ── LOOP ────────────────────────────────────────────────────────
        while iteration < self.max_iterations:
            iteration += 1
            console.print(
                Rule(
                    f"[bold yellow]Iteration {iteration} of {self.max_iterations}[/bold yellow]",
                    style="yellow",
                )
            )

            # ── ACT ─────────────────────────────────────────────────────
            new_summaries = self._act(queries, question, scraped_urls)
            all_summaries.extend(new_summaries)
            self.total_scraped = len(scraped_urls)

            # ── OBSERVE ─────────────────────────────────────────────────
            observation = self._observe(question, all_summaries, iteration)
            self._log_observation(observation)

            budget_exhausted = self.tokens_used >= self.token_budget
            if observation.get("sufficient") or iteration >= self.max_iterations or budget_exhausted:
                if budget_exhausted:
                    console.print(
                        "[red]⚑ Token budget exhausted — stopping loop and writing report.[/red]\n"
                    )
                break

            # ── ITERATE ─────────────────────────────────────────────────
            refinements = observation.get("refinements", [])
            if not refinements:
                console.print("[dim]No refinement queries suggested — stopping early.[/dim]\n")
                break
            queries = refinements

        # ── OUTPUT ──────────────────────────────────────────────────────
        return self._output(question, all_summaries)

    # ------------------------------------------------------------------
    # Phase: PLAN
    # ------------------------------------------------------------------

    def _plan(self, question: str) -> list[str]:
        """
        Ask Claude to generate focused search queries for the research question.

        Returns:
            A list of query strings.
        """
        console.print()
        console.print(
            Panel(
                "[bold]Deciding search strategy...[/bold]",
                title="[cyan]PLAN[/cyan]",
                border_style="cyan",
            )
        )

        prompt = f"""You are a research planning assistant.

Given the research question below, generate exactly {self.queries_per_plan} focused web search \
queries that together will help gather comprehensive, accurate information.

Research question: {question}

Rules:
- Each query should target a distinct angle or sub-topic of the question.
- Queries should be concise (3–8 words), specific, and optimised for search engines.
- Do NOT include explanations — return ONLY valid JSON.

Return this exact JSON structure (no markdown fences):
{{
  "reasoning": "One or two sentences explaining your search strategy.",
  "queries": ["query 1", "query 2", "query 3"]
}}"""

        response = self._call_claude(prompt, max_tokens=512, label="PLAN")
        parsed = self._parse_json(response)

        reasoning = parsed.get("reasoning", "")
        queries: list[str] = parsed.get("queries") or [question]

        if reasoning:
            console.print(f"  [dim italic]Strategy: {reasoning}[/dim italic]")
        console.print()

        table = Table(show_header=False, box=None, padding=(0, 2))
        for i, q in enumerate(queries, 1):
            table.add_row(
                Text(f"{i}.", style="cyan bold"),
                Text(q, style="white"),
            )
        console.print(table)
        console.print()

        return queries

    # ------------------------------------------------------------------
    # Phase: ACT
    # ------------------------------------------------------------------

    def _act(
        self,
        queries: list[str],
        question: str,
        scraped_urls: set[str],
    ) -> list[dict]:
        """
        For each query: search → scrape top results → summarize relevant content.

        Args:
            queries:      List of search query strings.
            question:     The original research question (used for summarization).
            scraped_urls: Set of URLs already scraped (mutated in-place to avoid duplicates).

        Returns:
            List of summary dicts: {url, title, summary, query}.
        """
        console.print(
            Panel(
                "[bold]Searching the web and extracting content...[/bold]",
                title="[yellow]ACT[/yellow]",
                border_style="yellow",
            )
        )

        all_summaries: list[dict] = []

        for query in queries:
            # Check cap before even searching
            if len(scraped_urls) >= self.max_sources:
                console.print(
                    f"\n[dim yellow]⚑ Source cap ({self.max_sources}) reached "
                    f"— skipping remaining queries.[/dim yellow]"
                )
                break

            console.print(f"\n[bold yellow]›[/bold yellow] Query: [italic]{query}[/italic]")

            # Search
            with Live(Spinner("dots", text="Searching DuckDuckGo..."), console=console, transient=True):
                results = web_search(query, max_results=self.results_per_query)

            if not results:
                console.print("  [red]No search results returned.[/red]")
                continue

            # Display results table
            t = Table(
                show_header=True,
                header_style="bold dim",
                box=None,
                padding=(0, 1),
            )
            t.add_column("#", style="dim", width=3)
            t.add_column("Title", style="white", max_width=48, no_wrap=True)
            t.add_column("URL", style="blue dim", max_width=52, no_wrap=True)
            for i, r in enumerate(results, 1):
                t.add_row(str(i), r["title"][:48], r["url"][:52])
            console.print(t)

            # Scrape + summarize
            for result in results:
                # Enforce global source cap
                if len(scraped_urls) >= self.max_sources:
                    console.print(
                        f"  [dim yellow]⚑ Source cap reached "
                        f"({self.max_sources} pages) — skipping remaining results.[/dim yellow]"
                    )
                    break

                url = result["url"]

                if url in scraped_urls:
                    console.print(f"  [dim]↷ Already scraped: {url[:65]}[/dim]")
                    continue
                scraped_urls.add(url)

                console.print(f"  [dim]↓ Scraping: {url[:70]}[/dim]")
                with Live(Spinner("dots", text="Fetching page..."), console=console, transient=True):
                    page = scrape_page(url)

                if not page["success"] or not page["text"].strip():
                    console.print("  [red]✗ Scrape failed or page is empty — skipping.[/red]")
                    continue

                # Token budget check before calling Claude
                if self.tokens_used >= self.token_budget:
                    console.print(
                        f"  [red]⚑ Token budget ({self.token_budget:,}) exhausted "
                        f"— skipping summarization.[/red]"
                    )
                    continue

                char_count = len(page["text"])
                console.print(f"  [dim]∑ Summarizing {char_count:,} chars...[/dim]")
                with Live(Spinner("dots", text="Summarizing with Claude..."), console=console, transient=True):
                    summary_result = summarize(page["text"], question)

                # Accumulate tokens regardless of success
                self.tokens_used += summary_result.get("tokens_used", 0)

                if not summary_result["success"]:
                    console.print(f"  [red]✗ Summarization error: {summary_result['error']}[/red]")
                    continue

                summary_text = summary_result["summary"]
                if summary_text == "No relevant information found.":
                    console.print("  [dim yellow]~ No relevant content on this page.[/dim yellow]")
                    continue

                console.print("  [green]✓ Relevant content extracted.[/green]")
                all_summaries.append(
                    {
                        "url": url,
                        "title": page["title"] or result["title"],
                        "summary": summary_text,
                        "query": query,
                    }
                )

        remaining = self.max_sources - len(scraped_urls)
        console.print()
        console.print(
            f"  [bold]ACT complete:[/bold] {len(all_summaries)} useful summary(ies) this round  "
            f"[dim]({len(scraped_urls)}/{self.max_sources} sources used, "
            f"{max(remaining, 0)} slot(s) remaining)[/dim]"
        )
        return all_summaries

    # ------------------------------------------------------------------
    # Phase: OBSERVE
    # ------------------------------------------------------------------

    def _observe(
        self,
        question: str,
        summaries: list[dict],
        iteration: int,
    ) -> dict:
        """
        Ask Claude to evaluate whether the gathered information is sufficient.

        Args:
            question:   The research question.
            summaries:  All summaries collected so far.
            iteration:  Current iteration number (used for context).

        Returns:
            Dict with keys: sufficient (bool), reason (str),
            coverage_score (int 1-10), refinements (list[str]).
        """
        console.print()
        console.print(
            Panel(
                "[bold]Evaluating gathered information...[/bold]",
                title="[magenta]OBSERVE[/magenta]",
                border_style="magenta",
            )
        )

        if not summaries:
            console.print("  [dim]No summaries available yet.[/dim]")
            return {
                "sufficient": False,
                "reason": "No information gathered yet.",
                "coverage_score": 0,
                "refinements": [question],
            }

        combined = "\n\n---\n\n".join(
            f"Source {i}: {s['url']}\n{s['summary']}"
            for i, s in enumerate(summaries, 1)
        )

        prompt = f"""You are evaluating the progress of a web research task.

Research question: {question}
Iteration: {iteration} of {self.max_iterations}

Information gathered so far ({len(summaries)} source(s)):
---
{combined[:10_000]}
---

Assess whether this information is sufficient to write a comprehensive, accurate answer \
to the research question.

Return ONLY valid JSON with this exact structure (no markdown fences):
{{
  "sufficient": true or false,
  "reason": "One or two sentences explaining your assessment.",
  "coverage_score": <integer 1-10>,
  "refinements": ["refined query 1", "refined query 2"]
}}

Guidelines:
- "sufficient": true if the information covers the question well; false if clear gaps remain.
- "coverage_score": 1 (nothing useful) to 10 (fully covered).
- "refinements": 1–3 new, targeted queries to fill the gaps (only if sufficient is false).
- If sufficient is true, "refinements" may be an empty list."""

        response = self._call_claude(prompt, max_tokens=512, label="OBSERVE")
        parsed = self._parse_json(response)

        parsed.setdefault("sufficient", len(summaries) >= self.queries_per_plan * 2)
        parsed.setdefault("reason", "")
        parsed.setdefault("coverage_score", 5)
        parsed.setdefault("refinements", [])

        return parsed

    def _log_observation(self, observation: dict) -> None:
        """Render the observation result to the terminal."""
        score: int | str = observation.get("coverage_score", "?")
        reason: str = observation.get("reason", "")
        sufficient: bool = observation.get("sufficient", False)
        refinements: list[str] = observation.get("refinements", [])

        score_color = "green" if int(score or 0) >= 7 else "yellow" if int(score or 0) >= 4 else "red"
        status_color = "green" if sufficient else "yellow"
        status_label = "SUFFICIENT ✓" if sufficient else "NEEDS MORE INFO"

        console.print(
            f"  Coverage: [bold {score_color}]{score}/10[/bold {score_color}]  "
            f"Status: [bold {status_color}]{status_label}[/bold {status_color}]"
        )
        if reason:
            console.print(f"  [dim italic]{reason}[/dim italic]")

        if not sufficient and refinements:
            console.print("  [magenta]Refinement queries:[/magenta]")
            for q in refinements:
                console.print(f"    [dim cyan]→ {q}[/dim cyan]")
        console.print()

    # ------------------------------------------------------------------
    # Phase: OUTPUT
    # ------------------------------------------------------------------

    def _output(self, question: str, summaries: list[dict]) -> str:
        """
        Compile all gathered summaries into a final Markdown report.

        Args:
            question:  The research question.
            summaries: All summaries collected across all iterations.

        Returns:
            Absolute path to the saved report file, or empty string on failure.
        """
        console.print(Rule("[bold green]Generating Final Report[/bold green]", style="green"))
        console.print(
            Panel(
                f"[bold]Compiling {len(summaries)} source(s) into a report...[/bold]",
                title="[green]OUTPUT[/green]",
                border_style="green",
            )
        )

        result = write_report(question=question, summaries=summaries)

        if result["success"]:
            pct = self.tokens_used / self.token_budget * 100
            console.print()
            console.print(
                Panel.fit(
                    f"[bold green]Research complete![/bold green]\n\n"
                    f"  Sources : [cyan]{len(summaries)}[/cyan]\n"
                    f"  Tokens  : [cyan]{self.tokens_used:,} / {self.token_budget:,} "
                    f"({pct:.0f}% of budget)[/cyan]\n"
                    f"  Report  : [cyan]{result['path']}[/cyan]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
            return result["path"]

        console.print(f"[red]Failed to write report: {result['error']}[/red]")
        return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_claude(self, prompt: str, max_tokens: int = 1024, label: str = "") -> str:
        """
        Send a single-turn prompt to Claude, track token usage, and return the text response.

        Skips the API call and returns "{}" immediately if the token budget is already
        exhausted, allowing callers to degrade gracefully.

        Args:
            prompt:     The user prompt string.
            max_tokens: Maximum tokens in the response.
            label:      Optional description logged alongside token usage (e.g. "PLAN").

        Returns:
            Raw text response from Claude, or "{}" on budget exhaustion or API error.
        """
        # ── Pre-call budget check ───────────────────────────────────────
        if self.tokens_used >= self.token_budget:
            console.print(
                f"  [red]⚑ Token budget ({self.token_budget:,}) exhausted "
                f"— skipping{' ' + label if label else ''} call.[/red]"
            )
            return "{}"

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            call_tokens = message.usage.input_tokens + message.usage.output_tokens
            self.tokens_used += call_tokens

            pct = self.tokens_used / self.token_budget * 100
            tag = f"[{label}] " if label else ""
            budget_color = "green" if pct < 70 else "yellow" if pct < 90 else "red"
            console.print(
                f"  [dim]{tag}Tokens: +{call_tokens:,} → "
                f"[{budget_color}]{self.tokens_used:,} / {self.token_budget:,} "
                f"({pct:.0f}%)[/{budget_color}][/dim]"
            )
            return message.content[0].text.strip()

        except anthropic.AuthenticationError:
            console.print("[red]Authentication failed — check ANTHROPIC_API_KEY.[/red]")
        except anthropic.RateLimitError:
            console.print("[yellow]Rate limit reached — consider reducing concurrency.[/yellow]")
        except anthropic.APIError as e:
            console.print(f"[red]Anthropic API error: {e}[/red]")
        return "{}"

    def _parse_json(self, text: str) -> dict:
        """
        Parse a JSON object from a Claude response, stripping markdown code fences if present.

        Args:
            text: Raw response text from Claude.

        Returns:
            Parsed dict, or {} if parsing fails.
        """
        try:
            # Strip ```json ... ``` or ``` ... ``` fences
            if "```" in text:
                parts = text.split("```")
                # parts[1] is inside the fences
                inner = parts[1]
                if inner.startswith("json"):
                    inner = inner[4:]
                text = inner
            return json.loads(text.strip())
        except (json.JSONDecodeError, IndexError):
            console.print("[yellow dim]Warning: Could not parse JSON from Claude response.[/yellow dim]")
            return {}

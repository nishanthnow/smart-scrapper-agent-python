"""
smart-scrapper-agent — CLI entry point.

Usage:
    python main.py "What is quantum computing?"
    python main.py "Best Python web frameworks 2025" --max-sources 6 --max-iterations 2
    python main.py "Explain zero-knowledge proofs" --token-budget 50000
    python main.py --model claude-haiku-4-5-20251001 "Rust vs Go benchmarks"
    python main.py          # interactive prompt fallback

Flags:
    question              Research question (positional, optional — prompted if omitted).
    --max-sources  N      Hard cap on total pages scraped across all iterations (default: 9).
    --max-iterations N    Maximum Plan→Act→Observe loops (default: 3).
    --token-budget N      Max total tokens (input+output) across all Claude calls (default: 200000).
    --model NAME          Anthropic model ID (default: claude-opus-4-6).
    --mode {1,2}          1 = ResearchAgent (default), 2 = ScraperAgent (single URL).
"""

import argparse
import os
import sys
import time

# ── Step 1: load .env if python-dotenv is installed ──────────────────────────
# This reads ANTHROPIC_API_KEY from a .env file in the project folder.
# If python-dotenv is not installed the key must be set as a system/shell variable.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

# ── Step 2: check all required packages are installed ────────────────────────
_MISSING: list[str] = []
for _pkg, _import in [
    ("rich",           "rich"),
    ("httpx",          "httpx"),
    ("beautifulsoup4", "bs4"),
    ("anthropic",      "anthropic"),
    ("markdownify",    "markdownify"),
    ("lxml",           "lxml"),
]:
    try:
        __import__(_import)
    except ModuleNotFoundError:
        _MISSING.append(_pkg)

if _MISSING:
    sys.exit(
        f"ERROR: Missing packages: {', '.join(_MISSING)}\n\n"
        f"Install everything at once:\n"
        f"    pip install -r requirements.txt\n\n"
        f"Then run again:\n"
        f"    python main.py \"your question here\""
    )

# ── Step 3: check ANTHROPIC_API_KEY is set ───────────────────────────────────
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit(
        "ERROR: ANTHROPIC_API_KEY is not set.\n\n"
        "Option A — set it in your shell for this session:\n"
        "    PowerShell:  $env:ANTHROPIC_API_KEY = \"sk-ant-...\"\n"
        "    CMD:         set ANTHROPIC_API_KEY=sk-ant-...\n\n"
        "Option B — create a .env file in the project folder:\n"
        "    echo ANTHROPIC_API_KEY=sk-ant-... > .env\n\n"
        "Option C — set it permanently via System Environment Variables\n"
        "    (Windows Settings → Search 'environment variables')"
    )

# ── Step 4: project imports (safe — all deps verified above) ─────────────────
from rich.console import Console        # noqa: E402
from rich.prompt import Prompt          # noqa: E402
from rich.rule import Rule              # noqa: E402
from rich.table import Table            # noqa: E402
from rich.text import Text              # noqa: E402

from agent.core import ResearchAgent            # noqa: E402
from agent.scraper_agent import ScraperAgent    # noqa: E402

console = Console()


# ── Argument parsing ─────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart-scrapper-agent",
        description="AI-powered research agent: search → scrape → summarize → report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Research question (prompted interactively if omitted).",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=9,
        metavar="N",
        help="Hard cap on total pages scraped across all iterations (default: 9).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        metavar="N",
        help="Maximum Plan→Act→Observe loops (default: 3).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-6",
        metavar="MODEL",
        help="Anthropic model ID (default: claude-opus-4-6).",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=200_000,
        metavar="N",
        help="Max total tokens (input+output) across all Claude calls (default: 200000).",
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2],
        default=1,
        help="1 = ResearchAgent (default), 2 = ScraperAgent (single URL).",
    )
    return parser


# ── Helpers ──────────────────────────────────────────────────────────────────


def print_header(args: argparse.Namespace) -> None:
    """Print a startup banner showing the active configuration."""
    console.print()
    console.print(Rule("[bold cyan]Smart Scrapper Agent[/bold cyan]"))
    console.print("[dim]Powered by Anthropic Claude[/dim]\n", justify="center")

    cfg = Table(show_header=False, box=None, padding=(0, 2))
    cfg.add_row(Text("Model", style="dim"), Text(args.model, style="cyan"))
    if args.mode == 1:
        cfg.add_row(Text("Max sources", style="dim"), Text(str(args.max_sources), style="cyan"))
        cfg.add_row(Text("Max iterations", style="dim"), Text(str(args.max_iterations), style="cyan"))
        cfg.add_row(Text("Token budget", style="dim"), Text(f"{args.token_budget:,}", style="cyan"))
    cfg.add_row(
        Text("Mode", style="dim"),
        Text("Research Agent" if args.mode == 1 else "Scraper Agent", style="cyan"),
    )
    console.print(cfg)
    console.print()


def print_summary(
    report_path: str,
    elapsed: float,
    sources: int,
    tokens_used: int,
    token_budget: int,
) -> None:
    """Print a final summary table after the agent finishes."""
    console.print()
    console.print(Rule("[bold green]Run Summary[/bold green]", style="green"))

    pct = tokens_used / token_budget * 100 if token_budget else 0
    budget_color = "green" if pct < 70 else "yellow" if pct < 90 else "red"

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_row(Text("Status", style="dim"), Text("Complete ✓", style="bold green"))
    t.add_row(Text("Sources scraped", style="dim"), Text(str(sources), style="cyan"))
    t.add_row(
        Text("Tokens used", style="dim"),
        Text(f"{tokens_used:,} / {token_budget:,} ({pct:.0f}%)", style=budget_color),
    )
    t.add_row(Text("Elapsed", style="dim"), Text(f"{elapsed:.1f}s", style="cyan"))
    t.add_row(Text("Report", style="dim"), Text(report_path or "—", style="cyan"))
    console.print(t)
    console.print()


# ── Mode runners ─────────────────────────────────────────────────────────────


def run_research_agent(args: argparse.Namespace, question: str) -> None:
    """Run the ReAct ResearchAgent and display a progress log."""
    agent = ResearchAgent(
        model=args.model,
        max_iterations=args.max_iterations,
        max_sources=args.max_sources,
        token_budget=args.token_budget,
    )

    start = time.perf_counter()
    report_path = agent.run(question=question)
    elapsed = time.perf_counter() - start

    print_summary(
        report_path=report_path,
        elapsed=elapsed,
        sources=agent.total_scraped,
        tokens_used=agent.tokens_used,
        token_budget=args.token_budget,
    )


def run_scraper_agent(args: argparse.Namespace) -> None:
    """Run the single-URL ScraperAgent."""
    url = Prompt.ask("\n[bold yellow]URL to scrape[/bold yellow]")
    task = Prompt.ask("[bold yellow]What to extract[/bold yellow]")

    agent = ScraperAgent(model=args.model)
    console.print("\n[bold green]Starting scraper...[/bold green]\n")

    start = time.perf_counter()
    result = agent.run(url=url, task=task)
    elapsed = time.perf_counter() - start

    console.print(result)
    console.print(
        f"\n[bold green]Done[/bold green] in [cyan]{elapsed:.1f}s[/cyan] — "
        f"output saved to [cyan]output/[/cyan]"
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print_header(args)

    if args.mode == 2:
        run_scraper_agent(args)
        return

    # Resolve question — CLI arg takes priority, otherwise prompt
    if args.question:
        question = args.question
        console.print(f"[bold yellow]Question:[/bold yellow] {question}\n")
    else:
        question = Prompt.ask("[bold yellow]Research question[/bold yellow]")

    if not question.strip():
        console.print("[red]No question provided. Exiting.[/red]")
        sys.exit(1)

    run_research_agent(args, question)


if __name__ == "__main__":
    main()

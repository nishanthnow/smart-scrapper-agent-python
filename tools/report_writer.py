"""
report_writer.py — Compiles a list of summaries into a clean Markdown report
and saves it to the output/ folder.
"""

from datetime import datetime
from pathlib import Path
from rich.console import Console

console = Console()

OUTPUT_DIR = Path("output")


def write_report(
    question: str,
    summaries: list[dict],
    filename: str | None = None,
) -> dict:
    """Compile summaries into a structured Markdown report and save to output/.

    Args:
        question:   The original research question or topic.
        summaries:  A list of summary dicts, each containing:
                        - url     (str): source URL
                        - title   (str): page title
                        - summary (str): extracted summary text
        filename:   Optional filename (without path). If omitted, a timestamped
                    name is generated automatically (e.g. report_20260226_153045.md).

    Returns:
        A dict with keys:
            - path    (str):  Absolute path to the saved report file.
            - success (bool): True if the file was written successfully.
            - error   (str):  Error message if success is False, else empty string.
    """
    result = {"path": "", "success": False, "error": ""}

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.md"

        output_path = OUTPUT_DIR / filename

        lines = _build_report(question, summaries)
        output_path.write_text("\n".join(lines), encoding="utf-8")

        console.print(f"[green]Report saved →[/green] [cyan]{output_path}[/cyan]")
        result["path"] = str(output_path.resolve())
        result["success"] = True

    except OSError as e:
        msg = f"Failed to write report: {e}"
        console.print(f"[red]{msg}[/red]")
        result["error"] = msg

    except Exception as e:
        msg = f"Unexpected error writing report: {e}"
        console.print(f"[red]{msg}[/red]")
        result["error"] = msg

    return result


def _build_report(question: str, summaries: list[dict]) -> list[str]:
    """Build report lines from question and summaries."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    valid = [s for s in summaries if s.get("summary") and
             s["summary"] != "No relevant information found."]
    skipped = len(summaries) - len(valid)

    lines = [
        "# Research Report",
        "",
        f"**Question:** {question}",
        f"**Generated:** {now}",
        f"**Sources:** {len(valid)} relevant / {len(summaries)} total",
        "",
        "---",
        "",
    ]

    if not valid:
        lines += [
            "> No relevant information was found across the searched sources.",
            "",
        ]
        return lines

    lines += ["## Findings", ""]

    for i, item in enumerate(valid, start=1):
        title = item.get("title") or item.get("url", f"Source {i}")
        url = item.get("url", "")
        summary = item.get("summary", "").strip()

        lines += [
            f"### {i}. {title}",
            "",
        ]
        if url:
            lines += [f"**Source:** [{url}]({url})", ""]

        lines += [summary, "", "---", ""]

    if skipped > 0:
        lines += [
            "",
            f"*{skipped} source(s) contained no relevant information and were omitted.*",
        ]

    return lines

"""
summarizer.py — Uses the Anthropic API to extract and summarize content
relevant to a specific question from raw text.
"""

import os
import anthropic
from rich.console import Console

console = Console()

DEFAULT_MODEL = "claude-opus-4-6"
MAX_INPUT_CHARS = 15_000  # Truncate very long pages before sending


def summarize(text: str, question: str, model: str = DEFAULT_MODEL) -> dict:
    """Summarize only the parts of `text` that are relevant to `question`.

    Args:
        text:     Raw text content scraped from a web page.
        question: The user's research question or topic of interest.
        model:    Anthropic model ID to use (default: claude-opus-4-6).

    Returns:
        A dict with keys:
            - summary (str):  The focused summary from Claude.
            - success (bool): True if the API call succeeded.
            - error   (str):  Error message if success is False, else empty string.
    """
    result = {"summary": "", "success": False, "error": "", "tokens_used": 0}

    if not text.strip():
        result["error"] = "Empty text provided — nothing to summarize."
        console.print("[yellow]Summarizer: received empty text, skipping.[/yellow]")
        return result

    truncated_text = text[:MAX_INPUT_CHARS]
    if len(text) > MAX_INPUT_CHARS:
        console.print(
            f"[dim]Summarizer: text truncated to {MAX_INPUT_CHARS:,} chars.[/dim]"
        )

    prompt = f"""You are a precise research assistant. Your job is to read the page content below and extract only the information that is directly relevant to the question.

Question: {question}

Page content:
---
{truncated_text}
---

Instructions:
- Focus strictly on information relevant to the question.
- Ignore navigation text, ads, and unrelated content.
- Be concise but complete — do not omit important relevant details.
- Write in clear, neutral prose. Use bullet points only when listing distinct items.
- If the page contains no relevant information, respond with: "No relevant information found."
"""

    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        result["summary"] = message.content[0].text.strip()
        result["tokens_used"] = message.usage.input_tokens + message.usage.output_tokens
        result["success"] = True

    except anthropic.AuthenticationError:
        msg = "Invalid or missing ANTHROPIC_API_KEY."
        console.print(f"[red]{msg}[/red]")
        result["error"] = msg

    except anthropic.RateLimitError:
        msg = "Anthropic rate limit reached. Try again shortly."
        console.print(f"[yellow]{msg}[/yellow]")
        result["error"] = msg

    except anthropic.APIError as e:
        msg = f"Anthropic API error: {e}"
        console.print(f"[red]{msg}[/red]")
        result["error"] = msg

    except Exception as e:
        msg = f"Unexpected error during summarization: {e}"
        console.print(f"[red]{msg}[/red]")
        result["error"] = msg

    return result

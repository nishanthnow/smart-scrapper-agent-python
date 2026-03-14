"""
serve.py — Start the Smart Scrapper Agent REST API server.

Usage:
    python serve.py                         # default: localhost:8000
    python serve.py --host 0.0.0.0          # expose to local network
    python serve.py --port 9000             # custom port
    python serve.py --reload                # auto-reload on code changes (dev mode)
    python serve.py --workers 4             # multiple worker processes (prod)

Interactive docs available at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

import argparse
import os
import sys

# ── Dependency + API key checks ───────────────────────────────────────────────

_MISSING: list[str] = []
for _pkg, _import in [
    ("fastapi",    "fastapi"),
    ("uvicorn",    "uvicorn"),
    ("rich",       "rich"),
    ("httpx",      "httpx"),
    ("anthropic",  "anthropic"),
    ("bs4",        "bs4"),
]:
    try:
        __import__(_import)
    except ModuleNotFoundError:
        _MISSING.append(_pkg)

if _MISSING:
    sys.exit(
        f"ERROR: Missing packages: {', '.join(_MISSING)}\n\n"
        f"Install everything:\n"
        f"    pip install -r requirements.txt\n"
    )

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit(
        "ERROR: ANTHROPIC_API_KEY is not set.\n\n"
        "Option A — create a .env file in this folder:\n"
        "    ANTHROPIC_API_KEY=sk-ant-...\n\n"
        "Option B — set it in your shell:\n"
        "    PowerShell:  $env:ANTHROPIC_API_KEY = 'sk-ant-...'\n"
        "    CMD:         set ANTHROPIC_API_KEY=sk-ant-...\n"
    )

import uvicorn  # noqa: E402 — imported after checks


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="serve.py",
        description="Start the Smart Scrapper Agent API server.",
    )
    parser.add_argument("--host",    default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port",    default=8000, type=int, help="Bind port (default: 8000)")
    parser.add_argument("--reload",  action="store_true", help="Enable auto-reload (dev mode)")
    parser.add_argument("--workers", default=1, type=int, help="Number of worker processes (default: 1)")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print(f"\n  Smart Scrapper Agent API")
    print(f"  ─────────────────────────────────────────")
    print(f"  Listening : http://{args.host}:{args.port}")
    print(f"  Docs      : http://{args.host}:{args.port}/docs")
    print(f"  ReDoc     : http://{args.host}:{args.port}/redoc")
    print(f"  Workers   : {args.workers}")
    print(f"  Reload    : {args.reload}")
    print(f"  ─────────────────────────────────────────\n")

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()

"""Package init.

Loads a local .env into the process environment so the reply-analysis backend
picks up an API key without the caller having to export it first. Only
`docker compose` reads .env on its own; `python run_daily.py` and
`uvicorn app.main:app` do not, and a missing key fails silently by falling
back to the lexicon, which is easy to mistake for the LLM path working.

Stdlib only, and real environment variables always win over the file.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> None:
    env_path = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    try:
        raw = env_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()

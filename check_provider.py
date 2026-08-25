"""Verify the configured reply-analysis backend actually works.

    python check_provider.py

Reads .env (via app/__init__.py), reports which provider is selected, makes one
live call, and prints the parsed analysis. Exit code 0 = the LLM path works,
1 = no key configured, 2 = a key is configured but the call failed.

Worth running because a bad key is silent: analyse() falls back to the lexicon
and the pipeline looks perfectly healthy.
"""
from __future__ import annotations

import sys
import time

from app import nlp

SAMPLE = ("Honestly I'm exhausted. My manager never acknowledges anything I do "
          "and I've started interviewing elsewhere.")


def main() -> int:
    active = nlp.active_provider()
    if active is None:
        print("no provider configured — reply analysis will use the lexicon.")
        print("set GROQ_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY in .env")
        return 1

    name, key, model = active
    print(f"provider : {name}")
    print(f"model    : {model}")
    print(f"key      : {key[:7]}…{key[-4:]} ({len(key)} chars)")
    print(f"\ncalling {name}…")

    t0 = time.time()
    try:
        result = nlp._analyse_llm(SAMPLE, name, key, model)
    except Exception as exc:                      # noqa: BLE001 - diagnostic
        print(f"FAILED after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}")
        return 2
    elapsed = time.time() - t0

    if result is None:
        print(f"FAILED after {elapsed:.1f}s — the call returned nothing.")
        print("The pipeline will still run, silently, on the lexicon. Likely causes:")
        print("  * key rejected (401)      -> regenerate it in the provider console")
        print("  * rate limited (429)      -> retries exhausted; wait and retry")
        print("  * model name wrong (404)  -> check VIBEBOT_MODEL / the default above")
        print("  * no network route        -> proxy or firewall blocking the host")
        return 2

    print(f"OK in {elapsed:.1f}s — the LLM path is live.\n")
    print(f"  sentiment : {result.sentiment}")
    print(f"  tone      : {result.tone}")
    print(f"  themes    : {result.themes}")
    print(f"  flags     : {result.flags}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

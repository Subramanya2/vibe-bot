"""Find out exactly why the provider call is failing.

    python diagnose_groq.py

check_provider.py tells you *that* it failed; this tells you *why*. It bypasses
the retry-and-swallow logic in nlp._analyse_llm and reports the raw status code
and response body from each step. Never prints the key itself.
"""
from __future__ import annotations

import os
import socket
import sys

import httpx

from app import nlp   # importing app loads .env


def mask(k: str) -> str:
    return f"{k[:7]}…{k[-4:]}" if len(k) > 12 else "…"


def step(n: str) -> None:
    print(f"\n{'-' * 62}\n{n}\n{'-' * 62}")


def main() -> int:
    active = nlp.active_provider()
    if active is None:
        print("No key found. Check that .env sits next to run_daily.py and "
              "contains GROQ_API_KEY=gsk_...")
        return 1
    provider, key, model = active

    step("1. the key as the code sees it")
    print(f"provider      : {provider}")
    print(f"model         : {model}")
    print(f"key           : {mask(key)}")
    print(f"length        : {len(key)}  (Groq keys are normally 56)")
    print(f"prefix ok     : {key.startswith('gsk_')}")
    problems = []
    if key != key.strip():
        problems.append("leading/trailing whitespace")
    if key[:1] in "\"'" or key[-1:] in "\"'":
        problems.append("wrapping quotes")
    if any(c in key for c in "\r\n"):
        problems.append("embedded newline (CRLF?)")
    if not key.replace("_", "").isalnum():
        problems.append("unexpected characters")
    print(f"hygiene       : {', '.join(problems) if problems else 'clean'}")

    step("2. can this machine reach api.groq.com at all")
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "NO_PROXY"):
        if os.getenv(var):
            print(f"proxy env     : {var}={os.getenv(var)}")
    try:
        ip = socket.gethostbyname("api.groq.com")
        print(f"DNS           : api.groq.com -> {ip}")
    except Exception as exc:
        print(f"DNS FAILED    : {exc}")
        print("  -> no name resolution. VPN, DNS filter or offline.")
        return 2
    try:
        with socket.create_connection(("api.groq.com", 443), timeout=10):
            print("TCP/443       : connected")
    except Exception as exc:
        print(f"TCP/443 FAILED: {type(exc).__name__}: {exc}")
        print("  -> firewall or corporate proxy blocking outbound 443.")
        return 2

    step("3. does Groq accept the key  (GET /openai/v1/models)")
    try:
        r = httpx.get("https://api.groq.com/openai/v1/models",
                      headers={"Authorization": f"Bearer {key}"}, timeout=20)
    except Exception as exc:
        print(f"REQUEST FAILED: {type(exc).__name__}: {exc}")
        return 2
    print(f"HTTP {r.status_code}")
    if r.status_code == 401:
        print(r.text[:500])
        print("\n  VERDICT: the key is rejected. It is revoked, deleted, or from")
        print("  a different account. Make a new one at console.groq.com/keys")
        print("  and replace the value in .env.")
        return 2
    if r.status_code == 403:
        print(r.text[:500])
        print("\n  VERDICT: authenticated but not permitted — check the key's")
        print("  scope, or whether the account is suspended/over quota.")
        return 2
    if r.status_code == 429:
        print(r.text[:500])
        print("\n  VERDICT: rate limited. The key is valid. Wait a minute.")
        return 2
    if r.status_code != 200:
        print(r.text[:500])
        return 2

    ids = [m["id"] for m in r.json().get("data", [])]
    print(f"key is VALID — account can see {len(ids)} models")
    print(f"configured model present: {model in ids}")
    if model not in ids:
        print(f"  -> '{model}' is not available to this account.")
        print(f"  -> available chat models: "
              f"{[i for i in ids if 'llama' in i or 'gpt' in i][:8]}")
        print("  -> set VIBEBOT_MODEL in .env to one of those.")
        return 2

    step("4. the real call the bot makes  (POST /chat/completions)")
    req = nlp._request(provider, key, model, "I am exhausted and thinking of leaving.")
    try:
        r = httpx.post(**req, timeout=30)
    except Exception as exc:
        print(f"REQUEST FAILED: {type(exc).__name__}: {exc}")
        return 2
    print(f"HTTP {r.status_code}")
    if r.status_code != 200:
        print(r.text[:700])
        return 2
    print("response body (truncated):")
    print("  " + nlp._extract(provider, r.json())[:300])
    print("\n  VERDICT: everything works. The LLM path is live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

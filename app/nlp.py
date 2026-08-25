"""Reads what the employee actually wrote.

Two interchangeable backends behind one function:

  * an LLM when a provider key is present (Anthropic, Groq or Gemini) — better
    at sarcasm, negation and mixed sentiment.
  * a deterministic lexicon otherwise, so the whole pipeline runs offline, in
    CI, and in a demo with no network.

The crisis check is deliberately NOT delegated to the model. It runs first, on
every reply, in both backends: a missed disclosure is the one failure mode here
that actually harms someone, so it stays as an auditable rule.
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict

import httpx

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

# Haiku by default: these calls are short JSON classifications, not reasoning,
# and it is the cheapest tier. A full 30-conversation day costs a few cents.
# ---------------------------------------------------------------- providers
# Any one of these, or none. Picked by whichever key is present unless
# VIBEBOT_PROVIDER names one explicitly. Groq and Gemini both have no-card free
# tiers that comfortably cover a day of check-ins (~120 calls).
#
# Privacy note: Google's free tier permits training on submitted prompts. These
# replies contain disclosures about burnout, harassment and health, so free-tier
# Gemini is a demo-only option. Confirm any provider's retention policy before
# pointing this at real employees.
PROVIDERS = {
    "anthropic": dict(env="ANTHROPIC_API_KEY", model="claude-haiku-4-5-20251001"),
    # llama-3.1-8b-instant was shut down for free/developer tiers on
    # 2026-08-16; calling it now errors. gpt-oss-20b is Groq's named
    # replacement and is faster (~1000 tok/s).
    "groq":      dict(env="GROQ_API_KEY",      model="openai/gpt-oss-20b"),
    "gemini":    dict(env="GEMINI_API_KEY",    model="gemini-2.5-flash"),
}
RETRIES = 2          # free tiers throttle at ~30 requests/minute
BACKOFF = 2.0        # seconds, doubled per retry
# Providers may answer a 429 with a Retry-After of many minutes when a daily or
# token bucket is exhausted, not just the per-minute one. Honouring that
# verbatim freezes the whole run on a single reply with no output. Cap the wait
# and fall through to the lexicon instead — a slightly weaker analysis beats a
# process that looks hung for a quarter of an hour.
MAX_RETRY_WAIT = float(os.getenv("VIBEBOT_MAX_RETRY_WAIT", "45"))
# A Retry-After longer than the cap means the bucket that is empty is not the
# per-minute one — it is the daily or token allowance, which will not refill
# during this run. Retrying into that costs ~90s per reply and falls back to the
# lexicon anyway. After this many consecutive give-ups, stop calling the
# provider for the rest of the process and say so once.
CIRCUIT_TRIP = int(os.getenv("VIBEBOT_CIRCUIT_TRIP", "3"))
_circuit = {"open": False, "consecutive": 0}
STATS = collections.Counter()   # rate_limited / capped / failed, for the caller


def active_provider() -> tuple[str, str, str] | None:
    """(name, api_key, model) for the configured backend, or None for lexicon."""
    named = os.getenv("VIBEBOT_PROVIDER", "").strip().lower()
    order = [named] if named in PROVIDERS else list(PROVIDERS)
    for name in order:
        spec = PROVIDERS[name]
        key = os.getenv(spec["env"], "").strip()
        if key:
            return name, key, os.getenv("VIBEBOT_MODEL", "").strip() or spec["model"]
    return None


# ---------------------------------------------------------------- lexicons
# Phrases that end the scripted conversation and page a human immediately.
CRISIS_PATTERNS = [
    r"\bkill (myself|me)\b", r"\bend(ing)? (it|my life|things)\b", r"\bsuicid",
    r"\bself[- ]harm\b", r"\bhurt myself\b", r"\bno reason to (live|go on)\b",
    r"\bdon'?t want to (live|be here)\b",
]
# Phrases that require a human but not an emergency response.
SERIOUS_PATTERNS = {
    "harassment": [r"\bharass", r"\bbullie?[ds]?\b", r"\bshout(ed|ing)? at me\b", r"\bhumiliat"],
    "discrimination": [r"\bdiscriminat", r"\bracist\b", r"\bsexist\b", r"\bcaste\b"],
    "retaliation": [r"\bretaliat", r"\bthreaten(ed|ing)? (me|my job)\b"],
    "attrition_intent": [r"\bquit(ting)?\b", r"\bresign", r"\b(an)?other offers?\b",
                         r"\blooking (else)?where\b", r"\bjob hunt",
                         r"\bstarted (looking|interviewing)\b", r"\bexit\b",
                         r"\bnotice period\b", r"\bleav(e|ing) the (company|firm)\b"],
    "health": [r"\bburn(ed|t) out\b", r"\bpanic attack", r"\banxiety\b", r"\bdepress",
               r"\bcan'?t sleep\b"],
}

NEGATIVE = {
    "exhausted": 2, "exhausting": 2, "drained": 2, "burnt": 2, "burned": 2, "overwhelmed": 2,
    "unbearable": 2, "hopeless": 2, "miserable": 2, "furious": 2, "unfair": 2, "ignored": 2,
    "invisible": 2, "stuck": 1, "tired": 1, "frustrated": 2, "stressed": 2, "worried": 1,
    "disappointed": 2, "demotivated": 2, "undervalued": 2, "overworked": 2, "pointless": 2,
    "difficult": 1, "hard": 1, "struggling": 2, "pressure": 1, "constant": 1, "never": 1,
    "nobody": 1, "no one": 1, "again": 1, "still": 1, "bad": 1, "worse": 2, "worst": 2,
    "terrible": 2, "awful": 2, "relentless": 2, "thankless": 2, "toxic": 2, "dread": 2,
    "resent": 2, "isolated": 2, "unsupported": 2, "chaos": 1, "constantly": 1,
}
POSITIVE = {
    "great": 2, "good": 1, "fine": 1, "happy": 2, "excited": 2, "enjoying": 2, "enjoy": 2,
    "supportive": 2, "helpful": 2, "manageable": 2, "fair": 1, "clear": 1, "better": 1,
    "improving": 2, "proud": 2, "appreciated": 2, "valued": 2, "settled": 1, "okay": 1,
    "ok": 1, "no issues": 2, "thanks": 1, "love": 2, "brilliant": 2,
}
NEGATORS = {"not", "n't", "never", "hardly", "barely", "no"}

THEMES = {
    "workload": ["hours", "workload", "deadline", "capacity", "bandwidth", "late", "weekend", "overtime"],
    "recognition": ["recognition", "credit", "noticed", "appreciated", "thankless", "visible", "award"],
    "career": ["promotion", "rating", "appraisal", "growth", "progression", "band", "review"],
    "manager": ["manager", "lead", "boss", "supervisor", "reporting"],
    "team": ["team", "colleagues", "peers", "handover", "attrition", "backfill"],
    "compensation": ["salary", "pay", "hike", "bonus", "compensation"],
    "onboarding": ["onboarding", "buddy", "mentor", "training", "induction", "new"],
    "wellbeing": ["sleep", "health", "family", "personal", "energy", "rest"],
}


@dataclass
class ReplyAnalysis:
    sentiment: float                 # -1..1
    tone: str                        # negative | neutral | positive
    themes: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    crisis: bool = False
    summary: str = ""
    backend: str = "lexicon"

    def to_dict(self) -> dict:
        return asdict(self)


def _match_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def detect_crisis(text: str) -> bool:
    return _match_any(text.lower(), CRISIS_PATTERNS)


def detect_flags(text: str) -> list[str]:
    low = text.lower()
    return [name for name, pats in SERIOUS_PATTERNS.items() if _match_any(low, pats)]


CLAUSE_SPLIT = re.compile(r"[,.;!?]|\bbut\b|\bthough\b|\bhowever\b")


def _score_clause(clause: str) -> tuple[float, int]:
    """Score one clause. Negation is scoped to the clause and to two tokens."""
    score, hits = 0.0, 0
    padded = f" {clause.lower().strip()} "
    for table, sign in ((POSITIVE, 1), (NEGATIVE, -1)):
        for phrase, weight in table.items():
            if " " in phrase and f" {phrase} " in padded:
                score += sign * weight
                hits += 1
                padded = padded.replace(f" {phrase} ", " ")
    tokens = re.findall(r"[a-z']+", padded)
    for i, tok in enumerate(tokens):
        window = tokens[max(0, i - 2): i]
        negated = any(w in NEGATORS or w.endswith("n't") for w in window)
        if tok in NEGATIVE:
            score += NEGATIVE[tok] * (-1 if not negated else 0.5)
            hits += 1
        elif tok in POSITIVE:
            score += POSITIVE[tok] * (1 if not negated else -1)
            hits += 1
    return score, hits


def _lexicon_sentiment(text: str) -> float:
    total, hits = 0.0, 0
    for clause in CLAUSE_SPLIT.split(text.lower()):
        if not clause.strip():
            continue
        s, h = _score_clause(clause)
        total += s
        hits += h
    if not hits:
        return 0.0
    return max(-1.0, min(1.0, total / (hits * 2)))


def _lexicon_themes(text: str) -> list[str]:
    low = text.lower()
    return [t for t, words in THEMES.items() if any(w in low for w in words)]


def _tone(sentiment: float) -> str:
    if sentiment <= -0.2:
        return "negative"
    if sentiment >= 0.25:
        return "positive"
    return "neutral"


def _analyse_lexicon(text: str) -> ReplyAnalysis:
    s = _lexicon_sentiment(text)
    return ReplyAnalysis(
        sentiment=round(s, 2), tone=_tone(s), themes=_lexicon_themes(text),
        flags=detect_flags(text), crisis=detect_crisis(text),
        summary=text.strip()[:160], backend="lexicon",
    )


SYSTEM = """You analyse a single employee reply in a workplace wellbeing check-in.
Return ONLY minified JSON, no prose, no code fences, with keys:
  sentiment: number from -1 (distressed) to 1 (positive)
  themes: array from [workload, recognition, career, manager, team, compensation, onboarding, wellbeing]
  flags: array from [harassment, discrimination, retaliation, attrition_intent, health]
  summary: one neutral sentence, under 20 words, no quotes
Judge the employee's own state, not the topic. Sarcasm and understatement are common."""


def _request(provider: str, key: str, model: str, text: str) -> dict:
    """Build the provider-specific request. Same prompt and JSON contract for all."""
    if provider == "anthropic":
        return dict(
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": model, "max_tokens": 400, "system": SYSTEM,
                  "messages": [{"role": "user", "content": text}]},
        )
    if provider == "groq":
        return dict(
            url="https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
            json={"model": model, "temperature": 0, "max_tokens": 400,
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": text}]},
        )
    if provider == "gemini":
        return dict(
            url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": key, "content-type": "application/json"},
            json={"system_instruction": {"parts": [{"text": SYSTEM}]},
                  "contents": [{"parts": [{"text": text}]}],
                  "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}},
        )
    raise ValueError(f"unknown provider {provider}")


def _extract(provider: str, payload: dict) -> str:
    """Pull the model's text out of whichever response shape came back."""
    if provider == "anthropic":
        return "".join(b.get("text", "") for b in payload.get("content", [])
                       if b.get("type") == "text")
    if provider == "groq":
        return payload["choices"][0]["message"]["content"]
    if provider == "gemini":
        return "".join(p.get("text", "") for p in
                       payload["candidates"][0]["content"]["parts"])
    return ""


def _retry_after_seconds(raw: str | None, fallback: float) -> float:
    """Retry-After is seconds OR an HTTP-date (RFC 9110).

    float() on a date string raised, the blanket handler below swallowed it, the
    429 went uncounted, and the wait cap never engaged.
    """
    if not raw:
        return fallback
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return fallback


def circuit_is_open() -> bool:
    """True once the provider has been given up on for this process."""
    return _circuit["open"]


def reset_circuit() -> None:
    """Close the breaker again — used by the tests."""
    _circuit["open"] = False
    _circuit["consecutive"] = 0


def _trip(provider: str, why: str) -> None:
    _circuit["consecutive"] += 1
    if _circuit["consecutive"] >= CIRCUIT_TRIP and not _circuit["open"]:
        _circuit["open"] = True
        STATS["circuit_opened"] += 1
        print(f"    …{provider} gave up after {_circuit['consecutive']} consecutive "
              f"failures ({why}). Using the lexicon for the rest of this run — "
              f"the daily quota is the likely cause, and it will not refill now.",
              file=sys.stderr, flush=True)


def _analyse_llm(text: str, provider: str, key: str, model: str) -> ReplyAnalysis | None:
    if _circuit["open"]:
        STATS["skipped_circuit_open"] += 1
        return None
    req = _request(provider, key, model, text)
    delay = BACKOFF
    for attempt in range(RETRIES + 1):
        try:
            r = httpx.post(**req, timeout=20)
            if r.status_code == 429 and attempt < RETRIES:
                # free tiers cap requests per minute; respect Retry-After if given
                asked = _retry_after_seconds(r.headers.get("retry-after"), delay)
                wait = min(asked, MAX_RETRY_WAIT)
                STATS["rate_limited"] += 1
                if asked > MAX_RETRY_WAIT:
                    STATS["capped"] += 1
                print(f"    …{provider} rate limited, asked for {asked:.0f}s, "
                      f"waiting {wait:.0f}s", file=sys.stderr, flush=True)
                if asked > MAX_RETRY_WAIT:
                    # Not a per-minute limit. Count it toward giving up.
                    _trip(provider, f"rate limited for {asked:.0f}s")
                    if _circuit["open"]:
                        return None
                time.sleep(wait)
                delay *= 2
                continue
            r.raise_for_status()
            _circuit["consecutive"] = 0     # a success closes the streak
            raw = _extract(provider, r.json())
            raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
            data = json.loads(raw)
            s = max(-1.0, min(1.0, float(data.get("sentiment", 0))))
            return ReplyAnalysis(
                sentiment=round(s, 2), tone=_tone(s),
                themes=[t for t in data.get("themes", []) if t in THEMES],
                # union with the rule-based flags: the model may miss, the rules may not
                flags=sorted(set(data.get("flags", [])) | set(detect_flags(text))),
                crisis=detect_crisis(text),
                summary=str(data.get("summary", ""))[:200], backend=provider,
            )
        except Exception as exc:
            if attempt >= RETRIES:
                STATS["failed"] += 1
                detail = type(exc).__name__
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status:
                    detail = f"HTTP {status}"
                STATS[f"failed:{detail}"] += 1
                print(f"    …{provider} call failed ({detail}), "
                      f"using the lexicon for this reply", file=sys.stderr, flush=True)
                _trip(provider, detail)
                return None      # fall through to the lexicon
            time.sleep(delay)
            delay *= 2
    return None


def analyse(text: str) -> ReplyAnalysis:
    """Analyse one employee reply. Never raises."""
    if not text or not text.strip():
        return ReplyAnalysis(0.0, "neutral", summary="(no reply)")
    active = active_provider()
    if active:
        result = _analyse_llm(text, *active)
        if result:
            return result
    return _analyse_lexicon(text)

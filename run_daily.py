"""Run one full day end to end, headless.

    python run_daily.py --capacity 30 --simulate

--simulate answers as the employees so the pipeline can be exercised without a
UI: replies are drawn from a per-archetype script, which is also how the report
gets populated for a demo. Without it, sessions are opened and left waiting for
real replies through the API.
"""
from __future__ import annotations

import argparse
import collections
import random
import time
import uuid
from datetime import date

from app import nlp
from app import conversation
from app.conversation import reply, start_session
from app.features import Datasets, build_features
from app.report import build_report, save, to_markdown
from app.selection import select_cohort

# Replies a person in each situation might plausibly give. Used only by --simulate.
SCRIPTS = {
    "workload": [
        "it's been relentless — two people left the team in March and nothing was backfilled",
        "not a busy week, it's been like this since April",
        "I'm exhausted honestly, and I don't think anyone above me has noticed",
    ],
    "leave_deficit": [
        "I keep postponing it, every time I plan leave something slips on the project",
        "if I take a week it just piles up and I pay for it after",
        "I'd take it if someone covered the client calls",
    ],
    "recognition": [
        "we shipped the migration in January and nobody outside the team knows it happened",
        "it's less the award, more that the work seems invisible",
        "I'd rather it was raised quietly with my lead than made a thing of",
    ],
    "career": [
        "I found out from the letter, there was no conversation before it",
        "nobody has told me what would actually change the outcome next cycle",
        "I've started looking at other offers, to be honest",
    ],
    "onboarding": [
        "I still don't have a buddy and I've been asking since I joined",
        "the hardest part is not knowing who to ask basic questions",
        "the training modules are half unassigned",
    ],
    "disengagement": [
        "the survey feels like noise honestly, nothing changes after it",
        "the week is fine, I just stopped filling it in",
    ],
    "positive": [
        "the team is genuinely good — my lead protects our focus time",
        "mostly the work itself, we get to own things end to end",
        "the weekly demo is worth copying, it keeps everyone honest",
    ],
    "generic": ["nothing specific, just a heavy stretch", "it's fine, manageable"],
}


def simulate(session, rng: random.Random):
    """Answer as the employee until the bot closes the session."""
    session = reply(session, rng.choice(["yes, that's fine", "sure, go ahead", "ok"]))
    if session.state == "closed":
        return session
    driver = session.drivers[0]["code"] if session.drivers else "generic"
    key = "positive" if session.band == "positive" else driver
    lines = list(SCRIPTS.get(key, SCRIPTS["generic"]))
    while session.state != "closed" and lines:
        session = reply(session, lines.pop(0))
    while session.state != "closed":
        session = reply(session, rng.choice(["yes please", "no, I'm alright for now"]))
    return session


def _install_pacer(rpm: int) -> None:
    """Space provider calls so we stay under the free-tier limit.

    Cheaper than hitting 429s: a throttled call burns the full retry-and-backoff
    budget and may still fall through to the lexicon, silently degrading the
    analysis. Waiting the gap costs the same wall-clock and keeps every reply on
    the LLM path.
    """
    gap = 60.0 / max(rpm, 1)
    last = [0.0]
    real_post = nlp.httpx.post

    def paced(**kw):
        wait = gap - (time.monotonic() - last[0])
        if wait > 0:
            time.sleep(wait)
        last[0] = time.monotonic()
        return real_post(**kw)

    nlp.httpx.post = paced


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=30)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--cooldown-days", type=int, default=14,
                    help="skip anyone contacted within this many days "
                         "(sustained distress always overrides it)")
    ap.add_argument("--rpm", type=int, default=0,
                    help="cap provider requests per minute (Groq free tier is 30). "
                         "0 disables pacing.")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    conversation.seed(args.seed)   # pin question choice too, not just replies

    if args.rpm:
        _install_pacer(args.rpm)

    print("loading datasets…")
    ds = Datasets.load()
    features = build_features(ds)
    print(f"  {len(features)} employees, {len(features.columns)} features")

    # Same cooldown the API applies, read from the same session store — so a
    # headless run and the console agree on who was spoken to recently.
    from app.main import _last_contacted
    cohort = select_cohort(features, capacity=args.capacity,
                           cooldown_days=args.cooldown_days,
                           last_contacted=_last_contacted())
    print(f"selected {len(cohort)} for contact today")

    backend = nlp.active_provider()
    if backend:
        print(f"reply analysis: {backend[0]} / {backend[2]}"
              f"{f' — paced to {args.rpm} req/min' if args.rpm else ''}")
    else:
        print("reply analysis: lexicon (no API key configured)")

    sessions = []
    t0 = time.time()
    for i, c in enumerate(cohort, 1):
        s = start_session(c, session_id=str(uuid.uuid4())[:8])
        if args.simulate:
            s = simulate(s, rng)
        sessions.append(s)
        if args.simulate:
            # A live LLM backend paces at the provider's rate limit, so this
            # loop can take minutes. Without a heartbeat it looks hung.
            done = "escalated" if s.escalation else (s.outcome or {}).get("status", "ok")
            print(f"  [{i:>3}/{len(cohort)}] {c.name[:22]:22} {done:>10}"
                  f"   {time.time() - t0:5.1f}s elapsed", flush=True)
    if args.simulate:
        used = collections.Counter(
            (t.analysis or {}).get("backend", "?")
            for s in sessions for t in s.turns
            if t.role == "employee" and t.analysis)
        print(f"ran {len(sessions)} conversations, "
              f"{sum(1 for s in sessions if s.escalation)} escalated "
              f"in {time.time() - t0:.0f}s")
        print(f"analysis backends actually used: {dict(used)}")
        if nlp.STATS.get("rate_limited"):
            print(f"rate limit hits: {nlp.STATS['rate_limited']}"
                  + (f" ({nlp.STATS['capped']} asked for longer than "
                     f"{nlp.MAX_RETRY_WAIT:.0f}s and were capped)"
                     if nlp.STATS.get("capped") else ""))
        if nlp.circuit_is_open():
            print(f"provider given up on mid-run: {nlp.STATS.get('skipped_circuit_open', 0)} "
                  f"later replies went straight to the lexicon without calling it.")
        if backend and used.get("lexicon"):
            print(f"  WARNING: {used['lexicon']} replies fell back to the lexicon "
                  f"— rate limited or the API errored. Lower --rpm (try 15) or "
                  f"--capacity, or wait for the quota window to reset.")

    rep = build_report(cohort, sessions, features, as_of=date.today())
    j, m = save(rep, args.outdir)
    print(f"\nreport written:\n  {j}\n  {m}\n")
    print(to_markdown(rep)[:1600])


if __name__ == "__main__":
    main()

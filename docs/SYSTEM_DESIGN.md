# System design

## What the brief asks for

Deloitte collects a Vibemeter reading every alternate day across 35,000+
employees. Today two manual steps follow: analysts correlate that mood data with
leave, activity, rewards, performance and onboarding records to find the likely
cause (Step 1), then people run individual meetings and focus groups to confirm
it (Step 2). Both are slow. The bot automates both, and reports back daily.

## Shape of the system

```
  six CSV sources
        │
        ▼
  features.py ──────── one row per employee, fixed lookbacks (28d vibe, 14d activity)
        │
        ▼
  selection.py ─────── driver rules → risk score → cohort rules
        │                    │
        │                    └── every candidate carries: trigger + drivers + evidence
        ▼
  conversation.py ──── state machine, questions drawn from question_bank.yaml
        │                    │
        │                    ├── nlp.py       reply → sentiment, themes, flags, crisis
        │                    └── escalation   thresholds → P1 / P2 with a reason
        ▼
  report.py ────────── org rollup + per-employee findings + actions (JSON and Markdown)
```

`main.py` exposes this over HTTP; `run_daily.py` runs the same pipeline headless
for a scheduled job. The two share every module, so the console and the cron job
cannot drift apart.

## Step 1 — who to talk to, and why

Two stages, kept apart deliberately.

**Driver rules.** Six deterministic rules over the joined feature table:
workload, leave deficit, recognition, career, onboarding, disengagement. Each
returns a severity in 0..1 *and the sentence of evidence that produced it*
("averaging 11.3 hours a day over the last two weeks, 2.5h above the Consulting
median"). Rules rather than a learned model, for three reasons: HR has to be
able to read the reason and disagree with it; there are no labels for "why was
this person unhappy"; and the same string is what the bot opens the conversation
with, so the explanation and the question can never disagree.

Workload compares against the employee's own team median, not a global
threshold — 9.5 hours means something different in Audit during close than it
does in Operations.

**Risk score.** `0.55 × vibe severity + 0.45 × weighted drivers`. The vibe keeps
the majority share because it is the signal we were asked to explain; the
drivers are candidate explanations. The score only *orders* the queue — it never
decides membership, so every contacted employee traces back to a named rule.

**Cohort rules**, in priority order:

| Rule | Condition | Band |
| --- | --- | --- |
| `sustained_distress` | 3+ consecutive Frustrated/Sad readings | critical |
| `sharp_decline` | drop of 2+ zones inside a week, mean ≤ 3.2 | elevated |
| `chronic_low` | 28-day mean ≤ 2.4 | elevated |
| `critical_driver` | any driver ≥ 0.7 severity with mean ≤ 3.4 | elevated |
| `went_quiet` | stopped answering the Vibemeter | watch |
| `thriving_sample` | 28-day mean ≥ 4.3 | positive |

Three constraints then shape the day:

- **Cooldown.** Nobody is contacted twice inside 14 days — except
  `sustained_distress`, which overrides it. A wellbeing bot that pesters people
  stops getting honest answers.
- **Positive share.** ~12% of seats are reserved for people doing well. The
  brief explicitly asks which happy employees to talk to; you cannot learn what
  is working by only interviewing people who are struggling.
- **Driver round-robin.** Straight worst-first triage returned 27 of 30 burnout
  cases on our sample, because the score ranks burnout above everything. Seats
  are filled worst-band-first but rotated across primary drivers, so the day's
  report covers the range of issues instead of one.

## Step 2 — the conversation

A bounded state machine, not a free-running agent:

`greeting → consent → probing → support offer → closed`, with `crisis` as a
priority path out of any state.

- Questions come from `data/question_bank.yaml`, which is reviewable by HR
  without touching code. The LLM reads replies; it never invents questions.
- The opener quotes the driver evidence, so the first question is specific
  ("your working pattern stood out — 11.3 hours a day…") rather than "how are
  you feeling?". Specific questions get specific answers.
- Follow-ups branch on the sentiment of the previous reply, so someone who says
  "busy quarter, manageable" is not interrogated like someone who says
  "exhausted and nobody noticed".
- A hard turn budget (7 questions) caps every session.
- A P1 signal — a crisis disclosure, or harassment, discrimination or
  retaliation — escalates on the turn it appears, at any stage, including the
  answer to "is now a good time?". Safety outranks the script, and the safety
  rules read every reply rather than the consent-excluded subset the sentiment
  average uses.
- Consent is asked first and a decline closes the session immediately. Refusal
  is matched on word boundaries, not substrings — "now is fine", "no problem"
  and "I know what this is about" are consent, not refusal. Consent is
  deliberately **not** gated on sentiment: a distressed "yeah, not great, but
  ok" is the strongest reason to keep talking, and treating it as a decline
  would drop exactly the people the bot exists to reach.

### Reading replies

`nlp.py` has four interchangeable backends behind one `analyse()` call: Groq,
Anthropic, Gemini, and a deterministic lexicon. Whichever key is present wins
(`VIBEBOT_PROVIDER` forces one); with no key at all the lexicon runs, so the
demo, CI and an offline laptop all work. All four return the same structure:
sentiment (−1..1), themes, flags, crisis.

A 30-person day is ~190 calls, inside every free tier's daily allowance. The
binding constraint is requests per minute (~30 on Groq, ~15 on Gemini Flash),
so 429s are retried with backoff, and any persistent failure or malformed
response falls through to the lexicon rather than losing the reply.

`Retry-After` is capped (`VIBEBOT_MAX_RETRY_WAIT`, default 45s). A provider
that has exhausted a daily or token bucket may ask for many minutes; obeying
that verbatim freezes the run on a single reply with no output, which is worse
than a degraded analysis. `run_daily.py --rpm` paces requests instead, which
costs the same wall-clock but keeps every reply on the LLM path.

The fallback is deliberate but it is also the sharpest edge in this design:
a revoked key, an expired model id and a rate limit all fail identically and
invisibly. `check_provider.py` and the per-run backend counter exist so that
degradation is never silent.

Provider choice is a privacy decision as much as a quality one: Google's free
tier permits training on submitted prompts, and these replies contain health
and harassment disclosures. Confirm retention terms before real data.

The crisis check is **not** delegated to the model. It runs first, on every
reply, in both backends, as an auditable regex rule. A missed disclosure is the
one failure here that actually harms someone, so it does not depend on a network
call succeeding or a model behaving.

### Consent

Two consent decisions are recorded, and both are opt-in rather than inferred.

Consent to *talk* is asked first; a refusal closes the session immediately.
Refusal is matched on word boundaries, so "now is fine" and "no problem" are
consent. It is deliberately not gated on sentiment — a distressed "yeah, not
great, but ok" is the strongest reason to keep talking.

Consent to be *contacted by a human* is recorded as yes, no, or unknown. An
earlier rule inferred it from sentiment (`>= 0`) plus a substring match, which
recorded "no", "I'd rather not" and "please don't" as agreement — the report
then told HR the employee had consented. Only an explicit affirmative records
yes; an explicit refusal records no; anything else is reported as "unclear —
ask", because manufacturing either answer misrepresents a decision the employee
made about their own disclosure.

### Escalation

| Priority | Trigger |
| --- | --- |
| P1 | personal-safety disclosure; harassment, discrimination or retaliation |
| P2 | mean sentiment ≤ −0.45 across 2+ consecutive negative replies |
| P2 | analysis risk ≥ 70 confirmed by negative conversation sentiment |
| P2 | stated intent to leave |

On a P1 crisis disclosure the bot stops the script, says plainly that it is not
the right kind of help, surfaces the EAP and the Tele-MANAS helpline (14416),
and hands to a human. It does not continue asking about workload.

## Privacy

The bot promises the employee "a theme, not a transcript", so the report stores
summaries, themes and flags — not verbatim replies. Escalations carry the reason
and the flags, and record whether the employee consented to being contacted.
Sessions persist to SQLite so a dropped connection does not lose a conversation.

## Known limits

- **The lexicon backend is shallow.** It handles negation and clause boundaries
  but inverts sarcasm ("great, another weekend gone" reads positive) and misses
  understatement. Set a provider key for anything beyond a demo — the brief
  weights emotion detection at 40%.
- **Synthetic data.** `data/generate.py` plants known archetypes so selection can
  be tested against ground truth. Thresholds tuned here will need re-tuning
  against real distributions.
- **No manager loop.** The report recommends actions but nothing closes the
  loop on whether the action happened, which is what would actually prove the
  system works.
- **English only**, and no accessibility pass on the console beyond keyboard
  focus and reduced-motion defaults.

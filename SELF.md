# Vibemeter People Experience Bot — project brief

Everything about this project in one place: what it does, how it is built, why
each decision was made, what went wrong and how it was found. Written so you can
answer follow-up questions without re-reading the source.

---

## 1. The problem

Deloitte runs a **Vibemeter** — every employee logs a mood score (1 Frustrated →
5 Excited) every other day, across 35,000+ people. Two manual steps follow:

1. **Step 1 — analysis.** Analysts correlate mood against leave, activity,
   rewards, performance and onboarding records to work out *why* someone's mood
   dropped.
2. **Step 2 — conversation.** People run one-to-one meetings and focus groups to
   confirm the cause.

Both are slow and neither scales. The brief was to automate both and report back
to the People Experience team daily.

**What this system does:** reads six data sources, decides who to talk to each
day *and why*, holds a short scripted conversation to find the cause, escalates
what needs a human, and writes a daily report.

---

## 2. Tech stack, and why

| Layer | Choice | Why this and not the alternative |
| --- | --- | --- |
| Language | **Python 3.12** | The data work (joins, rolling windows, group medians) is pandas' home turf. A JS/TS stack would have meant hand-rolling all of it. |
| Data | **pandas + numpy** | Six CSVs joined into one row per employee with fixed lookback windows. Vectorised group medians matter: workload is judged against the employee's own team, which is a `groupby().transform("median")`, not a loop. |
| API | **FastAPI** | Type-hint-driven validation via pydantic, so `capacity`, `positive_share` and `cooldown_days` are bounds-checked at the edge with no hand-written validation. Free OpenAPI docs at `/docs`. Async-capable if this ever needs to scale. |
| Server | **uvicorn** | The standard ASGI server for FastAPI. |
| Validation | **pydantic v2** | Request models are the validation layer. |
| Persistence | **SQLite** (stdlib) | Sessions must survive a restart, but the deliverable is a daily report, not a multi-tenant product. SQLite is zero-config, file-backed, and ships with Python. Postgres would have added an ops dependency for no benefit at this scale. |
| Config | **YAML** (PyYAML) | The question bank is content, edited by HR, not code. Keeping it in `data/question_bank.yaml` means changing what the bot *says* requires no Python. |
| HTTP client | **httpx** | Used for the LLM provider calls. Same API as requests, but with timeouts that actually work and HTTP/2 support if needed. |
| Frontend | **One HTML file, no framework** | The console is an internal review tool: a queue, a chat pane, a report view. React would have meant a build step, a bundler and a `node_modules` for maybe 250 lines of DOM work. Everything is in `static/index.html` — open it and you can read the whole UI. |
| Reply analysis | **LLM with a deterministic lexicon fallback** | See §6. This is the most interesting design decision in the project. |
| Tests | **pytest** | 97 tests, ~13 seconds, hermetic. |
| Container | **Docker + compose** | Reproducible run; the image generates its sample data at build so it runs standalone. |

**Deliberate non-choices worth being able to defend:**

- **No ML model for driver attribution.** There are no labels for "why was this
  person unhappy" — nobody has ever recorded the ground truth. A supervised
  model would need labels that don't exist. Rules are also *readable*: HR can
  disagree with "averaging 11.3 hours a day, 2.5h above the Consulting median"
  in a way they cannot disagree with a gradient-boosted score.
- **No vector database / RAG.** Nothing here needs semantic retrieval over a
  corpus. The question bank is small and keyed by driver.
- **No message queue.** A daily batch of ~30 conversations is not a streaming
  problem.

---

## 3. Folder structure

```
vibebot/
├── app/
│   ├── __init__.py        loads .env at import so entry points see the API key
│   ├── features.py        six CSV sources → one feature row per employee
│   ├── selection.py       driver rules, risk score, cohort rules  ← "who and why"
│   ├── conversation.py    the conversation state machine + escalation
│   ├── nlp.py             reply analysis: LLM backends + lexicon fallback
│   ├── report.py          the daily report, JSON and Markdown
│   └── main.py            FastAPI routes, SQLite session store, serves the console
├── data/
│   ├── generate.py        synthetic dataset generator with planted archetypes
│   ├── question_bank.yaml what the bot says, per driver — content, not code
│   └── raw/               generated CSVs (gitignored)
├── static/index.html      the HR console (single file, no build step)
├── tests/
│   ├── conftest.py        forces the lexicon backend so the suite is hermetic
│   ├── test_pipeline.py   features, selection, conversation
│   ├── test_providers.py  provider wire formats and fallback behaviour
│   ├── test_regressions.py one test per fixed defect, named for the defect
│   └── test_api.py        the HTTP surface end to end
├── docs/SYSTEM_DESIGN.md  selection logic, escalation, privacy, known limits
├── reports/               generated daily reports (gitignored)
├── run_daily.py           headless end-to-end day → reports/
├── check_provider.py      is the configured API key actually working?
├── diagnose_groq.py       why isn't it — raw HTTP status, key hygiene, models
├── Dockerfile             /  docker-compose.yml
└── requirements.txt
```

**The layering, in one sentence:** `features` knows about CSVs and nothing else;
`selection` knows about features and produces `Candidate` objects; `conversation`
knows about a Candidate and produces a `Session`; `report` turns cohorts and
sessions into a document; `main` is the only module that knows about HTTP.

---

## 4. Step 1 — who to talk to, and why

Two stages kept deliberately apart, because they answer different questions.

### Driver rules — *what is probably wrong*

Six deterministic rules over the joined feature table. Each returns a severity
in 0..1 **and the sentence of evidence that produced it**.

| Driver | Signal |
| --- | --- |
| `workload` | Hours vs the employee's own team median, after-hours message ratio, weekend days worked |
| `leave_deficit` | Days since last leave, total leave taken in the period |
| `recognition` | Months since last award, award count |
| `career` | Rating drop between cycles, absolute rating, years in role without promotion |
| `onboarding` | Incomplete induction, no assigned buddy, within first year |
| `disengagement` | Vibemeter non-response — silence as a signal |

**Why the evidence string is generated by the rule itself:** that exact string is
what the bot opens the conversation with, and what appears in the report. One
source, so the explanation and the question can never disagree.

### Risk score — *how urgent*

```
score = 0.55 × vibe_severity + 0.45 × Σ(driver_weight × severity)
```

`vibe_severity` blends level (45%), low streak (25%), trend (20%) and a
sharp 7-day drop (10%).

Driver weights: workload 0.22, career 0.20, leave_deficit / recognition /
onboarding 0.16 each, disengagement 0.10.

**Key point to be able to say out loud:** the score only *orders* the queue. It
never decides membership. Every contacted employee traces back to a named rule,
not to a number — which is what makes the selection defensible to an employee who
asks "why me?"

### Cohort rules — *who actually gets contacted*

In priority order, first match wins and is recorded as the `trigger`:

| Rule | Condition | Band |
| --- | --- | --- |
| `sustained_distress` | 3+ consecutive Frustrated/Sad readings | critical |
| `sharp_decline` | drop of 2+ zones and mean ≤ 3.2 | elevated |
| `chronic_low` | mean ≤ 2.4 | elevated |
| `critical_driver` | any driver ≥ 0.7 severity and mean ≤ 3.4 | elevated |
| `went_quiet` | Vibemeter non-response | watch |
| `thriving_sample` | mean ≥ 4.3 | positive |

Then three constraints:

1. **Capacity.** HR can only hold so many conversations a day.
2. **Cooldown** (default 14 days) — don't ask the same person the same thing
   every week. **Sustained distress overrides it**: that is exactly who must not
   be skipped.
3. **Diversification.** Straight worst-first triage produced a cohort of 27/30
   burnout cases on the sample data — a one-note report that hid the recognition
   and career problems entirely. `_diversify` keeps banding absolute (critical
   seats fill first) but round-robins across primary drivers to decide *which*
   critical employees get those seats.

A small `positive_share` of the cohort goes to thriving employees, so the report
can say what is *working*, not only what is broken.

---

## 5. Step 2 — the conversation

A state machine, not a free-running chatbot:

```
greeting → consent → probing → support_offer → closed
                        ↑
                     crisis (pre-empts everything)
```

- **Consent first.** A refusal closes the session immediately. Refusal is matched
  on word boundaries, so "now is fine" and "no problem" are consent, not refusal.
  Consent is deliberately **not** gated on sentiment — a distressed "yeah, not
  great, but ok" is the strongest reason to keep talking.
- **Opens with the evidence.** Not "how are you doing?" but the specific thing
  the data noticed.
- **Follow-ups branch on the sentiment of the previous reply**, so "busy quarter,
  manageable" is not interrogated like "exhausted and nobody noticed".
- **Hard turn budget: 7 questions.** Every session is bounded.
- **Consent to HR contact is a separate, explicit opt-in** — recorded as yes, no,
  or unknown, and the report prints "unclear — ask" rather than guessing.

### Escalation

| Trigger | Priority |
| --- | --- |
| Crisis disclosure (self-harm language) | P1 |
| Harassment / discrimination / retaliation flag | P1 |
| Mean sentiment ≤ −0.45 with 2+ consecutive negative replies | P2 |
| Risk ≥ 70 confirmed by conversation sentiment ≤ −0.3 | P2 |
| Stated intent to leave | P2 |

**A P1 escalates on the turn it appears, at any stage** — including the answer to
"is now a good time?".

---

## 6. Reply analysis — the most defensible design decision

`nlp.py` has four interchangeable backends behind one `analyse()` call: Groq,
Anthropic, Gemini, and a deterministic lexicon. Whichever API key is present
wins; with no key the lexicon runs, so the demo, CI and an offline laptop all
work.

All four return the same structure: sentiment (−1..1), themes, flags, crisis.

**The crisis check is deliberately NOT delegated to the model.** It runs first,
on every reply, in both backends, as an auditable regex rule. Reasoning: a missed
self-harm disclosure is the one failure here that actually harms someone, so it
must not depend on a network call succeeding or a model behaving. This is the
single best thing to talk about in an interview — it shows you thought about
which failures are recoverable and which are not.

The lexicon handles negation and clause boundaries ("it's fine, but I'm
exhausted" splits on `but`), so it is not a naive bag of words.

**Rate limiting and degradation.** A 30-person day is ~190 provider calls. Free
tiers cap at ~30 requests/minute, so `--rpm` paces requests. 429s are retried
with backoff; `Retry-After` is honoured but **capped at 45s**, because a provider
that has exhausted a *daily* bucket can ask for many minutes and freeze the whole
run on one reply. Every run ends by reporting which backend actually analysed the
replies, so silent degradation is visible.

**Circuit breaker.** A `Retry-After` longer than the cap means the *daily* or
token bucket is empty, not the per-minute one — and that will not refill during
the run. Retrying into it costs ~90s per reply and falls back to the lexicon
anyway. After 3 consecutive give-ups the provider is dropped for the rest of the
process. This turned a 12-person run that was crawling at ~4 minutes per
conversation into one that finishes in seconds. *Worth discussing: the right
response to an exhausted quota is not a longer backoff, it is to stop asking.*

---

## 7. Privacy and ethics

Worth raising unprompted — it signals judgement.

- **Consent is asked before the conversation and again before any human is
  involved.** Neither is inferred.
- **Nothing reaches a manager without the employee's say-so** — the greeting says
  so explicitly.
- The report gives HR the **driver and the evidence**, not a verbatim transcript
  dump of everything the employee said.
- **Free-tier LLM providers may train on submitted prompts.** These replies
  contain health and harassment disclosures. That is fine for a coding challenge
  on synthetic data and **not** fine for real employees — check any provider's
  retention policy first. Google's free tier explicitly permits it.
- The system can **misattribute a cause**, which is why the bot asks rather than
  concludes, and why HR sees the evidence and can disagree.

---

## 8. Testing

**99 tests, ~13 seconds, 93% line coverage.**

| File | Covers |
| --- | --- |
| `test_pipeline.py` | Feature engineering, driver rules, cohort selection, conversation flow |
| `test_providers.py` | Each provider's wire format, retry, fallback, and that crisis detection is provider-independent |
| `test_regressions.py` | One test per fixed defect, named for the defect |
| `test_api.py` | The HTTP surface end to end via `TestClient`, including a restart |

`conftest.py` neutralises provider keys for the whole suite. Without it the tests
make live API calls — non-deterministic, billable, and network-dependent. **This
is worth mentioning as a lesson learned:** adding `.env` auto-loading turned the
unit tests into integration tests overnight and took the suite from 1.3s to 315s
before it was caught with `pytest --durations`.

---

## 9. Bugs found and fixed — the best interview material

Real defects found by systematic review, each now guarded by a named test.

**1. A refusal of HR contact was recorded as consent.**
```python
agreed = analysis.sentiment >= 0 or any(w in text.lower() for w in ("yes","sure","ok","please"))
```
Both halves wrong. `"no"` scores 0.0 sentiment, and `>= 0` passes. `"please
don't"` matches `"please"`. Every one of *no / nope / I'd rather not / definitely
not / absolutely not / please don't* recorded as consent — and the report told HR
the employee had agreed. **Fix:** explicit opt-in, yes/no/unknown, word-anchored.
*Lesson: a permissive default is a bug when the field is a consent decision.*

**2. A crisis on the first reply lost its priority.**
The sentiment average deliberately excludes the consent-stage turn — but the
first reply *is* consent stage, so the escalation check saw an empty list and
produced a blank priority and blank reason. **Fix:** split the list in two —
safety rules read every reply, the sentiment average reads the substantive ones.
*Lesson: one data structure was doing two incompatible jobs.*

**3. `positive_share=0` still contacted a thriving employee.** A `max(1, ...)`
floor forced a positive seat: `capacity=1` contacted someone scoring 5.9 and
skipped all 134 people in sustained distress — violating the module's own first
promise.

**4. The cooldown was completely inert.** `select_cohort` implemented it
correctly but read a `last_contacted` map no caller ever supplied. It was
validated, documented, and shipped to HR as a live guarantee. The tests hid it by
passing the argument explicitly — proving the *function* worked while the
*system* never invoked it. *Lesson: unit tests that supply what production
forgets to supply will pass forever.*

**5. Stored XSS in the console.** Employee message text went into `innerHTML`
unescaped, was persisted to SQLite, and replayed into an HR reviewer's browser.
Stored, not reflected — it fires later, in a different user's session.

**6. A dead model id looked exactly like a bad API key.** Groq decommissioned
`llama-3.1-8b-instant` on 2026-08-16. Because `analyse()` catches every exception
and falls back, a revoked key, an expired model and a rate limit all failed
identically and invisibly. **Fix:** `check_provider.py` and `diagnose_groq.py`,
plus a per-run backend counter. *Lesson: a silent fallback is a good product
decision and a terrible debugging experience — instrument it.*

**7. Sessions did not survive a restart.** The docstring claimed they did; the
write path existed but nothing ever read it back.

**8. One bad vibe value destroyed the whole report.** A direct `VIBE_LABELS[...]`
subscript raised `KeyError` — at the *final* step, after a 20-minute paid run.

Also: `Retry-After` HTTP-dates crashed the parser; `--seed` didn't reproduce
because question selection used the global `random`; a session crossing midnight
moved days and vanished from its own report; message length was unbounded.

---

## 10. Known limits — say these before you're asked

- **The lexicon is shallow.** It handles negation and clauses, but sarcasm and
  understatement defeat it. The LLM backend exists because of this.
- **`disengagement` never fires on the sample data.** The non-response threshold
  is unreachable given how the generator produces responses, so one driver and
  the `went_quiet` rule are effectively untested in practice.
- **Single-process state.** `_state` is module-level, so running uvicorn with
  `--workers > 1` breaks it: one worker holds the cohort, another serves the next
  request. Fine for the demo; the fix is to move cohort state into SQLite too.
- **No authentication.** `/api/report` returns every employee's risk score and
  escalation reason to any caller. Acceptable for a challenge, not for anything
  real.
- **Synthetic data.** The generator plants known archetypes with a ground-truth
  file, so selection *can* be validated — but no test yet asserts a recall floor
  against it. That is the single highest-value test still missing.
- **Docker path is unverified.** The image builds by inspection but has not been
  run end to end.
- **`build_features` is 106 lines** of six inline per-source blocks, and the
  evidence prose is generated inside the scoring rules — so wording cannot change
  without editing the risk model. Both are known refactors, deliberately deferred
  as maintainability rather than correctness work.

---

## 11. Likely interview questions

**"Why rules instead of ML for driver attribution?"** No labels exist for "why
was this person unhappy". Rules are auditable, HR can disagree with them, and the
same evidence string opens the conversation — so the explanation and the question
can never diverge.

**"How do you know the bot talks to the right people?"** The score only orders;
named rules decide membership. Every contacted employee traces to a rule. The
generator plants known archetypes with ground truth so recall *can* be measured
— I'd add that test next.

**"What happens when the LLM is down?"** Deterministic lexicon fallback, so no
reply is ever lost. Crisis detection never depended on the model in the first
place. The run reports which backend was actually used, so degradation is
visible rather than silent.

**"What's the riskiest part of this system?"** Misattributing a cause, and a
missed crisis disclosure. The first is mitigated by asking rather than
concluding; the second by keeping crisis detection as a deterministic rule that
runs before and independently of any network call.

**"Would you deploy this at 35,000 employees?"** Not as-is. Selection is
vectorised and fine, but `_state` is single-process, there's no auth, and the
provider rate limit binds at ~190 calls per 30-person day. First three changes:
move cohort state to SQLite, add SSO, and batch or self-host the analysis model.

**"What did you get wrong?"** The consent-inversion bug — a permissive default on
a field recording an employee's decision about their own disclosure. And letting
`.env` auto-loading turn the unit tests into live API calls without noticing for
several runs.

---

## 12. Quick reference

```powershell
pip install -r requirements.txt
python data\generate.py --employees 500 --days 120
uvicorn app.main:app --reload            # console at http://localhost:8000
python run_daily.py --capacity 12 --simulate --rpm 15
python -m pytest tests -q                # 99 tests
python check_provider.py                 # is the API key live?
```

Numbers worth remembering: **6** data sources · **6** driver rules · **6** cohort
rules · risk = **0.55** vibe + **0.45** drivers · **7**-question turn budget ·
**14**-day cooldown · **28**-day vibe lookback, **14**-day activity lookback ·
**~190** provider calls per 30-person day · **99** tests at **93%** coverage.

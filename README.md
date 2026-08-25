# Vibemeter People Experience Bot

An AI check-in bot for the Deloitte coding challenge. It reads six employee data
sources, decides who to talk to each day and why, holds a short conversation to
find the cause, escalates what needs a human, and writes the People Experience
team a daily report.

Runs fully offline. An API key (Groq, Anthropic or Gemini) improves the reply
analysis but nothing requires it — with no key the built-in lexicon runs and
every feature still works.

## Quickstart

```powershell
pip install -r requirements.txt
python data\generate.py --employees 500 --days 120   # sample data (required first)
uvicorn app.main:app --reload                        # console at http://localhost:8000
```

Open the URL **uvicorn** prints. Serving `static/index.html` from a plain file
server (`python -m http.server`, VS Code Live Server) will load the page but
every `/api/*` call 404s, because only the FastAPI app serves both.

If you have an API key, put it in `.env` beside `run_daily.py`; it is loaded
automatically. To set one for a single session instead:
`$env:GROQ_API_KEY = "gsk_..."`. Check it before you rely on it:

```powershell
python check_provider.py     # exit code 0 = the LLM path is live
python diagnose_groq.py      # only if the above fails: prints the real HTTP status
```

PowerShell swallows exit codes silently, so check it explicitly:

```powershell
python check_provider.py; $LASTEXITCODE
```

Or the whole day headless, with the conversations simulated:

```powershell
python run_daily.py --capacity 30 --simulate
# -> reports\people-experience-YYYY-MM-DD.json and .md
```

With a live API key, pace the run to stay inside the provider's rate limit —
`--rpm 15` is a safe demo setting and finishes a 12-person cohort in ~5 minutes:

```powershell
python run_daily.py --capacity 12 --simulate --rpm 15
```

Other flags: `--cooldown-days` (default 14) skips anyone contacted recently —
sustained distress always overrides it — and `--seed` makes a run reproducible,
question selection included.

The run prints each conversation as it completes and ends with which backend
actually analysed the replies. If that line mentions `lexicon` while a key is
set, some replies were throttled and silently fell back — lower `--rpm`.

Docker:

```powershell
docker compose up --build        # http://localhost:8000
```

Tests:

```powershell
python -m pytest tests -q        # 99 tests, ~13s
```

## Using the console

1. **Find today's cohort** runs Step 1: correlates the six sources, scores every
   employee, and applies the selection rules. The queue shows who was picked,
   their risk score, the rule that picked them, and the drivers behind it.
2. **Click anyone** to open their check-in. The evidence panel shows exactly what
   the bot will quote. Type as the employee to walk the conversation; each reply
   is annotated with sentiment, themes and any flags.
3. **Day's report** renders the end-of-day summary.

Things worth trying: reply "not now, I'm busy" (session closes cleanly), reply
with sustained negativity (watch it escalate to P2 mid-conversation), disclose
harassment in the very first answer (P1 immediately, whatever stage), or open a
`positive`-band employee to see the what's-working script.

Consent to be contacted by a human is an explicit opt-in. "yes please" records
yes, "I'd rather not" records no, and anything the bot cannot read as either is
reported as **unclear — ask** rather than guessed at.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/analyse` | Step 1 — correlate sources, return today's cohort |
| `GET` | `/api/cohort` | current cohort |
| `POST` | `/api/chat/start` | open a session for one employee |
| `POST` | `/api/chat/message` | send a reply, get the bot's response |
| `GET` | `/api/report?fmt=json\|markdown` | end-of-day report |
| `GET` | `/health` | liveness |

PowerShell aliases `curl` to `Invoke-WebRequest`, whose flags are different, so
use `Invoke-RestMethod` — it parses the JSON response for you:

```powershell
$api = "http://localhost:8000/api"

# Step 1 — today's cohort
$cohort = Invoke-RestMethod -Method Post -Uri "$api/analyse" -ContentType "application/json" `
          -Body (@{ capacity = 25; cooldown_days = 14 } | ConvertTo-Json)
$cohort.cohort | Select-Object -First 5 name, team, risk_score, trigger

# Step 2 — hold a conversation. start takes employee_id; message takes session_id + message
$s = Invoke-RestMethod -Method Post -Uri "$api/chat/start" -ContentType "application/json" `
     -Body (@{ employee_id = $cohort.cohort[0].employee_id } | ConvertTo-Json)

$s = Invoke-RestMethod -Method Post -Uri "$api/chat/message" -ContentType "application/json" `
     -Body (@{ session_id = $s.session_id; message = "sure, go ahead" } | ConvertTo-Json)
$s.turns[-1].text

# The daily report
Invoke-RestMethod -Uri "$api/report?fmt=markdown"
```

If you prefer curl, call `curl.exe` explicitly so PowerShell does not intercept it:

```powershell
curl.exe -X POST localhost:8000/api/analyse -H "content-type: application/json" `
         -d '{\"capacity\":25}'
```

## Layout

```
app/
  features.py       six sources -> one row per employee
  selection.py      driver rules, risk score, cohort rules   <- the "who and why"
  question_bank.py  (data/question_bank.yaml)                <- the question bank
  conversation.py   state machine + escalation thresholds
  nlp.py            reply analysis: LLM backend or lexicon fallback
  report.py         daily report, JSON and Markdown
  main.py           FastAPI + console
data/
  generate.py       synthetic datasets with planted archetypes + ground truth
  question_bank.yaml
static/index.html   the HR console (served by main.py)
run_daily.py        headless end-to-end day -> reports/
check_provider.py   is the configured API key actually working?
diagnose_groq.py    why isn't it — raw status, key hygiene, model availability
tests/
  conftest.py       forces the lexicon so the suite is hermetic and fast
  test_pipeline.py  selection, features, conversation
  test_providers.py provider wire formats and fallback
  test_regressions.py  one test per fixed defect, named for the defect
  test_api.py       the HTTP surface end to end
docs/SYSTEM_DESIGN.md
```

## Configuration

Reply analysis runs on a lexicon by default and upgrades to an LLM when a key
is present. Set at most one:

| Provider | Variable | Default model | Free tier |
| --- | --- | --- | --- |
| Groq | `GROQ_API_KEY` | `openai/gpt-oss-20b` | no card; ~30 req/min, up to 14,400/day |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001` | starter credits, then ~6c/day here |
| Gemini | `GEMINI_API_KEY` | `gemini-2.5-flash` | no card; ~15 req/min, ~1,500/day |

`VIBEBOT_PROVIDER` forces one when several keys are set; `VIBEBOT_MODEL`
overrides the model. A 30-person day is ~190 calls, inside every free tier's
daily allowance — but requests per minute is the binding limit, so use `--rpm`
to pace rather than absorbing 429s.

429s are retried with backoff and then fall through to the lexicon rather than
losing a reply. `Retry-After` is honoured but capped at 45s
(`VIBEBOT_MAX_RETRY_WAIT`): a provider that has exhausted a *daily* bucket can
ask for many minutes, and obeying that freezes the whole run on one reply.

**If the daily quota is gone, `--rpm` will not save you.** A `Retry-After` of
100s+ means the empty bucket is the daily or token allowance, not the per-minute
one, and it will not refill during the run. After 3 consecutive give-ups
(`VIBEBOT_CIRCUIT_TRIP`) the provider is dropped for the rest of the process and
the run finishes on the lexicon in seconds instead of crawling for 20 minutes
while degrading anyway. The end-of-run summary says how many replies that
affected.

**Model ids expire.** Groq shut down `llama-3.1-8b-instant` and
`llama-3.3-70b-versatile` for free tiers on 2026-08-16. A dead model id fails
exactly like a bad key — silently, straight to the lexicon. `check_provider.py`
tells them apart.

**Before pointing this at real employees:** Google's free tier permits training
on submitted prompts, and these replies carry health and harassment
disclosures. Check any provider's retention policy first.

Selection knobs are request parameters (`capacity`, `positive_share`,
`cooldown_days`); escalation thresholds are named constants at the top of
`app/conversation.py`. `positive_share` is honoured exactly: `0` contacts nobody
who is thriving, and the "what's working" sample never takes the last seat of a
small cohort.

`VIBEBOT_DB` moves the session store (default `vibebot.db` beside the app). The
cooldown reads it, so both the console and `run_daily.py` know who was contacted
recently.

**Note for the Docker path:** `docker-compose.yml` mounts `./reports` but not the
session database, so `docker compose down` discards conversation history. Mount
it, or point `VIBEBOT_DB` at a mounted path, if you need it to persist.

See [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) for the selection logic,
escalation rules, privacy model and known limits.

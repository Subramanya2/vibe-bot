"""HTTP surface: one API for the employee chat and the HR console.

Sessions live in SQLite so a conversation survives a restart and the daily
report can be rebuilt from what actually happened rather than from memory.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import uuid
from contextlib import asynccontextmanager, closing
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .conversation import Session, Turn, reply, start_session
from .features import Datasets, build_features
from .report import build_report, to_markdown
from .selection import Candidate, Driver, select_cohort

ROOT = Path(__file__).resolve().parent.parent
# Overridable so tests (and a container with a mounted volume) can point the
# session store somewhere other than the repo root.
DB = Path(os.getenv("VIBEBOT_DB") or (ROOT / "vibebot.db"))
STATIC = ROOT / "static"

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the feature table once at boot.

    on_event("startup") is deprecated in FastAPI 0.115. Catch broadly here:
    a truncated CSV raises ParserError and a renamed column raises KeyError,
    and either used to abort startup with a traceback instead of the clean
    503 that /api/analyse already knows how to return.
    """
    try:
        _state["features"] = build_features(Datasets.load())
    except Exception as exc:                      # noqa: BLE001 — see docstring
        print(f"datasets not loaded ({type(exc).__name__}: {exc}) — "
              f"run `python data/generate.py`", file=sys.stderr, flush=True)
        _state["features"] = None
    yield


app = FastAPI(title="Vibemeter People Experience Bot", version="1.0",
              lifespan=lifespan)

_state: dict = {"features": None, "cohort": [], "sessions": {}}


# ---------------------------------------------------------------- storage
def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY, employee_id TEXT, day TEXT, payload TEXT)""")
    # /api/report and the cooldown lookup both filter on day.
    con.execute("CREATE INDEX IF NOT EXISTS idx_sessions_day ON sessions(day)")
    return con


def _persist(s: Session) -> None:
    """Write the session, keyed to the day it STARTED.

    str(date.today()) was re-evaluated on every write, so a conversation that
    began at 23:50 and continued past midnight silently moved to the next day
    and vanished from its own daily report. started_at is UTC, and so is this.
    """
    day = (s.started_at or "")[:10] or str(date.today())
    with closing(_db()) as con, con:
        con.execute(
            "REPLACE INTO sessions (session_id, employee_id, day, payload) VALUES (?,?,?,?)",
            (s.session_id, s.employee_id, day, json.dumps(s.to_dict())))


def _load_sessions(day: str | None = None) -> list[Session]:
    day = day or str(date.today())
    with closing(_db()) as con:
        rows = con.execute("SELECT payload FROM sessions WHERE day = ?", (day,)).fetchall()
    out = []
    for (payload,) in rows:
        d = json.loads(payload)
        turns = [Turn(**t) for t in d.pop("turns", [])]
        out.append(Session(**d, turns=turns))
    return out


def _load_session(session_id: str) -> Session | None:
    """One session by id, from disk. The fallback that makes the module
    docstring's 'survives a restart' claim actually true."""
    with closing(_db()) as con:
        row = con.execute("SELECT payload FROM sessions WHERE session_id = ?",
                          (session_id,)).fetchone()
    if not row:
        return None
    d = json.loads(row[0])
    turns = [Turn(**t) for t in d.pop("turns", [])]
    return Session(**d, turns=turns)


def _last_contacted() -> dict[str, date]:
    """employee_id -> the most recent day we spoke to them.

    select_cohort implements a cooldown but takes this map as an argument, and
    nothing used to supply it — so `cooldown_days` was validated, documented,
    echoed to HR as a live guarantee, and completely inert.
    """
    with closing(_db()) as con:
        rows = con.execute(
            "SELECT employee_id, MAX(day) FROM sessions GROUP BY employee_id").fetchall()
    out: dict[str, date] = {}
    for emp, day in rows:
        try:
            out[emp] = date.fromisoformat(day)
        except (TypeError, ValueError):
            continue
    return out


def _rehydrate(d: dict) -> Candidate:
    drivers = [Driver(**x) for x in d.get("drivers", [])]
    return Candidate(**{**d, "drivers": drivers})


# ---------------------------------------------------------------- schemas
class RunRequest(BaseModel):
    capacity: int = Field(25, ge=1, le=200)
    positive_share: float = Field(0.12, ge=0, le=0.5)
    cooldown_days: int = Field(14, ge=0, le=90)


class StartRequest(BaseModel):
    employee_id: str = Field(..., min_length=1, max_length=64)


class MessageRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    # Unbounded before: a 2M-character reply was stored, held in memory for the
    # process lifetime, echoed back, and sent verbatim to the provider.
    message: str = Field(..., min_length=1, max_length=4000)


# ---------------------------------------------------------------- routes


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "datasets_loaded": _state["features"] is not None}


@app.post("/api/analyse")
def analyse_cohort(req: RunRequest) -> dict:
    """Step 1: correlate the sources and choose today's cohort."""
    if _state["features"] is None:
        raise HTTPException(503, "datasets not loaded — run `python data/generate.py`")
    cohort = select_cohort(
        _state["features"], capacity=req.capacity,
        positive_share=req.positive_share, cooldown_days=req.cooldown_days,
        last_contacted=_last_contacted(),
    )
    _state["cohort"] = [c.to_dict() for c in cohort]
    return {"count": len(cohort), "cohort": _state["cohort"]}


@app.get("/api/cohort")
def cohort() -> dict:
    return {"count": len(_state["cohort"]), "cohort": _state["cohort"]}


@app.post("/api/chat/start")
def chat_start(req: StartRequest) -> dict:
    match = next((c for c in _state["cohort"] if c["employee_id"] == req.employee_id), None)
    if not match:
        raise HTTPException(404, "employee not in today's cohort — run /api/analyse first")
    s = start_session(_rehydrate(match), session_id=str(uuid.uuid4())[:8])
    _state["sessions"][s.session_id] = s
    _persist(s)
    return s.to_dict()


@app.post("/api/chat/message")
def chat_message(req: MessageRequest) -> dict:
    s = _state["sessions"].get(req.session_id) or _load_session(req.session_id)
    if not s:
        raise HTTPException(404, "unknown session")
    s = reply(s, req.message)
    _state["sessions"][s.session_id] = s
    _persist(s)
    return s.to_dict()


@app.get("/api/report")
def report(fmt: str = "json"):
    """Step 2 output: the end-of-day summary for the People Experience team."""
    if _state["features"] is None:
        raise HTTPException(503, "datasets not loaded")
    cohort = [_rehydrate(c) for c in _state["cohort"]]
    rep = build_report(cohort, _load_sessions(), _state["features"])
    if fmt not in ("json", "markdown"):
        raise HTTPException(400, "fmt must be 'json' or 'markdown'")
    if fmt == "markdown":
        return PlainTextResponse(to_markdown(rep))
    return rep


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text()


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

"""The conversation itself, plus the rules that hand it to a human.

A small state machine rather than a free-running agent. Three reasons: the turn
count is bounded so the bot cannot interrogate anyone, every question comes from
the reviewed bank in question_bank.yaml, and each session ends with a structured
outcome the report can aggregate. The LLM (when configured) reads replies; it
does not choose what to ask.
"""
from __future__ import annotations

import re

import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .nlp import ReplyAnalysis, analyse
from .selection import Candidate

BANK_PATH = Path(__file__).resolve().parent.parent / "data" / "question_bank.yaml"

# Escalation thresholds. Kept here, named, so HR can change them without code.
ESCALATE_SENTIMENT = -0.45      # mean sentiment across the session
ESCALATE_NEGATIVE_TURNS = 2     # consecutive negative replies
ESCALATE_RISK = 70.0            # analysis risk score

State = str  # greeting | consent | probing | support_offer | closed | crisis

# Question choice is random, which made --seed a lie: two runs with the same
# seed picked different follow-ups. Route it through a module RNG the caller
# can pin.
_RNG = random.Random()


def seed(value: int | None) -> None:
    """Pin question selection so a seeded run is reproducible."""
    _RNG.seed(value)


@dataclass
class Turn:
    role: str                    # bot | employee
    text: str
    stage: str = ""              # which state this reply answered
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    analysis: dict | None = None


@dataclass
class Session:
    session_id: str
    employee_id: str
    name: str
    trigger: str
    band: str
    risk_score: float
    drivers: list[dict]
    state: State = "greeting"
    turns: list[Turn] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    driver_index: int = 0
    probe_index: int = 0
    escalation: dict | None = None
    outcome: dict | None = None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def bot_questions(self) -> int:
        return sum(1 for t in self.turns if t.role == "bot" and t.text.rstrip().endswith("?"))

    @property
    def all_analyses(self) -> list[dict]:
        """Every analysed employee reply, including the consent answer.

        Anything that must never be missed — a crisis disclosure, a harassment
        flag, a theme — reads this. A disclosure made in answer to "is now a
        good time?" is still a disclosure.
        """
        return [t.analysis for t in self.turns
                if t.role == "employee" and t.analysis]

    @property
    def analyses(self) -> list[dict]:
        """Substantive replies only — the basis for the sentiment average.

        The consent answer ("yes, go ahead") is neutral by definition and was
        dragging the session mean toward zero, so it is excluded. Use
        `all_analyses` for anything safety-related: this list is empty after a
        single first-reply disclosure, which previously blanked the escalation.
        """
        return [t.analysis for t in self.turns
                if t.role == "employee" and t.analysis and t.stage != "consent"]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["turns"] = [asdict(t) if not isinstance(t, dict) else t for t in self.turns]
        return d


class Bank:
    def __init__(self, path: Path | str = BANK_PATH):
        self.data = yaml.safe_load(Path(path).read_text())

    @property
    def turn_budget(self) -> int:
        return int(self.data["meta"]["turn_budget"])

    def block(self, driver_code: str | None, band: str) -> dict:
        if band == "positive":
            return self.data["positive"]
        if driver_code and driver_code in self.data["drivers"]:
            return self.data["drivers"][driver_code]
        return self.data["generic"]

    def opening(self, key: str) -> str:
        return self.data["opening"][key].strip()

    def closing(self, key: str) -> str:
        return self.data["closing"][key].strip()

    def crisis(self, key: str) -> str:
        return self.data["crisis"][key].strip()


BANK = Bank()


def start_session(candidate: Candidate, session_id: str, bank: Bank = BANK) -> Session:
    s = Session(
        session_id=session_id,
        employee_id=candidate.employee_id,
        name=candidate.name,
        trigger=candidate.trigger,
        band=candidate.band,
        risk_score=candidate.risk_score,
        drivers=[d if isinstance(d, dict) else asdict(d) for d in candidate.drivers],
    )
    first = candidate.name.split()[0]
    greeting = bank.opening("greeting").format(first_name=first)
    s.turns.append(Turn("bot", f"{greeting} {bank.opening('consent')}"))
    s.state = "consent"
    return s


def _driver_at(s: Session, idx: int) -> dict | None:
    return s.drivers[idx] if idx < len(s.drivers) else None


def _opener_for(s: Session, bank: Bank) -> str:
    driver = _driver_at(s, s.driver_index)
    block = bank.block(driver["code"] if driver else None, s.band)
    evidence = driver["evidence"] if driver else ""
    return block["opener"].strip().format(evidence=evidence)


def _followup_for(s: Session, tone: str, bank: Bank) -> str | None:
    driver = _driver_at(s, s.driver_index)
    block = bank.block(driver["code"] if driver else None, s.band)
    options = list(block.get("followups", {}).get(tone, []))
    probes = list(block.get("probes", []))
    pool = [q for q in options + probes if q not in s.asked]
    if not pool:
        return None
    return _RNG.choice(pool[:2]) if len(pool) > 1 else pool[0]


def _raise(priority: str, reason: str) -> dict:
    """The escalation record. One constructor, so the three call sites agree."""
    return dict(priority=priority, reason=reason,
                raised_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def _should_escalate(s: Session) -> tuple[bool, str, str]:
    """(escalate, priority, reason) — evaluated after every employee reply."""
    # Safety rules look at EVERY reply. The sentiment rules below look only at
    # substantive ones, because a consent answer is neutral by definition.
    everything = s.all_analyses
    if any(x.get("crisis") for x in everything):
        return True, "P1", "Employee disclosed a personal-safety concern"
    serious = {f for x in everything for f in x.get("flags", []) if f in
               {"harassment", "discrimination", "retaliation"}}
    if serious:
        return True, "P1", f"Reply raised {', '.join(sorted(serious))}"
    a = s.analyses
    # The two rules below read the *conversation*, so they need a conversation.
    # Firing on a single reply cut sessions to one question in testing.
    if len(a) < 2:
        return False, "", ""
    mean = sum(x["sentiment"] for x in a) / len(a)
    negative_run = 0
    for x in a:
        negative_run = negative_run + 1 if x["tone"] == "negative" else 0
    if mean <= ESCALATE_SENTIMENT and negative_run >= ESCALATE_NEGATIVE_TURNS:
        return True, "P2", f"Sustained negative sentiment ({mean:+.2f}) across the check-in"
    if s.risk_score >= ESCALATE_RISK and mean <= -0.3 and len(a) >= 2:
        return True, "P2", f"High analysis risk ({s.risk_score}) confirmed by the conversation"
    if "attrition_intent" in {f for x in everything for f in x.get("flags", [])}:
        return True, "P2", "Employee signalled intent to leave"
    return False, "", ""


# Consent refusals. Anchored so "now", "know", "nothing" and "cannot" do not
# read as "no", and negated forms ("not busy") do not read as "busy".
HARD_DECLINE = [
    r"\bno\b", r"\bnope\b", r"\bnot (now|today|right now|a good time)\b",
    r"\blater\b", r"\banother time\b", r"\bcan'?t (talk|now|right now)\b",
    r"\bskip\b", r"\bleave me alone\b", r"\bnot interested\b",
]
SOFT_DECLINE = [r"\bbusy\b", r"\bswamped\b", r"\bin a meeting\b"]
# Phrases that look like refusals but are not, plus explicit go-aheads that
# override a soft signal ("I'm not busy, go ahead").
AFFIRMATIVE = [
    r"\bno problem\b", r"\bno worries\b", r"\bnot busy\b", r"\bnot too busy\b",
    r"\bgo ahead\b", r"\bask away\b", r"\bfire away\b",
]


def _declines(text: str) -> bool:
    """True when the consent answer is a refusal."""
    low = text.lower()
    affirmative = any(re.search(p, low) for p in AFFIRMATIVE)
    # A hard refusal stands even alongside politeness ("no thanks, ask away").
    for pat in HARD_DECLINE:
        if re.search(pat, low):
            if pat == r"\bno\b" and re.search(r"\bno (problem|worries)\b", low):
                continue
            return True
    if any(re.search(p, low) for p in SOFT_DECLINE):
        return not affirmative
    return False


# Answers to "shall I ask someone to reach out?". Consent is opt-in: only an
# explicit yes counts, an explicit no is recorded as no, and anything else is
# recorded as unclear rather than assumed.
CONTACT_YES = [
    r"\byes\b", r"\byeah\b", r"\byep\b", r"\bsure\b", r"\bplease do\b",
    r"\bgo ahead\b", r"\bthat would help\b", r"\bthat'?d help\b",
    r"\bsounds good\b", r"\bi'?d like that\b", r"\bok(ay)?\b", r"\bdo it\b",
]
CONTACT_NO = [
    r"\bno\b", r"\bnope\b", r"\bnot\b", r"\brather not\b", r"\bprefer not\b",
    r"\bdon'?t\b", r"\bdo not\b", r"\bno thanks?\b", r"\bleave it\b",
    r"\bi'?m (fine|alright|ok|okay)\b", r"\bmaybe later\b",
]


def _wants_contact(text: str) -> bool | None:
    """True = explicit yes, False = explicit no, None = could not tell."""
    low = text.lower()
    # Refusal is checked first: "no thanks" and "please don't" both contain
    # tokens that would otherwise read as agreement.
    if any(re.search(p, low) for p in CONTACT_NO):
        return False
    if any(re.search(p, low) for p in CONTACT_YES):
        return True
    return None


def reply(s: Session, text: str, bank: Bank = BANK) -> Session:
    """Feed one employee message in; the bot's response is appended to s.turns."""
    if s.state == "closed":
        return s

    analysis: ReplyAnalysis = analyse(text)
    s.turns.append(Turn("employee", text, stage=s.state, analysis=analysis.to_dict()))

    # --- priority path: a disclosure outranks the script -------------------
    if analysis.crisis:
        s.state = "crisis"
        s.turns.append(Turn("bot", bank.crisis("response")))
        s.turns.append(Turn("bot", bank.crisis("handoff")))
        # Never build an escalation from an empty verdict: a crisis on the very
        # first reply used to land in HR's report with a blank priority and a
        # blank reason. Fall back to the P1 the crisis rule guarantees.
        esc, prio, why = _should_escalate(s)
        if not esc:
            prio, why = "P1", "Employee disclosed a personal-safety concern"
        s.escalation = _raise(prio, why)
        return _close(s, bank, skip_wrap=True)

    # A P1 signal outranks the script wherever it appears. Harassment disclosed
    # in answer to "is now a good time?" used to sit unescalated until the next
    # reply — and if the employee stopped there, it never escalated at all.
    esc, prio, why = _should_escalate(s)
    if esc and prio == "P1" and not s.escalation:
        s.escalation = _raise(prio, why)
        s.turns.append(Turn("bot", bank.closing("escalation_notice")))
        s.state = "support_offer"
        return s

    if s.state == "consent":
        # Word-boundary match, not substring: "now is fine", "no problem" and
        # "I know" are consent, not refusal. And do NOT gate on sentiment here —
        # a distressed "yeah, not great but ok" is the strongest reason to talk,
        # not a decline. Sentiment on this turn is excluded from the average for
        # the same reason (see _conversation_sentiment).
        if _declines(text):
            s.turns.append(Turn("bot", bank.opening("consent_declined")))
            s.outcome = dict(status="declined")
            return _close(s, bank, skip_wrap=True)
        s.state = "probing"
        opener = _opener_for(s, bank)
        s.asked.append(opener)
        s.turns.append(Turn("bot", opener))
        return s

    if s.state == "probing":
        if s.bot_questions >= bank.turn_budget:
            return _wrap(s, bank)

        escalate, prio, why = _should_escalate(s)
        if escalate:
            s.escalation = _raise(prio, why)
            s.turns.append(Turn("bot", bank.closing("escalation_notice")))
            s.state = "support_offer"
            return s

        q = _followup_for(s, analysis.tone, bank)
        if q is None:
            # exhausted this driver — move to the next one, or wrap up
            driver = _driver_at(s, s.driver_index)
            if driver:
                s.turns.append(Turn("bot", bank.block(driver["code"], s.band)["close"].strip()))
            s.driver_index += 1
            if _driver_at(s, s.driver_index) and s.bot_questions < bank.turn_budget - 1:
                nxt = _opener_for(s, bank)
                s.asked.append(nxt)
                s.turns.append(Turn("bot", nxt))
                return s
            return _wrap(s, bank)

        s.asked.append(q)
        s.turns.append(Turn("bot", q))
        return s

    if s.state == "support_offer":
        # Consent to be contacted by a human is an explicit opt-in, never an
        # inference. The previous rule (sentiment >= 0 OR a substring match)
        # recorded "no", "I'd rather not" and "please don't" as consent —
        # neutral sentiment passed >= 0, and "please don't" contains "please".
        # Unclear answers record None, which the report shows as "unclear"
        # rather than guessing on the employee's behalf.
        agreed = _wants_contact(text)
        # Reaching this stage is not itself an escalation — every completed
        # check-in ends with the offer of a human. Only a raised escalation is
        # an escalation, or the report over-counts them badly.
        s.outcome = dict(status="escalated" if s.escalation else "completed",
                         wants_hr_contact=agreed)
        return _close(s, bank)

    return s


def _wrap(s: Session, bank: Bank) -> Session:
    escalate, prio, why = _should_escalate(s)
    if escalate and not s.escalation:
        s.escalation = _raise(prio, why)
        s.turns.append(Turn("bot", bank.closing("escalation_notice")))
        s.state = "support_offer"
        return s
    s.turns.append(Turn("bot", bank.closing("support_offer")))
    s.state = "support_offer"
    return s


def _close(s: Session, bank: Bank, skip_wrap: bool = False) -> Session:
    if not skip_wrap:
        s.turns.append(Turn("bot", bank.closing("wrap")))
    a = s.analyses
    mean = round(sum(x["sentiment"] for x in a) / len(a), 2) if a else None
    # themes and flags come from every reply — a flag raised in the consent
    # answer is still a flag HR needs to see.
    everything = s.all_analyses
    themes = sorted({t for x in everything for t in x.get("themes", [])})
    flags = sorted({f for x in everything for f in x.get("flags", [])})
    # A session that raised an escalation is "escalated", whichever path closed
    # it. The crisis path used to fall through to "completed", so a P1 landed in
    # the report under the same status as an uneventful chat.
    default_status = "escalated" if s.escalation else "completed"
    s.outcome = {**(s.outcome or {}),
                 "status": (s.outcome or {}).get("status", default_status),
                 "mean_sentiment": mean, "themes": themes, "flags": flags,
                 "turns": len(s.turns), "escalated": s.escalation is not None}
    s.state = "closed"
    return s

"""Who does the bot talk to today, and why.

Two stages, kept separate on purpose:

  1. DRIVERS — deterministic rules over the joined feature table. Each fires with
     a severity in 0..1 and a plain-English piece of evidence. Rules, not a
     model, because HR has to be able to read the reason and argue with it.

  2. SELECTION — decides the daily cohort from those drivers under three
     constraints: never miss someone in sustained distress, respect a contact
     cooldown so the bot isn't a nuisance, and keep a slice of the cohort for
     employees who are doing well (the brief asks what's working, not only
     what's broken).

The risk score orders the queue. It never decides membership on its own — the
rules do that, so an employee can always be traced to the rule that picked them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Callable

import pandas as pd

# Driver weights. Tuned so a single critical driver cannot by itself outrank
# sustained low vibe: the vibe is the signal we were asked to explain, the
# drivers are candidate explanations for it.
WEIGHTS = {
    "workload": 0.22,
    "leave_deficit": 0.16,
    "recognition": 0.16,
    "career": 0.20,
    "onboarding": 0.16,
    "disengagement": 0.10,
}

VIBE_WEIGHT = 0.55  # share of the final score owned by the vibe signal itself


@dataclass
class Driver:
    code: str
    label: str
    severity: float          # 0..1
    evidence: str            # quoted back to the employee, verbatim
    detail: dict = field(default_factory=dict)


@dataclass
class Candidate:
    employee_id: str
    name: str
    team: str
    level: str
    manager_id: str
    vibe_latest: int | None
    vibe_mean: float | None
    risk_score: float
    band: str                # critical | elevated | watch | positive
    trigger: str             # the rule that selected them
    drivers: list[Driver]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["drivers"] = [asdict(x) if not isinstance(x, dict) else x for x in self.drivers]
        return d

    @property
    def primary_driver(self) -> Driver | None:
        return self.drivers[0] if self.drivers else None


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


# --------------------------------------------------------------------------
# Driver rules
# --------------------------------------------------------------------------
def _workload(r) -> Driver | None:
    hours = r.get("avg_work_hours")
    if hours is None:
        return None
    over = hours - 9.0
    vs_team = r.get("hours_vs_team") or 0
    after_hours = r.get("after_hours_ratio") or 0
    if over <= 0 and vs_team < 1.0 and after_hours < 0.20:
        return None
    sev = _clamp(0.45 * _clamp(over / 2.5) + 0.30 * _clamp(vs_team / 2.0) + 0.25 * _clamp(after_hours / 0.35))
    if sev < 0.25:
        return None
    bits = [f"averaging {hours:.1f} hours a day over the last two weeks"]
    if vs_team >= 0.8:
        bits.append(f"{vs_team:.1f}h above the {r['team']} median")
    if after_hours >= 0.18:
        bits.append(f"{after_hours:.0%} of messages sent after hours")
    if r.get("weekend_days_worked", 0) >= 3:
        bits.append(f"{int(r['weekend_days_worked'])} weekend days worked")
    return Driver("workload", "Workload and hours", round(sev, 2), ", ".join(bits),
                  dict(avg_work_hours=hours, hours_vs_team=vs_team, after_hours_ratio=after_hours))


def _leave_deficit(r) -> Driver | None:
    gap = r.get("days_since_leave", 999)
    taken = r.get("leave_days_taken", 0)
    if gap < 45:
        return None
    sev = _clamp(0.6 * _clamp((gap - 45) / 120) + 0.4 * _clamp((6 - taken) / 6))
    if sev < 0.25:
        return None
    ev = ("no leave taken in the period we track" if gap >= 999
          else f"last leave was {int(gap)} days ago")
    if taken <= 2:
        ev += f", {int(taken)} day(s) total this period"
    return Driver("leave_deficit", "Rest and time off", round(sev, 2), ev,
                  dict(days_since_leave=gap, leave_days_taken=taken))


def _recognition(r) -> Driver | None:
    gap = r.get("days_since_award", 999)
    count = r.get("award_count", 0)
    if gap < 180:
        return None
    sev = _clamp(0.65 * _clamp((gap - 180) / 240) + 0.35 * _clamp((2 - count) / 2))
    if sev < 0.25:
        return None
    ev = ("no recognition on record" if count == 0
          else f"last recognition was about {int(gap / 30)} months ago")
    return Driver("recognition", "Recognition", round(sev, 2), ev,
                  dict(days_since_award=gap, award_count=count))


def _career(r) -> Driver | None:
    delta = r.get("rating_delta", 0)
    rating = r.get("latest_rating")
    promoted = r.get("promoted_recently", False)
    tenure = r.get("tenure_months", 0)
    if rating is None:
        return None
    sev = 0.0
    bits = []
    if delta <= -1:
        sev += 0.45 * _clamp(abs(delta) / 2)
        bits.append(f"rating moved from {rating - delta} to {rating} last cycle")
    if rating <= 2:
        sev += 0.35
        bits.append(f"latest rating {rating} of 5")
    if not promoted and tenure >= 30:
        sev += 0.25
        bits.append(f"{int(tenure / 12)} years in role without a promotion")
    sev = _clamp(sev)
    if sev < 0.25 or not bits:
        return None
    return Driver("career", "Career and performance", round(sev, 2), "; ".join(bits),
                  dict(rating=rating, rating_delta=delta, promoted_recently=promoted))


def _onboarding(r) -> Driver | None:
    score = r.get("onboarding_score")
    if score is None or r.get("tenure_months", 99) > 12:
        return None
    clarity = r.get("role_clarity") or 3
    mentor = r.get("mentor_assigned")
    training = r.get("training_pct") or 100
    sev = _clamp(0.4 * _clamp((3 - score) / 2) + 0.3 * _clamp((3 - clarity) / 2)
                 + 0.2 * (0.0 if mentor else 1.0) + 0.1 * _clamp((70 - training) / 70))
    if sev < 0.25:
        return None
    bits = [f"onboarding rated {score} of 5"]
    if not mentor:
        bits.append("no mentor assigned")
    if clarity <= 2:
        bits.append("low role clarity")
    if training < 70:
        bits.append(f"training {training}% complete")
    return Driver("onboarding", "Early experience", round(sev, 2), ", ".join(bits),
                  dict(onboarding_score=score, mentor_assigned=mentor, training_pct=training))


def _disengagement(r) -> Driver | None:
    """Silence is a signal: someone who has stopped answering the Vibemeter."""
    if not r.get("vibe_nonresponse"):
        return None
    responses = r.get("vibe_responses", 0)
    sev = _clamp((7 - responses) / 7)
    if sev < 0.25:
        return None
    return Driver("disengagement", "Survey disengagement", round(sev, 2),
                  f"answered the Vibemeter {int(responses)} times in the last four weeks",
                  dict(vibe_responses=responses))


RULES: list[Callable] = [_workload, _leave_deficit, _recognition, _career, _onboarding, _disengagement]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def vibe_severity(r) -> float:
    """0..1 severity from the Vibemeter alone."""
    mean = r.get("vibe_mean")
    if mean is None:
        return 0.35  # unknown mood is mildly concerning, not neutral
    level = _clamp((3.5 - mean) / 2.5)
    streak = _clamp(r.get("vibe_low_streak", 0) / 5)
    trend = _clamp(-(r.get("vibe_trend") or 0) / 0.05)
    drop = _clamp((r.get("vibe_drop_7d") or 0) / 3)
    return _clamp(0.45 * level + 0.25 * streak + 0.2 * trend + 0.1 * drop)


def score_employee(r: dict) -> tuple[float, list[Driver]]:
    drivers = [d for d in (rule(r) for rule in RULES) if d]
    drivers.sort(key=lambda d: d.severity * WEIGHTS.get(d.code, 0.1), reverse=True)
    driver_component = sum(WEIGHTS.get(d.code, 0.1) * d.severity for d in drivers)
    score = VIBE_WEIGHT * vibe_severity(r) + (1 - VIBE_WEIGHT) * _clamp(driver_component)
    return round(100 * _clamp(score), 1), drivers


# --------------------------------------------------------------------------
# Cohort selection
# --------------------------------------------------------------------------
def select_cohort(
    features: pd.DataFrame,
    *,
    capacity: int = 25,
    positive_share: float = 0.12,
    cooldown_days: int = 14,
    last_contacted: dict[str, date] | None = None,
    as_of: date | None = None,
) -> list[Candidate]:
    as_of = as_of or date.today()
    last_contacted = last_contacted or {}

    scored: list[Candidate] = []
    for r in features.to_dict("records"):
        score, drivers = score_employee(r)
        mean, streak = r.get("vibe_mean"), r.get("vibe_low_streak", 0)
        drop = r.get("vibe_drop_7d") or 0

        # -- the rules, in priority order. First match wins and is recorded. --
        if streak >= 3:
            trigger, band = "sustained_distress", "critical"
        elif drop >= 2 and (mean or 5) <= 3.2:
            trigger, band = "sharp_decline", "elevated"
        elif mean is not None and mean <= 2.4:
            trigger, band = "chronic_low", "elevated"
        elif any(d.severity >= 0.7 for d in drivers) and (mean or 5) <= 3.4:
            trigger, band = "critical_driver", "elevated"
        elif r.get("vibe_nonresponse"):
            trigger, band = "went_quiet", "watch"
        elif mean is not None and mean >= 4.3:
            trigger, band = "thriving_sample", "positive"
        else:
            continue

        scored.append(Candidate(
            employee_id=r["employee_id"], name=r["name"], team=r["team"], level=r["level"],
            manager_id=r["manager_id"], vibe_latest=r.get("vibe_latest"), vibe_mean=mean,
            risk_score=score, band=band, trigger=trigger, drivers=drivers,
        ))

    # Cooldown protects people from being asked the same thing every week.
    # Sustained distress overrides it — that is exactly who we must not skip.
    def eligible(c: Candidate) -> bool:
        if c.trigger == "sustained_distress":
            return True
        last = last_contacted.get(c.employee_id)
        return last is None or (as_of - last).days >= cooldown_days

    pool = [c for c in scored if eligible(c)]
    positives = sorted([c for c in pool if c.band == "positive"], key=lambda c: -(c.vibe_mean or 0))
    concerns = sorted([c for c in pool if c.band != "positive"], key=lambda c: -c.risk_score)

    # Two bugs lived here. `int()` truncated, so a 5-seat day at 15% asked for
    # zero positives; and a `max(1, ...)` floor then forced a positive seat even
    # when the caller asked for none — capacity=1 with positive_share=0
    # contacted someone scoring 5.9 and skipped all 134 employees in sustained
    # distress. Round instead of truncating, and never let the "what's working"
    # sample take the last seat: the module's first promise is that nobody in
    # distress is missed.
    want_positive = round(capacity * positive_share)
    n_positive = min(len(positives), want_positive, max(0, capacity - 1))
    cohort = _diversify(concerns, capacity - n_positive) + positives[:n_positive]
    cohort.sort(key=lambda c: (c.band != "critical", -c.risk_score))
    return cohort


def _diversify(candidates: list[Candidate], limit: int) -> list[Candidate]:
    """Fill seats worst-band-first, but round-robin across primary drivers.

    Straight worst-first triage buries every other issue type. On our sample,
    134 employees trip the sustained-distress rule and the risk score ranks
    burnout above all of them, so the top 30 came back 27/30 burnout — a
    one-note report that hides the recognition and career problems entirely.
    Banding is still absolute (critical seats are filled before anyone else);
    the round-robin only decides *which* critical employees get the seats, by
    taking the highest-risk unserved employee from each driver in turn.
    """
    if limit <= 0:
        return []
    chosen: list[Candidate] = []
    for band in ("critical", "elevated", "watch"):
        in_band = [c for c in candidates if c.band == band]
        if not in_band:
            continue
        buckets: dict[str, list[Candidate]] = {}
        for c in in_band:
            key = c.primary_driver.code if c.primary_driver else "unattributed"
            buckets.setdefault(key, []).append(c)
        for b in buckets.values():
            b.sort(key=lambda c: -c.risk_score)
        while len(chosen) < limit and any(buckets.values()):
            order = sorted((k for k in buckets if buckets[k]),
                           key=lambda k: -buckets[k][0].risk_score)
            for key in order:
                if len(chosen) >= limit:
                    break
                chosen.append(buckets[key].pop(0))
        if len(chosen) >= limit:
            break
    return chosen


def explain_selection() -> dict:
    """Machine-readable description of the rules, for the report and the docs."""
    return {
        "stages": ["driver_rules", "risk_score", "cohort_rules"],
        "vibe_weight": VIBE_WEIGHT,
        "driver_weights": WEIGHTS,
        "cohort_rules": [
            ("sustained_distress", "3+ consecutive Frustrated/Sad responses", "critical", "ignores cooldown"),
            ("sharp_decline", "drop of 2+ zones within a week and mean <= 3.2", "elevated", ""),
            ("chronic_low", "28-day mean vibe <= 2.4", "elevated", ""),
            ("critical_driver", "any driver at severity >= 0.7 with mean <= 3.4", "elevated", ""),
            ("went_quiet", "stopped answering the Vibemeter", "watch", ""),
            ("thriving_sample", "28-day mean >= 4.3", "positive", "reserved share of the cohort"),
        ],
    }

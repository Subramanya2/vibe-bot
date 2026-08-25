"""The end-of-day report for the People Experience team.

Deliberately two-layer: the org rollup answers "what is happening", the cohort
table answers "to whom and why", and every row carries the rule that selected
the employee plus the evidence quoted to them. No transcripts — the bot promises
the employee a theme, not a recording, so the report stores summaries and themes
and keeps verbatim text only where an escalation needs it.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from .conversation import Session
from .selection import Candidate, explain_selection

VIBE_LABELS = {1: "Frustrated", 2: "Sad", 3: "Okay", 4: "Happy", 5: "Excited"}

ACTIONS = {
    "workload": "Review staffing on the named accounts; confirm backfills for open roles.",
    "leave_deficit": "Manager to agree a dated break; check whether leave is being discouraged.",
    "recognition": "Surface recent delivery to the practice lead; nominate where warranted.",
    "career": "Schedule a calibration conversation explaining the rating and the path back.",
    "onboarding": "Assign a buddy this week; close the outstanding induction modules.",
    "disengagement": "Low-touch follow-up; do not re-survey for two weeks.",
}


# `cohort` is always list[Candidate] with list[Driver] inside: run_daily builds
# them directly and main.py rehydrates them before calling in. The defensive
# `d["code"] if isinstance(d, dict) else d.code` that used to appear at three
# sites here hid that contract rather than enforcing it.
def build_report(
    cohort: list[Candidate],
    sessions: list[Session],
    features: pd.DataFrame,
    as_of: date | None = None,
) -> dict:
    as_of = as_of or date.today()
    by_emp = {s.employee_id: s for s in sessions}

    # ---- organisation-level ----
    vibes = [v for v in features.vibe_latest.dropna().tolist()]
    # .get, not [] — a single out-of-range vibe score (an export artefact, a
    # "no answer" sentinel) used to KeyError the whole report, which in
    # run_daily.py meant losing a 20-minute paid run at the final step.
    dist = Counter(VIBE_LABELS.get(int(v), "Unknown") for v in vibes)
    at_risk = int((features.vibe_mean <= 2.4).sum())

    driver_counter = Counter()
    for c in cohort:
        for d in c.drivers:
            driver_counter[d.code] += 1

    theme_counter = Counter()
    sentiments = []
    for s in sessions:
        out = s.outcome or {}
        theme_counter.update(out.get("themes", []))
        if out.get("mean_sentiment") is not None:
            sentiments.append(out["mean_sentiment"])

    completed = [s for s in sessions if (s.outcome or {}).get("status") in ("completed", "escalated")]
    declined = [s for s in sessions if (s.outcome or {}).get("status") == "declined"]
    escalations = [s for s in sessions if s.escalation]

    # ---- per-employee ----
    rows = []
    for c in cohort:
        s = by_emp.get(c.employee_id)
        out = (s.outcome or {}) if s else {}
        drivers = [d.code for d in c.drivers]
        rows.append(dict(
            employee_id=c.employee_id, name=c.name, team=c.team, level=c.level,
            manager_id=c.manager_id, risk_score=c.risk_score, band=c.band,
            selected_because=c.trigger,
            vibe_latest=VIBE_LABELS.get(c.vibe_latest or 0, "—"),
            vibe_mean=c.vibe_mean,
            drivers=drivers,
            evidence=[d.evidence for d in c.drivers][:2],
            conversation=out.get("status", "not_started"),
            sentiment=out.get("mean_sentiment"),
            themes=out.get("themes", []),
            flags=out.get("flags", []),
            escalated=bool(s and s.escalation),
            escalation_priority=(s.escalation or {}).get("priority") if s else None,
            next_step=ACTIONS.get(drivers[0], "Manager check-in within the week.") if drivers
            else "Manager check-in within the week.",
        ))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": str(as_of),
        "organisation": {
            "employees_tracked": int(len(features)),
            "vibe_distribution": dict(dist),
            "employees_below_threshold": at_risk,
            "cohort_size": len(cohort),
            "conversations_started": len(sessions),
            "completed": len(completed),
            "declined": len(declined),
            "escalations": len(escalations),
            "mean_conversation_sentiment": round(sum(sentiments) / len(sentiments), 2) if sentiments else None,
            "top_drivers": driver_counter.most_common(6),
            "top_themes_from_replies": theme_counter.most_common(6),
        },
        "escalations": [
            dict(employee_id=s.employee_id, name=s.name, team=next(
                (r["team"] for r in rows if r["employee_id"] == s.employee_id), "—"),
                 priority=s.escalation["priority"], reason=s.escalation["reason"],
                 consented=(s.outcome or {}).get("wants_hr_contact"),
                 flags=(s.outcome or {}).get("flags", []))
            for s in sorted(escalations, key=lambda x: x.escalation["priority"])
        ],
        "employees": rows,
        "selection_logic": explain_selection(),
    }


def _consent_label(v) -> str:
    """None means the employee's answer was not a clear yes or no.

    Rendering that as "no" would misreport a decision; rendering it as "yes"
    would manufacture consent. HR is told to ask.
    """
    return "unclear — ask" if v is None else ("yes" if v else "no")


def to_markdown(rep: dict) -> str:
    o = rep["organisation"]
    lines = [
        f"# People Experience — daily summary, {rep['as_of']}",
        "",
        f"{o['employees_tracked']} employees tracked · {o['cohort_size']} contacted today · "
        f"{o['completed']} completed · {o['declined']} declined · **{o['escalations']} escalated**",
        "",
        "## Mood across the organisation",
        "",
        "| Zone | Employees |", "| --- | --- |",
    ]
    for zone in ("Frustrated", "Sad", "Okay", "Happy", "Excited"):
        lines.append(f"| {zone} | {o['vibe_distribution'].get(zone, 0)} |")
    lines += [
        "",
        f"Mean sentiment in today's conversations: **{o['mean_conversation_sentiment']}** "
        f"(-1 distressed, +1 positive)",
        "",
        "## Why people were contacted",
        "",
        "| Driver | Employees |", "| --- | --- |",
    ]
    lines += [f"| {code} | {n} |" for code, n in o["top_drivers"]]
    lines += ["", "## What they actually said", "", "| Theme | Mentions |", "| --- | --- |"]
    lines += [f"| {t} | {n} |" for t, n in o["top_themes_from_replies"]] or ["| — | — |"]

    if rep["escalations"]:
        lines += ["", "## Escalations", "",
                  "| Priority | Employee | Team | Reason | Consented |", "| --- | --- | --- | --- | --- |"]
        for e in rep["escalations"]:
            lines.append(f"| {e['priority']} | {e['name']} ({e['employee_id']}) | {e['team']} | "
                         f"{e['reason']} | {_consent_label(e['consented'])} |")

    lines += ["", "## Cohort", "",
              "| Employee | Team | Risk | Selected by | Driver | Sentiment | Next step |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rep["employees"]:
        lines.append(
            f"| {r['name']} ({r['employee_id']}) | {r['team']} | {r['risk_score']} | "
            f"{r['selected_because']} | {r['drivers'][0] if r['drivers'] else '—'} | "
            f"{r['sentiment'] if r['sentiment'] is not None else '—'} | {r['next_step']} |"
        )
    return "\n".join(lines)


def save(rep: dict, outdir: Path | str = "reports") -> tuple[Path, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"people-experience-{rep['as_of']}"
    j, m = outdir / f"{stem}.json", outdir / f"{stem}.md"
    j.write_text(json.dumps(rep, indent=2, default=str))
    m.write_text(to_markdown(rep))
    return j, m

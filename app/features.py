"""Step 1 of the brief: pull every source together into one row per employee.

Each feature is computed with a fixed lookback so the numbers mean the same
thing every day, and every driver keeps the raw evidence alongside the score —
the bot quotes that evidence back to the employee, and it lands in the report.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

VIBE_LOOKBACK_DAYS = 28
ACTIVITY_LOOKBACK_DAYS = 14


@dataclass
class Datasets:
    employees: pd.DataFrame
    vibemeter: pd.DataFrame
    leave: pd.DataFrame
    activity: pd.DataFrame
    rewards: pd.DataFrame
    performance: pd.DataFrame
    onboarding: pd.DataFrame

    @classmethod
    def load(cls, root: Path | str = RAW) -> "Datasets":
        root = Path(root)
        missing = [n for n in cls.__annotations__ if not (root / f"{n}.csv").exists()]
        if missing:
            raise FileNotFoundError(
                f"missing dataset(s) {missing} in {root} — run `python data/generate.py` first"
            )
        frames = {n: pd.read_csv(root / f"{n}.csv") for n in cls.__annotations__}
        for col, frame in (
            ("response_date", "vibemeter"),
            ("activity_date", "activity"),
            ("award_date", "rewards"),
            ("start_date", "leave"),
        ):
            frames[frame][col] = pd.to_datetime(frames[frame][col]).dt.date
        return cls(**frames)


def _slope(values: list[float]) -> float:
    """Least-squares slope in vibe-points per day. Negative means declining."""
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _streak_below(scores: list[int], threshold: int = 2) -> int:
    """Length of the current run of scores at or below `threshold`."""
    run = 0
    for s in reversed(scores):
        if s <= threshold:
            run += 1
        else:
            break
    return run


def build_features(ds: Datasets, as_of: date | None = None) -> pd.DataFrame:
    as_of = as_of or date.today()
    vibe_cut = as_of - timedelta(days=VIBE_LOOKBACK_DAYS)
    act_cut = as_of - timedelta(days=ACTIVITY_LOOKBACK_DAYS)

    rows = []
    vibe_by_emp = {k: v for k, v in ds.vibemeter.sort_values("response_date").groupby("employee_id")}
    act_by_emp = {k: v for k, v in ds.activity.groupby("employee_id")}
    leave_by_emp = {k: v for k, v in ds.leave.groupby("employee_id")}
    rew_by_emp = {k: v for k, v in ds.rewards.groupby("employee_id")}
    perf_by_emp = {k: v for k, v in ds.performance.groupby("employee_id")}
    onb_by_emp = {k: v for k, v in ds.onboarding.groupby("employee_id")}

    for emp in ds.employees.itertuples():
        eid = emp.employee_id
        f: dict[str, object] = dict(
            employee_id=eid,
            name=emp.name,
            team=emp.team,
            level=emp.level,
            manager_id=emp.manager_id,
            tenure_months=int(emp.tenure_months),
        )

        # ---- Vibemeter ----
        v = vibe_by_emp.get(eid)
        recent = v[v.response_date >= vibe_cut] if v is not None else None
        scores = recent.vibe_score.tolist() if recent is not None else []
        f["vibe_responses"] = len(scores)
        f["vibe_latest"] = int(scores[-1]) if scores else None
        f["vibe_mean"] = round(float(np.mean(scores)), 2) if scores else None
        f["vibe_trend"] = round(_slope(scores), 4)
        f["vibe_low_streak"] = _streak_below(scores)
        f["vibe_drop_7d"] = None
        if len(scores) >= 4:
            f["vibe_drop_7d"] = int(max(scores[-4:-1]) - scores[-1])
        f["vibe_nonresponse"] = len(scores) < (VIBE_LOOKBACK_DAYS / 2) * 0.5

        # ---- Activity ----
        a = act_by_emp.get(eid)
        a = a[a.activity_date >= act_cut] if a is not None else None
        if a is not None and len(a):
            f["avg_work_hours"] = round(float(a.work_hours.mean()), 2)
            f["max_work_hours"] = round(float(a.work_hours.max()), 2)
            f["after_hours_ratio"] = round(
                float(a.after_hours_messages.sum() / max(1, a.teams_messages.sum())), 3
            )
            f["meetings_per_day"] = round(float(a.meetings.mean()), 2)
            f["weekend_days_worked"] = int(
                sum(1 for d in a.activity_date if d.weekday() >= 5)
            )
        else:
            f.update(avg_work_hours=None, max_work_hours=None, after_hours_ratio=None,
                     meetings_per_day=None, weekend_days_worked=0)

        # ---- Leave ----
        lv = leave_by_emp.get(eid)
        approved = lv[lv.status == "Approved"] if lv is not None else None
        if approved is not None and len(approved):
            last = max(approved.start_date)
            f["days_since_leave"] = (as_of - last).days
            f["leave_days_taken"] = int(approved.days.sum())
            f["sick_leaves"] = int((approved.leave_type == "Sick").sum())
        else:
            f.update(days_since_leave=999, leave_days_taken=0, sick_leaves=0)
        f["leave_rejections"] = int((lv.status == "Rejected").sum()) if lv is not None else 0

        # ---- Rewards ----
        rw = rew_by_emp.get(eid)
        if rw is not None and len(rw):
            f["days_since_award"] = (as_of - max(rw.award_date)).days
            f["award_count"] = len(rw)
        else:
            f.update(days_since_award=999, award_count=0)

        # ---- Performance ----
        pf = perf_by_emp.get(eid)
        if pf is not None and len(pf):
            pf = pf.sort_values("review_cycle")
            ratings = pf.rating.tolist()
            f["latest_rating"] = int(ratings[-1])
            f["rating_delta"] = int(ratings[-1] - ratings[-2]) if len(ratings) > 1 else 0
            f["promoted_recently"] = "Yes" in pf.promoted.tolist()[-2:]
        else:
            f.update(latest_rating=None, rating_delta=0, promoted_recently=False)

        # ---- Onboarding (only meaningful in the first year) ----
        ob = onb_by_emp.get(eid)
        if ob is not None and len(ob) and f["tenure_months"] <= 12:
            r = ob.iloc[0]
            f["onboarding_score"] = int(r.onboarding_score)
            f["mentor_assigned"] = r.mentor_assigned == "Yes"
            f["training_pct"] = int(r.training_completed_pct)
            f["role_clarity"] = int(r.clarity_of_role)
        else:
            f.update(onboarding_score=None, mentor_assigned=None, training_pct=None,
                     role_clarity=None)

        rows.append(f)

    df = pd.DataFrame(rows)

    # Peer baselines make "long hours" mean long *for this team*, not in absolute terms.
    team_hours = df.groupby("team").avg_work_hours.transform("median")
    df["hours_vs_team"] = (df.avg_work_hours - team_hours).round(2)
    return df


if __name__ == "__main__":  # pragma: no cover
    d = Datasets.load()
    out = build_features(d)
    print(out.head(8).to_string())
    print(f"\n{len(out)} employees, {len(out.columns)} features")

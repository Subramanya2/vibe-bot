"""Generate the six input datasets described in the problem statement.

The challenge ships sample data; this reproduces its shape so the pipeline is
runnable standalone. Signals are planted deliberately (burnout, recognition
drought, promotion miss, rocky onboarding) so selection logic can be validated
against a known ground truth, written to ground_truth.csv.

    python data/generate.py --employees 500 --days 120
"""
from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "raw"

VIBE_ZONES = {1: "Frustrated", 2: "Sad", 3: "Okay", 4: "Happy", 5: "Excited"}

# Planted archetypes. weight = share of population.
ARCHETYPES = {
    "healthy":            dict(weight=0.55, vibe=(3.6, 0.7)),
    "burnout":            dict(weight=0.10, vibe=(2.0, 0.6)),
    "recognition_drought":dict(weight=0.09, vibe=(2.5, 0.6)),
    "promotion_miss":     dict(weight=0.08, vibe=(2.2, 0.7)),
    "rocky_onboarding":   dict(weight=0.07, vibe=(2.4, 0.8)),
    "leave_starved":      dict(weight=0.06, vibe=(2.6, 0.6)),
    "thriving":           dict(weight=0.05, vibe=(4.6, 0.4)),
}

FIRST = ["Aarav","Diya","Rohan","Ananya","Kabir","Meera","Arjun","Sara","Vikram","Nisha",
         "Ishaan","Priya","Aditya","Riya","Karthik","Tanvi","Rahul","Neha","Siddharth","Pooja",
         "Aman","Shreya","Nikhil","Divya","Varun","Ira","Manav","Lakshmi","Rohit","Anjali"]
LAST = ["Sharma","Nair","Iyer","Reddy","Patel","Bose","Menon","Gupta","Rao","Desai",
        "Kulkarni","Chatterjee","Joshi","Pillai","Verma","Shetty","Bhat","Kaur","Das","Mehta"]

TEAMS = ["Consulting", "Audit", "Tax", "Risk Advisory", "Technology", "Operations"]
LOCATIONS = ["Bengaluru", "Mumbai", "Gurugram", "Hyderabad", "Kolkata", "Pune"]
LEVELS = ["Analyst", "Consultant", "Senior Consultant", "Manager", "Senior Manager"]


def _pick_archetype(rng: random.Random) -> str:
    names = list(ARCHETYPES)
    weights = [ARCHETYPES[n]["weight"] for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def build(n_employees: int, days: int, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = random.Random(seed)
    npr = np.random.default_rng(seed)
    today = date.today()
    start = today - timedelta(days=days)

    employees, truth = [], []
    for i in range(n_employees):
        emp_id = f"EMP{1000 + i}"
        arch = _pick_archetype(rng)
        tenure_months = rng.randint(1, 5) if arch == "rocky_onboarding" else rng.randint(4, 96)
        employees.append(
            dict(
                employee_id=emp_id,
                name=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                team=rng.choice(TEAMS),
                location=rng.choice(LOCATIONS),
                level=rng.choice(LEVELS),
                manager_id=f"MGR{rng.randint(1, 40):03d}",
                tenure_months=tenure_months,
                joined_on=str(today - timedelta(days=tenure_months * 30)),
            )
        )
        truth.append(dict(employee_id=emp_id, archetype=arch))

    emp_df = pd.DataFrame(employees)
    truth_df = pd.DataFrame(truth)
    arch_of = dict(zip(truth_df.employee_id, truth_df.archetype))

    # ---------------- Vibemeter: every alternate day ----------------
    vibe_rows = []
    for emp in employees:
        arch = arch_of[emp["employee_id"]]
        mu, sigma = ARCHETYPES[arch]["vibe"]
        # a declining trend for the distressed archetypes
        drift = -0.9 if arch in {"burnout", "promotion_miss", "recognition_drought"} else 0.0
        d = start
        k = 0
        while d <= today:
            progress = k / max(1, days / 2)
            score = npr.normal(mu + drift * progress, sigma)
            score = int(np.clip(round(score), 1, 5))
            if rng.random() > 0.12:  # ~12% non-response
                vibe_rows.append(
                    dict(
                        employee_id=emp["employee_id"],
                        response_date=str(d),
                        vibe_score=score,
                        emotion_zone=VIBE_ZONES[score],
                    )
                )
            d += timedelta(days=2)
            k += 1
    vibe_df = pd.DataFrame(vibe_rows)

    # ---------------- Leave ----------------
    leave_rows = []
    for emp in employees:
        arch = arch_of[emp["employee_id"]]
        n_leaves = 0 if arch in {"leave_starved", "burnout"} else rng.randint(2, 9)
        for _ in range(n_leaves):
            s = start + timedelta(days=rng.randint(0, max(1, days - 5)))
            dur = rng.choice([1, 1, 2, 3, 5])
            leave_rows.append(
                dict(
                    employee_id=emp["employee_id"],
                    leave_type=rng.choice(["Casual", "Sick", "Earned", "Unpaid"]),
                    start_date=str(s),
                    end_date=str(s + timedelta(days=dur - 1)),
                    days=dur,
                    status=rng.choices(["Approved", "Rejected"], weights=[0.9, 0.1])[0],
                )
            )
    leave_df = pd.DataFrame(leave_rows)

    # ---------------- Activity tracker ----------------
    act_rows = []
    for emp in employees:
        arch = arch_of[emp["employee_id"]]
        base_hours = 10.6 if arch == "burnout" else (7.9 if arch == "thriving" else 8.7)
        for off in range(days):
            d = start + timedelta(days=off)
            if d.weekday() >= 5 and rng.random() > (0.55 if arch == "burnout" else 0.12):
                continue
            hours = float(np.clip(npr.normal(base_hours, 1.1), 3, 16))
            act_rows.append(
                dict(
                    employee_id=emp["employee_id"],
                    activity_date=str(d),
                    work_hours=round(hours, 2),
                    teams_messages=int(np.clip(npr.normal(48 if arch == "burnout" else 32, 12), 0, 300)),
                    emails_sent=int(np.clip(npr.normal(26 if arch == "burnout" else 17, 8), 0, 200)),
                    meetings=int(np.clip(npr.normal(6 if arch == "burnout" else 3.4, 1.8), 0, 14)),
                    after_hours_messages=int(
                        np.clip(npr.normal(14 if arch == "burnout" else 3, 4), 0, 80)
                    ),
                )
            )
    act_df = pd.DataFrame(act_rows)

    # ---------------- Rewards & recognition ----------------
    rew_rows = []
    for emp in employees:
        arch = arch_of[emp["employee_id"]]
        n = 0 if arch == "recognition_drought" else rng.choices([0, 1, 2, 3], weights=[0.25, 0.4, 0.25, 0.1])[0]
        if arch == "thriving":
            n = max(n, 2)
        for _ in range(n):
            rew_rows.append(
                dict(
                    employee_id=emp["employee_id"],
                    award_type=rng.choice(["Spot Award", "Applause", "Star Performer", "Client Kudos"]),
                    award_date=str(today - timedelta(days=rng.randint(5, 500))),
                    points=rng.choice([50, 100, 250, 500]),
                )
            )
    rew_df = pd.DataFrame(rew_rows)

    # ---------------- Performance ----------------
    perf_rows = []
    for emp in employees:
        arch = arch_of[emp["employee_id"]]
        for cycle_idx, cycle in enumerate(["H1-2025", "H2-2025", "H1-2026"]):
            if arch == "promotion_miss":
                rating = [4, 4, 2][cycle_idx]
            elif arch == "thriving":
                rating = rng.choice([4, 5])
            else:
                rating = int(np.clip(round(npr.normal(3.3, 0.8)), 1, 5))
            perf_rows.append(
                dict(
                    employee_id=emp["employee_id"],
                    review_cycle=cycle,
                    rating=rating,
                    promoted="No" if arch == "promotion_miss" else rng.choices(["Yes", "No"], weights=[0.15, 0.85])[0],
                    manager_feedback_length=int(np.clip(npr.normal(120, 60), 5, 400)),
                )
            )
    perf_df = pd.DataFrame(perf_rows)

    # ---------------- Onboarding ----------------
    onb_rows = []
    for emp in employees:
        if emp["tenure_months"] > 12:
            continue
        arch = arch_of[emp["employee_id"]]
        rocky = arch == "rocky_onboarding"
        onb_rows.append(
            dict(
                employee_id=emp["employee_id"],
                survey_date=emp["joined_on"],
                onboarding_score=rng.randint(1, 2) if rocky else rng.randint(3, 5),
                mentor_assigned="No" if rocky else rng.choices(["Yes", "No"], weights=[0.85, 0.15])[0],
                training_completed_pct=rng.randint(20, 55) if rocky else rng.randint(70, 100),
                clarity_of_role=rng.randint(1, 2) if rocky else rng.randint(3, 5),
            )
        )
    onb_df = pd.DataFrame(onb_rows)

    return {
        "employees": emp_df,
        "vibemeter": vibe_df,
        "leave": leave_df,
        "activity": act_df,
        "rewards": rew_df,
        "performance": perf_df,
        "onboarding": onb_df,
        "ground_truth": truth_df,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--employees", type=int, default=500)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    frames = build(args.employees, args.days, args.seed)
    for name, df in frames.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name:<14} {len(df):>7,} rows -> {OUT / (name + '.csv')}")


if __name__ == "__main__":
    main()

"""Tests for the parts that would quietly do damage if they broke:
selection rules, the crisis path, escalation thresholds, and report integrity.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from app.conversation import Bank, reply, start_session
from app.features import Datasets, build_features
from app.nlp import analyse, detect_crisis, detect_flags
from app.report import build_report
from app.selection import Candidate, score_employee, select_cohort, vibe_severity


# --------------------------------------------------------------- fixtures
def make_row(**over) -> dict:
    base = dict(
        employee_id="EMP1", name="Test Person", team="Consulting", level="Analyst",
        manager_id="MGR1", tenure_months=24, vibe_responses=12, vibe_latest=3,
        vibe_mean=3.4, vibe_trend=0.0, vibe_low_streak=0, vibe_drop_7d=0,
        vibe_nonresponse=False, avg_work_hours=8.2, max_work_hours=9.0,
        after_hours_ratio=0.05, meetings_per_day=3.0, weekend_days_worked=0,
        days_since_leave=20, leave_days_taken=8, sick_leaves=1, leave_rejections=0,
        days_since_award=90, award_count=2, latest_rating=3, rating_delta=0,
        promoted_recently=False, onboarding_score=None, mentor_assigned=None,
        training_pct=None, role_clarity=None, hours_vs_team=0.0,
    )
    base.update(over)
    return base


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return build_features(Datasets.load())


# --------------------------------------------------------------- features
def test_features_cover_every_employee(features):
    ds = Datasets.load()
    assert len(features) == len(ds.employees)
    assert features.employee_id.is_unique


def test_hours_are_measured_against_the_team_not_an_absolute(features):
    assert "hours_vs_team" in features
    # a team's own median is 0 by construction, so the column must centre near 0
    assert abs(features.hours_vs_team.median()) < 0.6


# --------------------------------------------------------------- drivers
def test_healthy_employee_trips_no_driver():
    score, drivers = score_employee(make_row())
    assert drivers == []
    assert score < 40


def test_long_hours_trip_workload_with_quotable_evidence():
    _, drivers = score_employee(make_row(avg_work_hours=11.4, hours_vs_team=2.4, after_hours_ratio=0.3))
    workload = next(d for d in drivers if d.code == "workload")
    assert workload.severity >= 0.5
    assert "11.4" in workload.evidence


def test_no_leave_trips_leave_deficit():
    _, drivers = score_employee(make_row(days_since_leave=999, leave_days_taken=0))
    assert any(d.code == "leave_deficit" for d in drivers)


def test_rating_drop_trips_career():
    _, drivers = score_employee(make_row(latest_rating=2, rating_delta=-2))
    career = next(d for d in drivers if d.code == "career")
    assert "2" in career.evidence


def test_onboarding_only_applies_in_the_first_year():
    early = make_row(tenure_months=4, onboarding_score=1, role_clarity=1,
                     mentor_assigned=False, training_pct=30)
    late = {**early, "tenure_months": 40}
    assert any(d.code == "onboarding" for d in score_employee(early)[1])
    assert not any(d.code == "onboarding" for d in score_employee(late)[1])


def test_vibe_severity_ranks_sustained_low_above_a_single_dip():
    sustained = vibe_severity(make_row(vibe_mean=1.6, vibe_low_streak=6, vibe_trend=-0.04))
    dip = vibe_severity(make_row(vibe_mean=3.3, vibe_low_streak=1, vibe_drop_7d=1))
    assert sustained > dip


# --------------------------------------------------------------- selection
def test_cohort_respects_capacity(features):
    assert len(select_cohort(features, capacity=10)) <= 10


def test_cohort_always_reserves_room_for_positive_conversations(features):
    cohort = select_cohort(features, capacity=25, positive_share=0.2)
    assert any(c.band == "positive" for c in cohort), "brief asks what is working, not only what is broken"


def test_cooldown_suppresses_a_recently_contacted_employee(features):
    first = select_cohort(features, capacity=12)
    target = next(c for c in first if c.trigger != "sustained_distress")
    again = select_cohort(features, capacity=12,
                          last_contacted={target.employee_id: date.today() - timedelta(days=2)})
    assert target.employee_id not in {c.employee_id for c in again}


def test_sustained_distress_overrides_the_cooldown(features):
    first = select_cohort(features, capacity=30)
    critical = next(c for c in first if c.trigger == "sustained_distress")
    again = select_cohort(features, capacity=30,
                          last_contacted={critical.employee_id: date.today()})
    assert critical.employee_id in {c.employee_id for c in again}


def test_cohort_is_not_dominated_by_one_driver(features):
    """Worst-first triage returned 27/30 burnout before the round-robin was added."""
    cohort = select_cohort(features, capacity=30)
    primaries = [c.primary_driver.code for c in cohort if c.primary_driver]
    top = max(set(primaries), key=primaries.count)
    assert primaries.count(top) <= 0.6 * len(primaries)


def test_every_selected_employee_carries_the_rule_that_picked_them(features):
    assert all(c.trigger for c in select_cohort(features, capacity=15))


# --------------------------------------------------------------- nlp
@pytest.mark.parametrize("text", [
    "some days I don't want to be here anymore",
    "I've thought about ending it",
    "honestly I want to kill myself",
])
def test_crisis_language_is_caught(text):
    assert detect_crisis(text) is True
    assert analyse(text).crisis is True


@pytest.mark.parametrize("text", [
    "it's been a heavy quarter but manageable",
    "the deadline killed our weekend",       # figurative, must not trip
])
def test_ordinary_complaints_are_not_crises(text):
    assert detect_crisis(text) is False


def test_serious_flags_are_detected():
    assert "harassment" in detect_flags("my lead humiliated me in the standup")
    assert "attrition_intent" in detect_flags("I've started looking at other offers")


def test_negation_is_handled():
    assert analyse("not happy at all").sentiment < 0
    assert analyse("no issues, genuinely enjoying it").sentiment > 0


# --------------------------------------------------------------- conversation
def _candidate(**over) -> Candidate:
    base = dict(employee_id="EMP1", name="Test Person", team="Consulting", level="Analyst",
                manager_id="MGR1", vibe_latest=2, vibe_mean=2.1, risk_score=72.0,
                band="critical", trigger="sustained_distress", drivers=[])
    base.update(over)
    return Candidate(**base)


def test_session_opens_by_asking_permission():
    s = start_session(_candidate(), "t1")
    assert s.state == "consent"
    assert s.turns[0].text.rstrip().endswith("?")


def test_declining_ends_the_session_immediately():
    s = reply(start_session(_candidate(), "t2"), "not now, I'm busy")
    assert s.state == "closed"
    assert s.outcome["status"] == "declined"
    assert len(s.turns) == 3  # greeting, decline, acknowledgement


def test_crisis_disclosure_stops_the_script_and_escalates_p1():
    s = reply(start_session(_candidate(), "t3"), "yes ok")
    s = reply(s, "I don't want to be here anymore")
    assert s.state == "closed"
    assert s.escalation["priority"] == "P1"
    # the bot must not carry on interrogating after a disclosure
    assert not any(t.text.endswith("hours?") for t in s.turns if t.role == "bot")
    assert "14416" in " ".join(t.text for t in s.turns)


def test_turn_budget_is_enforced():
    bank = Bank()
    s = start_session(_candidate(), "t4")
    s = reply(s, "yes")
    for _ in range(15):
        if s.state == "closed":
            break
        s = reply(s, "it's been difficult and quite draining")
    assert s.bot_questions <= bank.turn_budget + 1


def test_consent_reply_is_excluded_from_session_sentiment():
    s = reply(start_session(_candidate(), "t5"), "yes")
    assert s.analyses == [], "the neutral consent turn was dragging session means toward zero"


def test_sustained_negativity_escalates():
    s = reply(start_session(_candidate(), "t6"), "yes")
    for _ in range(3):
        if s.state == "closed":
            break
        s = reply(s, "I am exhausted, overwhelmed and completely unsupported")
    assert s.escalation is not None
    assert s.escalation["priority"] == "P2"


def test_a_positive_conversation_does_not_escalate():
    c = _candidate(band="positive", trigger="thriving_sample", risk_score=12.0, vibe_mean=4.6)
    s = reply(start_session(c, "t7"), "sure")
    for _ in range(4):
        if s.state == "closed":
            break
        s = reply(s, "genuinely enjoying it, the team is supportive and the work is good")
    assert s.escalation is None


# --------------------------------------------------------------- report
def test_report_counts_only_real_escalations(features):
    cohort = select_cohort(features, capacity=6)
    sessions = []
    for c in cohort:
        s = reply(start_session(c, f"r{c.employee_id}"), "yes")
        for _ in range(3):
            if s.state == "closed":
                break
            s = reply(s, "it's fine honestly, no issues")
        sessions.append(s)
    rep = build_report(cohort, sessions, features)
    assert rep["organisation"]["escalations"] == len(rep["escalations"])
    assert rep["organisation"]["escalations"] == sum(1 for s in sessions if s.escalation)


def test_report_explains_its_own_selection_rules(features):
    rep = build_report(select_cohort(features, capacity=5), [], features)
    rules = rep["selection_logic"]["cohort_rules"]
    assert any(r[0] == "sustained_distress" for r in rules)
    assert rep["selection_logic"]["driver_weights"]


def test_every_reported_employee_has_a_reason_and_a_next_step(features):
    rep = build_report(select_cohort(features, capacity=12), [], features)
    for row in rep["employees"]:
        assert row["selected_because"]
        assert row["next_step"]

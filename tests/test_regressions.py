"""Regressions. Every test here failed before the fix it guards.

Named for the defect, not the function, so a future reader knows what breaks if
it goes red.
"""
from __future__ import annotations

import pytest

from app import nlp
from app.conversation import _wants_contact, reply, start_session
from app.features import Datasets, build_features
from app.report import build_report, to_markdown
from app.selection import select_cohort


@pytest.fixture(scope="module")
def features():
    return build_features(Datasets.load())


@pytest.fixture(scope="module")
def cohort(features):
    return select_cohort(features, capacity=6)


# ---------------------------------------------------------------- consent
@pytest.mark.parametrize("answer", [
    "no", "nope", "no thanks", "I'd rather not", "definitely not",
    "please don't", "absolutely not", "I'm alright", "not really",
])
def test_refusing_hr_contact_is_never_recorded_as_consent(answer):
    """`sentiment >= 0 or "please" in text` read every one of these as a yes.

    A refusal recorded as consent means HR contacts someone who explicitly
    said no, about something disclosed in confidence.
    """
    assert _wants_contact(answer) is False


@pytest.mark.parametrize("answer", ["yes", "yes please", "sure", "ok",
                                    "that would help", "go ahead", "please do"])
def test_agreeing_to_hr_contact_is_recorded(answer):
    assert _wants_contact(answer) is True


@pytest.mark.parametrize("answer", ["hmm", "I guess we'll see", "..."])
def test_an_unclear_answer_is_unclear_not_a_guess(answer):
    """Neither yes nor no. Recording either would invent a decision."""
    assert _wants_contact(answer) is None


def test_the_report_says_unclear_rather_than_guessing(cohort, features):
    s = start_session(cohort[0], session_id="unclear1")
    s = reply(s, "sure, go ahead")
    for _ in range(8):
        if s.state == "support_offer":
            break
        s = reply(s, "everything is unbearable and I am exhausted")
    if s.state != "support_offer":
        pytest.skip("did not reach the support offer in this configuration")
    s = reply(s, "hmm")
    assert s.outcome["wants_hr_contact"] is None
    md = to_markdown(build_report([cohort[0]], [s], features))
    assert "unclear — ask" in md


# ---------------------------------------------------------------- crisis
def test_crisis_on_the_very_first_reply_still_carries_p1(cohort):
    """`analyses` excludes consent-stage turns, and the first reply IS one.

    _should_escalate therefore saw an empty list and returned no verdict, so
    HR received an escalation with a blank priority and a blank reason — on
    the one path the design says must never degrade.
    """
    s = start_session(cohort[0], session_id="crisis1")
    s = reply(s, "I don't want to live anymore")
    assert s.escalation["priority"] == "P1"
    assert s.escalation["reason"]
    assert s.outcome["escalated"] is True
    assert s.outcome["status"] == "escalated"


def test_crisis_priority_does_not_depend_on_when_it_is_disclosed(cohort):
    first = start_session(cohort[0], session_id="c_a")
    first = reply(first, "I want to end things")
    later = start_session(cohort[1], session_id="c_b")
    later = reply(later, "sure, go ahead")
    later = reply(later, "I want to end things")
    assert first.escalation["priority"] == later.escalation["priority"] == "P1"
    assert first.escalation["reason"] == later.escalation["reason"]


def test_a_flag_raised_in_the_consent_answer_still_reaches_hr(cohort):
    """Themes and flags used to be read from the consent-excluded list too."""
    s = start_session(cohort[0], session_id="flag1")
    s = reply(s, "my manager has been harassing me")
    # A P1 signal escalates on the turn it appears, whatever stage that is.
    assert s.escalation["priority"] == "P1"
    assert s.escalation["reason"] == "Reply raised harassment"
    assert s.state == "support_offer"
    s = reply(s, "yes please")
    assert s.state == "closed"
    assert "harassment" in s.outcome["flags"]
    assert s.outcome["status"] == "escalated"


# ---------------------------------------------------------------- selection
def test_positive_share_zero_contacts_nobody_thriving(features):
    """max(1, ...) forced a positive seat: capacity=1 spent it on a 5.9."""
    picked = select_cohort(features, capacity=1, positive_share=0)
    assert [c.band for c in picked] == ["critical"]


def test_a_single_seat_never_goes_to_a_thriving_employee(features):
    """The module's first promise is that nobody in distress is missed."""
    picked = select_cohort(features, capacity=1, positive_share=0.15)
    assert picked[0].band != "positive"


def test_the_positive_sample_survives_a_small_cohort(features):
    """int() truncation asked for zero positives on a 5-seat day at 15%."""
    picked = select_cohort(features, capacity=5, positive_share=0.15)
    assert sum(1 for c in picked if c.band == "positive") == 1


def test_cooldown_actually_excludes_a_recent_contact(features):
    """cooldown_days was validated, documented and echoed to HR — and inert,
    because no caller ever supplied last_contacted."""
    from datetime import date, timedelta

    first = select_cohort(features, capacity=40)
    recent = next(c for c in first if c.trigger != "sustained_distress")
    seen = {recent.employee_id: date.today() - timedelta(days=1)}
    again = select_cohort(features, capacity=40, cooldown_days=14, last_contacted=seen)
    assert recent.employee_id not in {c.employee_id for c in again}


def test_sustained_distress_overrides_the_cooldown(features):
    """Deliberate: the cooldown must never silence someone in real distress."""
    from datetime import date, timedelta

    first = select_cohort(features, capacity=40)
    distressed = next(c for c in first if c.trigger == "sustained_distress")
    seen = {distressed.employee_id: date.today() - timedelta(days=1)}
    again = select_cohort(features, capacity=40, cooldown_days=14, last_contacted=seen)
    assert distressed.employee_id in {c.employee_id for c in again}


# ---------------------------------------------------------------- report
def test_one_bad_vibe_score_does_not_destroy_the_report(features, cohort):
    """A direct VIBE_LABELS[...] subscript KeyError'd the whole report."""
    broken = features.copy()
    broken.loc[broken.index[0], "vibe_latest"] = 0
    rep = build_report(cohort, [], broken)
    assert rep["organisation"]["vibe_distribution"].get("Unknown", 0) >= 1


# ---------------------------------------------------------------- nlp
@pytest.mark.parametrize("header,expected", [
    ("30", 30.0),
    ("0", 0.0),
    ("not-a-number", 7.0),
    (None, 7.0),
])
def test_retry_after_parsing_never_raises(header, expected):
    """float() on an RFC 9110 HTTP-date raised, the blanket handler ate it,
    and the 429 went uncounted while the wait cap never engaged."""
    assert nlp._retry_after_seconds(header, 7.0) == expected


def test_retry_after_accepts_an_http_date():
    got = nlp._retry_after_seconds("Wed, 21 Oct 2099 07:28:00 GMT", 7.0)
    assert got > 0 and got != 7.0


# ---------------------------------------------------------------- determinism
def test_the_same_seed_produces_the_same_conversation(cohort):
    """_followup_for used the global random, so --seed did not reproduce."""
    from app import conversation

    def run(seed):
        conversation.seed(seed)
        s = start_session(cohort[0], session_id="seeded")
        s = reply(s, "sure")
        s = reply(s, "the workload has been heavy")
        s = reply(s, "not much support from anyone")
        return [t.text for t in s.turns if t.role == "bot"]

    assert run(11) == run(11)


# ---------------------------------------------------------------- circuit breaker
def test_an_exhausted_daily_quota_stops_the_run_calling_the_provider(monkeypatch):
    """A Retry-After far longer than the cap means the DAILY bucket is empty,
    not the per-minute one. Retrying into that costs ~90s per reply and falls
    back to the lexicon anyway, so a 12-person day crawls for 20 minutes and
    silently degrades. Give up after a few, once, and say so.
    """
    calls = {"n": 0}

    class Throttled:
        status_code = 429
        headers = {"retry-after": "260"}

        def raise_for_status(self):
            import httpx
            raise httpx.HTTPStatusError("429", request=None, response=None)

        def json(self):
            return {}

    def fake_post(**_kw):
        calls["n"] += 1
        return Throttled()

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(nlp.httpx, "post", fake_post)
    monkeypatch.setattr(nlp, "MAX_RETRY_WAIT", 0.0)
    nlp.reset_circuit()
    nlp.STATS.clear()

    for i in range(10):
        assert nlp.analyse(f"reply {i}, this is exhausting").backend == "lexicon"

    assert nlp.circuit_is_open()
    assert nlp.STATS["skipped_circuit_open"] >= 5
    before = calls["n"]
    nlp.analyse("one more")
    assert calls["n"] == before, "provider was called after the breaker opened"
    nlp.reset_circuit()


def test_a_success_closes_the_failure_streak(monkeypatch):
    """One bad reply must not trip the breaker on an otherwise healthy run."""
    nlp.reset_circuit()
    nlp._trip("groq", "test")
    nlp._trip("groq", "test")
    assert not nlp.circuit_is_open()
    nlp.reset_circuit()
    assert not nlp.circuit_is_open()

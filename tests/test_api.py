"""API-level tests. app/main.py had no coverage at all before these.

They run against a temporary database so a test run never touches the real one.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBEBOT_DB", str(tmp_path / "test.db"))
    from app import main as main_module
    importlib.reload(main_module)
    with TestClient(main_module.app) as c:
        yield c


def test_health_reports_dataset_state(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["datasets_loaded"] is True


def test_the_full_check_in_lifecycle(client):
    cohort = client.post("/api/analyse", json={"capacity": 5}).json()
    assert cohort["count"] == 5
    emp = cohort["cohort"][0]["employee_id"]

    started = client.post("/api/chat/start", json={"employee_id": emp}).json()
    sid = started["session_id"]
    assert started["state"] == "consent"

    said = client.post("/api/chat/message",
                       json={"session_id": sid, "message": "sure, go ahead"}).json()
    assert said["state"] == "probing"

    md = client.get("/api/report", params={"fmt": "markdown"})
    assert md.status_code == 200
    assert md.headers["content-type"].startswith("text/plain")
    assert "People Experience" in md.text


def test_a_session_survives_a_restart(client, tmp_path, monkeypatch):
    """main.py's docstring promised this; chat_message only ever read memory,
    so every in-flight conversation 404'd after a restart."""
    from app import main as main_module

    cohort = client.post("/api/analyse", json={"capacity": 3}).json()
    emp = cohort["cohort"][0]["employee_id"]
    sid = client.post("/api/chat/start", json={"employee_id": emp}).json()["session_id"]
    client.post("/api/chat/message", json={"session_id": sid, "message": "sure"})

    main_module._state["sessions"].clear()          # simulate a process restart

    again = client.post("/api/chat/message",
                        json={"session_id": sid, "message": "the hours are brutal"})
    assert again.status_code == 200
    assert again.json()["session_id"] == sid


def test_unknown_session_and_unknown_employee_are_404s(client):
    client.post("/api/analyse", json={"capacity": 3})
    assert client.post("/api/chat/start", json={"employee_id": "NOPE"}).status_code == 404
    assert client.post("/api/chat/message",
                       json={"session_id": "nosuch", "message": "hi"}).status_code == 404


def test_an_unbounded_message_is_rejected(client):
    """A 2M-character reply used to be stored, held in memory for the process
    lifetime, echoed back, and sent verbatim to the provider."""
    cohort = client.post("/api/analyse", json={"capacity": 2}).json()
    emp = cohort["cohort"][0]["employee_id"]
    sid = client.post("/api/chat/start", json={"employee_id": emp}).json()["session_id"]
    r = client.post("/api/chat/message", json={"session_id": sid, "message": "x" * 100_000})
    assert r.status_code == 422


def test_an_unknown_report_format_is_rejected(client):
    client.post("/api/analyse", json={"capacity": 2})
    assert client.get("/api/report", params={"fmt": "bogus"}).status_code == 400


@pytest.mark.parametrize("payload", [
    {"capacity": 0}, {"capacity": 5000}, {"positive_share": 0.9}, {"cooldown_days": -1},
])
def test_selection_knobs_are_bounded(client, payload):
    assert client.post("/api/analyse", json=payload).status_code == 422


def test_the_console_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "People Experience" in r.text

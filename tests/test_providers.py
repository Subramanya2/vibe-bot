"""Provider backends.

The sandbox has no keys and no route to Groq or Gemini, so these mock the HTTP
layer and assert the two things that actually break in the field: that we send
each provider the shape it expects, and that we read back whatever shape it
returns. Plus the fallback behaviour, which is what keeps the demo alive.
"""
from __future__ import annotations

import json

import pytest

from app import nlp


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for spec in nlp.PROVIDERS.values():
        monkeypatch.delenv(spec["env"], raising=False)
    monkeypatch.delenv("VIBEBOT_PROVIDER", raising=False)
    monkeypatch.delenv("VIBEBOT_MODEL", raising=False)


class FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self._payload, self.status_code = payload, status
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


BODY = {"sentiment": -0.8, "themes": ["workload"], "flags": [], "summary": "Employee is overloaded"}

SHAPES = {
    "anthropic": {"content": [{"type": "text", "text": json.dumps(BODY)}]},
    "groq": {"choices": [{"message": {"content": json.dumps(BODY)}}]},
    "gemini": {"candidates": [{"content": {"parts": [{"text": json.dumps(BODY)}]}}]},
}


# ------------------------------------------------------------- selection
def test_no_key_means_lexicon(monkeypatch):
    assert nlp.active_provider() is None
    assert nlp.analyse("I am exhausted").backend == "lexicon"


@pytest.mark.parametrize("name", list(nlp.PROVIDERS))
def test_a_key_activates_its_provider(monkeypatch, name):
    monkeypatch.setenv(nlp.PROVIDERS[name]["env"], "test-key")
    provider, key, model = nlp.active_provider()
    assert (provider, key) == (name, "test-key")
    assert model == nlp.PROVIDERS[name]["model"]


def test_explicit_provider_wins_over_key_order(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("VIBEBOT_PROVIDER", "groq")
    assert nlp.active_provider()[0] == "groq"


def test_model_can_be_overridden(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("VIBEBOT_MODEL", "llama-3.3-70b-versatile")
    assert nlp.active_provider()[2] == "llama-3.3-70b-versatile"


# ------------------------------------------------------------- wire format
@pytest.mark.parametrize("name", list(nlp.PROVIDERS))
def test_request_carries_the_system_prompt_and_the_reply(name):
    req = nlp._request(name, "k", "m", "I am exhausted")
    blob = json.dumps(req["json"])
    assert "sentiment" in blob, "system prompt must reach the model"
    assert "I am exhausted" in blob
    assert req["url"].startswith("https://")


def test_each_provider_authenticates_the_way_it_expects():
    assert "x-api-key" in nlp._request("anthropic", "k", "m", "t")["headers"]
    assert nlp._request("groq", "k", "m", "t")["headers"]["Authorization"] == "Bearer k"
    assert nlp._request("gemini", "k", "m", "t")["headers"]["x-goog-api-key"] == "k"


@pytest.mark.parametrize("name,payload", SHAPES.items())
def test_each_response_shape_is_parsed(name, payload):
    assert json.loads(nlp._extract(name, payload))["sentiment"] == -0.8


@pytest.mark.parametrize("name", list(nlp.PROVIDERS))
def test_end_to_end_against_a_mocked_provider(monkeypatch, name):
    monkeypatch.setenv(nlp.PROVIDERS[name]["env"], "k")
    monkeypatch.setattr(nlp.httpx, "post", lambda **kw: FakeResponse(SHAPES[name]))
    a = nlp.analyse("relentless quarter, no end in sight")
    assert a.backend == name
    assert a.sentiment == -0.8
    assert a.themes == ["workload"]


# ------------------------------------------------------------- resilience
def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(nlp.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse({}, status=429, headers={"retry-after": "0"})
        return FakeResponse(SHAPES["groq"])

    monkeypatch.setattr(nlp.httpx, "post", flaky)
    assert nlp.analyse("exhausted").backend == "groq"
    assert calls["n"] == 2, "free tiers throttle at ~30 rpm; one 429 must not lose the reply"


def test_persistent_failure_falls_back_to_the_lexicon(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(nlp.time, "sleep", lambda s: None)
    monkeypatch.setattr(nlp.httpx, "post", lambda **kw: FakeResponse({}, status=500))
    a = nlp.analyse("I am exhausted and overwhelmed")
    assert a.backend == "lexicon"
    assert a.sentiment < 0, "the check-in must still produce a reading"


def test_malformed_json_falls_back(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(nlp.time, "sleep", lambda s: None)
    bad = {"choices": [{"message": {"content": "sorry, I can't help with that"}}]}
    monkeypatch.setattr(nlp.httpx, "post", lambda **kw: FakeResponse(bad))
    assert nlp.analyse("exhausted").backend == "lexicon"


def test_crisis_detection_does_not_depend_on_the_provider(monkeypatch):
    """The safety rule must hold when the model is down, wrong, or absent."""
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(nlp.time, "sleep", lambda s: None)
    calm = {"choices": [{"message": {"content": json.dumps(
        {"sentiment": 0.4, "themes": [], "flags": [], "summary": "fine"})}}]}
    monkeypatch.setattr(nlp.httpx, "post", lambda **kw: FakeResponse(calm))
    assert nlp.analyse("I don't want to be here anymore").crisis is True

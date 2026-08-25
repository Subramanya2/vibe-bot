"""Shared test setup.

app/__init__.py loads .env at import time so the entry points pick up an API
key without the caller exporting one. That is right for running the bot and
wrong for the tests: with a real key present, every reply analysed in
test_pipeline.py becomes a live provider call, so the suite turns into a
non-deterministic, billable, network-dependent integration run that asserts on
whatever the model happened to say.

Neutralise the keys for every test. Provider behaviour is covered properly in
test_providers.py, which mocks the transport and asserts on the wire format.
"""
from __future__ import annotations

import pytest

from app import nlp


@pytest.fixture(autouse=True)
def no_live_provider(monkeypatch):
    """Force the deterministic lexicon backend for the whole suite."""
    for spec in nlp.PROVIDERS.values():
        monkeypatch.delenv(spec["env"], raising=False)
    monkeypatch.delenv("VIBEBOT_PROVIDER", raising=False)
    monkeypatch.delenv("VIBEBOT_MODEL", raising=False)
    # Belt and braces: if a test ever does configure a provider, never let a
    # real retry sleep run inside the suite.
    monkeypatch.setattr(nlp.time, "sleep", lambda *_: None)

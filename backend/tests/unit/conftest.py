"""
Unit tests never talk to Groq. backend/.env usually holds a real key, so a
test that forgets to fake one new Groq call would otherwise spend real
quota and pass or fail on the live model's mood. This turns any such call
into an immediate, loud failure. Tests that fake requests.post themselves
(test_groq_retry, test_shared_pacing) replace this with their own fake.
"""

import pytest
import requests

from app import groq_client

_real_post = requests.post


@pytest.fixture(autouse=True)
def _no_live_groq(monkeypatch):
    attempts = []

    def guarded_post(url, *args, **kwargs):
        if url == groq_client.GROQ_ENDPOINT:
            attempts.append(url)
            raise AssertionError("A unit test tried to call the real Groq API -- fake the call instead.")
        return _real_post(url, *args, **kwargs)

    monkeypatch.setattr(groq_client.requests, "post", guarded_post)
    yield
    # Several Groq helpers catch every exception and fall back quietly, which
    # would swallow the error above -- so fail the test here as well.
    assert not attempts, "A unit test tried to call the real Groq API -- fake the call instead."

"""
_post_chat's handling of Groq 429s: per-minute limits are waited out and
retried; daily limits (a long Retry-After) fail straight away.
"""

import pytest
import requests

from app import groq_client


class _Response:
    def __init__(self, status_code, retry_after=None):
        self.status_code = status_code
        self.headers = {"retry-after": retry_after} if retry_after is not None else {}
        self.text = "{}"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def json(self):
        return {"choices": [{"message": {"content": "described"}}]}


@pytest.fixture
def fake_groq(monkeypatch):
    """Serves the queued responses in order and records every sleep."""
    queue, sleeps = [], []
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(groq_client.requests, "post", lambda *a, **k: queue.pop(0))
    monkeypatch.setattr(groq_client.time, "sleep", sleeps.append)
    return queue, sleeps


def test_rate_limit_waits_retry_after_then_succeeds(fake_groq):
    queue, sleeps = fake_groq
    queue.extend([_Response(429, "7"), _Response(429, "3"), _Response(200)])
    assert groq_client._post_chat({"model": "m"}) == "described"
    assert sleeps == [7.0, 3.0]


def test_rate_limit_gives_up_after_max_waits(fake_groq):
    queue, sleeps = fake_groq
    queue.extend([_Response(429, "2")] * (groq_client.MAX_RATE_LIMIT_WAITS + 1))
    with pytest.raises(requests.HTTPError):
        groq_client._post_chat({"model": "m"})
    assert len(sleeps) == groq_client.MAX_RATE_LIMIT_WAITS


def test_daily_limit_fails_without_waiting(fake_groq):
    queue, sleeps = fake_groq
    queue.append(_Response(429, "10800"))  # "try again in 3h"
    with pytest.raises(requests.HTTPError):
        groq_client._post_chat({"model": "m"})
    assert sleeps == []


def test_missing_retry_after_uses_default(fake_groq):
    queue, sleeps = fake_groq
    queue.extend([_Response(429), _Response(200)])
    assert groq_client._post_chat({"model": "m"}) == "described"
    assert sleeps == [5.0]


# --- proactive pacing from the x-ratelimit-* headers ---

def _budget_headers(remaining, limit=8000, full_in="60s"):
    return {"x-ratelimit-limit-tokens": str(limit),
            "x-ratelimit-remaining-tokens": str(remaining),
            "x-ratelimit-reset-tokens": full_in}


def _with_headers(response, **kwargs):
    response.headers.update(_budget_headers(**kwargs))
    return response


@pytest.fixture
def frozen_clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(groq_client.time, "monotonic", lambda: now[0])
    return now


def test_parse_duration_formats():
    assert groq_client._parse_duration("18.495s") == pytest.approx(18.495)
    assert groq_client._parse_duration("21m36s") == 1296
    assert groq_client._parse_duration("1h2m3.5s") == 3723.5
    assert groq_client._parse_duration("250ms") == pytest.approx(0.25)
    assert groq_client._parse_duration(None) is None


def test_first_request_is_not_paced(fake_groq, frozen_clock):
    queue, sleeps = fake_groq
    queue.append(_with_headers(_Response(200), remaining=5000))
    groq_client._post_chat({"model": "m"})
    assert sleeps == []


def test_waits_for_budget_before_a_request_that_would_not_fit(fake_groq, frozen_clock):
    queue, sleeps = fake_groq
    # Request 1: nothing known yet, budget is learned from the reply.
    queue.append(_with_headers(_Response(200), remaining=6000))
    groq_client._post_chat({"model": "m"})
    # Request 2 sent at once: ~6000 available, Groq leaves 3000 -> it cost 3000.
    queue.append(_with_headers(_Response(200), remaining=3000, full_in="60s"))
    groq_client._post_chat({"model": "m"})
    assert sleeps == []
    assert groq_client._token_budget["m"]["header_cost"] == pytest.approx(3000)

    # Budget is now 3000/8000, refilling 5000 over 60s (~83/s). A 3000-token
    # request fits right away...
    queue.append(_with_headers(_Response(200), remaining=0, full_in="60s"))
    groq_client._post_chat({"model": "m"})
    assert sleeps == []
    # ...but now 0 is left, refilling 8000/60s: the next 3000 needs 22.5s.
    queue.append(_with_headers(_Response(200), remaining=0))
    groq_client._post_chat({"model": "m"})
    assert sleeps == [pytest.approx(22.5)]


def test_refill_over_time_avoids_waiting(fake_groq, frozen_clock):
    queue, sleeps = fake_groq
    groq_client._token_budget["m"] = {"limit": 8000, "remaining": 0, "full_in": 60.0,
                                      "at": frozen_clock[0], "header_cost": 3000}
    frozen_clock[0] += 30  # half refilled: 4000 available
    queue.append(_with_headers(_Response(200), remaining=1000))
    groq_client._post_chat({"model": "m"})
    assert sleeps == []


def test_429_teaches_a_limit_missing_from_the_headers(fake_groq, frozen_clock):
    """The vision model case: headers say 8000 TPM, but the binding limit is
    7000 input tokens/min, and only the 429 says so. After one 429, requests
    are paced against a local copy of that bucket."""
    queue, sleeps = fake_groq
    roomy = dict(remaining=7900)  # the header budget never gets in the way here
    drain_per_s = 7000 / 60

    # Two requests go through before anything is known about the limit.
    queue.extend([_with_headers(_Response(200), **roomy), _with_headers(_Response(200), **roomy)])
    groq_client._post_chat({"model": "vision"})
    frozen_clock[0] += 1
    groq_client._post_chat({"model": "vision"})

    # The third is rejected; Groq names the limit, its level and the size.
    limited = _with_headers(_Response(429, "2"), **roomy)
    limited.text = '{"error":{"message":"... (ITPM): Limit 7000, Used 5530, Requested 2184. ..."}}'
    queue.extend([limited, _with_headers(_Response(200), **roomy)])
    frozen_clock[0] += 1
    groq_client._post_chat({"model": "vision"})
    state = groq_client._token_budget["vision"]
    # "Used 5530" over our 2 recent requests = 2765 each -- what Groq really
    # charged, more than its stated "Requested 2184".
    assert (state["window_limit"], state["window_cost"]) == (7000, 2765)
    # After the Retry-After, the retry still waits until 5530 + 2765 fits
    # under 7000: (8295 - 7000) / (7000/60 per s) = 11.1s.
    assert sleeps == [2.0, pytest.approx(1295 / drain_per_s)]

    # The bucket is now at 8295 (frozen clock): the next one waits 34.8s.
    queue.append(_with_headers(_Response(200), **roomy))
    groq_client._post_chat({"model": "vision"})
    assert sleeps[2] == pytest.approx((8295 + 2765 - 7000) / drain_per_s)


def test_no_wait_once_the_learned_bucket_has_drained(fake_groq, frozen_clock):
    queue, sleeps = fake_groq
    groq_client._token_budget["vision"] = {"window_limit": 7000, "window_cost": 2765,
                                           "window_used": 7000, "window_used_at": frozen_clock[0] - 60}
    queue.append(_with_headers(_Response(200), remaining=7900))
    groq_client._post_chat({"model": "vision"})
    assert sleeps == []


def test_budgets_are_per_model(fake_groq, frozen_clock):
    queue, sleeps = fake_groq
    groq_client._token_budget["vision"] = {"limit": 8000, "remaining": 0, "full_in": 60.0,
                                           "at": frozen_clock[0], "header_cost": 3000}
    queue.append(_with_headers(_Response(200), remaining=7000))
    groq_client._post_chat({"model": "text"})
    assert sleeps == []


# --- progress reporting ---

def test_rate_limit_wait_is_reported_as_progress(fake_groq):
    from app import progress
    queue, _ = fake_groq
    queue.extend([_Response(429, "4"), _Response(200)])
    events = []
    with progress.listening(lambda stage, detail: events.append((stage, detail))):
        progress.stage("extraction")
        groq_client._post_chat({"model": "m"})
    assert events == [("extraction", ""), ("extraction", "Rate-limited by Groq -- retrying in 4s")]


def test_json_validate_failed_400_is_its_own_error_with_groqs_message(fake_groq):
    queue, _ = fake_groq
    bad = _Response(400)
    bad.text = '{"error":{"code":"json_validate_failed","failed_generation":"max completion tokens reached"}}'
    queue.append(bad)
    with pytest.raises(groq_client.GroqJSONGenerationFailed, match="max completion tokens"):
        groq_client._post_chat({"model": "m"})


def test_other_http_errors_carry_groqs_message(fake_groq):
    queue, _ = fake_groq
    bad = _Response(400)
    bad.text = '{"error":{"message":"model does not support reasoning_effort"}}'
    queue.append(bad)
    with pytest.raises(requests.HTTPError, match="does not support reasoning_effort"):
        groq_client._post_chat({"model": "m"})

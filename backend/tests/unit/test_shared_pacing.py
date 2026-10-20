"""
Running several jobs at once: Groq pacing state shared through a store
(Redis in production, faked here) with reservations, so concurrent worker
processes don't all see the same free budget; the figure cache merging
concurrent saves; and the Celery pool chosen per platform.
"""

import fnmatch
import sys
from contextlib import contextmanager

import pytest
import requests

from app import groq_client
from tests.unit.test_groq_retry import _Response, _with_headers


@pytest.fixture
def fake_groq(monkeypatch):
    queue, sleeps = [], []
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(groq_client.requests, "post", lambda *a, **k: queue.pop(0))
    monkeypatch.setattr(groq_client.time, "sleep", sleeps.append)
    return queue, sleeps


@pytest.fixture
def frozen_clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(groq_client.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(groq_client.time, "time", lambda: now[0])
    return now


# ---------- reservations ----------

def test_a_sent_request_reserves_budget_until_its_response_arrives(fake_groq, frozen_clock):
    _, sleeps = fake_groq
    groq_client._token_budget["m"] = {"limit": 8000, "remaining": 4000, "full_in": 60.0,
                                      "at": frozen_clock[0], "header_cost": 3000}
    # Two processes decide before either response is back: the first fits...
    first = groq_client._wait_for_token_budget("m")
    assert sleeps == [] and first["available"] == 4000
    # ...and the second sees only 1000 left, so it waits for 2000 to refill
    # at 4000/60s, instead of both going out and one getting a 429.
    second = groq_client._wait_for_token_budget("m")
    assert sleeps == [pytest.approx(30.0)]
    assert len(groq_client._token_budget["m"]["inflight"]) == 2
    assert second["available"] == 1000


def test_the_response_releases_the_reservation(fake_groq, frozen_clock):
    queue, _ = fake_groq
    groq_client._token_budget["m"] = {"limit": 8000, "remaining": 8000, "full_in": 60.0,
                                      "at": frozen_clock[0], "header_cost": 3000}
    queue.append(_with_headers(_Response(200), remaining=5000))
    groq_client._post_chat({"model": "m"})
    assert groq_client._token_budget["m"]["inflight"] == []


def test_a_failed_request_releases_its_reservation(fake_groq, frozen_clock, monkeypatch):
    groq_client._token_budget["m"] = {"limit": 8000, "remaining": 8000, "full_in": 60.0,
                                      "at": frozen_clock[0], "header_cost": 3000}

    def refuse(*a, **k):
        raise requests.exceptions.ConnectionError("reset")

    monkeypatch.setattr(groq_client.requests, "post", refuse)
    with pytest.raises(RuntimeError):
        groq_client._post_chat({"model": "m"})
    assert groq_client._token_budget["m"]["inflight"] == []


def test_stale_reservations_expire(frozen_clock):
    state = {"limit": 8000, "remaining": 8000, "full_in": 60.0, "at": frozen_clock[0],
             "header_cost": 3000, "inflight": [["lost", frozen_clock[0] - 120, 8000]]}
    assert groq_client._header_budget_wait(state, frozen_clock[0]) == 0.0


def test_learned_bucket_is_charged_once_per_request(fake_groq, frozen_clock):
    queue, _ = fake_groq
    groq_client._token_budget["vision"] = {"window_limit": 7000, "window_cost": 2000,
                                           "window_used": 0, "window_used_at": frozen_clock[0]}
    queue.append(_Response(200))
    groq_client._post_chat({"model": "vision"})
    assert groq_client._token_budget["vision"]["window_used"] == 2000


def test_wait_is_rechecked_when_another_process_moved_the_budget(fake_groq, frozen_clock, monkeypatch):
    """While this request slept, another process reserved budget: the wait
    is recomputed instead of sending into an already-spent budget."""
    _, sleeps = fake_groq
    groq_client._token_budget["m"] = {"limit": 8000, "remaining": 0, "full_in": 60.0,
                                      "at": frozen_clock[0], "header_cost": 4000}

    def sleep_while_another_process_sends(seconds):
        sleeps.append(seconds)
        frozen_clock[0] += seconds
        if len(sleeps) == 1:
            groq_client._wait_for_token_budget("m")  # the other process

    monkeypatch.setattr(groq_client.time, "sleep", sleep_while_another_process_sends)
    groq_client._wait_for_token_budget("m")
    # 4000 needs 30s to refill; the other process takes it, so 30s more.
    assert sleeps == [pytest.approx(30.0), pytest.approx(30.0)]


# ---------- the Redis-backed store ----------

class _FakeRedis:
    """Just the commands _RedisBudgetStore uses, over a shared dict --
    two instances on one dict behave like two processes on one Redis."""

    def __init__(self, data):
        self.data = data

    def ping(self):
        return True

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value.encode()

    @contextmanager
    def lock(self, name, timeout=None, blocking_timeout=None):
        yield

    def keys(self, pattern):
        return [k for k in self.data if fnmatch.fnmatch(k, pattern)]


def _redis_store(data):
    store = groq_client._RedisBudgetStore.__new__(groq_client._RedisBudgetStore)
    store._redis = _FakeRedis(data)
    return store


def test_processes_sharing_redis_pace_against_one_budget(fake_groq, frozen_clock, monkeypatch):
    queue, sleeps = fake_groq
    redis_data = {}
    process_a, process_b = _redis_store(redis_data), _redis_store(redis_data)

    monkeypatch.setattr(groq_client, "_store", process_a)
    queue.append(_with_headers(_Response(200), remaining=6000))
    groq_client._post_chat({"model": "m"})
    queue.append(_with_headers(_Response(200), remaining=0, full_in="60s"))
    groq_client._post_chat({"model": "m"})  # learns: one request costs 6000

    # Process B has never sent anything itself, but sees A's budget.
    monkeypatch.setattr(groq_client, "_store", process_b)
    assert groq_client._token_budget == {}
    queue.append(_with_headers(_Response(200), remaining=0))
    groq_client._post_chat({"model": "m"})
    assert sleeps == [pytest.approx(45.0)]  # 6000 at 8000/60s
    assert "researchlens:groq_budget:m" in redis_data


def test_redis_failure_falls_back_to_per_process_pacing(fake_groq, monkeypatch):
    queue, _ = fake_groq

    class Broken:
        shared = True

        def get(self, model):
            raise ConnectionError("redis went away")

        update = get

    monkeypatch.setattr(groq_client, "_store", Broken())
    queue.append(_with_headers(_Response(200), remaining=5000))
    assert groq_client._post_chat({"model": "m"}) == "described"
    assert groq_client._store is groq_client._local_store
    assert groq_client._token_budget["m"]["remaining"] == 5000


def test_store_is_local_without_a_redis_url(monkeypatch):
    monkeypatch.setattr(groq_client, "_store", None)
    monkeypatch.setattr(groq_client, "GROQ_RATE_LIMIT_REDIS_URL", "")
    assert groq_client._budget_store() is groq_client._local_store


def test_store_is_local_when_redis_is_unreachable(monkeypatch):
    monkeypatch.setattr(groq_client, "_store", None)
    monkeypatch.setattr(groq_client, "GROQ_RATE_LIMIT_REDIS_URL", "redis://127.0.0.1:1/0")
    assert groq_client._budget_store() is groq_client._local_store


# ---------- figure cache under concurrent workers ----------

def test_concurrent_figure_cache_saves_keep_both_descriptions(tmp_path, monkeypatch):
    from app import doc_parser
    cache_path = tmp_path / "figure_cache.json"
    monkeypatch.setattr(doc_parser.groq_client, "describe_figure", lambda png, caption: f"desc of {caption}")
    # Two workers load the (empty) cache at the same time...
    worker_a = doc_parser._FigureDescriber(str(cache_path))
    worker_b = doc_parser._FigureDescriber(str(cache_path))
    worker_a.describe(b"png-a", "A")
    worker_b.describe(b"png-b", "B")
    # ...and save one after the other: the second must not erase the first.
    worker_a.save_cache()
    worker_b.save_cache()
    fresh = doc_parser._FigureDescriber(str(cache_path))
    assert fresh.describe(b"png-a", "A") == "desc of A"
    assert fresh.describe(b"png-b", "B") == "desc of B"
    assert fresh.requests_made == 0
    assert list(tmp_path.glob("*.tmp")) == []


# ---------- worker pool ----------

def test_worker_pool_matches_the_platform():
    from app import config
    from app.celery_app import celery_app
    if sys.platform == "win32":
        assert (celery_app.conf.worker_pool, celery_app.conf.worker_concurrency) == ("solo", 1)
    else:
        assert celery_app.conf.worker_pool == "prefork"
        assert celery_app.conf.worker_concurrency == config.CELERY_WORKER_CONCURRENCY

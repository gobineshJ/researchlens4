"""
Thin wrapper around Groq's chat completions endpoint (OpenAI-compatible,
so plain `requests` is enough -- no extra SDK dependency).

Jobs:
  filter_relevant_pages() -- Day 2: score discovered pages, decide which
                              are worth fully scraping.
  expand_query()           -- rewrites a short/generic instruction into a
                              few more specific search queries, so retrieval
                              isn't stuck matching against one narrow phrasing.
  generate_report()       -- Day 3: turn retrieved chunks into a grounded,
                              citation-aware report.
  describe_figure()       -- turn a rendered PDF figure into a text
                              description that can be chunked like any text.

All of them raise GroqNotConfigured with a clear message if no API key
is set, rather than failing with a confusing HTTP error -- the pipeline
can catch this and tell the user exactly what's missing.

Every request goes through _post_chat, which also handles Groq's
per-minute rate limits: requests are paced so they fit the model's
token budget, and a 429 that happens anyway is waited out and retried.
"""

import base64
import copy
import json
import logging
import re
import threading
import time
import uuid
import requests

from . import chunker, progress

from .config import (
    GROQ_API_KEY, GROQ_FILTER_MODEL, GROQ_GENERATION_MODEL, GROQ_MAX_CONTEXT_TOKENS,
    GROQ_VISION_MODEL, QUERY_EXPANSION_VARIANTS, GROQ_RATE_LIMIT_REDIS_URL,
    GROQ_JUDGE_MODEL, ENTAILMENT_MAX_SOURCE_TOKENS, ENTAILMENT_MAX_COMPLETION_TOKENS,
    MULTI_HOP_EVIDENCE_TOKENS,
)

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2  # doubles each retry: 2s, 4s, 8s
MAX_RATE_LIMIT_WAITS = 6       # 429 retries per request, separate from MAX_RETRIES
MAX_RATE_LIMIT_WAIT_SECONDS = 60


class GroqNotConfigured(Exception):
    pass


class GroqJSONGenerationFailed(requests.HTTPError):
    """Groq's 400 "json_validate_failed": in JSON mode the model didn't
    produce a valid document -- with reasoning models, usually because the
    reasoning used up max_completion_tokens before the JSON was written."""


# --- Proactive pacing ---
#
# Groq enforces several per-minute limits per model, and a request that
# breaks one comes back as a 429. Pacing tracks two of them so a request
# that won't fit WAITS instead of being sent, rejected and retried:
#
#   1. The budget in the response headers (x-ratelimit-limit-tokens /
#      -remaining-tokens / -reset-tokens = time until full again).
#   2. Limits that are NOT in the headers, learned from the first 429 that
#      names them. The vision model's binding limit is one of these: 7000
#      *input* tokens/min, against which Groq counts each image request as
#      ~2184 tokens ("Limit 7000, Used 5530, Requested 2184"). From then on
#      a local copy of that bucket -- refilling Limit/60 tokens a second,
#      re-synced to Groq's "Used" on every 429 -- decides when the next
#      request fits.
#
# Every worker process sharing the API key spends the same budget, so the
# state lives in a store: Redis when GROQ_RATE_LIMIT_REDIS_URL is set (the
# Celery broker by default), so all processes pace against one copy, or a
# per-process dict otherwise. A request RESERVES its expected cost in the
# store as it is sent, so a second process deciding at the same moment sees
# the budget already spoken for, instead of both seeing room and both
# being rejected.
#
# The 429 retry in _post_chat stays as the fallback: the first request(s)
# to a model go unpaced (nothing is known yet), and anything else using the
# same key outside this app spends the budget without being seen.

logger = logging.getLogger(__name__)

_token_budget: dict[str, dict] = {}
# model -> {
#   header budget:  "limit", "remaining", "full_in" (s), "at" (clock time),
#                   "header_cost" (tokens one request took from it, measured),
#                   "inflight" [[reservation id, sent at, cost], ...] (sent,
#                   not yet reflected in the headers)
#   learned limit:  "window_limit", "window_cost" (tokens Groq charges one
#                   request), "window_used" + "window_used_at" (bucket level
#                   and when it was that), "unpriced" [send times before the
#                   cost was known]
#   "version":      bumped on every change, so a waiting request can tell
#                   whether another process moved the budget while it slept
# }

_WINDOW_SECONDS = 60
_INFLIGHT_SECONDS = 60  # a reservation whose response never came is dropped after this
_LIMIT_IN_429 = re.compile(r"Limit (\d+), Used (\d+), Requested (\d+)")

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_DURATION_SECONDS = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}


class _LocalBudgetStore:
    """Pacing state for this process only (tests, eager mode, no Redis)."""
    shared = False

    def __init__(self, states: dict[str, dict]):
        self._states = states
        self._lock = threading.Lock()

    def get(self, model: str) -> dict | None:
        with self._lock:
            state = self._states.get(model)
            return copy.deepcopy(state) if state is not None else None

    def update(self, model: str, change):
        with self._lock:
            return change(self._states.setdefault(model, {}))


class _RedisBudgetStore:
    """Pacing state shared by every process pointed at the same Redis."""
    shared = True
    _KEY = "researchlens:groq_budget:{}"

    def __init__(self, url: str):
        import redis
        self._redis = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
        self._redis.ping()

    def get(self, model: str) -> dict | None:
        raw = self._redis.get(self._KEY.format(model))
        return json.loads(raw) if raw else None

    def update(self, model: str, change):
        key = self._KEY.format(model)
        with self._redis.lock(key + ":lock", timeout=10, blocking_timeout=10):
            raw = self._redis.get(key)
            state = json.loads(raw) if raw else {}
            result = change(state)
            self._redis.set(key, json.dumps(state), ex=24 * 3600)
            return result


_local_store = _LocalBudgetStore(_token_budget)
_store = None  # chosen on first use -- see _budget_store()


def _budget_store():
    global _store
    if _store is None:
        _store = _local_store
        if GROQ_RATE_LIMIT_REDIS_URL:
            try:
                _store = _RedisBudgetStore(GROQ_RATE_LIMIT_REDIS_URL)
            except Exception as exc:
                logger.warning("Groq rate-limit pacing is per-process: Redis at %s is unreachable (%s)",
                               GROQ_RATE_LIMIT_REDIS_URL, exc)
    return _store


def _store_call(method: str, *args):
    """Runs one store operation; a Redis failure mid-run downgrades to
    per-process pacing instead of failing the Groq request."""
    global _store
    store = _budget_store()
    try:
        return getattr(store, method)(*args)
    except Exception as exc:
        if store is _local_store:
            raise
        logger.warning("Shared Groq rate-limit state failed (%s) -- pacing per-process from now on", exc)
        _store = _local_store
        return getattr(_local_store, method)(*args)


def _now() -> float:
    # A shared store compares timestamps written by different processes,
    # which only wall-clock time allows; monotonic time is per-process.
    return time.time() if _budget_store().shared else time.monotonic()


def _parse_duration(text: str | None) -> float | None:
    """Groq's reset format: "18.495s", "21m36s", "1h2m3.5s", "250ms"."""
    parts = _DURATION_PART.findall(text or "")
    if not parts:
        return None
    return sum(float(value) * _DURATION_SECONDS[unit] for value, unit in parts)


def _available_tokens(state: dict, now: float) -> float:
    """The header budget refills steadily, reaching `limit` after `full_in` seconds."""
    if state["full_in"] <= 0:
        return state["limit"]
    refill_per_second = (state["limit"] - state["remaining"]) / state["full_in"]
    return min(state["limit"], state["remaining"] + refill_per_second * (now - state["at"]))


def _live_inflight(state: dict, now: float) -> list:
    return [entry for entry in state.get("inflight", []) if now - entry[1] < _INFLIGHT_SECONDS]


def _header_budget_wait(state: dict, now: float) -> float:
    cost = state.get("header_cost", 0)
    if not cost or "limit" not in state or state["full_in"] <= 0:
        return 0.0
    available = _available_tokens(state, now) - sum(e[2] for e in _live_inflight(state, now))
    if available >= cost:
        return 0.0
    refill_per_second = (state["limit"] - state["remaining"]) / state["full_in"]
    return (cost - available) / refill_per_second if refill_per_second > 0 else 0.0


def _learned_used(state: dict, now: float) -> float:
    """Our copy of the learned bucket: drains Limit/60 tokens per second."""
    drain = state["window_limit"] / _WINDOW_SECONDS * (now - state["window_used_at"])
    return max(0.0, state["window_used"] - drain)


def _learned_limit_wait(state: dict, now: float) -> float:
    limit, cost = state.get("window_limit"), state.get("window_cost")
    if not limit or not cost:
        return 0.0
    over = _learned_used(state, now) + cost - limit
    return over / (limit / _WINDOW_SECONDS) if over > 0 else 0.0


def _wait_for_token_budget(model: str) -> dict | None:
    """
    Sleeps until the model's known limits can take one more request like
    the previous ones, then reserves that request's cost in the store.
    Returns the reservation ("id", "window": whether the learned bucket was
    charged, "available": header-budget tokens free as it is sent), or None
    when nothing is known about the model yet.

    After a sleep the budget is only re-checked if another process changed
    it in the meantime -- otherwise the wait was already computed against
    the state that still holds. Total waiting is capped at
    MAX_RATE_LIMIT_WAIT_SECONDS; past that the request goes anyway and the
    429 retry takes over.
    """
    waited, seen_version = 0.0, None
    while True:
        state = _store_call("get", model)
        if state is None:
            return None
        if seen_version is not None and state.get("version") == seen_version:
            break
        now = _now()
        wait = min(max(_header_budget_wait(state, now), _learned_limit_wait(state, now)),
                   MAX_RATE_LIMIT_WAIT_SECONDS - waited)
        if wait <= 0:
            break
        if wait > 0.5:
            progress.detail(f"Waiting {wait:.0f}s for Groq's per-minute token limit")
        time.sleep(wait)
        waited += wait
        seen_version = state.get("version")

    reservation_id = uuid.uuid4().hex

    def reserve(state: dict) -> dict:
        now = _now()
        inflight = _live_inflight(state, now)
        available = (_available_tokens(state, now) - sum(e[2] for e in inflight)
                     if "limit" in state else None)
        if state.get("header_cost"):
            inflight.append([reservation_id, now, state["header_cost"]])
        state["inflight"] = inflight
        charge_window = bool(state.get("window_cost") and state.get("window_limit"))
        if charge_window:
            state["window_used"] = _learned_used(state, now) + state["window_cost"]
            state["window_used_at"] = now
        state["version"] = state.get("version", 0) + 1
        return {"id": reservation_id, "window": charge_window, "available": available}

    return _store_call("update", model, reserve)


def _release_reservation(model: str, reservation: dict | None) -> None:
    """Drops a reservation whose request never got a response."""
    if reservation is None:
        return

    def release(state: dict) -> None:
        state["inflight"] = [e for e in state.get("inflight", []) if e[0] != reservation["id"]]
        state["version"] = state.get("version", 0) + 1

    _store_call("update", model, release)


def _record_token_budget(model: str, response, reservation: dict | None, sent_at: float) -> None:
    available_at_send = reservation["available"] if reservation else None

    def record(state: dict) -> None:
        state["version"] = state.get("version", 0) + 1
        if reservation:
            state["inflight"] = [e for e in state.get("inflight", []) if e[0] != reservation["id"]]

        if response.status_code == 429:
            # Groq naming the limit this request broke, how full it is, and
            # how big it judged the request to be.
            match = _LIMIT_IN_429.search(response.text or "")
            if match:
                limit, used, requested = (int(g) for g in match.groups())
                now = _now()
                if not state.get("window_cost"):
                    # First sight of this limit. "Requested" is Groq's
                    # up-front estimate; what it actually charged shows in
                    # "Used" -- measured at ~2700 per image against a stated
                    # 2184 -- so spread "Used" over our recent sends too and
                    # keep the larger.
                    recent = [t for t in state.get("unpriced", []) if now - t < _WINDOW_SECONDS]
                    state["window_cost"] = round(max(requested, used / len(recent) if recent else 0))
                else:
                    state["window_cost"] = max(state["window_cost"], requested)
                state["window_limit"] = min(limit, state.get("window_limit") or limit)
                # Groq's "Used" is authoritative: it already counts every
                # process's requests, so it replaces any reservations.
                state["window_used"], state["window_used_at"] = used, now
                state.pop("unpriced", None)
        elif response.status_code == 200:
            if reservation and reservation["window"]:
                pass  # already charged to the learned bucket when reserved
            elif state.get("window_cost"):
                state["window_used"] = _learned_used(state, sent_at) + state["window_cost"]
                state["window_used_at"] = sent_at
            else:
                state.setdefault("unpriced", []).append(sent_at)

        headers = response.headers
        try:
            limit = int(headers["x-ratelimit-limit-tokens"])
            remaining = int(headers["x-ratelimit-remaining-tokens"])
        except (KeyError, TypeError, ValueError):
            return
        if response.status_code == 200 and available_at_send is not None:
            spent = available_at_send - remaining
            if spent > 0:
                state["header_cost"] = spent
        state.update(limit=limit, remaining=remaining, at=_now(),
                     full_in=_parse_duration(headers.get("x-ratelimit-reset-tokens")) or 0.0)

    _store_call("update", model, record)


def _retry_after_seconds(response) -> float:
    """How long a 429 says to wait: the Retry-After header (whole seconds,
    rounded up by Groq), or 5s if it's missing."""
    try:
        return max(float(response.headers.get("retry-after", "")), 1.0)
    except ValueError:
        return 5.0


def _call_groq(model: str, system_prompt: str, user_prompt: str, json_mode: bool = True,
                temperature: float = 0.2, **options) -> str:
    """`options` go into the payload as-is (e.g. reasoning_effort,
    max_completion_tokens)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    payload.update(options)
    return _post_chat(payload)


def _read_stream(response, on_text) -> str:
    """
    Reads a streamed (server-sent events) completion, calling
    on_text(text_so_far) as content arrives, and returns the full text.
    Reasoning models stream their reasoning in a separate "reasoning"
    field first; only "content" is the answer.
    """
    text = ""
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            delta = json.loads(data)["choices"][0].get("delta", {})
        except (ValueError, KeyError, IndexError):
            continue
        piece = delta.get("content")
        if piece:
            text += piece
            on_text(text)
    return text


def _post_chat(payload: dict, timeout: int = 30, on_text=None) -> str:
    """
    Sends one chat-completions payload and returns the reply text, with
    the retry/error handling every Groq call shares. Split out of
    _call_groq so describe_figure can send multi-part (text + image)
    messages through the same path.

    With `on_text`, the reply is streamed: on_text(text_so_far) is called
    as it arrives (rate-limit and error handling are unchanged -- those are
    decided by the status code, before any of the body is read).
    """
    if on_text is not None:
        payload = {**payload, "stream": True}
    if not GROQ_API_KEY:
        raise GroqNotConfigured(
            "GROQ_API_KEY is not set. Get a free key at console.groq.com/keys "
            "and add it to backend/.env"
        )
    model = payload["model"]

    last_error = None
    rate_limit_waits = 0
    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            reservation = _wait_for_token_budget(model)
            sent_at = _now()
            try:
                response = requests.post(
                    GROQ_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=timeout,
                    stream=on_text is not None,
                )
            except Exception:
                _release_reservation(model, reservation)
                raise
            _record_token_budget(model, response, reservation, sent_at)
            if response.status_code == 413:
                # Groq's own message says which limit was hit (e.g. "Request
                # too large ... on tokens per minute (TPM): Limit 8000,
                # Requested 11234") -- surface it instead of a bare HTTPError.
                raise RuntimeError(
                    f"Groq rejected the request as too large for model '{model}'. "
                    f"Lower GROQ_MAX_CONTEXT_TOKENS in backend/.env or use fewer/smaller "
                    f"sources. Groq said: {response.text}"
                )
            if (response.status_code == 429 and rate_limit_waits < MAX_RATE_LIMIT_WAITS
                    and _retry_after_seconds(response) <= MAX_RATE_LIMIT_WAIT_SECONDS):
                # Per-minute limit hit (e.g. the vision model's free tier
                # is 7000 input tokens/min and one figure is ~2200, so a
                # paper's figures can't all go in the same minute). Groq
                # says how long until there's room -- wait that long and
                # resend. Doesn't use up a network-error attempt. A longer
                # wait means a daily limit ("try again in 3h"): that falls
                # through to raise_for_status rather than stalling the run.
                rate_limit_waits += 1
                attempt -= 1
                wait = _retry_after_seconds(response)
                progress.detail(f"Rate-limited by Groq -- retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            if response.status_code >= 400:
                # Keep Groq's own explanation -- a bare "400 Bad Request"
                # says nothing about which limit or rule was broken.
                if response.status_code == 400 and "json_validate_failed" in (response.text or ""):
                    raise GroqJSONGenerationFailed(
                        f"Groq could not produce valid JSON for model '{model}': {response.text[:500]}",
                        response=response,
                    )
                try:
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    raise requests.HTTPError(f"{exc} -- Groq said: {(response.text or '')[:500]}",
                                             response=response) from None
            if on_text is not None:
                return _read_stream(response, on_text)
            return response.json()["choices"][0]["message"]["content"]

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            # Transient network issue (firewall/AV resetting the connection,
            # a dropped Wi-Fi packet, etc.) -- worth retrying a few times
            # before giving up, rather than failing the whole pipeline on
            # one bad connection.
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise RuntimeError(
                f"Could not reach Groq after {MAX_RETRIES} attempts "
                f"(connection kept resetting). This usually means a firewall, "
                f"antivirus, or network is blocking/interfering with the "
                f"connection to api.groq.com. Original error: {last_error}"
            ) from exc


def filter_relevant_pages(candidates: list[dict], topic: str, instructions: str) -> list[dict]:
    """
    Scores each discovered candidate page for relevance to the topic +
    instructions. Returns the same list with two fields added:
        "relevant": bool, "reason": str

    Only candidates with "relevant": True should go on to scrape_page().
    """
    if not candidates:
        return []

    listing = "\n".join(
        f"{i}. title: {c['title'] or '(no title)'} | snippet: {c['snippet'] or '(none)'} | url: {c['url']}"
        for i, c in enumerate(candidates)
    )

    system_prompt = (
        "You are a research relevance filter. Given a topic, instructions, "
        "and a numbered list of candidate web pages (title + short snippet), "
        "decide which pages are worth fully reading. Respond ONLY with JSON: "
        '{"decisions": [{"index": 0, "relevant": true, "reason": "..."}, ...]} '
        "with one entry per candidate index, no extra text."
    )
    user_prompt = f"Topic/instructions: {topic}\n{instructions}\n\nCandidates:\n{listing}"

    # temperature=0 -- this is a mechanical filtering DECISION (relevant
    # or not), not creative writing. Sampling here would make which pages
    # get scraped vary between identical runs for no good reason.
    raw = _call_groq(GROQ_FILTER_MODEL, system_prompt, user_prompt, json_mode=True, temperature=0)
    parsed = json.loads(raw)
    decisions = {d["index"]: d for d in parsed.get("decisions", [])}

    results = []
    for i, candidate in enumerate(candidates):
        decision = decisions.get(i, {"relevant": False, "reason": "not scored"})
        results.append({
            **candidate,
            "relevant": bool(decision.get("relevant", False)),
            "reason": decision.get("reason", ""),
        })
    return results


def expand_query(instructions: str, num_variants: int = QUERY_EXPANSION_VARIANTS,
                  avoid: list[str] | None = None) -> list[str]:
    """
    Rewrites the user's instructions into a few more specific search
    queries. A short generic instruction ("focus on research gaps")
    embeds poorly against long, specific chunks -- retrieving separately
    for a handful of more concrete rewordings ("limitations of current
    methods", "authors' suggested future work") catches relevant chunks
    the original phrasing alone would miss.

    `avoid`, when given, is a list of queries already tried (e.g. from a
    first retrieval attempt that came back poorly grounded) -- the model
    is asked to propose genuinely different angles instead of near-
    duplicates of them. Used by pipeline.py's self-correction retry.

    Always returns the ORIGINAL instructions as the first element, even
    if the Groq call fails -- so a failure here degrades gracefully to
    the old single-query behavior instead of breaking retrieval entirely.
    """
    queries = [instructions]
    if not GROQ_API_KEY:
        return queries  # can't expand without a key; fall back silently

    avoid_clause = ""
    if avoid:
        avoid_list = "\n".join(f"- {q}" for q in avoid)
        avoid_clause = (
            "\n\nThese queries were already tried and didn't surface good results -- "
            f"propose genuinely DIFFERENT angles, not close rewordings of them:\n{avoid_list}"
        )

    system_prompt = (
        f"You rewrite a research report instruction into {num_variants} alternative, "
        "more specific search queries that would each surface different relevant "
        "passages from a document. Keep each query short (under 15 words) and "
        "focused on one concrete angle. Respond ONLY with JSON: "
        '{"queries": ["...", "...", "..."]} with exactly '
        f"{num_variants} entries, no extra text.{avoid_clause}"
    )
    try:
        # temperature=0 -- query expansion feeds retrieval directly, so it
        # needs to be deterministic. Sampling here (as it was before) meant
        # identical instructions on identical documents could produce
        # different rerank scores run to run, purely from wording drift in
        # the expanded queries -- confirmed as the cause of exactly that
        # symptom on a real "summary" test.
        raw = _call_groq(GROQ_FILTER_MODEL, system_prompt, instructions, json_mode=True, temperature=0)
        parsed = json.loads(raw)
        variants = [q.strip() for q in parsed.get("queries", []) if isinstance(q, str) and q.strip()]
        queries.extend(variants[:num_variants])
    except Exception:
        # Any failure here (network, bad JSON, rate limit) -- just use the
        # original instructions alone rather than failing the whole pipeline
        # over a "nice to have" retrieval improvement.
        pass
    return queries


def decompose_query(instructions: str) -> list[str]:
    """
    Breaks a compound instruction into 2-3 independent sub-questions when
    it genuinely asks about more than one thing (e.g. "what dataset did
    they use, and what are the paper's main limitations?" -> two
    sub-questions). Each sub-question then gets its own retrieval pass
    (via expand_query + hybrid search), so evidence for EACH part gets
    pulled in, rather than one query embedding averaging across both
    topics and potentially missing one of them entirely.

    This is query decomposition / "compositional" multi-hop: sub-
    questions are retrieved independently and in parallel, not as a
    chain where each hop's retrieved content shapes the next query (that
    would be iterative multi-hop, which this does NOT implement -- being
    upfront about that distinction matters for describing this honestly).

    Returns [instructions] unchanged -- i.e. "no decomposition needed" --
    when the instruction is already a single focused question/topic, or
    on any failure. Never returns an empty list.
    """
    if not GROQ_API_KEY:
        return [instructions]

    system_prompt = (
        "Decide whether this research report instruction asks about MULTIPLE "
        "distinct things that would each need their own search of the source "
        "document (e.g. two separate questions, or two unrelated aspects/themes "
        "joined by 'and'), or whether it's really just ONE focused question or "
        "topic even if it's phrased with multiple clauses.\n"
        "If it's genuinely compound, split it into 2-3 independent "
        "sub-questions, each answerable on its own. If it's already a single "
        "focused question or topic, return it unchanged as the only entry.\n"
        'Respond ONLY with JSON: {"sub_questions": ["...", "..."]}, no extra text.'
    )
    try:
        # temperature=0 -- this decides how many separate retrieval passes
        # run, so it needs to be deterministic like the other routing calls.
        raw = _call_groq(GROQ_FILTER_MODEL, system_prompt, instructions, json_mode=True, temperature=0)
        parsed = json.loads(raw)
        sub_questions = [q.strip() for q in parsed.get("sub_questions", []) if isinstance(q, str) and q.strip()]
        return sub_questions[:3] if sub_questions else [instructions]
    except Exception:
        return [instructions]


def classify_intent(instructions: str) -> str:
    """
    Classifies `instructions` into one of three retrieval strategies:
      "summary"      -- asks for coverage of the WHOLE document, no
                         specific sub-topic named (e.g. "summarize this").
      "targeted_qa"  -- asks a specific, narrow question with one
                         findable answer.
      "analysis"     -- asks for discussion, comparison, or synthesis of
                         a real topic/theme -- broader than one fact, but
                         not asking to cover the whole document.

    This exists because a vague "summarize" instruction and a narrow
    factual question need fundamentally different retrieval strategies,
    not just different prompts: "summarize" isn't asking for the 5
    chunks most SIMILAR to the word "summarize" (which is meaningless),
    it's asking for chunks spread across the document's full span. A
    narrow question, by contrast, genuinely benefits from similarity-
    and keyword-based retrieval honing in on one place in the text.

    Returns "targeted_qa" on any failure or when Groq isn't configured --
    the safest default, since it's the strategy the pipeline already used
    before intent routing existed.
    """
    if not GROQ_API_KEY:
        return "targeted_qa"

    system_prompt = (
        "Classify the following research report instruction into exactly "
        "one of three categories.\n"
        '"summary": asks for an overview or summary of the ENTIRE '
        "document, with no specific sub-topic named (e.g. \"summarize "
        "this paper\", \"give me an overview\").\n"
        '"targeted_qa": asks a specific, narrow question with one '
        "findable answer (e.g. \"what dataset did they use?\").\n"
        '"analysis": asks for discussion, comparison, or synthesis of a '
        "real topic or theme -- broader than one fact, but not asking to "
        "cover the whole document (e.g. \"discuss the limitations\", "
        "\"compare the two approaches\").\n"
        'Respond ONLY with JSON: {"intent": "summary"} or '
        '{"intent": "targeted_qa"} or {"intent": "analysis"}, no extra text.'
    )
    try:
        # temperature=0 -- this decides which retrieval code path runs,
        # so it needs to be deterministic like the other routing/filtering
        # decisions in this module.
        raw = _call_groq(GROQ_FILTER_MODEL, system_prompt, instructions, json_mode=True, temperature=0)
        parsed = json.loads(raw)
        intent = parsed.get("intent", "targeted_qa")
        return intent if intent in ("summary", "targeted_qa", "analysis") else "targeted_qa"
    except Exception:
        return "targeted_qa"


FIGURE_PROMPT_VERSION = "1"
# Part of the figure cache key (see doc_parser): bump it whenever the prompt
# below changes, so cached descriptions written by the old prompt are not
# silently reused.

_FIGURE_PROMPT = (
    "This image is a figure from a research paper. Describe it so that the "
    "description alone can answer questions about the figure. State the kind "
    "of figure (bar chart, line plot, diagram, photo, ...), the axis labels and "
    "units, the legend entries or panel labels, and every number or piece of "
    "text you can actually read. Then state the main comparison or trend it "
    "shows. Do NOT guess values you cannot read -- say they are not legible. "
    "Plain prose, no markdown, under 250 words.\n\n"
    "The paper's own caption for this figure: {caption}"
)


def describe_figure(image_png: bytes, caption: str) -> str:
    """
    Returns a text description of one rendered figure. `caption` is the
    paper's own caption text, given to the model as context -- it tells
    the model what the figure is meant to show, which makes misreading a
    chart much less likely than describing it cold.

    Raises GroqNotConfigured without a key, and lets any other request
    failure propagate: doc_parser decides what to do with a figure that
    could not be described.
    """
    image_b64 = base64.b64encode(image_png).decode("ascii")
    payload = {
        "model": GROQ_VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _FIGURE_PROMPT.format(caption=caption[:1500] or "(none found)")},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }],
        # temperature=0 -- a description is transcription, not writing, and
        # it gets cached, so whatever comes back first is what every later
        # run sees.
        "temperature": 0,
        # The model reasons by default; reading labels off a chart doesn't
        # need it, and the reasoning tokens would only add latency.
        "reasoning_effort": "none",
        "max_completion_tokens": 600,
    }
    # Images make for slower requests than the text-only calls above.
    return _post_chat(payload, timeout=60).strip()


# Lives in chunker now, so the summary sampler can size its own output
# against this same budget (chunker.plan_summary_excerpts). Kept bound here
# under the original name because callers and tests reference it.
_estimate_tokens = chunker.estimate_tokens


def _endpoint_first_order(indices: list[int], chunks: list[dict]) -> list[int]:
    """
    Reorders one source's chunk indices so that taking ANY prefix of the
    result gives the widest-spread subset of that size: first the lowest
    and highest document positions, then repeatedly whichever remaining
    chunk sits nearest the middle of the widest gap left behind.

    This is what makes budget trimming safe for "summary" intent. The
    trim keeps a prefix of each source's queue and discards the tail, so
    with the default relevance ordering the chunks it discards are
    position-blind -- a document's opening chunks can rank lowest on the
    cross-encoder and get dropped, collapsing the positional span that
    summary retrieval exists to produce. Ordering endpoints first makes
    that outcome structurally impossible at any trim depth.

    Chunks without a "position" keep their given order (nothing to spread
    across), so this degrades to the default behaviour.
    """
    positioned = [i for i in indices if isinstance(chunks[i].get("position"), int)]
    if len(positioned) < 2:
        return list(indices)
    unpositioned = [i for i in indices if i not in set(positioned)]

    remaining = sorted(positioned, key=lambda i: chunks[i]["position"])
    order = [remaining.pop(0), remaining.pop()]
    while remaining:
        # Pick the chunk that best splits the widest gap among the ones
        # already chosen -- classic farthest-point sampling.
        chosen = sorted(chunks[i]["position"] for i in order)
        best = max(
            remaining,
            key=lambda i: min(abs(chunks[i]["position"] - p) for p in chosen),
        )
        order.append(best)
        remaining.remove(best)
    return order + unpositioned


def fit_chunks_to_budget(chunks: list[dict], max_tokens: int = GROQ_MAX_CONTEXT_TOKENS,
                          preserve_span: bool = False) -> list[dict]:
    """
    Trims `chunks` so their combined text fits in `max_tokens`, so the
    generation request stays under Groq's per-request size limit (see
    GROQ_MAX_CONTEXT_TOKENS in config.py).

    Chunks are taken round-robin across sources (each source's list in
    its existing ranked order), so trimming drops the lowest-ranked chunks
    of EVERY source evenly rather than silently cutting whole documents
    off the end. The kept chunks are returned in their original order.
    Always keeps at least one chunk, truncating its text if it alone is
    over budget.

    preserve_span: consider each source's chunks for keeping in
        endpoint-first order instead of their given (relevance) order, so
        trimming can never collapse a document's positional span. Set this
        for "summary" intent, where selection is positional and relevance
        rank is the wrong thing to trim by. Only the ORDER OF
        CONSIDERATION changes -- the returned list is still in input
        order, so the generation prompt sees chunks as it did before.
    """
    if not chunks:
        return []

    by_source: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        by_source.setdefault(c.get("source_name", ""), []).append(i)

    if preserve_span:
        by_source = {
            name: _endpoint_first_order(indices, chunks)
            for name, indices in by_source.items()
        }

    kept: set[int] = set()
    used = 0
    queues = list(by_source.values())

    if preserve_span:
        # Two round-robin passes first -- every document's lowest position,
        # then every document's highest -- so the available room buys full
        # spans for as many documents as it can before any interior chunk
        # is considered. Interiors are then filled document by document
        # rather than round-robin: once endpoints are secured, the room
        # that is left is better spent deepening coverage than scattering
        # single chunks. Pass 1 alone already guarantees every document
        # keeps representation (source_coverage).
        order = []
        for _ in range(2):
            order += [queue.pop(0) for queue in queues if queue]
        order += [i for queue in queues for i in queue]
        for i in order:
            cost = _estimate_tokens(chunks[i]["text"])
            if used + cost <= max_tokens:
                kept.add(i)
                used += cost
    else:
        while any(queues):
            for queue in queues:
                if not queue:
                    continue
                i = queue.pop(0)
                cost = _estimate_tokens(chunks[i]["text"])
                if used + cost <= max_tokens:
                    kept.add(i)
                    used += cost
            # Keep cycling -- a later, shorter chunk may still fit after a
            # longer one didn't.

    if not kept:
        first = chunks[0]
        return [{**first, "text": first["text"][: max_tokens * 4]}]
    return [c for i, c in enumerate(chunks) if i in kept]


_CITATION_MARKER = re.compile(r"\[(\d+(?:\s*[-–,]\s*\d+)*)\]")


def cited_indices(report_text: str, num_sources: int) -> list[int]:
    """Every source index the report's [n] / [1, 3] / [2-4] markers cite,
    in first-cited order."""
    seen: list[int] = []
    for body in _CITATION_MARKER.findall(report_text or ""):
        for part in re.split(r"\s*,\s*", body):
            bounds = [int(b) for b in re.split(r"\s*[-–]\s*", part) if b.isdigit()]
            numbers = range(bounds[0], bounds[-1] + 1) if len(bounds) == 2 and 0 < bounds[1] - bounds[0] < 20 else bounds
            for n in numbers:
                if 1 <= n <= num_sources and n not in seen:
                    seen.append(n)
    return seen


def citations_for(report_text: str, chunks: list[dict]) -> list[str]:
    """The source labels a report actually cites, read off its markers."""
    return [chunks[i - 1]["source_label"] for i in cited_indices(report_text, len(chunks))]


DRAFT_UPDATE_SECONDS = 0.75


def generate_report(chunks: list[dict], instructions: str) -> dict:
    """
    chunks: list of {"text": ..., "source_label": ..., "source_type": ...}
            (source_label is the human-readable citation string, e.g.
            "paper.pdf, page 3" or "Page Title, https://...")

    Returns {"report_text": str, "citations_used": [source_label, ...]}

    The report is streamed: progress.draft() receives the text written so
    far (at most every DRAFT_UPDATE_SECONDS), which the UI shows while the
    run is still going. Plain text rather than JSON mode, so the stream is
    readable as it arrives -- the cited sources are read off the [n]
    markers in the text, which is also what the entailment check reads.
    """
    numbered_context = "\n\n".join(
        f"[{i+1}] Source: {c['source_label']}\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    system_prompt = (
        "You are a research report writer. Using ONLY the numbered source "
        "excerpts provided, write a report that follows the user's "
        "instructions. Every factual claim must be followed by a citation "
        "marker like [1] or [2] referring to the source excerpt it came "
        "from. Do not invent facts not present in the sources.\n"
        # The entailment check's two most common findings on real reports:
        # a source described from its title alone ("provides a
        # comprehensive overview") and a cited sentence padded with
        # details from a different excerpt.
        "Each sentence may only say what the excerpt(s) it cites actually "
        "state. Never describe a source's scope, purpose, findings or "
        "structure beyond what its excerpt shows -- a title or author list "
        "alone supports only the title and authors. Do not attach details "
        "from one excerpt to a citation of another: if a sentence combines "
        "facts from several excerpts, cite each one, or split the sentence. "
        "If the excerpts don't cover something the instructions ask for, "
        "say so rather than filling the gap.\n"
        "Respond with the report text only: no preamble, no closing remarks, "
        "no JSON, no list of sources at the end (the [n] markers are the citations)."
    )
    user_prompt = f"Instructions: {instructions}\n\nSource excerpts:\n{numbered_context}"

    last_sent = [0.0]

    def on_text(text_so_far: str) -> None:
        now = time.monotonic()
        if now - last_sent[0] >= DRAFT_UPDATE_SECONDS:
            last_sent[0] = now
            progress.draft(text_so_far)

    payload = {
        "model": GROQ_GENERATION_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    # A long report can take a while to finish; the timeout is per read,
    # and tokens keep arriving throughout.
    report_text = _post_chat(payload, timeout=60, on_text=on_text).strip()
    progress.draft(report_text)
    return {"report_text": report_text, "citations_used": citations_for(report_text, chunks)}


def _excerpts_within(chunks: list[dict], max_tokens: int) -> list[str]:
    """Each chunk's text, cut to an equal share of `max_tokens` only if
    they don't all fit whole."""
    if sum(_estimate_tokens(c["text"]) for c in chunks) <= max_tokens:
        return [c["text"] for c in chunks]
    share_chars = max(200, max_tokens * 4 // max(1, len(chunks)))
    return [c["text"][:share_chars] for c in chunks]


def plan_follow_up(instructions: str, evidence: list[dict], tried_queries: list[str],
                   max_tokens: int = MULTI_HOP_EVIDENCE_TOKENS) -> dict:
    """
    One step of iterative multi-hop retrieval: reads the evidence retrieved
    so far and decides whether answering `instructions` needs a fact the
    evidence POINTS TO but doesn't contain -- the bridge in "what dataset did
    the method that beat the 2019 baseline use?" is the method's name, which
    only the first hop's results reveal. If so, returns follow-up search
    queries that name that thing, so the next hop can look it up.

    Returns {"complete": bool, "missing": str, "queries": [str, ...]}.
    complete=True (and no queries) on any failure or without a key, so a
    planner problem just ends the hop loop instead of failing the run.
    """
    done = {"complete": True, "missing": "", "queries": []}
    if not GROQ_API_KEY or not evidence:
        return done

    excerpts = "\n\n".join(
        f"[{i + 1}] {c.get('source_label', '')}\n{text}"
        for i, (c, text) in enumerate(zip(evidence, _excerpts_within(evidence, max_tokens)))
    )
    tried = "\n".join(f"- {q}" for q in tried_queries)
    system_prompt = (
        "You plan document retrieval for a research question, one hop at a time. "
        "Given the question and the evidence retrieved so far, decide whether the "
        "question can be answered from this evidence. If answering it needs a fact "
        "that the evidence points to but does not contain -- for example the "
        "evidence names a method, dataset, model, author or result, and the "
        "question needs a further fact ABOUT that named thing -- write 1-2 short "
        "search queries (under 15 words) that use the exact name from the "
        "evidence. Do not repeat or reword the queries already tried. If the "
        "evidence already suffices, or further searching would not help, set "
        'complete to true. Respond ONLY with JSON: {"complete": true or false, '
        '"missing": "what is still missing, or empty", "queries": ["..."]}, no extra text.'
    )
    user_prompt = (f"Question: {instructions}\n\nQueries already tried:\n{tried}\n\n"
                   f"Evidence so far:\n{excerpts}")
    try:
        # temperature=0 -- this decides whether (and what) to retrieve next,
        # a routing decision like classify_intent/decompose_query.
        parsed = json.loads(_call_groq(GROQ_FILTER_MODEL, system_prompt, user_prompt,
                                       json_mode=True, temperature=0))
    except Exception:
        return done
    tried_lower = {q.strip().lower() for q in tried_queries}
    queries = [q.strip() for q in parsed.get("queries", []) if isinstance(q, str) and q.strip()
               and q.strip().lower() not in tried_lower][:2]
    complete = bool(parsed.get("complete", True)) or not queries
    return {"complete": complete, "missing": str(parsed.get("missing", "") or ""),
            "queries": [] if complete else queries}


_VERDICTS = ("supported", "unsupported", "contradicted")


def _pack_claim_batches(claims: list[dict], chunks: list[dict], max_tokens: int) -> list[list[int]]:
    """
    Groups claim indices so each batch's distinct cited excerpts fit
    `max_tokens`. Claims citing the same excerpts share a batch for free,
    so a report built from a handful of excerpts is usually one request.
    """
    batches: list[list[int]] = []
    batch: list[int] = []
    batch_sources: set[int] = set()
    for ci, claim in enumerate(claims):
        sources = set(claim["sources"])
        combined = batch_sources | sources
        cost = sum(_estimate_tokens(chunks[i - 1]["text"]) for i in combined)
        if batch and cost > max_tokens:
            batches.append(batch)
            batch, batch_sources = [], set()
            combined = sources
        batch.append(ci)
        batch_sources = combined
    if batch:
        batches.append(batch)
    return batches


def judge_claims(claims: list[dict], chunks: list[dict],
                 max_tokens: int = ENTAILMENT_MAX_SOURCE_TOKENS) -> list[dict]:
    """
    Entailment check: for each claim ({"text": sentence, "sources": [cited
    1-based indices into `chunks`]}), asks an LLM judge whether the excerpts
    it cites actually SUPPORT what it says -- catching a report that cites
    the right source but misstates it, which vocabulary-overlap
    groundedness can't.

    Returns one {"claim", "sources", "verdict", "reason"} per claim, in
    order; verdict is "supported", "unsupported", "contradicted", or
    "unjudged" when the judge skipped it. Raises on request failures --
    the caller decides how a check that couldn't run is reported.
    """
    results = [{"claim": c["text"], "sources": c["sources"], "verdict": "unjudged", "reason": ""}
                for c in claims]
    system_prompt = (
        "You check whether sentences from a generated research report are faithful "
        "to the source excerpts they cite. For each numbered claim, judge it ONLY "
        "against the excerpts it cites:\n"
        '"supported": the cited excerpts state or directly imply everything the claim says.\n'
        '"unsupported": the claim adds, overstates, or changes something the cited '
        "excerpts don't say (including wrong numbers, wrong attribution, or a "
        "stronger conclusion than the source draws).\n"
        '"contradicted": the cited excerpts say the opposite.\n'
        "Be strict about numbers and attributions, lenient about paraphrase. "
        'Respond ONLY with JSON: {"verdicts": [{"claim": 1, "verdict": "supported", '
        '"reason": "one short sentence"}, ...]} with one entry per claim, no extra text.'
    )
    pending = _pack_claim_batches(claims, chunks, max_tokens)
    while pending:
        batch = pending.pop(0)
        source_ids = sorted({i for ci in batch for i in claims[ci]["sources"]})
        texts = _excerpts_within([chunks[i - 1] for i in source_ids], max_tokens)
        excerpts = "\n\n".join(f"[{i}] {text}" for i, text in zip(source_ids, texts))
        listing = "\n".join(
            f"{n + 1}. (cites {', '.join(f'[{i}]' for i in claims[ci]['sources'])}) {claims[ci]['text']}"
            for n, ci in enumerate(batch)
        )
        user_prompt = f"Source excerpts:\n{excerpts}\n\nClaims:\n{listing}"
        try:
            # temperature=0 -- a judgement that feeds a pass/fail check.
            # The judge is a reasoning model, and its reasoning tokens come
            # out of the same completion budget as the JSON: at Groq's
            # default it can think through the whole budget and return
            # truncated JSON (claims left unjudged) or none at all (400
            # json_validate_failed). Low effort + an explicit budget keeps
            # room for the answer.
            parsed = json.loads(_call_groq(
                GROQ_JUDGE_MODEL, system_prompt, user_prompt, json_mode=True, temperature=0,
                reasoning_effort="low", max_completion_tokens=ENTAILMENT_MAX_COMPLETION_TOKENS,
            ))
        except (GroqJSONGenerationFailed, json.JSONDecodeError):
            # Still too much to answer in one go: judge each half
            # separately. A single claim that fails stays "unjudged".
            if len(batch) > 1:
                half = len(batch) // 2
                pending[:0] = [batch[:half], batch[half:]]
            continue
        for entry in parsed.get("verdicts", []):
            if not isinstance(entry, dict):
                continue
            try:
                n = int(entry.get("claim")) - 1
            except (TypeError, ValueError):
                continue
            verdict = str(entry.get("verdict", "")).strip().lower()
            if 0 <= n < len(batch) and verdict in _VERDICTS:
                results[batch[n]].update(verdict=verdict, reason=str(entry.get("reason", "") or ""))
    return results



def revise_claims(flagged: list[dict], chunks: list[dict],
                  max_tokens: int = ENTAILMENT_MAX_SOURCE_TOKENS) -> list[dict | None]:
    """
    Targeted revision: rewrites ONLY the sentences the entailment check
    flagged, each against the excerpts it cites, instead of regenerating
    the whole report. flagged: [{"claim", "sources", "reason"}].

    Returns one entry per flagged sentence, in order: {"action": "rewrite",
    "text": new sentence} (keeping as much as its excerpts support),
    {"action": "remove"} when they support none of it, or None if the model
    skipped it. Raises on request failures -- the pipeline then keeps the
    report as it was.
    """
    source_ids = sorted({i for item in flagged for i in item["sources"]})
    texts = _excerpts_within([chunks[i - 1] for i in source_ids], max_tokens)
    excerpts = "\n\n".join(f"[{i}] {text}" for i, text in zip(source_ids, texts))
    listing = "\n".join(
        f"{n + 1}. (cites {', '.join(f'[{i}]' for i in item['sources'])}) {item['claim']}\n"
        f"   Problem: {item.get('reason') or 'not supported by the cited excerpts'}"
        for n, item in enumerate(flagged)
    )
    system_prompt = (
        "You fix sentences in a research report that a fact-checker flagged as not "
        "supported by the source excerpts they cite. Rewrite each flagged sentence so "
        "it states ONLY what its cited excerpts actually say: keep as much of the "
        "original wording and meaning as the excerpts support, drop or correct the "
        "rest, keep it a single sentence, and do not add citation markers. If the "
        "cited excerpts support nothing in the sentence, remove it instead.\n"
        'Respond ONLY with JSON: {"revisions": [{"sentence": 1, "action": "rewrite", '
        '"text": "..."}, {"sentence": 2, "action": "remove"}, ...]} with one entry per '
        "sentence, no extra text."
    )
    user_prompt = f"Source excerpts:\n{excerpts}\n\nFlagged sentences:\n{listing}"
    # temperature=0 and low reasoning effort, for the same reasons as the
    # judge: a reasoning model must leave room in its budget for the answer.
    parsed = json.loads(_call_groq(
        GROQ_GENERATION_MODEL, system_prompt, user_prompt, json_mode=True, temperature=0,
        reasoning_effort="low", max_completion_tokens=ENTAILMENT_MAX_COMPLETION_TOKENS,
    ))
    results: list[dict | None] = [None] * len(flagged)
    for entry in parsed.get("revisions", []):
        if not isinstance(entry, dict):
            continue
        try:
            n = int(entry.get("sentence")) - 1
        except (TypeError, ValueError):
            continue
        action = str(entry.get("action", "")).strip().lower()
        if not 0 <= n < len(flagged):
            continue
        if action == "remove":
            results[n] = {"action": "remove"}
        elif action == "rewrite" and isinstance(entry.get("text"), str) and entry["text"].strip():
            results[n] = {"action": "rewrite", "text": entry["text"].strip()}
    return results

"""
Thin wrapper around Groq's chat completions endpoint (OpenAI-compatible,
so plain `requests` is enough -- no extra SDK dependency).

Three jobs:
  filter_relevant_pages() -- Day 2: score discovered pages, decide which
                              are worth fully scraping.
  expand_query()           -- rewrites a short/generic instruction into a
                              few more specific search queries, so retrieval
                              isn't stuck matching against one narrow phrasing.
  generate_report()       -- Day 3: turn retrieved chunks into a grounded,
                              citation-aware report.
  describe_figure()       -- turn a rendered PDF figure into a text
                              description that can be chunked like any text.

Both raise GroqNotConfigured with a clear message if no API key is set,
rather than failing with a confusing HTTP error -- the pipeline can catch
this and tell the user exactly what's missing.
"""

import base64
import json
import re
import threading
import time
import requests

from . import chunker, progress

from .config import (
    GROQ_API_KEY, GROQ_FILTER_MODEL, GROQ_GENERATION_MODEL, GROQ_MAX_CONTEXT_TOKENS,
    GROQ_VISION_MODEL, QUERY_EXPANSION_VARIANTS,
)

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2  # doubles each retry: 2s, 4s, 8s
MAX_RATE_LIMIT_WAITS = 6       # 429 retries per request, separate from MAX_RETRIES
MAX_RATE_LIMIT_WAIT_SECONDS = 60


class GroqNotConfigured(Exception):
    pass


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
# The 429 retry in _post_chat stays as the fallback: the first request(s)
# to a model go unpaced (nothing is known yet), and other processes using
# the same API key spend the same budget without this one seeing it.

_budget_lock = threading.Lock()
_token_budget: dict[str, dict] = {}
# model -> {
#   header budget:  "limit", "remaining", "full_in" (s), "at" (monotonic),
#                   "header_cost" (tokens one request took from it, measured)
#   learned limit:  "window_limit", "window_cost" (tokens Groq charges one
#                   request), "window_used" + "window_used_at" (bucket level
#                   and when it was that), "unpriced" [send times before the
#                   cost was known]
# }

_WINDOW_SECONDS = 60
_LIMIT_IN_429 = re.compile(r"Limit (\d+), Used (\d+), Requested (\d+)")

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_DURATION_SECONDS = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}


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


def _header_budget_wait(state: dict, now: float) -> float:
    cost = state.get("header_cost", 0)
    if not cost or "limit" not in state or state["full_in"] <= 0:
        return 0.0
    available = _available_tokens(state, now)
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


def _wait_for_token_budget(model: str) -> float | None:
    """
    Sleeps until the model's known limits can take one more request like
    the previous ones, then returns how many header-budget tokens are
    available as it is sent (None when nothing is known about the model).
    """
    with _budget_lock:
        state = dict(_token_budget[model]) if model in _token_budget else None
    if state is None:
        return None

    now = time.monotonic()
    wait = min(max(_header_budget_wait(state, now), _learned_limit_wait(state, now)),
               MAX_RATE_LIMIT_WAIT_SECONDS)
    if wait > 0.5:
        progress.detail(f"Waiting {wait:.0f}s for Groq's per-minute token limit")
    if wait > 0:
        time.sleep(wait)
    return _available_tokens(state, time.monotonic()) if "limit" in state else None


def _record_token_budget(model: str, response, available_at_send: float | None,
                         sent_at: float) -> None:
    with _budget_lock:
        state = _token_budget.setdefault(model, {})

        if response.status_code == 429:
            # Groq naming the limit this request broke, how full it is, and
            # how big it judged the request to be.
            match = _LIMIT_IN_429.search(response.text or "")
            if match:
                limit, used, requested = (int(g) for g in match.groups())
                now = time.monotonic()
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
                state["window_used"], state["window_used_at"] = used, now
                state.pop("unpriced", None)
        elif response.status_code == 200:
            if state.get("window_cost"):
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
        state.update(limit=limit, remaining=remaining, at=time.monotonic(),
                     full_in=_parse_duration(headers.get("x-ratelimit-reset-tokens")) or 0.0)


def _retry_after_seconds(response) -> float:
    """How long a 429 says to wait: the Retry-After header (whole seconds,
    rounded up by Groq), or 5s if it's missing."""
    try:
        return max(float(response.headers.get("retry-after", "")), 1.0)
    except ValueError:
        return 5.0


def _call_groq(model: str, system_prompt: str, user_prompt: str, json_mode: bool = True,
                temperature: float = 0.2) -> str:
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
    return _post_chat(payload)


def _post_chat(payload: dict, timeout: int = 30) -> str:
    """
    Sends one chat-completions payload and returns the reply text, with
    the retry/error handling every Groq call shares. Split out of
    _call_groq so describe_figure can send multi-part (text + image)
    messages through the same path.
    """
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
            available_at_send = _wait_for_token_budget(model)
            sent_at = time.monotonic()
            response = requests.post(
                GROQ_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            _record_token_budget(model, response, available_at_send, sent_at)
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
            response.raise_for_status()
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


def generate_report(chunks: list[dict], instructions: str) -> dict:
    """
    chunks: list of {"text": ..., "source_label": ..., "source_type": ...}
            (source_label is the human-readable citation string, e.g.
            "paper.pdf, page 3" or "Page Title, https://...")

    Returns {"report_text": str, "citations_used": [source_label, ...]}
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
        "from. Do not invent facts not present in the sources. Respond ONLY "
        'with JSON: {"report_text": "...", "citations_used": [1, 2, ...]} '
        "where citations_used lists every source index you actually cited."
    )
    user_prompt = f"Instructions: {instructions}\n\nSource excerpts:\n{numbered_context}"

    raw = _call_groq(GROQ_GENERATION_MODEL, system_prompt, user_prompt, json_mode=True)
    parsed = json.loads(raw)

    cited_indices = parsed.get("citations_used", [])
    citations_used = [
        chunks[i - 1]["source_label"] for i in cited_indices
        if 1 <= i <= len(chunks)
    ]

    return {
        "report_text": parsed.get("report_text", ""),
        "citations_used": citations_used,
    }

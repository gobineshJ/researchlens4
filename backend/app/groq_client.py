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

Both raise GroqNotConfigured with a clear message if no API key is set,
rather than failing with a confusing HTTP error -- the pipeline can catch
this and tell the user exactly what's missing.
"""

import json
import time
import requests

from . import chunker

from .config import (
    GROQ_API_KEY, GROQ_FILTER_MODEL, GROQ_GENERATION_MODEL, GROQ_MAX_CONTEXT_TOKENS,
    QUERY_EXPANSION_VARIANTS,
)

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2  # doubles each retry: 2s, 4s, 8s


class GroqNotConfigured(Exception):
    pass


def _call_groq(model: str, system_prompt: str, user_prompt: str, json_mode: bool = True,
                temperature: float = 0.2) -> str:
    if not GROQ_API_KEY:
        raise GroqNotConfigured(
            "GROQ_API_KEY is not set. Get a free key at console.groq.com/keys "
            "and add it to backend/.env"
        )

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

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                GROQ_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )
            if response.status_code == 413:
                # Groq's own message says which limit was hit (e.g. "Request
                # too large ... on tokens per minute (TPM): Limit 8000,
                # Requested 11234") -- surface it instead of a bare HTTPError.
                raise RuntimeError(
                    f"Groq rejected the request as too large for model '{model}'. "
                    f"Lower GROQ_MAX_CONTEXT_TOKENS in backend/.env or use fewer/smaller "
                    f"sources. Groq said: {response.text}"
                )
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

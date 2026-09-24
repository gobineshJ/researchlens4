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

from .config import GROQ_API_KEY, GROQ_FILTER_MODEL, GROQ_GENERATION_MODEL, QUERY_EXPANSION_VARIANTS

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2  # doubles each retry: 2s, 4s, 8s


class GroqNotConfigured(Exception):
    pass


def _call_groq(model: str, system_prompt: str, user_prompt: str, json_mode: bool = True) -> str:
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
        "temperature": 0.2,
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

    raw = _call_groq(GROQ_FILTER_MODEL, system_prompt, user_prompt, json_mode=True)
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


def expand_query(instructions: str, num_variants: int = QUERY_EXPANSION_VARIANTS) -> list[str]:
    """
    Rewrites the user's instructions into a few more specific search
    queries. A short generic instruction ("focus on research gaps")
    embeds poorly against long, specific chunks -- retrieving separately
    for a handful of more concrete rewordings ("limitations of current
    methods", "authors' suggested future work") catches relevant chunks
    the original phrasing alone would miss.

    Always returns the ORIGINAL instructions as the first element, even
    if the Groq call fails -- so a failure here degrades gracefully to
    the old single-query behavior instead of breaking retrieval entirely.
    """
    queries = [instructions]
    if not GROQ_API_KEY:
        return queries  # can't expand without a key; fall back silently

    system_prompt = (
        f"You rewrite a research report instruction into {num_variants} alternative, "
        "more specific search queries that would each surface different relevant "
        "passages from a document. Keep each query short (under 15 words) and "
        "focused on one concrete angle. Respond ONLY with JSON: "
        '{"queries": ["...", "...", "..."]} with exactly '
        f"{num_variants} entries, no extra text."
    )
    try:
        raw = _call_groq(GROQ_FILTER_MODEL, system_prompt, instructions, json_mode=True)
        parsed = json.loads(raw)
        variants = [q.strip() for q in parsed.get("queries", []) if isinstance(q, str) and q.strip()]
        queries.extend(variants[:num_variants])
    except Exception:
        # Any failure here (network, bad JSON, rate limit) -- just use the
        # original instructions alone rather than failing the whole pipeline
        # over a "nice to have" retrieval improvement.
        pass
    return queries


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

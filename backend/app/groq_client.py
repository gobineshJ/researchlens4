"""
Thin wrapper around Groq's chat completions endpoint (OpenAI-compatible,
so plain `requests` is enough -- no extra SDK dependency).

Two jobs:
  filter_relevant_pages() -- Day 2: score discovered pages, decide which
                              are worth fully scraping.
  generate_report()       -- Day 3: turn retrieved chunks into a grounded,
                              citation-aware report.

Both raise GroqNotConfigured with a clear message if no API key is set,
rather than failing with a confusing HTTP error -- the pipeline can catch
this and tell the user exactly what's missing.
"""

import json
import requests

from .config import GROQ_API_KEY, GROQ_FILTER_MODEL, GROQ_GENERATION_MODEL

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


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

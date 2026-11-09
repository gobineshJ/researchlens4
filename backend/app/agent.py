"""
Agentic retrieval: an LLM decides, step by step, which tool to call to
gather evidence for a question, and when it has enough.

    search_documents(query, source?)   hybrid BM25 + vector search, reranked
    read_passage(passage_id)           the full text of one passage
    graph_lookup(entity)               what the knowledge graph links to an entity
    graph_path(from_entity, to_entity) the chain of relations linking two entities
    finish(passage_ids, reason)        stop; these passages are the evidence

Only RETRIEVAL is agentic. The caller (pipeline.py) still writes the report
from the chosen passages and always runs the claim check and revision on
it -- an agent must never be able to skip verification.

Passages have stable ids "<document>.<position>" (e.g. "3.1" = document 3,
second chunk). Everything a tool shows the model is recorded in a ledger,
so the final evidence is always real, citable chunks -- whatever the model
writes in finish() is checked against the ledger, never trusted blindly.

Limits, so a confused model can't burn the free tier's quota:
AGENT_MAX_STEPS model calls, AGENT_MAX_TOKENS total tokens,
AGENT_MAX_SECONDS wall time; repeated identical calls return the earlier
result instead of running again; older tool results are condensed so the
context sent each step stays small.
"""

import json
import time
from dataclasses import dataclass, field

from . import groq_client, hybrid_search, progress, reranker, vectorstore
from .config import (
    AGENT_MAX_OBSERVATION_CHARS, AGENT_MAX_SECONDS, AGENT_MAX_STEPS, AGENT_MAX_TOKENS,
    AGENT_MODEL, AGENT_REASONING_EFFORT, AGENT_SEARCH_RESULTS,
)

PREVIEW_CHARS = 320
READ_CHARS = 2800
KEEP_FULL_RESULTS = 3      # most recent tool results kept verbatim; older ones condensed


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


SEARCH_TOOLS = [
    # No document filter: small models lock onto one document with it and
    # reword the same search there (seen in live traces). To target a
    # document, put its distinctive words in the query.
    _tool("search_documents",
          "Search ALL uploaded documents (keyword + semantic search, reranked). Returns the best "
          "passages with ids and short previews.",
          {"query": {"type": "string", "description": "What to look for; use specific names and terms."}},
          ["query"]),
    _tool("read_passage",
          "Read the full text of one passage by id (e.g. '3.1'), when a preview isn't enough.",
          {"passage_id": {"type": "string"}}, ["passage_id"]),
]
GRAPH_TOOLS = [
    _tool("graph_lookup",
          "Knowledge graph: list the entities the documents connect to an entity (a product, "
          "dataset, organisation, place or person), each with the sentence and passage id that "
          "connects them. Good for finding the NAME of the next thing to look up.",
          {"entity": {"type": "string"}}, ["entity"]),
    _tool("graph_path",
          "Knowledge graph: find the chain of relations linking two entities across the documents, "
          "with the passage id for every link. Good for questions that connect two named things.",
          {"from_entity": {"type": "string"}, "to_entity": {"type": "string"}},
          ["from_entity", "to_entity"]),
]
FINISH_TOOL = _tool(
    "finish",
    "Stop searching. List the ids of the passages that contain the evidence the answer needs "
    "(most important first). Call this as soon as you have enough.",
    {"passage_ids": {"type": "array", "items": {"type": "string"}},
     "reason": {"type": "string", "description": "One sentence: why this evidence suffices."}},
    ["passage_ids"],
)

SYSTEM_PROMPT = (
    "You are the retrieval step of a research assistant. Your only job is to find the passages "
    "in the user's documents that answer the question; another step writes the answer from "
    "them. Work step by step with the tools:\n"
    "- Start with search_documents using specific terms from the question.\n"
    "- Questions often need a chain of 2-4 facts from different documents, and the question may "
    "not name the things in between. When a passage names something the question depends on (a "
    "product, supplier, test site, author or place), your NEXT step should look up that name -- "
    "rewording the same search will not find a fact that lives in another document.\n"
    "{graph_hint}"
    # Examples here must never come from the eval sets -- that would hand
    # the agent hints about the very questions it is scored on.
    "- Watch for look-alike names (e.g. 'Model-X2' vs 'Model-X', 'Springfield' vs 'Springvale'): "
    "check that a passage is about exactly the thing asked.\n"
    "- Call finish with the passage ids that contain the needed facts, most important first, "
    "usually 2-6 of them. If the documents don't contain the answer, finish with the passages "
    "closest to it and say so in the reason.\n"
    "You have at most {max_steps} steps. Be economical: don't repeat a search."
)
GRAPH_HINT = ("- When the question describes something without naming it ('the drug used in the "
              "Riverside trial', 'the lab where ... was measured'), FIRST call graph_lookup on the "
              "thing it does name, to see what the documents connect it to; then follow those names. "
              "graph_path finds the chain between two named things. Read the passages they cite.\n")


@dataclass
class AgentResult:
    passages: list[dict]                       # chunk dicts, evidence order
    trace: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    steps: int = 0
    tokens: int = 0


class _Session:
    """The documents as the agent's tools see them."""

    def __init__(self, chunks: list[dict], collection, graph=None):
        self.collection = collection
        self.graph = graph
        self.source_names = sorted({c["source_name"] for c in chunks})
        self.by_cid: dict[str, dict] = {}
        self.cid_of: dict[tuple, str] = {}
        for c in chunks:
            cid = f"{self.source_names.index(c['source_name']) + 1}.{c.get('position', 0)}"
            self.by_cid[cid] = c
            self.cid_of[(c["source_name"], c.get("position", 0))] = cid
        self.chunks = chunks
        self.bm25_index, self.bm25_chunks = hybrid_search.build_index(chunks)
        # passage id -> the graph's entities in it, best-connected first.
        # Shown with search results so the agent sees the NAMES a passage
        # gives it to follow (the bridge in a multi-document chain), not just
        # the words that matched its query.
        self.mentions: dict[str, list[str]] = {}
        if graph is not None:
            for key, cids in graph.passages.items():
                for cid in cids:
                    self.mentions.setdefault(cid, []).append(key)
            for cid, keys in self.mentions.items():
                keys.sort(key=lambda k: -len(graph.adjacent[k]))
                self.mentions[cid] = [graph.names[k] for k in keys[:8]]
        self.ledger: list[str] = []   # every passage id shown to the model, first-seen order
        self.read: set[str] = set()
        self.shown: list[str] = []    # ids shown by the CURRENT tool call
        self.searches: list[tuple[set[str], tuple[str, ...]]] = []  # (query words, sources)

    def _see(self, cid: str) -> None:
        if cid not in self.by_cid:
            return
        if cid not in self.ledger:
            self.ledger.append(cid)
        if cid not in self.shown:
            self.shown.append(cid)

    def _cid(self, chunk: dict) -> str | None:
        return self.cid_of.get((chunk.get("source_name"), chunk.get("position", 0)))

    def source_list(self) -> str:
        lines = []
        for n, name in enumerate(self.source_names, 1):
            ids = sorted((cid for cid in self.by_cid if cid.split(".")[0] == str(n)),
                         key=lambda c: int(c.split(".")[1]))
            first = next(c for c in self.chunks if c["source_name"] == name)
            lines.append(f"{n}. {name} -- passages {ids[0]}..{ids[-1]} -- begins: "
                         f"{' '.join(first['text'].split()[:14])}...")
        return "\n".join(lines)

    # ---------- tools ----------

    def search_documents(self, query: str, source: str | None = None) -> str:
        names = self.source_names
        if source:
            matches = [n for n in names if source.lower() in n.lower()]
            if not matches:
                return f"No document named {source!r}. Documents: {', '.join(names)}"
            names = matches
        words = _query_words(query)
        for earlier_words, earlier_sources in self.searches:
            if earlier_sources == tuple(names) and _similar(words, earlier_words):
                return ("This search is nearly the same as one you already ran, so the results would "
                        "be the same. Change approach: search for a NAME the "
                        "passages you found mention, read one of those passages in full, or use the "
                        "graph tools if available.")
        self.searches.append((words, tuple(names)))
        already_seen = set(self.ledger)
        pool = 4 * AGENT_SEARCH_RESULTS
        # One BM25 index over ALL documents, filtered afterwards: BM25's
        # term weighting needs the whole corpus -- built per document, a
        # one-chunk "corpus" gives every term a negative weight.
        keyword = [c for c in hybrid_search.search(self.bm25_index, self.bm25_chunks, query, top_k=pool * 3)
                   if c["source_name"] in names][:pool]
        vector = []
        for name in names:
            vector.extend(vectorstore.retrieve_candidates(self.collection, [query], k_per_query=pool, source_name=name))
        vector.sort(key=lambda c: c.get("distance", 1.0))
        candidates = hybrid_search.fuse_rrf(vector[:pool * 2], keyword)
        unique = {}
        for c in candidates:
            cid = self._cid(c)
            if cid and cid not in unique:
                unique[cid] = {**self.by_cid[cid], "rrf_score": c.get("rrf_score", 0)}
        ranked = reranker.rerank_with_fallback(query, list(unique.values()), top_k=AGENT_SEARCH_RESULTS)
        if not ranked:
            return "No passages found."
        lines = []
        for c in ranked:
            cid = self._cid(c)
            self._see(cid)
            mentions = self.mentions.get(cid)
            lines.append(f"[{cid}] ({c['source_name']}) {_preview(c['text'], query)}"
                         + (f"\n    Mentions: {', '.join(mentions)}" if mentions else ""))
        if all(cid in already_seen for cid in self.shown):
            lines.append("(No new passages -- you have seen all of these. If they mention a name the "
                         "question depends on, look that name up next.)")
        return "\n".join(lines)

    def read_passage(self, passage_id: str) -> str:
        cid = str(passage_id).strip().strip("[]")
        chunk = self.by_cid.get(cid)
        if chunk is None:
            return f"No passage {passage_id!r}. Ids look like '3.1'."
        self._see(cid)
        self.read.add(cid)
        text = chunk["text"]
        return f"[{cid}] ({chunk['source_name']})\n{text[:READ_CHARS]}" + (" ..." if len(text) > READ_CHARS else "")

    def graph_lookup(self, entity: str) -> str:
        result = self.graph.lookup(entity)
        if not result["found"]:
            return f"No entity matching {entity!r} in the knowledge graph."
        lines = [f"{result['entity']} appears in passages {', '.join(result['passages'][:8])}."]
        for rel in result["relations"]:
            self._see(rel["evidence"]["cid"])
            lines.append(f"- {rel['to']} ({rel['strength']}, passage {rel['evidence']['cid']}): "
                         f"{rel['evidence']['text'][:160]}")
        if result["alternatives"]:
            lines.append(f"Similar names: {', '.join(result['alternatives'])}")
        return "\n".join(lines)

    def graph_path(self, from_entity: str, to_entity: str) -> str:
        result = self.graph.path(from_entity, to_entity)
        if not result["found"]:
            return result["reason"]
        lines = [f"{result['from']} -> {result['to']} in {len(result['steps'])} link(s):"]
        for step in result["steps"]:
            self._see(step["evidence"]["cid"])
            lines.append(f"- {step['from']} -> {step['to']} (passage {step['evidence']['cid']}): "
                         f"{step['evidence']['text'][:160]}")
        return "\n".join(lines)

    def evidence(self, chosen: list[str]) -> list[dict]:
        """
        The passages to write from. If finish() named passages the agent
        was actually shown, exactly those, in its order -- adding the rest
        of what it merely SAW would hand the writer the look-alikes it
        deliberately rejected. Without a usable finish (a limit hit first),
        what it read, then what it saw. Ids it was never shown are ignored.
        """
        picked = [cid for cid in dict.fromkeys(str(c).strip().strip("[]") for c in chosen)
                  if cid in self.by_cid and cid in self.ledger]
        if not picked:
            picked = list(dict.fromkeys([c for c in self.ledger if c in self.read] + self.ledger))
        return [self.by_cid[cid] for cid in picked]


def _query_words(query: str) -> set[str]:
    return {w for w in (x.lower().strip(".,;:()\"'?") for x in query.split()) if len(w) > 2}


def _similar(a: set[str], b: set[str]) -> bool:
    """Same search in different words: Jaccard overlap of at least 0.7.
    ("Acme Lab director" vs "Acme Lab Springfield" share 2 of 4
    words -- a genuinely different search.)"""
    if not a or not b:
        return a == b
    return len(a & b) / len(a | b) >= 0.7


def _preview(text: str, query: str) -> str:
    """The sentence-ish window of `text` sharing the most words with the
    query, so the preview shows WHY the passage matched."""
    words = text.split()
    if len(" ".join(words)) <= PREVIEW_CHARS:
        return " ".join(words)
    terms = {w.lower().strip(".,;:()") for w in query.split() if len(w) > 2}
    window = 45
    best, best_score = 0, -1
    for start in range(0, max(1, len(words) - window), 10):
        score = sum(w.lower().strip(".,;:()") in terms for w in words[start:start + window])
        if score > best_score:
            best, best_score = start, score
    snippet = " ".join(words[best:best + window])
    return ("..." if best else "") + snippet[:PREVIEW_CHARS] + "..."


def _condense(messages: list[dict], ids_by_message: dict[int, list[str]]) -> list[dict]:
    """Shrinks all but the last KEEP_FULL_RESULTS tool results to just the
    passage ids they showed -- the model already saw the full text, and
    resending it every step is what makes agent loops expensive."""
    tool_positions = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    old = set(tool_positions[:-KEEP_FULL_RESULTS])
    out = []
    for i, m in enumerate(messages):
        if i in old:
            ids = ids_by_message.get(i, [])
            m = {**m, "content": f"(earlier result, condensed) passages shown: {', '.join(ids) or 'none'}"}
        out.append(m)
    return out


def _sort_key(cid: str) -> tuple[int, ...]:
    return tuple(int(part) for part in cid.split("."))


def _summarise(name: str, result: str, ids: list[str]) -> str:
    if name == "finish":
        return result
    if ids:
        return f"{len(ids)} passage(s): {', '.join(ids[:6])}{'...' if len(ids) > 6 else ''}"
    return result.splitlines()[0][:100] if result else "no result"


def run_agent(question: str, chunks: list[dict], collection, graph=None,
              call_model=None) -> AgentResult:
    """
    Runs the tool-calling loop for one question. `call_model(payload) ->
    response body` defaults to Groq; tests pass a scripted fake.
    Raises only if the very first model call fails (the caller then falls
    back to ordinary retrieval); later failures end the loop with the
    evidence gathered so far.
    """
    call_model = call_model or (lambda payload: groq_client._post_chat(payload, timeout=90, full=True))
    session = _Session(chunks, collection, graph)
    tools = SEARCH_TOOLS + (GRAPH_TOOLS if graph is not None else []) + [FINISH_TOOL]
    # A stray "source" argument (from habit) is ignored rather than an error.
    runners = {"search_documents": lambda query, **_: session.search_documents(query), "read_passage": session.read_passage,
               "graph_lookup": session.graph_lookup if graph is not None else None,
               "graph_path": session.graph_path if graph is not None else None}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(graph_hint=GRAPH_HINT if graph is not None else "",
                                                           max_steps=AGENT_MAX_STEPS)},
        {"role": "user", "content": f"Question: {question}\n\nDocuments:\n{session.source_list()}"},
    ]
    result = AgentResult(passages=[])
    seen_calls: dict[str, tuple[str, list[str]]] = {}
    ids_by_message: dict[int, list[str]] = {}  # tool message index -> passage ids it showed
    chosen: list[str] = []
    started = time.monotonic()
    invalid_calls = 0

    while True:
        if result.steps >= AGENT_MAX_STEPS:
            result.stop_reason = f"step limit ({AGENT_MAX_STEPS})"
            break
        if result.tokens >= AGENT_MAX_TOKENS:
            result.stop_reason = f"token limit ({result.tokens:,} >= {AGENT_MAX_TOKENS:,})"
            break
        if time.monotonic() - started > AGENT_MAX_SECONDS:
            result.stop_reason = f"time limit ({AGENT_MAX_SECONDS}s)"
            break
        payload = {"model": AGENT_MODEL, "messages": _condense(messages, ids_by_message), "tools": tools,
                   "tool_choice": "auto", "temperature": 0, "max_completion_tokens": 1500}
        if AGENT_REASONING_EFFORT:
            payload["reasoning_effort"] = AGENT_REASONING_EFFORT
        try:
            body = call_model(payload)
        except Exception as exc:
            # Groq checks tool-call arguments against the schema and rejects
            # the whole reply on a mismatch (seen live with Qwen). That's the
            # model's mistake, not an outage: say what was wrong and let it
            # try again, a couple of times per question.
            if "tool call validation failed" in str(exc) and invalid_calls < 2:
                invalid_calls += 1
                result.steps += 1
                messages.append({"role": "user", "content": (
                    "Your last tool call was rejected because its arguments did not match the tool's "
                    "parameters. Use only the parameters each tool defines, with the right types.")})
                continue
            if result.steps == 0:
                raise
            result.stop_reason = f"model call failed: {str(exc)[:160]}"
            break
        result.steps += 1
        result.tokens += (body.get("usage") or {}).get("total_tokens", 0)
        message = body["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if not calls:
            result.stop_reason = "model answered without calling finish"
            break
        messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": calls})

        finished = False
        for call in calls[:3]:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            key = f"{name}:{json.dumps(args, sort_keys=True)}"
            session.shown = []
            if name == "finish":
                chosen = [str(x) for x in args.get("passage_ids", []) if x]
                output = f"Finished with {len(chosen)} passage(s)."
                finished = True
                result.stop_reason = "finish: " + str(args.get("reason", ""))[:200]
            elif key in seen_calls:
                output = "You already made this exact call. Its result was:\n" + seen_calls[key][0]
                session.shown = list(seen_calls[key][1])
            elif runners.get(name) is None:
                output = f"Unknown tool {name!r}."
            else:
                try:
                    output = runners[name](**{k: v for k, v in args.items() if isinstance(v, str)})
                except TypeError as exc:
                    output = f"Bad arguments for {name}: {exc}"
                except Exception as exc:
                    output = f"{name} failed: {str(exc)[:200]}"
                seen_calls[key] = (output, list(session.shown))
            output = output[:AGENT_MAX_OBSERVATION_CHARS]
            if name != "finish":
                # Without this, models run straight into the step limit:
                # in live traces neither gpt-oss model ever chose to stop.
                left = AGENT_MAX_STEPS - result.steps
                output += (f"\n[{left} step(s) left.]" if left > 1 else
                           "\n[Last step: call finish now with the passages you have.]")
            ids_by_message[len(messages)] = list(session.shown)
            messages.append({"role": "tool", "tool_call_id": call.get("id") or f"{name}-{result.steps}",
                             "content": output})
            step = {"step": result.steps, "tool": name, "args": args,
                    "result": _summarise(name, output, session.shown), "tokens": result.tokens}
            result.trace.append(step)
            progress.agent_step(step)
            progress.detail(f"Step {result.steps}: {name}({', '.join(str(v)[:40] for v in args.values() if not isinstance(v, list))}) -> {step['result']}")
        # Every tool call in the assistant message needs an answer, or the
        # NEXT request is rejected (400) -- including the forced finish.
        for call in calls[3:]:
            messages.append({"role": "tool", "tool_call_id": call.get("id") or "extra",
                             "content": "Skipped: at most 3 tool calls per step."})
        if finished:
            break

    if not chosen and session.ledger and not result.stop_reason.startswith("finish"):
        chosen = _forced_finish(call_model, messages, ids_by_message, result, session)
    result.passages = session.evidence(chosen)
    return result


def _forced_finish(call_model, messages, ids_by_message, result: AgentResult, session) -> list[str]:
    """A limit hit before the agent called finish: one last call that MUST
    be finish, so the agent -- not a blind fallback -- picks its evidence.
    The old fallback handed the writer everything the agent had seen,
    look-alikes included, which is how a correct search turned into a wrong
    answer in live runs. If even this call fails, the fallback still runs."""
    messages = messages + [{"role": "user", "content": (
        f"You have run out of steps ({result.stop_reason}). Call finish now with the passage ids "
        "that contain the facts the question needs, most important first.")}]
    payload = {"model": AGENT_MODEL, "messages": _condense(messages, ids_by_message), "tools": [FINISH_TOOL],
               "tool_choice": {"type": "function", "function": {"name": "finish"}},
               "temperature": 0, "max_completion_tokens": 800}
    try:
        body = call_model(payload)
        call = (body["choices"][0]["message"].get("tool_calls") or [None])[0]
        args = json.loads(call["function"].get("arguments") or "{}") if call else {}
    except Exception as exc:
        result.stop_reason += f"; forced finish failed: {str(exc)[:160]}"
        return []
    result.tokens += (body.get("usage") or {}).get("total_tokens", 0)
    chosen = [str(x) for x in args.get("passage_ids", []) if x]
    step = {"step": result.steps + 1, "tool": "finish", "args": args,
            "result": f"Finished with {len(chosen)} passage(s) (forced: {result.stop_reason}).",
            "tokens": result.tokens}
    result.trace.append(step)
    progress.agent_step(step)
    return chosen

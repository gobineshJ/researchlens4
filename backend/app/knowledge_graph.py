"""
A knowledge graph of the uploaded documents, for the retrieval agent's
graph_lookup and graph_path tools.

Built from the chunks with rules, not an LLM, so it costs no Groq quota
and the same documents always give the same graph:

  entities  product/standard codes ("VX-9", "EN 6240"), CamelCase names
            ("KestrelCell"), multi-word proper names ("Varn Materials",
            "Halvik Institute of Electrochemistry") and single proper
            nouns found mid-sentence ("Norrby"). A surname seen alone
            ("Kade") is merged into the full name seen elsewhere
            ("Ilse Kade").
  edges     entities in the same SENTENCE (strong, cost 1), labelled with
            the words between them ("used ... supplied by"), and entities
            in the same PASSAGE (weak, cost 2) -- the only way to link,
            say, a paper's title to a fact in its methods section.

Every edge keeps the passage id(s) it came from, so whatever the agent
finds through the graph is still a citable passage.
"""

import heapq
import re
from collections import defaultdict

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])|\n+")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9]|[A-Za-z0-9]")
_STANDARD = re.compile(r"\b[A-Z]{2,5} \d{3,5}\b")          # "EN 6240"
_PROPER = re.compile(
    r"\b(?:(?:Dr|Prof|Mr|Ms|Mrs)\.\s)?[A-Z][a-zA-Z'\-]+(?:\s(?:of|for|and|de)\s[A-Z][a-zA-Z'\-]+|\s[A-Z][a-zA-Z'\-]+)*"
)

# Capitalised words that are not entities: sentence openers, section
# words, months, units.
_STOP = {
    "the", "this", "that", "these", "those", "a", "an", "in", "on", "at", "for", "of", "to", "by", "with",
    "we", "our", "it", "its", "they", "their", "all", "each", "every", "after", "before", "during", "between",
    "most", "some", "any", "no", "not", "one", "two", "three", "four", "five", "both", "only", "since",
    "when", "while", "if", "however", "finally", "first", "second", "third", "also", "other", "over", "under",
    "abstract", "introduction", "background", "summary", "results", "methods", "method", "conclusion",
    "limitations", "discussion", "figure", "table", "section", "entry", "edition", "contact", "fees", "safety",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "celsius", "us", "is", "are", "was", "were", "has", "have", "had", "can", "could",
    "cells", "cell", "capacity", "efficiency", "availability", "costs", "cost", "lessons", "recommendation",
    "developer", "chemistry", "electrolyte", "tested", "reported", "calendar", "projected", "prepared",
    "compiled", "chief", "centre", "plans", "production", "specifications", "incidents", "customers",
    "pricing", "receiving", "data", "reporting", "who", "what", "where", "which", "how", "why",
}
_CONNECTORS = {"of", "for", "and", "de"}
_TITLES = ("dr. ", "prof. ", "mr. ", "ms. ", "mrs. ")
MAX_ENTITIES_PER_PASSAGE = 30
MAX_PHRASE_WORDS = 5          # longer capitalised runs are titles, not names
EVIDENCE_PER_EDGE = 3


def _key(name: str) -> str:
    key = name.lower().strip()
    for title in _TITLES:
        if key.startswith(title):
            key = key[len(title):]
    key = re.sub(r"^the\s+", "", key)
    key = re.sub(r"'s$", "", key)
    return " ".join(key.split())


def _is_named_token(token: str) -> bool:
    """A product, model or code name: "VX-9", "KS-3", "Merlin-Na-3",
    "Gannet-S", "KestrelCell", "SodaCore-X", "A100" -- but not an ordinary
    hyphenated word like "Room-temperature"."""
    if not token[0].isupper():
        return False
    parts = token.split("-")
    if len(parts) > 1 and any(part[:1].isupper() or part[:1].isdigit() for part in parts[1:]):
        return True
    if re.search(r"[a-z][A-Z]", token):           # CamelCase
        return True
    return len(token) >= 3 and any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token)


def _clean_proper(phrase: str, sentence_start: bool) -> str | None:
    words = phrase.split()
    # Drop leading stopwords ("The Oskarsund Battery Test Centre" -> "Oskarsund ...").
    while words and words[0].lower().rstrip(".") in _STOP:
        words = words[1:]
    while words and words[-1].lower() in _CONNECTORS:
        words = words[:-1]
    if not words or len(words) > MAX_PHRASE_WORDS:
        return None
    if len(words) == 1:
        word = words[0]
        # A lone capitalised word that opens the sentence is usually just
        # an ordinary word ("Training took..."); keep it only if it opens
        # nothing (i.e. was found mid-sentence).
        if sentence_start and phrase.split()[0] == word:
            return None
        if len(word) < 3 or word.lower() in _STOP or re.fullmatch(r"[A-Z]{2,}s", word):
            return None   # "GPUs": a plural acronym is a common noun
    return " ".join(words)


def extract_entities(sentence: str) -> list[tuple[str, int]]:
    """(entity name, character offset) for every entity in one sentence."""
    found: dict[int, str] = {}
    for m in _STANDARD.finditer(sentence):
        found[m.start()] = m.group(0)
    for m in _TOKEN.finditer(sentence):
        if m.start() not in found and _is_named_token(m.group(0)):
            found[m.start()] = m.group(0)
    covered = [(start, start + len(name)) for start, name in found.items()]
    for m in _PROPER.finditer(sentence):
        if any(s <= m.start() < e or m.start() <= s < m.end() for s, e in covered):
            continue
        # "Sentence start" = nothing but numbers/punctuation before it, so a
        # section heading like "2 Production" doesn't make "Production" a name.
        starts = not re.search(r"[A-Za-z]", sentence[:m.start()])
        name = _clean_proper(m.group(0), sentence_start=starts)
        if name:
            found.setdefault(m.start() + m.group(0).find(name.split()[0]), name)
    return sorted(((name, start) for start, name in found.items()), key=lambda x: x[1])


class KnowledgeGraph:
    def __init__(self):
        self.names: dict[str, str] = {}                      # key -> display name
        self.passages: dict[str, set[str]] = defaultdict(set)  # key -> passage ids
        # (a, b) sorted keys -> {"cost": 1|2, "weight": n, "evidence": [{"cid", "text"}]}
        self.edges: dict[tuple[str, str], dict] = {}
        self.adjacent: dict[str, set[str]] = defaultdict(set)

    # ---------- building ----------

    def _add_edge(self, a: str, b: str, cost: int, cid: str, text: str) -> None:
        if a == b:
            return
        pair = tuple(sorted((a, b)))
        edge = self.edges.setdefault(pair, {"cost": cost, "weight": 0, "evidence": []})
        edge["cost"] = min(edge["cost"], cost)
        edge["weight"] += 1
        if len(edge["evidence"]) < EVIDENCE_PER_EDGE and all(e["cid"] != cid or e["text"] != text
                                                            for e in edge["evidence"]):
            # Sentence evidence first: it says HOW the two are related.
            entry = {"cid": cid, "text": text}
            if cost == 1:
                edge["evidence"].insert(0, entry)
            else:
                edge["evidence"].append(entry)
            del edge["evidence"][EVIDENCE_PER_EDGE:]
        self.adjacent[a].add(b)
        self.adjacent[b].add(a)

    @classmethod
    def build(cls, passages: list[dict], documents: list[dict] | None = None) -> "KnowledgeGraph":
        """
        passages:  [{"cid": "3.1", "source_name": ..., "text": ...}] -- the chunks
        documents: [{"source_name": ..., "text": ...}] -- the full original
                   texts, optional. Chunk text has lost its line breaks, so
                   a title runs straight into the next line ("...Long Cycle
                   Life Ilse Kade"); sentences are read from the original
                   text when it's given, and each is mapped to the passage
                   it falls in.
        """
        graph = cls()
        raw: list[tuple[str, list[tuple[str, list[str]]]]] = []  # cid, [(sentence, names)]
        by_cid: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
        sources_with_text = set()
        for document in documents or []:
            own = [p for p in passages if p.get("source_name") == document["source_name"]]
            if not own:
                continue
            sources_with_text.add(document["source_name"])
            squashed = [(p["cid"], _squash(p["text"])) for p in own]
            for sentence in _SENTENCE.split(document["text"]):
                sentence = " ".join(sentence.split())
                if not sentence:
                    continue
                probe = _squash(sentence)[:80]
                cid = next((c for c, t in squashed if probe in t), None)
                if cid is None:  # straddles a chunk boundary: use where it starts
                    head = _squash(" ".join(sentence.split()[:6]))
                    cid = next((c for c, t in squashed if head in t), own[0]["cid"])
                by_cid[cid].append((sentence, [n for n, _ in extract_entities(sentence)]))
        for passage in passages:
            if passage.get("source_name") in sources_with_text:
                raw.append((passage["cid"], by_cid.get(passage["cid"], [])))
                continue
            sentences = []
            for sentence in _SENTENCE.split(passage["text"]):
                sentence = sentence.strip()
                if sentence:
                    sentences.append((sentence, [n for n, _ in extract_entities(sentence)]))
            raw.append((passage["cid"], sentences))

        # Aliases: a lone surname -> the one full person-like name ending in it.
        multi = {_key(n) for _, sents in raw for _, names in sents for n in names if " " in n}
        alias: dict[str, str] = {}
        for full in multi:
            last = full.split()[-1]
            if not any(w in _CONNECTORS for w in full.split()) and len(full.split()) <= 3:
                alias.setdefault(last, full) if last not in alias else alias.update({last: ""})
        alias = {k: v for k, v in alias.items() if v}

        def resolve(name: str) -> str:
            key = _key(name)
            return alias.get(key, key)

        for cid, sentences in raw:
            in_passage: dict[str, int] = defaultdict(int)
            for sentence, names in sentences:
                keys = []
                for name in names:
                    key = resolve(name)
                    if key not in graph.names or (" " in name and " " not in graph.names[key]):
                        graph.names[key] = name if key == _key(name) else graph.names.get(key, name)
                    graph.passages[key].add(cid)
                    in_passage[key] += 1
                    keys.append((key, name))
                for i, (a, name_a) in enumerate(keys):
                    for b, name_b in keys[i + 1:]:
                        graph._add_edge(a, b, 1, cid, _between(sentence, name_a, name_b))
            # Weaker same-passage links, among the passage's most-mentioned entities.
            top = sorted(in_passage, key=lambda k: -in_passage[k])[:MAX_ENTITIES_PER_PASSAGE]
            for i, a in enumerate(top):
                for b in top[i + 1:]:
                    if tuple(sorted((a, b))) not in graph.edges:
                        graph._add_edge(a, b, 2, cid, "(mentioned in the same passage)")
        for key in alias:
            graph.names.setdefault(alias[key], alias[key])
        return graph

    # ---------- querying ----------

    def match(self, query: str, limit: int = 3) -> list[str]:
        """Entity keys best matching a free-text name: exact, then alias /
        substring, then word overlap; ties go to the better-connected."""
        q = _key(query)
        if not q:
            return []
        scored = []
        q_words = set(q.split())
        q_alnum = re.sub(r"[^a-z0-9]", "", q)
        for key in self.names:
            if key == q or re.sub(r"[^a-z0-9]", "", key) == q_alnum:   # "vx 9" == "VX-9"
                score = 100
            elif q in key or key in q:
                score = 60 - abs(len(key) - len(q)) / 10
            else:
                overlap = len(q_words & set(key.split()))
                if not overlap:
                    continue
                score = 30 * overlap / max(len(q_words), len(key.split()))
            scored.append((score, len(self.adjacent[key]), key))
        scored.sort(reverse=True)
        return [key for _, _, key in scored[:limit]]

    def lookup(self, entity: str, limit: int = 8) -> dict:
        """The best-matching entity's neighbours: strong (same-sentence)
        relations first, then by how often they co-occur."""
        keys = self.match(entity)
        if not keys:
            return {"entity": entity, "found": False, "relations": [], "alternatives": []}
        key = keys[0]
        neighbours = []
        for other in self.adjacent[key]:
            edge = self.edges[tuple(sorted((key, other)))]
            neighbours.append((edge["cost"], -edge["weight"], other, edge))
        neighbours.sort()
        return {
            "entity": self.names[key], "found": True,
            "passages": sorted(self.passages[key]),
            "relations": [{"to": self.names[o], "strength": "same sentence" if e["cost"] == 1 else "same passage",
                           "evidence": e["evidence"][0]} for _, _, o, e in neighbours[:limit]],
            "alternatives": [self.names[k] for k in keys[1:]],
        }

    def path(self, source: str, target: str, max_hops: int = 5) -> dict:
        """Cheapest chain of relations linking two entities (Dijkstra:
        same-sentence edges cost 1, same-passage edges 2)."""
        a, b = (self.match(source) or [None])[0], (self.match(target) or [None])[0]
        if a is None or b is None:
            missing = source if a is None else target
            return {"found": False, "reason": f"No entity matching {missing!r} in the documents.", "steps": []}
        best = {a: 0}
        previous: dict[str, str] = {}
        queue = [(0, 0, a)]
        while queue:
            cost, hops, node = heapq.heappop(queue)
            if node == b:
                break
            if cost > best.get(node, float("inf")) or hops >= max_hops:
                continue
            for other in self.adjacent[node]:
                step = self.edges[tuple(sorted((node, other)))]["cost"]
                if cost + step < best.get(other, float("inf")):
                    best[other] = cost + step
                    previous[other] = node
                    heapq.heappush(queue, (cost + step, hops + 1, other))
        if b not in best:
            return {"found": False, "reason": f"No chain of at most {max_hops} relations links them.", "steps": []}
        chain = [b]
        while chain[-1] != a:
            chain.append(previous[chain[-1]])
        chain.reverse()
        steps = []
        for x, y in zip(chain, chain[1:]):
            edge = self.edges[tuple(sorted((x, y)))]
            steps.append({"from": self.names[x], "to": self.names[y], "evidence": edge["evidence"][0]})
        return {"found": True, "from": self.names[a], "to": self.names[b], "steps": steps}

    def stats(self) -> dict:
        strong = sum(1 for e in self.edges.values() if e["cost"] == 1)
        return {"entities": len(self.names), "edges": len(self.edges), "sentence_edges": strong}


def _squash(text: str) -> str:
    return " ".join(text.split()).lower()


def _between(sentence: str, name_a: str, name_b: str, max_words: int = 14) -> str:
    """The sentence, trimmed to the part linking the two names when that
    part is short -- the closest thing to a relation label rules can give."""
    i, j = sentence.find(name_a), sentence.find(name_b)
    if i >= 0 and j >= 0:
        lo, hi = sorted((i, j))
        end = hi + len(name_b if hi == j else name_a)
        span = sentence[lo:end]
        if len(span.split()) <= max_words + 6:
            return span.strip()
    return sentence[:240].strip()

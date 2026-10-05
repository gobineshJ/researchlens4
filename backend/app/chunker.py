"""
Splits text records (from doc_parser or scraper) into overlapping
chunks sized in tokens, per the brief's 500-800 token / 10-20% overlap
spec. Uses a simple whitespace-word approximation for "tokens" rather
than a real tokenizer -- close enough for chunk sizing, and keeps this
module dependency-free.

Every input record must have "text" plus enough fields to build a
human-readable citation label -- this module adds a "source_label" and
carries the rest through untouched.
"""

import re

from .config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_RATIO


def _label_for(record: dict) -> str:
    if record["source_type"] == "file":
        return f"{record['filename']}, {record['location']}"
    return f"{record['title']}, {record['url']}"


def _source_name_for(record: dict) -> str:
    """
    Identifies which DOCUMENT a chunk came from, as opposed to source_label
    which identifies a specific page/section within it. Needed so retrieval
    can guarantee every uploaded document gets representation, rather than
    treating all chunks from all documents as one undifferentiated pool.
    """
    if record["source_type"] == "file":
        return record["filename"]
    return record["url"]


def chunk_record(record: dict,
                  chunk_size: int = CHUNK_SIZE_WORDS,
                  overlap_ratio: float = CHUNK_OVERLAP_RATIO) -> list[dict]:
    """
    Splits one record's text into word-bounded chunks. Returns a list of:
        {"text": ..., "source_label": ..., "source_type": ..., "source_name": ...,
         "content_type": ...}

    content_type is "text" unless the record says otherwise -- doc_parser
    marks figure records, so evaluation can tell model-generated figure
    descriptions apart from text taken verbatim from a source.
    """
    words = record["text"].split()
    if not words:
        return []

    step = max(1, int(chunk_size * (1 - overlap_ratio)))
    label = _label_for(record)
    source_name = _source_name_for(record)
    content_type = record.get("content_type", "text")

    chunks = []
    start = 0
    while start < len(words):
        piece = words[start:start + chunk_size]
        chunks.append({
            "text": " ".join(piece),
            "source_label": label,
            "source_type": record["source_type"],
            "source_name": source_name,
            "content_type": content_type,
        })
        if start + chunk_size >= len(words):
            break
        start += step

    return chunks


def chunk_all(records: list[dict]) -> list[dict]:
    """
    Chunk every record in a list and flatten the result.

    Each chunk also gets a "position": its 0-based index among all chunks
    of the SAME source document, in reading order. Numbering here rather
    than in chunk_record is deliberate -- one PDF produces one record per
    page, so a per-record index would restart at 0 on every page and be
    useless for measuring how much of a document a selection spans. This
    numbering matches the index a chunk has in pipeline's
    `chunks_by_source[name]` list exactly, which is what lets
    evaluation.validate_span_coverage read positions directly instead of
    reconstructing them by matching chunk text.
    """
    all_chunks = []
    position_by_source: dict[str, int] = {}
    for record in records:
        for chunk in chunk_record(record):
            source_name = chunk["source_name"]
            chunk["position"] = position_by_source.get(source_name, 0)
            position_by_source[source_name] = chunk["position"] + 1
            all_chunks.append(chunk)
    return all_chunks


def estimate_tokens(text: str) -> int:
    """
    ~4 characters per token is the usual rule of thumb for English text --
    close enough for staying under a limit, and avoids a tokenizer
    dependency.

    Note this is a DIFFERENT unit from CHUNK_SIZE_WORDS above, which
    counts whitespace words. A 650-word chunk is roughly 1000 of these.
    """
    return len(text) // 4 + 1


def sample_representative(chunks: list[dict], k: int) -> list[dict]:
    """
    Picks k chunks EVENLY SPACED across the list, by position -- not by
    similarity to any query. This is for "summary"-intent requests: a
    generic instruction like "summarize this document" has no specific
    topic for similarity search to match against, so the top-5-most-
    similar chunks would just reflect whatever happens to embed closest
    to the word "summarize" -- not a representative sample of the whole
    document, which is what the user actually wants covered.

    Both ENDPOINTS are always included: the first chunk and the last one.
    The earlier `int(i * len(chunks) / k)` spacing topped out at about
    0.8 * len(chunks), so the end of every document -- conclusions,
    limitations, future work -- was structurally unreachable, and span
    coverage could never exceed ~80% even on a perfectly healthy run.

    `chunks` is assumed to already be in the document's original reading
    order (chunker.chunk_record produces chunks that way).

    Returns all of `chunks` unchanged if there are k or fewer of them --
    nothing to sample down.
    """
    if len(chunks) <= k:
        return list(chunks)
    if k <= 0:
        return []

    if k == 1:
        indices = [0]
    else:
        last = len(chunks) - 1
        indices = sorted({round(i * last / (k - 1)) for i in range(k)})
    if len(indices) < k:
        # Rounding collisions (two computed positions landed on the same
        # index) can leave us short of k -- top up with the next unused
        # indices so the caller reliably gets k chunks back.
        remaining = [i for i in range(len(chunks)) if i not in indices]
        indices = sorted(indices + remaining[: k - len(indices)])
    return [chunks[i] for i in indices]


# ---------- Back matter (for "summary" sampling) ----------
#
# A paper's last chunks are rarely its conclusions. On real Nature Medicine
# papers they are Extended Data figure captions; on an MDPI review, the
# reference list. Sampling a document "end to end" without knowing this
# hands the summary a page of citations and calls it full coverage. These
# patterns were tuned on three real papers (two Nature Medicine articles, one
# 52-page review): every reference, caption, affiliation and acknowledgement
# chunk was caught, and no chunk of the main text or Methods was.

_NUMBERED_REFERENCE = re.compile(r"(?:^|\s)\d{1,3}\.\s+[A-Z][\w'\-]+,?\s+[A-Z]\.")
# "12. Smith, J." / "122. Gupta, S.;" -- the start of a numbered citation.
# Reference chunks measured 2.2-4.2 per 100 words; main text at most 1.7,
# and only where the discussion runs into the start of the reference list.
_REFERENCE_DENSITY = 2.0

_AFFILIATION_WORD = re.compile(
    r"\b(?:Department|University|Hospital|Institute|School|Cent(?:er|re)|Faculty|College|"
    r"Laborator(?:y|ies))\b"
)
# Affiliation lists measured 10-12 per 100 words; the densest real content
# (a Methods section describing participating hospitals) 5.7.
_AFFILIATION_DENSITY = 8.0

_INITIALS_LIST = re.compile(r"\b(?:[A-Z]\.){2,4},")
# "P.R., E.H., G.L.," -- author-contribution statements (~20 per 100 words).
_INITIALS_DENSITY = 10.0

_CAPTION_START = re.compile(r"(?:Extended Data|Supplementary) (?:Fig\.|Figure|Table) \d+\s*\|")
# Must appear near the START of a chunk: "Extended Data Fig. 3 | ..." opens a
# caption page, while main text only refers to one ("see Extended Data Fig.
# 6c"), with no "|".
_ADMIN_START = re.compile(r"\b(?:Acknowledge?ments?|Author contributions|Competing interests)\b")
_LICENSE = re.compile(r"Creative Commons|reproduction in any medium")
_HEAD_CHARS = 120


def back_matter_kind(text: str) -> str | None:
    """
    Names the kind of back matter a chunk is -- "figure caption",
    "references", "affiliations", "acknowledgements" or "license" -- or None
    for the document's actual content. Heuristic and deliberately
    conservative: a chunk that is only partly back matter (the last
    paragraph of a discussion followed by the first citations) stays
    content.
    """
    words = max(1, len(text.split()))
    head = text[:_HEAD_CHARS]

    def per_100_words(pattern: re.Pattern) -> float:
        return 100 * len(pattern.findall(text)) / words

    if _CAPTION_START.search(head):
        return "figure caption"
    if per_100_words(_NUMBERED_REFERENCE) >= _REFERENCE_DENSITY:
        return "references"
    if per_100_words(_AFFILIATION_WORD) >= _AFFILIATION_DENSITY:
        return "affiliations"
    if _ADMIN_START.search(head) or per_100_words(_INITIALS_LIST) >= _INITIALS_DENSITY:
        return "acknowledgements"
    if words < 300 and _LICENSE.search(text):
        return "license"
    return None


def content_chunks(chunks: list[dict]) -> list[dict]:
    """
    One document's chunks minus its back matter, in reading order, each
    keeping its original "position". Falls back to every chunk when nothing
    would be left, so a document is never dropped from a summary entirely.
    """
    kept = [c for c in chunks if back_matter_kind(c["text"]) is None]
    return kept or list(chunks)


def _excerpt(text: str, max_chars: int) -> str:
    """The start of `text`, at most max_chars, ending at a sentence boundary
    when one falls in the last 40% -- otherwise at a word boundary."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    sentence_end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if sentence_end >= max_chars * 0.6:
        return cut[: sentence_end + 1]
    space = cut.rfind(" ")
    return cut[:space] if space > 0 else cut


def _water_fill_cap(lengths: list[int], budget: int) -> int:
    """Largest per-item cap such that sum(min(length, cap)) fits the budget:
    short items keep their full length and the room they leave goes to the
    long ones, instead of every item being cut to the same size."""
    if sum(lengths) <= budget:
        return max(lengths)
    low, high = 1, max(lengths)
    while low < high:
        mid = (low + high + 1) // 2
        if sum(min(length, mid) for length in lengths) <= budget:
            low = mid
        else:
            high = mid - 1
    return low


def plan_summary_excerpts(chunks_by_source: dict[str, list[dict]], max_per_source: int,
                           max_tokens: int, min_excerpt_tokens: int) -> tuple[dict[str, list[dict]], int]:
    """
    What a single "summary" request sees: for every document, excerpts taken
    at evenly spaced positions from first content chunk to last, trimmed so
    that all of them together fit max_tokens.

    Breadth over depth, on purpose. Real chunks are ~1000 tokens, so a
    5000-token request holds about five WHOLE chunks -- under two per
    document for three papers, which is not enough to span even one of
    them, let alone describe it. A summary needs to see how each document
    begins, develops and ends far more than it needs any one passage in
    full, so this spends the budget on positions first:

      1. Number of positions: as many as max_tokens allows at no less than
         min_excerpt_tokens each, at least 2 per document (both ends), at
         most max_per_source per document. Extra positions go to whichever
         document is most thinly covered -- usually the longest.
      2. Excerpt length: one shared cap, water-filled so short chunks stay
         whole and the room they leave goes to the long ones.

    Returns ({source_name: [excerpt chunk, ...]}, cap) where each excerpt is
    a copy of its chunk with "text" shortened (position, labels and source
    unchanged, so citations and span coverage still work), and cap is the
    per-excerpt token limit that was applied. A cap below min_excerpt_tokens
    means there were too many documents for even two decent excerpts each.
    """
    sources = {name: chunks for name, chunks in chunks_by_source.items() if chunks}
    if not sources:
        return {}, 0

    slots = max(2 * len(sources), max_tokens // max(1, min_excerpt_tokens))
    sizes = {name: min(2, len(chunks)) for name, chunks in sources.items()}

    def spacing(name: str) -> float:
        return (len(sources[name]) - 1) / max(1, sizes[name] - 1)

    while sum(sizes.values()) < slots:
        growable = [name for name, chunks in sources.items()
                    if sizes[name] < min(max_per_source, len(chunks))]
        if not growable:
            break
        sizes[max(growable, key=spacing)] += 1

    samples = {name: sample_representative(sources[name], sizes[name]) for name in sources}
    lengths = [estimate_tokens(c["text"]) for sample in samples.values() for c in sample]
    cap = _water_fill_cap(lengths, max_tokens)
    # estimate_tokens is len // 4 + 1, so (cap - 1) * 4 characters stays within cap.
    max_chars = max(4, (cap - 1) * 4)
    excerpts = {
        name: [{**c, "text": _excerpt(c["text"], max_chars)} for c in sample]
        for name, sample in samples.items()
    }
    return excerpts, cap

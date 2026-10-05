"""
Validation/evaluation checks for each pipeline stage, matching the
brief's evaluation panel:

    Extraction -> Completeness + Metadata
    Chunking   -> Size + Overlap
    Crawler    -> Depth + Domain + Duplicates
    Retrieval  -> Precision@K + Recall@K + MRR   (needs labeled data --
                  see evaluate_retrieval; not run automatically on a
                  live request since there's no ground truth for an
                  arbitrary user query)
    Generation -> Groundedness + Citation Accuracy
    System     -> Processing Time + Error Rate

An EvaluationReport instance is built up across one /process run and
returned to the frontend alongside the actual results, so "did this
work correctly" is a real answer, not an assumption.
"""

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from . import chunker, progress
from .config import (
    CHUNK_SIZE_WORDS, CHUNK_OVERLAP_RATIO, CRAWL_MAX_DEPTH, CRAWL_SAME_DOMAIN_ONLY,
    RERANK_SCORE_WARNING_THRESHOLD, EVALUATION_LOG_PATH, GROUNDEDNESS_THRESHOLD,
    GROQ_MAX_CONTEXT_TOKENS,
)


class EvaluationReport:
    def __init__(self):
        self.checks: list[dict] = []
        self.stage_durations: dict[str, float] = {}
        self.errors: list[str] = []

    def add_check(self, stage: str, name: str, passed: bool, detail: str, value: float | None = None,
                  subject: str | None = None):
        """
        `value` is the actual number the check was judged against (e.g.
        average rerank score, groundedness percentage) when there is one --
        stored separately from `detail` so runs can be compared numerically
        instead of by re-parsing sentences.

        `subject` names WHAT this check was run against when one check runs
        several times per run -- span_coverage emits one row per uploaded
        document, and without this the panel shows three rows all reading
        "span_coverage", distinguishable only by reading the prose.
        """
        self.checks.append({"stage": stage, "name": name, "passed": passed, "detail": detail,
                            "value": value, "subject": subject})

    def record_error(self, stage: str, message: str):
        self.errors.append(f"[{stage}] {message}")

    @contextmanager
    def timed_stage(self, stage_name: str):
        progress.stage(stage_name)
        start = time.perf_counter()
        try:
            yield
        except Exception as exc:
            self.record_error(stage_name, str(exc))
            raise
        finally:
            self.stage_durations[stage_name] = round(time.perf_counter() - start, 3)

    def to_dict(self) -> dict:
        total_checks = len(self.checks)
        passed_checks = sum(1 for c in self.checks if c["passed"])
        return {
            "checks": self.checks,
            "summary": f"{passed_checks}/{total_checks} checks passed" if total_checks else "no checks run",
            "stage_durations_seconds": self.stage_durations,
            "error_count": len(self.errors),
            "errors": self.errors,
        }


# ---------- Extraction ----------

def validate_extraction(records: list[dict], report: EvaluationReport):
    stage = "extraction"
    if not records:
        report.add_check(stage, "completeness", False, "No records extracted.")
        return

    missing_text = [r for r in records if not r.get("text", "").strip()]
    report.add_check(
        stage, "completeness", len(missing_text) == 0,
        f"{len(missing_text)} of {len(records)} record(s) had empty text.",
    )

    required_fields_by_type = {"file": {"filename", "location"}, "web": {"url", "title"}}
    missing_metadata = []
    for r in records:
        required = required_fields_by_type.get(r.get("source_type"), set())
        if not required.issubset(r.keys()):
            missing_metadata.append(r)
    report.add_check(
        stage, "metadata", len(missing_metadata) == 0,
        f"{len(missing_metadata)} of {len(records)} record(s) missing required metadata.",
    )

    # Only when the documents had figures at all -- a run on text-only
    # sources gets no row rather than a vacuous "0 of 0" pass.
    figures = [r for r in records if r.get("content_type", "").startswith("figure")]
    if figures:
        described = sum(1 for r in figures if r["content_type"] == "figure_description")
        report.add_check(
            stage, "figure_descriptions", described == len(figures),
            f"{described} of {len(figures)} figure(s) described by the vision model"
            + ("." if described == len(figures) else
               f"; {len(figures) - described} kept as caption-only (no API key, request "
               f"failure, or over the per-document limit)."),
            value=round(100 * described / len(figures), 1),
        )


# ---------- Chunking ----------

def validate_chunking(chunks: list[dict], report: EvaluationReport):
    stage = "chunking"
    if not chunks:
        report.add_check(stage, "size", False, "No chunks produced.")
        return

    sizes = [len(c["text"].split()) for c in chunks]
    oversized = [s for s in sizes if s > CHUNK_SIZE_WORDS * 1.1]
    # Stated in budget tokens too, because the two units are ~1.5x apart and
    # the chunk count the budget can actually hold is the number that
    # determines how much of a document a summary gets to see.
    max_tokens_per_chunk = max(chunker.estimate_tokens(c["text"]) for c in chunks)
    report.add_check(
        stage, "size", len(oversized) == 0,
        f"{len(oversized)} of {len(chunks)} chunk(s) exceeded ~{CHUNK_SIZE_WORDS} words. "
        f"Largest is ~{max_tokens_per_chunk} tokens, so the "
        f"{GROQ_MAX_CONTEXT_TOKENS}-token generation budget holds roughly "
        f"{max(1, GROQ_MAX_CONTEXT_TOKENS // max_tokens_per_chunk)} of them at once "
        f"(CHUNK_SIZE_WORDS counts WORDS; the budget counts tokens).",
    )

    expected_step = max(1, int(CHUNK_SIZE_WORDS * (1 - CHUNK_OVERLAP_RATIO)))
    overlap_applied = expected_step < CHUNK_SIZE_WORDS
    report.add_check(
        stage, "overlap", overlap_applied,
        f"Configured overlap ratio {CHUNK_OVERLAP_RATIO:.0%}, step size {expected_step} words.",
    )


# ---------- Crawler ----------

def validate_crawler(candidates: list[dict], seed_url: str, report: EvaluationReport):
    stage = "crawler"
    if not candidates:
        report.add_check(stage, "depth", True, "No pages discovered (nothing to check).")
        return

    max_depth_seen = max(c["depth"] for c in candidates)
    report.add_check(
        stage, "depth", max_depth_seen <= CRAWL_MAX_DEPTH,
        f"Max depth seen: {max_depth_seen} (limit {CRAWL_MAX_DEPTH}).",
    )

    if CRAWL_SAME_DOMAIN_ONLY:
        seed_domain = urlparse(seed_url).netloc
        off_domain = [c for c in candidates if urlparse(c["url"]).netloc != seed_domain]
        report.add_check(
            stage, "domain", len(off_domain) == 0,
            f"{len(off_domain)} candidate(s) were off the seed domain ({seed_domain}).",
        )

    urls = [c["url"] for c in candidates]
    duplicates = len(urls) - len(set(urls))
    report.add_check(
        stage, "duplicates", duplicates == 0,
        f"{duplicates} duplicate URL(s) found among {len(urls)} candidates.",
    )


# ---------- Retrieval ----------
# Precision/Recall/MRR need a labeled test set (see evaluate_retrieval below,
# unchanged -- still not run automatically). But ChromaDB already returns a
# similarity distance for every retrieved chunk, and that number is discarded
# after retrieval without ever being checked. A high distance means "closest
# match ChromaDB could find was still not very close" -- a real, automatic
# quality signal that needs no ground truth at all.

RETRIEVAL_DISTANCE_WARNING_THRESHOLD = 1.0  # cosine distance (0=identical, 1=unrelated, 2=opposite)

def validate_retrieval_confidence(retrieved_chunks: list[dict], report: EvaluationReport):
    """
    Diagnostic only -- reports the average rerank score (or distance,
    when the reranker's fallback path was used) but ALWAYS passes.

    This used to be a pass/fail gate on a threshold. It was demoted after
    three real, independently-verified runs on the same document, varying
    only the instruction wording, showed it tracked the WRONG direction:
    it failed the run with the best actual groundedness (52%) and passed
    a weaker one (42%). A short, near-verbatim-heading query can score
    the cross-encoder very highly while still producing a shallow, poorly
    grounded report -- and a genuinely well-grounded run on a harder,
    more natural-language question can score the cross-encoder poorly.
    Absolute cross-encoder score magnitude doesn't reliably predict real
    output quality here, so gating on it produced false alarms and false
    confidence in roughly equal measure.

    Real quality gating now lives in validate_generation's groundedness
    check (vocabulary overlap between the report and its cited sources),
    which has tracked actual quality correctly in every real comparison
    run so far. This check is kept because the number itself is still
    useful context when reading a report -- just not as a pass/fail gate.
    """
    stage = "retrieval"
    if not retrieved_chunks:
        report.add_check(stage, "confidence", False, "No chunks retrieved -- nothing to generate from.")
        return

    has_rerank_scores = all("rerank_score" in c for c in retrieved_chunks)

    if has_rerank_scores:
        scores = [c["rerank_score"] for c in retrieved_chunks]
        avg_score = sum(scores) / len(scores)
        weak_matches = [s for s in scores if s < RERANK_SCORE_WARNING_THRESHOLD]
        report.add_check(
            stage, "confidence", True,
            f"Average rerank score {avg_score:.3f}; {len(weak_matches)} of {len(scores)} "
            f"chunk(s) scored below {RERANK_SCORE_WARNING_THRESHOLD} (informational only -- "
            f"see groundedness for the actual quality gate)." if weak_matches
            else f"Average rerank score {avg_score:.3f} -- all chunks scored as relevant.",
            value=round(avg_score, 3),
        )
        return

    distances = [c["distance"] for c in retrieved_chunks if "distance" in c]
    if not distances:
        # "summary" intent selects by position and runs neither similarity
        # search nor the cross-encoder, so there is no relevance score to
        # report. Say that explicitly rather than silently omitting the row,
        # which reads like the check failed to run.
        report.add_check(
            stage, "confidence", True,
            f"No relevance score applies: all {len(retrieved_chunks)} chunk(s) were selected "
            f"by position across each document, not by matching a query. See span_coverage "
            f"for whether that selection actually covered the documents.",
        )
        return
    avg_distance = sum(distances) / len(distances)
    weak_matches = [d for d in distances if d > RETRIEVAL_DISTANCE_WARNING_THRESHOLD]
    report.add_check(
        stage, "confidence", True,
        f"Average retrieval distance {avg_distance:.3f}; {len(weak_matches)} of {len(distances)} "
        f"chunk(s) exceeded {RETRIEVAL_DISTANCE_WARNING_THRESHOLD} (fallback distance-based check -- "
        f"reranker was unavailable for this run; informational only -- see groundedness for the "
        f"actual quality gate)." if weak_matches
        else f"Average retrieval distance {avg_distance:.3f} -- all chunks within confidence threshold "
             f"(fallback distance-based check -- reranker was unavailable for this run).",
        value=round(avg_distance, 3),
    )


def validate_span_coverage(chunks_by_source: dict[str, list[dict]], retrieved_chunks: list[dict],
                            report: EvaluationReport):
    """
    Only meaningful for "summary" intent, where selection is supposed to
    be POSITIONAL (chunker.sample_representative), not similarity-based.
    Confirms the selected chunks actually span a wide range of each
    document, rather than clustering near the start (or anywhere else) --
    which would defeat the point of positional sampling and silently
    degrade back into a narrow, unrepresentative summary.

    `chunks_by_source` is, per source, the chunks selection was made FROM,
    in reading order. For "summary" that is each document's content with
    its back matter removed (chunker.content_chunks), so coverage is judged
    against the text that can actually be summarized -- a selection
    reaching a paper's reference list is not better coverage.

    Span and gap are measured in steps through that list, not in raw
    positions: with back matter removed, raw positions have holes in them
    (a Nature paper's Methods follow its references), and a hole the
    sampler was right to skip must not count as unsampled text.

    Positions come from each chunk's "position" field (chunker.chunk_all).
    Chunks without one fall back to their index in the list, matched on
    text -- only approximate, since identical texts collapse to one entry.
    """
    stage = "retrieval"
    for name, source_chunks in chunks_by_source.items():
        if len(source_chunks) <= 1:
            continue  # nothing to spread across

        def position_of(chunk: dict, fallback: int | None) -> int | None:
            position = chunk.get("position")
            return position if isinstance(position, int) else fallback

        available = [position_of(c, i) for i, c in enumerate(source_chunks)]
        step_of = {position: step for step, position in enumerate(sorted(set(available)))}
        text_to_position = {c["text"]: p for c, p in zip(source_chunks, available)}

        selected = sorted({
            p for c in retrieved_chunks if c.get("source_name") == name
            for p in [position_of(c, text_to_position.get(c["text"]))]
            if p in step_of
        })
        if not selected:
            continue

        steps = [step_of[p] for p in selected]
        full_span = len(step_of) - 1
        span_fraction = (steps[-1] - steps[0]) / full_span
        covers_everything_available = len(steps) >= len(step_of) - 1
        passed = span_fraction >= 0.5 or covers_everything_available

        # Span alone is a RANGE, so the first and last chunk of a 100-chunk
        # document score 100% with everything between them unread. The
        # largest gap says how much went unrepresented in one stretch,
        # including before the first and after the last selected chunk --
        # informational, deliberately not a gate: it needs real runs before
        # a threshold would mean anything.
        boundaries = [0] + steps + [full_span]
        largest_gap = max(b - a for a, b in zip(boundaries, boundaries[1:]))
        gap_fraction = largest_gap / full_span

        content_note = (f" (its {len(step_of)} content chunks run from position "
                        f"{min(step_of)} to {max(step_of)})"
                        if len(step_of) != max(step_of) - min(step_of) + 1
                        else f" (positions {min(step_of)}-{max(step_of)})")
        report.add_check(
            stage, "span_coverage", passed,
            f"Selected chunks span {span_fraction:.0%} of the document's content, from "
            f"position {selected[0]} to {selected[-1]}{content_note}.",
            value=round(span_fraction, 3),
            subject=name,
        )
        report.add_check(
            stage, "span_largest_gap", True,
            f"Widest unsampled stretch is {largest_gap} of {full_span} step(s) through the "
            f"content, {gap_fraction:.0%} of it, with {len(selected)} chunk(s) selected "
            f"(informational only -- span coverage measures the range, this measures the "
            f"holes inside it).",
            value=round(gap_fraction, 3),
            subject=name,
        )


def validate_source_coverage(all_source_names: list[str], retrieved_chunks: list[dict],
                              report: EvaluationReport):
    """
    Catches exactly the multi-PDF bug: a global top-k retrieval can
    silently drop an entire uploaded document if none of its chunks rank
    high enough, even though the user explicitly uploaded it to be
    covered. Confirms every source that went in has at least one chunk
    that came out.
    """
    stage = "retrieval"
    if len(all_source_names) <= 1:
        return  # nothing to compare against with a single source
    retrieved_names = {c["source_name"] for c in retrieved_chunks}
    missing = [name for name in all_source_names if name not in retrieved_names]
    report.add_check(
        stage, "source_coverage", len(missing) == 0,
        f"{len(missing)} of {len(all_source_names)} uploaded source(s) got zero representation "
        f"in retrieval: {missing}" if missing
        else f"All {len(all_source_names)} uploaded sources are represented in retrieval.",
    )


def evaluate_retrieval(retrieved_labels: list[str], relevant_labels: set[str], k: int) -> dict:
    """
    Standard IR metrics. Call this from a test with known-relevant
    source labels for a query -- there's no ground truth for an
    arbitrary live user query, so this isn't run automatically inside
    /process.
    """
    top_k = retrieved_labels[:k]
    hits = [label for label in top_k if label in relevant_labels]

    precision = len(hits) / len(top_k) if top_k else 0.0
    recall = len(hits) / len(relevant_labels) if relevant_labels else 0.0

    mrr = 0.0
    for rank, label in enumerate(top_k, start=1):
        if label in relevant_labels:
            mrr = 1.0 / rank
            break

    return {"precision_at_k": round(precision, 3), "recall_at_k": round(recall, 3), "mrr": round(mrr, 3)}


# ---------- Generation ----------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "this", "that", "these", "those",
    "it", "its", "as", "by", "at", "from", "which", "their", "has", "have",
    "can", "also", "such", "into", "than", "then", "not", "no", "so",
}


def _significant_words(text: str) -> set[str]:
    words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    return {w for w in words if len(w) > 3 and w not in _STOPWORDS}


def compute_groundedness(report_text: str, citations_used: list[str], retrieved_chunks: list[dict]) -> float:
    """
    Returns the raw overlap fraction (0.0-1.0) between the report's
    significant vocabulary and the text of its cited chunks. A coarse
    proxy (word overlap, not entailment) -- catches wholesale fabrication
    or drift far from the source material, though a report could still
    reuse a source's words while getting the meaning wrong.

    Pulled out of validate_generation so pipeline.py's self-correction
    retry can compute the SAME number to decide whether a retry is worth
    trying, instead of a second, potentially inconsistent definition of
    "grounded".
    """
    if not citations_used:
        return 0.0
    cited_text = " ".join(c["text"] for c in retrieved_chunks if c["source_label"] in citations_used)
    source_words = _significant_words(cited_text)
    report_words = _significant_words(report_text)
    return len(report_words & source_words) / len(report_words) if report_words else 0.0


def validate_generation(report_text: str, citations_used: list[str], retrieved_chunks: list[dict],
                         report: EvaluationReport):
    stage = "generation"
    retrieved_labels = [c["source_label"] for c in retrieved_chunks]

    has_citations = "[" in report_text and "]" in report_text
    report.add_check(
        stage, "citation_markers_present", has_citations,
        "Report text contains bracketed citation markers." if has_citations
        else "No [n] citation markers found in generated report text.",
    )

    unknown_citations = [label for label in citations_used if label not in retrieved_labels]
    report.add_check(
        stage, "citation_accuracy", len(unknown_citations) == 0,
        f"{len(unknown_citations)} cited source(s) were not among the retrieved chunks.",
    )

    # Groundedness: not just "did it cite something" (too weak -- a report can
    # cite a source while actually saying something the source never mentions).
    # Instead, check how much of the report's actual vocabulary shows up
    # anywhere in the cited chunks' text.
    if not citations_used:
        report.add_check(stage, "groundedness", False, "Report cites zero sources -- not grounded in retrieved evidence.")
    else:
        overlap = compute_groundedness(report_text, citations_used, retrieved_chunks)
        grounded = overlap >= GROUNDEDNESS_THRESHOLD
        report.add_check(
            stage, "groundedness", grounded,
            f"{overlap:.0%} of the report's significant vocabulary appears in its cited sources "
            f"(threshold {GROUNDEDNESS_THRESHOLD:.0%})." +
            ("" if grounded else " Low overlap suggests possible drift from source content."),
            value=round(overlap * 100, 1),
        )

    # Groundedness above is measured against chunk TEXT, and for a figure
    # that text is the vision model's description -- a chart value the model
    # misread still counts as grounded. Informational, not pass/fail: it
    # says how much of the report rests on evidence nobody transcribed by hand.
    generated_labels = {c["source_label"] for c in retrieved_chunks
                        if c.get("content_type") == "figure_description"}
    figure_citations = [label for label in citations_used if label in generated_labels]
    if figure_citations:
        report.add_check(
            stage, "figure_citations", True,
            f"{len(figure_citations)} of {len(citations_used)} cited source(s) are model-generated "
            f"figure descriptions ({', '.join(figure_citations)}); groundedness for these is "
            f"checked against the description, not the figure itself.",
            value=len(figure_citations),
        )


# ---------- PDF export ----------
# This replaces an earlier, misdirected check that scanned the RAW extracted
# document text for typographic punctuation. That was checking the wrong
# thing: raw extracted text is never written into the PDF -- only the LLM's
# generated report_text and the citation labels are. This check instead
# verifies the text that actually gets rendered, after pdf_export's own
# sanitization step, so it's a real regression check on the output rather
# than a check that fails on every academic PDF regardless of correctness.

_PROBLEM_CHARS = set("\u2010\u2011\u2012\u2013\u2014\u2018\u2019\u201c\u201d\u2026")


def validate_pdf_encoding_safety(sanitized_report_text: str, sanitized_citations: list[str],
                                  report: EvaluationReport):
    stage = "pdf_export"
    combined = sanitized_report_text + " ".join(sanitized_citations)
    found = _PROBLEM_CHARS & set(combined)
    report.add_check(
        stage, "encoding_safety", len(found) == 0,
        "No unrenderable typographic characters remain in the text actually written to the PDF."
        if not found else
        f"Sanitization missed character(s) {found} -- these will render as blank boxes in the output PDF.",
    )


# ---------- Persistent evaluation history ----------
# Every run's evaluation summary gets appended here as one JSON line.
# This is what turns "before vs after" comparisons into something you
# can actually query, rather than reconstructing them from screenshots
# and memory (which is what every comparison so far in this project has
# had to do). Never raises -- a logging failure should never break an
# actual /process response.

def append_run_to_log(report: "EvaluationReport", run_id: str | None = None,
                       instructions: str = "", log_path: str | None = None) -> None:
    # Resolved at call time, not as a default argument: a default is fixed
    # when the module loads, so tests could not redirect it -- and every
    # test run wrote into the real Run history.
    log_path = log_path or EVALUATION_LOG_PATH
    run_id = run_id or uuid.uuid4().hex[:12]
    summary = report.to_dict()
    entry = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "instructions": instructions,
        "summary": summary["summary"],
        "error_count": summary["error_count"],
        "stage_durations_seconds": summary["stage_durations_seconds"],
        "checks": [
            {"stage": c["stage"], "name": c["name"], "passed": c["passed"], "value": c.get("value")}
            for c in summary["checks"]
        ],
    }
    try:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # logging must never break the actual response


def read_run_history(log_path: str | None = None, limit: int = 50) -> list[dict]:
    """
    Reads the evaluation log back out, newest first, for comparison
    across runs. Returns [] if the log doesn't exist yet (first run
    hasn't happened) rather than raising -- an empty history is a
    normal, expected state, not an error.
    """
    path = Path(log_path or EVALUATION_LOG_PATH)
    if not path.exists():
        return []

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a corrupted line rather than failing the whole read
    entries.reverse()  # newest first
    return entries[:limit]

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

import time
from contextlib import contextmanager
from urllib.parse import urlparse

from .config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_RATIO, CRAWL_MAX_DEPTH, CRAWL_SAME_DOMAIN_ONLY, RERANK_SCORE_WARNING_THRESHOLD


class EvaluationReport:
    def __init__(self):
        self.checks: list[dict] = []
        self.stage_durations: dict[str, float] = {}
        self.errors: list[str] = []

    def add_check(self, stage: str, name: str, passed: bool, detail: str):
        self.checks.append({"stage": stage, "name": name, "passed": passed, "detail": detail})

    def record_error(self, stage: str, message: str):
        self.errors.append(f"[{stage}] {message}")

    @contextmanager
    def timed_stage(self, stage_name: str):
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


# ---------- Chunking ----------

def validate_chunking(chunks: list[dict], report: EvaluationReport):
    stage = "chunking"
    if not chunks:
        report.add_check(stage, "size", False, "No chunks produced.")
        return

    sizes = [len(c["text"].split()) for c in chunks]
    oversized = [s for s in sizes if s > CHUNK_SIZE_TOKENS * 1.1]
    report.add_check(
        stage, "size", len(oversized) == 0,
        f"{len(oversized)} of {len(chunks)} chunk(s) exceeded ~{CHUNK_SIZE_TOKENS} tokens.",
    )

    expected_step = max(1, int(CHUNK_SIZE_TOKENS * (1 - CHUNK_OVERLAP_RATIO)))
    overlap_applied = expected_step < CHUNK_SIZE_TOKENS
    report.add_check(
        stage, "overlap", overlap_applied,
        f"Configured overlap ratio {CHUNK_OVERLAP_RATIO:.0%}, step size {expected_step} tokens.",
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
# Raised from 0.8 -> 1.0 after real evidence: a confirmed well-grounded,
# accurately-cited run (see conversation log) produced an average distance
# of 0.856, which the 0.8 threshold flagged as a false failure. Short
# instruction-style queries ("focus on X") compared against long paragraph
# chunks structurally produce higher distances than same-length text pairs
# would, even for genuinely relevant matches -- this is a property of how
# sentence embedding models handle length asymmetry, not a sign of a bad
# match. 1.0 leaves headroom above the observed good-match value while
# still catching genuinely unrelated retrievals (distance approaching or
# past the "unrelated" midpoint of the 0-2 cosine range).

def validate_retrieval_confidence(retrieved_chunks: list[dict], report: EvaluationReport):
    """
    Prefers "rerank_score" when present (the cross-encoder's direct
    relevance judgment -- more accurate than embedding distance). Falls
    back to "distance" for chunks that went through reranker's fallback
    path (reranker model unavailable), so this check still means
    something even when reranking itself degraded.
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
            stage, "confidence", len(weak_matches) == 0,
            f"Average rerank score {avg_score:.3f}; {len(weak_matches)} of {len(scores)} "
            f"chunk(s) scored below the relevance threshold ({RERANK_SCORE_WARNING_THRESHOLD}) -- "
            f"the source material may not actually cover the given instructions well." if weak_matches
            else f"Average rerank score {avg_score:.3f} -- all chunks scored as relevant.",
        )
        return

    distances = [c["distance"] for c in retrieved_chunks if "distance" in c]
    if not distances:
        return
    avg_distance = sum(distances) / len(distances)
    weak_matches = [d for d in distances if d > RETRIEVAL_DISTANCE_WARNING_THRESHOLD]
    report.add_check(
        stage, "confidence", len(weak_matches) == 0,
        f"Average retrieval distance {avg_distance:.3f}; {len(weak_matches)} of {len(distances)} "
        f"chunk(s) exceeded the {RETRIEVAL_DISTANCE_WARNING_THRESHOLD} weak-match threshold -- "
        f"the source material may not actually cover the given instructions well." if weak_matches
        else f"Average retrieval distance {avg_distance:.3f} -- all chunks within confidence threshold "
             f"(fallback distance-based check -- reranker was unavailable for this run).",
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
    # anywhere in the cited chunks' text. This is a coarse proxy (word overlap,
    # not entailment), but it catches wholesale fabrication or drift far
    # from the source material, which "citations_used is non-empty" cannot.
    if not citations_used:
        report.add_check(stage, "groundedness", False, "Report cites zero sources -- not grounded in retrieved evidence.")
    else:
        cited_text = " ".join(c["text"] for c in retrieved_chunks if c["source_label"] in citations_used)
        source_words = _significant_words(cited_text)
        report_words = _significant_words(report_text)
        overlap = len(report_words & source_words) / len(report_words) if report_words else 0.0
        grounded = overlap >= 0.3
        report.add_check(
            stage, "groundedness", grounded,
            f"{overlap:.0%} of the report's significant vocabulary appears in its cited sources "
            f"(threshold 30%)." + ("" if grounded else " Low overlap suggests possible drift from source content."),
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


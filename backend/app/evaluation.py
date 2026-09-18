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

from .config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_RATIO, CRAWL_MAX_DEPTH, CRAWL_SAME_DOMAIN_ONLY


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


# ---------- Retrieval (needs labeled data -- run manually against a test set) ----------

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

def validate_generation(report_text: str, citations_used: list[str], retrieved_labels: list[str],
                         report: EvaluationReport):
    stage = "generation"

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

    report.add_check(
        stage, "groundedness", len(citations_used) > 0,
        f"Report cites {len(citations_used)} distinct source(s)." if citations_used
        else "Report cites zero sources -- likely not grounded in retrieved evidence.",
    )

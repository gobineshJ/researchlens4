"""
Tests every validate_* function with BOTH a good case (should pass) and
a genuinely bad case (should fail). A check that only ever gets tested
against good input is a check you can't trust -- these deliberately
construct broken input for each one.
"""

from app import evaluation, pdf_export


# ---------- Extraction ----------

def test_extraction_completeness_passes_on_good_records():
    report = evaluation.EvaluationReport()
    records = [{"source_type": "file", "filename": "x.pdf", "location": "page 1", "text": "real text"}]
    evaluation.validate_extraction(records, report)
    completeness = next(c for c in report.checks if c["name"] == "completeness")
    assert completeness["passed"]


def test_extraction_completeness_fails_on_empty_text():
    report = evaluation.EvaluationReport()
    records = [{"source_type": "file", "filename": "x.pdf", "location": "page 1", "text": ""}]
    evaluation.validate_extraction(records, report)
    completeness = next(c for c in report.checks if c["name"] == "completeness")
    assert not completeness["passed"]


def test_extraction_metadata_fails_when_fields_missing():
    report = evaluation.EvaluationReport()
    records = [{"source_type": "file", "text": "text with no filename or location"}]
    evaluation.validate_extraction(records, report)
    metadata = next(c for c in report.checks if c["name"] == "metadata")
    assert not metadata["passed"]


# ---------- Chunking ----------

def test_chunking_size_fails_on_oversized_chunk():
    report = evaluation.EvaluationReport()
    huge_chunk = {"text": " ".join(["word"] * 2000), "source_label": "x", "source_type": "file"}
    evaluation.validate_chunking([huge_chunk], report)
    size_check = next(c for c in report.checks if c["name"] == "size")
    assert not size_check["passed"]


def test_chunking_size_passes_on_normal_chunk():
    report = evaluation.EvaluationReport()
    normal_chunk = {"text": " ".join(["word"] * 500), "source_label": "x", "source_type": "file"}
    evaluation.validate_chunking([normal_chunk], report)
    size_check = next(c for c in report.checks if c["name"] == "size")
    assert size_check["passed"]


# ---------- Crawler ----------

def test_crawler_fails_on_off_domain_pages():
    report = evaluation.EvaluationReport()
    candidates = [
        {"url": "https://example.com/a", "title": "A", "snippet": "", "depth": 0},
        {"url": "https://totally-different.com/b", "title": "B", "snippet": "", "depth": 1},
    ]
    evaluation.validate_crawler(candidates, "https://example.com/a", report)
    domain_check = next(c for c in report.checks if c["name"] == "domain")
    assert not domain_check["passed"]


def test_crawler_fails_on_duplicate_urls():
    report = evaluation.EvaluationReport()
    candidates = [
        {"url": "https://example.com/a", "title": "A", "snippet": "", "depth": 0},
        {"url": "https://example.com/a", "title": "A again", "snippet": "", "depth": 1},
    ]
    evaluation.validate_crawler(candidates, "https://example.com/a", report)
    dup_check = next(c for c in report.checks if c["name"] == "duplicates")
    assert not dup_check["passed"]


def test_crawler_passes_on_clean_same_domain_pages():
    report = evaluation.EvaluationReport()
    candidates = [
        {"url": "https://example.com/a", "title": "A", "snippet": "", "depth": 0},
        {"url": "https://example.com/b", "title": "B", "snippet": "", "depth": 1},
    ]
    evaluation.validate_crawler(candidates, "https://example.com/a", report)
    assert all(c["passed"] for c in report.checks)


# ---------- Retrieval confidence ----------

def test_retrieval_confidence_passes_on_close_match():
    report = evaluation.EvaluationReport()
    chunks = [{"source_label": "x", "text": "t", "distance": 0.3}]
    evaluation.validate_retrieval_confidence(chunks, report)
    assert report.checks[0]["passed"]


def test_retrieval_confidence_is_informational_on_distant_match():
    """
    validate_retrieval_confidence is diagnostic-only (see its docstring
    for why): across real runs, absolute score/distance magnitude didn't
    reliably predict actual output quality, so it no longer gates the
    report. It should still report the real average distance, but always
    pass -- groundedness is the actual quality gate now.
    """
    report = evaluation.EvaluationReport()
    chunks = [{"source_label": "x", "text": "t", "distance": 1.8}]  # near "opposite" end of 0-2 range
    evaluation.validate_retrieval_confidence(chunks, report)
    assert report.checks[0]["passed"]
    assert report.checks[0]["value"] == 1.8


def test_retrieval_confidence_fails_on_empty_retrieval():
    report = evaluation.EvaluationReport()
    evaluation.validate_retrieval_confidence([], report)
    assert not report.checks[0]["passed"]


def test_retrieval_confidence_uses_rerank_score_when_present():
    """When chunks carry rerank_score (the reranker actually ran), that
    should be preferred over distance -- a high distance but positive
    rerank_score should still PASS, since rerank_score is the more
    accurate signal."""
    report = evaluation.EvaluationReport()
    chunks = [{"source_label": "x", "text": "t", "distance": 1.5, "rerank_score": 4.0}]
    evaluation.validate_retrieval_confidence(chunks, report)
    assert report.checks[0]["passed"]
    assert "rerank score" in report.checks[0]["detail"]


def test_retrieval_confidence_is_informational_on_negative_rerank_score():
    """A negative rerank score is still reported (via detail/value), but
    no longer fails the check -- see the informational-only test above."""
    report = evaluation.EvaluationReport()
    chunks = [{"source_label": "x", "text": "t", "distance": 0.3, "rerank_score": -3.0}]
    evaluation.validate_retrieval_confidence(chunks, report)
    assert report.checks[0]["passed"]
    assert report.checks[0]["value"] == -3.0


def test_retrieval_confidence_falls_back_to_distance_when_reranker_degraded():
    """Chunks from reranker's fallback path have no rerank_score at all --
    should fall back to the distance-based check (still informational-
    only), not silently skip."""
    report = evaluation.EvaluationReport()
    chunks = [{"source_label": "x", "text": "t", "distance": 1.8}]  # no rerank_score key
    evaluation.validate_retrieval_confidence(chunks, report)
    assert report.checks[0]["passed"]
    assert "distance" in report.checks[0]["detail"]


# ---------- Span coverage (positional sampling for "summary" intent) ----------

def _source_chunks(n: int, name: str = "doc.pdf") -> list[dict]:
    return [{"text": f"chunk-{i}", "source_name": name} for i in range(n)]


def test_span_coverage_passes_when_selection_spans_wide_range():
    report = evaluation.EvaluationReport()
    source_chunks = _source_chunks(100)
    # picks positions 0, 25, 50, 75, 99 -- spans the full range
    retrieved = [source_chunks[i] for i in (0, 25, 50, 75, 99)]
    evaluation.validate_span_coverage({"doc.pdf": source_chunks}, retrieved, report)
    assert report.checks[0]["passed"]


def test_span_coverage_fails_when_selection_clusters_near_start():
    report = evaluation.EvaluationReport()
    source_chunks = _source_chunks(100)
    # all picks bunched in the first 10% of the document
    retrieved = [source_chunks[i] for i in (0, 2, 4, 6, 8)]
    evaluation.validate_span_coverage({"doc.pdf": source_chunks}, retrieved, report)
    assert not report.checks[0]["passed"]


def test_span_coverage_passes_for_small_documents_regardless_of_spread():
    """When a document has few chunks, selecting nearly all of them
    trivially satisfies "representative", even if the numeric span
    fraction alone would look narrow."""
    report = evaluation.EvaluationReport()
    source_chunks = _source_chunks(3)
    retrieved = [source_chunks[0], source_chunks[1]]
    evaluation.validate_span_coverage({"doc.pdf": source_chunks}, retrieved, report)
    assert report.checks[0]["passed"]


def test_span_coverage_skips_documents_with_one_chunk():
    report = evaluation.EvaluationReport()
    source_chunks = _source_chunks(1)
    evaluation.validate_span_coverage({"doc.pdf": source_chunks}, source_chunks, report)
    assert report.checks == []


def test_span_coverage_checks_each_source_independently():
    report = evaluation.EvaluationReport()
    wide = _source_chunks(100, name="wide.pdf")
    narrow = _source_chunks(100, name="narrow.pdf")
    retrieved = (
        [wide[i] for i in (0, 25, 50, 75, 99)]      # spans fully
        + [narrow[i] for i in (0, 1, 2, 3, 4)]      # clustered near start
    )
    evaluation.validate_span_coverage(
        {"wide.pdf": wide, "narrow.pdf": narrow}, retrieved, report
    )
    results = {c["detail"].split(":")[0]: c["passed"] for c in report.checks}
    assert results["wide.pdf"]
    assert not results["narrow.pdf"]


# ---------- Generation / groundedness ----------

def test_compute_groundedness_returns_zero_with_no_citations():
    assert evaluation.compute_groundedness("some report text", [], []) == 0.0


def test_compute_groundedness_matches_validate_generation_value():
    """compute_groundedness (used by pipeline.py's self-correction retry)
    and validate_generation's own "groundedness" check must agree exactly --
    otherwise the retry could decide to keep/discard a result based on a
    number that doesn't match what the report itself will show."""
    retrieved = [{"source_label": "paper.pdf, page 1",
                  "text": "machine learning models require large labeled datasets for training"}]
    report_text = "The paper states that machine learning models require large labeled datasets [1]."
    citations = ["paper.pdf, page 1"]

    raw_overlap = evaluation.compute_groundedness(report_text, citations, retrieved)

    report = evaluation.EvaluationReport()
    evaluation.validate_generation(report_text, citations, retrieved, report)
    grounded_check = next(c for c in report.checks if c["name"] == "groundedness")

    assert round(raw_overlap * 100, 1) == grounded_check["value"]


def test_groundedness_passes_when_report_matches_sources():
    report = evaluation.EvaluationReport()
    retrieved = [{"source_label": "paper.pdf, page 1",
                  "text": "machine learning models require large labeled datasets for training"}]
    report_text = "The paper states that machine learning models require large labeled datasets [1]."
    evaluation.validate_generation(report_text, ["paper.pdf, page 1"], retrieved, report)
    grounded = next(c for c in report.checks if c["name"] == "groundedness")
    assert grounded["passed"]


def test_groundedness_fails_when_report_is_unrelated_to_sources():
    report = evaluation.EvaluationReport()
    retrieved = [{"source_label": "paper.pdf, page 1",
                  "text": "machine learning models require large labeled datasets for training"}]
    report_text = "The stock market experienced volatility due to unrelated geopolitical tensions [1]."
    evaluation.validate_generation(report_text, ["paper.pdf, page 1"], retrieved, report)
    grounded = next(c for c in report.checks if c["name"] == "groundedness")
    assert not grounded["passed"]


def test_citation_accuracy_fails_on_fabricated_citation():
    report = evaluation.EvaluationReport()
    retrieved = [{"source_label": "real.pdf, page 1", "text": "real content"}]
    report_text = "Some claim [1]."
    # Citing a source that was never actually retrieved
    evaluation.validate_generation(report_text, ["fabricated.pdf, page 99"], retrieved, report)
    accuracy = next(c for c in report.checks if c["name"] == "citation_accuracy")
    assert not accuracy["passed"]


def test_groundedness_fails_with_zero_citations():
    report = evaluation.EvaluationReport()
    evaluation.validate_generation("A report with no citations at all.", [], [], report)
    grounded = next(c for c in report.checks if c["name"] == "groundedness")
    assert not grounded["passed"]


# ---------- PDF encoding safety ----------

def test_pdf_encoding_safety_passes_after_real_sanitization():
    report = evaluation.EvaluationReport()
    raw = "disease\u2010specific and missing\u2010value handling"
    sanitized = pdf_export._sanitize_for_pdf(raw)
    evaluation.validate_pdf_encoding_safety(sanitized, [], report)
    assert report.checks[0]["passed"]


def test_pdf_encoding_safety_fails_if_sanitization_is_skipped():
    """
    Deliberately passes UNSANITIZED text to prove the check would catch
    a future regression where someone removes or breaks the sanitizer.
    """
    report = evaluation.EvaluationReport()
    raw = "disease\u2010specific diagnosis"  # smart hyphen, never sanitized
    evaluation.validate_pdf_encoding_safety(raw, [], report)
    assert not report.checks[0]["passed"]


# ---------- Source coverage (the 3-PDF bug) ----------

def test_source_coverage_fails_when_a_document_is_dropped():
    """
    Reproduces the exact reported bug: 3 uploaded documents, retrieval
    only returned chunks from 2 of them.
    """
    report = evaluation.EvaluationReport()
    all_sources = ["pdf1.pdf", "pdf2.pdf", "pdf3.pdf"]
    retrieved = [
        {"source_name": "pdf1.pdf", "source_label": "pdf1.pdf, page 1", "text": "t", "distance": 0.3},
        {"source_name": "pdf2.pdf", "source_label": "pdf2.pdf, page 1", "text": "t", "distance": 0.3},
    ]
    evaluation.validate_source_coverage(all_sources, retrieved, report)
    assert not report.checks[0]["passed"]
    assert "pdf3.pdf" in report.checks[0]["detail"]


def test_source_coverage_passes_when_every_document_represented():
    report = evaluation.EvaluationReport()
    all_sources = ["pdf1.pdf", "pdf2.pdf", "pdf3.pdf"]
    retrieved = [
        {"source_name": name, "source_label": f"{name}, page 1", "text": "t", "distance": 0.3}
        for name in all_sources
    ]
    evaluation.validate_source_coverage(all_sources, retrieved, report)
    assert report.checks[0]["passed"]


def test_source_coverage_skipped_for_single_document():
    """With only one uploaded document, coverage is trivially satisfied
    by the existing single-source retrieval path -- no check should even
    fire, since there's nothing to compare against."""
    report = evaluation.EvaluationReport()
    evaluation.validate_source_coverage(["only.pdf"], [{"source_name": "only.pdf"}], report)
    assert report.checks == []

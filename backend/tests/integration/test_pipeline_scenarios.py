"""
End-to-end scenario tests against the REAL pipeline (real Groq calls,
real embeddings, real ChromaDB). These need GROQ_API_KEY set and normal
internet access, so they're skipped automatically if that's not
available -- run with: pytest tests/integration/ -v

These are the 6 scenarios discussed earlier, each designed to produce
a specific, predictable outcome -- unlike casual manual testing, each
assertion here states in advance what SHOULD happen, so a failure
tells you something concrete broke, not just "something's different."
"""

import os
import pytest

from app.pipeline import run_pipeline
from app.config import GROQ_API_KEY

pytestmark = pytest.mark.skipif(
    not GROQ_API_KEY,
    reason="GROQ_API_KEY not set -- integration tests need a real key. "
           "Set it in backend/.env to run these.",
)


def test_scenario_1_deliberately_mismatched_topic(sample_pdf):
    """
    Scenario 1: ask for something the document doesn't cover at all.
    Expect retrieval confidence and/or groundedness to genuinely fail --
    if they don't, the thresholds are too loose to catch real problems.
    """
    result = run_pipeline(
        file_paths=[sample_pdf],
        seed_urls=[],
        instructions="Summarize this paper's discussion of 16th-century naval warfare tactics.",
    )
    checks = result["evaluation"]["checks"]
    retrieval_confidence = next((c for c in checks if c["name"] == "confidence"), None)
    groundedness = next((c for c in checks if c["name"] == "groundedness"), None)

    # At least one of these should catch the mismatch -- assert on whichever
    # ran (groundedness only runs if something got cited at all).
    caught_mismatch = (retrieval_confidence and not retrieval_confidence["passed"]) or \
                       (groundedness and not groundedness["passed"])
    assert caught_mismatch, (
        "Neither retrieval confidence nor groundedness caught a deliberately "
        "unrelated topic -- thresholds may be too loose. "
        f"Checks: {checks}"
    )


def test_scenario_2_image_only_pdf_degrades_gracefully(empty_pdf):
    """
    Scenario 2: a PDF with no extractable text. Should fail extraction
    completeness cleanly, NOT crash the pipeline.
    """
    result = run_pipeline(
        file_paths=[empty_pdf],
        seed_urls=[],
        instructions="Summarize this document.",
    )
    checks = result["evaluation"]["checks"]
    completeness = next((c for c in checks if c["name"] == "completeness"), None)
    assert completeness is not None
    assert not completeness["passed"]
    # Should not have produced a fabricated report from nothing
    assert result["report_text"] == "" or result["chunks_created"] == 0


def test_scenario_5_tiny_input_fewer_chunks_than_top_k(tmp_path):
    """
    Scenario 5: input smaller than TOP_K_CHUNKS (5). Should not crash
    when there are fewer chunks available than the requested top-k.
    """
    tiny_file = tmp_path / "tiny.txt"
    tiny_file.write_text("A single short sentence about artificial intelligence.")

    result = run_pipeline(
        file_paths=[str(tiny_file)],
        seed_urls=[],
        instructions="Summarize this in detail.",
    )
    assert result["chunks_created"] >= 1
    assert len(result["retrieved_chunks"]) <= result["chunks_created"]


@pytest.mark.skipif(
    not os.getenv("RUN_NETWORK_URL_TESTS"),
    reason="Set RUN_NETWORK_URL_TESTS=1 to run tests that crawl a live URL.",
)
def test_scenario_3_irrelevant_seed_url_gets_filtered():
    """
    Scenario 3: a seed URL with nothing to do with the instructions.
    The Groq relevance filter should reject it -- filtered_pages should
    end up empty (or very small), not everything discovered getting
    scraped indiscriminately.
    """
    result = run_pipeline(
        file_paths=[],
        seed_urls=["https://en.wikipedia.org/wiki/Chocolate_chip_cookie"],
        instructions="Discuss recent advances in machine learning for medical diagnosis.",
    )
    assert len(result["filtered_pages"]) == 0, (
        "Relevance filter let through pages unrelated to the instructions: "
        f"{result['filtered_pages']}"
    )


@pytest.mark.skipif(
    not os.getenv("RUN_NETWORK_URL_TESTS"),
    reason="Set RUN_NETWORK_URL_TESTS=1 to run tests that crawl a live URL.",
)
def test_scenario_4_web_only_run_no_file():
    """
    Scenario 4: only a seed URL, no uploaded file at all. This path
    (crawl -> filter -> scrape -> chunk -> embed) has never been
    exercised by a file-only test, so this confirms it works too.
    """
    result = run_pipeline(
        file_paths=[],
        seed_urls=["https://en.wikipedia.org/wiki/Machine_learning"],
        instructions="Summarize what machine learning is.",
    )
    assert result["discovered_pages"] > 0
    assert result["chunks_created"] > 0
    assert result["report_text"] != ""

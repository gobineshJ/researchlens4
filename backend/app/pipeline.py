"""
Orchestrates the full flow: parse files -> discover pages -> Groq filter
-> scrape selected pages -> chunk -> embed/store -> retrieve -> Groq
generate -> PDF export. Every stage's evaluation check runs inline, so
the returned result always includes a real evaluation report, not a
separate afterthought.
"""

import uuid
from pathlib import Path

from . import doc_parser, scraper, groq_client, chunker, vectorstore, pdf_export, evaluation, reranker, hybrid_search
from .config import (
    TOP_K_CHUNKS, TOP_K_CHUNKS_PER_SOURCE, CANDIDATE_OVERSAMPLE_FACTOR,
    ANALYSIS_TOP_K_MULTIPLIER, RRF_K, GROUNDEDNESS_THRESHOLD, MAX_RETRIEVAL_ATTEMPTS,
)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _hybrid_retrieve_and_rerank(source_names: list[str], chunks_by_source: dict[str, list[dict]],
                                 collection, queries: list[str], final_k: int, oversample_k: int) -> list[dict]:
    """
    Runs the hybrid (BM25 + vector, RRF-fused) retrieval + cross-encoder
    reranking pipeline, per source document, for a given list of search
    queries. Shared by the initial retrieval attempt and the self-
    correction retry (which calls this again with reformulated queries)
    so both go through exactly the same retrieval logic.
    """
    retrieved: list[dict] = []
    for name in source_names:
        source_chunks = chunks_by_source[name]
        bm25_index, bm25_chunks = hybrid_search.build_index(source_chunks)

        fused_by_text: dict[str, dict] = {}
        for query in queries:
            vector_ranked = vectorstore.retrieve_candidates(
                collection, [query], k_per_query=oversample_k, source_name=name
            )
            bm25_ranked = hybrid_search.search(bm25_index, bm25_chunks, query, top_k=oversample_k)
            fused = hybrid_search.fuse_rrf(vector_ranked, bm25_ranked, k_rrf=RRF_K)
            for chunk in fused:
                text = chunk["text"]
                if text not in fused_by_text or chunk["rrf_score"] > fused_by_text[text]["rrf_score"]:
                    fused_by_text[text] = chunk

        candidates = sorted(fused_by_text.values(), key=lambda c: c["rrf_score"], reverse=True)[:oversample_k]
        retrieved.extend(reranker.rerank_with_fallback(queries, candidates, top_k=final_k))
    return retrieved


def run_pipeline(file_paths: list[str], seed_urls: list[str], instructions: str) -> dict:
    report = evaluation.EvaluationReport()
    all_records: list[dict] = []

    # --- Stage 1: parse uploaded files ---
    with report.timed_stage("extraction"):
        for path in file_paths:
            all_records.extend(doc_parser.parse_document(path))
    evaluation.validate_extraction(all_records, report)

    # --- Stage 2: crawl discovery ---
    all_candidates: list[dict] = []
    with report.timed_stage("crawl_discovery"):
        for seed in seed_urls:
            found = scraper.discover(seed)
            all_candidates.extend(found)
            evaluation.validate_crawler(found, seed, report)

    # --- Stage 3: Groq relevance filter ---
    filtered_candidates: list[dict] = []
    with report.timed_stage("relevance_filter"):
        if all_candidates:
            filtered_candidates = groq_client.filter_relevant_pages(
                all_candidates, topic=instructions, instructions=instructions
            )

    # --- Stage 4: scrape only the pages that passed the filter ---
    web_records: list[dict] = []
    with report.timed_stage("scraping"):
        for candidate in filtered_candidates:
            if candidate.get("relevant"):
                scraped = scraper.scrape_page(candidate["url"], candidate.get("title", ""))
                if scraped:
                    web_records.append(scraped)
        all_records.extend(web_records)

    # --- Stage 5: chunk everything ---
    with report.timed_stage("chunking"):
        chunks = chunker.chunk_all(all_records)
    evaluation.validate_chunking(chunks, report)

    # --- Stage 6: embed + store in a fresh ChromaDB session collection ---
    with report.timed_stage("embedding_and_storage"):
        collection, session_id = vectorstore.new_session_collection()
        vectorstore.add_chunks(collection, chunks)

    # --- Stage 7: retrieve for the report query ---
    # An intent router decides HOW to retrieve, because different
    # instructions need fundamentally different strategies, not just
    # different prompts:
    #
    #   "summary" -- the user wants the whole document covered. There is
    #       no specific topic to search FOR, so similarity/keyword
    #       search is the wrong tool entirely: it would just surface
    #       whatever happens to embed closest to the word "summarize".
    #       Instead, chunks are sampled evenly by POSITION across each
    #       source document (chunker.sample_representative), then
    #       reranked against the expanded queries only to pick the best
    #       ordering/labeling among the positional sample -- not to
    #       filter it down by relevance.
    #
    #   "targeted_qa" / "analysis" -- a real topic exists to search for.
    #       First, query decomposition (groq_client.decompose_query) checks
    #       whether the instruction actually asks about MULTIPLE distinct
    #       things ("what dataset did they use, and what are the paper's
    #       limitations?") -- if so, each sub-question gets expanded and
    #       retrieved independently, so evidence for each part gets pulled
    #       in rather than one averaged embedding missing one of them. This
    #       is query decomposition / "compositional" multi-hop: sub-
    #       questions are retrieved independently and in parallel, not as
    #       a chain where one hop's results shape the next query (that
    #       would be iterative multi-hop, which this does not implement).
    #       For each source document and each resulting query, both dense
    #       vector search AND BM25 keyword search run independently, then
    #       their two RANKINGS (not raw scores, which live on incomparable
    #       scales) are combined with Reciprocal Rank Fusion. This catches
    #       both exact-term matches (a specific dataset/method name BM25
    #       catches but embeddings might not rank highly) and semantic
    #       matches (paraphrases BM25 would miss). The best-scoring fused
    #       candidates per source are then reranked by the cross-encoder.
    #       "analysis" gets a wider final_k than "targeted_qa" --
    #       synthesizing a theme genuinely needs more supporting evidence
    #       than answering one narrow question.
    with report.timed_stage("retrieval"):
        source_names = sorted({c["source_name"] for c in chunks})
        chunks_by_source: dict[str, list[dict]] = {}
        for c in chunks:
            chunks_by_source.setdefault(c["source_name"], []).append(c)

        intent = groq_client.classify_intent(instructions)
        report.add_check(
            "retrieval", "intent", True,
            f"Classified as '{intent}' -- "
            + {
                "summary": "sampling chunks spread across each document's full span, not by similarity.",
                "targeted_qa": "hybrid vector + BM25 retrieval (RRF-fused), then reranked.",
                "analysis": "hybrid vector + BM25 retrieval (RRF-fused, wider net), then reranked.",
            }[intent],
        )

        multi_source = len(source_names) > 1
        base_k = TOP_K_CHUNKS_PER_SOURCE if multi_source else TOP_K_CHUNKS
        final_k = round(base_k * ANALYSIS_TOP_K_MULTIPLIER) if intent == "analysis" else base_k
        oversample_k = final_k * CANDIDATE_OVERSAMPLE_FACTOR

        sub_questions: list[str] = [instructions]
        retrieved: list[dict] = []

        if intent == "summary":
            expanded_queries = groq_client.expand_query(instructions)
            for name in source_names:
                sample = chunker.sample_representative(chunks_by_source[name], final_k)
                retrieved.extend(reranker.rerank_with_fallback(expanded_queries, sample, top_k=final_k))
        else:
            sub_questions = groq_client.decompose_query(instructions)
            is_decomposed = len(sub_questions) > 1
            report.add_check(
                "retrieval", "query_decomposition", True,
                (f"Instruction decomposed into {len(sub_questions)} independent sub-questions, "
                 f"each retrieved separately: {sub_questions}") if is_decomposed
                else f"Instruction treated as a single focused question (no decomposition needed): {sub_questions[0]!r}",
            )
            expanded_queries = _dedupe_preserve_order(
                q for sq in sub_questions for q in groq_client.expand_query(sq)
            )
            retrieved = _hybrid_retrieve_and_rerank(
                source_names, chunks_by_source, collection, expanded_queries, final_k, oversample_k
            )

    # --- Stage 8: Groq report generation (with one self-correction retry) ---
    generation_result = {"report_text": "", "citations_used": []}
    with report.timed_stage("generation"):
        if retrieved:
            generation_result = groq_client.generate_report(retrieved, instructions)

    # Self-correction: if the first pass came back poorly grounded, retry
    # retrieval once with reformulated queries before settling for it.
    # Only for "targeted_qa"/"analysis" -- "summary"'s positional sampling
    # doesn't depend on the search query, so reformulating it wouldn't
    # change which chunks get picked.
    if intent != "summary" and MAX_RETRIEVAL_ATTEMPTS > 1 and retrieved:
        original_groundedness = evaluation.compute_groundedness(
            generation_result["report_text"], generation_result["citations_used"], retrieved
        )
        if original_groundedness < GROUNDEDNESS_THRESHOLD:
            with report.timed_stage("self_correction_retry"):
                retry_queries = _dedupe_preserve_order(
                    q for sq in sub_questions
                    for q in groq_client.expand_query(sq, avoid=expanded_queries)
                )
                retried = _hybrid_retrieve_and_rerank(
                    source_names, chunks_by_source, collection, retry_queries, final_k, oversample_k
                )
                retry_generation = groq_client.generate_report(retried, instructions) if retried else generation_result
                retry_groundedness = (
                    evaluation.compute_groundedness(
                        retry_generation["report_text"], retry_generation["citations_used"], retried
                    ) if retried else original_groundedness
                )
                improved = bool(retried) and retry_groundedness > original_groundedness
                report.add_check(
                    "generation", "self_correction", True,
                    f"Initial groundedness {original_groundedness:.0%} was below the "
                    f"{GROUNDEDNESS_THRESHOLD:.0%} threshold; retried retrieval with "
                    f"{len(retry_queries)} reformulated quer{'y' if len(retry_queries) == 1 else 'ies'} "
                    f"-> {retry_groundedness:.0%}. " +
                    ("Retry improved groundedness -- using the retried result."
                     if improved else
                     "Retry did not improve groundedness -- keeping the original result."),
                    value=round((retry_groundedness - original_groundedness) * 100, 1),
                )
                if improved:
                    retrieved = retried
                    generation_result = retry_generation

    # Evaluation checks that depend on the FINAL retrieved/generated
    # result run once, here -- after any self-correction retry has been
    # decided, so they reflect what was actually returned, not a
    # since-discarded first attempt.
    evaluation.validate_retrieval_confidence(retrieved, report)
    evaluation.validate_source_coverage(source_names, retrieved, report)
    if intent == "summary":
        evaluation.validate_span_coverage(chunks_by_source, retrieved, report)
    evaluation.validate_generation(
        generation_result["report_text"], generation_result["citations_used"], retrieved, report
    )

    # --- Stage 9: PDF export ---
    pdf_path = None
    with report.timed_stage("pdf_export"):
        if generation_result["report_text"]:
            filename = f"report_{uuid.uuid4().hex[:8]}.pdf"
            pdf_path = pdf_export.build_pdf(
                generation_result["report_text"], generation_result["citations_used"], filename
            )
            # Verify against the SAME sanitization pdf_export just applied,
            # so this checks the actual rendered output, not the raw input.
            evaluation.validate_pdf_encoding_safety(
                pdf_export._sanitize_for_pdf(generation_result["report_text"]),
                [pdf_export._sanitize_for_pdf(c) for c in generation_result["citations_used"]],
                report,
            )

    evaluation.append_run_to_log(report, instructions=instructions)

    return {
        "extracted_records": len(all_records),
        "discovered_pages": len(all_candidates),
        "filtered_pages": [c for c in filtered_candidates if c.get("relevant")],
        "chunks_created": len(chunks),
        "retrieved_chunks": retrieved,
        "report_text": generation_result["report_text"],
        "citations_used": generation_result["citations_used"],
        "pdf_path": pdf_path,
        # Just the filename, OS-agnostic -- the frontend uses this directly
        # for the download link instead of trying to parse pdf_path itself,
        # which broke on Windows because it uses backslashes, not "/".
        "pdf_filename": Path(pdf_path).name if pdf_path else None,
        "evaluation": report.to_dict(),
    }

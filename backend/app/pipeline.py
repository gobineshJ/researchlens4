"""
Orchestrates the full flow: parse files -> discover pages -> Groq filter
-> scrape selected pages -> chunk -> embed/store -> retrieve -> Groq
generate -> PDF export. Every stage's evaluation check runs inline, so
the returned result always includes a real evaluation report, not a
separate afterthought.
"""

import uuid
from pathlib import Path

from . import doc_parser, scraper, groq_client, chunker, vectorstore, pdf_export, evaluation, reranker
from .config import TOP_K_CHUNKS, TOP_K_CHUNKS_PER_SOURCE, CANDIDATE_OVERSAMPLE_FACTOR


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
    # Three things happen here, in order:
    #   1. Query expansion -- the raw instructions get rewritten into a
    #      few more specific search queries (falls back to just the
    #      original instructions if Groq is unavailable).
    #   2. Oversampled retrieval PER SOURCE DOCUMENT -- guarantees every
    #      uploaded file/page gets its own candidate pool, so reranking
    #      in step 3 can't silently squeeze an entire document out the
    #      way one global top-k used to.
    #   3. Reranking -- a cross-encoder re-scores each source's candidate
    #      pool against ALL of the expanded queries (not just the raw
    #      instructions) and keeps each candidate's BEST score across
    #      them. A vague instruction like "summary" gives the
    #      cross-encoder nothing specific to match against on its own;
    #      scoring against every expanded variant means a chunk only
    #      needs to genuinely answer ONE reasonable phrasing of what was
    #      asked. Falls back to distance-sorted results if the reranker
    #      model can't load.
    with report.timed_stage("retrieval"):
        source_names = sorted({c["source_name"] for c in chunks})
        expanded_queries = groq_client.expand_query(instructions)
        multi_source = len(source_names) > 1
        final_k = TOP_K_CHUNKS_PER_SOURCE if multi_source else TOP_K_CHUNKS
        oversample_k = final_k * CANDIDATE_OVERSAMPLE_FACTOR

        retrieved: list[dict] = []
        if multi_source:
            for name in source_names:
                candidates = vectorstore.retrieve_candidates(
                    collection, expanded_queries, k_per_query=oversample_k, source_name=name
                )
                retrieved.extend(reranker.rerank_with_fallback(expanded_queries, candidates, top_k=final_k))
        else:
            candidates = vectorstore.retrieve_candidates(
                collection, expanded_queries, k_per_query=oversample_k
            )
            retrieved = reranker.rerank_with_fallback(expanded_queries, candidates, top_k=final_k)
    evaluation.validate_retrieval_confidence(retrieved, report)
    evaluation.validate_source_coverage(source_names, retrieved, report)

    # --- Stage 8: Groq report generation ---
    generation_result = {"report_text": "", "citations_used": []}
    with report.timed_stage("generation"):
        if retrieved:
            generation_result = groq_client.generate_report(retrieved, instructions)
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

    evaluation.append_run_to_log(report)

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

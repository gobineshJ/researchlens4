"""
Orchestrates the full flow: parse files -> discover pages -> Groq filter
-> scrape selected pages -> chunk -> embed/store -> retrieve -> Groq
generate -> PDF export. Every stage's evaluation check runs inline, so
the returned result always includes a real evaluation report, not a
separate afterthought.
"""

import uuid
from collections import Counter
from pathlib import Path

from . import doc_parser, scraper, groq_client, chunker, vectorstore, pdf_export, evaluation, reranker, hybrid_search, mindmap
from .config import (
    TOP_K_CHUNKS, TOP_K_CHUNKS_PER_SOURCE, CANDIDATE_OVERSAMPLE_FACTOR,
    ANALYSIS_TOP_K_MULTIPLIER, RRF_K, GROUNDEDNESS_THRESHOLD, MAX_RETRIEVAL_ATTEMPTS,
    GROQ_MAX_CONTEXT_TOKENS, SUMMARY_MAX_EXCERPTS_PER_SOURCE, SUMMARY_MIN_EXCERPT_TOKENS,
    MULTI_HOP_ENABLED, MULTI_HOP_MAX_EXTRA_HOPS, ENTAILMENT_CHECK_ENABLED, ENTAILMENT_MAX_CLAIMS,
    CLAIM_REVISION_ENABLED, MIND_MAP_ENABLED,
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


def _interleave_by_source(earlier: list[dict], later: list[dict]) -> list[dict]:
    """
    Merges a follow-up hop's chunks into what earlier hops found,
    alternating within each source (earlier[0], later[0], earlier[1], ...).
    fit_chunks_to_budget keeps a prefix of each source's list, so simply
    appending would make the hop's results -- the very facts the follow-up
    went looking for -- the first thing a budget trim drops.
    """
    by_source: dict[str, tuple[list[dict], list[dict]]] = {}
    for c in earlier:
        by_source.setdefault(c["source_name"], ([], []))[0].append(c)
    for c in later:
        by_source.setdefault(c["source_name"], ([], []))[1].append(c)
    merged: list[dict] = []
    for first, second in by_source.values():
        for i in range(max(len(first), len(second))):
            merged.extend(group[i] for group in (first, second) if i < len(group))
    return merged


def _multi_hop_retrieve(instructions: str, queries: list[str], source_names: list[str],
                        chunks_by_source: dict[str, list[dict]], collection,
                        final_k: int, oversample_k: int) -> tuple[list[dict], list[dict]]:
    """
    Iterative multi-hop retrieval. Hop 1 is the ordinary hybrid retrieval
    for `queries`. Then, up to MULTI_HOP_MAX_EXTRA_HOPS times, Groq reads
    the evidence found so far (groq_client.plan_follow_up) and decides
    whether the question needs a fact that evidence points to but doesn't
    contain -- "what dataset did the method that beat the 2019 baseline
    use?" can only look up the dataset once hop 1 has revealed the
    method's name. Its follow-up queries are retrieved as the next hop
    and merged in. Stops as soon as the planner says the evidence
    suffices, or a hop turns up nothing new.

    Returns (retrieved chunks, one record per hop for the evaluation panel).
    """
    retrieved = _hybrid_retrieve_and_rerank(
        source_names, chunks_by_source, collection, queries, final_k, oversample_k
    )
    hops = [{"queries": list(queries), "new_chunks": len(retrieved), "missing": ""}]
    if not MULTI_HOP_ENABLED or not retrieved:
        return retrieved, hops

    tried = list(queries)
    for _ in range(MULTI_HOP_MAX_EXTRA_HOPS):
        plan = groq_client.plan_follow_up(instructions, retrieved, tried)
        if plan["complete"] or not plan["queries"]:
            break
        found = _hybrid_retrieve_and_rerank(
            source_names, chunks_by_source, collection, plan["queries"], final_k, oversample_k
        )
        seen = {c["text"] for c in retrieved}
        new = [c for c in found if c["text"] not in seen]
        hops.append({"queries": plan["queries"], "new_chunks": len(new), "missing": plan["missing"]})
        tried += plan["queries"]
        if not new:
            break
        retrieved = _interleave_by_source(retrieved, new)
    return retrieved, hops


_FLAGGED = ("unsupported", "contradicted")


def _revise_flagged_claims(generation_result: dict, claims: list[dict], verdicts: list[dict],
                           retrieved: list[dict], report) -> dict:
    """
    Targeted revision: rewrites only the sentences the entailment judge
    flagged (groq_client.revise_claims), splices them back into the report
    in place, and judges the rewrites again. `verdicts` is updated in
    place: a revised claim gets its new verdict, with the original sentence
    kept under "original"; a removed one gets verdict "removed".

    Returns the generation result to use from here on -- the original one
    if nothing was flagged or the revision request failed.
    """
    flagged = [i for i, v in enumerate(verdicts) if v["verdict"] in _FLAGGED]
    if not flagged:
        return generation_result

    with report.timed_stage("claim_revision"):
        try:
            revisions = groq_client.revise_claims(
                [{"claim": verdicts[i]["claim"], "sources": verdicts[i]["sources"],
                  "reason": verdicts[i]["reason"]} for i in flagged],
                retrieved,
            )
            applied = [(i, r) for i, r in zip(flagged, revisions) if r is not None]
            rewrites = [(i, r) for i, r in applied if r["action"] == "rewrite"]
            new_verdicts = groq_client.judge_claims(
                [{"text": evaluation.clean_claim_text(r["text"]), "sources": claims[i]["sources"]}
                 for i, r in rewrites],
                retrieved,
            ) if rewrites else []
        except Exception as exc:
            report.record_error("claim_revision", f"Flagged claims could not be revised: {exc}")
            return generation_result

    report_text = evaluation.apply_revisions(
        generation_result["report_text"],
        [{**r, "span": claims[i]["span"]} for i, r in applied],
    )
    for i, r in applied:
        verdicts[i]["original"] = verdicts[i]["claim"]
        verdicts[i]["original_verdict"] = verdicts[i]["verdict"]
        if r["action"] == "remove":
            verdicts[i].update(verdict="removed", claim="", reason="Removed: its cited excerpts support none of it.")
    for (i, r), new in zip(rewrites, new_verdicts):
        verdicts[i].update(claim=new["claim"], verdict=new["verdict"], reason=new["reason"])

    fixed = sum(1 for i, _ in rewrites if verdicts[i]["verdict"] == "supported")
    removed = sum(1 for _, r in applied if r["action"] == "remove")
    report.add_check(
        "generation", "claim_revision", True,
        f"{len(flagged)} flagged sentence(s): {len(rewrites)} rewritten against their cited "
        f"excerpts ({fixed} now judged supported), {removed} removed"
        + (f", {len(flagged) - len(applied)} left unchanged (the reviser skipped them)"
           if len(applied) < len(flagged) else "")
        + ". The rest of the report is unchanged; originals are listed in the Claim check.",
        value=fixed + removed,
    )
    return {"report_text": report_text,
            "citations_used": groq_client.citations_for(report_text, retrieved)}


def _hop_detail(hops: list[dict]) -> str:
    if len(hops) == 1:
        return ("Single hop: the evidence from the first retrieval pass was judged sufficient, "
                "so no follow-up lookups were needed.")
    steps = "; ".join(
        f"hop {n + 2} looked up {hop['missing'] or 'a follow-up'} via {hop['queries']} "
        f"-> {hop['new_chunks']} new chunk(s)"
        for n, hop in enumerate(hops[1:])
    )
    return (f"{len(hops)} hops: each follow-up was planned from the evidence the previous hop "
            f"retrieved. {steps}.")


def run_pipeline(file_paths: list[str], seed_urls: list[str], instructions: str,
                 username: str | None = None, run_id: str | None = None, *,
                 extra_records: list[dict] | None = None, intent: str | None = None,
                 export_pdf: bool = True, mind_map: bool = True) -> dict:
    """Runs every stage (see _run_pipeline) and always drops the run's
    vector collection afterwards, even if a stage failed.

    For callers that know more than the instructions say (the daily digest):
      extra_records  records shaped like doc_parser's, placed BEFORE the
                     parsed files -- e.g. a paper's arXiv abstract, which
                     then becomes the first chunk of that paper
      intent         skip intent classification and use this one
      export_pdf     False when the caller lays out its own PDF
      mind_map       False to skip the mind map (and its Groq call)
    """
    collections: list = []
    try:
        return _run_pipeline(file_paths, seed_urls, instructions, username, run_id, collections,
                             extra_records or [], intent, export_pdf, mind_map)
    finally:
        for collection in collections:
            vectorstore.drop_session_collection(collection)


def _run_pipeline(file_paths: list[str], seed_urls: list[str], instructions: str,
                  username: str | None, run_id: str | None, collections: list,
                  extra_records: list[dict], forced_intent: str | None, export_pdf: bool,
                  mind_map: bool) -> dict:
    report = evaluation.EvaluationReport()
    all_records: list[dict] = list(extra_records)

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
        collections.append(collection)
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
    #       questions are retrieved independently and in parallel. On top
    #       of that, _multi_hop_retrieve adds ITERATIVE multi-hop: after
    #       each pass, Groq reads what was found and, if the question needs
    #       a fact that evidence points to but doesn't contain, retrieves
    #       for it as the next hop.
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

        intent = forced_intent or groq_client.classify_intent(instructions)
        report.add_check(
            "retrieval", "intent", True,
            f"{'Set by the caller to' if forced_intent else 'Classified as'} '{intent}' -- "
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
            # No query expansion and no cross-encoder on this path. Both only
            # make sense when there is a topic to rank AGAINST, and there
            # isn't one: positional sampling has already chosen the chunks,
            # and reranking with top_k == len(sample) cannot filter anything,
            # it can only reorder. That reordering is what let the context
            # budget trim drop a document's opening chunks by relevance rank
            # and collapse its span. Document order is also simply the right
            # order for a summary prompt -- a model asked to summarize reads
            # better from a document's beginning to its end than from its
            # most "relevant" passage to its least.
            expanded_queries = [instructions]
            # Sample from each document's CONTENT. A paper's last chunks are
            # usually reference lists or figure captions, so "first and last
            # chunk" would otherwise mean "abstract and bibliography".
            summary_chunks_by_source = {
                name: chunker.content_chunks(chunks_by_source[name]) for name in source_names
            }
            excluded: dict[str, Counter] = {
                name: Counter(
                    kind for c in chunks_by_source[name]
                    if (kind := chunker.back_matter_kind(c["text"])) is not None
                )
                for name in source_names
                if len(summary_chunks_by_source[name]) < len(chunks_by_source[name])
            }
            report.add_check(
                "retrieval", "back_matter", True,
                "Excluded from summary sampling: " + "; ".join(
                    f"{name}: {sum(kinds.values())} of {len(chunks_by_source[name])} chunk(s) "
                    f"(" + ", ".join(f"{n} {kind}" for kind, n in kinds.most_common()) + ")"
                    for name, kinds in excluded.items()
                ) + "."
                if excluded
                else "No reference lists, figure captions, affiliations or acknowledgements "
                     "detected -- sampling from every chunk.",
                value=sum(sum(k.values()) for k in excluded.values()),
            )

            # One request has to cover every document, and real chunks are
            # ~1000 tokens -- whole chunks would mean under two per paper.
            # So spend the budget on POSITIONS: evenly spaced excerpts from
            # each document's first content chunk to its last, each trimmed
            # to an equal share of the budget (chunker.plan_summary_excerpts).
            # Sizing this up front also means the budget trim below never
            # has to drop anything, which is what collapsed span coverage
            # when it dropped whole chunks by relevance rank.
            samples, excerpt_cap = chunker.plan_summary_excerpts(
                summary_chunks_by_source, SUMMARY_MAX_EXCERPTS_PER_SOURCE,
                GROQ_MAX_CONTEXT_TOKENS, SUMMARY_MIN_EXCERPT_TOKENS,
            )
            sampled = [c for name in source_names for c in samples.get(name, [])]
            full_tokens = [chunker.estimate_tokens(c["text"])
                           for chunks in summary_chunks_by_source.values() for c in chunks]
            typical = sorted(full_tokens)[len(full_tokens) // 2] if full_tokens else 0
            trimmed = sum(
                1 for name in source_names for excerpt, original in zip(
                    samples.get(name, []),
                    chunker.sample_for_summary(summary_chunks_by_source[name],
                                               len(samples.get(name, []))),
                ) if excerpt["text"] != original["text"]
            )
            too_thin = excerpt_cap < SUMMARY_MIN_EXCERPT_TOKENS and trimmed > 0
            sizes_text = ", ".join(f"{name}: {len(samples.get(name, []))}" for name in source_names)
            report.add_check(
                "retrieval", "budget_fit", not too_thin,
                f"{len(source_names)} documents need at least 2 excerpts each, which leaves "
                f"only ~{excerpt_cap} tokens per excerpt in the {GROQ_MAX_CONTEXT_TOKENS}-token "
                f"budget -- below the {SUMMARY_MIN_EXCERPT_TOKENS}-token minimum, so each one "
                f"says little. Summarize fewer documents per run or raise "
                f"GROQ_MAX_CONTEXT_TOKENS."
                if too_thin
                else f"{len(sampled)} excerpt(s) at section-anchored and evenly spaced positions ({sizes_text}); "
                     f"{trimmed} trimmed to ~{excerpt_cap} tokens (a typical chunk is "
                     f"~{typical}) so all of them fit the {GROQ_MAX_CONTEXT_TOKENS}-token "
                     f"budget. A summary needs to see how each document begins, develops "
                     f"and ends more than it needs any one passage whole."
                if trimmed
                else f"{len(sampled)} whole chunk(s) at section-anchored and evenly spaced positions ({sizes_text}) "
                     f"fit the {GROQ_MAX_CONTEXT_TOKENS}-token budget without trimming.",
                value=len(sampled),
            )
            retrieved.extend(sampled)
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
            retrieved, hops = _multi_hop_retrieve(
                instructions, expanded_queries, source_names, chunks_by_source, collection,
                final_k, oversample_k,
            )
            if MULTI_HOP_ENABLED:
                report.add_check("retrieval", "multi_hop", True, _hop_detail(hops), value=len(hops))

    # --- Stage 8: Groq report generation (with one self-correction retry) ---
    # Trim to the generation context budget BEFORE generating, and keep the
    # trimmed list as `retrieved`, so the evaluation checks below measure
    # the chunks the model actually saw.
    # preserve_span: for "summary", consider chunks for keeping in
    # endpoint-first position order, so even if the estimate above is
    # slightly off this trim cannot drop a document's first or last
    # sampled chunk and undo the positional sampling.
    retrieved = groq_client.fit_chunks_to_budget(retrieved, preserve_span=(intent == "summary"))

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
                retried_hops, _ = _multi_hop_retrieve(
                    instructions, retry_queries, source_names, chunks_by_source, collection,
                    final_k, oversample_k,
                )
                retried = groq_client.fit_chunks_to_budget(retried_hops)
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
        evaluation.validate_span_coverage(summary_chunks_by_source, retrieved, report)
    # Entailment: does each cited sentence say what its cited excerpt says?
    # The report's [n] markers index `retrieved` -- the exact list the final
    # generation call was numbered from.
    claim_verdicts: list[dict] = []
    if ENTAILMENT_CHECK_ENABLED and generation_result["report_text"]:
        claims = evaluation.extract_cited_claims(
            generation_result["report_text"], len(retrieved), max_claims=ENTAILMENT_MAX_CLAIMS
        )
        with report.timed_stage("entailment"):
            try:
                claim_verdicts = groq_client.judge_claims(claims, retrieved) if claims else []
                judged = True
            except Exception as exc:
                # The report itself is fine; only the check couldn't run.
                # Recorded as an error rather than a pass or a fail.
                report.record_error("entailment", f"Entailment check could not run: {exc}")
                judged = False
        if judged and CLAIM_REVISION_ENABLED:
            generation_result = _revise_flagged_claims(
                generation_result, claims, claim_verdicts, retrieved, report
            )
        if judged:
            evaluation.validate_entailment(claims, claim_verdicts, report)

    # After any revision, so these describe the report that is returned.
    evaluation.validate_generation(
        generation_result["report_text"], generation_result["citations_used"], retrieved, report
    )

    # --- Stage 9: mind map ---
    # From the FINAL report, so its claims match what the user reads. A map
    # that can't be built is recorded, never a reason to fail the run.
    mind_map_graph = None
    if mind_map and MIND_MAP_ENABLED and generation_result["report_text"]:
        try:
            with report.timed_stage("mind_map"):
                mind_map_graph = mindmap.build_mind_map(
                    generation_result["report_text"], retrieved, instructions, claim_verdicts, report
                )
        except Exception:
            pass  # timed_stage already recorded it

    # --- Stage 10: PDF export ---
    # Numbered by the report's own [n] markers -- see groq_client.numbered_citations.
    citations = groq_client.numbered_citations(generation_result["report_text"], retrieved)
    pdf_path = None
    with report.timed_stage("pdf_export"):
        if generation_result["report_text"] and export_pdf:
            filename = f"report_{uuid.uuid4().hex[:8]}.pdf"
            pdf_path = pdf_export.build_pdf(generation_result["report_text"], citations, filename)
            # Verify against the SAME sanitization pdf_export just applied,
            # so this checks the actual rendered output, not the raw input.
            evaluation.validate_pdf_encoding_safety(
                pdf_export._sanitize_for_pdf(generation_result["report_text"]),
                [pdf_export._sanitize_for_pdf(c["label"]) for c in citations],
                report,
            )

    evaluation.append_run_to_log(report, run_id=run_id, instructions=instructions, username=username)

    return {
        "extracted_records": len(all_records),
        "discovered_pages": len(all_candidates),
        "filtered_pages": [c for c in filtered_candidates if c.get("relevant")],
        "chunks_created": len(chunks),
        "retrieved_chunks": retrieved,
        "report_text": generation_result["report_text"],
        "citations_used": generation_result["citations_used"],
        # [{"n": 4, "label": ...}] in marker order -- what a printed source
        # list needs (citations_used has no numbers).
        "citations": citations,
        "claim_verdicts": claim_verdicts,
        # {"themes_from", "nodes", "edges"} -- see mindmap.build_mind_map.
        "mind_map": mind_map_graph,
        "pdf_path": pdf_path,
        # Just the filename, OS-agnostic -- the frontend uses this directly
        # for the download link instead of trying to parse pdf_path itself,
        # which broke on Windows because it uses backslashes, not "/".
        "pdf_filename": Path(pdf_path).name if pdf_path else None,
        "evaluation": report.to_dict(),
    }

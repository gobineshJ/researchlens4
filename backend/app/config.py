"""
Central settings for the whole backend. Nothing here should require
touching pipeline code to tune — crawl limits, chunk size, model
names, all live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Groq ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_FILTER_MODEL = "openai/gpt-oss-20b"       # fast, cheap — used for many small relevance checks
GROQ_GENERATION_MODEL = "openai/gpt-oss-120b"  # higher quality — used once per report
# Note: llama-3.1-8b-instant and llama-3.3-70b-versatile (the original choices)
# were decommissioned by Groq on 2026-08-16. These are Groq's recommended
# replacements as of this update -- check console.groq.com/docs/deprecations
# if you hit a 404 again in future, since Groq's free-tier lineup does change.
GROQ_MAX_CONTEXT_TOKENS = int(os.getenv("GROQ_MAX_CONTEXT_TOKENS", "5000"))
# Upper bound on source-excerpt tokens sent to generate_report in one
# request. Groq rejects any single request larger than the model's
# tokens-per-minute limit with "413 Payload Too Large" -- on the free tier
# that's ~8000 TPM for gpt-oss-120b, and the budget has to leave room for
# the prompt and the generated report itself. Multi-source runs retrieve
# 5-8 chunks PER SOURCE at ~650 tokens each, so without a cap a handful of
# documents blows straight past that limit. Raise this on a paid tier.

# --- Crawl / discovery ---
CRAWL_MAX_DEPTH = 2
CRAWL_MAX_PAGES = 15
CRAWL_SAME_DOMAIN_ONLY = True
REQUEST_TIMEOUT = 8
USER_AGENT = "ResearchLens-Bot/0.1 (training project; respects robots.txt)"

# --- Chunking ---
CHUNK_SIZE_WORDS = 650
# WORDS, not model tokens -- chunker splits on whitespace, and this was
# named CHUNK_SIZE_TOKENS while the context budget above counts tokens as
# len(text)//4. The two units differ by roughly 1.5x (a 650-word chunk is
# ~1000 budget tokens), so reading one as the other silently understates
# how much of GROQ_MAX_CONTEXT_TOKENS each chunk actually consumes --
# which is how a run ended up fitting half as many chunks as expected.
CHUNK_OVERLAP_RATIO = 0.15

# --- Retrieval ---
TOP_K_CHUNKS = 5              # used when there's exactly one source document
TOP_K_CHUNKS_PER_SOURCE = 5   # used per document when there are multiple --
                              # guarantees every uploaded file/page gets
                              # representation instead of one global top-k
                              # that can silently drop an entire document
ANALYSIS_TOP_K_MULTIPLIER = 1.6
# "analysis" instructions (discuss/compare/synthesize a theme) genuinely
# need more supporting evidence than a narrow "targeted_qa" lookup -- a
# wider net than the base per-source k, but still nowhere near a full
# document sample (that's what "summary" intent is for).

# --- Summary sampling ---
SUMMARY_MAX_EXCERPTS_PER_SOURCE = 8
SUMMARY_MIN_EXCERPT_TOKENS = 250
# "summary" sends ONE request, and real chunks are ~1000 tokens, so whole
# chunks would mean under two per document -- not enough to span a paper.
# Instead each document is sampled at up to SUMMARY_MAX_EXCERPTS_PER_SOURCE
# evenly spaced positions, and every excerpt is trimmed so they all fit
# GROQ_MAX_CONTEXT_TOKENS, but to no less than SUMMARY_MIN_EXCERPT_TOKENS
# (~160 words; shorter excerpts stop carrying a coherent point). With the
# default 5000-token budget that is up to 20 excerpts per request.

# --- Query expansion ---
# A short generic instruction ("focus on research gaps") embeds poorly
# against long specific chunks -- rewriting it into a few more specific
# search queries first and retrieving for each catches relevant chunks
# the original phrasing alone would miss.
QUERY_EXPANSION_VARIANTS = 3   # extra rewordings generated per request (plus the original)

# --- Reranking ---
# Vector search is retrieved cheaply in bulk (oversampled), then a
# cross-encoder re-scores that pool for actual relevance to the query --
# catches cases where embedding similarity and true relevance diverge,
# which plain vector distance alone can miss.
CANDIDATE_OVERSAMPLE_FACTOR = 3     # retrieve this many times TOP_K before reranking down
RRF_K = 60
# Reciprocal Rank Fusion constant for combining the BM25 and vector
# rankings: score = sum(1 / (RRF_K + rank)) across both ranked lists.
# 60 is the standard default from the original RRF paper (Cormack et al.)
# and widely reused as-is -- it just controls how quickly a rank's
# contribution decays, not something this project has reason to tune.

# --- Self-correction (adaptive retry) ---
GROUNDEDNESS_THRESHOLD = 0.3
# Shared between evaluation.py's pass/fail groundedness check and
# pipeline.py's self-correction retry decision, so "grounded enough"
# means the exact same thing in both places.
MAX_RETRIEVAL_ATTEMPTS = 2
# 1 initial attempt + 1 retry. If the first pass's report comes back
# below GROUNDEDNESS_THRESHOLD, the pipeline reformulates its search
# queries (avoiding the ones already tried) and retries retrieval +
# generation once before settling for whichever attempt scored higher.
# Only applies to "targeted_qa"/"analysis" intent -- "summary"'s
# positional sampling doesn't depend on the search query at all, so
# retrying with different search terms wouldn't change which chunks
# get picked.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# ms-marco cross-encoders conventionally treat 0 as the relevant/irrelevant
# boundary (positive = relevant, negative = not) -- this is a standard
# heuristic for this model family, not a value empirically calibrated
# against this project's own corpus (same caveat as the retrieval
# distance threshold below: worth checking against a deliberately
# irrelevant test case before trusting it fully).
RERANK_SCORE_WARNING_THRESHOLD = 0.0
# ms-marco-MiniLM-L-6-v2 was trained on MS MARCO passages averaging
# ~55-60 words. Our chunks are ~650 tokens -- roughly 10x longer than
# what the model learned to score, which produces systematically
# negative scores even for genuinely relevant chunks (confirmed: a
# demonstrably well-grounded real run scored -5.970 average). Scoring
# only a short leading snippet of each chunk, closer to the model's
# trained input length, keeps scores in the regime the 0.0 threshold
# actually means something in.
RERANK_SNIPPET_CHARS = 400

# --- ChromaDB ---
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
CHROMA_COLLECTION_NAME = "researchlens"

# --- Embeddings ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Output ---
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./generated_reports")

# --- Evaluation history ---
# Every /process run appends its evaluation summary here as one JSON
# line. This is what lets threshold recalibration be based on a real,
# growing history instead of screenshots and memory -- see
# evaluation.append_run_to_log().
EVALUATION_LOG_PATH = os.getenv("EVALUATION_LOG_PATH", "./evaluation_log.jsonl")

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

# --- Crawl / discovery ---
CRAWL_MAX_DEPTH = 2
CRAWL_MAX_PAGES = 15
CRAWL_SAME_DOMAIN_ONLY = True
REQUEST_TIMEOUT = 8
USER_AGENT = "ResearchLens-Bot/0.1 (training project; respects robots.txt)"

# --- Chunking ---
CHUNK_SIZE_TOKENS = 650
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

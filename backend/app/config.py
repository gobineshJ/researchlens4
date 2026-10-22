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

# --- Figure captioning (multimodal) ---
# Figures in uploaded PDFs are rendered to PNG, described by a vision model,
# and injected as ordinary text records ("paper.pdf, page 7, Figure 3"), so
# the rest of the pipeline treats them exactly like any other chunk.
GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
# Groq's only vision model as of 2026-10 (max 3 images / 20MB per request).
# Same caveat as the models above: check console.groq.com/docs/vision on a 404.
FIGURE_CAPTIONING_ENABLED = os.getenv("FIGURE_CAPTIONING_ENABLED", "true").lower() in ("1", "true", "yes")
FIGURE_RENDER_DPI = 150
# High enough for axis labels and legend text to stay legible to the model,
# low enough that a full-width figure stays well under a megabyte of PNG.
FIGURE_MAX_PER_DOCUMENT = 30
# Each described figure is one Groq request at extraction time. Past this
# many, a document's remaining figures are kept as caption-only records
# rather than risking the free tier's per-minute request limit.
FIGURE_CACHE_PATH = os.getenv("FIGURE_CACHE_PATH", "./figure_cache.json")
# Descriptions keyed by a hash of the rendered image. Every /process run
# builds a fresh collection from scratch, so without this the same paper's
# figures would be re-described (and re-billed) on every run.

# --- Crawl / discovery ---
CRAWL_MAX_DEPTH = 2
CRAWL_MAX_PAGES = 15
CRAWL_SAME_DOMAIN_ONLY = True
REQUEST_TIMEOUT = 8
USER_AGENT = "ResearchLens-Bot/0.1 (training project; respects robots.txt)"
CRAWL_RESPECT_ROBOTS_TXT = os.getenv("CRAWL_RESPECT_ROBOTS_TXT", "true").lower() in ("1", "true", "yes")
# The user-agent above makes a promise; this keeps it. Every fetch (crawl
# discovery and full scrape alike) is checked against the site's
# robots.txt first, and a Crawl-delay is honoured between requests to the
# same site. Only turn it off for sites you own.
ROBOTS_MAX_CRAWL_DELAY_SECONDS = 10
# A Crawl-delay longer than this is treated as "don't crawl this site"
# (pages are skipped) rather than stalling a worker for minutes.
ROBOTS_CACHE_SECONDS = 3600

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
SUMMARY_SECTION_AWARE = True
# Evenly spaced positions miss what a reader of a paper summary wants most:
# a page-9 excerpt is as likely to be a related-work paragraph as a result.
# With this on, the positions are anchored first on the chunks where the
# Results/Experiments, Limitations and Conclusion/Discussion sections start
# (detected from their headings, e.g. "5 Results", "Limitations"), each
# excerpt starting at the heading, and only the remaining slots are spread
# evenly. Documents with no recognisable headings sample exactly as before.

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
RERANK_STRIP_BOILERPLATE = True
# On real academic PDFs a chunk's first RERANK_SNIPPET_CHARS are often a
# running page header ("arXiv:2101.00001v2 [cs.CL] ..."), a figure caption
# or a row of table numbers -- and that is what the cross-encoder then
# scores. With this on, those are stripped before the snippet is taken
# (reranker._scoring_snippet). The full chunk text is unaffected.

# --- Iterative multi-hop retrieval ---
MULTI_HOP_ENABLED = os.getenv("MULTI_HOP_ENABLED", "true").lower() in ("1", "true", "yes")
MULTI_HOP_MAX_EXTRA_HOPS = 2
# After the first retrieval pass ("targeted_qa"/"analysis" only), Groq reads
# the evidence found so far and decides whether the question needs a fact
# the evidence POINTS TO but doesn't contain (e.g. it names the method that
# beat a baseline, and the question asks which dataset that method used).
# If so, it writes follow-up queries naming that thing, and they are
# retrieved as the next hop. At most this many follow-up hops per attempt;
# each costs one small Groq call plus one retrieval pass.
MULTI_HOP_EVIDENCE_TOKENS = 3000
# How much of the evidence-so-far the hop planner gets to read, split
# evenly across the retrieved chunks.

# --- Entailment (claim-level faithfulness) ---
ENTAILMENT_CHECK_ENABLED = os.getenv("ENTAILMENT_CHECK_ENABLED", "true").lower() in ("1", "true", "yes")
GROQ_JUDGE_MODEL = GROQ_FILTER_MODEL
ENTAILMENT_THRESHOLD = 0.7
# Groundedness above is vocabulary overlap: a report can reuse a source's
# words while misstating what it says and still pass. The entailment check
# asks an LLM judge, for each sentence that carries a citation marker,
# whether the excerpt it cites actually supports it. It fails when fewer
# than this share of the judged claims are supported, or when any claim
# contradicts its source.
CLAIM_REVISION_ENABLED = os.getenv("CLAIM_REVISION_ENABLED", "true").lower() in ("1", "true", "yes")
# When the entailment check flags sentences, rewrite ONLY those sentences
# against the excerpts they cite (or remove them if nothing supports them),
# then judge the rewrites again. One generation call plus one judge call,
# and the rest of the report is left exactly as written. The flagged
# originals stay visible in the UI's Claim check.
ENTAILMENT_MAX_CLAIMS = 25
# Longer reports are sampled evenly, so the check's cost stays bounded.
ENTAILMENT_MAX_SOURCE_TOKENS = 3000
# Cited excerpts sent per judging request. Claims are packed into as few
# requests as fit; a claim whose cited excerpts alone exceed this gets them
# truncated.
ENTAILMENT_MAX_COMPLETION_TOKENS = 3000
# Room for the judge's reasoning AND its JSON answer. Groq counts this
# against the per-minute limit up front (8000 TPM for gpt-oss-20b on the
# free tier), so excerpts + claims + this must stay well under 8000.

# --- ChromaDB ---
CHROMA_MODE = os.getenv("CHROMA_MODE", "ephemeral").lower()
# "ephemeral" (default): each run's vectors live in memory and are dropped
# when the run ends. Every run builds a fresh collection and nothing ever
# reads an old one, so there is nothing worth keeping on disk -- and an
# on-disk store is shared by every worker process, which ChromaDB doesn't
# support writing to concurrently. "persistent": collections are written to
# CHROMA_PERSIST_DIR (useful to inspect a run's index while debugging) and
# still deleted when the run ends.
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

# --- Authentication (OAuth2 password flow + JWT bearer tokens) ---
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
# Signs every access token. Set this in .env -- when it's empty a random
# key is generated at startup (see auth.py), which works but logs every
# user out whenever the API restarts.
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
# Short on purpose: the browser keeps the access token only in memory and
# silently swaps it for a new one via the refresh cookie (POST /auth/refresh).
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
# Refresh tokens live in an httpOnly cookie (scripts -- including injected
# ones -- can't read it), are stored hashed in AUTH_DB_PATH, and rotate on
# every use: presenting an already-used one revokes that whole login.
REFRESH_COOKIE_NAME = "researchlens_refresh"
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() in ("1", "true", "yes")
# Set true whenever the app is served over HTTPS, so the cookie is never
# sent over plain HTTP. Must stay false for http://localhost development.
CORS_ORIGINS = [o.strip() for o in os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
# Browser origins allowed to call the API directly. The frontend normally
# goes through a same-origin /api proxy (Vite in development, nginx in
# Docker), which needs no CORS at all; this list only matters when the
# frontend is pointed straight at the API (VITE_API_BASE).
AUTH_DB_PATH = os.getenv("AUTH_DB_PATH", "./researchlens.db")
# SQLite file holding user accounts and which user submitted which job.
ALLOW_REGISTRATION = os.getenv("ALLOW_REGISTRATION", "true").lower() in ("1", "true", "yes")
# Whether the login screen's "Create account" option works. Turn it off
# once your accounts exist; `python -m app.create_user` still works.

# --- Celery (background pipeline runs) ---
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
CELERY_TASK_ALWAYS_EAGER = os.getenv("CELERY_TASK_ALWAYS_EAGER", "false").lower() in ("1", "true", "yes")
# true = run the pipeline inside the API process instead of on a worker,
# so the app still works without Redis or a worker running. /process then
# blocks until the report is done, exactly like before Celery was added.
CELERY_WORKER_CONCURRENCY = int(os.getenv("CELERY_WORKER_CONCURRENCY", "2"))
# Pipeline runs one worker process can execute at once on Linux/macOS/WSL
# (prefork pool). Each process loads its own embedding + reranker models,
# so this is bounded by RAM more than by CPU. Ignored on native Windows,
# where only the solo pool works (one run at a time per worker).
GROQ_RATE_LIMIT_REDIS_URL = os.getenv(
    "GROQ_RATE_LIMIT_REDIS_URL", "" if CELERY_TASK_ALWAYS_EAGER else CELERY_BROKER_URL
)
# Where Groq rate-limit pacing state lives. With several worker processes
# sharing one API key, each one pacing only against its OWN requests still
# collides with the others -- sharing the state through Redis lets every
# process see (and reserve against) the same budget. Empty = per-process
# pacing. Falls back to per-process automatically if Redis is unreachable.
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./job_uploads")
# Uploaded files wait here until a worker picks the job up -- a temp dir
# deleted at the end of the request (the old approach) would be gone by
# then. The API and the worker must both be able to read this path.

# --- API keys (machine clients: n8n, scripts, an MCP server) ---
API_KEY_HEADER_NAME = "X-API-Key"
# Long-lived keys for clients that can't do the browser's login + refresh
# cookie dance. Sent as "X-API-Key: rl_..." on any protected route instead
# of "Authorization: Bearer". Created with POST /auth/api-keys (from a
# logged-in session) or `python -m app.create_api_key <username> <name>`;
# stored hashed, shown once, revoked with DELETE /auth/api-keys/{key_id}.

# --- arXiv (daily paper digests) ---
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_PDF_URL = "https://export.arxiv.org/pdf/{arxiv_id}"
# export.arxiv.org is the host arXiv asks programmatic clients to use, for
# both the search API and PDFs, so automated traffic stays off arxiv.org.
ARXIV_REQUEST_INTERVAL_SECONDS = 3.0
# arXiv's API terms: no more than one request every three seconds. Applied
# per process to searches and to every PDF download.
ARXIV_TIMEOUT_SECONDS = 30
ARXIV_MAX_PDF_BYTES = 50 * 1024 * 1024
ARXIV_MAX_PAPERS_PER_RUN = int(os.getenv("ARXIV_MAX_PAPERS_PER_RUN", "8"))
# Papers one /papers/arxiv/process job may include. Every paper is one more
# source sharing the same GROQ_MAX_CONTEXT_TOKENS budget, so past a handful
# each one is represented by too few excerpts to summarise well.
ARXIV_DEFAULT_INSTRUCTIONS = (
    "Summarize each paper: the problem it addresses, its method, its key "
    "results, and its stated limitations. Then note any themes the papers share."
)

# --- Daily digest (app/daily_digest.py) ---
DIGEST_PAPER_INSTRUCTIONS = (
    "Summarize this one paper for a researcher deciding whether to read it in full. "
    "Begin with a single line of the form 'TL;DR: <one sentence>'. Then write four "
    "short labelled parts -- Problem, Method, Key results, Limitations -- each a few "
    "sentences. Under Key results, give the concrete numbers (scores, gains, "
    "datasets, baselines) the excerpts state. If the excerpts don't state "
    "something, say so in one short sentence instead of guessing."
)
# Used once PER PAPER: each paper gets its own pipeline run and the whole
# GROQ_MAX_CONTEXT_TOKENS budget, instead of five papers sharing one budget
# (about one excerpt each). A feed's `instructions` in digest.toml replaces it.
DIGEST_MIN_RELEVANCE = 0.20
# Papers are ranked by cosine similarity between the feed's topic and each
# paper's title + abstract (the local embedding model -- no API calls), and
# the top max_papers are kept. Measured on a week of real cs.CL/IR/AI
# "retrieval augmented generation" results (2026-10): core RAG papers
# 0.45-0.55, a RAG prompt-injection paper 0.24, an unrelated survey dataset
# 0.14. So this floor only drops clear misses; ranking does the rest.
# Per feed: min_relevance. A one-sentence `topic` ranks better than keywords.

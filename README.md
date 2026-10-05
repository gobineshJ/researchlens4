# ResearchLens

## Description

A RAG (Retrieval-Augmented Generation) research paper summarizer, built as a 3-day AI/ML intern training project. It combines evidence from two sources into one citation-aware report:

* Uploaded research documents (PDF, DOCX, TXT)
* A bounded web crawl starting from user-supplied reference URLs — never open-ended search

The crawler never does open-ended web search — every crawl starts from a URL you supply, follows links a bounded distance, and a Groq-based relevance filter decides which discovered pages are actually worth scraping before any full extraction happens.

Stack: **React** (frontend) + **FastAPI** (backend) + **ChromaDB** (vector store) + **Groq** (LLM filtering and generation).

## Architecture

```
USER
  │
  ├── Research files (PDF/DOCX/TXT) ──► EXTRACTION (PyMuPDF, python-docx)
  ├── Seed URLs ─────────────────────► CRAWLER (Requests + BeautifulSoup) ──► Candidate pages
  └── Instructions ───────────────────────────────────────────────────────┐
                                                                            │
                                              Candidate pages ──► GROQ RELEVANCE FILTER (openai/gpt-oss-20b)
                                                                            │
                                                        selected pages ──► full scrape
                                                                            │
              Extracted file text  +  Scraped web text ──► CHUNKING (500-800 tokens, 10-20% overlap)
                                                                            │
                                          CHUNKS ──► EMBEDDING (sentence-transformers, all-MiniLM-L6-v2)
                                                                            │
                                                               ──► CHROMADB (vectors + metadata, cosine distance)
                                                                            │
                                     Instructions ──► GROQ INTENT ROUTER (openai/gpt-oss-20b)
                                                        classifies: summary / targeted_qa / analysis
                                                                            │
                              ┌─────────────────────────────────────────────┴─────────────────────────────────┐
                              │ intent = "summary"                                    intent = "targeted_qa" / "analysis"
                              ▼                                                                    ▼
                  No query expansion or reranking:                             Instructions ──► GROQ QUERY DECOMPOSITION
                        there is no topic to rank against                                        (splits a COMPOUND instruction into
                              │                                                                   2-3 independent sub-questions, each
                  PER-SOURCE positional sampling                                                  retrieved separately -- "compositional"
                  (chunker.sample_representative --                                               multi-hop; NOT iterative -- see note below)
                   picks chunks evenly spaced across                                              │
                   each document's full span, NOT by                          Each sub-question ──► GROQ QUERY EXPANSION
                   similarity -- "summarize" has no                                              │
                   topic for similarity search to match)                      PER-SOURCE, per expanded query:
                              │                                                 ├─ DENSE vector search (ChromaDB)
                              │                                                 └─ BM25 keyword search (rank-bm25)
                              │                                                               │
                              │                                               RECIPROCAL RANK FUSION (RRF)
                              │                                               combines both rankings by position,
                              │                                               not raw score (incomparable scales) --
                              │                                               "analysis" widens the final pool via
                              │                                               ANALYSIS_TOP_K_MULTIPLIER
                              └─────────────────────────────────────────────┬─────────────────────────────────┘
                                                                            │
                                               Candidate pool ──► CROSS-ENCODER RERANKING
                                                                    (cross-encoder/ms-marco-MiniLM-L-6-v2,
                                                                     scores a short snippet of each candidate
                                                                     against every expanded query, keeps each
                                                                     candidate's best score, per source.
                                                                     "targeted_qa"/"analysis" ONLY -- a
                                                                     positional sample has no query to rank
                                                                     against, and reordering it by relevance
                                                                     is what let the budget trim drop a
                                                                     document's opening chunks)
                                                                            │
                                                              FINAL CHUNKS + source metadata
                                                               trimmed to GROQ_MAX_CONTEXT_TOKENS
                                                               (for "summary": reference lists, figure
                                                                captions, affiliations and
                                                                acknowledgements are excluded first, so
                                                                "the end of the document" means its last
                                                                real content; then up to 8 evenly spaced
                                                                EXCERPTS per document, each trimmed to an
                                                                equal share of this budget -- breadth over
                                                                depth, since whole ~1000-token chunks would
                                                                allow under two per paper. The trim keeps
                                                                each document's first and last chunk before
                                                                any interior one as a safety net)
                                                                            │
                                                    GROQ REPORT GENERATION (openai/gpt-oss-120b)
                                                                            │
                            groundedness < 30%? ──► SELF-CORRECTION RETRY: reformulate queries
                            (targeted_qa/analysis    (avoiding the ones already tried), retrieve +
                             only, one retry max)     generate again, keep whichever attempt scored higher
                                                                            │
                                                      Grounded report + citations ──► REPORTLAB ──► PDF
                                                                            │
                                                                     REACT UI ──► Download
                                                                            │
                                                     Every run's evaluation ──► evaluation_log.jsonl
                                                     (viewable in the UI's "Run history" panel)
```

## Project status

Backend and frontend are both built, including an intent router, query decomposition (compositional multi-hop), hybrid (BM25 + vector) retrieval fused with Reciprocal Rank Fusion, section-aware summarization, cross-encoder reranking, a self-correction retry loop, and persistent run history. Every module has unit tests (113 as of this writing), including the FastAPI routes directly via `TestClient` and the pipeline's own orchestration logic (query decomposition wiring, self-correction retry) via mocked Groq/embedding boundaries. What's been verified with real network access (confirmed working by direct testing): the embedding model, the cross-encoder reranker, and Groq calls. The intent router, hybrid retrieval, RRF fusion, query decomposition, and self-correction retry are new and unit-tested with mocked/synthetic data, but have not yet been run against a live document end-to-end with a real Groq key — that's the next thing to confirm on a machine with network access.

What's still worth doing before fully trusting this on a new machine: `RERANK_SCORE_WARNING_THRESHOLD`-based retrieval confidence (in `config.py`/`evaluation.py`) is now **informational only** — across three real comparison runs it tracked the wrong direction relative to actual report quality (it failed the best run and passed a weaker one), so it no longer gates the report. `validate_generation`'s groundedness check (vocabulary overlap between the report and its cited sources) is the real quality gate now, and was correct in all three of those comparisons; it's also what the new self-correction retry decides on. `tests/integration/test_pipeline_scenarios.py`'s `test_scenario_1_deliberately_mismatched_topic` exists specifically to exercise this with a real document, and `tests/integration/test_retrieval_metrics.py` (new -- see Testing below) runs Precision@K/Recall@K/MRR against real, labeled retrieval for the first time. **Run the smoke test below first** to confirm the environment works before relying on the full pipeline.

**On "multi-hop":** what's implemented is query *decomposition* -- a compound instruction gets split into independent sub-questions, each retrieved on its own -- not iterative multi-hop, where each retrieval step's results would shape the next query. That distinction matters: this catches a request that asks about two separate things at once, but it can't chain facts together (e.g. "what dataset did the method that beat the 2019 baseline use?" needs the first fact to look up the second, which this doesn't do).

## File structure

```
researchlens/
├── .gitignore
├── README.md
├── .github/workflows/backend-tests.yml   # runs unit tests on every push
├── backend/
│   ├── requirements.txt
│   ├── requirements-dev.txt   # adds pytest, pytest-cov
│   ├── pytest.ini
│   ├── .env.example
│   ├── evaluation_log.jsonl   # created on first run; gitignored, per-machine history
│   ├── app/
│   │   ├── main.py            # FastAPI app + endpoints; all but /health behind bearer-token auth
│   │   ├── auth.py            # OAuth2 password flow: login/register, JWT issue + validation
│   │   ├── db.py              # SQLite: user accounts + which user owns which job
│   │   ├── create_user.py     # `python -m app.create_user <name>` CLI
│   │   ├── celery_app.py      # Celery app + the task that runs the pipeline on a worker
│   │   ├── config.py          # all tunable settings, env vars
│   │   ├── doc_parser.py      # PDF/DOCX/TXT -> text + metadata
│   │   ├── scraper.py         # crawl discovery + full-page scrape
│   │   ├── groq_client.py     # relevance filter + intent classification + query decomposition + query expansion + report generation
│   │   ├── chunker.py         # token-bounded chunking with overlap; also positional sampling for "summary" intent
│   │   ├── vectorstore.py     # ChromaDB + embeddings, per-source oversampled retrieval
│   │   ├── hybrid_search.py   # BM25 keyword search + Reciprocal Rank Fusion (RRF) with vector search
│   │   ├── reranker.py        # cross-encoder reranking, multi-query, graceful fallback
│   │   ├── pdf_export.py      # ReportLab PDF generation + Unicode sanitization
│   │   ├── evaluation.py      # validation checks per stage + persistent run logging
│   │   └── pipeline.py        # orchestrates all stages, including the self-correction retry loop
│   └── tests/
│       ├── conftest.py        # shared fixtures (sample PDF/DOCX/TXT generation)
│       ├── unit/               # no network needed, runs in seconds (includes pipeline orchestration tests, mocked)
│       └── integration/        # needs GROQ_API_KEY + (for some tests) internet -- see Testing below
└── frontend/
    ├── package.json
    ├── index.html
    └── src/
        ├── App.jsx             # auth gate, form, job polling, results, evaluation table, run history panel
        ├── LoginScreen.jsx     # login / create-account form
        ├── api.js              # API_BASE, token storage, authenticated fetch, blob downloads
        └── main.jsx
```

## Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Then open `.env` and add your Groq API key (free — see below) and a `JWT_SECRET_KEY`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Pipeline runs execute on a **Celery** worker, with **Redis** as the broker. Start Redis (e.g. `docker run -p 6379:6379 redis`, or `sudo apt install redis-server` inside WSL), then run the API and a worker in two terminals:

```bash
uvicorn app.main:app --reload
celery -A app.celery_app worker --loglevel=info --pool=solo   # --pool=solo is required on Windows
```

This starts the API at `http://127.0.0.1:8000`. Check `http://127.0.0.1:8000/health` — it should return `{"status": "ok"}`.

No Redis? Set `CELERY_TASK_ALWAYS_EAGER=true` in `.env` and skip the worker: the pipeline then runs inside the API process, and `/process` blocks until the report is done.

### Authentication

Every endpoint except `/health`, `/auth/token` and `/auth/register` needs an `Authorization: Bearer <token>` header. Tokens come from the OAuth2 password flow:

```bash
curl -X POST http://127.0.0.1:8000/auth/token -d "username=alice&password=..."
# -> {"access_token": "eyJ...", "token_type": "bearer"}
```

The frontend's login screen does this for you, and has a "Create account" option while `ALLOW_REGISTRATION=true` (the default). Once your accounts exist, set it to `false` and add users from the command line instead:

```bash
python -m app.create_user alice
```

In Swagger (`http://127.0.0.1:8000/docs`), use the **Authorize** button to log in before trying the protected endpoints.

`/process` now returns `{"job_id": ...}` immediately (HTTP 202). Poll `GET /jobs/{job_id}` until `status` is `done` (the pipeline's output is in `result`) or `error` (with `error`). Users can only see their own jobs and download their own reports.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

This opens the app at `http://localhost:5173`. It talks to the backend at `http://127.0.0.1:8000` by default (change `API_BASE` at the top of `src/api.js` if you run the backend elsewhere).

### Getting a free Groq API key

Groq's API is genuinely free — no credit card:

1. Go to **console.groq.com** and sign up (email or Google/GitHub).
2. Go to **console.groq.com/keys** → "Create API Key" → copy it (starts with `gsk_`).
3. Paste it into `backend/.env` as `GROQ_API_KEY=gsk_...`.

Free tier gives ~30 requests/minute and ~14,400 requests/day — enough for this project.

## Smoke test (run this first)

Before trusting the full pipeline, confirm each layer works on your machine, in this order:

1. **Backend health check** — with the backend running, open `http://127.0.0.1:8000/health` in a browser. Expect `{"status": "ok"}`.
2. **Embedding model download** — the first `/process` call downloads `all-MiniLM-L6-v2` from Hugging Face (a few hundred MB). This needs normal internet access and may take a minute the first time only; it's cached after that. If this step fails with a connection error, check your internet connection — nothing else in the code is the cause.
3. **Full run** — in the frontend, upload one small PDF or DOCX, leave the URL box empty, type any instruction ("summarize this"), click Process. With no seed URLs, the crawler/filter/scrape stages are skipped automatically, so this isolates: extraction → chunking → embedding → storage → retrieval → Groq generation → PDF. If this works, add a real seed URL (e.g. a Wikipedia page) next to bring the crawler and relevance filter into the run.
4. **Check the evaluation panel** — the results screen shows a table of pass/fail checks per stage, plus timings. A run isn't "working" just because it didn't crash — check that the evaluation checks actually passed too.

## Evaluation / validation

Each pipeline stage has a real check, not just a "looks fine" assumption. Every check's numeric value (not just pass/fail) is stored, so results are comparable across runs, not just per-run:

| Stage | What's checked |
|---|---|
| Extraction | Every record has non-empty text and required metadata (filename+location, or url+title) |
| Chunking | Chunk sizes stay within the configured token limit; overlap is actually applied between consecutive chunks (checked on real produced output, not just the config) |
| Crawler | Depth limit respected, all URLs on the seed domain, zero duplicates |
| Retrieval — intent | Which of `summary` / `targeted_qa` / `analysis` the instructions were classified as, and which retrieval strategy that triggered — always "passes"; this is a transparency record, not a gate |
| Retrieval — query decomposition | **Only for `targeted_qa`/`analysis`.** Whether the instruction was split into independent sub-questions or treated as a single focused question, and what those sub-questions were — a transparency record, not a gate |
| Retrieval — confidence | **Informational only.** Reports the average cross-encoder rerank score (or embedding distance, if the reranker's fallback path ran) as a numeric value, but never fails the report — see Project status above for why |
| Retrieval — source coverage | With multiple uploaded documents, confirms every one of them actually contributed at least one retrieved chunk — catches a document being silently dropped |
| Retrieval — span coverage | **Only for `summary` intent.** Confirms the positionally-sampled chunks actually spread across at least half of each document's position range (or cover nearly everything, for a short document) — catches positional sampling silently degrading into a narrow cluster |
| Retrieval — Precision@K/Recall@K/MRR | Computed via `evaluate_retrieval()`. Previously only unit-tested as a metric formula; now also run against a real, labeled retrieval in `tests/integration/test_retrieval_metrics.py` (three topically distinct synthetic documents, so "relevant" is unambiguous ground truth) — not run automatically on a live query, since there's no ground truth for an arbitrary user question |
| Generation — self-correction | **Only fires for `targeted_qa`/`analysis` when the first pass's groundedness came back below threshold.** Records the groundedness before and after one retry with reformulated queries, and whether the retry's result was adopted — always "passes"; a transparency record of whether/how self-correction acted, not a gate itself |
| Generation | Citation markers present, every cited source was actually retrieved (citation accuracy), and **groundedness** — the percentage of the report's significant vocabulary that actually appears in its cited sources (not just "cited something"). This is the real pass/fail quality gate for retrieval+generation as a whole, and what the self-correction retry itself decides on. |
| PDF export | The text actually written to the PDF (after sanitization) contains no characters common fonts can't render — checked against the real rendered output, not the raw source text |
| System | Processing time per stage, error count |

**Run history**: every `/process` call appends its evaluation summary (with numeric values, not just pass/fail) to `backend/evaluation_log.jsonl`, viewable via the "Run history" panel in the UI or the `/evaluation-history` endpoint directly. This is what makes before/after comparisons (e.g. after changing a threshold or retrieval strategy) an actual query instead of screenshots.

## Testing

Two kinds of tests, in `backend/tests/`:

- **`tests/unit/`** — no network, no Groq key, runs in seconds. Tests document parsing, chunking (including real overlap between consecutive chunks and positional sampling for "summary" intent, not just config), BM25 search and RRF fusion (`hybrid_search.py`) with synthetic corpora, intent classification and query decomposition's fallback behavior, scraper HTML logic, every evaluation check with both a passing and a genuinely failing case, and `test_pipeline.py` — the pipeline's own orchestration logic (does decomposition get skipped for "summary" intent, does the self-correction retry actually fire and adopt the better result) with every Groq/embedding call mocked, so it's testing the WIRING, not re-testing the mocked-out model behavior.
- **`tests/integration/`** — runs real components. `test_pipeline_scenarios.py` and `test_retrieval_metrics.py` need `GROQ_API_KEY`/`RUN_EMBEDDING_TESTS` respectively; both skip automatically (not an error) if their requirement is missing:
  - **`test_pipeline_scenarios.py`** (needs `GROQ_API_KEY`): a topic mismatch that should fail groundedness on purpose, an image-only PDF that should fail extraction cleanly instead of crashing, and a tiny input smaller than the top-k retrieval count. Two further scenarios (an irrelevant seed URL, a web-only run with no file) hit live URLs and only run if you also set `RUN_NETWORK_URL_TESTS=1`.
  - **`test_retrieval_metrics.py`** (needs `RUN_EMBEDDING_TESTS=1`, downloads the embedding model on first run): runs `evaluate_retrieval()` (Precision@K/Recall@K/MRR) against a real, labeled retrieval for the first time — three topically distinct synthetic documents (machine learning, whales, the printing press) give unambiguous ground truth for what "relevant" means, closing a gap that was previously just a formula with no real evidence behind it.

Run everything:

```bash
cd backend
pip install -r requirements-dev.txt
pytest tests/unit/ -v          # fast, always safe to run
pytest tests/integration/test_pipeline_scenarios.py -v   # needs GROQ_API_KEY in .env
RUN_EMBEDDING_TESTS=1 pytest tests/integration/test_retrieval_metrics.py -v   # downloads embedding model
RUN_NETWORK_URL_TESTS=1 pytest tests/integration/ -v   # also hits live URLs
```

**Coverage** is measured automatically on every `pytest` run (configured in `pytest.ini` / `.coveragerc`) and prints a per-file breakdown. From unit tests alone: **77% overall** — `config.py` and `hybrid_search.py` at 100%, `pipeline.py` at 92% (up from 6% before `test_pipeline.py` existed, by mocking out Groq/embeddings rather than needing them live), but `vectorstore.py` still at 20.4% since its real logic only runs against a live ChromaDB/embedding model, which the integration tests exercise (not counted in that baseline). Run `pytest tests/unit/ tests/integration/` together with a real `GROQ_API_KEY` to see the fuller picture.

**Automation:** `.github/workflows/backend-tests.yml` runs the unit tests automatically on every push and pull request via GitHub Actions — no setup needed once this repo is on GitHub, since unit tests need no secrets. The integration tests are deliberately left out of that workflow because they need a real Groq key; if you want those automated too, add `GROQ_API_KEY` as a repository secret (Settings → Secrets and variables → Actions) and extend the workflow to pass it through and run `pytest tests/integration/`.

## Roadmap

This started as a one-day build and has since grown well past the original three-day brief's scope:

* File parsing + crawl discovery — **Done**
* Groq relevance filter + full scraping of selected pages — **Done**
* Chunking + embeddings + ChromaDB storage/retrieval — **Done**, verified with real network access
* Query expansion (Groq rewrites vague instructions into more specific search queries) — **Done**
* Per-document retrieval with guaranteed coverage (no document silently dropped when multiple are uploaded) — **Done**
* Cross-encoder reranking, scored on short snippets to match the model's trained input length, multi-query aware — **Done**, verified with real network access
* Groq report generation + citation-aware PDF export — **Done**
* Persistent evaluation history + in-app run comparison — **Done**
* React frontend wired to all of the above — **Done**
* Automated test suite (unit + integration) with CI on every push — **Done**
* Intent router (`summary` / `targeted_qa` / `analysis`) directing retrieval strategy per request — **Done**, unit-tested; not yet run end-to-end with a live Groq key and real document
* Positional (section-aware) sampling for "summary" intent, instead of similarity search against a topic that doesn't exist for vague requests — **Done**, unit-tested
* Hybrid retrieval (BM25 keyword search + dense vector search) fused via Reciprocal Rank Fusion, for `targeted_qa`/`analysis` intents — **Done**, unit-tested
* Retrieval-confidence check demoted to informational-only after real evidence showed it didn't track actual output quality; groundedness established as the real quality gate — **Done**
* Query decomposition ("compositional" multi-hop): a compound instruction is split into independent sub-questions, each retrieved separately — **Done**, unit-tested. Not iterative multi-hop (see "On multi-hop" note in Project status above) — that's a real, acknowledged limitation, not an oversight.
* Self-correcting retry: if the first pass's groundedness comes back below threshold, the pipeline reformulates its search queries and retries retrieval + generation once, keeping whichever attempt scored higher — **Done**, unit-tested (only applies to `targeted_qa`/`analysis`; "summary"'s positional sampling doesn't depend on the search query, so retrying it wouldn't change anything)
* Precision@K/Recall@K/MRR run against a real, labeled retrieval (`tests/integration/test_retrieval_metrics.py`) — **Done**, closing a gap that was previously just an untested formula

**Still open, honestly:**
* The intent router, hybrid+RRF retrieval, query decomposition, self-correction retry, and span-coverage check are unit-tested against synthetic/mocked data but have not yet been exercised end-to-end against a real document with a live Groq key — that's the next real-world check to run.
* Groundedness is vocabulary overlap, not true entailment — a report could reuse a source's words while misstating their meaning and still pass. Self-correction inherits this: it optimizes for the same word-overlap number, so it can't catch a retry that's equally fluent but still wrong in substance.
* True iterative multi-hop (each retrieval step's results shaping the next query) is not implemented -- only compositional decomposition (see above). This mainly matters for cross-document questions that need chaining a fact found in one document to look something up in another.
* Some chunks' first `RERANK_SNIPPET_CHARS` characters can be page headers, figure captions, or table numbers rather than real content (seen on a real academic PDF) — this affects what the reranker actually scores. Trimming boilerplate before the snippet is taken would likely help; not yet done.
* The self-correction retry costs an extra Groq call (or several, with decomposition) plus a second retrieval pass whenever the first attempt is poorly grounded -- for a document genuinely unrelated to the instructions (like the deliberate-mismatch test scenario), this means paying for a retry that was never going to succeed. Not a bug, just a real cost/latency trade-off worth knowing about.

## Support

This is a personal practice project completed during an AI/ML internship. For questions, contact [your name / email].

## Contributing

This is a solo learning project and not currently open to external contributions. To run it locally for review, follow the Setup steps above, then work through the Smoke test to confirm each stage.

## Authors and acknowledgment

Built by Jayasundaram Gobinesh as part of an AI/ML internship practice project.

## License

Not currently licensed for public/open-source use — internal practice project.

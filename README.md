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
                                     Instructions ──► GROQ QUERY EXPANSION (a few more specific rewordings)
                                                                            │
                                     Expanded queries ──► PER-SOURCE oversampled ChromaDB retrieval
                                                          (every uploaded document/page gets its own pool --
                                                           prevents one document being silently dropped)
                                                                            │
                                               Candidate pool ──► CROSS-ENCODER RERANKING
                                                                    (cross-encoder/ms-marco-MiniLM-L-6-v2,
                                                                     scores a short snippet of each candidate
                                                                     against every expanded query, keeps each
                                                                     candidate's best score, per source)
                                                                            │
                                                              FINAL CHUNKS + source metadata
                                                                            │
                                                    GROQ REPORT GENERATION (openai/gpt-oss-120b)
                                                                            │
                                                      Grounded report + citations ──► REPORTLAB ──► PDF
                                                                            │
                                                                     REACT UI ──► Download
                                                                            │
                                                     Every run's evaluation ──► evaluation_log.jsonl
                                                     (viewable in the UI's "Run history" panel)
```

## Project status

Backend and frontend are both built, including query expansion, per-document coverage-preserving retrieval, cross-encoder reranking, and persistent run history. Every module has unit tests (70 as of this writing), including the FastAPI routes directly via `TestClient`. What's been verified with real network access (confirmed working by direct testing): the embedding model, the cross-encoder reranker, and Groq calls. What's still worth doing before fully trusting this on a new machine: `RERANK_SCORE_WARNING_THRESHOLD` (in `config.py`) has only ever been calibrated against *good* matches from real runs — it's never been tested against a genuinely irrelevant document/instruction pairing, which `tests/integration/test_pipeline_scenarios.py`'s `test_scenario_1_deliberately_mismatched_topic` exists specifically to do. **Run the smoke test below first** to confirm the environment works before relying on the full pipeline.

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
│   │   ├── main.py            # FastAPI app + endpoints (including /evaluation-history)
│   │   ├── config.py          # all tunable settings, env vars
│   │   ├── doc_parser.py      # PDF/DOCX/TXT -> text + metadata
│   │   ├── scraper.py         # crawl discovery + full-page scrape
│   │   ├── groq_client.py     # relevance filter + query expansion + report generation
│   │   ├── chunker.py         # token-bounded chunking with overlap, tags each chunk's source_name
│   │   ├── vectorstore.py     # ChromaDB + embeddings, per-source oversampled retrieval
│   │   ├── reranker.py        # cross-encoder reranking, multi-query, graceful fallback
│   │   ├── pdf_export.py      # ReportLab PDF generation + Unicode sanitization
│   │   ├── evaluation.py      # validation checks per stage + persistent run logging
│   │   └── pipeline.py        # orchestrates all stages in order
│   └── tests/
│       ├── conftest.py        # shared fixtures (sample PDF/DOCX/TXT generation)
│       ├── unit/               # no network needed, runs in seconds
│       └── integration/        # needs GROQ_API_KEY + (for 2 tests) internet
└── frontend/
    ├── package.json
    ├── index.html
    └── src/
        ├── App.jsx             # form, results, evaluation table, run history panel
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

Then open `.env` and add your Groq API key (free — see below). Run the server:

```bash
uvicorn app.main:app --reload
```

This starts the API at `http://127.0.0.1:8000`. Check `http://127.0.0.1:8000/health` — it should return `{"status": "ok"}`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

This opens the app at `http://localhost:5173`. It talks to the backend at `http://127.0.0.1:8000` by default (change `API_BASE` at the top of `src/App.jsx` if you run the backend elsewhere).

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
| Retrieval — confidence | Prefers the cross-encoder's rerank score when reranking succeeded (falls back to embedding distance if the reranker was unavailable) — average score/distance is stored as a numeric value for comparison across runs |
| Retrieval — source coverage | With multiple uploaded documents, confirms every one of them actually contributed at least one retrieved chunk — catches a document being silently dropped |
| Retrieval — Precision@K/Recall@K/MRR | Computed via `evaluate_retrieval()` against a labeled test set (not run automatically on a live query, since there's no ground truth for an arbitrary user question) |
| Generation | Citation markers present, every cited source was actually retrieved (citation accuracy), and groundedness — the percentage of the report's significant vocabulary that actually appears in its cited sources (not just "cited something") |
| PDF export | The text actually written to the PDF (after sanitization) contains no characters common fonts can't render — checked against the real rendered output, not the raw source text |
| System | Processing time per stage, error count |

**Run history**: every `/process` call appends its evaluation summary (with numeric values, not just pass/fail) to `backend/evaluation_log.jsonl`, viewable via the "Run history" panel in the UI or the `/evaluation-history` endpoint directly. This is what makes before/after comparisons (e.g. after changing a threshold or retrieval strategy) an actual query instead of screenshots.

## Testing

Two kinds of tests, in `backend/tests/`:

- **`tests/unit/`** — no network, no Groq key, runs in under a second. Tests document parsing, chunking (including real overlap between consecutive chunks, not just config), scraper HTML logic, and every evaluation check with both a passing and a genuinely failing case.
- **`tests/integration/`** — runs the real pipeline against real Groq calls and embeddings. Needs `GROQ_API_KEY` set in `.env`; skips automatically (not an error) if it's missing. Implements deliberate scenario tests: a topic mismatch that should fail retrieval/groundedness on purpose, an image-only PDF that should fail extraction cleanly instead of crashing, and a tiny input smaller than the top-k retrieval count. Two further scenarios (an irrelevant seed URL, a web-only run with no file) hit live URLs and only run if you set `RUN_NETWORK_URL_TESTS=1`, since they're slower and depend on an external page staying up.

Run everything:

```bash
cd backend
pip install -r requirements-dev.txt
pytest tests/unit/ -v          # fast, always safe to run
pytest tests/integration/ -v   # needs GROQ_API_KEY in .env
RUN_NETWORK_URL_TESTS=1 pytest tests/integration/ -v   # also hits live URLs
```

**Coverage** is measured automatically on every `pytest` run (configured in `pytest.ini` / `.coveragerc`) and prints a per-file breakdown. Baseline from unit tests alone: **61.8% overall** — `chunker.py` and `config.py` at 100%, but `pipeline.py` at just 7.9% and `vectorstore.py` at 20.4%, since their real logic only runs against live ChromaDB/embeddings/Groq calls, which the integration tests exercise (not counted in that 61.8% baseline). Run `pytest tests/unit/ tests/integration/` together with a real `GROQ_API_KEY` to see the fuller picture.

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

**Still open, honestly:**
* `RERANK_SCORE_WARNING_THRESHOLD` has only ever been calibrated against good matches — never against a confirmed-bad one. `test_scenario_1_deliberately_mismatched_topic` in `tests/integration/` exists specifically to close this gap; run it.
* Retrieval Precision@K/Recall@K/MRR (`evaluate_retrieval()`) exists but has no real labeled test set behind it yet.
* Groundedness is vocabulary overlap, not true entailment — a report could reuse a source's words while misstating their meaning and still pass.

## Support

This is a personal practice project completed during an AI/ML internship. For questions, contact [your name / email].

## Contributing

This is a solo learning project and not currently open to external contributions. To run it locally for review, follow the Setup steps above, then work through the Smoke test to confirm each stage.

## Authors and acknowledgment

Built by Jayasundaram Gobinesh as part of an AI/ML internship practice project.

## License

Not currently licensed for public/open-source use — internal practice project.

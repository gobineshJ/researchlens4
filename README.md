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
                                              Candidate pages ──► GROQ RELEVANCE FILTER (llama-3.1-8b-instant)
                                                                            │
                                                        selected pages ──► full scrape
                                                                            │
              Extracted file text  +  Scraped web text ──► CHUNKING (500-800 tokens, 10-20% overlap)
                                                                            │
                                          CHUNKS ──► EMBEDDING (sentence-transformers, all-MiniLM-L6-v2)
                                                                            │
                                                               ──► CHROMADB (vectors + metadata)
                                                                            │
                                       User query/instructions ──► QUERY EMBEDDING ──► ChromaDB retrieval
                                                                            │
                                                              TOP-K CHUNKS (k=5) + source metadata
                                                                            │
                                                    GROQ REPORT GENERATION (llama-3.3-70b-versatile)
                                                                            │
                                                      Grounded report + citations ──► REPORTLAB ──► PDF
                                                                            │
                                                                     REACT UI ──► Download
```

## Project status

Backend and frontend are both built. Every module has been tested individually (extraction, chunking, crawl-discovery logic, PDF export, evaluation checks, FastAPI routing). The one piece not yet confirmed end-to-end is the embedding step (`sentence-transformers` downloading its model from Hugging Face) — this needs normal internet access to run, which the environment this was built in didn't have. **Run the smoke test below first** to confirm it on your machine before relying on the full pipeline.

## File structure

```
researchlens/
├── .gitignore
├── README.md
├── backend/
│   ├── requirements.txt
│   ├── .env.example
│   ├── app/
│   │   ├── main.py            # FastAPI app + endpoints
│   │   ├── config.py          # all tunable settings, env vars
│   │   ├── doc_parser.py      # PDF/DOCX/TXT -> text + metadata
│   │   ├── scraper.py         # crawl discovery + full-page scrape
│   │   ├── groq_client.py     # relevance filter + report generation
│   │   ├── chunker.py         # token-bounded chunking with overlap
│   │   ├── vectorstore.py     # ChromaDB + embeddings
│   │   ├── pdf_export.py      # ReportLab PDF generation
│   │   ├── evaluation.py      # validation checks per pipeline stage
│   │   └── pipeline.py        # orchestrates all stages in order
│   └── tests/
└── frontend/
    ├── package.json
    ├── index.html
    └── src/
        ├── App.jsx             # entire UI: form, results, evaluation table
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

Each pipeline stage has a real check, not just a "looks fine" assumption:

| Stage | What's checked |
|---|---|
| Extraction | Every record has non-empty text and required metadata (filename+location, or url+title) |
| Chunking | Chunk sizes stay within the configured token limit; overlap is actually applied |
| Crawler | Depth limit respected, all URLs on the seed domain, zero duplicates |
| Retrieval | Precision@K, Recall@K, MRR — computed via `evaluate_retrieval()` against a labeled test set (not run automatically on a live query, since there's no ground truth for an arbitrary user question) |
| Generation | Citation markers present in the report, every cited source was actually retrieved (citation accuracy), report cites at least one source (groundedness) |
| System | Processing time per stage, error count — both surfaced in the evaluation panel |

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

**Automation:** `.github/workflows/backend-tests.yml` runs the unit tests automatically on every push and pull request via GitHub Actions — no setup needed once this repo is on GitHub, since unit tests need no secrets. The integration tests are deliberately left out of that workflow because they need a real Groq key; if you want those automated too, add `GROQ_API_KEY` as a repository secret (Settings → Secrets and variables → Actions) and extend the workflow to pass it through and run `pytest tests/integration/`.

## Roadmap

This was built in one focused session rather than day-by-day, but functionally covers what the three-day brief asked for:

* File parsing + crawl discovery — **Done**
* Groq relevance filter + full scraping of selected pages — **Done**
* Chunking + embeddings + ChromaDB storage/retrieval — **Done** (retrieval untested end-to-end due to sandbox network limits — see Project status)
* Groq report generation + citation-aware PDF export — **Done**
* React frontend wired to all of the above — **Done**

## Support

This is a personal practice project completed during an AI/ML internship. For questions, contact [your name / email].

## Contributing

This is a solo learning project and not currently open to external contributions. To run it locally for review, follow the Setup steps above, then work through the Smoke test to confirm each stage.

## Authors and acknowledgment

Built by Jayasundaram Gobinesh as part of an AI/ML internship practice project.

## License

Not currently licensed for public/open-source use — internal practice project.

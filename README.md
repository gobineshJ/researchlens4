# ResearchLens

## Description

A RAG (Retrieval-Augmented Generation) research paper summarizer, built as a AI/ML intern training project. It combines evidence from two sources into one citation-aware report:

* Uploaded research documents (PDF, DOCX, TXT)
* A bounded web crawl starting from user-supplied reference URLs — never open-ended search

The crawler never does open-ended web search — every crawl starts from a URL you supply, follows links a bounded distance, and a Groq-based relevance filter decides which discovered pages are actually worth scraping before any full extraction happens.

Stack: **React** (frontend) + **FastAPI** (backend) + **Celery + Redis** (background pipeline runs) + **ChromaDB** (vector store) + **Groq** (LLM filtering, figure description and generation). Every API endpoint except the health check is protected by **OAuth2 password-flow login with JWT bearer tokens**.

## Architecture

```
USER ──► LOGIN SCREEN ──► POST /auth/token (OAuth2 password flow) ──► JWT bearer token
  │                         (every later request carries "Authorization: Bearer <token>")
  │
  ├── POST /process ──► files saved to job_uploads/<job_id>/ ──► job queued on REDIS ──► returns {job_id}
  │                                                                     │
  │                                                     CELERY WORKER runs the pipeline below,
  │                                                     publishing progress ("Describing Figure 3...")
  │                                                                     │
  ├── GET /jobs/{job_id} every 2s ◄── queued / running + progress / done + result / error
  │
  │   ─── the pipeline (runs on the worker) ───
  │
  ├── Research files (PDF/DOCX/TXT) ──► EXTRACTION (PyMuPDF, python-docx)
  │                                      PDF figures ──► rendered to PNG ──► GROQ VISION (qwen/qwen3.8-27b)
  │                                      ──► text description, cited as "paper.pdf, page 7, Figure 3"
  ├── Seed URLs ─────────────────────► CRAWLER (Requests + BeautifulSoup) ──► Candidate pages
  │                                      (every fetch checked against the site's robots.txt;
  │                                       Crawl-delay honoured)
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
                                                                     -- page headers, captions, table-number
                                                                     rows stripped first --
                                                                     against every expanded query, keeps each
                                                                     candidate's best score, per source.
                                                                     "targeted_qa"/"analysis" ONLY -- a
                                                                     positional sample has no query to rank
                                                                     against, and reordering it by relevance
                                                                     is what let the budget trim drop a
                                                                     document's opening chunks)
                                                                            │
                              targeted_qa/analysis: ITERATIVE MULTI-HOP (openai/gpt-oss-20b)
                              reads the evidence so far; if the question needs a fact that
                              evidence names but doesn't contain, retrieves for it as the next
                              hop (up to 2 follow-up hops) and merges the results in
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
                                                    GROQ REPORT GENERATION (openai/gpt-oss-120b),
                                                    streamed: the UI shows the report as it's written
                                                                            │
                            groundedness < 30%? ──► SELF-CORRECTION RETRY: reformulate queries
                            (targeted_qa/analysis    (avoiding the ones already tried), retrieve +
                             only, one retry max)     generate again, keep whichever attempt scored higher
                                                                            │
                              ENTAILMENT CHECK (LLM judge): each cited sentence vs. the excerpt
                              it cites -- supported / unsupported / contradicted
                                                                            │
                              TARGETED REVISION: only the flagged sentences are rewritten against
                              their own excerpts (or removed), spliced back in, and judged again
                                                                            │
                                                      Grounded report + citations ──► REPORTLAB ──► PDF
                                                                            │
                                                     REACT UI ──► report with clickable [n] citations (each opens
                                                                  the exact excerpt); Download (fetched with the bearer
                                                                  token; only the user who ran the job can download it)
                                                                            │
                                                     Every run's evaluation ──► evaluation_log.jsonl
                                                     (viewable in the UI's "Run history" panel --
                                                      each user sees only their own runs)
```

## Project status

Backend and frontend are both built, including an intent router, query decomposition (compositional multi-hop), hybrid (BM25 + vector) retrieval fused with Reciprocal Rank Fusion, section-aware summarization, cross-encoder reranking, a self-correction retry loop, and persistent run history. Every module has unit tests (240 as of this writing), including the FastAPI routes directly via `TestClient` and the pipeline's own orchestration logic (query decomposition wiring, self-correction retry) via mocked Groq/embedding boundaries. What's been verified with real network access (confirmed working by direct testing): the embedding model, the cross-encoder reranker, and Groq calls. The intent router, hybrid retrieval, RRF fusion, query decomposition, and self-correction retry are new and unit-tested with mocked/synthetic data, but have not yet been run against a live document end-to-end with a real Groq key — that's the next thing to confirm on a machine with network access.

What's still worth doing before fully trusting this on a new machine: `RERANK_SCORE_WARNING_THRESHOLD`-based retrieval confidence (in `config.py`/`evaluation.py`) is now **informational only** — across three real comparison runs it tracked the wrong direction relative to actual report quality (it failed the best run and passed a weaker one), so it no longer gates the report. `validate_generation`'s groundedness check (vocabulary overlap between the report and its cited sources) is the real quality gate now, and was correct in all three of those comparisons; it's also what the new self-correction retry decides on. `tests/integration/test_pipeline_scenarios.py`'s `test_scenario_1_deliberately_mismatched_topic` exists specifically to exercise this with a real document, and `tests/integration/test_retrieval_metrics.py` (new -- see Testing below) runs Precision@K/Recall@K/MRR against real, labeled retrieval for the first time. **Run the smoke test below first** to confirm the environment works before relying on the full pipeline.

**Login, background jobs and figures** (latest additions): the app opens on a login screen (OAuth2 password flow, bcrypt-hashed passwords, 15-minute JWTs kept only in memory, renewed through a rotating refresh token in an httpOnly cookie), every endpoint except `/health` requires a bearer token, and users only see their own jobs and reports. `/process` no longer runs the pipeline inside the HTTP request: it queues a Celery job and the UI polls for the result, showing live progress (current stage, which figure is being described, rate-limit waits). Figures in uploaded PDFs are rendered and described by Groq's vision model. Verified end to end with Redis and a Celery worker on Windows (`--pool=solo`).

**On "multi-hop":** two kinds are implemented. Query *decomposition* splits a compound instruction into independent sub-questions, each retrieved on its own. *Iterative* multi-hop then chains facts: after each retrieval pass, Groq reads what was found and, if the question needs a fact that evidence points to but doesn't contain, writes follow-up queries naming it and retrieves them as the next hop. "What dataset did the method that beat the 2019 baseline use?" works this way: hop 1 finds the method's name, hop 2 looks up its dataset. There are at most `MULTI_HOP_MAX_EXTRA_HOPS` (2) follow-up hops, and the loop stops as soon as the evidence is judged sufficient. Each hop is recorded in the evaluation panel.

**Several jobs at once:** on Linux, macOS or WSL the Celery worker runs a prefork pool of `CELERY_WORKER_CONCURRENCY` processes (default 2), so that many reports run in parallel. Groq rate-limit pacing state lives in Redis, shared by every process, and each request reserves its cost as it is sent, so parallel workers don't all spend the same free budget and hit 429s. Native Windows still runs one job at a time per worker (solo pool).

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
│   ├── researchlens.db        # created on first run; gitignored -- user accounts + job ownership (SQLite)
│   ├── figure_cache.json      # created on first run; gitignored -- figure descriptions, so each is only paid for once
│   ├── job_uploads/           # gitignored -- uploaded files waiting for a worker; deleted when the job finishes
│   ├── app/
│   │   ├── main.py            # FastAPI app + endpoints; all but /health behind bearer-token auth
│   │   ├── auth.py            # OAuth2 password flow: login/register, JWT issue + validation
│   │   ├── db.py              # SQLite: user accounts + which user owns which job
│   │   ├── create_user.py     # `python -m app.create_user <name>` CLI
│   │   ├── celery_app.py      # Celery app + the task that runs the pipeline on a worker
│   │   ├── progress.py        # stage/detail progress updates, published by the Celery task for the UI
│   │   ├── config.py          # all tunable settings, env vars
│   │   ├── doc_parser.py      # PDF/DOCX/TXT -> text + metadata; PDF figures rendered and described
│   │   ├── scraper.py         # crawl discovery + full-page scrape; robots.txt + Crawl-delay
│   │   ├── groq_client.py     # relevance filter + intent classification + query decomposition + query expansion + multi-hop planning + figure description + report generation + entailment judge; rate-limit pacing (shared via Redis) + 429 retry
│   │   ├── chunker.py         # token-bounded chunking with overlap; also positional sampling for "summary" intent
│   │   ├── vectorstore.py     # ChromaDB (in-memory per run, dropped afterwards) + embeddings, per-source oversampled retrieval
│   │   ├── hybrid_search.py   # BM25 keyword search + Reciprocal Rank Fusion (RRF) with vector search
│   │   ├── reranker.py        # cross-encoder reranking, multi-query, boilerplate-stripped snippets, graceful fallback
│   │   ├── pdf_export.py      # ReportLab PDF generation + Unicode sanitization
│   │   ├── evaluation.py      # validation checks per stage + persistent run logging
│   │   └── pipeline.py        # orchestrates all stages, including iterative multi-hop retrieval and the self-correction retry loop
│   ├── eval/                  # measuring quality on a golden set (see "Measuring quality" below)
│   │   ├── run_eval.py        # runs the real pipeline on every golden question and scores it
│   │   ├── judge_agreement.py # how often the entailment judge agrees with human labels
│   │   ├── export_claims.py   # turns claims from your own runs into a file to label
│   │   ├── metrics.py         # the scoring functions (unit-tested)
│   │   ├── data/golden/       # 4 fictional documents + 20 questions with reference answers and evidence
│   │   └── data/judge_labels.jsonl  # 40 labelled claims for judge agreement
│   ├── Dockerfile             # one image for the API and the worker
│   └── tests/
│       ├── conftest.py        # shared fixtures (sample PDF/DOCX/TXT generation)
│       ├── unit/               # no network needed, runs in seconds (includes pipeline orchestration tests, mocked)
│       └── integration/        # needs GROQ_API_KEY + (for some tests) internet -- see Testing below
├── docker-compose.yml          # redis + api + worker + frontend in one command
└── frontend/
    ├── package.json
    ├── index.html
    ├── vite.config.js          # dev proxy: /api -> FastAPI
    ├── Dockerfile, nginx.conf  # production build served by nginx, which proxies /api
    └── src/
        ├── App.jsx             # auth gate: restores the session from the refresh cookie
        ├── LoginScreen.jsx     # login / create-account form
        ├── api.js              # in-memory access token, silent refresh, authenticated fetch, downloads
        ├── styles.js
        └── components/
            ├── Workspace.jsx       # upload form + job polling
            ├── ProgressBox.jsx     # live stage, detail and the report as it streams in
            ├── ResultsPanel.jsx    # counts, report, download
            ├── ReportView.jsx      # report text with clickable [n] citations -> source excerpt
            ├── ClaimVerdicts.jsx   # flagged and revised sentences
            ├── EvaluationPanel.jsx # every check, timings, errors
            └── HistoryPanel.jsx    # your past runs
```

## Setup

### Quickest: Docker Compose

With Docker installed and `backend/.env` filled in (at least `GROQ_API_KEY` and `JWT_SECRET_KEY`, see below):

```bash
docker compose up --build
```

Open `http://localhost:8080`. This starts Redis, the API, a worker running `CELERY_WORKER_CONCURRENCY` jobs at once, and the frontend behind nginx. The embedding and reranker models are baked into the image. Accounts, reports and run history live in the `app-data` volume. `docker compose down -v` wipes them. Swagger is still at `http://localhost:8000/docs`.

The manual setup below is for development.

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

Pipeline runs execute on a **Celery** worker, with **Redis** as the broker (and as the shared store for Groq rate-limit pacing). Start Redis first:

* **Windows (WSL):** `wsl`, then once: `sudo apt update && sudo apt install -y redis-server`; after that just `redis-server` (leave that window open).
* **Docker:** `docker run -p 6379:6379 redis`
* **Linux/macOS:** your package manager's `redis-server`.

Then run the API and a worker in two more terminals (both from `backend/`, venv activated):

```bash
uvicorn app.main:app --reload
celery -A app.celery_app worker --loglevel=info
```

The worker picks its pool for the platform: **solo** on native Windows (one job at a time; prefork doesn't work there) and **prefork** with `CELERY_WORKER_CONCURRENCY` processes (default 2) everywhere else. To run several jobs at once from a Windows machine, run the worker inside **WSL**. Use a separate venv inside WSL (a Windows venv won't run there) and point it at the same `backend/` folder:

```bash
# inside WSL, from /mnt/c/.../researchlens/backend
python3 -m venv .venv-linux && source .venv-linux/bin/activate
pip install -r requirements.txt
CELERY_WORKER_CONCURRENCY=3 celery -A app.celery_app worker --loglevel=info
```

Each worker process loads its own embedding and reranker models (a few hundred MB of RAM each), so size the concurrency to your memory. `--pool` / `--concurrency` on the command line still override.

The worker does **not** reload on code changes the way `uvicorn --reload` does -- restart it (Ctrl+C, run again) after changing anything under `app/`.

This starts the API at `http://127.0.0.1:8000`. Check `http://127.0.0.1:8000/health` — it should return `{"status": "ok"}`.

No Redis? Set `CELERY_TASK_ALWAYS_EAGER=true` in `.env` and skip the worker: the pipeline then runs inside the API process, and `/process` blocks until the report is done.

### Authentication

Every endpoint except `/health`, `/auth/token` and `/auth/register` needs an `Authorization: Bearer <token>` header. Tokens come from the OAuth2 password flow:

```bash
curl -X POST http://127.0.0.1:8000/auth/token -d "username=alice&password=..."
# -> {"access_token": "eyJ...", "token_type": "bearer"}
```

The same response sets a **refresh token** in an httpOnly, `SameSite=Strict` cookie. Access tokens last 15 minutes (`ACCESS_TOKEN_EXPIRE_MINUTES`). The frontend keeps them only in memory, never in `localStorage`, and silently swaps the cookie for a new one through `POST /auth/refresh` when one expires or the page reloads. Refresh tokens are stored hashed and rotate on every use. Presenting an already-used one (a sign it was copied) ends that whole login. `POST /auth/logout` revokes it. Sessions last `REFRESH_TOKEN_EXPIRE_DAYS` (7) without activity. Set `COOKIE_SECURE=true` when serving over HTTPS. CORS only admits `CORS_ORIGINS`.

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

This opens the app at `http://localhost:5173`. The app calls `/api/...` on its own origin and Vite's dev server forwards that to the backend at `http://127.0.0.1:8000` (set `API_TARGET` to change it). Going through the same origin is what lets the refresh cookie work without any cross-site rules. If you'd rather call the API directly, set `VITE_API_BASE` and add the frontend's origin to `CORS_ORIGINS`.

### Getting a free Groq API key

Groq's API is genuinely free — no credit card:

1. Go to **console.groq.com** and sign up (email or Google/GitHub).
2. Go to **console.groq.com/keys** → "Create API Key" → copy it (starts with `gsk_`).
3. Paste it into `backend/.env` as `GROQ_API_KEY=gsk_...`.

Free tier gives ~30 requests/minute and ~14,400 requests/day — enough for this project.

**Figures and the free tier:** the vision model's free tier allows ~7,000 input tokens per minute and Groq charges ~2,700 per figure, so only about 3 figures can be described per minute. A paper with many figures therefore takes a couple of minutes on its first run (the UI shows "Waiting Ns for Groq's per-minute token limit" while it paces itself); descriptions are cached in `figure_cache.json`, so later runs of the same PDF skip that. Set `FIGURE_CAPTIONING_ENABLED=false` in `.env` to skip figure descriptions entirely, or upgrade to Groq's paid Dev tier for higher limits.

## Smoke test (run this first)

Before trusting the full pipeline, confirm each layer works on your machine, in this order:

1. **Backend health check** — with the backend running, open `http://127.0.0.1:8000/health` in a browser. Expect `{"status": "ok"}`.
2. **Worker check** — the Celery worker's terminal shows `ready.` and lists the task `researchlens.run_pipeline`. (Skip this if you set `CELERY_TASK_ALWAYS_EAGER=true`.)
3. **Login** — open the frontend; you should see the login screen, not the app. Use "Create one" to make an account; you land in the app logged in.
4. **Embedding model download** — the first `/process` call downloads `all-MiniLM-L6-v2` from Hugging Face (a few hundred MB). This needs normal internet access and may take a minute the first time only; it's cached after that. If this step fails with a connection error, check your internet connection — nothing else in the code is the cause.
5. **Full run** — in the frontend, upload one small PDF or DOCX, leave the URL box empty, type any instruction ("summarize this"), click Process. The status box should move through the stages ("Reading your documents…", "Writing the report…"), and the worker's terminal logs the run. With no seed URLs, the crawler/filter/scrape stages are skipped automatically, so this isolates: extraction → chunking → embedding → storage → retrieval → Groq generation → PDF. If this works, add a real seed URL (e.g. a Wikipedia page) next to bring the crawler and relevance filter into the run.
6. **Check the evaluation panel** — the results screen shows a table of pass/fail checks per stage, plus timings. A run isn't "working" just because it didn't crash — check that the evaluation checks actually passed too.

## Evaluation / validation

Each pipeline stage has a real check, not just a "looks fine" assumption. Every check's numeric value (not just pass/fail) is stored, so results are comparable across runs, not just per-run:

| Stage | What's checked |
|---|---|
| Extraction | Every record has non-empty text and required metadata (filename+location, or url+title) |
| Extraction — figure descriptions | **Only when PDFs contain captioned figures.** How many were described by the vision model vs. kept caption-only. Fails if any couldn't be described (no API key, a request that still failed after rate-limit retries, or over `FIGURE_MAX_PER_DOCUMENT`) — the backend log says which figure and why |
| Chunking | Chunk sizes stay within the configured token limit; overlap is actually applied between consecutive chunks (checked on real produced output, not just the config) |
| Crawler | Depth limit respected, all URLs on the seed domain, zero duplicates. (Pages the site's robots.txt disallows are never fetched at all, so they never reach this check.) |
| Retrieval — intent | Which of `summary` / `targeted_qa` / `analysis` the instructions were classified as, and which retrieval strategy that triggered — always "passes"; this is a transparency record, not a gate |
| Retrieval — query decomposition | **Only for `targeted_qa`/`analysis`.** Whether the instruction was split into independent sub-questions or treated as a single focused question, and what those sub-questions were — a transparency record, not a gate |
| Retrieval — multi-hop | **Only for `targeted_qa`/`analysis`.** How many hops retrieval took, and for each follow-up hop what it looked up, with which queries, and how many new chunks it found. A transparency record, not a gate |
| Retrieval — confidence | **Informational only.** Reports the average cross-encoder rerank score (or embedding distance, if the reranker's fallback path ran) as a numeric value, but never fails the report — see Project status above for why |
| Retrieval — source coverage | With multiple uploaded documents, confirms every one of them actually contributed at least one retrieved chunk — catches a document being silently dropped |
| Retrieval — span coverage | **Only for `summary` intent.** Confirms the positionally-sampled chunks actually spread across at least half of each document's position range (or cover nearly everything, for a short document) — catches positional sampling silently degrading into a narrow cluster |
| Retrieval — Precision@K/Recall@K/MRR | Computed via `evaluate_retrieval()`. Previously only unit-tested as a metric formula; now also run against a real, labeled retrieval in `tests/integration/test_retrieval_metrics.py` (three topically distinct synthetic documents, so "relevant" is unambiguous ground truth) — not run automatically on a live query, since there's no ground truth for an arbitrary user question |
| Generation — self-correction | **Only fires for `targeted_qa`/`analysis` when the first pass's groundedness came back below threshold.** Records the groundedness before and after one retry with reformulated queries, and whether the retry's result was adopted — always "passes"; a transparency record of whether/how self-correction acted, not a gate itself |
| Generation | Citation markers present, every cited source was actually retrieved (citation accuracy), and **groundedness** — the percentage of the report's significant vocabulary that actually appears in its cited sources (not just "cited something"). This is the real pass/fail quality gate for retrieval+generation as a whole, and what the self-correction retry itself decides on. |
| Generation — entailment | An LLM judge (`GROQ_JUDGE_MODEL`) checks each sentence that carries a `[n]` marker against the excerpt(s) it cites: **supported**, **unsupported** (adds, overstates or changes something) or **contradicted**. Fails if fewer than `ENTAILMENT_THRESHOLD` (70%) of judged claims are supported, or if any claim contradicts its source. The UI's "Claim check" lists the flagged sentences. Unlike groundedness, this catches a report that reuses a source's words but misstates them. If the judge can't run (e.g. Groq's daily limit), it is recorded as an error, not as a pass or a fail |
| Generation — claim revision | **Only when the entailment check flags sentences.** Those sentences, and only those, are rewritten against the excerpts they cite (or removed if nothing supports them), put back in place with their original citation markers, and judged again. The rest of the report is untouched. The entailment check then scores the revised report, and the UI's Claim check shows each original next to its rewrite. If the revision request fails, the original report is kept and an error is recorded. A transparency record, not a gate |
| PDF export | The text actually written to the PDF (after sanitization) contains no characters common fonts can't render — checked against the real rendered output, not the raw source text |
| System | Processing time per stage, error count |

**Run history**: every `/process` call appends its evaluation summary (with numeric values, not just pass/fail) to `backend/evaluation_log.jsonl`, tagged with the user who ran it and with the job id as its `run_id`. It is viewable via the "Run history" panel in the UI or the `/evaluation-history` endpoint, and each user only sees their own runs. Entries logged before this change have no user and are therefore hidden from everyone. They are still in the file. This is what makes before/after comparisons (e.g. after changing a threshold or retrieval strategy) an actual query instead of screenshots.

## Measuring quality

The per-run checks above say whether *one* report looks trustworthy. `backend/eval/` answers the bigger question: is the system getting better or worse when something changes?

**Golden set** (`eval/data/golden/`): four short **fictional** documents (a research paper, a dataset datasheet, a survey and a hospital deployment report) whose facts chain across documents, plus 20 questions. Fictional on purpose: the model can't answer from memory, so a right answer means retrieval worked. Each question has a reference answer, the exact evidence passages, and the facts a correct answer must state. The types are `targeted_qa`, `multi_hop` (needs facts from two or three documents in sequence), `analysis`, `compound`, `summary` and `unanswerable` (the right answer is "the documents don't say"). A unit test checks the set itself: every evidence quote really is in its document, and every multi-hop question spans at least two.

```bash
cd backend
python -m eval.run_eval --judge-answers                                   # whole set, about 20 min on the free tier
python -m eval.run_eval --only mh- qa-corpus                              # a subset, by id prefix
python -m eval.run_eval --set MULTI_HOP_ENABLED=false --tag no-multihop   # ablation: turn a feature off
python -m eval.run_eval --compare eval/results/A.json eval/results/B.json # before/after table
python -m eval.run_eval --retry-incomplete eval/results/A.json            # rerun questions the daily limit cut off
```

If a Groq call inside a question fails (in practice, the free tier's daily limit), the question is marked **incomplete** and kept out of the averages, so "the quota ran out" is never mixed with "the system got it wrong". The run stops at that point instead of failing the remaining questions one by one. `--retry-incomplete` reruns just those questions later, with the same settings, and saves a merged file. `--compare` warns when two runs averaged different sets of questions.

Each question is scored on `evidence_recall` (did the gold passages reach the report's context), `evidence_mrr`, `fact_recall` (required facts stated), `answer_score` (LLM-graded against the reference, with `--judge-answers`), plus the pipeline's own entailment and groundedness, hops taken, Groq calls and seconds. Results go to `eval/results/`. The whole set uses a noticeable part of the free tier's daily token allowance, so use `--only` for quick checks.

First live run (4 questions, `--only qa-corpus mh-countries sum-datasheet un-radiology`):

| question type | evidence recall | fact recall | answer score | entailment | Groq calls |
|---|---|---|---|---|---|
| targeted_qa | 1.00 | 1.00 | 1.00 | 100% | 7 |
| multi_hop (3 documents chained) | 1.00 | 1.00 | 1.00 | 100% | 7 |
| summary | — | 0.75 | 1.00 | 100% | 4 |
| unanswerable | — | — | 1.00 (said "not reported") | 100% | 7 |

That's 4 questions, not a benchmark. Run the full set to get numbers worth quoting. One finding already: the hop planner took a second hop even on the simple lookup question, which is an extra call you could tune away.

**Can the entailment judge be trusted?** `python -m eval.judge_agreement` runs the real judge on `eval/data/judge_labels.jsonl`. That's 40 claims, each labelled against its excerpt, covering verbatim and paraphrased claims plus the failure modes that matter: wrong numbers, swapped figures, wrong attribution, overstatement, added detail, title-only scope and reversed facts. First run:

| | result |
|---|---|
| supported vs. flagged: accuracy / Cohen's κ | 1.00 / 1.00 |
| bad claims caught (flag recall) | 20 of 20 |
| flags that were right (flag precision) | 20 of 20 |
| 3-way accuracy / κ | 0.80 / 0.69 |

All 8 three-way disagreements are one pattern: where the label said "unsupported" for a wrong number or name, the judge said "contradicted". That's arguably correct, and it doesn't change the pass/fail outcome, since both count as flagged. Caveats: the starter labels were written along with this tool, not by an independent annotator, and synthetic claims are easier than claims from real reports, so treat this as an upper bound. To measure it on your own reports, run `python -m eval.export_claims`. It turns claims from your recent runs into a file with empty labels. Label them yourself, then run `python -m eval.judge_agreement --labels eval/data/my_labels.jsonl`.

## Testing

Two kinds of tests, in `backend/tests/`:

- **`tests/unit/`** — no network, no Groq key, runs in seconds. Tests document parsing, chunking (including real overlap between consecutive chunks and positional sampling for "summary" intent, not just config), BM25 search and RRF fusion (`hybrid_search.py`) with synthetic corpora, intent classification and query decomposition's fallback behavior, scraper HTML logic, every evaluation check with both a passing and a genuinely failing case, login and token validation (`test_auth.py`: wrong passwords, expired/forged tokens, deleted users, registration rules), the protected routes and job flow (`test_main.py`: every route rejects missing tokens, jobs are enqueued without needing Redis, users can't see each other's jobs or reports), Groq rate-limit pacing and 429 retries (`test_groq_retry.py`, with a fake clock), progress reporting (`test_progress.py`), figure detection (`test_figures.py`), robots.txt handling (`test_robots.py`), boilerplate stripping for reranker snippets (`test_rerank_snippet.py`), claim extraction and the entailment judge (`test_entailment.py`), iterative multi-hop retrieval (`test_multi_hop.py`), rate-limit pacing shared across processes plus concurrent figure-cache writes (`test_shared_pacing.py`), refresh-token rotation, reuse detection and CORS (`test_refresh_tokens.py`), targeted revision and splicing (`test_claim_revision.py`), the eval harness and golden-set consistency (`test_eval_metrics.py`), and `test_pipeline.py` — the pipeline's own orchestration logic (does decomposition get skipped for "summary" intent, does the self-correction retry actually fire and adopt the better result) with every Groq/embedding call mocked, so it's testing the WIRING, not re-testing the mocked-out model behavior.
  `tests/unit/conftest.py` makes any real request to Groq fail the test, so a forgotten mock can't quietly spend your quota or depend on the live model.
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

**Coverage** is measured automatically on every `pytest` run (configured in `pytest.ini` / `.coveragerc`) and prints a per-file breakdown. From unit tests alone: **88% overall** (376 tests; was 85% / 240 before robots.txt, entailment, multi-hop, shared pacing, refresh tokens, revision and the eval harness) — `config.py` and `hybrid_search.py` at 100%, `pipeline.py` at 92% (up from 6% before `test_pipeline.py` existed, by mocking out Groq/embeddings rather than needing them live), but `vectorstore.py` still at 20.4% since its real logic only runs against a live ChromaDB/embedding model, which the integration tests exercise (not counted in that baseline). Run `pytest tests/unit/ tests/integration/` together with a real `GROQ_API_KEY` to see the fuller picture.

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

* Background pipeline runs on a Celery worker (Redis broker), with job polling and live progress in the UI — **Done**, verified with a real worker on Windows
* Login screen + OAuth2 password flow with JWT bearer tokens protecting every endpoint; per-user job and report access — **Done**, unit-tested and verified in the browser
* PDF figures rendered and described by Groq's vision model, cached per image — **Done**
* Groq rate limits handled: requests paced against the model's per-minute budget (including the vision model's input-token limit, learned from Groq's 429 message), remaining 429s waited out and retried — **Done**, verified against the live API (8 of 8 figures described, 1 rejected request instead of 6)

* Crawler checks robots.txt (and honours Crawl-delay) before every fetch, as its user-agent says it does — **Done**, unit-tested
* Entailment check: an LLM judge verifies each cited sentence against the excerpt it cites, beside the vocabulary-overlap groundedness — **Done**, unit-tested with a faked judge
* Per-user run history — **Done**, unit-tested
* Page headers, captions, identifiers and table-number rows stripped before the reranker's snippet is taken — **Done**, unit-tested
* Several jobs at once: prefork worker pool on Linux/WSL, Groq rate-limit pacing shared through Redis with per-request reservations, figure cache safe under concurrent writers — **Done**, unit-tested with a faked Redis
* Iterative multi-hop retrieval (each hop's evidence shapes the next query) — **Done**, unit-tested with a faked planner

* Evaluation harness: golden set with reference answers and evidence, end-to-end runner with ablations and before/after comparison — **Done**, run live on 4 questions
* Entailment-judge agreement against labelled claims, plus an exporter to label claims from real runs — **Done**, run live (binary κ 1.00 on the starter set)
* In-memory ChromaDB per run, dropped when the run ends (no disk growth, no shared on-disk store across workers) — **Done**
* Refresh tokens in an httpOnly cookie with rotation and reuse detection; in-memory access tokens; restricted CORS — **Done**, unit-tested and verified through the dev proxy
* Docker Compose for the whole stack — **Done** (written, not yet run on a machine with Docker)
* Targeted revision of flagged sentences — **Done**, unit-tested and verified live
* Report streamed into the UI as it's written; clickable citations showing the source excerpt; frontend split into components — **Done**

**Still open, honestly:**
* Only 4 of the 20 golden questions have been run live so far. The full run (`python -m eval.run_eval --judge-answers`) and a multi-hop on/off comparison are the next things to do. Those are the numbers to quote.
* The entailment check is itself an LLM (`openai/gpt-oss-20b` by default), so it can be wrong in both directions. Treat a flagged claim as "check this by hand", not as proof. It costs one or two extra Groq calls per report. Self-correction still decides on groundedness, not entailment.
* Multi-hop planning sees each chunk only up to an equal share of `MULTI_HOP_EVIDENCE_TOKENS`, so a bridge fact buried deep in a long chunk can be missed. Each follow-up hop costs a Groq call plus a retrieval pass. The self-correction retry runs its own hops, so a poorly grounded run can make several extra calls.
* The Docker setup hasn't been run yet (it was written on a machine without Docker). Expect to fix small things on the first `docker compose up --build`.
* Revision trusts the reviser not to introduce new errors. The rewrites are judged again, but by the same family of model. The originals stay visible in the Claim check for that reason.
* `CHROMA_MODE=persistent` remains available for debugging, but persistent mode with several worker processes still shares one on-disk store, which ChromaDB doesn't support for concurrent writers. The default in-memory mode avoids this.
* Rate-limit pacing is shared only between processes using the same Redis. Anything else spending the same Groq key (another machine, a script) is invisible to it, and its 429s are still waited out and retried. If Redis is unreachable, pacing quietly falls back to per-process.
* The self-correction retry costs an extra Groq call (or several, with decomposition) plus a second retrieval pass whenever the first attempt is poorly grounded -- for a document genuinely unrelated to the instructions (like the deliberate-mismatch test scenario), this means paying for a retry that was never going to succeed. Not a bug, just a real cost/latency trade-off worth knowing about.

## Support

This is a personal practice project completed during an AI/ML internship. For questions, contact [your name / email].

## Contributing

This is a solo learning project and not currently open to external contributions. To run it locally for review, follow the Setup steps above, then work through the Smoke test to confirm each stage.

## Authors and acknowledgment

Built by Jayasundaram Gobinesh as part of an AI/ML internship practice project.

## License

Not currently licensed for public/open-source use — internal practice project.

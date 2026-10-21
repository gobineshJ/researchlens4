"""
Daily arXiv digest, as one command -- no API, Redis or Celery worker needed.
Built to run on a schedule in GitHub Actions (.github/workflows/daily-digest.yml),
so nothing has to stay running anywhere:

    python -m app.daily_digest --config ../digest.toml --state-dir ../digests

For every [[feed]] in the config:
  1. search arXiv for its keywords/categories over the last `days` days
  2. drop papers this feed already covered (state-dir/seen.json)
  3. download up to `max_papers` PDFs and run the normal pipeline on them
  4. write state-dir/reports/<date>-<feed>.md and .pdf
  5. record the papers as covered -- only after the report was written
Then one email with every feed's report attached (when SMTP_* is set).

The state dir is plain files on purpose: the workflow commits it back to
the repo, which is what carries "already covered" from one run to the next.

A feed that fails doesn't stop the others; the exit code is 1 if any feed
or the email failed, so the workflow run shows red (and GitHub emails you).

    --dry-run    search only: print what would be processed, change nothing
    --no-email   write reports and state, but don't send the email
"""

import argparse
import json
import os
import re
import shutil
import smtplib
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from . import config, papers, progress

_FEED_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


@dataclass
class Feed:
    name: str
    keywords: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    days: int | None = 3
    max_papers: int = 5
    max_results: int = 50
    instructions: str = config.ARXIV_DEFAULT_INSTRUCTIONS


@dataclass
class DigestConfig:
    feeds: list[Feed]
    email_when_nothing_new: bool = False


@dataclass
class FeedResult:
    feed: str
    papers: list[dict] = field(default_factory=list)     # the papers the report covers
    failed_downloads: list[str] = field(default_factory=list)
    markdown_path: Path | None = None
    pdf_path: Path | None = None
    report_text: str = ""
    error: str = ""


# --- config + state ---

def load_config(path: str | Path) -> DigestConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    defaults = raw.get("defaults", {})
    feeds = []
    for entry in raw.get("feed", []):
        merged = {**defaults, **entry}
        unknown = set(merged) - set(Feed.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown setting(s) {sorted(unknown)} in feed {merged.get('name')!r}.")
        feed = Feed(**merged)
        if not _FEED_NAME.match(feed.name):
            raise ValueError(f"Feed name {feed.name!r}: use letters, digits, '_', '.', '-' only.")
        if not 1 <= feed.max_papers <= config.ARXIV_MAX_PAPERS_PER_RUN:
            raise ValueError(f"Feed {feed.name!r}: max_papers must be 1-{config.ARXIV_MAX_PAPERS_PER_RUN}.")
        if not 1 <= feed.max_results <= 100:
            raise ValueError(f"Feed {feed.name!r}: max_results must be 1-100.")
        papers.build_query(feed.keywords, feed.categories, feed.days)  # raises on a bad query
        feeds.append(feed)
    if not feeds:
        raise ValueError(f"{path} defines no [[feed]].")
    names = [f.name for f in feeds]
    if len(set(names)) != len(names):
        raise ValueError("Feed names must be unique.")
    return DigestConfig(feeds=feeds, email_when_nothing_new=raw.get("email_when_nothing_new", False))


def load_seen(path: Path) -> dict[str, dict[str, str]]:
    """{feed: {paper id (no version): date it was covered}}"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_seen(path: Path, seen: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sorted, one id per line: the daily commit's diff shows exactly what was added.
    path.write_text(json.dumps(seen, indent=1, sort_keys=True) + "\n", encoding="utf-8")


# --- one feed ---

def _log(text: str) -> None:
    print(text, flush=True)


def find_new_papers(feed: Feed, seen_ids: set[str]) -> tuple[int, list[dict]]:
    query = papers.build_query(feed.keywords, feed.categories, feed.days)
    _log(f"[{feed.name}] arXiv query: {query}")
    total, found = papers.search_arxiv(query, max_results=feed.max_results)
    new = [p for p in found if p["id"] not in seen_ids]
    _log(f"[{feed.name}] {total} matching, {len(found)} fetched, {len(new)} not covered yet")
    return total, new[:feed.max_papers]


def _markdown(feed: Feed, day: str, included: list[dict], result: dict, failed: list[str]) -> str:
    lines = [f"# {feed.name} -- {day}", "", f"## Papers ({len(included)})", ""]
    for p in included:
        authors = ", ".join(p["authors"][:4]) + (" et al." if len(p["authors"]) > 4 else "")
        lines.append(f"- **{p['title']}** -- {authors} ([arXiv {p['arxiv_id']}]({p['abs_url']}))")
    if failed:
        lines += ["", "Could not include: " + "; ".join(failed)]
    lines += ["", "## Report", "", result["report_text"].strip()]
    if result.get("citations_used"):
        lines += ["", "## Sources", ""] + [f"{i}. {c}" for i, c in enumerate(result["citations_used"], 1)]
    return "\n".join(lines) + "\n"


def run_feed(feed: Feed, new_papers: list[dict], reports_dir: Path, day: str) -> FeedResult:
    # Imported here so --dry-run doesn't load the embedding and reranker models.
    from .pipeline import run_pipeline

    result = FeedResult(feed=feed.name)
    by_id = {p["arxiv_id"]: p for p in new_papers}
    with tempfile.TemporaryDirectory() as tmp:
        with progress.listening(lambda stage, detail: _log(f"[{feed.name}]   {stage}: {detail}" if detail
                                                           else f"[{feed.name}]   {stage}")):
            pdfs, downloaded, result.failed_downloads = papers.download_arxiv_pdfs(list(by_id), tmp)
            output = run_pipeline(pdfs, [], feed.instructions, run_id=f"digest-{feed.name}-{day}")
    if not output.get("report_text"):
        raise RuntimeError("The pipeline produced no report text.")

    result.papers = [by_id[i] for i in downloaded]
    result.report_text = output["report_text"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    result.markdown_path = reports_dir / f"{day}-{feed.name}.md"
    result.markdown_path.write_text(
        _markdown(feed, day, result.papers, output, result.failed_downloads), encoding="utf-8")
    if output.get("pdf_path"):
        result.pdf_path = reports_dir / f"{day}-{feed.name}.pdf"
        shutil.move(output["pdf_path"], result.pdf_path)
    return result


# --- email ---

def build_email(results: list[FeedResult], day: str, sender: str, recipient: str,
                repo_url: str = "") -> EmailMessage:
    covered = sum(len(r.papers) for r in results)
    msg = EmailMessage()
    msg["Subject"] = f"ResearchLens digest {day}: {covered} new paper{'s' * (covered != 1)}"
    msg["From"] = sender
    msg["To"] = recipient

    parts = []
    for r in results:
        if r.error:
            parts.append(f"=== {r.feed}: FAILED ===\n{r.error}")
        elif not r.papers:
            parts.append(f"=== {r.feed}: nothing new ===")
        else:
            listing = "\n".join(f"- {p['title']}\n  {p['abs_url']}" for p in r.papers)
            link = f"\nOnline: {repo_url}/blob/main/digests/reports/{r.markdown_path.name}" if repo_url else ""
            parts.append(f"=== {r.feed}: {len(r.papers)} paper(s) ===\n{listing}{link}\n\n{r.report_text.strip()}")
    msg.set_content("\n\n\n".join(parts) + "\n")

    for r in results:
        if r.pdf_path and r.pdf_path.exists():
            msg.add_attachment(r.pdf_path.read_bytes(), maintype="application", subtype="pdf",
                               filename=r.pdf_path.name)
    return msg


def send_email(msg: EmailMessage) -> None:
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(host, port, timeout=60) as smtp:
        smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(msg)


# --- main ---

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the daily arXiv digest.")
    parser.add_argument("--config", default="../digest.toml")
    parser.add_argument("--state-dir", default="../digests")
    parser.add_argument("--dry-run", action="store_true", help="search only; change nothing")
    parser.add_argument("--no-email", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    state_dir = Path(args.state_dir)
    seen_path = state_dir / "seen.json"
    seen = load_seen(seen_path)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    results: list[FeedResult] = []
    for feed in cfg.feeds:
        feed_seen = seen.setdefault(feed.name, {})
        try:
            _, new = find_new_papers(feed, set(feed_seen))
            if args.dry_run:
                for p in new:
                    _log(f"[{feed.name}]   would process {p['arxiv_id']}: {p['title']}")
                continue
            if not new:
                results.append(FeedResult(feed=feed.name))
                continue
            result = run_feed(feed, new, state_dir / "reports", day)
            for p in result.papers:
                feed_seen[p["id"]] = day
            # Saved after every feed, so a later feed crashing the process
            # can't lose the earlier ones' progress.
            save_seen(seen_path, seen)
            _log(f"[{feed.name}] wrote {result.markdown_path}")
            results.append(result)
        except Exception as exc:
            _log(f"[{feed.name}] FAILED: {type(exc).__name__}: {exc}")
            results.append(FeedResult(feed=feed.name, error=f"{type(exc).__name__}: {exc}"))

    if args.dry_run:
        return 0

    ok = not any(r.error for r in results)
    anything = any(r.papers or r.error for r in results)
    if args.no_email or not os.getenv("SMTP_USER"):
        _log("Email skipped (--no-email, or SMTP_USER not set).")
    elif not anything and not cfg.email_when_nothing_new:
        _log("Nothing new in any feed -- no email.")
    else:
        sender = os.environ["SMTP_USER"]
        repo = os.getenv("GITHUB_REPOSITORY")
        repo_url = f"{os.getenv('GITHUB_SERVER_URL', 'https://github.com')}/{repo}" if repo else ""
        try:
            send_email(build_email(results, day, sender, os.getenv("DIGEST_EMAIL_TO") or sender, repo_url))
            _log(f"Emailed the digest to {os.getenv('DIGEST_EMAIL_TO') or sender}.")
        except Exception as exc:
            _log(f"Email FAILED: {type(exc).__name__}: {exc}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

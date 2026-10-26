"""
Daily arXiv digest, as one command -- no API, Redis or Celery worker needed.
Built to run on a schedule in GitHub Actions (.github/workflows/daily-digest.yml),
so nothing has to stay running anywhere:

    python -m app.daily_digest --config ../digest.toml --state-dir ../digests

For every [[feed]] in the config:
  1. search arXiv for its keywords/categories over the last `days` days
  2. drop papers this feed already covered (state-dir/seen.json)
  3. rank the rest by relevance to the feed's topic -- the local embedding
     model against title + abstract, no API calls -- and keep the best
     `max_papers` that clear `min_relevance`
  4. summarise EACH paper in its own pipeline run ("summary" mode, its
     arXiv abstract as an extra source, section-aware excerpts), so every
     paper gets the whole context budget instead of a fifth of it
  5. one more Groq call reads those summaries and writes the themes they
     share plus the one paper most worth reading in full
  6. write state-dir/reports/<date>-<feed>.md and .pdf, then record the
     papers that made it in as covered
Then one email with every feed's overview, the PDFs attached (when SMTP_* is set).

The state dir is plain files on purpose: the workflow commits it back to
the repo, which is what carries "already covered" from one run to the next.

A feed that fails doesn't stop the others, and one paper that fails doesn't
stop its feed; the exit code is 1 if a whole feed or the email failed, so
the workflow run shows red (and GitHub emails you).

    --dry-run    search and rank only: print what would be processed, change nothing
    --no-email   write reports and state, but don't send the email
"""

import argparse
import json
import os
import re
import smtplib
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from . import config, groq_client, papers, pdf_export, progress

_FEED_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
# "TL;DR: ...", "**TL;DR:** ...", "TLDR - ..." -- only trailing ** is stripped,
# so a closing citation like "[1]." survives.
_TLDR = re.compile(r"^\W*TL\s*;?\s*DR\W*?[:\-]\W*\s*(.+?)[\s*]*$", re.IGNORECASE)


@dataclass
class Feed:
    name: str
    keywords: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    topic: str = ""
    days: int | None = 3
    max_papers: int = 5
    max_results: int = 50
    min_relevance: float = config.DIGEST_MIN_RELEVANCE
    instructions: str = config.DIGEST_PAPER_INSTRUCTIONS

    @property
    def topic_text(self) -> str:
        """What relevance is measured against: the feed's `topic` sentence,
        else its keywords. Empty for a categories-only feed (no ranking)."""
        return self.topic.strip() or "; ".join(k.strip() for k in self.keywords if k.strip())


@dataclass
class DigestConfig:
    feeds: list[Feed]
    email_when_nothing_new: bool = False


@dataclass
class PaperSummary:
    paper: dict
    relevance: float | None
    tldr: str
    body: str
    citations: list[dict]   # [{"n", "label"}], numbered by this summary's own markers


@dataclass
class FeedResult:
    feed: str
    summaries: list[PaperSummary] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)       # "id: reason", downloads and summaries
    themes: str = ""
    themes_note: str = ""                                 # what check_themes did
    read_in_full: int | None = None                       # 1-based index into summaries
    why: str = ""
    markdown_path: Path | None = None
    pdf_path: Path | None = None
    error: str = ""

    @property
    def papers(self) -> list[dict]:
        return [s.paper for s in self.summaries]


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
        if not -1 <= feed.min_relevance <= 1:
            raise ValueError(f"Feed {feed.name!r}: min_relevance must be between -1 and 1.")
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


def _log(text: str) -> None:
    print(text, flush=True)


# --- choosing papers ---

def find_new_papers(feed: Feed, seen_ids: set[str]) -> list[dict]:
    query = papers.build_query(feed.keywords, feed.categories, feed.days)
    _log(f"[{feed.name}] arXiv query: {query}")
    total, found = papers.search_arxiv(query, max_results=feed.max_results)
    new = [p for p in found if p["id"] not in seen_ids]
    _log(f"[{feed.name}] {total} matching, {len(found)} fetched, {len(new)} not covered yet")
    return new


def relevance_scores(topic: str, candidates: list[dict]) -> list[float]:
    """Cosine similarity of each paper's title + abstract to `topic`."""
    from . import vectorstore  # loads the embedding model; --dry-run on a categories-only feed skips it
    vectors = vectorstore.embed([topic] + [f"{p['title']}. {p['abstract']}" for p in candidates])
    query = vectors[0]
    return [sum(a * b for a, b in zip(query, v)) for v in vectors[1:]]


def choose_papers(feed: Feed, candidates: list[dict]) -> list[tuple[float | None, dict]]:
    """
    The papers worth summarising today: most relevant first, at most
    max_papers, none below min_relevance. A categories-only feed has nothing
    to rank against, so it keeps arXiv's newest-first order.

    Papers left out are not marked seen: one that just missed the cut can
    still make it on a quieter day while it's inside the `days` window.
    """
    if not candidates:
        return []
    topic = feed.topic_text
    if not topic:
        return [(None, p) for p in candidates[:feed.max_papers]]
    ranked = sorted(zip(relevance_scores(topic, candidates), candidates), key=lambda sp: sp[0], reverse=True)
    for score, paper in ranked:
        mark = "keep" if score >= feed.min_relevance else "skip"
        _log(f"[{feed.name}]   {score:.2f} {mark} {paper['arxiv_id']}: {paper['title'][:90]}")
    kept = [(score, paper) for score, paper in ranked if score >= feed.min_relevance]
    if len(kept) < len(ranked):
        _log(f"[{feed.name}] {len(ranked) - len(kept)} below min_relevance {feed.min_relevance}")
    return kept[:feed.max_papers]


# --- summarising ---

def abstract_record(paper: dict, filename: str) -> dict:
    """The paper's arXiv metadata as an extra source for its pipeline run,
    filed under the same document as its PDF so it becomes that paper's
    first chunk -- the densest statement of what the paper claims."""
    authors = ", ".join(paper["authors"])
    return {
        "source_type": "file",
        "filename": filename,
        "location": "arXiv abstract",
        "text": f"Title: {paper['title']}. Authors: {authors}. Abstract: {paper['abstract']}",
    }


def split_tldr(text: str) -> tuple[str, str]:
    """('one sentence', rest of the summary) when the summary opens with a
    'TL;DR: ...' line as instructed, else ('', text)."""
    lines = text.strip().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        match = _TLDR.match(line.strip())
        if match:
            return match.group(1).strip(), "\n".join(lines[i + 1:]).strip()
        break
    return "", text.strip()


def summarize_paper(feed: Feed, paper: dict, relevance: float | None, pdf_path: str, day: str) -> PaperSummary:
    # Imported here so --dry-run doesn't load the embedding and reranker models.
    from .pipeline import run_pipeline

    output = run_pipeline(
        [pdf_path], [], feed.instructions, run_id=f"digest-{feed.name}-{day}-{paper['id']}",
        extra_records=[abstract_record(paper, Path(pdf_path).name)], intent="summary", export_pdf=False,
        mind_map=False,
    )
    if not output.get("report_text"):
        raise RuntimeError("the pipeline produced no summary")
    tldr, body = split_tldr(output["report_text"])
    return PaperSummary(paper, relevance, tldr, body, output.get("citations", []))


_PAPER_MARKER = re.compile(r"\[(P\d+(?:\s*[,–-]\s*P?\d+)*)\]")
_NUMBER_MARKER = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
_LEADING_PAPER_MARKER = re.compile(r"^(\[(P\d+(?:\s*[,–-]\s*P?\d+)*)\])\s+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=\S)")


def markers_to_sentence_ends(text: str) -> str:
    """
    "[P1] introduces X." -> "P1 introduces X [P1]."

    The claim checker reads a marker at the START of a sentence as the tail
    of the sentence before it ("... improved. [2]" is a common way to cite),
    so a sentence opening with its citation would be judged against the
    wrong paper -- and removing the previous sentence would take that
    paper's marker with it. Moving the marker to the end keeps each
    sentence and its paper together.
    """
    paragraphs = []
    for paragraph in text.split("\n\n"):
        sentences = []
        for sentence in _SENTENCE_BREAK.split(paragraph.strip()):
            match = _LEADING_PAPER_MARKER.match(sentence)
            if match:
                marker, papers_cited = match.group(1), match.group(2)
                rest = sentence[match.end():]
                end = len(rest.rstrip()) - (1 if rest.rstrip()[-1:] in ".!?" else 0)
                tail = "" if marker in rest else f" {marker}"
                sentence = f"{papers_cited} {rest[:end].rstrip()}{tail}{rest[end:]}"
            sentences.append(sentence)
        paragraphs.append(" ".join(sentences))
    return "\n\n".join(paragraphs)


def check_themes(themes: str, summaries: list[PaperSummary]) -> tuple[str, str]:
    """
    Holds the themes to the same standard as the summaries: every sentence
    citing [P2] is judged against paper 2's summary by the entailment judge,
    and flagged sentences are rewritten or removed by the same reviser the
    pipeline uses. The summaries stand in as the "excerpts", [Pn] becomes
    [n] for the checker and back afterwards.

    Returns (themes to publish, a one-line note on what the check did).
    Raises on request failures -- the caller then keeps the themes unchecked.
    """
    from . import evaluation

    sources = [{"text": f"{s.paper['title']}\nTL;DR: {s.tldr}\n{s.body}"} for s in summaries]
    themes = markers_to_sentence_ends(themes)
    numbered = _PAPER_MARKER.sub(lambda m: "[" + m.group(1).replace("P", "") + "]", themes)
    claims = evaluation.extract_cited_claims(numbered, len(sources))
    if not claims:
        return themes, "Themes cite no papers, so they could not be checked."
    verdicts = groq_client.judge_claims(claims, sources)
    flagged = [i for i, v in enumerate(verdicts) if v["verdict"] in ("unsupported", "contradicted")]
    note = f"{len(claims)} theme sentence(s) checked against the paper summaries"
    if flagged:
        revisions = groq_client.revise_claims(
            [{"claim": verdicts[i]["claim"], "sources": verdicts[i]["sources"], "reason": verdicts[i]["reason"]}
             for i in flagged],
            sources,
        )
        applied = [{**r, "span": claims[i]["span"]} for i, r in zip(flagged, revisions) if r is not None]
        numbered = evaluation.apply_revisions(numbered, applied)
        removed = sum(r["action"] == "remove" for r in applied)
        note += f"; {len(applied) - removed} corrected, {removed} removed"
    note += "."
    restored = _NUMBER_MARKER.sub(
        lambda m: "[" + re.sub(r"(\d+)", r"P\1", m.group(1)) + "]", numbered)
    return restored, note


def run_feed(feed: Feed, chosen: list[tuple[float | None, dict]], reports_dir: Path, day: str) -> FeedResult:
    result = FeedResult(feed=feed.name)
    relevance = {p["arxiv_id"]: score for score, p in chosen}
    by_id = {p["arxiv_id"]: p for _, p in chosen}
    with tempfile.TemporaryDirectory() as tmp:
        with progress.listening(lambda stage, detail: _log(f"[{feed.name}]   {stage}: {detail}" if detail
                                                           else f"[{feed.name}]   {stage}")):
            pdfs, downloaded, result.failed = papers.download_arxiv_pdfs(list(by_id), tmp)
            for i, (arxiv_id, pdf) in enumerate(zip(downloaded, pdfs), start=1):
                _log(f"[{feed.name}] summarising {i}/{len(pdfs)}: {arxiv_id}")
                try:
                    result.summaries.append(summarize_paper(feed, by_id[arxiv_id], relevance[arxiv_id], pdf, day))
                except Exception as exc:
                    _log(f"[{feed.name}] could not summarise {arxiv_id}: {type(exc).__name__}: {exc}")
                    result.failed.append(f"{arxiv_id}: {type(exc).__name__}: {exc}")
    if not result.summaries:
        raise RuntimeError("No paper could be summarised -- " + "; ".join(result.failed))

    if len(result.summaries) > 1:
        try:
            synthesis = groq_client.synthesize_digest(
                feed.topic_text or feed.name,
                [{"title": s.paper["title"], "summary": f"TL;DR: {s.tldr}\n{s.body}"} for s in result.summaries],
            )
            result.themes, result.read_in_full, result.why = (
                synthesis["themes"], synthesis["read_in_full"], synthesis["why"])
        except Exception as exc:
            # The per-paper summaries are the substance; the overview is a bonus.
            _log(f"[{feed.name}] overview skipped: {type(exc).__name__}: {exc}")
        if result.themes:
            try:
                result.themes, result.themes_note = check_themes(result.themes, result.summaries)
            except Exception as exc:
                result.themes_note = f"Themes could not be fact-checked ({type(exc).__name__})."
            _log(f"[{feed.name}] {result.themes_note}")

    reports_dir.mkdir(parents=True, exist_ok=True)
    result.markdown_path = reports_dir / f"{day}-{feed.name}.md"
    result.markdown_path.write_text(render_markdown(feed, day, result), encoding="utf-8")
    result.pdf_path = reports_dir / f"{day}-{feed.name}.pdf"
    render_pdf(feed, day, result, result.pdf_path)
    return result


# --- rendering ---

def _authors(paper: dict) -> str:
    return ", ".join(paper["authors"][:4]) + (" et al." if len(paper["authors"]) > 4 else "")


def _relevance_note(s: PaperSummary) -> str:
    return f" · relevance {s.relevance:.2f}" if s.relevance is not None else ""


def _overview_parts(feed: Feed, result: FeedResult) -> list[tuple[str, str]]:
    """[(heading, markdown text)] shared by the Markdown and PDF reports."""
    parts = []
    if result.read_in_full:
        s = result.summaries[result.read_in_full - 1]
        parts.append(("Worth reading in full",
                      f"**[P{result.read_in_full}] {s.paper['title']}** -- {result.why}"))
    if result.themes:
        parts.append(("Themes", result.themes + (f"\n\n*{result.themes_note}*" if result.themes_note else "")))
    glance = "\n".join(
        f"**[P{i}] {s.paper['title']}**{_relevance_note(s)}\n{s.tldr or '(no TL;DR)'}"
        for i, s in enumerate(result.summaries, start=1)
    )
    parts.append(("At a glance", glance.replace("\n**", "\n\n**")))
    if result.failed:
        parts.append(("Not included", "\n".join(result.failed)))
    return parts


def render_markdown(feed: Feed, day: str, result: FeedResult) -> str:
    topic = f" · topic: {feed.topic_text}" if feed.topic_text else ""
    lines = [f"# {feed.name} -- {day}", "", f"{len(result.summaries)} new paper(s){topic}", ""]
    for heading, text in _overview_parts(feed, result):
        lines += [f"## {heading}", "", text, ""]
    lines += ["Numbers like [P2] refer to the papers below; [n] inside a paper's summary "
              "refers to that paper's own sources.", ""]
    for i, s in enumerate(result.summaries, start=1):
        p = s.paper
        lines += ["---", "", f"## [P{i}] {p['title']}", "",
                  f"{_authors(p)} · [arXiv {p['arxiv_id']}]({p['abs_url']}){_relevance_note(s)}", ""]
        if s.tldr:
            lines += [f"**TL;DR:** {s.tldr}", ""]
        lines += [s.body, ""]
        if s.citations:
            lines += ["**Sources**", ""] + [f"- [{c['n']}] {c['label']}" for c in s.citations] + [""]
    return "\n".join(lines)


def render_pdf(feed: Feed, day: str, result: FeedResult, path: Path) -> None:
    topic = f" · topic: {feed.topic_text}" if feed.topic_text else ""
    pdf_export.build_digest_pdf(
        str(path), title=f"{feed.name} -- {day}", subtitle=f"{len(result.summaries)} new paper(s){topic}",
        overview=_overview_parts(feed, result),
        papers=[{
            "heading": f"[P{i}] {s.paper['title']}",
            "meta": f"{_authors(s.paper)} · arXiv {s.paper['arxiv_id']} · {s.paper['abs_url']}{_relevance_note(s)}",
            "text": (f"**TL;DR:** {s.tldr}\n\n" if s.tldr else "") + s.body,
            "citations": s.citations,
        } for i, s in enumerate(result.summaries, start=1)],
    )


# --- email ---

def build_email(results: list[FeedResult], day: str, sender: str, recipient: str,
                repo_url: str = "") -> EmailMessage:
    covered = sum(len(r.summaries) for r in results)
    msg = EmailMessage()
    msg["Subject"] = f"ResearchLens digest {day}: {covered} new paper{'s' * (covered != 1)}"
    msg["From"] = sender
    msg["To"] = recipient

    parts = []
    for r in results:
        if r.error:
            parts.append(f"=== {r.feed}: FAILED ===\n{r.error}")
            continue
        if not r.summaries:
            parts.append(f"=== {r.feed}: nothing new ===")
            continue
        section = [f"=== {r.feed}: {len(r.summaries)} paper(s) ==="]
        if r.read_in_full:
            s = r.summaries[r.read_in_full - 1]
            section.append(f"Worth reading in full: {s.paper['title']}\n{r.why}\n{s.paper['abs_url']}")
        if r.themes:
            section.append("Themes:\n" + r.themes)
        section.append("At a glance:\n" + "\n\n".join(
            f"[P{i}] {s.paper['title']}\n{s.tldr or '(no TL;DR)'}\n{s.paper['abs_url']}"
            for i, s in enumerate(r.summaries, start=1)))
        if r.failed:
            section.append("Not included: " + "; ".join(r.failed))
        online = f" -- online: {repo_url}/blob/main/digests/reports/{r.markdown_path.name}" if repo_url else ""
        section.append(f"Full summaries with sources: the attached PDF{online}")
        parts.append("\n\n".join(section))
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
    parser.add_argument("--dry-run", action="store_true", help="search and rank only; change nothing")
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
            chosen = choose_papers(feed, find_new_papers(feed, set(feed_seen)))
            if args.dry_run:
                for score, p in chosen:
                    note = f" (relevance {score:.2f})" if score is not None else ""
                    _log(f"[{feed.name}]   would process {p['arxiv_id']}{note}: {p['title']}")
                continue
            if not chosen:
                results.append(FeedResult(feed=feed.name))
                continue
            result = run_feed(feed, chosen, state_dir / "reports", day)
            for s in result.summaries:
                feed_seen[s.paper["id"]] = day
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
    anything = any(r.summaries or r.error for r in results)
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

"""
Tiny SQLite store for what the API has to remember between requests:
user accounts, their refresh tokens and API keys, which user submitted
which background job (so one user can't poll -- or download the report
of -- another user's run), and which papers each digest feed has already
covered. Plain sqlite3, no ORM: five small tables don't justify one.
"""

import sqlite3
from contextlib import contextmanager

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS refresh_tokens (
    token_hash TEXT PRIMARY KEY,               -- sha256 of the token; the token itself is never stored
    username   TEXT NOT NULL REFERENCES users(username),
    family_id  TEXT NOT NULL,                  -- one login; every rotation stays in the family
    expires_at TEXT NOT NULL,
    revoked    INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    username     TEXT NOT NULL REFERENCES users(username),
    pdf_filename TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS api_keys (
    key_id       TEXT PRIMARY KEY,             -- public handle, used to list and revoke
    key_hash     TEXT NOT NULL UNIQUE,         -- sha256 of the key; the key itself is never stored
    prefix       TEXT NOT NULL,                -- first characters, so a listing can say which key is which
    username     TEXT NOT NULL REFERENCES users(username),
    name         TEXT NOT NULL,
    revoked      INTEGER NOT NULL DEFAULT 0,
    last_used_at TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS seen_papers (
    username TEXT NOT NULL REFERENCES users(username),
    feed     TEXT NOT NULL,                    -- one digest's history, e.g. "daily-nlp"
    source   TEXT NOT NULL,                    -- "arxiv"
    paper_id TEXT NOT NULL,                    -- arXiv id without its version suffix
    job_id   TEXT,                             -- the run that covered it, if any
    seen_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (username, feed, source, paper_id)
);
"""


@contextmanager
def connect():
    # config.AUTH_DB_PATH is read on every call (not imported once) so
    # tests can point it at a temp file with monkeypatch.
    conn = sqlite3.connect(config.AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


# --- users ---

def get_user(username: str) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def create_user(username: str, password_hash: str) -> bool:
    """False if the username is already taken."""
    with connect() as conn:
        try:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash),
            )
        except sqlite3.IntegrityError:
            return False
    return True


# --- jobs ---

def record_job(job_id: str, username: str) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO jobs (job_id, username) VALUES (?, ?)", (job_id, username))


def get_job_owner(job_id: str) -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT username FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return row["username"] if row else None


def set_job_pdf(job_id: str, pdf_filename: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE jobs SET pdf_filename = ? WHERE job_id = ?", (pdf_filename, job_id))


def user_owns_pdf(username: str, pdf_filename: str) -> bool:
    with connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE username = ? AND pdf_filename = ?",
            (username, pdf_filename),
        ).fetchone()
    return row is not None


# --- refresh tokens ---

def store_refresh_token(token_hash: str, username: str, family_id: str, expires_at: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO refresh_tokens (token_hash, username, family_id, expires_at) VALUES (?, ?, ?, ?)",
            (token_hash, username, family_id, expires_at),
        )


def get_refresh_token(token_hash: str) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM refresh_tokens WHERE token_hash = ?", (token_hash,)).fetchone()


def revoke_refresh_token(token_hash: str) -> bool:
    """Marks one token used. False if it was already revoked -- checked and
    set in one statement, so two concurrent refreshes can't both succeed."""
    with connect() as conn:
        cursor = conn.execute(
            "UPDATE refresh_tokens SET revoked = 1 WHERE token_hash = ? AND revoked = 0", (token_hash,)
        )
    return cursor.rowcount == 1


def revoke_refresh_family(family_id: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE refresh_tokens SET revoked = 1 WHERE family_id = ?", (family_id,))


def delete_expired_refresh_tokens(now_iso: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM refresh_tokens WHERE expires_at < ?", (now_iso,))


# --- API keys ---

def store_api_key(key_id: str, key_hash: str, prefix: str, username: str, name: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO api_keys (key_id, key_hash, prefix, username, name) VALUES (?, ?, ?, ?, ?)",
            (key_id, key_hash, prefix, username, name),
        )


def get_api_key_by_hash(key_hash: str) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)).fetchone()


def touch_api_key(key_id: str, now_iso: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE api_keys SET last_used_at = ? WHERE key_id = ?", (now_iso, key_id))


def list_api_keys(username: str) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT key_id, prefix, name, revoked, last_used_at, created_at FROM api_keys "
            "WHERE username = ? ORDER BY created_at DESC",
            (username,),
        ).fetchall()


def revoke_api_key(username: str, key_id: str) -> bool:
    """False if the key doesn't exist or belongs to someone else."""
    with connect() as conn:
        cursor = conn.execute(
            "UPDATE api_keys SET revoked = 1 WHERE key_id = ? AND username = ?", (key_id, username)
        )
    return cursor.rowcount == 1


# --- seen papers ---

def seen_paper_ids(username: str, feed: str, source: str, paper_ids: list[str]) -> set[str]:
    """The subset of paper_ids this feed has already covered."""
    if not paper_ids:
        return set()
    placeholders = ",".join("?" * len(paper_ids))
    with connect() as conn:
        rows = conn.execute(
            f"SELECT paper_id FROM seen_papers WHERE username = ? AND feed = ? AND source = ? "
            f"AND paper_id IN ({placeholders})",
            (username, feed, source, *paper_ids),
        ).fetchall()
    return {row["paper_id"] for row in rows}


def mark_papers_seen(username: str, feed: str, source: str, paper_ids: list[str],
                     job_id: str | None = None) -> int:
    """Records paper_ids as covered by this feed; already-seen ones are left
    as they were. Returns how many were newly recorded."""
    with connect() as conn:
        cursor = conn.executemany(
            "INSERT OR IGNORE INTO seen_papers (username, feed, source, paper_id, job_id) "
            "VALUES (?, ?, ?, ?, ?)",
            [(username, feed, source, paper_id, job_id) for paper_id in paper_ids],
        )
    return cursor.rowcount

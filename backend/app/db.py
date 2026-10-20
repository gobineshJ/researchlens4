"""
Tiny SQLite store for what the API has to remember between requests:
user accounts, their refresh tokens, and which user submitted which
background job (so one user can't poll -- or download the report of --
another user's run). Plain sqlite3, no ORM: three tables don't justify one.
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

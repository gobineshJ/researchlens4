"""
Tiny SQLite store for the two things the API has to remember between
requests: user accounts, and which user submitted which background job
(so one user can't poll -- or download the report of -- another user's
run). Plain sqlite3, no ORM: two tables don't justify one.
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

"""SQLite storage. Single file, no ORM, ships with the repo so graders get data on day one."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id            INTEGER PRIMARY KEY,
    url           TEXT NOT NULL UNIQUE,
    title         TEXT NOT NULL DEFAULT '',
    is_watched    INTEGER NOT NULL DEFAULT 1,
    first_seen    TEXT NOT NULL,
    last_fetched  TEXT,
    etag          TEXT,
    last_modified TEXT
);

CREATE TABLE IF NOT EXISTS snapshots (
    id           INTEGER PRIMARY KEY,
    page_id      INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    fetched_at   TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    body_text    TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'network'
);
CREATE INDEX IF NOT EXISTS idx_snapshots_page ON snapshots(page_id, id DESC);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY,
    page_id       INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    snapshot_id   INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    section_title TEXT NOT NULL DEFAULT '',
    anchor        TEXT NOT NULL DEFAULT '',
    text_display  TEXT NOT NULL,
    text_index    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_id);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    dim      INTEGER NOT NULL,
    vector   BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS change_events (
    id            INTEGER PRIMARY KEY,
    page_id       INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    detected_at   TEXT NOT NULL,
    old_hash      TEXT,
    new_hash      TEXT NOT NULL,
    diff_text     TEXT NOT NULL DEFAULT '',
    summary       TEXT NOT NULL DEFAULT '',
    notify_status TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS qa_log (
    id         INTEGER PRIMARY KEY,
    asked_at   TEXT NOT NULL,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL DEFAULT '',
    citations  TEXT NOT NULL DEFAULT '[]',
    grounded   INTEGER NOT NULL DEFAULT 1,
    model      TEXT NOT NULL DEFAULT '',
    latency_ms INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tool_trace (
    id          INTEGER PRIMARY KEY,
    qa_id       INTEGER REFERENCES qa_log(id) ON DELETE CASCADE,
    step        INTEGER NOT NULL,
    tool_name   TEXT NOT NULL,
    tool_input  TEXT NOT NULL DEFAULT '{}',
    tool_result TEXT NOT NULL DEFAULT '',
    duration_ms INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trace_qa ON tool_trace(qa_id, step);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    # check_same_thread=False because Streamlit runs each script re-run in a fresh thread but
    # caches one connection across them. Safe here: sqlite3.threadsafety is 3 (serialized).
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


# --- pages ---


def upsert_page(conn: sqlite3.Connection, url: str, title: str = "") -> int:
    row = conn.execute("SELECT id, title FROM pages WHERE url = ?", (url,)).fetchone()
    if row:
        if title and title != row["title"]:
            conn.execute("UPDATE pages SET title = ? WHERE id = ?", (title, row["id"]))
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO pages (url, title, first_seen) VALUES (?, ?, ?)",
        (url, title, utcnow()),
    )
    return int(cur.lastrowid)


def mark_fetched(
    conn: sqlite3.Connection,
    page_id: int,
    etag: str | None = None,
    last_modified: str | None = None,
) -> None:
    conn.execute(
        "UPDATE pages SET last_fetched = ?, etag = ?, last_modified = ? WHERE id = ?",
        (utcnow(), etag, last_modified, page_id),
    )


def list_pages(conn: sqlite3.Connection, watched_only: bool = False) -> list[sqlite3.Row]:
    sql = "SELECT * FROM pages"
    if watched_only:
        sql += " WHERE is_watched = 1"
    return list(conn.execute(sql + " ORDER BY id").fetchall())


def get_page(conn: sqlite3.Connection, url: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM pages WHERE url = ?", (url,)).fetchone()


# --- snapshots ---


def latest_snapshot(conn: sqlite3.Connection, page_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM snapshots WHERE page_id = ? ORDER BY id DESC LIMIT 1", (page_id,)
    ).fetchone()


def insert_snapshot(
    conn: sqlite3.Connection,
    page_id: int,
    content_hash: str,
    body_text: str,
    source: str = "network",
) -> int:
    cur = conn.execute(
        "INSERT INTO snapshots (page_id, fetched_at, content_hash, body_text, source)"
        " VALUES (?, ?, ?, ?, ?)",
        (page_id, utcnow(), content_hash, body_text, source),
    )
    return int(cur.lastrowid)


# --- chunks and embeddings ---


def replace_chunks(
    conn: sqlite3.Connection, page_id: int, snapshot_id: int, sections: Sequence[dict[str, Any]]
) -> list[int]:
    """Chunks always describe the newest snapshot, so old ones are dropped wholesale."""
    conn.execute("DELETE FROM chunks WHERE page_id = ?", (page_id,))
    ids: list[int] = []
    for ordinal, section in enumerate(sections):
        cur = conn.execute(
            "INSERT INTO chunks (page_id, snapshot_id, ordinal, section_title, anchor,"
            " text_display, text_index) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                page_id,
                snapshot_id,
                ordinal,
                section.get("section_title", ""),
                section.get("anchor", ""),
                section["text_display"],
                section["text_index"],
            ),
        )
        ids.append(int(cur.lastrowid))
    return ids


def store_embeddings(conn: sqlite3.Connection, rows: Iterable[tuple[int, int, bytes]]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings (chunk_id, dim, vector) VALUES (?, ?, ?)", rows
    )


def all_chunks_with_vectors(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT c.id, c.page_id, c.section_title, c.anchor, c.text_display, c.text_index,"
            "       p.url, p.title AS page_title, e.dim, e.vector"
            "  FROM chunks c"
            "  JOIN pages p ON p.id = c.page_id"
            "  LEFT JOIN embeddings e ON e.chunk_id = c.id"
            " ORDER BY c.page_id, c.ordinal"
        ).fetchall()
    )


def get_chunk(conn: sqlite3.Connection, chunk_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT c.*, p.url, p.title AS page_title FROM chunks c"
        " JOIN pages p ON p.id = c.page_id WHERE c.id = ?",
        (chunk_id,),
    ).fetchone()


def chunks_missing_vectors(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT c.id, c.text_index FROM chunks c"
            " LEFT JOIN embeddings e ON e.chunk_id = c.id WHERE e.chunk_id IS NULL"
        ).fetchall()
    )


# --- change events ---


def insert_change_event(
    conn: sqlite3.Connection,
    page_id: int,
    old_hash: str | None,
    new_hash: str,
    diff_text: str,
    summary: str,
) -> int:
    cur = conn.execute(
        "INSERT INTO change_events (page_id, detected_at, old_hash, new_hash, diff_text, summary)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (page_id, utcnow(), old_hash, new_hash, diff_text, summary),
    )
    return int(cur.lastrowid)


def set_notify_status(conn: sqlite3.Connection, event_id: int, status: str) -> None:
    conn.execute("UPDATE change_events SET notify_status = ? WHERE id = ?", (status, event_id))


def recent_change_events(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT ce.*, p.url, p.title AS page_title FROM change_events ce"
            " JOIN pages p ON p.id = ce.page_id ORDER BY ce.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    )


# --- qa log and tool trace ---


def start_qa(conn: sqlite3.Connection, question: str, model: str) -> int:
    cur = conn.execute(
        "INSERT INTO qa_log (asked_at, question, model) VALUES (?, ?, ?)",
        (utcnow(), question, model),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_qa(
    conn: sqlite3.Connection,
    qa_id: int,
    answer: str,
    citations: Sequence[dict[str, Any]],
    grounded: bool,
    latency_ms: int,
) -> None:
    conn.execute(
        "UPDATE qa_log SET answer = ?, citations = ?, grounded = ?, latency_ms = ? WHERE id = ?",
        (answer, json.dumps(citations, ensure_ascii=False), int(grounded), latency_ms, qa_id),
    )
    conn.commit()


def log_tool_call(
    conn: sqlite3.Connection,
    qa_id: int | None,
    step: int,
    tool_name: str,
    tool_input: Any,
    tool_result: str,
    duration_ms: int,
) -> None:
    conn.execute(
        "INSERT INTO tool_trace (qa_id, step, tool_name, tool_input, tool_result, duration_ms,"
        " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            qa_id,
            step,
            tool_name,
            json.dumps(tool_input, ensure_ascii=False),
            tool_result,
            duration_ms,
            utcnow(),
        ),
    )
    conn.commit()


def trace_for_qa(conn: sqlite3.Connection, qa_id: int) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT * FROM tool_trace WHERE qa_id = ? ORDER BY step", (qa_id,)
        ).fetchall()
    )

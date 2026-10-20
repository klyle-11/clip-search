"""SQLite storage: episodes, timed segments, search chunks (+FTS5, +embedding BLOBs)."""
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY,
    source      TEXT NOT NULL,          -- youtube | patreon_rss | obsidian
    source_id   TEXT NOT NULL,          -- video id / guid / note path
    title       TEXT,
    url         TEXT,
    published   TEXT,
    media_path  TEXT,                   -- local media file to cut from, if any
    indexed     INTEGER DEFAULT 0,
    UNIQUE(source, source_id)
);
CREATE TABLE IF NOT EXISTS segments (
    id          INTEGER PRIMARY KEY,
    episode_id  INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    start       REAL NOT NULL,
    end         REAL NOT NULL,
    text        TEXT NOT NULL,
    speaker     TEXT                    -- filled later by diarization
);
CREATE INDEX IF NOT EXISTS seg_ep ON segments(episode_id, start);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    episode_id  INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    start       REAL NOT NULL,
    end         REAL NOT NULL,
    text        TEXT NOT NULL,
    speakers    TEXT,                   -- comma-joined speakers in window
    embedding   BLOB
);
CREATE INDEX IF NOT EXISTS chunk_ep ON chunks(episode_id, start);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""


def connect(path: str | None = None) -> sqlite3.Connection:
    path = path or os.getenv("JBP_DB", "data/jbp.db")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    return conn


def upsert_episode(conn, *, source, source_id, title=None, url=None, published=None, media_path=None) -> tuple[int, bool]:
    """Return (episode_id, is_new). Existing episodes are left alone (re-ingest = no-op)."""
    row = conn.execute(
        "SELECT id FROM episodes WHERE source=? AND source_id=?", (source, source_id)
    ).fetchone()
    if row:
        return row["id"], False
    cur = conn.execute(
        "INSERT INTO episodes(source, source_id, title, url, published, media_path) VALUES (?,?,?,?,?,?)",
        (source, source_id, title, url, published, media_path),
    )
    return cur.lastrowid, True


def add_segments(conn, episode_id: int, segments: list[tuple[float, float, str]] | list[tuple[float, float, str, str]]):
    rows = [(episode_id, s[0], s[1], s[2], s[3] if len(s) > 3 else None) for s in segments if s[2].strip()]
    conn.executemany(
        "INSERT INTO segments(episode_id, start, end, text, speaker) VALUES (?,?,?,?,?)", rows
    )
    return len(rows)

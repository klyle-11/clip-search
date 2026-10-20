"""SQLite storage: episodes, timed segments, search chunks (+FTS5, +embedding BLOBs)."""
import os
import re
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
    diarized    INTEGER DEFAULT 0,
    UNIQUE(source, source_id)
);
CREATE TABLE IF NOT EXISTS segments (
    id          INTEGER PRIMARY KEY,
    episode_id  INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    start       REAL NOT NULL,
    end         REAL NOT NULL,
    text        TEXT NOT NULL,
    speaker     TEXT,                   -- voiceprint name, else raw cluster label
    cluster     TEXT                    -- raw diarization label (SPEAKER_00 ...)
);
CREATE INDEX IF NOT EXISTS seg_ep ON segments(episode_id, start);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    episode_id  INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    start       REAL NOT NULL,
    end         REAL NOT NULL,
    text        TEXT NOT NULL,
    norm        TEXT,                   -- textnorm.norm(text): what FTS indexes
    speakers    TEXT,                   -- comma-joined speakers in window
    embedding   BLOB
);
CREATE TABLE IF NOT EXISTS voiceprints (
    name        TEXT PRIMARY KEY,
    embedding   BLOB NOT NULL,          -- running mean of enrolled embeddings (L2-normalized)
    n           INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS ep_speakers (
    episode_id  INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    cluster     TEXT NOT NULL,
    seconds     REAL NOT NULL,          -- total talk time in the episode
    embedding   BLOB,                   -- cluster voice embedding
    name        TEXT,                   -- matched/labelled voiceprint
    score       REAL,                   -- cosine to that voiceprint
    manual      INTEGER DEFAULT 0,      -- set by `jbp label`; never overwritten by auto-matching
    PRIMARY KEY (episode_id, cluster)
);
CREATE INDEX IF NOT EXISTS chunk_ep ON chunks(episode_id, start);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    norm, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vocab USING fts5vocab(chunks_fts, row);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, norm) VALUES (new.id, new.norm);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, norm) VALUES ('delete', old.id, old.norm);
END;
"""


# columns added after v0.1 — ALTER existing DBs in place
MIGRATIONS = [("episodes", "diarized", "INTEGER DEFAULT 0"), ("segments", "cluster", "TEXT"), ("chunks", "norm", "TEXT")]


def _regexp(pattern, value):
    return value is not None and re.search(pattern, value, re.I) is not None


def _migrate_fts(conn) -> None:
    """v0.2 indexed raw chunks.text; now FTS indexes normalized text. Rebuild once if needed."""
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(chunks_fts)")]
    if cols == ["norm"]:
        return
    from .textnorm import norm
    conn.executescript("DROP TRIGGER IF EXISTS chunks_ai; DROP TRIGGER IF EXISTS chunks_ad; DROP TABLE IF EXISTS chunks_fts;")
    rows = conn.execute("SELECT id, text FROM chunks WHERE norm IS NULL").fetchall()
    conn.executemany("UPDATE chunks SET norm=? WHERE id=?", [(norm(r["text"]), r["id"]) for r in rows])
    conn.executescript(FTS_SCHEMA)
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
    conn.commit()


def connect(path: str | None = None) -> sqlite3.Connection:
    path = path or os.getenv("JBP_DB", "data/jbp.db")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    for table, col, decl in MIGRATIONS:
        if col not in {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
    _migrate_fts(conn)
    conn.create_function("REGEXP", 2, _regexp, deterministic=True)
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

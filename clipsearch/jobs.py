"""Batch ingest: classify a pile of sources into jobs, persist them in SQLite, run them one at a time.

Used by both `clipsearch add` (runs inline) and `clipsearch ui` (background worker thread). One worker on purpose:
whisper and yt-dlp are heavy, and SQLite likes a single writer.
"""
import contextlib
import json
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import db

YT_RE = re.compile(r"(?:youtube\.com|youtu\.be)/", re.I)


def classify(item: str) -> tuple[str, str] | None:
    """One line of user input -> (kind, source). None for blanks/comments/unknowns."""
    item = item.strip().strip('"').strip("'")
    if not item or item.startswith("#"):
        return None
    if YT_RE.search(item):
        return "youtube", item
    path = Path(item).expanduser()
    if path.is_dir():
        return "obsidian", str(path)
    if path.suffix.lower() in {".xml", ".rss"} and path.is_file():
        return "rss", str(path)
    if item.startswith(("http://", "https://")):
        return "rss", item
    return None


def expand(items: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    """Inputs may be sources or .txt files listing sources (one per line). Returns (sources, unrecognized)."""
    out, bad = [], []
    for it in items:
        p = Path(it.strip()).expanduser()
        if p.suffix.lower() in {".txt", ".list"} and p.is_file():
            sub, sub_bad = expand(p.read_text(encoding="utf-8").splitlines())
            out += sub
            bad += sub_bad
            continue
        c = classify(it)
        if c:
            out.append(c)
        elif it.strip() and not it.strip().startswith("#"):
            bad.append(it.strip())
    seen = set()
    return [c for c in out if not (c in seen or seen.add(c))], bad


def enqueue(conn, sources: list[tuple[str, str]], *, show=None, limit=None, match=None, model=None,
            index=True, embed=False) -> str:
    batch = uuid.uuid4().hex[:8]
    for kind, src in sources:
        params = {"source": src, "show": show, "limit": limit, "match": match}
        if kind == "rss":
            params["model"] = model
        conn.execute("INSERT INTO jobs(kind, params, batch) VALUES (?,?,?)", (kind, json.dumps(params), batch))
    if index and sources:
        conn.execute("INSERT INTO jobs(kind, params, batch) VALUES ('index', ?, ?)",
                     (json.dumps({"embed": embed}), batch))
    conn.commit()
    return batch


def enqueue_one(conn, kind: str, params: dict) -> int:
    cur = conn.execute("INSERT INTO jobs(kind, params) VALUES (?,?)", (kind, json.dumps(params)))
    conn.commit()
    return cur.lastrowid


def run_job(kind: str, p: dict) -> None:
    if kind == "youtube":
        from .ingest_youtube import ingest
        ingest(p["source"], limit=p.get("limit"), title_filter=p.get("match"), show=p.get("show"))
    elif kind == "rss":
        from .ingest_rss import ingest
        ingest(p["source"], limit=p.get("limit"), title_filter=p.get("match"), model_size=p.get("model"), show=p.get("show"))
    elif kind == "obsidian":
        from .ingest_obsidian import ingest
        ingest(p["source"], show=p.get("show"))
    elif kind == "index":
        from .index import build
        build(embeddings=p.get("embed", False))
    elif kind == "cut":
        from .cut import run
        run(p["cutlist"], p["out"], reencode=not p.get("copy"))
    else:
        raise ValueError(f"unknown job kind {kind!r}")


class _JobLog:
    """File-like: buffers job output and flushes it to jobs.log at most once a second."""

    def __init__(self, conn, job_id: int, echo=None):
        self.conn, self.job_id, self.echo, self.buf, self.last = conn, job_id, echo, "", 0.0

    def write(self, s: str) -> int:
        if self.echo:
            self.echo.write(s)
        self.buf += s
        if time.time() - self.last > 1:
            self.flush()
        return len(s)

    def flush(self) -> None:
        if self.buf:
            self.conn.execute("UPDATE jobs SET log = log || ? WHERE id=?", (self.buf, self.job_id))
            self.conn.commit()
            self.buf = ""
        self.last = time.time()


def run_next(conn, echo=None) -> bool:
    """Run the oldest queued job. Returns False if the queue is empty."""
    job = conn.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
    if not job:
        return False
    conn.execute("UPDATE jobs SET status='running', started=datetime('now') WHERE id=?", (job["id"],))
    conn.commit()
    if echo:
        echo.write(f"\n── job {job['id']}: {job['kind']} {json.loads(job['params']).get('source', '')}\n")
    log = _JobLog(conn, job["id"], echo)
    status = "done"
    with contextlib.redirect_stdout(log):
        try:
            run_job(job["kind"], json.loads(job["params"]))
        except BaseException as e:      # SystemExit from ingest helpers included
            status = "failed"
            print(f"\n!! {e}" if isinstance(e, SystemExit) else "\n" + traceback.format_exc())
    log.flush()
    conn.execute("UPDATE jobs SET status=?, finished=datetime('now') WHERE id=?", (status, job["id"]))
    conn.commit()
    return True


def run_all(conn) -> None:
    """Run the queue to empty, echoing job output to the terminal."""
    import sys
    while run_next(conn, sys.stdout):
        pass


class Worker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="clipsearch-worker")
        self.wake = threading.Event()

    def run(self) -> None:
        conn = db.connect()
        # anything left 'running' was interrupted by a restart
        conn.execute("UPDATE jobs SET status='failed', log = log || '\n!! interrupted (server restarted)' "
                     "WHERE status='running'")
        conn.commit()
        while True:
            if not run_next(conn):
                self.wake.wait(5)
                self.wake.clear()

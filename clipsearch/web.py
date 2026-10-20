"""`clipsearch ui` — local web UI: batch-add sources, watch the job queue, search dialogue, build cutlists.

Stdlib only (http.server + a JSON API + one static page). Binds 127.0.0.1 by default; there is no auth,
so only expose it (--host 0.0.0.0) on a network you trust.
"""
import csv
import json
import mimetypes
import re
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import db, jobs, search, shows
from .media import VID_RE
from .timeutil import fmt_ts, yt_link

PAGE = Path(__file__).with_name("ui.html")
FEED_DIR = Path("data/feeds")
CUT_DIR = Path("cutlists")


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", s).strip("-")[:60] or "untitled"


def _hit_json(conn, h: search.Hit, ctx: int) -> dict:
    vid = VID_RE.search(h.url or "")
    return {
        "episode_id": h.episode_id, "title": h.title, "url": h.url, "video_id": vid.group(1) if vid else None,
        "has_media": bool(h.media_path and Path(h.media_path).exists()),
        "start": h.start, "end": h.end, "at": h.at, "score": h.score, "via": sorted(h.via),
        "speakers": h.speakers, "text": h.text,
        "context": search.context(conn, h, ctx) if h.at is not None else None,
    }


# ---------- handlers: each returns a JSON-able object ----------

def api_stats(conn, q):
    eps = {r["source"]: r["n"] for r in conn.execute("SELECT source, COUNT(*) n FROM episodes GROUP BY source")}
    chunks, emb = conn.execute("SELECT COUNT(*), COUNT(embedding) FROM chunks").fetchone()
    lines = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    unindexed = conn.execute("SELECT COUNT(*) FROM episodes WHERE indexed=0").fetchone()[0]
    themes = list(tomllib.loads(Path("themes.toml").read_text())) if Path("themes.toml").exists() else []
    return {"episodes": eps, "lines": lines, "chunks": chunks, "embedded": emb, "unindexed": unindexed,
            "themes": themes, "shows": shows.listing(conn)}


def api_search(conn, q):
    get = lambda k, d=None: q.get(k, [d])[0]
    query, mode = get("q", ""), get("mode", "grep")
    title, speaker, show = get("match") or None, get("speaker") or None, get("show") or None
    limit, near, ctx = int(get("limit", 100)), int(get("near", 15)), int(get("ctx", 1))
    if mode == "theme":
        themes = tomllib.loads(Path("themes.toml").read_text())
        hits = search.run_theme(themes[query], speaker, title, show)
    elif mode == "grep":
        hits = search.grep(conn, query, speaker=speaker, title=title, near=near, limit=limit, show=show)
    else:
        hits = search.run(query, mode, limit, speaker, 0.0, title, near, show)
    names = dict(conn.execute("SELECT e.id, sh.name FROM episodes e JOIN shows sh ON sh.id=e.show_id").fetchall())
    return {"hits": [_hit_json(conn, h, ctx) | {"show": names.get(h.episode_id)} for h in hits],
            "episodes": len({h.episode_id for h in hits})}


def api_episodes(conn, q):
    rows = conn.execute(
        """SELECT e.id, e.source, e.title, e.url, e.published, e.media_path, e.indexed, sh.name AS show,
                  (SELECT COUNT(*) FROM segments s WHERE s.episode_id=e.id) AS lines
           FROM episodes e LEFT JOIN shows sh ON sh.id=e.show_id ORDER BY e.published DESC, e.id DESC""").fetchall()
    return [dict(r) | {"has_media": bool(r["media_path"] and Path(r["media_path"]).exists())} for r in rows]


def api_jobs(conn, q):
    rows = conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT 300").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["params"] = json.loads(d["params"])
        d["log"] = d["log"][-4000:]
        out.append(d)
    return out


def api_add(conn, body, worker):
    srcs, bad = jobs.expand(body.get("text", "").splitlines())
    FEED_DIR.mkdir(parents=True, exist_ok=True)
    for f in body.get("files", []):              # uploaded .xml feeds / .txt lists
        name = _slug(Path(f["name"]).stem) + Path(f["name"]).suffix.lower()
        if name.endswith((".txt", ".list")):
            s2, b2 = jobs.expand(f["content"].splitlines())
            srcs, bad = srcs + s2, bad + b2
        else:
            dest = FEED_DIR / name
            dest.write_text(f["content"], encoding="utf-8")
            srcs.append(("rss", str(dest)))
    if body.get("kind") in {"youtube", "rss"}:   # override auto-detection for URL items
        srcs = [(body["kind"] if k in {"youtube", "rss"} else k, s) for k, s in srcs]
    lim = int(body["limit"]) if body.get("limit") else None
    batch = jobs.enqueue(conn, srcs, show=body.get("show") or None, limit=lim, match=body.get("match") or None,
                         model=body.get("model") or None, index=body.get("index", True), embed=body.get("embed", False))
    worker.wake.set()
    return {"batch": batch, "queued": len(srcs), "unrecognized": bad}


def api_job_action(conn, body, worker):
    jid, action = int(body["id"]), body["action"]
    if action == "cancel":
        conn.execute("UPDATE jobs SET status='cancelled' WHERE id=? AND status='queued'", (jid,))
    elif action == "retry":
        conn.execute("UPDATE jobs SET status='queued', log='', started=NULL, finished=NULL WHERE id=? "
                     "AND status IN ('failed','cancelled')", (jid,))
        worker.wake.set()
    elif action == "clear":
        conn.execute("DELETE FROM jobs WHERE status IN ('done','cancelled')")
    conn.commit()
    return {"ok": True}


def api_set_show(conn, body, worker):
    ids = [int(i) for i in body["ids"]]
    sid = db.show_id(conn, body.get("show"))
    conn.executemany("UPDATE episodes SET show_id=? WHERE id=?", [(sid, i) for i in ids])
    conn.commit()
    return {"ok": True}


def api_delete_episode(conn, body, worker):
    conn.execute("DELETE FROM episodes WHERE id=?", (int(body["id"]),))
    conn.commit()
    return {"ok": True}


# ---------- clips / cutlists ----------

def api_clips(conn, q):
    lists = [r[0] for r in conn.execute("SELECT DISTINCT list FROM clips ORDER BY list")]
    name = q.get("list", [lists[0] if lists else "default"])[0]
    rows = conn.execute(
        """SELECT c.*, e.title, e.url, e.media_path FROM clips c JOIN episodes e ON e.id=c.episode_id
           WHERE c.list=? ORDER BY c.position""", (name,)).fetchall()
    clips = []
    for r in rows:
        vid = VID_RE.search(r["url"] or "")
        clips.append(dict(r) | {"video_id": vid.group(1) if vid else None,
                                "has_media": bool(r["media_path"] and Path(r["media_path"]).exists())})
    return {"lists": lists, "list": name, "clips": clips}


def api_clip_add(conn, body, worker):
    pos = conn.execute("SELECT COALESCE(MAX(position), 0) + 1 FROM clips WHERE list=?", (body["list"],)).fetchone()[0]
    conn.execute("INSERT INTO clips(list, position, episode_id, start, end, at, text) VALUES (?,?,?,?,?,?,?)",
                 (body["list"], pos, body["episode_id"], body["start"], body["end"], body.get("at"), body.get("text")))
    conn.commit()
    return {"ok": True}


def api_clip_update(conn, body, worker):
    cid = int(body["id"])
    if body.get("delete"):
        conn.execute("DELETE FROM clips WHERE id=?", (cid,))
    elif "move" in body:                        # -1 up, +1 down: swap with neighbour
        c = conn.execute("SELECT list, position FROM clips WHERE id=?", (cid,)).fetchone()
        op, order = ("<", "DESC") if body["move"] < 0 else (">", "ASC")
        n = conn.execute(f"SELECT id, position FROM clips WHERE list=? AND position {op} ? ORDER BY position {order} LIMIT 1",
                         (c["list"], c["position"])).fetchone()
        if n:
            conn.execute("UPDATE clips SET position=? WHERE id=?", (n["position"], cid))
            conn.execute("UPDATE clips SET position=? WHERE id=?", (c["position"], n["id"]))
    else:
        for k in ("start", "end", "offset", "note"):
            if k in body:
                conn.execute(f"UPDATE clips SET {k}=? WHERE id=?", (body[k], cid))
    conn.commit()
    return {"ok": True}


def export_cutlist(conn, name: str) -> Path:
    rows = api_clips(conn, {"list": [name]})["clips"]
    CUT_DIR.mkdir(parents=True, exist_ok=True)
    path = CUT_DIR / f"{_slug(name)}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=search.CUT_FIELDS)
        w.writeheader()
        for c in rows:
            w.writerow({"keep": "y", "title": c["title"], "speakers": "", "start": fmt_ts(c["start"]),
                        "end": fmt_ts(c["end"]), "pad": 0, "offset": c["offset"], "media_path": c["media_path"] or "",
                        "link": yt_link(c["url"], c["at"] if c["at"] is not None else c["start"]),
                        "text": (c["text"] or "")[:500]})
    return path


def api_export(conn, body, worker):
    path = export_cutlist(conn, body["list"])
    out = {"path": str(path)}
    if body.get("render"):
        dest = Path("out") / f"{_slug(body['list'])}.{body.get('format', 'mp4')}"
        out["job"] = jobs.enqueue_one(conn, "cut", {"cutlist": str(path), "out": str(dest)})
        out["out"] = str(dest)
        worker.wake.set()
    return out


GET = {"/api/stats": api_stats, "/api/search": api_search, "/api/episodes": api_episodes,
       "/api/jobs": api_jobs, "/api/clips": api_clips}
POST = {"/api/add": api_add, "/api/job": api_job_action, "/api/episode/delete": api_delete_episode, "/api/episode/show": api_set_show,
        "/api/clip/add": api_clip_add, "/api/clip": api_clip_update, "/api/export": api_export}


class Handler(BaseHTTPRequestHandler):
    worker: jobs.Worker = None

    def log_message(self, fmt, *args):     # keep the terminal quiet; errors still surface via _send
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json", extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=str).encode())

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/":
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if u.path.startswith("/media/"):
                return self._media(int(u.path.rsplit("/", 1)[1]))
            if u.path == "/api/cutlist.csv":
                path = export_cutlist(db.connect(), parse_qs(u.query)["list"][0])
                return self._send(200, path.read_bytes(), "text/csv",
                                  {"Content-Disposition": f'attachment; filename="{path.name}"'})
            fn = GET.get(u.path)
            if not fn:
                return self._json({"error": "not found"}, 404)
            self._json(fn(db.connect(), parse_qs(u.query)))
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 400)

    def do_POST(self):
        fn = POST.get(urlparse(self.path).path)
        if not fn:
            return self._json({"error": "not found"}, 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self._json(fn(db.connect(), body, self.worker))
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 400)

    def _media(self, episode_id: int):
        """Serve an episode's local media file (only paths registered in the DB), with Range for seeking."""
        row = db.connect().execute("SELECT media_path FROM episodes WHERE id=?", (episode_id,)).fetchone()
        path = Path(row["media_path"]) if row and row["media_path"] else None
        if not path or not path.is_file():
            return self._json({"error": "no media"}, 404)
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if not m:
            start, end, code = 0, size - 1, 200
        else:
            a, b = m.groups()
            start = int(a) if a else max(0, size - int(b))
            end = min(int(b), size - 1) if a and b else size - 1
            code = 206
        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            while length > 0:
                block = f.read(min(1 << 20, length))
                if not block:
                    break
                try:
                    self.wfile.write(block)
                except (BrokenPipeError, ConnectionResetError):
                    return
                length -= len(block)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    db.connect()                            # create/migrate before threads start
    worker = jobs.Worker()
    worker.start()
    Handler.worker = worker
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"clipsearch ui -> http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}  (ctrl-c to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass

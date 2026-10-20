"""Keyword (FTS5), semantic (cosine over stored embeddings), and hybrid (RRF) search."""
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import db
from .timeutil import fmt_ts, yt_link


@dataclass
class Hit:
    chunk_id: int
    episode_id: int
    title: str
    url: str | None
    media_path: str | None
    start: float
    end: float
    text: str
    speakers: str | None
    score: float = 0.0
    via: set = field(default_factory=set)


_COLS = """c.id AS chunk_id, c.episode_id, e.title, e.url, e.media_path, c.start, c.end, c.text, c.speakers"""


def _fts_query(q: str) -> str:
    """Pass through quoted phrases / FTS operators; otherwise AND the bare words together."""
    if any(ch in q for ch in '"*') or re.search(r"\b(AND|OR|NOT|NEAR)\b", q):
        return q
    words = re.findall(r"[\w']+", q)
    return " ".join(f'"{w}"' for w in words)


def _speaker_ok(h: Hit, speaker: str | None) -> bool:
    if not speaker or not h.speakers:   # no diarization yet -> don't filter
        return True
    return speaker.lower() in h.speakers.lower()


def keyword(conn, q: str, limit: int = 50, speaker: str | None = None) -> list[Hit]:
    rows = conn.execute(
        f"""SELECT {_COLS}, bm25(chunks_fts) AS rank
            FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid JOIN episodes e ON e.id = c.episode_id
            WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?""",
        (_fts_query(q), limit * 3),
    ).fetchall()
    hits = [Hit(**{k: r[k] for k in r.keys() if k != "rank"}, score=-r["rank"], via={"kw"}) for r in rows]
    return [focus(conn, h, q) for h in hits if _speaker_ok(h, speaker)][:limit]


def semantic(conn, q: str, limit: int = 50, speaker: str | None = None, min_score: float = 0.0) -> list[Hit]:
    from .index import embed

    rows = conn.execute(
        f"SELECT {_COLS}, c.embedding FROM chunks c JOIN episodes e ON e.id = c.episode_id WHERE c.embedding IS NOT NULL"
    ).fetchall()
    if not rows:
        print("no embeddings yet — run `jbp index` with sentence-transformers installed")
        return []
    mat = np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    qv = embed([q])[0]
    scores = mat @ qv
    order = np.argsort(-scores)
    hits = []
    for i in order:
        if scores[i] < min_score or len(hits) >= limit:
            break
        r = rows[i]
        h = Hit(**{k: r[k] for k in r.keys() if k != "embedding"}, score=float(scores[i]), via={"sem"})
        if _speaker_ok(h, speaker):
            hits.append(h)
    return hits


def rrf(*lists: list[Hit], k: int = 60) -> list[Hit]:
    merged: dict[int, Hit] = {}
    for lst in lists:
        for rank, h in enumerate(lst):
            m = merged.setdefault(h.chunk_id, Hit(**{**h.__dict__, "score": 0.0, "via": set()}))
            m.score += 1 / (k + rank + 1)
            m.via |= h.via
    return sorted(merged.values(), key=lambda h: -h.score)


def focus(conn, hit: Hit, q: str, before: float = 12.0, after: float = 30.0) -> Hit:
    """Tighten a keyword hit to the first segment in the chunk that contains a query word."""
    words = [w.lower() for w in re.findall(r"[\w']{3,}", q) if w.upper() not in {"AND", "OR", "NOT", "NEAR"}]
    if not words:
        return hit
    segs = conn.execute(
        "SELECT start, end, text FROM segments WHERE episode_id=? AND start>=? AND start<? ORDER BY start",
        (hit.episode_id, hit.start, hit.end),
    ).fetchall()
    best = max(segs, key=lambda s: sum(w in s["text"].lower() for w in words), default=None)
    if best is None or not any(w in best["text"].lower() for w in words):
        return hit
    hit.start = max(hit.start, best["start"] - before)
    hit.end = min(hit.end, best["end"] + after)
    return hit


def dedupe_overlaps(hits: list[Hit], gap: float = 10.0) -> list[Hit]:
    """Drop lower-ranked hits that overlap an already-kept hit in the same episode."""
    kept: list[Hit] = []
    for h in hits:
        dup = next((k for k in kept if k.episode_id == h.episode_id
                    and h.start <= k.end + gap and h.end >= k.start - gap), None)
        if dup:
            dup.via |= h.via
        else:
            kept.append(Hit(**h.__dict__))
    return kept


def run(q: str, mode: str = "hybrid", limit: int = 20, speaker: str | None = None, min_score: float = 0.0) -> list[Hit]:
    conn = db.connect()
    if mode == "kw":
        hits = keyword(conn, q, limit * 2, speaker)
    elif mode == "sem":
        hits = semantic(conn, q, limit * 2, speaker, min_score)
    else:
        hits = rrf(keyword(conn, q, limit * 3, speaker), semantic(conn, q, limit * 3, speaker, min_score))
    return dedupe_overlaps(hits)[:limit]


def run_theme(theme: dict, speaker: str | None = None) -> list[Hit]:
    conn = db.connect()
    limit = theme.get("limit", 30)
    lists = [semantic(conn, theme["description"], limit * 3, speaker, theme.get("min_score", 0.0))]
    for kw in theme.get("keywords", []):
        try:
            lists.append(keyword(conn, kw, limit * 2, speaker))
        except Exception as e:  # bad FTS syntax shouldn't kill the run
            print(f"  ! keyword {kw!r}: {e}")
    return dedupe_overlaps(rrf(*lists))[:limit]


def print_hits(hits: list[Hit], width: int = 220) -> None:
    for n, h in enumerate(hits, 1):
        snippet = h.text if len(h.text) <= width else h.text[:width] + "…"
        print(f"\n[{n}] {h.title}  {fmt_ts(h.start)}–{fmt_ts(h.end)}  ({'+'.join(sorted(h.via))} {h.score:.3g})")
        if h.url:
            print(f"    {yt_link(h.url, h.start)}")
        print(f"    {snippet}")


CUT_FIELDS = ["keep", "title", "start", "end", "pad", "offset", "media_path", "link", "text"]


def write_cutlist(hits: list[Hit], out: str, pad: float = 3.0) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CUT_FIELDS)
        w.writeheader()
        for h in hits:
            w.writerow({
                "keep": "y", "title": h.title, "start": fmt_ts(h.start), "end": fmt_ts(h.end),
                "pad": pad, "offset": 0, "media_path": h.media_path or "",
                "link": yt_link(h.url, h.start), "text": h.text[:500],
            })
    print(f"wrote {len(hits)} rows -> {out}  (edit keep/start/end/offset/media_path, then `jbp cut`)")

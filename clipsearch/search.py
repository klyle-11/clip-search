"""Keyword (FTS5), semantic (cosine over stored embeddings), hybrid (RRF), and grep (every occurrence)."""
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import db, dialogue
from .index import speakers_by_time
from .textnorm import norm, stem, tokens
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
    line_speaker: str | None = None   # speaker of the line a keyword hit was focused on
    at: float | None = None           # time of the matched words (keyword/grep hits)
    at_last: float | None = None      # start of the last line the match touches


_COLS = """c.id AS chunk_id, c.episode_id, e.title, e.url, e.media_path, c.start, c.end, c.text, c.speakers"""


BEFORE, AFTER = 12.0, 30.0   # default clip window around a matched line


def _ep_sql(title: str | None = None, show: str | None = None) -> tuple[str, tuple]:
    """Extra WHERE clauses restricting to a show (by name) and/or episode-title regex."""
    sql, args = "", ()
    if show:
        sql, args = " AND e.show_id IN (SELECT id FROM shows WHERE name = ?)", (show,)
    if title:
        sql, args = sql + " AND e.title REGEXP ?", args + (title,)
    return sql, args


MIN_SHARE = 0.4   # speaker must hold at least this much of the hit window's talk time


def speaker_share(conn, h: Hit, speaker: str) -> float | None:
    """Fraction of diarized talk time in [h.start, h.end) that belongs to `speaker` (None = not diarized)."""
    rows = conn.execute(
        """SELECT speaker, SUM(end - start) AS secs FROM segments
           WHERE episode_id=? AND start>=? AND start<? AND speaker IS NOT NULL GROUP BY speaker""",
        (h.episode_id, h.start, h.end),
    ).fetchall()
    total = sum(r["secs"] for r in rows)
    if not total:
        return None
    return sum(r["secs"] for r in rows if r["speaker"].lower() == speaker.lower()) / total


def _speaker_ok(conn, h: Hit, speaker: str | None) -> bool:
    if not speaker:
        return True
    if h.line_speaker and h.line_speaker.lower() == speaker.lower():
        return True
    share = speaker_share(conn, h, speaker)
    return share is None or share >= MIN_SHARE   # undiarized episodes pass through


def keyword(conn, q: str, limit: int = 50, speaker: str | None = None, title: str | None = None,
            near: int = 15, show: str | None = None) -> list[Hit]:
    query = dialogue.parse(q)
    tsql, targs = _ep_sql(title, show)
    rows = conn.execute(
        f"""SELECT {_COLS}, bm25(chunks_fts) AS rank
            FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid JOIN episodes e ON e.id = c.episode_id
            WHERE chunks_fts MATCH ?{tsql} ORDER BY rank LIMIT ?""",
        (query.fts(), *targs, limit * 3),
    ).fetchall()
    hits = [Hit(**{k: r[k] for k in r.keys() if k != "rank"}, score=-r["rank"], via={"kw"}) for r in rows]
    hits = [anchor(conn, h, query, speaker, near) for h in hits]
    return [h for h in hits if _speaker_ok(conn, h, speaker)][:limit]


def semantic(conn, q: str, limit: int = 50, speaker: str | None = None, min_score: float = 0.0,
             title: str | None = None, show: str | None = None) -> list[Hit]:
    from .index import embed

    tsql, targs = _ep_sql(title, show)
    rows = conn.execute(
        f"SELECT {_COLS}, c.embedding FROM chunks c JOIN episodes e ON e.id = c.episode_id WHERE c.embedding IS NOT NULL{tsql}",
        targs,
    ).fetchall()
    if not rows:
        print("no embeddings yet — run `clipsearch index` with sentence-transformers installed")
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
        if _speaker_ok(conn, h, speaker):
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


def _segs(conn, episode_id: int, lo: float, hi: float):
    return conn.execute(
        "SELECT start, end, text, speaker FROM segments WHERE episode_id=? AND start>=? AND start<? ORDER BY start",
        (episode_id, lo, hi),
    ).fetchall()


def _place(hit: Hit, segs, i: int, t: float, last: int) -> Hit:
    hit.at, hit.at_last, hit.line_speaker = t, segs[last]["start"], segs[i]["speaker"]
    hit.start = max(0.0, t - BEFORE)
    hit.end = segs[last]["end"] + AFTER
    return hit


def anchor(conn, hit: Hit, query: dialogue.Query, speaker: str | None = None, near: int = 15) -> Hit:
    """Pin a keyword hit to where the query actually occurs in the chunk (preferring `speaker`'s lines).

    Falls back to the line sharing the most query words when the scanner can't confirm a match
    (FTS's porter stemmer is looser than ours).
    """
    segs = _segs(conn, hit.episode_id, hit.start, hit.end)
    if not segs:
        return hit
    occs = dialogue.find(segs, query, near)
    if occs:
        mine = [o for o in occs if speaker and (segs[o.seg]["speaker"] or "").lower() == speaker.lower()]
        o = (mine or occs)[0]
        return _place(hit, segs, o.seg, o.t, o.seg + o.span - 1)
    want = {stem(w) for alt in query.alts for t in alt for w in t.words}
    overlap = [len(want & {stem(w) for w in tokens(norm(s["text"]))}) for s in segs]
    i = max(range(len(segs)), key=overlap.__getitem__)
    return _place(hit, segs, i, segs[i]["start"], i) if overlap[i] else hit


def grep(conn, q: str, *, speaker: str | None = None, title: str | None = None, near: int = 15,
         limit: int | None = None, show: str | None = None) -> list[Hit]:
    """Every occurrence of `q` across the corpus, in episode/time order (not ranked)."""
    query = dialogue.parse(q)
    tsql, targs = _ep_sql(title, show)
    rows = conn.execute(
        f"""SELECT c.episode_id, c.start, c.end, e.title, e.url, e.media_path
            FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid JOIN episodes e ON e.id = c.episode_id
            WHERE chunks_fts MATCH ?{tsql} ORDER BY e.published, c.episode_id, c.start""",
        (query.fts(), *targs),
    ).fetchall()
    # merge overlapping candidate chunks so each stretch of dialogue is scanned once
    spans: list[list] = []
    for r in rows:
        if spans and spans[-1][0]["episode_id"] == r["episode_id"] and r["start"] <= spans[-1][2]:
            spans[-1][2] = max(spans[-1][2], r["end"])
        else:
            spans.append([r, r["start"], r["end"]])
    hits: list[Hit] = []
    for r, lo, hi in spans:
        segs = _segs(conn, r["episode_id"], lo, hi)
        for o in dialogue.find(segs, query, near):
            last = o.seg + o.span - 1
            h = Hit(chunk_id=-1, episode_id=r["episode_id"], title=r["title"], url=r["url"],
                    media_path=r["media_path"], start=0, end=0,
                    text=" ".join(x["text"] for x in segs[o.seg:last + 1]), speakers=None, via={"grep"})
            _place(h, segs, o.seg, o.t, last)
            if not speaker or not h.line_speaker or h.line_speaker.lower() == speaker.lower():
                hits.append(h)
        if limit and len(hits) >= limit:
            break
    return hits[:limit] if limit else hits


def merge_close(hits: list[Hit], gap: float = 10.0) -> list[Hit]:
    """Time-ordered hits -> clips: occurrences close together in one episode become one longer clip."""
    out: list[Hit] = []
    for h in hits:
        if out and out[-1].episode_id == h.episode_id and h.start <= out[-1].end + gap:
            out[-1].end = max(out[-1].end, h.end)
            out[-1].text += " … " + h.text
        else:
            out.append(Hit(**h.__dict__))
    return out


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


def window_speakers(conn, hits: list[Hit]) -> list[Hit]:
    """Re-derive each hit's speakers from its (possibly focused) window rather than the whole chunk."""
    for h in hits:
        segs = conn.execute("SELECT start, end, speaker FROM segments WHERE episode_id=? AND start>=? AND start<?",
                            (h.episode_id, h.start, h.end)).fetchall()
        h.speakers = speakers_by_time(segs) or h.speakers
    return hits


def run(q: str, mode: str = "hybrid", limit: int = 20, speaker: str | None = None, min_score: float = 0.0,
        title: str | None = None, near: int = 15, show: str | None = None) -> list[Hit]:
    conn = db.connect()
    if mode == "kw":
        hits = keyword(conn, q, limit * 2, speaker, title, near, show)
    elif mode == "sem":
        hits = semantic(conn, q, limit * 2, speaker, min_score, title, show)
    else:
        hits = rrf(keyword(conn, q, limit * 3, speaker, title, near, show),
                   semantic(conn, q, limit * 3, speaker, min_score, title, show))
    return window_speakers(conn, dedupe_overlaps(hits)[:limit])


def run_theme(theme: dict, speaker: str | None = None, title: str | None = None, show: str | None = None) -> list[Hit]:
    """A theme may pin its own `show` in themes.toml; an explicit `show` overrides it."""
    conn = db.connect()
    limit = theme.get("limit", 30)
    show = show or theme.get("show")
    lists = [semantic(conn, theme["description"], limit * 3, speaker, theme.get("min_score", 0.0), title, show)]
    for kw in theme.get("keywords", []):
        try:
            lists.append(keyword(conn, kw, limit * 2, speaker, title, show=show))
        except Exception as e:  # bad FTS syntax shouldn't kill the run
            print(f"  ! keyword {kw!r}: {e}")
    return window_speakers(conn, dedupe_overlaps(rrf(*lists))[:limit])


def _bold(line: str) -> str:
    return f"\033[1m{line}\033[0m" if sys.stdout.isatty() else line


def context(conn, h: Hit, ctx: int = 1) -> list[dict]:
    """The matched line(s) plus `ctx` lines either side: [{start, end, text, speaker, match}]."""
    segs = _segs(conn, h.episode_id, h.at - 120, h.at + 120)
    first = max((i for i, s in enumerate(segs) if s["start"] <= h.at + 0.01), default=0)
    last = max((i for i, s in enumerate(segs) if s["start"] <= (h.at_last or h.at) + 0.01), default=first)
    return [{**dict(segs[i]), "match": first <= i <= last}
            for i in range(max(0, first - ctx), min(len(segs), last + 1 + ctx))]


def context_lines(conn, h: Hit, ctx: int = 1) -> list[str]:
    """context() formatted for the terminal; matched lines marked >."""
    out = []
    for s in context(conn, h, ctx):
        spk = f"{s['speaker']}: " if s["speaker"] else ""
        line = f"{fmt_ts(s['start'])}  {spk}{s['text']}"
        out.append(_bold(f"  > {line}") if s["match"] else f"    {line}")
    return out


def print_hits(hits: list[Hit], width: int = 220, ctx: int = 1) -> None:
    conn = db.connect()
    for n, h in enumerate(hits, 1):
        when = fmt_ts(h.at) if h.at is not None else f"{fmt_ts(h.start)}–{fmt_ts(h.end)}"
        spk = f"  [{h.speakers}]" if h.speakers else ""
        print(f"\n[{n}] {h.title}  {when}  ({'+'.join(sorted(h.via))} {h.score:.3g}){spk}")
        if h.url:
            print(f"    {yt_link(h.url, h.at if h.at is not None else h.start)}")
        if h.at is not None:
            print("\n".join(context_lines(conn, h, ctx)))
        else:
            print(f"    {h.text if len(h.text) <= width else h.text[:width] + '…'}")


def print_grep(hits: list[Hit], ctx: int = 1) -> None:
    conn = db.connect()
    last_ep = None
    for h in hits:
        if h.episode_id != last_ep:
            print(f"\n== {h.title}")
            last_ep = h.episode_id
        link = f"   {yt_link(h.url, h.at)}" if h.url else ""
        print(f"  @ {fmt_ts(h.at)}{link}")
        print("\n".join(context_lines(conn, h, ctx)))
    eps = len({h.episode_id for h in hits})
    print(f"\n{len(hits)} occurrences in {eps} episodes")


def similar_terms(conn, word: str, n: int = 20) -> list[tuple[str, int, int]]:
    """Indexed terms spelled like `word` -> [(term, chunks, occurrences)]. Terms are porter stems."""
    import difflib
    rows = {r["term"]: (r["doc"], r["cnt"]) for r in conn.execute("SELECT term, doc, cnt FROM chunks_vocab")}
    w = (tokens(norm(word)) or [word.lower()])[0]
    close = difflib.get_close_matches(w, rows, n=n, cutoff=0.6)
    close += [t for t in rows if t.startswith(w[:4]) and t not in close][: max(0, n - len(close))]
    return sorted(((t, *rows[t]) for t in close), key=lambda x: -x[2])


CUT_FIELDS = ["keep", "title", "speakers", "start", "end", "pad", "offset", "media_path", "link", "text"]


def write_cutlist(hits: list[Hit], out: str, pad: float = 3.0) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CUT_FIELDS)
        w.writeheader()
        for h in hits:
            w.writerow({
                "keep": "y", "title": h.title, "speakers": h.speakers or "", "start": fmt_ts(h.start), "end": fmt_ts(h.end),
                "pad": pad, "offset": 0, "media_path": h.media_path or "",
                "link": yt_link(h.url, h.start), "text": h.text[:500],
            })
    print(f"wrote {len(hits)} rows -> {out}  (edit keep/start/end/offset/media_path, then `clipsearch cut`)")

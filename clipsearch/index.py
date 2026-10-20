"""Group segments into overlapping ~45s chunks, then (optionally) embed them."""
import os

import numpy as np

from . import db
from .config import env
from .textnorm import norm

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    return _model


def embed(texts: list[str]) -> np.ndarray:
    return get_model().encode(texts, batch_size=64, normalize_embeddings=True,
                              show_progress_bar=len(texts) > 200).astype(np.float32)


def speakers_by_time(segs) -> str | None:
    """Comma-joined speakers in these segments, most talk time first."""
    secs: dict[str, float] = {}
    for s in segs:
        if s["speaker"]:
            secs[s["speaker"]] = secs.get(s["speaker"], 0.0) + s["end"] - s["start"]
    return ",".join(sorted(secs, key=lambda k: -secs[k])) or None


def make_chunks(segs, window: float = 45.0, stride: float = 30.0):
    """segs: rows with start/end/text/speaker sorted by start. Yields (start, end, text, speakers)."""
    if not segs:
        return
    t = segs[0]["start"]
    last = segs[-1]["end"]
    i0 = 0
    while t < last:
        while i0 < len(segs) and segs[i0]["end"] <= t:
            i0 += 1
        win = []
        j = i0
        while j < len(segs) and segs[j]["start"] < t + window:
            win.append(segs[j])
            j += 1
        if win:
            yield (win[0]["start"], win[-1]["end"], " ".join(s["text"] for s in win), speakers_by_time(win))
        t += stride


def build(*, embeddings: bool = True, rechunk: bool = False) -> None:
    conn = db.connect()
    where = "" if rechunk else "WHERE indexed=0"
    eps = conn.execute(f"SELECT id, title FROM episodes {where}").fetchall()
    for ep in eps:
        conn.execute("DELETE FROM chunks WHERE episode_id=?", (ep["id"],))
        segs = conn.execute(
            "SELECT start, end, text, speaker FROM segments WHERE episode_id=? ORDER BY start", (ep["id"],)
        ).fetchall()
        conn.executemany(
            "INSERT INTO chunks(episode_id, start, end, text, norm, speakers) VALUES (?,?,?,?,?,?)",
            [(ep["id"], st, en, text, norm(text), spk) for st, en, text, spk in make_chunks(segs)],
        )
        conn.execute("UPDATE episodes SET indexed=1 WHERE id=?", (ep["id"],))
        conn.commit()
    print(f"chunked {len(eps)} episodes")

    if not embeddings:
        return
    try:
        get_model()
    except ImportError:
        print("sentence-transformers not installed; skipping embeddings (keyword search still works).")
        return
    todo = conn.execute("SELECT id, text FROM chunks WHERE embedding IS NULL").fetchall()
    for k in range(0, len(todo), 2048):
        batch = todo[k:k + 2048]
        vecs = embed([r["text"] for r in batch])
        conn.executemany("UPDATE chunks SET embedding=? WHERE id=?",
                         [(v.tobytes(), r["id"]) for v, r in zip(vecs, batch)])
        conn.commit()
        print(f"embedded {min(k + 2048, len(todo))}/{len(todo)}")

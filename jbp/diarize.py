"""Speaker diarization + voiceprint matching, written back into segments.speaker / chunks.speakers.

Flow: pyannote splits an episode's audio into speaker turns (SPEAKER_00, SPEAKER_01, ...). Each cluster
gets a voice embedding (mean over its longest turns). Clusters whose embedding is close enough to an
enrolled voiceprint (e.g. "joe") take that name; the rest keep their raw label. Every transcript line
gets the speaker it overlaps most.

Enrollment: diarize one episode, `jbp speakers <ep>` to see who's who, then `jbp label <ep> SPEAKER_01 joe`.
That cluster's embedding becomes (or is averaged into) Joe's voiceprint and all diarized episodes are
re-matched — no audio reprocessing needed.

Needs `pip install -e ".[diarize]"`, ffmpeg, and HF_TOKEN in .env with the pyannote model terms accepted.
"""
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import db
from .index import speakers_by_time
from .media import select_episodes
from .timeutil import fmt_ts, yt_link

MIN_TURN = 1.5        # seconds; shorter turns make noisy embeddings
MAX_CROP = 10.0       # embed at most this much of any one turn
MAX_TURNS = 8         # per cluster


def _threshold() -> float:
    return float(os.getenv("JBP_VOICE_THRESHOLD", "0.55"))


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n else v


# ---------- audio / model backend ----------

def to_wav(src: str, dest: str, start: float | None = None, end: float | None = None) -> str:
    """16 kHz mono WAV: decodes reliably in every pyannote/torchaudio version."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.2f}"]
    cmd += ["-i", src]
    if end is not None:
        cmd += ["-t", f"{end - (start or 0):.2f}"]
    subprocess.run(cmd + ["-vn", "-ac", "1", "-ar", "16000", dest], check=True)
    return dest


def _from_pretrained(cls, name: str, token: str | None):
    try:
        return cls.from_pretrained(name, token=token)            # pyannote.audio >= 4
    except TypeError:
        return cls.from_pretrained(name, use_auth_token=token)   # 3.x


class Pyannote:
    def __init__(self):
        import torch
        from pyannote.audio import Inference, Model, Pipeline

        token = os.getenv("HF_TOKEN")
        self.pipeline = _from_pretrained(
            Pipeline, os.getenv("JBP_DIARIZE_MODEL", "pyannote/speaker-diarization-3.1"), token)
        model = _from_pretrained(Model, os.getenv("JBP_VOICE_MODEL", "pyannote/wespeaker-voxceleb-resnet34-LM"), token)
        if self.pipeline is None or model is None:
            raise SystemExit("couldn't load pyannote models — set HF_TOKEN and accept the model terms on huggingface.co")
        self.inference = Inference(model, window="whole")
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = None
        if dev is not None:
            self.pipeline.to(dev)
            self.inference.to(dev)

    def diarize(self, wav: str, **kw) -> list[tuple[float, float, str]]:
        out = self.pipeline(wav, **{k: v for k, v in kw.items() if v is not None})
        ann = (getattr(out, "exclusive_speaker_diarization", None)
               or getattr(out, "speaker_diarization", None) or out)
        return [(t.start, t.end, lbl) for t, _, lbl in ann.itertracks(yield_label=True)]

    def embed(self, wav: str, spans: list[tuple[float, float]]) -> np.ndarray:
        from pyannote.core import Segment
        return np.stack([_norm(self.inference.crop(wav, Segment(s, e))) for s, e in spans])


# ---------- pure logic ----------

def cluster_spans(turns: list[tuple[float, float, str]]) -> dict[str, list[tuple[float, float]]]:
    """Longest turns per cluster, each cropped to its middle MAX_CROP seconds."""
    by: dict[str, list[tuple[float, float]]] = {}
    for s, e, lbl in sorted(turns, key=lambda t: t[0] - t[1]):
        if e - s < MIN_TURN or len(by.setdefault(lbl, [])) >= MAX_TURNS:
            continue
        mid, half = (s + e) / 2, min(e - s, MAX_CROP) / 2
        by[lbl].append((mid - half, mid + half))
    return by


def assign(segments: list[tuple[float, float]], turns: list[tuple[float, float, str]]) -> list[str | None]:
    """Label each (start, end) segment with the turn label it overlaps most (None if no overlap)."""
    if not turns:
        return [None] * len(segments)
    labels = sorted({t[2] for t in turns})
    ts = np.array([t[0] for t in turns])
    te = np.array([t[1] for t in turns])
    tl = np.array([labels.index(t[2]) for t in turns])
    out = []
    for s, e in segments:
        e = max(e, s + 0.01)
        ov = np.clip(np.minimum(te, e) - np.maximum(ts, s), 0, None)
        if not ov.any():
            out.append(None)
            continue
        per = np.bincount(tl, weights=ov, minlength=len(labels))
        out.append(labels[int(per.argmax())])
    return out


def match(emb: np.ndarray | None, voiceprints: dict[str, np.ndarray], threshold: float) -> tuple[str | None, float | None]:
    if emb is None or not voiceprints:
        return None, None
    name, score = max(((n, float(v @ emb)) for n, v in voiceprints.items()), key=lambda x: x[1])
    return (name, score) if score >= threshold else (None, score)


# ---------- db ----------

def voiceprints(conn) -> dict[str, np.ndarray]:
    return {r["name"]: np.frombuffer(r["embedding"], dtype=np.float32)
            for r in conn.execute("SELECT name, embedding FROM voiceprints")}


def enroll_vec(conn, name: str, vec: np.ndarray) -> int:
    """Fold an embedding into a voiceprint (running mean). Returns the new sample count."""
    vec = _norm(vec)
    row = conn.execute("SELECT embedding, n FROM voiceprints WHERE name=?", (name,)).fetchone()
    if row:
        old = np.frombuffer(row["embedding"], dtype=np.float32)
        if old.shape != vec.shape:
            raise SystemExit(f"voiceprint {name!r} was made with a different voice model ({old.shape} vs {vec.shape})")
        vec, n = _norm(old * row["n"] + vec), row["n"] + 1
    else:
        n = 1
    conn.execute("INSERT OR REPLACE INTO voiceprints(name, embedding, n) VALUES (?,?,?)", (name, vec.tobytes(), n))
    return n


def refresh_chunk_speakers(conn, episode_id: int) -> None:
    segs = conn.execute("SELECT start, end, speaker FROM segments WHERE episode_id=? ORDER BY start",
                        (episode_id,)).fetchall()
    starts = np.array([s["start"] for s in segs])
    rows = []
    for c in conn.execute("SELECT id, start, end FROM chunks WHERE episode_id=?", (episode_id,)).fetchall():
        lo, hi = np.searchsorted(starts, c["start"]), np.searchsorted(starts, c["end"])
        rows.append((speakers_by_time(segs[lo:hi]), c["id"]))
    conn.executemany("UPDATE chunks SET speakers=? WHERE id=?", rows)


def apply_names(conn, episode_id: int, threshold: float | None = None) -> None:
    """Re-match this episode's non-manual clusters against voiceprints, then rewrite speakers."""
    threshold = _threshold() if threshold is None else threshold
    vps = voiceprints(conn)
    for r in conn.execute("SELECT cluster, embedding FROM ep_speakers WHERE episode_id=? AND manual=0",
                          (episode_id,)).fetchall():
        emb = np.frombuffer(r["embedding"], dtype=np.float32) if r["embedding"] else None
        name, score = match(emb, vps, threshold)
        conn.execute("UPDATE ep_speakers SET name=?, score=? WHERE episode_id=? AND cluster=?",
                     (name, score, episode_id, r["cluster"]))
    conn.execute(
        """UPDATE segments SET speaker = COALESCE(
               (SELECT name FROM ep_speakers p WHERE p.episode_id=segments.episode_id AND p.cluster=segments.cluster),
               cluster)
           WHERE episode_id=?""", (episode_id,))
    refresh_chunk_speakers(conn, episode_id)


def store(conn, episode_id: int, turns: list[tuple[float, float, str]], embs: dict[str, np.ndarray]) -> None:
    segs = conn.execute("SELECT id, start, end FROM segments WHERE episode_id=? ORDER BY start",
                        (episode_id,)).fetchall()
    labels = assign([(s["start"], s["end"]) for s in segs], turns)
    conn.executemany("UPDATE segments SET cluster=? WHERE id=?", [(l, s["id"]) for l, s in zip(labels, segs)])
    secs: dict[str, float] = {}
    for s, e, lbl in turns:
        secs[lbl] = secs.get(lbl, 0.0) + (e - s)
    conn.execute("DELETE FROM ep_speakers WHERE episode_id=?", (episode_id,))
    conn.executemany(
        "INSERT INTO ep_speakers(episode_id, cluster, seconds, embedding) VALUES (?,?,?,?)",
        [(episode_id, lbl, sec, embs[lbl].tobytes() if lbl in embs else None) for lbl, sec in secs.items()])
    apply_names(conn, episode_id)
    conn.execute("UPDATE episodes SET diarized=1 WHERE id=?", (episode_id,))


def diarize_episode(conn, ep, backend, **kw) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wav = to_wav(ep["media_path"], str(Path(tmp) / "a.wav"))
        turns = backend.diarize(wav, **kw)
        embs = {lbl: _norm(backend.embed(wav, spans).mean(axis=0))
                for lbl, spans in cluster_spans(turns).items() if spans}
    store(conn, ep["id"], turns, embs)


# ---------- commands ----------

def run(*, episode=None, match_title=None, limit=None, redo=False, num_speakers=None,
        min_speakers=None, max_speakers=None, backend=None) -> None:
    conn = db.connect()
    where = "media_path IS NOT NULL AND media_path != ''" + ("" if redo else " AND diarized=0")
    eps = [e for e in select_episodes(conn, episode=episode, match=match_title, limit=None, where=where)
           if Path(e["media_path"]).exists()]
    eps = eps[:limit] if limit else eps
    if not eps:
        print("nothing to diarize — episodes need a local media_path (`jbp fetch-media` for YouTube)")
        return
    backend = backend or Pyannote()
    for ep in eps:
        print(f"  ◎ {ep['title']}")
        diarize_episode(conn, ep, backend, num_speakers=num_speakers,
                        min_speakers=min_speakers, max_speakers=max_speakers)
        conn.commit()
        show(conn, ep["id"], samples=0)
    if not voiceprints(conn):
        print("\nno voiceprints yet: `jbp speakers <episode id>` to see who's who, then `jbp label <ep> SPEAKER_XX joe`")


def show(conn, episode_id: int, samples: int = 3) -> None:
    ep = conn.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()
    if not ep:
        raise SystemExit(f"no episode {episode_id}")
    rows = conn.execute("SELECT * FROM ep_speakers WHERE episode_id=? ORDER BY seconds DESC", (episode_id,)).fetchall()
    if not rows:
        print(f"[{episode_id}] {ep['title']}: not diarized")
        return
    if samples:
        print(f"[{episode_id}] {ep['title']}")
    for r in rows:
        tag = r["name"] or "?"
        if r["manual"]:
            tag += " (labelled)"
        elif r["score"] is not None:
            tag += f" ({r['score']:.2f})"
        print(f"    {r['cluster']:11s} {r['seconds'] / 60:5.1f} min  -> {tag}")
        if samples:
            for s in conn.execute(
                """SELECT start, text FROM segments WHERE episode_id=? AND cluster=?
                   ORDER BY length(text) DESC LIMIT ?""", (episode_id, r["cluster"], samples)):
                link = yt_link(ep["url"], s["start"]) if ep["url"] else ""
                print(f"        {fmt_ts(s['start'])}  {s['text'][:90]}  {link}")


def overview(conn) -> None:
    for name, n in conn.execute("SELECT name, n FROM voiceprints ORDER BY name"):
        eps, mins = conn.execute(
            "SELECT COUNT(DISTINCT episode_id), COALESCE(SUM(seconds), 0) / 60 FROM ep_speakers WHERE name=?", (name,)
        ).fetchone()
        print(f"voiceprint {name:12s} {n:3d} samples   found in {eps} episodes ({mins:.0f} min)")
    for ep in conn.execute("SELECT id, title FROM episodes WHERE diarized=1 ORDER BY id"):
        names = [r[0] for r in conn.execute(
            "SELECT COALESCE(name, cluster) FROM ep_speakers WHERE episode_id=? ORDER BY seconds DESC", (ep["id"],))]
        print(f"  [{ep['id']}] {ep['title'][:60]:60s} {', '.join(names)}")


def relabel_all(conn) -> None:
    for (ep_id,) in conn.execute("SELECT id FROM episodes WHERE diarized=1").fetchall():
        apply_names(conn, ep_id)
    conn.commit()


def label(episode_id: int, cluster: str, name: str, *, enroll: bool = True) -> None:
    conn = db.connect()
    row = conn.execute("SELECT * FROM ep_speakers WHERE episode_id=? AND cluster=?", (episode_id, cluster)).fetchone()
    if not row:
        raise SystemExit(f"episode {episode_id} has no cluster {cluster!r} (see `jbp speakers {episode_id}`)")
    conn.execute("UPDATE ep_speakers SET name=?, score=NULL, manual=1 WHERE episode_id=? AND cluster=?",
                 (name, episode_id, cluster))
    if enroll and row["embedding"]:
        n = enroll_vec(conn, name, np.frombuffer(row["embedding"], dtype=np.float32))
        print(f"voiceprint {name!r}: {n} sample(s)")
    relabel_all(conn)
    overview(conn)


def enroll_audio(name: str, spans: list[tuple[float, float]], *, audio: str | None = None,
                 episode: int | None = None, backend=None) -> None:
    """Enroll from hand-picked ranges where only `name` is talking."""
    conn = db.connect()
    if episode is not None:
        ep = conn.execute("SELECT media_path FROM episodes WHERE id=?", (episode,)).fetchone()
        audio = ep and ep["media_path"]
    if not audio or not Path(audio).exists():
        raise SystemExit("need --audio FILE or --episode ID with a local media_path")
    backend = backend or Pyannote()
    with tempfile.TemporaryDirectory() as tmp:
        for i, (s, e) in enumerate(spans or [(None, None)]):
            wav = to_wav(audio, str(Path(tmp) / f"{i}.wav"), s, e)
            n = enroll_vec(conn, name, backend.embed(wav, _whole(wav))[0])
    conn.commit()
    print(f"voiceprint {name!r}: {n} sample(s)")
    relabel_all(conn)


def _whole(wav: str) -> list[tuple[float, float]]:
    import wave
    with wave.open(wav) as w:
        return [(0.0, w.getnframes() / w.getframerate())]

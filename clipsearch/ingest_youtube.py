"""Pull YouTube auto-captions (no media) for a channel / playlist / video and store as segments."""
import html
import re
from pathlib import Path

from . import db
from .timeutil import parse_ts

CUE_RE = re.compile(r"^([\d:.]+)\s+-->\s+([\d:.]+)")
TAG_RE = re.compile(r"<[^>]+>")


def parse_vtt(text: str) -> list[tuple[float, float, str]]:
    """Parse a YouTube auto-caption VTT into clean, de-duplicated timed lines.

    Auto-captions use 'rolling' cues: each cue repeats the previous line and adds a new one,
    plus near-zero-length cues. We keep each distinct line once, at the cue where it first appears.
    """
    out: list[tuple[float, float, str]] = []
    seen_recent: list[str] = []
    start = end = None
    for raw in text.splitlines():
        line = raw.strip()
        m = CUE_RE.match(line)
        if m:
            start, end = parse_ts(m.group(1)), parse_ts(m.group(2))
            continue
        if start is None or not line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        clean = html.unescape(TAG_RE.sub("", line)).strip()
        if not clean or clean in seen_recent:
            continue
        out.append((start, end, clean))
        seen_recent = (seen_recent + [clean])[-3:]
    # close small gaps between consecutive lines (but don't bridge real silences/ad breaks)
    def _end(i, s, e):
        if i + 1 < len(out) and 0 <= out[i + 1][0] - e <= 2.0:
            return max(e, out[i + 1][0])
        return max(e, s + 0.5)
    return [(s, _end(i, s, e), t) for i, (s, e, t) in enumerate(out)]


def ingest(url: str, *, limit: int | None = None, lang: str = "en", subs_dir: str = "data/subs",
           title_filter: str | None = None, show: str | None = None) -> None:
    import yt_dlp

    Path(subs_dir).mkdir(parents=True, exist_ok=True)
    conn = db.connect()

    # 1) list entries cheaply
    with yt_dlp.YoutubeDL({"extract_flat": "in_playlist", "quiet": True, "playlistend": limit}) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = info.get("entries") or [info]
    entries = [e for e in entries if e]
    if title_filter:
        rx = re.compile(title_filter, re.I)
        entries = [e for e in entries if rx.search(e.get("title") or "")]
    print(f"{len(entries)} videos to check")

    opts = {
        "skip_download": True,
        "writeautomaticsub": True,
        "writesubtitles": True,          # prefer uploaded subs if they exist
        "subtitleslangs": [lang, f"{lang}-orig", f"{lang}.*"],
        "subtitlesformat": "vtt",
        "outtmpl": f"{subs_dir}/%(id)s.%(ext)s",
        "quiet": True,
        "ignoreerrors": True,
        "sleep_interval_subtitles": 1,
    }
    added = 0
    with yt_dlp.YoutubeDL(opts) as ydl:
        for e in entries:
            vid = e.get("id")
            if db.claim(conn, "youtube", vid, show):
                continue
            vurl = e.get("url") or f"https://www.youtube.com/watch?v={vid}"
            if not vurl.startswith("http"):
                vurl = f"https://www.youtube.com/watch?v={vid}"
            meta = ydl.extract_info(vurl, download=True)
            if not meta:
                print(f"  ! failed {vid}")
                continue
            vtts = sorted(Path(subs_dir).glob(f"{vid}*.vtt"))
            if not vtts:
                print(f"  - no captions: {meta.get('title')}")
                continue
            segs = parse_vtt(vtts[0].read_text(encoding="utf-8", errors="replace"))
            ep_id, _ = db.upsert_episode(
                conn, source="youtube", source_id=vid, title=meta.get("title"),
                url=f"https://www.youtube.com/watch?v={vid}", published=meta.get("upload_date"), show=show,
            )
            n = db.add_segments(conn, ep_id, segs)
            conn.commit()
            added += 1
            print(f"  + {meta.get('title')}  ({n} lines)")
    print(f"done: {added} new episodes. run `clipsearch index` next.")

"""Ingest Obsidian Web Clipper YouTube notes.

Clipper output varies by template, so this is lenient: it looks for any line that starts with a
timestamp (optionally inside a markdown link like [12:34](https://youtu.be/...?t=754)) and treats the
rest of the line as text. The video URL comes from frontmatter `source:`/`url:` if present.
"""
import re
from pathlib import Path

from . import db
from .timeutil import parse_ts

LINE_RE = re.compile(r"^\s*[-*]?\s*\[?\(?((?:\d+:)?\d{1,2}:\d{2})\)?\]?(?:\([^)]*\))?\s*[-–:]?\s*(.+)$")
URL_RE = re.compile(r"^(?:source|url)\s*:\s*\"?(https?://\S+?)\"?\s*$", re.M)
VID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")


def parse_note(text: str) -> list[tuple[float, float, str]]:
    raw = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if m:
            try:
                raw.append((parse_ts(m.group(1)), m.group(2).strip()))
            except ValueError:
                pass
    return [(t, raw[i + 1][0] if i + 1 < len(raw) else t + 5, s) for i, (t, s) in enumerate(raw)]


def ingest(folder: str, show: str | None = None) -> None:
    conn = db.connect()
    added = 0
    for path in sorted(Path(folder).expanduser().rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        segs = parse_note(text)
        if len(segs) < 10:          # not a transcript note
            continue
        url_m = URL_RE.search(text)
        url = url_m.group(1) if url_m else None
        vid_m = VID_RE.search(url or "")
        # if we already have this video from yt-dlp, skip the duplicate
        if vid_m and db.claim(conn, "youtube", vid_m.group(1), show):
            continue
        ep_id, new = db.upsert_episode(conn, source="obsidian", source_id=str(path), title=path.stem, url=url, show=show)
        if not new:
            continue
        db.add_segments(conn, ep_id, segs)
        conn.commit()
        added += 1
        print(f"  + {path.stem} ({len(segs)} lines)")
    print(f"done: {added} notes")

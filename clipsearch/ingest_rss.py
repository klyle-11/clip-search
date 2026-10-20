"""Podcast RSS (public, or a private member feed like Patreon's) -> local audio -> faster-whisper transcript.

Only fetches the feed URL/file you give it (or CLIPSEARCH_RSS). Audio is kept locally in media/rss/
so it can also be the cut source for compilations.
"""
import os
import re
from pathlib import Path

from . import db
from .config import env


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:80]


def ingest(feed_url: str | None = None, *, limit: int | None = None, media_dir: str = "media/rss",
           title_filter: str | None = None, model_size: str | None = None, show: str | None = None) -> None:
    import feedparser
    import requests
    from faster_whisper import WhisperModel

    feed_url = feed_url or env("RSS")
    if not feed_url:
        raise SystemExit("pass a feed URL/file (or set CLIPSEARCH_RSS in .env)")
    Path(media_dir).mkdir(parents=True, exist_ok=True)
    conn = db.connect()

    feed = feedparser.parse(feed_url)
    show = show or feed.feed.get("title")          # default: the podcast's own name
    items = feed.entries
    if title_filter:
        rx = re.compile(title_filter, re.I)
        items = [i for i in items if rx.search(i.get("title", ""))]
    items = items[:limit] if limit else items

    model = None
    for item in items:
        guid = item.get("id") or item.get("link") or item.get("title")
        if db.claim(conn, "rss", guid, show):
            continue
        enc = next((l for l in item.get("links", []) if l.get("rel") == "enclosure"), None)
        if not enc:
            print(f"  - no audio: {item.get('title')}")
            continue
        ext = Path(enc["href"].split("?")[0]).suffix or ".mp3"
        dest = Path(media_dir) / f"{_slug(item.get('title', guid))}{ext}"
        if not dest.exists():
            print(f"  ↓ {item.get('title')}")
            with requests.get(enc["href"], stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)

        if model is None:
            size = model_size or env("WHISPER_MODEL", "medium")
            model = WhisperModel(size, device="auto", compute_type="auto")
        print(f"  ✎ transcribing {dest.name}")
        segments, _ = model.transcribe(str(dest), language="en", vad_filter=True)
        segs = [(s.start, s.end, s.text.strip()) for s in segments]

        ep_id, _ = db.upsert_episode(
            conn, source="rss", source_id=guid, title=item.get("title"),
            url=item.get("link"), published=item.get("published"), media_path=str(dest), show=show,
        )
        db.add_segments(conn, ep_id, segs)
        conn.commit()
        print(f"  + {item.get('title')} ({len(segs)} segments)")

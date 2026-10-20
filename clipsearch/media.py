"""Download local media for YouTube episodes (audio by default) and record it as the episode's media_path.

Pulling from the same video the captions came from means transcript time == media time (offset 0),
and gives diarization something to listen to.
"""
import re
from pathlib import Path

from . import db

VID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")


def select_episodes(conn, *, episode: int | None = None, match: str | None = None,
                    limit: int | None = None, where: str = "1") -> list:
    rows = conn.execute(f"SELECT * FROM episodes WHERE {where} ORDER BY id").fetchall()
    if episode is not None:
        rows = [r for r in rows if r["id"] == episode]
    if match:
        rx = re.compile(match, re.I)
        rows = [r for r in rows if rx.search(r["title"] or "")]
    return rows[:limit] if limit else rows


def fetch(*, episode: int | None = None, match: str | None = None, limit: int | None = None,
          video: bool = False, media_dir: str = "media/youtube", force: bool = False) -> None:
    import yt_dlp

    conn = db.connect()
    where = "url IS NOT NULL" if force else "url IS NOT NULL AND (media_path IS NULL OR media_path = '')"
    eps = [e for e in select_episodes(conn, episode=episode, match=match, where=where)
           if VID_RE.search(e["url"])]
    eps = eps[:limit] if limit else eps
    print(f"{len(eps)} episodes to fetch")
    Path(media_dir).mkdir(parents=True, exist_ok=True)

    opts = {
        "format": "bv*[height<=1080]+ba/b" if video else "bestaudio/best",
        "outtmpl": f"{media_dir}/%(id)s.%(ext)s",
        "quiet": True,
        "noprogress": True,
        "ignoreerrors": True,
    }
    if video:
        opts["merge_output_format"] = "mp4"
    else:
        opts["postprocessors"] = [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}]

    with yt_dlp.YoutubeDL(opts) as ydl:
        for e in eps:
            vid = VID_RE.search(e["url"]).group(1)
            print(f"  ↓ {e['title']}")
            if not ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True):
                print(f"  ! failed {vid}")
                continue
            ext = ".mp4" if video else ".m4a"
            path = next(iter(sorted(Path(media_dir).glob(f"{vid}{ext}"))), None) \
                or next(iter(sorted(Path(media_dir).glob(f"{vid}.*"))), None)
            if not path:
                print(f"  ! downloaded but no file found for {vid}")
                continue
            conn.execute("UPDATE episodes SET media_path=? WHERE id=?", (str(path), e["id"]))
            conn.commit()
            print(f"  + {path}")

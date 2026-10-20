"""Turn a reviewed cutlist.csv into clips + one stitched compilation with ffmpeg.

Cuts only from local media files you supply (media_path column). `offset` (seconds, can be negative)
shifts transcript time to media time when the file's intro/ads differ from the transcript source.
"""
import csv
import shutil
import subprocess
from pathlib import Path

from .timeutil import fmt_ts, parse_ts


def run(cutlist: str, out: str, *, reencode: bool = True, dry_run: bool = False) -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH")
    rows = [r for r in csv.DictReader(open(cutlist, encoding="utf-8")) if r.get("keep", "y").lower().startswith("y")]
    missing = [r["title"] for r in rows if not r.get("media_path") or not Path(r["media_path"]).exists()]
    if missing:
        print("rows with no usable media_path (skipped):")
        for t in missing:
            print(f"  - {t}")
    rows = [r for r in rows if r.get("media_path") and Path(r["media_path"]).exists()]
    if not rows:
        raise SystemExit("nothing to cut")

    outp = Path(out)
    clip_dir = outp.parent / f"{outp.stem}_clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, r in enumerate(rows, 1):
        pad, off = float(r.get("pad") or 0), float(r.get("offset") or 0)
        s = max(0.0, parse_ts(r["start"]) + off - pad)
        e = parse_ts(r["end"]) + off + pad
        clip = clip_dir / f"{i:03d}{outp.suffix}"
        codec = (["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "192k"]
                 if reencode else ["-c", "copy"])
        if outp.suffix.lower() in {".mp3", ".m4a", ".wav"}:
            codec = ["-vn", "-c:a", "aac" if outp.suffix == ".m4a" else "libmp3lame", "-b:a", "192k"] if reencode else ["-c", "copy"]
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{s:.2f}", "-i", r["media_path"],
               "-t", f"{e - s:.2f}", *codec, str(clip)]
        print(f"[{i}/{len(rows)}] {r['title']} {fmt_ts(s)}–{fmt_ts(e)}")
        if dry_run:
            print("   ", " ".join(cmd))
        else:
            subprocess.run(cmd, check=True)
        clips.append(clip)

    listfile = clip_dir / "concat.txt"
    listfile.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(outp)]
    if dry_run:
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True)
    print(f"compilation -> {outp}  ({len(clips)} clips in {clip_dir})")

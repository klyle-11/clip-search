# Clip Search

Find the exact moments in podcasts and streams where something was said, and turn them into
compilation cut lists. Every line of every episode is searchable. You can step through each hit
with the player cued to that second, collect clips, and export or render them.

Runs locally. Point it at YouTube channels/playlists/videos, podcast RSS feeds (public or private
member feeds), or Obsidian Web Clipper transcript notes. Organize episodes into **shows** and
search one show or all of them.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .               # core: yt-dlp, numpy
pip install -e ".[whisper]"    # + faster-whisper, feedparser: transcribe RSS audio
pip install -e ".[semantic]"   # + sentence-transformers: search by meaning, themes
```
Needs `ffmpeg` on PATH for rendering. Python 3.11+.

## The UI
```bash
clipsearch ui                  # http://localhost:8765  (--host 0.0.0.0 for your LAN; there's no auth)
```
- **Sources:** paste any mix of YouTube video/playlist/channel URLs, RSS feed URLs and folder paths,
  or drop `.xml` feeds or `.txt` lists. Pick the show to file them under. Everything is queued and runs
  in the background with live logs, retry, and an index step at the end.
- **Search:** every occurrence of a phrase, or ranked keyword, semantic and hybrid search, or a saved theme.
  Results show the matched lines with timestamps and context. Click a result (or use `j`/`k`, `space`)
  to play it from that second: embedded YouTube, or local audio for RSS episodes. `a` adds it to a clip list.
- **Clips:** named cut lists. Adjust start/end/offset, reorder, download the CSV, or render with ffmpeg.
- **Library:** every episode, filed by show. Bulk-assign shows, delete bad ingests.

## CLI
```bash
clipsearch add urls.txt https://youtu.be/... feed.xml --show "My Podcast"   # batch ingest + index
clipsearch shows                                       # list shows
clipsearch shows assign "My Podcast" --unassigned      # file loose episodes (also --match/--source/--episode)

clipsearch grep '"4 in the morning" OR "5am"' -C 2     # every occurrence, with context + links
clipsearch grep 'noon alarm' --near 10 --show "My Podcast"
clipsearch grep '"4 in the morning"' --out cutlists/4am.csv --gap 15   # nearby hits merged into one clip
clipsearch search "waking up early while everyone sleeps" --mode sem   # by meaning (needs [semantic])
clipsearch vocab imani                                 # how the captions actually spelled it
clipsearch theme early-riser                           # saved theme from themes.toml -> cutlist
clipsearch cut cutlists/4am.csv --out out/4am.mp4      # cut + stitch from local media
```
Single-source commands are still there too: `ingest-youtube`, `ingest-rss --feed`, `ingest-obsidian`, `index`.

## Query syntax
- `"exact phrase"` matches consecutive words, even across caption line breaks.
- `word word` means all of the words within `--near` words of each other (default 15).
- `OR`, `NOT`, and `prefix*` work as you'd expect.
- Numbers and times are normalized on both sides: `four` = `4`, `4 a.m.` = `4am` = `4:00 AM`,
  `don't` = `dont`. Light stemming: `wake` ~ `waking`.

## Media and offsets
Rendering cuts from each episode's local media file. RSS episodes download their audio when they're
ingested. For YouTube episodes, `clipsearch fetch-media` downloads audio (or `--video`) from the same
video the captions came from, so timestamps line up. When the media comes from somewhere else (a
different upload, ads added), set the clip's `offset`.

## Optional: speaker labels
`pip install -e ".[diarize]"` + `HF_TOKEN` in `.env`. Then `clipsearch diarize`, `clipsearch speakers <ep>`,
and `clipsearch label <ep> SPEAKER_01 <name>` to enroll a voice. After that, `--speaker <name>` keeps
only hits where that person is talking. Untested on real audio so far.

## Settings
See `.env.example`. Every `CLIPSEARCH_*` setting also accepts its legacy `JBP_*` name, and an
existing `data/jbp.db` is picked up automatically.

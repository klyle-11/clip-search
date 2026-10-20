# jbp-search

Search the whole JBP catalog by keyword or theme and turn hits into compilation cut lists.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core: yt-dlp, numpy
pip install -e ".[semantic]"   # + sentence-transformers for theme search
pip install -e ".[whisper]"    # + faster-whisper, feedparser for Patreon audio
```
Needs `ffmpeg` on PATH for cutting.

## Typical flow
```bash
# 1. Pull captions for the channel (or a playlist / single video URL)
jbp ingest-youtube "https://www.youtube.com/@JoeBudden/videos" --limit 50

# 2. Patreon audio via your private patron RSS feed (put JBP_PATREON_RSS in .env)
jbp ingest-rss --limit 10

# 3. Obsidian Clipper notes you already saved
jbp ingest-obsidian ~/Vault/Clippings

# 4. Chunk + embed everything new
jbp index

# 5. Search
jbp search "wake up at 4 in the morning" --mode kw
jbp search "joe bragging about waking up early while everyone else sleeps" --mode sem
jbp theme early-riser --out cutlists/early-riser.csv

# 6. Edit the CSV (drop rows, set media_path/offset), then cut
jbp cut cutlists/early-riser.csv --out out/early-riser.mp4
```

## Themes
Add reusable compilations to `themes.toml` — a theme is a description (semantic) plus keyword queries.

# jbp-search

Search the whole JBP catalog by keyword or theme and turn hits into compilation cut lists.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core: yt-dlp, numpy
pip install -e ".[semantic]"   # + sentence-transformers for theme search
pip install -e ".[whisper]"    # + faster-whisper, feedparser for Patreon audio
pip install -e ".[diarize]"    # + pyannote.audio for speaker labels (HF_TOKEN in .env)
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
jbp grep '"4 in the morning"'                 # every occurrence, with context lines + links
jbp search "wake up at 4 in the morning" --mode kw
jbp search "joe bragging about waking up early while everyone else sleeps" --mode sem
jbp theme early-riser --out cutlists/early-riser.csv

# 6. Edit the CSV (drop rows, set media_path/offset), then cut
jbp cut cutlists/early-riser.csv --out out/early-riser.mp4
```

## Searching dialogue
`grep` lists **every** occurrence in episode/time order; `search --mode kw` ranks chunks (BM25) and pins
each hit to the line where the words occur. Both print the matched line(s) with timestamps, context,
and a YouTube link at that second.
```bash
jbp grep '"4 in the morning" OR "5am" OR "wake up early"' -C 2
jbp grep 'noon alarm' --near 10               # unquoted words: all within N words of each other
jbp grep 'morning NOT "good morning"' --match 'Episode 9\d\d'   # title regex filter
jbp grep '"4 in the morning"' --out cutlists/4am.csv --gap 15    # nearby hits merged into one clip
jbp vocab imani                               # how captions actually spelled it (amani? a money?)
```
- Phrases match across caption-line breaks ("…since four in the" / "morning that's why…").
- Numbers and times are normalized on both sides: `four` = `4`, `4 a.m.` = `4am` = `4:00 AM`,
  `don't` = `dont`. Light stemming: `wake` ~ `waking` ~ `wakes`. `wak*` = prefix.
- Upgrading from an older DB rebuilds the FTS index on first run (no re-ingest, no re-embed).

## Speakers (filter to Joe)
Diarization needs local audio. For YouTube episodes, `fetch-media` pulls it from the same video the
captions came from, so offset stays 0 and the file doubles as the cut source.
```bash
jbp fetch-media --episode 1            # or --match 'Episode 97\d' --limit 10 ; --video for mp4
jbp diarize --episode 1
jbp speakers 1                         # clusters + sample lines + links: find which one is Joe
jbp label 1 SPEAKER_01 joe             # enrolls Joe's voiceprint, re-matches every diarized episode
jbp diarize --limit 20                 # later episodes auto-label joe
jbp search "4 in the morning" --mode kw --speaker joe
```
Label a few more episodes to firm up the voiceprint (samples are averaged). Auto-match cutoff is
`JBP_VOICE_THRESHOLD` (cosine, default 0.55): check `jbp speakers` scores and tune. `--speaker` keeps a
hit if the matched line is theirs, or if they hold ≥40% of the hit window's talk time. Undiarized
episodes always pass through.

## Themes
Add reusable compilations to `themes.toml` — a theme is a description (semantic) plus keyword queries.

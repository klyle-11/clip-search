# jbp-search — handoff notes

## What this is
A personal pipeline for finding themed moments across The Joe Budden Podcast (main feed + Patreon)
and turning them into compilation cut lists. First target compilation: Joe talking about waking up
at 3–5am, clowning co-hosts who sleep late, and tying early rising to his success.

## Owner context
- Owner is a former full-stack engineer (Python/TS/Java), runs a homelab. Prefers local-first,
  self-hosted tooling. Don't over-explain basics.
- Obsidian Web Clipper is already used to save some YouTube transcripts as markdown. Those can be
  ingested too (`jbp ingest-obsidian`).

## Architecture (current state)
1. **Ingest**
   - `ingest_youtube.py` — yt-dlp pulls auto-captions (VTT) for a channel/playlist/video, no media download.
   - `ingest_rss.py` — Patreon private RSS (patron's own feed) → audio → faster-whisper transcript.
   - `ingest_obsidian.py` — parses Obsidian Clipper transcript notes (timestamped lines).
2. **Store** — single SQLite file (`data/jbp.db`): `episodes`, `segments` (raw timed lines),
   `chunks` (~45s windows), `chunks_fts` (FTS5), embeddings stored as float32 BLOBs on `chunks`.
3. **Search** — keyword (FTS5/BM25), semantic (sentence-transformers, brute-force cosine in numpy),
   or hybrid (reciprocal rank fusion). Saved themes live in `themes.toml`.
4. **Cut** — search hits → `cutlist.csv` (review/edit by hand) → ffmpeg commands that cut from the
   owner's own local media files and concat them.

## Status (v0.2)
- Tested offline with synthetic data: VTT rolling-caption parsing, Obsidian note parsing, chunking,
  FTS5 keyword search (with hit "focus" to the matching line), semantic + hybrid (RRF) with a stub
  embedder, themes, cutlist CSV, ffmpeg cut + concat.
- NOT yet tested against live sources: yt-dlp on the real channel, real sentence-transformers model,
  faster-whisper on Patreon audio. First run on the owner's machine should be:
  `jbp ingest-youtube <one episode URL>` → `jbp index` → `jbp search "4 in the morning" --mode kw`.
- Obsidian parser is template-agnostic but unverified against the owner's actual Clipper template —
  ask for a sample note if ingest reports 0 notes.

## Known gaps / next steps (in priority order)
1. **Speaker diarization — built (v0.2), untested on real audio.** `fetch-media` (yt-dlp audio for
   YouTube episodes → `media_path`), `diarize` (pyannote 3.1 turns + wespeaker cluster embeddings →
   `segments.cluster`/`speaker`, `ep_speakers`), `speakers`/`label`/`enroll` (voiceprints table, running
   mean). Search `--speaker` = matched line is theirs OR ≥40% of window talk time. Tested offline with a
   stub backend only. First live run: fetch + diarize one episode, `jbp speakers <id>`, label Joe, then
   diarize a few more and tune `JBP_VOICE_THRESHOLD` from the printed scores. If pyannote's API differs
   from the 3.x/4.x handling in `diarize.Pyannote`, fix it there; the rest is backend-agnostic.
2. Timestamp offset between transcript source and the media file being cut (intros/ads differ between
   YouTube and Patreon audio). Cut list has a per-row `offset` column; consider auto-alignment.
3. Swap brute-force cosine for sqlite-vec once the corpus is large (~200k chunks for the full catalog).
4. Optional LLM rerank pass over top-N semantic hits using the theme description.

## Ground rules
- Personal use only. Downloaded media stays local, used for the owner's own edits.
- Don't scrape Patreon pages; only use the patron RSS feed URL the owner supplies (keep it in `.env`).

## Leads already found (from research, verify speaker before cutting)
- Ep 971 "Off The Clock Cuck" ~00:03:04–00:07:15 — Imani's noon alarm, room clowns him, 4am/6:30 talk.
- Ep 787 "Shartonay" ~00:01:35–00:03:14 — "you said you wake up at 4 in the morning", bedtime 9–10.
- Ep 897 "Company Tardy Policy" ~00:02:47–00:06:15 — tardiness policy (contrast beat, not on-theme).

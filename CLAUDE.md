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

## Status (v0.3)
- Tested offline with synthetic data: VTT rolling-caption parsing, Obsidian note parsing, chunking,
  FTS5 keyword search (with hit "focus" to the matching line), semantic + hybrid (RRF) with a stub
  embedder, themes, cutlist CSV, ffmpeg cut + concat.
- NOT yet tested against live sources: yt-dlp on the real channel, real sentence-transformers model,
  faster-whisper on Patreon audio. First run on the owner's machine should be:
  `jbp ingest-youtube <one episode URL>` → `jbp index --no-embed` → `jbp grep '"4 in the morning"'`.
- Obsidian parser is template-agnostic but unverified against the owner's actual Clipper template —
  ask for a sample note if ingest reports 0 notes.

## Known gaps / next steps (in priority order)
**Owner priority: searching dialogue >> speaker diarization.** Put effort into finding lines, not who said them.
0. **Dialogue search — built (v0.3), offline-tested.** `textnorm.py` (number words→digits, a.m./pm,
   apostrophes; FTS indexes `chunks.norm`), `dialogue.py` (query parse: phrases/OR/NOT/prefix/near;
   line-level scanner that matches across caption lines and interpolates the word's time),
   `jbp grep` (every occurrence, context, merged cutlist), `jbp vocab` (spelling variants via fts5vocab),
   `--match` title regex on search/grep/theme. Ideas next: fuzzy/phonetic matching for caption errors
   (e.g. trigram FTS side-index), episode-number/date filters parsed from titles, a small TUI or web
   page to step through grep hits with an embedded YouTube player at the timestamp.
1. Timestamp offset between transcript source and the media file being cut (intros/ads differ between
   YouTube and Patreon audio). Cut list has a per-row `offset` column; consider auto-alignment.
2. Swap brute-force cosine for sqlite-vec once the corpus is large (~200k chunks for the full catalog).
3. Optional LLM rerank pass over top-N semantic hits using the theme description.
4. **Speaker diarization — built (v0.2), untested on real audio, deprioritized by owner.** `fetch-media`,
   `diarize` (pyannote 3.1 + wespeaker cluster embeddings), `speakers`/`label`/`enroll` voiceprints.
   `--speaker` = matched line is theirs OR ≥40% of window talk time; undiarized episodes pass through.
   If pyannote's API differs, fix `diarize.Pyannote`; the rest is backend-agnostic.

## Ground rules
- Personal use only. Downloaded media stays local, used for the owner's own edits.
- Don't scrape Patreon pages; only use the patron RSS feed URL the owner supplies (keep it in `.env`).

## Leads already found (from research, verify speaker before cutting)
- Ep 971 "Off The Clock Cuck" ~00:03:04–00:07:15 — Imani's noon alarm, room clowns him, 4am/6:30 talk.
- Ep 787 "Shartonay" ~00:01:35–00:03:14 — "you said you wake up at 4 in the morning", bedtime 9–10.
- Ep 897 "Company Tardy Policy" ~00:02:47–00:06:15 — tardiness policy (contrast beat, not on-theme).

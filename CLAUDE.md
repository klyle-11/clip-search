# Clip Search — handoff notes

## What this is
A tool for clippers: search everything said across podcasts/streams, step through every hit with a
player at that second, collect clips, export/render compilations. Started as a personal Joe Budden
Podcast (JBP) pipeline; **now a general product**. JBP is just one show (library) among many. The owner
plans to give it to other clippers and possibly monetize, so nothing in commands, UI or defaults
should be show-specific.

## Owner context
- Owner is a former full-stack engineer (Python/TS/Java), runs a homelab. Prefers local-first,
  self-hosted tooling. Don't over-explain basics.
- **Priority: searching dialogue >> speaker diarization.** Put effort into finding lines and making the
  review/clip loop fast, not into who said them.
- Uses Obsidian Web Clipper for some YouTube transcripts (`ingest-obsidian`, or a folder path in the UI).

## Architecture (package `clipsearch`, CLI `clipsearch`)
1. **Ingest:** `ingest_youtube.py` (yt-dlp captions, no media), `ingest_rss.py` (feed URL or .xml file →
   audio → faster-whisper; show defaults to the feed title), `ingest_obsidian.py`. All take `show=`.
   `jobs.py` classifies mixed input (URLs, .xml, folders, .txt lists) into a SQLite-backed job queue.
   `clipsearch add` runs it inline; the UI runs one background worker.
2. **Store:** one SQLite file (`data/clipsearch.db`, falls back to an existing `data/jbp.db`): `shows`,
   `episodes` (show_id), `segments` (timed lines), `chunks` (~45s windows, `norm` text, embedding BLOB),
   `chunks_fts` (FTS5 over `norm`), `jobs`, `clips` (UI cut lists), and diarization tables.
   Migrations run in `db.connect()`. Any data change there must `commit()`, or the open transaction
   locks out other connections.
3. **Search:** `textnorm.py` (numbers/times/apostrophes normalized for index + query), `dialogue.py`
   (query parser and a line-level scanner: phrases across caption breaks, near, OR/NOT/prefix,
   interpolated word time), `search.py` (grep = every occurrence; kw = BM25 chunks anchored to the
   matched line; sem/hybrid; themes; filters `show`, title regex, speaker).
4. **UI:** `web.py` (stdlib http.server + JSON API + Range-capable `/media/<episode>`), `ui.html`
   (single file, vanilla JS: Search / Sources / Clips / Library, YouTube IFrame player or local audio,
   keyboard j/k/space/a). No auth. Binds 127.0.0.1 by default.
5. **Cut:** cutlist CSV → ffmpeg clips + concat (`cut.py`); the UI's render button queues a `cut` job.
6. **Config:** `config.env("X")` reads `CLIPSEARCH_X`, falling back to legacy `JBP_X`.

## Status (v0.4)
- Offline-tested with synthetic data: VTT/Obsidian parsing, normalization, phrase/near/OR/NOT matching
  across line breaks, grep/kw/hybrid (stub embedder), shows (assign/rename/filter), job queue (success
  + failure paths), UI flows in headless Chromium (desktop + 390px dark), clip lists + CSV export,
  DB migrations from v0.1/v0.2/v0.3 schemas.
- NOT yet run against live sources: yt-dlp on real channels, faster-whisper, sentence-transformers,
  pyannote. First real run: `clipsearch ui` → paste one episode URL with a show name → search.

## Next steps (priority order)
1. **Live shakedown** on real captions; fix whatever yt-dlp/whisper throw. Check grep quality on
   messy auto-captions.
2. **Caption-error tolerance:** fuzzy/phonetic matching (e.g. trigram FTS side index, or a
   sound-alike pass over `vocab`) for names and slang auto-captions mangle.
3. **Distribution:** proper packaging (pipx / Docker image with ffmpeg), first-run onboarding in the UI,
   settings page instead of `.env`. If shared beyond the LAN: auth, per-user data, and rethink
   the single-worker queue.
4. Timestamp offset auto-alignment between transcript source and media (intros/ads differ).
5. sqlite-vec instead of brute-force cosine once the corpus is large (~200k chunks for a full catalog).
6. Optional LLM rerank of top-N semantic hits using a theme description.
7. Speaker diarization: built (v0.2) but untested on real audio, and deprioritized. `diarize.Pyannote`
   is the only backend-specific code.

## Ground rules
- Users bring their own sources. Only fetch the feed URLs/files a user supplies. Never scrape Patreon
  or other membership pages.
- Downloaded media stays local for the user's own edits.
- Productizing: yt-dlp downloading and redistributing others' content raise platform-ToS and
  copyright questions for a paid product. Keep the tool "search + cut lists over sources the user has
  access to", and flag anything that would host or redistribute media.

## JBP library: leads found earlier (verify speaker before cutting)
- Ep 971 "Off The Clock Cuck" ~00:03:04–00:07:15: Imani's noon alarm, room clowns him, 4am/6:30 talk.
- Ep 787 "Shartonay" ~00:01:35–00:03:14: "you said you wake up at 4 in the morning", bedtime 9–10.
- Ep 897 "Company Tardy Policy" ~00:02:47–00:06:15: tardiness policy (contrast beat, not on-theme).

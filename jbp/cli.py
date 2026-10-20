import argparse
import tomllib
from pathlib import Path


def db_connect():
    from . import db
    return db.connect()


def main(argv=None):
    p = argparse.ArgumentParser(prog="jbp", description="Search JBP transcripts and build compilation cut lists.")
    sub = p.add_subparsers(dest="cmd", required=True)

    y = sub.add_parser("ingest-youtube", help="pull auto-captions for a channel/playlist/video URL")
    y.add_argument("url")
    y.add_argument("--limit", type=int)
    y.add_argument("--match", help="regex on video title, e.g. 'Episode \\d+'")

    r = sub.add_parser("ingest-rss", help="Patreon/podcast RSS -> audio -> whisper")
    r.add_argument("--feed")
    r.add_argument("--limit", type=int)
    r.add_argument("--match")
    r.add_argument("--model", help="whisper size: small|medium|large-v3")

    o = sub.add_parser("ingest-obsidian", help="ingest Obsidian Clipper transcript notes")
    o.add_argument("folder")

    f = sub.add_parser("fetch-media", help="download audio (or --video) for YouTube episodes -> media_path")
    f.add_argument("--episode", type=int)
    f.add_argument("--match", help="regex on episode title")
    f.add_argument("--limit", type=int)
    f.add_argument("--video", action="store_true", help="mp4 instead of audio-only")
    f.add_argument("--force", action="store_true", help="re-download even if media_path is set")

    d = sub.add_parser("diarize", help="label who's talking (needs local media + pyannote)")
    d.add_argument("--episode", type=int)
    d.add_argument("--match")
    d.add_argument("--limit", type=int)
    d.add_argument("--redo", action="store_true")
    d.add_argument("--num-speakers", type=int)
    d.add_argument("--min-speakers", type=int)
    d.add_argument("--max-speakers", type=int)

    sp = sub.add_parser("speakers", help="voiceprints + diarized episodes, or one episode's clusters with sample lines")
    sp.add_argument("episode", type=int, nargs="?")

    lb = sub.add_parser("label", help="name a cluster (and enroll it as a voiceprint), e.g. `jbp label 3 SPEAKER_01 joe`")
    lb.add_argument("episode", type=int)
    lb.add_argument("cluster")
    lb.add_argument("name")
    lb.add_argument("--no-enroll", action="store_true", help="name it in this episode only")

    en = sub.add_parser("enroll", help="enroll a voiceprint from hand-picked solo ranges")
    en.add_argument("name")
    en.add_argument("--audio")
    en.add_argument("--episode", type=int)
    en.add_argument("--at", action="append", default=[], help="START-END, e.g. 00:10:00-00:11:30 (repeatable)")

    i = sub.add_parser("index", help="chunk + embed new episodes")
    i.add_argument("--no-embed", action="store_true")
    i.add_argument("--rechunk", action="store_true")

    s = sub.add_parser("search", help="search transcripts")
    s.add_argument("query")
    s.add_argument("--mode", choices=["kw", "sem", "hybrid"], default="hybrid")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--speaker", help="only hits where this speaker dominates (undiarized episodes pass through)")
    s.add_argument("--min-score", type=float, default=0.0)
    s.add_argument("--match", help="regex on episode title, e.g. 'Episode 9[0-9]{2}'")
    s.add_argument("--near", type=int, default=15, help="max words apart for unquoted words")
    s.add_argument("-C", "--context", type=int, default=1, help="lines of context around the match")
    s.add_argument("--out", help="write a cutlist CSV")

    g = sub.add_parser("grep", help="every occurrence of a phrase across the corpus, in episode/time order")
    g.add_argument("query", help='e.g. \'"4 in the morning" OR "5am"\' — four==4, "4 a.m."==4am')
    g.add_argument("--match", help="regex on episode title")
    g.add_argument("--speaker")
    g.add_argument("--near", type=int, default=15)
    g.add_argument("-C", "--context", type=int, default=1)
    g.add_argument("--limit", type=int)
    g.add_argument("--gap", type=float, default=10.0, help="merge occurrences this close into one clip (cutlist)")
    g.add_argument("--out", help="write a cutlist CSV (nearby occurrences merged)")

    v = sub.add_parser("vocab", help="indexed spellings close to a word (catch caption misspellings)")
    v.add_argument("word")
    v.add_argument("-n", type=int, default=20)

    t = sub.add_parser("theme", help="run a saved theme from themes.toml")
    t.add_argument("name", nargs="?")
    t.add_argument("--file", default="themes.toml")
    t.add_argument("--speaker")
    t.add_argument("--match", help="regex on episode title")
    t.add_argument("--out")

    c = sub.add_parser("cut", help="cut + stitch a reviewed cutlist with ffmpeg")
    c.add_argument("cutlist")
    c.add_argument("--out", required=True, help="e.g. out/early-riser.mp4 (or .m4a/.mp3 for audio)")
    c.add_argument("--copy", action="store_true", help="stream copy instead of re-encode (faster, less exact)")
    c.add_argument("--dry-run", action="store_true")

    sub.add_parser("stats", help="what's in the database")

    a = p.parse_args(argv)

    if a.cmd == "ingest-youtube":
        from .ingest_youtube import ingest
        ingest(a.url, limit=a.limit, title_filter=a.match)
    elif a.cmd == "ingest-rss":
        from .ingest_rss import ingest
        ingest(a.feed, limit=a.limit, title_filter=a.match, model_size=a.model)
    elif a.cmd == "ingest-obsidian":
        from .ingest_obsidian import ingest
        ingest(a.folder)
    elif a.cmd == "fetch-media":
        from .media import fetch
        fetch(episode=a.episode, match=a.match, limit=a.limit, video=a.video, force=a.force)
    elif a.cmd == "diarize":
        from .diarize import run
        run(episode=a.episode, match_title=a.match, limit=a.limit, redo=a.redo, num_speakers=a.num_speakers,
            min_speakers=a.min_speakers, max_speakers=a.max_speakers)
    elif a.cmd == "speakers":
        from . import db, diarize
        conn = db.connect()
        diarize.show(conn, a.episode) if a.episode else diarize.overview(conn)
    elif a.cmd == "label":
        from .diarize import label
        label(a.episode, a.cluster, a.name, enroll=not a.no_enroll)
    elif a.cmd == "enroll":
        from .diarize import enroll_audio
        from .timeutil import parse_ts
        spans = [tuple(parse_ts(x) for x in r.split("-", 1)) for r in a.at]
        enroll_audio(a.name, spans, audio=a.audio, episode=a.episode)
    elif a.cmd == "index":
        from .index import build
        build(embeddings=not a.no_embed, rechunk=a.rechunk)
    elif a.cmd == "search":
        from . import search
        hits = search.run(a.query, a.mode, a.limit, a.speaker, a.min_score, a.match, a.near)
        search.print_hits(hits, ctx=a.context)
        if a.out:
            search.write_cutlist(hits, a.out)
    elif a.cmd == "grep":
        from . import search
        hits = search.grep(db_connect(), a.query, speaker=a.speaker, title=a.match, near=a.near, limit=a.limit)
        search.print_grep(hits, ctx=a.context)
        if a.out:
            search.write_cutlist(search.merge_close(hits, a.gap), a.out)
    elif a.cmd == "vocab":
        from . import search
        for term, docs, cnt in search.similar_terms(db_connect(), a.word, a.n):
            print(f"{cnt:7d}  {term}  ({docs} chunks)")
    elif a.cmd == "theme":
        from . import search
        themes = tomllib.loads(Path(a.file).read_text())
        if not a.name:
            for n, th in themes.items():
                print(f"{n}: {th['description'][:100]}")
            return
        hits = search.run_theme(themes[a.name], a.speaker, a.match)
        search.print_hits(hits)
        search.write_cutlist(hits, a.out or f"cutlists/{a.name}.csv")
    elif a.cmd == "cut":
        from .cut import run
        run(a.cutlist, a.out, reencode=not a.copy, dry_run=a.dry_run)
    elif a.cmd == "stats":
        from . import db
        conn = db.connect()
        for src, n, segs in conn.execute(
            "SELECT e.source, COUNT(DISTINCT e.id), COUNT(s.id) FROM episodes e LEFT JOIN segments s ON s.episode_id=e.id GROUP BY e.source"
        ):
            print(f"{src:12s} {n:5d} episodes  {segs:8d} lines")
        ch, emb = conn.execute("SELECT COUNT(*), COUNT(embedding) FROM chunks").fetchone()
        print(f"chunks: {ch}  embedded: {emb}")
        media, diar = conn.execute("SELECT COUNT(media_path), SUM(diarized) FROM episodes WHERE media_path != ''").fetchone()
        vps = [r[0] for r in conn.execute("SELECT name FROM voiceprints")]
        print(f"with media: {media}  diarized: {diar or 0}  voiceprints: {', '.join(vps) or '-'}")


if __name__ == "__main__":
    main()

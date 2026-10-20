import argparse
import tomllib
from pathlib import Path


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

    i = sub.add_parser("index", help="chunk + embed new episodes")
    i.add_argument("--no-embed", action="store_true")
    i.add_argument("--rechunk", action="store_true")

    s = sub.add_parser("search", help="search transcripts")
    s.add_argument("query")
    s.add_argument("--mode", choices=["kw", "sem", "hybrid"], default="hybrid")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--speaker", help="filter by diarized speaker label (no-op until diarization exists)")
    s.add_argument("--min-score", type=float, default=0.0)
    s.add_argument("--out", help="write a cutlist CSV")

    t = sub.add_parser("theme", help="run a saved theme from themes.toml")
    t.add_argument("name", nargs="?")
    t.add_argument("--file", default="themes.toml")
    t.add_argument("--speaker")
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
    elif a.cmd == "index":
        from .index import build
        build(embeddings=not a.no_embed, rechunk=a.rechunk)
    elif a.cmd == "search":
        from . import search
        hits = search.run(a.query, a.mode, a.limit, a.speaker, a.min_score)
        search.print_hits(hits)
        if a.out:
            search.write_cutlist(hits, a.out)
    elif a.cmd == "theme":
        from . import search
        themes = tomllib.loads(Path(a.file).read_text())
        if not a.name:
            for n, th in themes.items():
                print(f"{n}: {th['description'][:100]}")
            return
        hits = search.run_theme(themes[a.name], a.speaker)
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


if __name__ == "__main__":
    main()

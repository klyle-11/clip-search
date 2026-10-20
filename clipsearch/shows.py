"""Shows (libraries): group episodes so one install can hold many podcasts/channels."""
from . import db


def listing(conn) -> list[dict]:
    return [dict(r) for r in conn.execute(
        """SELECT s.id, s.name, COUNT(e.id) AS episodes,
                  (SELECT COUNT(*) FROM segments g JOIN episodes x ON x.id=g.episode_id WHERE x.show_id=s.id) AS lines
           FROM shows s LEFT JOIN episodes e ON e.show_id=s.id GROUP BY s.id ORDER BY s.name""")]


def assign(conn, name: str, *, episode: int | None = None, match: str | None = None,
           source: str | None = None, unassigned: bool = False) -> int:
    """File matching episodes under show `name`. Returns how many moved."""
    where, args = ["1"], []
    if episode is not None:
        where.append("id = ?"); args.append(episode)
    if match:
        where.append("title REGEXP ?"); args.append(match)
    if source:
        where.append("source = ?"); args.append(source)
    if unassigned:
        where.append("show_id IS NULL")
    if len(where) == 1:
        raise SystemExit("say which episodes: --episode, --match, --source and/or --unassigned")
    cur = conn.execute(f"UPDATE episodes SET show_id=? WHERE {' AND '.join(where)}", (db.show_id(conn, name), *args))
    conn.commit()
    return cur.rowcount


def rename(conn, old: str, new: str) -> None:
    if not conn.execute("SELECT 1 FROM shows WHERE name=?", (old,)).fetchone():
        raise SystemExit(f"no show called {old!r}")
    existing = conn.execute("SELECT id FROM shows WHERE name=?", (new,)).fetchone()
    if existing:      # merge into the existing show
        conn.execute("UPDATE episodes SET show_id=? WHERE show_id=(SELECT id FROM shows WHERE name=?)", (existing["id"], old))
        conn.execute("DELETE FROM shows WHERE name=?", (old,))
    else:
        conn.execute("UPDATE shows SET name=? WHERE name=?", (new, old))
    conn.commit()

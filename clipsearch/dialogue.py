"""Line-level dialogue matching: query parsing, FTS query building, and exact-ish occurrence finding.

Query syntax (same for `search --mode kw` and `grep`):
    "4 in the morning"            phrase — consecutive words, may span caption lines
    wake early                    all words within --near words of each other (default 15)
    wak*                          prefix
    "4 in the morning" OR "5am"   alternatives
    morning NOT "good morning"    exclude chunks containing the phrase
Numbers/times are normalized on both sides, so four == 4 and "4 a.m." == 4am.
"""
import re
from dataclasses import dataclass

from .textnorm import norm, stem, tokens

_LEX = re.compile(r'"([^"]*)"|(\S+)')


@dataclass(frozen=True)
class Term:
    words: tuple[str, ...]      # normalized tokens; >1 = phrase
    prefix: bool = False        # last word is a prefix (wak*)


@dataclass
class Query:
    alts: list[list[Term]]      # OR of ANDs
    nots: list[Term]

    def fts(self) -> str:
        def term(t: Term) -> str:
            return f'"{" ".join(t.words)}"' + ("*" if t.prefix else "")
        expr = " OR ".join(f"({' '.join(term(t) for t in alt)})" for alt in self.alts)
        if self.nots:
            expr = f"({expr}) NOT ({' OR '.join(term(t) for t in self.nots)})"
        return expr


def _terms(raw: str, quoted: bool) -> list[Term]:
    prefix = raw.endswith("*")
    words = tokens(norm(raw.rstrip("*")))
    if not words:
        return []
    if quoted:
        return [Term(tuple(words), prefix)]
    return [Term((w,), prefix and i == len(words) - 1) for i, w in enumerate(words)]


def parse(q: str) -> Query:
    alts: list[list[Term]] = [[]]
    nots: list[Term] = []
    negate = False
    for m in _LEX.finditer(q):
        phrase, word = m.groups()
        if word in ("OR", "AND", "NOT"):
            if word == "OR" and alts[-1]:
                alts.append([])
            negate = word == "NOT"
            continue
        ts = _terms(phrase if phrase is not None else word, quoted=phrase is not None)
        (nots if negate else alts[-1]).extend(ts)
        negate = False
    alts = [a for a in alts if a]
    if not alts:
        raise ValueError(f"nothing searchable in query {q!r}")
    return Query(alts, nots)


# ---------- scanning ----------

def _match_word(qw: str, w: str, prefix: bool) -> bool:
    return w.startswith(qw) if prefix else (w == qw or stem(w) == stem(qw))


def _term_positions(stream: list[str], t: Term) -> list[int]:
    n, out = len(t.words), []
    for i in range(len(stream) - n + 1):
        if all(_match_word(qw, stream[i + k], t.prefix and k == n - 1) for k, qw in enumerate(t.words)):
            out.append(i)
    return out


@dataclass
class Occurrence:
    seg: int        # index of the segment the match starts in
    t: float        # estimated start time of the first matched word
    span: int       # how many segments the match touches (>= 1)


def find(segs, query: Query, near: int = 15) -> list[Occurrence]:
    """All occurrences of `query` in time-ordered segments (rows with start/end/text)."""
    stream: list[str] = []
    owner: list[tuple[int, float]] = []      # token -> (segment index, estimated time)
    for si, s in enumerate(segs):
        ws = tokens(norm(s["text"]))
        dur = max(s["end"] - s["start"], 0.0)
        for k, w in enumerate(ws):
            stream.append(w)
            owner.append((si, s["start"] + dur * k / len(ws)))

    hits: dict[int, int] = {}                # start token -> end token
    for alt in query.alts:
        pos = [_term_positions(stream, t) for t in alt]
        if any(not p for p in pos):
            continue
        if len(alt) == 1:
            for p in pos[0]:
                hits.setdefault(p, p + len(alt[0].words) - 1)
            continue
        # every term within `near` tokens: anchor on each occurrence of the rarest term
        rare = min(range(len(alt)), key=lambda i: len(pos[i]))
        for p in pos[rare]:
            picks = []
            for i, ps in enumerate(pos):
                cands = [x for x in ps if abs(x - p) <= near]
                if not cands:
                    break
                picks.append(min(cands, key=lambda x: abs(x - p)))
                picks.append(picks[-1] + len(alt[i].words) - 1)
            else:
                lo = min(picks)
                hits.setdefault(lo, max(picks))

    occs, last_end = [], -1
    for lo in sorted(hits):
        if lo <= last_end:                   # overlapping match of the same moment
            continue
        hi = hits[lo]
        last_end = hi
        occs.append(Occurrence(owner[lo][0], owner[lo][1], owner[hi][0] - owner[lo][0] + 1))
    return occs

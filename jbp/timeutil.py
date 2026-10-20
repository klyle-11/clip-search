import re

_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?")


def parse_ts(s: str) -> float:
    """'01:02:03.500' | '2:03' | '00:01:35,000' -> seconds."""
    m = _TS.fullmatch(s.strip())
    if not m:
        raise ValueError(f"bad timestamp: {s!r}")
    h, mnt, sec, ms = m.groups()
    return int(h or 0) * 3600 + int(mnt) * 60 + int(sec) + (int(ms.ljust(3, "0")) / 1000 if ms else 0)


def fmt_ts(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def yt_link(url: str | None, sec: float) -> str:
    if not url:
        return ""
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={int(sec)}s"

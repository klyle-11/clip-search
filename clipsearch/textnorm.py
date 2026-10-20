"""Text normalization shared by the FTS index, queries, and the line-level phrase scanner.

Captions and whisper disagree on surface forms ("four in the morning" / "4 in the morning",
"4 a.m." / "4am", "don't" / "dont"). Index and query both go through `norm`, so they meet in the middle.
Display always uses the original text.
"""
import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ONE_N = {w: i for i, w in enumerate(_ONES)}
_TEN_N = {w: i * 10 for i, w in enumerate(_TENS) if w}

_TENS_RE = re.compile(rf"\b({'|'.join(_TEN_N)})(?:[\s-]({'|'.join(_ONES[1:10])}))?\b")
_ONES_RE = re.compile(rf"\b({'|'.join(_ONES)})\b")
_AMPM_RE = re.compile(r"\b(\d{1,2})(?::(\d\d))?\s*([ap])\.?\s?m\b\.?")
_APOS_RE = re.compile(r"['’‘`]")
TOKEN_RE = re.compile(r"[^\W_]+")


def _ampm(m: re.Match) -> str:
    h, mm, ap = m.groups()
    return f"{h}:{mm}{ap}m" if mm and mm != "00" else f"{h}{ap}m"


def norm(text: str) -> str:
    s = _APOS_RE.sub("", text.lower())
    s = _TENS_RE.sub(lambda m: str(_TEN_N[m.group(1)] + (_ONE_N[m.group(2)] if m.group(2) else 0)), s)
    s = _ONES_RE.sub(lambda m: str(_ONE_N[m.group(1)]), s)
    return _AMPM_RE.sub(_ampm, s)


def tokens(text: str) -> list[str]:
    """Tokens of already-normalized text — same split as FTS5's unicode61 tokenizer."""
    return TOKEN_RE.findall(text)


def stem(w: str) -> str:
    """Crude suffix strip so the scanner roughly agrees with FTS5's porter stemmer (waking ~ wake ~ wakes)."""
    if w.isdigit():
        return w
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    return w[:-1] if w.endswith("e") and len(w) > 3 else w

"""Cluster-aware fine token grid and whitespace word grid (S1, decision C11).

Both grids work on a paragraph's logical text (Python str, code-point offsets).

Clusters approximate extended grapheme clusters with the standard library only (risk 4 in
docs/04): a base code point followed by any number of extenders:
  * combining marks (Unicode categories Mn, Mc, Me, e.g. U+0301, keycap U+20E3)
  * variation selectors (U+FE00-U+FE0F, U+E0100-U+E01EF)
  * emoji skin-tone modifiers (U+1F3FB-U+1F3FF)
  * emoji tag characters (U+E0020-U+E007F)
  * ZERO WIDTH JOINER (U+200D) together with the code point it joins
and a pair of regional-indicator symbols (a flag) forms one cluster.

Fine tokens (used by the resolver to check that a finding starts and ends on real boundaries):
  * a maximal run of word clusters (base is a Unicode letter or number), or
  * a single other non-whitespace cluster (punctuation, symbol, emoji).
Whitespace is never part of a token. Fine-token boundaries are always cluster boundaries.

Word grid (the ``WORDS: 0:Our 1:team ...`` hints): maximal runs of non-whitespace characters, as
``str.split()`` defines them, with their offsets.
"""

from __future__ import annotations

import re
import unicodedata

from proofreader.models import Token, Word

_ZWJ = 0x200D
_RI_FIRST, _RI_LAST = 0x1F1E6, 0x1F1FF
_EXTENDER_RANGES = (
    (0xFE00, 0xFE0F),  # variation selectors
    (0xE0100, 0xE01EF),  # variation selectors supplement
    (0x1F3FB, 0x1F3FF),  # emoji modifiers (skin tones)
    (0xE0020, 0xE007F),  # emoji tag sequence characters
)
_WORD_RE = re.compile(r"\S+")


def _is_extender(cp: int) -> bool:
    if unicodedata.category(chr(cp)) in ("Mn", "Mc", "Me"):
        return True
    return any(lo <= cp <= hi for lo, hi in _EXTENDER_RANGES)


def _is_regional_indicator(cp: int) -> bool:
    return _RI_FIRST <= cp <= _RI_LAST


def clusters(text: str) -> list[tuple[int, int]]:
    """(start, end) of every approximate grapheme cluster, covering ``text`` exactly."""
    out: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        start = i
        cp = ord(text[i])
        i += 1
        if _is_regional_indicator(cp) and i < n and _is_regional_indicator(ord(text[i])):
            i += 1  # flag = pair of regional indicators
        while i < n:
            nxt = ord(text[i])
            if _is_extender(nxt):
                i += 1
            elif nxt == _ZWJ:
                i += 1
                if i < n:  # ZWJ joins the following code point (and its extenders) into this cluster
                    i += 1
            else:
                break
        out.append((start, i))
    return out


def _cluster_class(text: str, start: int) -> str:
    base = text[start]
    if base.isspace():
        return "space"
    if unicodedata.category(base)[0] in ("L", "N"):
        return "word"
    return "other"


def fine_tokens(text: str) -> tuple[Token, ...]:
    """Word runs and single non-word clusters, in order; whitespace excluded."""
    tokens: list[Token] = []
    run_start: int | None = None
    run_end = 0
    for start, end in clusters(text):
        kind = _cluster_class(text, start)
        if kind == "word":
            if run_start is None:
                run_start = start
            run_end = end
            continue
        if run_start is not None:
            tokens.append(Token(run_start, run_end, text[run_start:run_end]))
            run_start = None
        if kind == "other":
            tokens.append(Token(start, end, text[start:end]))
    if run_start is not None:
        tokens.append(Token(run_start, run_end, text[run_start:run_end]))
    return tuple(tokens)


def fine_boundaries(text: str) -> frozenset[int]:
    """All offsets at which a fine token starts or ends."""
    out: set[int] = set()
    for tok in fine_tokens(text):
        out.add(tok.start)
        out.add(tok.end)
    return frozenset(out)


def is_aligned(text: str, start: int, end: int) -> bool:
    """True if ``text[start:end]`` is non-empty and both ends lie on fine-token boundaries."""
    if not 0 <= start < end <= len(text):
        return False
    bounds = fine_boundaries(text)
    return start in bounds and end in bounds


def words(text: str) -> tuple[Word, ...]:
    """Whitespace-separated words with indices and offsets (same split as ``str.split()``)."""
    return tuple(Word(i, m.start(), m.end(), m.group()) for i, m in enumerate(_WORD_RE.finditer(text)))

"""Quotation protection (S3.2; D6, docs/02 section 6 as finalised for S3).

Quoted material must never be edited. This module finds quotation marks in a paragraph's logical
text, pairs them conservatively and returns the protected ranges (marks included). Uncertainty
always protects MORE text (a lost correction is acceptable; editing a quotation is not).

Classification context (decision E): for a mark at offset i, p is the base character before it
(combining marks are skipped) and n the character after it.
  * boundary: paragraph start/end, whitespace, ( [ { em dash, en dash, or a nested-<p> barrier
  * punct:    Unicode category P* that is not itself a quotation character
  * L/N:      Unicode letter / number

Pairing (docs/02 section 6, made precise in the S3 plan):
  1. Definite closers pop the nearest matching opener of their family -> "pair" range. Without one,
     protect from the paragraph start to the closer ("empty_stack_close", open_start).
  2. Rule S (single family except the single guillemets): an OPEN / OPEN_CANDIDATE mark protects to
     the LAST CLOSE_OR_APOSTROPHE after it, else to the paragraph end. CLOSE_OR_APOSTROPHE is never
     a definite close; one left outside every range is an apostrophe.
  3. Unclosed openers protect to the paragraph end ("unclosed", open_end).
  4. AMBIGUOUS marks protect from the mark to the paragraph end; an odd count of one kind of
     straight double mark protects from its first unmatched occurrence to the end ("ambiguous").
  5. Guard (decision F): more than 64 non-apostrophe marks, or more than 8 overlapping ranges
     (computed before merging), protects the whole paragraph ("guard").

Nested paragraphs (S3.3, decision U3): each paragraph is analysed on its own text; a child
paragraph inherits whole-paragraph protection only when a protected range of its parent actually
spans the child's barrier (or the parent itself is entirely protected: transitive). See
``analyse_document``.
"""

from __future__ import annotations

import unicodedata
from bisect import bisect_left
from collections.abc import Mapping

from proofreader.models import (
    Paragraph,
    ProtectedRange,
    QuoteAnalysis,
    QuoteFamily,
    QuoteMark,
    QuoteRole,
    RangeOrigin,
    RejectCode,
    ScanResult,
)

MAX_QUOTE_MARKS = 64
MAX_QUOTE_DEPTH = 8

# Quotation characters (decision E table). Primes U+2032/U+2033 are deliberately NOT included.
LEFT_DOUBLE, RIGHT_DOUBLE, LOW_DOUBLE, REVERSED_DOUBLE = chr(0x201C), chr(0x201D), chr(0x201E), chr(0x201F)
STRAIGHT_DOUBLE, FULLWIDTH_DOUBLE = chr(0x22), chr(0xFF02)
LEFT_GUILLEMET, RIGHT_GUILLEMET = chr(0xAB), chr(0xBB)
LEFT_SINGLE_GUILLEMET, RIGHT_SINGLE_GUILLEMET = chr(0x2039), chr(0x203A)
LEFT_SINGLE, RIGHT_SINGLE, STRAIGHT_SINGLE = chr(0x2018), chr(0x2019), chr(0x27)
LOW_SINGLE, REVERSED_SINGLE = chr(0x201A), chr(0x201B)
LEFT_CORNER, RIGHT_CORNER = chr(0x300C), chr(0x300D)
LEFT_WHITE_CORNER, RIGHT_WHITE_CORNER = chr(0x300E), chr(0x300F)
REVERSED_DOUBLE_PRIME, DOUBLE_PRIME, LOW_DOUBLE_PRIME = chr(0x301D), chr(0x301E), chr(0x301F)

_D, _S, _C, _W = QuoteFamily.DOUBLE, QuoteFamily.SINGLE, QuoteFamily.CORNER, QuoteFamily.WHITE_CORNER
_O, _CL = QuoteRole.OPEN, QuoteRole.CLOSE

# Marks whose role does not depend on context.
_FIXED: dict[str, tuple[QuoteFamily, QuoteRole]] = {
    LEFT_DOUBLE: (_D, _O), RIGHT_DOUBLE: (_D, _CL), LOW_DOUBLE: (_D, _O), REVERSED_DOUBLE: (_D, _O),
    LEFT_GUILLEMET: (_D, _O), RIGHT_GUILLEMET: (_D, _CL),
    LEFT_SINGLE_GUILLEMET: (_S, _O), RIGHT_SINGLE_GUILLEMET: (_S, _CL),
    LOW_SINGLE: (_S, _O), REVERSED_SINGLE: (_S, _O),
    LEFT_CORNER: (_C, _O), RIGHT_CORNER: (_C, _CL),
    LEFT_WHITE_CORNER: (_W, _O), RIGHT_WHITE_CORNER: (_W, _CL),
    REVERSED_DOUBLE_PRIME: (_D, _O), DOUBLE_PRIME: (_D, _CL), LOW_DOUBLE_PRIME: (_D, _CL),
}
_STRAIGHT_DOUBLES = (STRAIGHT_DOUBLE, FULLWIDTH_DOUBLE)  # "-rule; each pairs only with itself
_APOSTROPHE_LIKE = (RIGHT_SINGLE, STRAIGHT_SINGLE)  # apostrophe / closer / opener by context

# Definite openers and the closers that end them.
_CLOSERS: dict[str, frozenset[str]] = {
    LEFT_DOUBLE: frozenset({RIGHT_DOUBLE}),
    LOW_DOUBLE: frozenset({RIGHT_DOUBLE}),
    REVERSED_DOUBLE: frozenset({RIGHT_DOUBLE}),
    LEFT_GUILLEMET: frozenset({RIGHT_GUILLEMET}),
    LEFT_SINGLE_GUILLEMET: frozenset({RIGHT_SINGLE_GUILLEMET}),
    LEFT_CORNER: frozenset({RIGHT_CORNER}),
    LEFT_WHITE_CORNER: frozenset({RIGHT_WHITE_CORNER}),
    REVERSED_DOUBLE_PRIME: frozenset({DOUBLE_PRIME, LOW_DOUBLE_PRIME}),
    STRAIGHT_DOUBLE: frozenset({STRAIGHT_DOUBLE}),
    FULLWIDTH_DOUBLE: frozenset({FULLWIDTH_DOUBLE}),
}
_RULE_S_OPENERS = (LEFT_SINGLE, LOW_SINGLE, REVERSED_SINGLE)  # plus OPEN_CANDIDATE apostrophe-likes

QUOTE_CHARACTERS = frozenset(_FIXED) | frozenset(_STRAIGHT_DOUBLES) | frozenset(_APOSTROPHE_LIKE) | {LEFT_SINGLE}
_BOUNDARY_PUNCT = frozenset("([{") | {chr(0x2014), chr(0x2013)}  # ( [ { em dash, en dash


def is_quote_character(ch: str) -> bool:
    """True for every character treated as a quotation delimiter (table + other Pi/Pf), primes excluded."""
    return ch in QUOTE_CHARACTERS or unicodedata.category(ch) in ("Pi", "Pf")


# --------------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------------


def classify(text: str, barriers: tuple[int, ...] = ()) -> tuple[QuoteMark, ...]:
    """Every quotation character in ``text`` with its family and LOCAL (context) role."""
    barrier_set = frozenset(barriers)
    marks: list[QuoteMark] = []
    for i, ch in enumerate(text):
        if ch in QUOTE_CHARACTERS:
            pass
        elif ord(ch) < 0x80 or unicodedata.category(ch) not in ("Pi", "Pf"):
            continue
        marks.append(_classify_one(text, i, ch, barrier_set))
    return tuple(marks)


def _classify_one(text: str, i: int, ch: str, barriers: frozenset[int]) -> QuoteMark:
    if ch in _FIXED:
        family, role = _FIXED[ch]
        return QuoteMark(i, ch, family, role)
    p = _prev_char(text, i, barriers)
    n = _next_char(text, i, barriers)
    if ch in _STRAIGHT_DOUBLES:
        if _is_boundary(p) and _is_non_space(n):
            role = QuoteRole.OPEN
        elif _is_non_space(p) and (_is_boundary(n) or _is_punct(n)):
            role = QuoteRole.CLOSE
        else:
            role = QuoteRole.AMBIGUOUS
        return QuoteMark(i, ch, QuoteFamily.DOUBLE, role)
    if ch == LEFT_SINGLE:
        role = QuoteRole.AMBIGUOUS if (_is_letter(p) and _is_letter(n)) else QuoteRole.OPEN
        return QuoteMark(i, ch, QuoteFamily.SINGLE, role)
    if ch in _APOSTROPHE_LIKE:
        if _is_letter(p) and _is_letter(n):
            role = QuoteRole.APOSTROPHE
        elif _is_number(p) and _is_letter(n):  # digit-letter, e.g. a plural year (S4.8e): never a quotation mark
            role = QuoteRole.APOSTROPHE
        elif _is_boundary(p) and _is_number(n):
            role = QuoteRole.APOSTROPHE
        elif (_is_letter(p) or _is_number(p)) and (_is_boundary(n) or _is_punct(n)):
            role = QuoteRole.CLOSE_OR_APOSTROPHE
        elif _is_boundary(p) and _is_letter(n):
            role = QuoteRole.OPEN_CANDIDATE
        else:
            role = QuoteRole.AMBIGUOUS
        return QuoteMark(i, ch, QuoteFamily.SINGLE, role)
    return QuoteMark(i, ch, QuoteFamily.OTHER, QuoteRole.AMBIGUOUS)  # other Unicode Pi/Pf


# A neighbour is a character, or None for a boundary position (paragraph edge or barrier).
def _prev_char(text: str, i: int, barriers: frozenset[int]) -> str | None:
    j = i
    while True:
        if j in barriers or j == 0:
            return None
        j -= 1
        if not unicodedata.category(text[j]).startswith("M"):  # skip combining marks to the base
            return text[j]


def _next_char(text: str, i: int, barriers: frozenset[int]) -> str | None:
    j = i + 1
    if j in barriers or j >= len(text):
        return None
    return text[j]


def _is_boundary(c: str | None) -> bool:
    return c is None or c.isspace() or c in _BOUNDARY_PUNCT


def _is_non_space(c: str | None) -> bool:
    return c is not None and not c.isspace()


def _is_punct(c: str | None) -> bool:
    return c is not None and unicodedata.category(c).startswith("P") and not is_quote_character(c)


def _is_letter(c: str | None) -> bool:
    return c is not None and unicodedata.category(c).startswith("L")


def _is_number(c: str | None) -> bool:
    return c is not None and unicodedata.category(c).startswith("N")


# --------------------------------------------------------------------------------------------
# Pairing, guard and protected ranges
# --------------------------------------------------------------------------------------------


def analyse_paragraph(paragraph: Paragraph) -> QuoteAnalysis:
    """Quotation analysis of one paragraph's own logical text (barriers count as boundaries)."""
    return analyse_text(paragraph.text, paragraph.barriers, paragraph.id)


def analyse_text(text: str, barriers: tuple[int, ...] = (), paragraph_id: int = 0) -> QuoteAnalysis:
    """Marks (with FINAL roles) and merged protected ranges for ``text``."""
    return _analyse(text, barriers, paragraph_id)[0]


def _analyse(
    text: str, barriers: tuple[int, ...], paragraph_id: int
) -> tuple[QuoteAnalysis, tuple[ProtectedRange, ...]]:
    """The analysis plus the INDIVIDUAL protected ranges (before touching/overlapping ones are merged).

    U3 propagation must ask whether one quotation actually spans a barrier; after merging, two
    separate quotations that merely touch at a barrier would look like one range spanning it.
    """
    marks = classify(text, barriers)
    n = len(text)
    counted = sum(1 for m in marks if m.role is not QuoteRole.APOSTROPHE)
    if counted > MAX_QUOTE_MARKS:
        guarded = _guarded(paragraph_id, marks, n, f"more than {MAX_QUOTE_MARKS} quotation marks ({counted})")
        return guarded, guarded.ranges
    raw = _raw_ranges(marks, n)
    depth = _max_depth(raw)
    if depth > MAX_QUOTE_DEPTH:
        guarded = _guarded(paragraph_id, marks, n, f"quotation nesting deeper than {MAX_QUOTE_DEPTH} ({depth})")
        return guarded, guarded.ranges
    ranges = _merge(raw)
    return QuoteAnalysis(paragraph_id, _final_roles(marks, ranges), ranges, None, None), tuple(raw)


def _guarded(paragraph_id: int, marks: tuple[QuoteMark, ...], n: int, reason: str) -> QuoteAnalysis:
    ranges = (ProtectedRange(0, n, True, True, RangeOrigin.GUARD),) if n else ()
    return QuoteAnalysis(paragraph_id, marks, ranges, reason, None)


def _raw_ranges(marks: tuple[QuoteMark, ...], n: int) -> list[ProtectedRange]:
    ranges: list[ProtectedRange] = []
    stacks: dict[QuoteFamily, list[QuoteMark]] = {}
    paired: set[int] = set()  # offsets of marks that ended up in a definite pair
    last_close_or_apostrophe = max((m.offset for m in marks if m.role is QuoteRole.CLOSE_OR_APOSTROPHE), default=-1)

    for m in marks:
        if m.role is QuoteRole.OPEN and m.char in _CLOSERS:  # definite opener
            stacks.setdefault(m.family, []).append(m)
        elif m.role is QuoteRole.OPEN_CANDIDATE or (m.role is QuoteRole.OPEN and m.char in _RULE_S_OPENERS):
            if last_close_or_apostrophe > m.offset:  # rule S: to the LAST close-or-apostrophe
                ranges.append(ProtectedRange(m.offset, last_close_or_apostrophe + 1, False, False, RangeOrigin.PAIR))
            else:
                ranges.append(ProtectedRange(m.offset, n, False, True, RangeOrigin.UNCLOSED))
        elif m.role is QuoteRole.CLOSE:
            stack = stacks.get(m.family, [])
            for k in range(len(stack) - 1, -1, -1):  # nearest matching opener; others stay open
                if m.char in _CLOSERS[stack[k].char]:
                    opener = stack.pop(k)
                    paired.update((opener.offset, m.offset))
                    ranges.append(ProtectedRange(opener.offset, m.offset + 1, False, False, RangeOrigin.PAIR))
                    break
            else:
                ranges.append(ProtectedRange(0, m.offset + 1, True, False, RangeOrigin.EMPTY_STACK_CLOSE))
        elif m.role is QuoteRole.AMBIGUOUS:
            ranges.append(ProtectedRange(m.offset, n, False, True, RangeOrigin.AMBIGUOUS))

    for stack in stacks.values():
        for opener in stack:
            ranges.append(ProtectedRange(opener.offset, n, False, True, RangeOrigin.UNCLOSED))

    for ch in _STRAIGHT_DOUBLES:  # odd count: protect from the first unmatched occurrence to the end
        same = [m.offset for m in marks if m.char == ch]
        if len(same) % 2:
            first_unmatched = min(o for o in same if o not in paired)
            ranges.append(ProtectedRange(first_unmatched, n, False, True, RangeOrigin.AMBIGUOUS))
    return ranges


def _max_depth(ranges: list[ProtectedRange]) -> int:
    events = sorted([(r.start, 1) for r in ranges] + [(r.end, -1) for r in ranges], key=lambda e: (e[0], e[1]))
    depth = best = 0
    for _, delta in events:  # ends sort before starts at equal offsets (half-open ranges)
        depth += delta
        best = max(best, depth)
    return best


def _merge(ranges: list[ProtectedRange]) -> tuple[ProtectedRange, ...]:
    """Sorted, merged ranges (overlapping or touching ranges combine; first member's origin kept)."""
    merged: list[ProtectedRange] = []
    for r in sorted(ranges, key=lambda r: (r.start, -r.end)):
        if merged and r.start <= merged[-1].end:
            cur = merged[-1]
            open_start = cur.open_start or (r.open_start and r.start == cur.start)
            if r.end > cur.end:
                merged[-1] = ProtectedRange(cur.start, r.end, open_start, r.open_end, cur.origin)
            else:
                merged[-1] = ProtectedRange(cur.start, cur.end, open_start, cur.open_end or (r.open_end and r.end == cur.end), cur.origin)
        else:
            merged.append(r)
    return tuple(merged)


def _final_roles(marks: tuple[QuoteMark, ...], ranges: tuple[ProtectedRange, ...]) -> tuple[QuoteMark, ...]:
    """A CLOSE_OR_APOSTROPHE outside every protected range is an apostrophe."""
    out = []
    for m in marks:
        if m.role is QuoteRole.CLOSE_OR_APOSTROPHE and not any(r.start <= m.offset < r.end for r in ranges):
            m = QuoteMark(m.offset, m.char, m.family, QuoteRole.APOSTROPHE)
        out.append(m)
    return tuple(out)


# --------------------------------------------------------------------------------------------
# Nested paragraphs (S3.3, decision U3)
# --------------------------------------------------------------------------------------------


def _parent_paragraphs(scan: ScanResult) -> dict[int, int | None]:
    """Paragraph id -> id of the nearest enclosing paragraph (any element with local name ``p``, D2)."""
    elements = {e.id: e for e in scan.elements}
    paragraph_of_element = {p.element_id: p.id for p in scan.paragraphs}
    parents: dict[int, int | None] = {}
    for p in scan.paragraphs:
        parent_element = elements[p.element_id].parent_id
        while parent_element is not None and elements[parent_element].local_name != "p":
            parent_element = elements[parent_element].parent_id
        parents[p.id] = None if parent_element is None else paragraph_of_element[parent_element]
    return parents


def _child_barriers(
    scan: ScanResult, elements: Mapping[int, object], paragraph: Paragraph, children: list[int]
) -> tuple[tuple[int, int], ...]:
    raw_starts = [a.raw_start for a in paragraph.atoms]
    out = []
    for child_id in children:
        start_byte = elements[scan.paragraphs[child_id].element_id].start_tag[0]  # type: ignore[attr-defined]
        before = bisect_left(raw_starts, start_byte)  # outer atoms that start before the child's start tag
        out.append((child_id, paragraph.atoms[before - 1].char_end if before else 0))
    return tuple(out)


def nested_barriers(scan: ScanResult, paragraph_id: int) -> tuple[tuple[int, int], ...]:
    """(child paragraph id, barrier char offset) for every directly nested paragraph, in document order.

    Derived from S1 data only (element tree + atom byte ranges); the set of offsets equals
    ``paragraph.barriers`` (several children may share one barrier offset).
    """
    parents = _parent_paragraphs(scan)
    children = [pid for pid, parent in sorted(parents.items()) if parent == paragraph_id]
    return _child_barriers(scan, {e.id: e for e in scan.elements}, scan.paragraphs[paragraph_id], children)


def _covers_barrier(r: ProtectedRange, barrier: int, text_length: int) -> bool:
    """True if the protected range spans a nested paragraph sitting at ``barrier`` (U3).

    Strictly inside the range, or at a paragraph edge the range reaches only because a mark is missing
    (an unmatched closer reaching the start, an unclosed opener reaching the end, or the guard).
    """
    after_start = r.start < barrier or (r.open_start and barrier == r.start == 0)
    before_end = barrier < r.end or (r.open_end and barrier == r.end == text_length)
    return after_start and before_end


def analyse_document(scan: ScanResult) -> Mapping[int, QuoteAnalysis]:
    """Quotation analysis of every paragraph, with conditional propagation into nested paragraphs.

    A child inherits whole-paragraph protection only if one of its parent's INDIVIDUAL protected ranges
    (before merging) spans the child's barrier, or if the parent itself inherited protection
    (transitive). Two quotations that merely touch at the barrier do not protect the child. Siblings
    outside the parent's quotations keep their own analysis.
    """
    parents = _parent_paragraphs(scan)
    children: dict[int, list[int]] = {}
    for pid, parent in sorted(parents.items()):
        if parent is not None:
            children.setdefault(parent, []).append(pid)

    elements = {e.id: e for e in scan.elements}
    result: dict[int, QuoteAnalysis] = {}
    inherited_from: dict[int, int] = {}
    for paragraph in scan.paragraphs:  # document order: parents before their children
        own, own_individual = _analyse(paragraph.text, paragraph.barriers, paragraph.id)
        source = inherited_from.get(paragraph.id)
        if source is None:
            analysis = own
        else:
            n = len(paragraph.text)
            whole = (ProtectedRange(0, n, True, True, RangeOrigin.INHERITED),) if n else ()
            analysis = QuoteAnalysis(paragraph.id, own.marks, whole, own.unsafe_reason, source)
        result[paragraph.id] = analysis

        n = len(paragraph.text)
        for child_id, barrier in _child_barriers(scan, elements, paragraph, children.get(paragraph.id, [])):
            if source is not None or any(_covers_barrier(r, barrier, n) for r in own_individual):
                inherited_from[child_id] = paragraph.id
    return result


# --------------------------------------------------------------------------------------------
# Span check (docs/02 section 6, decision G): used by the resolver in S3.6
# --------------------------------------------------------------------------------------------


def span_verdict(analysis: QuoteAnalysis, start: int, end: int) -> RejectCode | None:
    """Quotation verdict for the span [start, end) of the analysed text, or None if it is allowed.

    Order: the span contains a quotation mark -> TOUCHES_QUOTATION; it overlaps a protected range ->
    IN_QUOTATION; it is adjacent to a protected range -> TOUCHES_QUOTATION. Apostrophes never block.
    """
    if any(start <= m.offset < end and m.role is not QuoteRole.APOSTROPHE for m in analysis.marks):
        return RejectCode.TOUCHES_QUOTATION
    if any(r.start < end and start < r.end for r in analysis.ranges):
        return RejectCode.IN_QUOTATION
    if any(end == r.start or start == r.end for r in analysis.ranges):
        return RejectCode.TOUCHES_QUOTATION
    return None

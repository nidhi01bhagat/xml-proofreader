"""Finding -> resolved span or rejection; anchors are authoritative, word index is an exact tiebreaker only (D9).

ENTRY POINTS (API safety):
  * ``resolve_document(scan, payloads)`` is THE production entry point. It computes document-level
    quotation protection once (``quotes.analyse_document``: nested-paragraph propagation included)
    and applies it to every paragraph; callers cannot switch it off.
  * ``resolve_paragraph(paragraph, payload, quotes)`` requires a quotation analysis (no default).
  * ``resolve_findings(..., quotes=None)`` / ``resolve_finding`` are lower-level building blocks.
    With ``quotes=None`` they keep the S3.5 behaviour and provide NO quotation protection; use them
    only in tests or with an analysis from ``quotes.analyse_document``.

Stages: S3.4 parsing and validation of a paragraph's raw model response (steps 1-7 below);
S3.5 candidate matching and anchor resolution; S3.6 quotation integration, paragraph/document
resolution and tracing (docs/03 section 3).

Interface note (intentional S3.4 deviation from the S3 plan): step 1 needs the scanned paragraph,
which ``parse_findings(paragraph_id, payload)`` (approved signature) does not receive. Step 1 is
therefore the separate ``paragraph_rejection(scan, paragraph_id)``; S3.6 calls it first and only
parses the payload when it returns None.

Paragraph ids come from the scan, never from the model. A non-int id (including bool) is a caller
error and raises ``ResolverInputError`` instead of being stored in a rejection.

Order, per paragraph (deterministic; depends only on the inputs):
  1. paragraph check (``paragraph_rejection``): negative id -> INVALID_FIELD; unknown id or a
     paragraph with a skip reason -> PARA_NOT_ANNOTATABLE (S2 precedent). Nothing is parsed.
  2. envelope: a dict whose "findings" value is a list, else SCHEMA_INVALID (whole paragraph)
  3. RAW count > 30 -> TOO_MANY_FINDINGS (whole paragraph), before any item is validated, so
     invalid and duplicate items count; exactly 30 is allowed
  4. any other top-level key -> SCHEMA_INVALID (whole paragraph)
  5. each item validated on its own (strict Pydantic ``Finding``); a failure rejects only that
     index (INVALID_FIELD), the others continue
  6. conversion to ``CheckedFinding``: edge whitespace stripped from ``original`` (flagged)
  7. span cap: more than 6 words (20 for clarity) in ``original`` -> SPAN_TOO_LONG

Rejection details name fields and rules only, never document or model text.
"""

from __future__ import annotations

import unicodedata
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from proofreader import quotes as q
from proofreader import tokens
from proofreader.models import (
    MAX_FINDINGS_PER_PARAGRAPH,
    Annotation,
    AnnotationOutcome,
    Category,
    CheckedFinding,
    Finding,
    FindingRejection,
    GridWord,
    MatchMode,
    Paragraph,
    QuoteAnalysis,
    QuoteRole,
    RejectCode,
    ResolutionResult,
    ResolvedFinding,
    ScanResult,
    TraceRecord,
    TraceStatus,
)

MAX_SPAN_WORDS = 6
MAX_CLARITY_SPAN_WORDS = 20
_ENVELOPE_KEY = "findings"


class ResolverInputError(TypeError):
    """The caller passed a paragraph id that is not an int (a programming error, never model output)."""


def _require_int_id(paragraph_id: object) -> int:
    if isinstance(paragraph_id, bool) or not isinstance(paragraph_id, int):
        raise ResolverInputError(f"paragraph id must be an int, got {type(paragraph_id).__name__}")
    return paragraph_id


def find_paragraph(scan: ScanResult, paragraph_id: int) -> Paragraph | None:
    """The paragraph whose ``id`` field equals ``paragraph_id``, or None.

    S1's ``scan()`` assigns ids 0..n-1 in document order and stores ``scan.paragraphs`` in that
    order, so the positional lookup is verified against the id field and normally used. A
    ScanResult built differently (ids not equal to positions) is still resolved by its id field.
    """
    paragraphs = scan.paragraphs
    if 0 <= paragraph_id < len(paragraphs) and paragraphs[paragraph_id].id == paragraph_id:
        return paragraphs[paragraph_id]
    return next((p for p in paragraphs if p.id == paragraph_id), None)


def paragraph_rejection(scan: ScanResult, paragraph_id: int) -> FindingRejection | None:
    """Step 1: None if findings for ``paragraph_id`` may be resolved, else a whole-paragraph rejection.

    Raises ``ResolverInputError`` for a non-int id (bool included).
    """
    pid = _require_int_id(paragraph_id)
    if pid < 0:
        return FindingRejection(pid, None, RejectCode.INVALID_FIELD, "paragraph id must be >= 0")
    paragraph = find_paragraph(scan, pid)
    if paragraph is None:
        return FindingRejection(pid, None, RejectCode.PARA_NOT_ANNOTATABLE, "unknown paragraph id")
    return _skip_rejection(paragraph)


def _skip_rejection(paragraph: Paragraph) -> FindingRejection | None:
    if paragraph.skip_reason is not None:
        return FindingRejection(paragraph.id, None, RejectCode.PARA_NOT_ANNOTATABLE,
                                f"paragraph skipped: {paragraph.skip_reason.value}")
    return None


def parse_findings(
    paragraph_id: int, payload: object
) -> tuple[tuple[CheckedFinding, ...], tuple[FindingRejection, ...], int | None]:
    """Steps 2-7: validated findings, rejections and the raw findings count (None if unreadable).

    Raises ``ResolverInputError`` for a non-int ``paragraph_id`` (bool included).
    """
    paragraph_id = _require_int_id(paragraph_id)

    def whole(code: RejectCode, detail: str) -> FindingRejection:
        return FindingRejection(paragraph_id, None, code, detail)

    if not isinstance(payload, dict):
        return (), (whole(RejectCode.SCHEMA_INVALID, "response is not a JSON object"),), None
    if _ENVELOPE_KEY not in payload:
        return (), (whole(RejectCode.SCHEMA_INVALID, "response has no 'findings' array"),), None
    raw = payload[_ENVELOPE_KEY]
    if not isinstance(raw, list):
        return (), (whole(RejectCode.SCHEMA_INVALID, "'findings' is not an array"),), None
    raw_count = len(raw)
    if raw_count > MAX_FINDINGS_PER_PARAGRAPH:  # raw count first: never SCHEMA_INVALID for this
        return (), (whole(RejectCode.TOO_MANY_FINDINGS, f"{raw_count} findings (limit {MAX_FINDINGS_PER_PARAGRAPH})"),), raw_count
    extra = len(payload) - 1
    if extra:
        return (), (whole(RejectCode.SCHEMA_INVALID, f"{extra} unexpected top-level key(s)"),), raw_count

    checked: list[CheckedFinding] = []
    rejected: list[FindingRejection] = []
    for index, item in enumerate(raw):
        try:
            finding = Finding.model_validate(item)
        except ValidationError as exc:
            rejected.append(FindingRejection(paragraph_id, index, RejectCode.INVALID_FIELD, _describe(exc)))
            continue
        result = _check(paragraph_id, index, finding)
        (checked if isinstance(result, CheckedFinding) else rejected).append(result)
    return tuple(checked), tuple(rejected), raw_count


def _check(paragraph_id: int, index: int, finding: Finding) -> CheckedFinding | FindingRejection:
    original = finding.original.strip()
    if not original:  # cannot happen after Finding validation; fail closed anyway
        return FindingRejection(paragraph_id, index, RejectCode.INVALID_FIELD, "original: whitespace only")
    category = Category(finding.type)
    limit = MAX_CLARITY_SPAN_WORDS if category is Category.CLARITY else MAX_SPAN_WORDS
    words = len(original.split())
    if words > limit:
        return FindingRejection(paragraph_id, index, RejectCode.SPAN_TOO_LONG,
                                f"original: {words} words (limit {limit} for {category.value})")
    return CheckedFinding(
        index=index,
        original=original,
        before=finding.before,
        after=finding.after,
        word_start=finding.word_start,
        word_end=finding.word_end,
        correction=finding.correction,
        category=category,
        reason=finding.reason,
        original_stripped=original != finding.original,
    )


def _describe(exc: ValidationError) -> str:
    """Field names and Pydantic error types only (no messages, no input values, no unknown key names)."""
    parts = []
    for error in exc.errors():
        loc = error.get("loc", ())
        if not loc:
            field = "item"
        elif error.get("type") == "extra_forbidden":
            field = "<unexpected field>"
        else:
            field = str(loc[0])
        parts.append(f"{field}: {error.get('type', 'invalid')}")
    return "; ".join(parts)


# ============================================================================================
# S3.5: candidate matching and anchor resolution (D9; S3 plan section 5, decisions A, C, D, U2)
# ============================================================================================
#
# Everything works on the paragraph's logical text (code points). Nested-<p> barriers split the
# text into segments; a barrier is a hard edge for words, fine tokens, anchors and candidates.
#
# Tiers (decision A), cumulative C(T1) <= C(T2) <= C(T3):
#   T1 exact            text[s:e] == original
#   T2 whitespace gap   equal after collapsing every whitespace run to one space (both sides)
#   T3 NFC              equal after NFC-normalising each cluster, with T2's whitespace rule
# Every candidate starts and ends on a fine-token boundary. The first tier with an anchor-valid
# candidate decides; when a tier's candidates all fail their anchors the search continues.
# Anchors (authoritative): compared after edge stripping, whitespace collapsing and per-cluster
# NFC (quote marks stay exact: NFC never changes them); the anchor's far end must lie on a
# fine-token boundary; an empty anchor is valid only at the paragraph edge or a barrier.
# Word indices (0-based, inclusive, barrier-aware grid) only break ties among anchor-valid
# candidates of the deciding tier, by exact equality. No fuzzy matching, no +/-1 fallback.
#
# Quotation checks (S3.6, see _resolve) run between span resolution and the correction check.


def _normalize(text: str, nfc: bool = True) -> str:
    """Whitespace runs -> one space, each cluster NFC-normalised (if ``nfc``), edges stripped."""
    parts: list[str] = []
    previous_space = False
    for start, end in tokens.clusters(text):
        chunk = text[start:end]
        if chunk.isspace():
            if not previous_space:
                parts.append(" ")
                previous_space = True
            continue
        previous_space = False
        parts.append(unicodedata.normalize("NFC", chunk) if nfc else chunk)
    return "".join(parts).strip(" ")


@dataclass(frozen=True)
class _Context:
    """Per-paragraph data shared by all findings of that paragraph (built once)."""

    paragraph: Paragraph
    text: str
    bounds: tuple[int, ...]  # sorted {0, len(text)} | barriers: segment edges
    fine: frozenset[int]  # segment-aware fine-token boundaries (source offsets)
    grid: tuple[GridWord, ...]
    grid_starts: tuple[int, ...]
    view: str  # per-cluster NFC text with whitespace runs collapsed, segment by segment
    view_source_start: tuple[int, ...]  # view char -> source offset where its unit starts
    view_source_end: tuple[int, ...]  # view char -> source offset where its unit ends
    view_units: frozenset[int]  # view indices where a unit (cluster / whitespace run) starts, + len
    view_fine: frozenset[int]  # fine-token boundaries of the view (segment-aware)
    view_of_start: dict[int, int]  # source offset of a non-space cluster start -> view index
    view_of_end: dict[int, int]  # source offset of a non-space cluster end -> view index
    view_of_bound: dict[int, int]  # segment edge (source) -> view index


def _context(paragraph: Paragraph) -> _Context:
    text = paragraph.text
    n = len(text)
    bounds = tuple(sorted({0, n} | {b for b in paragraph.barriers if 0 <= b <= n}))
    fine: set[int] = set()
    grid: list[GridWord] = []
    view: list[str] = []
    source_start: list[int] = []
    source_end: list[int] = []
    units: set[int] = set()
    view_fine: set[int] = set()
    of_start: dict[int, int] = {}
    of_end: dict[int, int] = {}
    of_bound: dict[int, int] = {}
    for seg_start, seg_end in zip(bounds, bounds[1:]):
        of_bound[seg_start] = len(view)
        segment = text[seg_start:seg_end]
        fine.update(seg_start + b for b in tokens.fine_boundaries(segment))
        first_index = len(grid)  # fixed before extending: indices continue across segments
        grid.extend(GridWord(first_index + k, seg_start + w.start, seg_start + w.end, w.text)
                    for k, w in enumerate(tokens.words(segment)))
        view_segment_start = len(view)
        previous_space = False
        for cs, ce in tokens.clusters(segment):
            a, b = seg_start + cs, seg_start + ce
            chunk = segment[cs:ce]
            if chunk.isspace():
                if previous_space:
                    source_end[-1] = b
                else:
                    units.add(len(view))
                    view.append(" ")
                    source_start.append(a)
                    source_end.append(b)
                    previous_space = True
                continue
            previous_space = False
            units.add(len(view))
            of_start[a] = len(view)
            for ch in unicodedata.normalize("NFC", chunk):
                view.append(ch)
                source_start.append(a)
                source_end.append(b)
            of_end[b] = len(view)
        view_text = "".join(view[view_segment_start:])
        view_fine.update(view_segment_start + b for b in tokens.fine_boundaries(view_text))
    of_bound[bounds[-1]] = len(view)
    units.add(len(view))
    return _Context(
        paragraph, text, bounds, frozenset(fine), tuple(grid), tuple(w.start for w in grid),
        "".join(view), tuple(source_start), tuple(source_end), frozenset(units), frozenset(view_fine),
        of_start, of_end, of_bound,
    )


def word_grid(paragraph: Paragraph) -> tuple[GridWord, ...]:
    """Barrier-aware word grid (U2): ``str.split()`` words of each barrier-delimited segment,
    indexed 0, 1, ... in document order. A barrier always ends a word. Source of truth for S4."""
    return _context(paragraph).grid


def _tier_candidates(ctx: _Context, original: str) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Candidate spans for (T1, T2, T3); each tier contains the previous one (cumulative)."""
    text, fine = ctx.text, ctx.fine
    exact: set[tuple[int, int]] = set()
    i = text.find(original)
    while i != -1:
        j = i + len(original)
        if i in fine and j in fine:
            exact.add((i, j))
        i = text.find(original, i + 1)

    normalized: set[tuple[int, int]] = set()
    target = _normalize(original)
    if target:
        view = ctx.view
        k = view.find(target)
        while k != -1:
            end = k + len(target)
            if k in ctx.view_units and end in ctx.view_units:
                s, e = ctx.view_source_start[k], ctx.view_source_end[end - 1]
                if s in fine and e in fine:
                    normalized.add((s, e))
            k = view.find(target, k + 1)

    collapsed = _normalize(original, nfc=False)
    gap = {c for c in normalized if _normalize(text[c[0]:c[1]], nfc=False) == collapsed}
    t1 = exact
    t2 = t1 | gap
    t3 = t2 | normalized
    return tuple(sorted(t1)), tuple(sorted(t2)), tuple(sorted(t3))


def _anchors_valid(ctx: _Context, span: tuple[int, int], before: str, after: str) -> bool:
    """``before`` / ``after`` are already normalised. Anchors never extend across a barrier."""
    s, e = span
    seg_start = ctx.bounds[bisect_right(ctx.bounds, s) - 1]
    seg_end = ctx.bounds[bisect_left(ctx.bounds, e)]
    vs, ve = ctx.view_of_start.get(s), ctx.view_of_end.get(e)
    if vs is None or ve is None:
        return False  # not a cluster boundary: fail closed
    lo, hi = ctx.view_of_bound[seg_start], ctx.view_of_bound[seg_end]
    view = ctx.view

    p = vs - 1 if vs > lo and view[vs - 1] == " " else vs  # skip the separating whitespace
    if before:
        k = p - len(before)
        if k < lo or not view.startswith(before, k) or (k != lo and k not in ctx.view_fine):
            return False
    elif p != lo:
        return False

    q = ve + 1 if ve < hi and view[ve] == " " else ve
    if after:
        m = q + len(after)
        if m > hi or not view.startswith(after, q) or (m != hi and m not in ctx.view_fine):
            return False
    elif q != hi:
        return False
    return True


def _word_range(ctx: _Context, span: tuple[int, int]) -> tuple[int, int] | None:
    def word_at(offset: int) -> int | None:
        k = bisect_right(ctx.grid_starts, offset) - 1
        if k >= 0 and ctx.grid[k].start <= offset < ctx.grid[k].end:
            return k
        return None

    first, last = word_at(span[0]), word_at(span[1] - 1)
    return None if first is None or last is None else (first, last)


def _resolve_span(ctx: _Context, finding: CheckedFinding):
    """(span, match_mode, hint_drift) or (RejectCode, detail)."""
    tiers = _tier_candidates(ctx, finding.original)
    before, after = _normalize(finding.before), _normalize(finding.after)
    hint = (finding.word_start, finding.word_end)
    occurrences = 0
    for mode, candidates in zip(MatchMode, tiers):
        if not candidates:
            continue
        occurrences = len(candidates)
        valid = [c for c in candidates if _anchors_valid(ctx, c, before, after)]
        if not valid:
            continue  # decision A: fall through to the next tier
        if len(valid) == 1:
            chosen, drift = valid[0], _word_range(ctx, valid[0]) != hint
        else:
            hinted = [c for c in valid if _word_range(ctx, c) == hint]
            if len(hinted) != 1:
                return RejectCode.AMBIGUOUS, f"{len(valid)} anchor-valid candidates ({mode.value} tier)"
            chosen, drift = hinted[0], False
        if any(chosen[0] < b < chosen[1] for b in ctx.paragraph.barriers):
            return RejectCode.CROSSES_PARAGRAPH, "span crosses a nested paragraph"
        return chosen, mode, drift
    if occurrences:
        return RejectCode.ANCHOR_MISMATCH, f"{occurrences} occurrence(s), none with matching anchors"
    return RejectCode.NOT_FOUND, "no token-aligned occurrence of the original"


def _resolve(ctx: _Context, finding: CheckedFinding, quotes: QuoteAnalysis | None) -> ResolvedFinding | FindingRejection:
    pid = ctx.paragraph.id
    result = _resolve_span(ctx, finding)
    if isinstance(result[0], RejectCode):
        return FindingRejection(pid, finding.index, result[0], result[1])
    (s, e), mode, drift = result
    if quotes is not None:  # S3.6 quotation checks, before the correction checks
        verdict = q.span_verdict(quotes, s, e)  # contains a mark -> overlap -> adjacency (decision G)
        if verdict is not None:
            return FindingRejection(pid, finding.index, verdict, _SPAN_VERDICT_DETAIL[verdict])
        problem = _introduced_apostrophe_verdict(quotes, ctx.text, ctx.paragraph.barriers, s, e, finding.correction)
        if problem is not None:
            return FindingRejection(pid, finding.index, RejectCode.QUOTATION_IN_CORRECTION, problem)
    if unicodedata.normalize("NFC", finding.correction) == unicodedata.normalize("NFC", ctx.text[s:e]):
        return FindingRejection(pid, finding.index, RejectCode.INVALID_FIELD,
                                "correction: canonically equivalent to the original span (no-op)")
    annotation = Annotation(pid, s, e, finding.category, finding.correction, finding.reason)  # reason for all (I)
    return ResolvedFinding(finding.index, annotation, mode, drift)


def resolve_findings(
    paragraph: Paragraph, findings: Sequence[CheckedFinding], *, quotes: QuoteAnalysis | None = None
) -> tuple[ResolvedFinding | FindingRejection, ...]:
    """Resolve each checked finding of one paragraph independently (one result per finding, in
    the given order; raw indices preserved). The paragraph check (``paragraph_rejection``) is the
    caller's job and must come first.

    WARNING: with ``quotes=None`` (the default, legacy S3.5 behaviour) NO quotation protection is
    applied. Production code must use ``resolve_document``, which always applies it."""
    ctx = _context(paragraph)
    return tuple(_resolve(ctx, f, quotes) for f in findings)


def resolve_finding(
    paragraph: Paragraph, finding: CheckedFinding, *, quotes: QuoteAnalysis | None = None
) -> ResolvedFinding | FindingRejection:
    """Single-finding form of ``resolve_findings`` (same WARNING: ``quotes=None`` = no quotation protection)."""
    return resolve_findings(paragraph, [finding], quotes=quotes)[0]


# ============================================================================================
# S3.6: quotation integration, paragraph/document resolution and traceability
# ============================================================================================
#
# Per paragraph, in this order (S3 plan section 4):
#   1. paragraph eligibility (unknown / malformed id, skip reason)       -> whole paragraph
#   2-7. envelope, raw count, per-item validation, stripping, span caps  (parse_findings)
#   8. tiers, anchors, word tie-break, barriers                          (S3.5)
#   9. quotation span checks (quotes.span_verdict): the span contains a quotation mark ->
#      TOUCHES_QUOTATION; overlaps protected text -> IN_QUOTATION; is adjacent to it ->
#      TOUCHES_QUOTATION. Protection comes from quotes.analyse_document (nested propagation, U3).
#   10. correction checks: quotation delimiters (QUOTATION_IN_CORRECTION, revision-3 semantics
#       below), then the NFC no-op (INVALID_FIELD)
# Then annotation.annotate (S2) decides what is applied; trace() reports every raw finding.

_APOSTROPHE_CHARACTERS = (q.RIGHT_SINGLE, q.STRAIGHT_SINGLE)
_SPAN_VERDICT_DETAIL = {
    RejectCode.IN_QUOTATION: "span overlaps protected quotation content",
    RejectCode.TOUCHES_QUOTATION: "span contains a quotation mark or touches a protected quotation",
}


def _introduced_apostrophe_verdict(
    quotes: QuoteAnalysis, text: str, barriers: tuple[int, ...], s: int, e: int, correction: str
) -> str | None:
    """Revision-3 policy for quotation delimiters in a correction; None if allowed, else a detail.

    The SOURCE quotation structure (rule S, unchanged) is the only authority; corrections never
    re-pair it. In the corrected text T' = text[:s] + correction + text[e:]:
      * any delimiter other than ' and U+2019 (including other Pi/Pf characters) -> rejected
      * each ' / U+2019 the correction introduces, classified locally in T':
        APOSTROPHE -> allowed; CLOSE_OR_APOSTROPHE -> allowed (a possessive outside every completed
        quotation; the span checks already guarantee the position is outside and not adjacent to any
        source range); OPEN_CANDIDATE / AMBIGUOUS -> rejected (would open / create an ambiguous quotation)
      * neighbour stability: existing quotation marks at s-1 and e must keep their local role
      * defensive: an introduced mark at a position covered by / adjacent to a source range -> rejected
    """
    for ch in correction:
        if q.is_quote_character(ch) and ch not in _APOSTROPHE_CHARACTERS:
            return "correction contains a quotation delimiter"
    introduced = [k for k, ch in enumerate(correction) if ch in _APOSTROPHE_CHARACTERS]
    neighbours = [o for o in (s - 1, e) if 0 <= o < len(text) and q.is_quote_character(text[o])]
    if not introduced and not neighbours:
        return None
    shift = len(correction) - (e - s)
    corrected = text[:s] + correction + text[e:]
    corrected_barriers = tuple(b if b <= s else b + shift for b in barriers)
    new_roles = {m.offset: m.role for m in q.classify(corrected, corrected_barriers)}
    if introduced and q.span_verdict(quotes, s, e) is not None:
        return "correction places an apostrophe inside or next to protected quotation content"
    for k in introduced:
        if new_roles[s + k] in (QuoteRole.OPEN_CANDIDATE, QuoteRole.AMBIGUOUS):
            return "correction would open a quotation or create an ambiguous quotation mark"
    if neighbours:
        old_roles = {m.offset: m.role for m in q.classify(text, barriers)}
        for o in neighbours:
            if new_roles.get(o if o < s else o + shift) is not old_roles.get(o):
                return "correction changes the role of an adjacent quotation mark"
    return None


def _raw_count(payload: object) -> int | None:
    if isinstance(payload, dict) and isinstance(payload.get("findings"), list):
        return len(payload["findings"])
    return None


def _rejection_order(r: FindingRejection) -> tuple[bool, int]:
    return (r.finding_index is not None, -1 if r.finding_index is None else r.finding_index)


def resolve_paragraph(paragraph: Paragraph, payload: object, quotes: QuoteAnalysis) -> ResolutionResult:
    """Steps 1-10 for one paragraph. ``quotes`` must be this paragraph's analysis from
    ``quotes.analyse_document`` (so nested-paragraph protection is included)."""
    if quotes.paragraph_id != paragraph.id:
        raise ResolverInputError("quotation analysis belongs to another paragraph")
    skip = _skip_rejection(paragraph)
    if skip is not None:
        return ResolutionResult(paragraph.id, _raw_count(payload), (), (skip,))
    checked, rejected, raw_count = parse_findings(paragraph.id, payload)
    outcomes = resolve_findings(paragraph, checked, quotes=quotes)
    resolved = sorted((x for x in outcomes if isinstance(x, ResolvedFinding)), key=lambda x: x.finding_index)
    rejections = sorted([*rejected, *(x for x in outcomes if isinstance(x, FindingRejection))], key=_rejection_order)
    return ResolutionResult(paragraph.id, raw_count, tuple(resolved), tuple(rejections))


def resolve_document(scan: ScanResult, payloads: Mapping[int, object]) -> tuple[ResolutionResult, ...]:
    """Resolve every paragraph's raw model response; results sorted by paragraph id.

    Recommended production entry point: quotation protection comes from ``quotes.analyse_document``
    (computed once, nested-paragraph propagation included) and is always applied. Keys must be int
    paragraph ids (else ``ResolverInputError``); unknown / skipped / negative ids are rejected as a
    whole paragraph."""
    for key in payloads:
        _require_int_id(key)
    analyses = q.analyse_document(scan)
    results = []
    for pid in sorted(payloads):
        payload = payloads[pid]
        rejection = paragraph_rejection(scan, pid)
        if rejection is not None:
            results.append(ResolutionResult(pid, _raw_count(payload), (), (rejection,)))
            continue
        paragraph = find_paragraph(scan, pid)
        if paragraph is None:  # cannot happen: paragraph_rejection found it
            raise ResolverInputError("paragraph lookup is inconsistent")
        results.append(resolve_paragraph(paragraph, payload, analyses[paragraph.id]))
    return tuple(results)


def annotations(results: Sequence[ResolutionResult]) -> list[Annotation]:
    """The annotations to pass, unchanged, to ``annotation.annotate`` (paragraph, then finding order)."""
    ordered = sorted(results, key=lambda r: r.paragraph_id)
    return [rf.annotation for result in ordered for rf in result.resolved]


def trace(results: Sequence[ResolutionResult], outcome: AnnotationOutcome) -> tuple[TraceRecord, ...]:
    """One record per raw finding, sorted by (paragraph id, raw index); whole-paragraph rejections
    expand to every raw index (or a single record with index None when no list was readable).

    Status of a resolved finding: S2 rejected it in prepare -> REJECTED_S2 (code kept);
    rolled back after verification -> ROLLED_BACK (VERIFY_FAILED); annotate() failed closed ->
    DOCUMENT_FAILED; otherwise it must be in ``outcome.applied`` -> APPLIED.
    Raises ValueError if the outcome does not belong to these results.
    """
    applied = set(outcome.applied)
    s2_code: dict[Annotation, RejectCode] = {}
    for rejection in outcome.rejected:
        s2_code.setdefault(rejection.annotation, rejection.code)

    records: list[TraceRecord] = []
    for result in results:
        pid = result.paragraph_id
        for rej in result.rejections:
            if rej.finding_index is None:
                indices: Sequence[int | None] = range(result.raw_count) if result.raw_count else [None]
            else:
                indices = [rej.finding_index]
            records.extend(TraceRecord(pid, i, TraceStatus.REJECTED_S3, rej.code, None, None, None, None, rej.detail)
                           for i in indices)
        for rf in result.resolved:
            a = rf.annotation
            code = s2_code.get(a)
            if code is RejectCode.VERIFY_FAILED:
                status = TraceStatus.ROLLED_BACK
            elif code is not None:
                status = TraceStatus.REJECTED_S2
            elif not outcome.ok:
                status = TraceStatus.DOCUMENT_FAILED
            elif a in applied:
                status = TraceStatus.APPLIED
            else:
                raise ValueError("annotation outcome does not correspond to these resolution results")
            records.append(TraceRecord(pid, rf.finding_index, status, code, rf.match_mode, a.start, a.end,
                                       rf.hint_drift, ""))
    records.sort(key=lambda t: (t.paragraph_id, t.finding_index is not None, t.finding_index or 0))
    return tuple(records)

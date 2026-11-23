"""Annotation of resolved spans with ``<error>`` elements (S2).

S2.1: field validation of resolved spans (``validate_annotation``).
S2.2: span safety (``check_span``, X1-X8 and X10 with D4 OFF) and deterministic overlap
      handling (``select_disjoint``, ``prepare``) - disjoint spans only, no D15 priority (S3).
S2.3: deterministic rendering (``build_insertions``, ``render``): byte insertion only, no
      re-serialisation. Independent verification and rollback follow in S2.4/S2.5.

Field rules (docs/03 section 3, R09, AC-14, decision 4 of the S2 review):
* ``category`` is a :class:`Category` (one of the six lowercase types).
* ``paragraph_id``, ``start``, ``end`` are integers (``bool`` is rejected); ``paragraph_id >= 0``.
* ``start == end`` -> EMPTY_SPAN; ``start < 0``, ``end > len(text)`` or ``start > end`` -> OUT_OF_RANGE.
* ``correction`` is a string of at most 300 characters (empty = deletion) and differs from
  ``text[start:end]``.
* ``reason`` is required for ``styleguide`` and optional otherwise; when present it is a string of
  1-200 characters that is not only whitespace.
* ``correction`` and ``reason`` contain no character that XML 1.0 forbids: U+0000-U+0008, U+000B,
  U+000C, U+000E-U+001F (docs/03 section 3), plus surrogates U+D800-U+DFFF and U+FFFE/U+FFFF
  (XML 1.0 ``Char`` production; writing them would make the output ill-formed).

Not checked here (S3 finding contract): word-count caps, ``original`` length, anchor fields.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from proofreader.models import (
    Annotation,
    AnnotationOutcome,
    AtomKind,
    Category,
    Insertion,
    Paragraph,
    RejectCode,
    Rejection,
    ScanResult,
)
from proofreader.verify import VerificationReport, verify
from proofreader.xml_source import MappingRejected, close_byte, open_byte

MAX_CORRECTION_CHARS = 300
MAX_REASON_CHARS = 200
_CATEGORY_ORDER = {c: i for i, c in enumerate(Category)}


def _xml_illegal(text: str) -> int | None:
    """Index of the first character not allowed in XML 1.0 text, or None."""
    for i, ch in enumerate(text):
        cp = ord(ch)
        if (
            cp <= 0x08
            or cp in (0x0B, 0x0C)
            or 0x0E <= cp <= 0x1F
            or 0xD800 <= cp <= 0xDFFF
            or cp in (0xFFFE, 0xFFFF)
        ):
            return i
    return None


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_annotation(annotation: Annotation, paragraph_text: str) -> Rejection | None:
    """Check the fields of ``annotation`` against its paragraph's text. None means valid."""

    def reject(code: RejectCode, detail: str) -> Rejection:
        return Rejection(annotation, code, detail)

    a = annotation
    if not isinstance(a.category, Category):
        return reject(RejectCode.INVALID_FIELD, f"category must be a Category, got {a.category!r}")
    for name in ("paragraph_id", "start", "end"):
        if not _is_int(getattr(a, name)):
            return reject(RejectCode.INVALID_FIELD, f"{name} must be an int, got {getattr(a, name)!r}")
    if a.paragraph_id < 0:
        return reject(RejectCode.INVALID_FIELD, f"paragraph_id must be >= 0, got {a.paragraph_id}")

    if a.start == a.end:
        return reject(RejectCode.EMPTY_SPAN, f"empty span at {a.start}")
    if a.start < 0 or a.end > len(paragraph_text) or a.start > a.end:
        return reject(RejectCode.OUT_OF_RANGE, f"span [{a.start}, {a.end}) outside text of length {len(paragraph_text)}")

    if not isinstance(a.correction, str):
        return reject(RejectCode.INVALID_FIELD, f"correction must be a str, got {type(a.correction).__name__}")
    if len(a.correction) > MAX_CORRECTION_CHARS:
        return reject(RejectCode.INVALID_FIELD, f"correction longer than {MAX_CORRECTION_CHARS} characters")
    bad = _xml_illegal(a.correction)
    if bad is not None:
        return reject(RejectCode.INVALID_FIELD, f"correction contains XML-illegal character U+{ord(a.correction[bad]):04X}")
    if a.correction == paragraph_text[a.start : a.end]:
        return reject(RejectCode.INVALID_FIELD, "correction is identical to the original span")

    if a.reason is None:
        if a.category is Category.STYLEGUIDE:
            return reject(RejectCode.INVALID_FIELD, "reason is required for styleguide errors")
        return None
    if not isinstance(a.reason, str):
        return reject(RejectCode.INVALID_FIELD, f"reason must be a str or None, got {type(a.reason).__name__}")
    if not a.reason.strip():
        return reject(RejectCode.INVALID_FIELD, "reason must not be empty or whitespace-only")
    if len(a.reason) > MAX_REASON_CHARS:
        return reject(RejectCode.INVALID_FIELD, f"reason longer than {MAX_REASON_CHARS} characters")
    bad = _xml_illegal(a.reason)
    if bad is not None:
        return reject(RejectCode.INVALID_FIELD, f"reason contains XML-illegal character U+{ord(a.reason[bad]):04X}")
    return None


# --------------------------------------------------------------------------------------------
# S2.2 span safety (docs/02 section 5: X1-X8, X10; D4 OFF)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SafeSpan:
    """An annotation that passed field validation and span safety, with its insertion offsets."""

    annotation: Annotation
    open_offset: int  # byte offset where the opening tag goes
    close_offset: int  # byte offset where the closing tag goes


_MAPPING_CODES = {
    "SPLITS_ATOMIC": RejectCode.SPLITS_ATOMIC,
    "IN_CDATA": RejectCode.IN_CDATA,
    "PARA_UNMAPPABLE": RejectCode.PARA_NOT_ANNOTATABLE,
}


def check_span(data: bytes, scan: ScanResult, annotation: Annotation) -> SafeSpan | Rejection:
    """Validate ``annotation`` against the scanned document. Never guesses or moves boundaries.

    Checks, in this order (the first failure wins):
      1. ``paragraph_id`` well-formed (INVALID_FIELD), known and annotatable (PARA_NOT_ANNOTATABLE:
         unknown id or any skip reason - X6 existing <error>, X10 unmappable, C2 unverifiable,
         EMPTY, CDATA_ONLY)
      2. field validation (``validate_annotation``: INVALID_FIELD / EMPTY_SPAN / OUT_OF_RANGE)
      3. X7  no nested-<p> barrier strictly inside the span            -> CROSSES_PARAGRAPH
      4. X5  no covered CDATA text                                      -> IN_CDATA
      5. X2  both endpoints map through S1 open_byte / close_byte       -> SPLITS_ATOMIC (refs, CRLF)
      6. X3/X4 (D4 OFF) all covered atoms in one container              -> CROSSES_ELEMENT
      7. X3/X5/X8 consecutive covered atoms byte-adjacent: any gap is markup (comment, PI, empty
         element, empty CDATA section)                                  -> CROSSES_ELEMENT / IN_CDATA
    """
    a = annotation
    if not _is_int(a.paragraph_id):  # floats, strings, None, bool ...
        return Rejection(a, RejectCode.INVALID_FIELD, f"paragraph_id must be an int, got {a.paragraph_id!r}")
    if a.paragraph_id < 0:
        return Rejection(a, RejectCode.INVALID_FIELD, f"paragraph_id must be >= 0, got {a.paragraph_id}")
    if a.paragraph_id >= len(scan.paragraphs):
        return Rejection(a, RejectCode.PARA_NOT_ANNOTATABLE, f"unknown paragraph id {a.paragraph_id}")
    paragraph = scan.paragraphs[a.paragraph_id]
    if paragraph.skip_reason is not None:
        return Rejection(a, RejectCode.PARA_NOT_ANNOTATABLE, f"paragraph skipped: {paragraph.skip_reason.value}")

    invalid = validate_annotation(a, paragraph.text)
    if invalid is not None:
        return invalid

    if any(a.start < b < a.end for b in paragraph.barriers):
        return Rejection(a, RejectCode.CROSSES_PARAGRAPH, "span crosses a nested <p>")

    covered = _covered_atoms(paragraph, a.start, a.end)
    for atom in covered:
        if atom.kind is AtomKind.CDATA:
            return Rejection(a, RejectCode.IN_CDATA, "span touches CDATA text")
        if atom.kind is AtomKind.BAD:  # cannot happen for annotatable paragraphs; fail closed anyway
            return Rejection(a, RejectCode.PARA_NOT_ANNOTATABLE, "span covers unmappable content")

    codec = scan.encoding.codec
    try:
        open_offset = open_byte(paragraph, a.start, codec)
        close_offset = close_byte(paragraph, a.end, codec)
    except MappingRejected as exc:
        return Rejection(a, _MAPPING_CODES.get(exc.code, RejectCode.PARA_NOT_ANNOTATABLE), str(exc))

    if len({atom.container_id for atom in covered}) != 1:
        return Rejection(a, RejectCode.CROSSES_ELEMENT, "span crosses inline element boundaries (D4 off)")
    for prev, nxt in zip(covered, covered[1:]):
        if prev.raw_end != nxt.raw_start:
            gap = data[prev.raw_end : nxt.raw_start]
            if b"<![CDATA[" in gap:
                return Rejection(a, RejectCode.IN_CDATA, "span contains a CDATA section")
            return Rejection(a, RejectCode.CROSSES_ELEMENT, "span contains markup (element, comment or PI)")

    return SafeSpan(a, open_offset, close_offset)


def _covered_atoms(paragraph: Paragraph, start: int, end: int):
    return [atom for atom in paragraph.atoms if atom.char_start < end and atom.char_end > start]


# --------------------------------------------------------------------------------------------
# S2.2 deterministic overlap handling (S2 review decisions 3, 5, 6; no D15 priority)
# --------------------------------------------------------------------------------------------


def annotation_sort_key(a: Annotation):
    """Canonical order: independent of the order in which annotations were produced."""
    return (
        a.paragraph_id,
        a.start,
        a.end,
        _CATEGORY_ORDER[a.category],
        a.correction,
        a.reason is not None,
        a.reason or "",
    )


def select_disjoint(spans: Iterable[SafeSpan]) -> tuple[list[SafeSpan], list[Rejection]]:
    """Keep only mutually disjoint spans; never pick a winner.

    * exact duplicates (identical annotations) collapse into one;
    * same span, one category, differing correction/reason -> every one rejected as CONFLICT;
    * same span, several categories                         -> every one rejected as OVERLAP;
    * spans that nest or partially overlap another span     -> every one involved rejected as OVERLAP;
    * adjacent spans (one ends where the next starts) do not overlap and are kept.
    Results are sorted canonically, so input order never matters.
    Every SafeSpan holds a fully validated (hence hashable) annotation, so hashing is safe here.
    """
    unique = {s.annotation: s for s in spans}  # exact duplicates collapse
    ordered = sorted(unique.values(), key=lambda s: annotation_sort_key(s.annotation))

    by_span: dict[tuple[int, int, int], list[SafeSpan]] = {}
    for s in ordered:
        a = s.annotation
        by_span.setdefault((a.paragraph_id, a.start, a.end), []).append(s)

    code_of: dict[Annotation, tuple[RejectCode, str]] = {}
    for (_pid, start, end), group in by_span.items():
        if len(group) > 1:
            if len({s.annotation.category for s in group}) > 1:
                code = (RejectCode.OVERLAP, f"same span [{start}, {end}) proposed with different categories")
            else:
                code = (RejectCode.CONFLICT, f"same span [{start}, {end}) and category, different corrections")
            for s in group:
                code_of[s.annotation] = code

    keys = sorted(by_span)
    for i, (pid, start, end) in enumerate(keys):
        for j in range(i + 1, len(keys)):  # index loop: no per-iteration slice copy (O(n^2) otherwise)
            pid2, start2, end2 = keys[j]
            if pid2 != pid or start2 >= end:
                break  # sorted by (paragraph, start): no later span in this paragraph overlaps this one
            for key, other in (((pid, start, end), (start2, end2)), ((pid2, start2, end2), (start, end))):
                for s in by_span[key]:
                    code_of.setdefault(s.annotation, (RejectCode.OVERLAP, f"overlaps span [{other[0]}, {other[1]})"))

    accepted = [s for s in ordered if s.annotation not in code_of]
    rejected = [Rejection(s.annotation, *code_of[s.annotation]) for s in ordered if s.annotation in code_of]
    return accepted, rejected


def prepare(
    data: bytes, scan: ScanResult, annotations: Iterable[Annotation]
) -> tuple[list[SafeSpan], list[Rejection]]:
    """Span-check every annotation, then keep the mutually disjoint ones (all deterministic).

    Overlap detection considers only annotations that passed the span checks: an invalid or unsafe
    annotation is rejected for its own reason and cannot knock out a valid one.
    """
    safe: list[SafeSpan] = []
    rejected: list[Rejection] = []  # a list, not a dict/set: malformed annotations may be unhashable
    for a in annotations:
        result = check_span(data, scan, a)
        if isinstance(result, SafeSpan):
            safe.append(result)
        else:
            rejected.append(result)
    accepted, overlap_rejected = select_disjoint(safe)
    return accepted, sorted([*_dedup_rejections(rejected), *overlap_rejected], key=_rejection_sort_key)


def _dedup_rejections(rejected: list[Rejection]) -> list[Rejection]:
    """Equality-based dedup that never hashes: sort by the crash-free key, then compare by ``==``
    only within runs of equal keys (equal annotations have equal keys). O(n log n), not O(n^2)."""
    unique: list[Rejection] = []
    run_key: object = None
    run: list[Rejection] = []
    for r in sorted(rejected, key=_rejection_sort_key):
        key = _rejection_sort_key(r)
        if key != run_key:
            run_key, run = key, []
        if not any(u.annotation == r.annotation for u in run):
            run.append(r)
            unique.append(r)
    return unique


def _is_well_formed(a: Annotation) -> bool:
    """True if every field has the type the canonical sort key relies on (values may still be invalid)."""
    return (
        isinstance(a.category, Category)
        and all(_is_int(v) for v in (a.paragraph_id, a.start, a.end))
        and isinstance(a.correction, str)
        and (a.reason is None or isinstance(a.reason, str))
    )


def _rejection_sort_key(r: Rejection):
    """Total, crash-free order: well-formed annotations canonically, then malformed ones by their text."""
    if _is_well_formed(r.annotation):
        return (0, annotation_sort_key(r.annotation), "", r.code.value)
    return (1, (), repr(r.annotation), r.code.value)


# --------------------------------------------------------------------------------------------
# S2.3 deterministic rendering (byte insertion only; X11 ordering; E02 attribute format)
# --------------------------------------------------------------------------------------------

_CLOSE_TAG = b"</error>"
_ATTR_ESCAPES = (  # "&" first; \n \r \t as references so attribute-value normalisation cannot alter them
    ("&", "&amp;"),
    ("<", "&lt;"),
    (">", "&gt;"),
    ('"', "&quot;"),
    ("\n", "&#10;"),
    ("\r", "&#13;"),
    ("\t", "&#9;"),
)


class RenderInputError(ValueError):
    """Rendering was given input that ``prepare`` would never produce (a programming error).

    The renderer never repairs, drops or rewrites such input; it refuses to render.
    """


def escape_attribute(value: str, codec: str) -> bytes:
    """Escape ``value`` for a double-quoted XML attribute and encode it in the document's codec.

    Characters the codec cannot represent (e.g. non-ASCII in a US-ASCII document) become numeric
    character references, so the value round-trips exactly through any XML parser.
    """
    bad = _xml_illegal(value)
    if bad is not None:
        raise RenderInputError(f"attribute value contains XML-illegal character U+{ord(value[bad]):04X}")
    for old, new in _ATTR_ESCAPES:
        value = value.replace(old, new)
    return value.encode(codec, "xmlcharrefreplace")


def opening_tag(annotation: Annotation, codec: str) -> bytes:
    """``<error type="..." correction="..." [reason="..."]>`` (attribute order and quoting per E02)."""
    a = annotation
    if not isinstance(a.category, Category):
        raise RenderInputError(f"category must be a Category, got {a.category!r}")
    if not isinstance(a.correction, str):
        raise RenderInputError("correction must be a str")
    if a.reason is not None and (not isinstance(a.reason, str) or not a.reason.strip()):
        raise RenderInputError("reason must be None or a non-empty str")
    if a.category is Category.STYLEGUIDE and a.reason is None:
        raise RenderInputError("styleguide errors require a reason")
    tag = b'<error type="' + a.category.value.encode("ascii") + b'" correction="' + escape_attribute(a.correction, codec) + b'"'
    if a.reason is not None:
        tag += b' reason="' + escape_attribute(a.reason, codec) + b'"'
    return tag + b">"


def build_insertions(spans: Iterable[SafeSpan], codec: str, data_length: int) -> list[Insertion]:
    """Opening/closing tag insertions for validated, mutually disjoint spans, in render order.

    Order: ascending byte offset; at an equal offset the closing tag of one span goes before the
    opening tag of the next (adjacent spans). Input order never matters.
    """
    # Validate every span and build its tag BEFORE any hashing: invalid input must raise
    # RenderInputError, never TypeError (malformed annotations may hold unhashable values).
    tagged: list[tuple[SafeSpan, bytes]] = []
    for s in spans:
        if not isinstance(s, SafeSpan):
            raise RenderInputError(f"render() only accepts SafeSpan objects, got {type(s).__name__}")
        if not (_is_int(s.open_offset) and _is_int(s.close_offset) and 0 <= s.open_offset < s.close_offset <= data_length):
            raise RenderInputError(f"invalid byte range [{s.open_offset}, {s.close_offset}) for {data_length} bytes")
        if not _is_int(s.annotation.paragraph_id):
            raise RenderInputError(f"paragraph_id must be an int, got {s.annotation.paragraph_id!r}")
        tagged.append((s, opening_tag(s.annotation, codec)))

    # Sort by byte range (stable), then compare neighbours: O(n log n), never hashes. Identical
    # spans share a range and are therefore adjacent; a different span on the same range is an error.
    tagged.sort(key=lambda item: (item[0].open_offset, item[0].close_offset))
    unique: list[tuple[SafeSpan, bytes]] = []
    for s, tag in tagged:
        if unique:
            prev = unique[-1][0]
            if (s.open_offset, s.close_offset) == (prev.open_offset, prev.close_offset):
                if s == prev:
                    continue  # exact duplicate collapses (equality, as in select_disjoint)
                raise RenderInputError("two different annotations share one byte range")
        unique.append((s, tag))
    for (prev, _), (nxt, _) in zip(unique, unique[1:]):
        if nxt.open_offset < prev.close_offset:
            raise RenderInputError(
                f"overlapping byte ranges [{prev.open_offset}, {prev.close_offset}) and [{nxt.open_offset}, {nxt.close_offset})"
            )
    insertions: list[Insertion] = []
    for s, tag in unique:
        pid = s.annotation.paragraph_id
        insertions.append(Insertion(s.open_offset, 1, tag, pid))
        insertions.append(Insertion(s.close_offset, 0, _CLOSE_TAG, pid))
    insertions.sort(key=lambda ins: (ins.offset, ins.order))
    return insertions


def render(data: bytes, spans: Iterable[SafeSpan], codec: str) -> tuple[bytes, tuple[tuple[int, int], ...]]:
    """Insert ``<error>`` tags into ``data``; every original byte is kept, in order, unchanged.

    Returns the new document and the byte ranges (in the output) occupied by inserted tags, in order.
    No parsing, normalisation or re-serialisation happens here.
    """
    if not isinstance(data, bytes):
        raise RenderInputError("data must be bytes")
    insertions = build_insertions(spans, codec, len(data))
    out = bytearray()
    ranges: list[tuple[int, int]] = []
    prev = 0
    for ins in insertions:
        out += data[prev : ins.offset]
        ranges.append((len(out), len(out) + len(ins.data)))
        out += ins.data
        prev = ins.offset
    out += data[prev:]
    return bytes(out), tuple(ranges)


# --------------------------------------------------------------------------------------------
# S2.5 annotate with independent verification and per-paragraph rollback (X12)
# --------------------------------------------------------------------------------------------


def _describe(issue) -> str:
    where = "document" if issue.paragraph_id is None else f"paragraph {issue.paragraph_id}"
    return f"{issue.code} ({where}): {issue.message}"


def annotate(
    data: bytes,
    scan: ScanResult,
    annotations: Iterable[Annotation],
    *,
    render_fn=None,
    verify_fn=None,
) -> AnnotationOutcome:
    """Prepare, render and independently verify; roll back failing paragraphs; fail closed.

    Every round re-renders from the ORIGINAL bytes with only the annotations of paragraphs that
    have not failed (no byte-range surgery on rendered output), then verifies the exact bytes that
    would be returned. Rollback is per paragraph id, so nested paragraphs are independent.

    Fail closed (``ok=False``, ``output=None``) when verification reports a document-level issue,
    blames a paragraph that has no retained annotations, the renderer refuses its input, or the
    retained set stops shrinking. ``ok=True`` is returned only for output that verified cleanly.

    ``render_fn`` / ``verify_fn`` default to :func:`render` and :func:`proofreader.verify.verify`;
    they are parameters only so tests can inject faults.
    """
    render_fn = render_fn or render
    verify_fn = verify_fn or verify
    codec = scan.encoding.codec

    accepted, rejected = prepare(data, scan, annotations)
    rejected_out = list(rejected)
    retained = list(accepted)
    rolled_back: set[int] = set()
    rollback_notes: list[str] = []
    max_rounds = len({s.annotation.paragraph_id for s in accepted}) + 1

    def fail(reasons, rounds) -> AnnotationOutcome:
        return AnnotationOutcome(False, None, (), tuple(sorted(rolled_back)),
                                 tuple(sorted(rejected_out, key=_rejection_sort_key)),
                                 tuple(rollback_notes) + tuple(reasons), rounds)

    for round_no in range(1, max_rounds + 1):
        try:
            output, _ = render_fn(data, retained, codec)
        except RenderInputError as exc:
            return fail([f"RENDER_REFUSED (document): {exc}"], round_no)
        applied = [s.annotation for s in retained]
        report: VerificationReport = verify_fn(data, output, applied)
        if report.ok:
            return AnnotationOutcome(True, output, tuple(applied), tuple(sorted(rolled_back)),
                                     tuple(sorted(rejected_out, key=_rejection_sort_key)),
                                     tuple(rollback_notes), round_no)
        if report.document_issues:
            return fail([_describe(i) for i in report.document_issues], round_no)
        failed = set(report.failed_paragraphs)
        annotated = {s.annotation.paragraph_id for s in retained}
        unexplained = failed - annotated
        if unexplained:  # the verifier blames paragraphs we did not touch: not recoverable by rollback
            return fail([_describe(i) for pid in sorted(unexplained) for i in report.issues_for(pid)], round_no)
        for pid in sorted(failed):
            detail = "; ".join(f"{i.code}: {i.message}" for i in report.issues_for(pid))
            rollback_notes.append(f"rolled back paragraph {pid}: {detail}")
            for s in retained:
                if s.annotation.paragraph_id == pid:
                    rejected_out.append(Rejection(s.annotation, RejectCode.VERIFY_FAILED, detail))
        rolled_back |= failed
        retained = [s for s in retained if s.annotation.paragraph_id not in failed]

    return fail(["ROLLBACK_EXHAUSTED (document): verification did not converge"], max_rounds)

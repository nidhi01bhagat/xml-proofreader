"""Independent output verification (S2.4; docs/02 section 8: G1, V1-V6).

The verifier establishes correctness from three inputs only: the original bytes, the rendered
output bytes and the applied annotations. It does NOT use the renderer's insertion offsets or
byte ranges, nor S1's source map: it re-derives everything with its own parsing.

Two parsing paths, both separate from the renderer:
* ElementTree (the approved verifier path): paragraph own-text, the text, logical position and
  attributes of every inserted ``<error>`` (V1, V2, V3, V6).
* A minimal, namespace-aware expat event recorder: byte positions of ``<error>`` tags in the
  output (so G1 can be recomputed independently), nesting and paragraph placement (V4), the
  structural event stream (V5) and namespace inheritance (inserted tags must be unprefixed and
  declare no namespace, so they inherit the default namespace in scope - S2.3 decision).
  It never fetches external DTDs or entities.

Paragraphs are identified as in S1: elements with local name ``p`` in document order (D2).
"Inserted" error elements are the ``<error>`` elements whose nearest ``<p>`` is a paragraph that
received annotations; such paragraphs contained no ``<error>`` in the original (X6), so any
``<error>`` there must correspond to an applied annotation.

Results are structured: document-level issues make the whole output unusable; paragraph-level
issues identify paragraphs that S2.5 can roll back independently.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import xml.parsers.expat as expat
from collections.abc import Sequence
from dataclasses import dataclass

from proofreader.models import Annotation, Category

_NS = " "  # namespace separator for the expat recorder


class IssueCode:
    ORIGINAL_UNPARSEABLE = "ORIGINAL_UNPARSEABLE"  # verification impossible (document level)
    OUTPUT_NOT_WELL_FORMED = "OUTPUT_NOT_WELL_FORMED"  # document level
    UNCHANGED_EXPECTED = "UNCHANGED_EXPECTED"  # no annotations applied, but output differs (document level)
    INVALID_ANNOTATION = "INVALID_ANNOTATION"  # annotation refers to an unknown paragraph (document level)
    PARAGRAPH_COUNT_CHANGED = "PARAGRAPH_COUNT_CHANGED"  # document level
    BYTES_NOT_RESTORED = "BYTES_NOT_RESTORED"  # G1
    STRUCTURE_CHANGED = "STRUCTURE_CHANGED"  # V5
    TEXT_CHANGED = "TEXT_CHANGED"  # V6
    MISSING_ERROR = "MISSING_ERROR"  # V4: annotation without an inserted element
    EXTRA_ERROR = "EXTRA_ERROR"  # V4: inserted element without an annotation
    SPAN_MISMATCH = "SPAN_MISMATCH"  # V1/V2
    ATTRIBUTES_MISMATCH = "ATTRIBUTES_MISMATCH"  # V3
    NESTED_ERROR = "NESTED_ERROR"  # V4: <error> inside <error>
    ERROR_OUTSIDE_PARAGRAPH = "ERROR_OUTSIDE_PARAGRAPH"  # V4: new <error> with no <p> ancestor
    NAMESPACE_MISMATCH = "NAMESPACE_MISMATCH"  # prefixed tag or namespace declaration on an inserted <error>
    TAG_NOT_CANONICAL = "TAG_NOT_CANONICAL"  # inserted tag bytes deviate from the E02 byte layout (S2.6)
    ENCODING_UNSUPPORTED = "ENCODING_UNSUPPORTED"  # original's encoding not UTF-8/US-ASCII or inconsistent (document level)


# The verifier's own view of the supported encodings (deliberately not imported from S1).
# Names are compared case-insensitively; no declaration means UTF-8 (XML 1.0, section 4.3.3).
_UTF8_BOM = b"\xef\xbb\xbf"
_UTF8_NAMES = frozenset({"utf-8", "utf8"})
_ASCII_NAMES = frozenset({"us-ascii", "ascii"})
_ESCAPED_CONTROLS = frozenset({9, 10, 13})  # TAB, LF, CR: always written as numeric references
_DECL_SPACE = (b" ", b"\t", b"\r", b"\n")  # XML 1.0 "S" after "<?xml" in a declaration
# Expat auto-detects UTF-16 from a BOM or from its NUL-byte pattern even when told the input is UTF-8,
# so these are rejected on the raw bytes, before any parsing. UTF-32 BOMs first (prefix of UTF-16 LE).
_FOREIGN_BOMS = (
    (b"\x00\x00\xfe\xff", "UTF-32 (big-endian)"),
    (b"\xff\xfe\x00\x00", "UTF-32 (little-endian)"),
    (b"\xfe\xff", "UTF-16 (big-endian)"),
    (b"\xff\xfe", "UTF-16 (little-endian)"),
)


# Canonical E02 byte layout of inserted tags (checked on the raw bytes, independent of the renderer):
# '<error type="T" correction="C"[ reason="R"]>' ... '</error>' - single spaces, double quotes, no
# other whitespace; values use only &amp; &lt; &gt; &quot; and decimal references for TAB/LF/CR or
# non-ASCII characters. Attribute VALUES are compared separately (V3, via ElementTree).
_VALUE = rb'(?:[^"<>&\t\n\r]|&(?:amp|lt|gt|quot);|&#[0-9]+;)*'
_CANONICAL_OPEN = re.compile(rb'<error type="[a-z]+" correction="' + _VALUE + rb'"(?: reason="' + _VALUE + rb'")?>\Z')
_CHAR_REF = re.compile(rb"&#([0-9]+);")


@dataclass(frozen=True)
class VerificationIssue:
    code: str
    message: str
    paragraph_id: int | None = None
    # Where the problem was found, if known: an ORIGINAL-document offset for BYTES_NOT_RESTORED and
    # UNCHANGED_EXPECTED (first differing byte), an OUTPUT offset (start of the tag) otherwise.
    byte_offset: int | None = None


@dataclass(frozen=True)
class VerificationReport:
    document_issues: tuple[VerificationIssue, ...]
    paragraph_issues: tuple[tuple[int, tuple[VerificationIssue, ...]], ...]  # sorted by paragraph id

    @property
    def ok(self) -> bool:
        return not self.document_issues and not self.paragraph_issues

    @property
    def failed_paragraphs(self) -> tuple[int, ...]:
        return tuple(pid for pid, _ in self.paragraph_issues)

    def issues_for(self, paragraph_id: int) -> tuple[VerificationIssue, ...]:
        return dict(self.paragraph_issues).get(paragraph_id, ())


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------


def verify(original: bytes, output: bytes, applied: Sequence[Annotation]) -> VerificationReport:
    """Check ``output`` against ``original`` and the annotations that were supposed to be applied."""
    doc: list[VerificationIssue] = []
    para: dict[int, list[VerificationIssue]] = {}

    def add(issue: VerificationIssue) -> None:
        if issue.paragraph_id is None:
            doc.append(issue)
        else:
            para.setdefault(issue.paragraph_id, []).append(issue)

    # Raw byte check first, on every path (also when nothing was applied): a byte scan, no parsing,
    # so the fast path for unchanged UTF-8/US-ASCII documents stays cheap.
    raw_problem = _raw_encoding_problem(original)
    if raw_problem:  # e.g. UTF-16/UTF-32 without a declaration: never classified as UTF-8
        add(VerificationIssue(IssueCode.ENCODING_UNSUPPORTED, raw_problem))
        return _report(doc, para)

    applied = list(applied)
    if not applied:
        # The declared encoding is checked here too (cheap: nothing is parsed without a declaration).
        well_formed, declared = _declaration_encoding(original)
        if not well_formed:
            add(VerificationIssue(IssueCode.ORIGINAL_UNPARSEABLE, "the XML declaration of the original is not well-formed"))
            return _report(doc, para)
        _, noop_problem = _document_encoding(original, declared)
        if noop_problem:
            add(VerificationIssue(IssueCode.ENCODING_UNSUPPORTED, noop_problem))
            return _report(doc, para)
        if output != original:
            add(VerificationIssue(IssueCode.UNCHANGED_EXPECTED, "no annotations were applied but the output differs",
                                  byte_offset=_first_difference(original, output)))
        return _report(doc, para)

    orig_tree = _parse_tree(original)
    orig_events = _record(original)
    if orig_tree is None or orig_events is None:
        add(VerificationIssue(IssueCode.ORIGINAL_UNPARSEABLE, "the original document cannot be parsed for verification"))
        return _report(doc, para)
    encoding, encoding_problem = _document_encoding(original, orig_events.declared_encoding)
    if encoding_problem:  # never fall back to UTF-8 for input we cannot classify
        add(VerificationIssue(IssueCode.ENCODING_UNSUPPORTED, encoding_problem))
        return _report(doc, para)
    out_tree = _parse_tree(output)
    out_events = _record(output)
    if out_tree is None or out_events is None:
        add(VerificationIssue(IssueCode.OUTPUT_NOT_WELL_FORMED, "the output is not well-formed XML"))
        return _report(doc, para)

    orig_paras = _paragraph_elements(orig_tree)
    out_paras = _paragraph_elements(out_tree)
    if len(orig_paras) != len(out_paras) or len(orig_events.paragraphs) != len(out_events.paragraphs):
        add(VerificationIssue(IssueCode.PARAGRAPH_COUNT_CHANGED,
                              f"paragraph count changed: {len(orig_paras)} -> {len(out_paras)}"))
        return _report(doc, para)

    expected: dict[int, list[Annotation]] = {}
    for a in applied:
        if not _is_int(a.paragraph_id) or not 0 <= a.paragraph_id < len(orig_paras):
            add(VerificationIssue(IssueCode.INVALID_ANNOTATION, f"annotation refers to unknown paragraph {a.paragraph_id!r}"))
            continue
        if not (_is_int(a.start) and _is_int(a.end) and 0 <= a.start < a.end):
            add(VerificationIssue(IssueCode.INVALID_ANNOTATION, f"annotation has an invalid span [{a.start!r}, {a.end!r})"))
            continue
        bucket = expected.setdefault(a.paragraph_id, [])
        if not any(a == b for b in bucket):  # equality-based dedup: never hashes caller-supplied values
            bucket.append(a)
    if doc:
        return _report(doc, para)
    for pid in expected:
        expected[pid].sort(key=lambda a: (a.start, a.end))

    # ---- byte/structure checks from the event recorder (independent of the renderer) ----
    inserted = [e for e in out_events.errors if e.paragraph in expected]
    orig_outside = sum(1 for e in orig_events.errors if e.paragraph is None)
    out_outside = [e for e in out_events.errors if e.paragraph is None]
    if len(out_outside) > orig_outside:
        add(VerificationIssue(IssueCode.ERROR_OUTSIDE_PARAGRAPH,
                              f"{len(out_outside) - orig_outside} new <error> element(s) without a <p> ancestor",
                              byte_offset=out_outside[-1].start[0]))
    for e in inserted:
        if e.nested:
            add(VerificationIssue(IssueCode.NESTED_ERROR, "inserted <error> is nested inside another <error>",
                                  e.paragraph, e.start[0]))
        if not e.unprefixed or e.declares_namespace:
            add(VerificationIssue(IssueCode.NAMESPACE_MISMATCH,
                                  "inserted <error> must be unprefixed and declare no namespace (it inherits the one in scope)",
                                  e.paragraph, e.start[0]))
        if not _canonical_tags(output[e.start[0] : e.start[1]], output[e.end[0] : e.end[1]], encoding):
            add(VerificationIssue(IssueCode.TAG_NOT_CANONICAL,
                                  f"inserted <error> tags are not in the canonical E02 byte layout for {encoding}",
                                  e.paragraph, e.start[0]))

    tag_ranges = sorted(r for e in inserted for r in (e.start, e.end))
    restored = _remove(output, tag_ranges)
    if restored != original:
        offset = _first_difference(original, restored)
        culprit = _paragraph_at(orig_events, offset)
        add(VerificationIssue(IssueCode.BYTES_NOT_RESTORED,
                              f"removing the inserted tags does not restore the original (first difference at original byte {offset})",
                              culprit if culprit in expected else None, offset))

    # V5: skip the inserted error tags themselves and any namespace declaration events attached to
    # them (those are reported as NAMESPACE_MISMATCH for the paragraph, not as a document failure).
    tag_bytes = {e.start[0] for e in inserted} | {e.end[0] for e in inserted}
    skip = {e.start_index for e in inserted} | {e.end_index for e in inserted}
    skip |= {i for i, (idx, kind, _) in enumerate(out_events.events) if kind in ("N", "n") and idx in tag_bytes}
    orig_struct = _structure(orig_events.events, set())
    out_struct = _structure(out_events.events, skip)
    if [item[1:] for item in orig_struct] != [item[1:] for item in out_struct]:
        k = next((i for i, (a, b) in enumerate(zip(orig_struct, out_struct)) if a[1:] != b[1:]), min(len(orig_struct), len(out_struct)))
        at = out_struct[k][0] if k < len(out_struct) else len(output)
        culprit = _paragraph_at(out_events, at)
        add(VerificationIssue(IssueCode.STRUCTURE_CHANGED, "markup outside the inserted <error> tags changed (V5)",
                              culprit if culprit in expected else None, at))

    # ---- text/attribute checks from ElementTree (independent of the renderer) ----
    for pid, (orig_p, out_p) in enumerate(zip(orig_paras, out_paras)):
        orig_text = _own_text(orig_p)
        out_text, errors = _own_text_and_errors(out_p)
        if out_text != orig_text:
            add(VerificationIssue(IssueCode.TEXT_CHANGED, "paragraph text changed (V6)", pid))
        if pid not in expected:
            continue
        _compare_errors(pid, orig_text, errors, expected[pid], add)

    return _report(doc, para)


# --------------------------------------------------------------------------------------------
# Error correspondence (V1-V4)
# --------------------------------------------------------------------------------------------


def _compare_errors(pid, text, found, wanted, add) -> None:
    found_by_span = {}
    for start, end, el in found:
        found_by_span.setdefault((start, end), []).append(el)
    for a in wanted:
        hits = found_by_span.pop((a.start, a.end), [])
        if not hits:
            add(VerificationIssue(IssueCode.MISSING_ERROR, f"no <error> covers annotated span [{a.start}, {a.end})", pid))
            continue
        if len(hits) > 1:
            add(VerificationIssue(IssueCode.EXTRA_ERROR, f"{len(hits)} <error> elements cover span [{a.start}, {a.end})", pid))
        el = hits[0]
        flattened = "".join(el.itertext())
        if flattened != text[a.start : a.end]:
            add(VerificationIssue(IssueCode.SPAN_MISMATCH,
                                  f"<error> text {flattened!r} != original {text[a.start:a.end]!r}", pid))
        want_attrs = [("type", a.category.value if isinstance(a.category, Category) else a.category),
                      ("correction", a.correction)]
        if a.reason is not None:
            want_attrs.append(("reason", a.reason))
        got_attrs = list(el.attrib.items())
        if got_attrs != want_attrs:
            add(VerificationIssue(IssueCode.ATTRIBUTES_MISMATCH,
                                  f"attributes {got_attrs!r} != expected {want_attrs!r} for span [{a.start}, {a.end})", pid))
    for (start, end), els in sorted(found_by_span.items()):
        add(VerificationIssue(IssueCode.EXTRA_ERROR,
                              f"{len(els)} unexpected <error> element(s) at span [{start}, {end})", pid))


# --------------------------------------------------------------------------------------------
# ElementTree path
# --------------------------------------------------------------------------------------------


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _parse_tree(data: bytes):
    try:
        # UTF-8 override: encoding.check() guarantees UTF-8 or pure ASCII input
        return ET.fromstring(data, parser=ET.XMLParser(encoding="utf-8"))
    except ET.ParseError:
        return None


def _paragraph_elements(root) -> list:
    return [el for el in root.iter() if _local(el.tag) == "p"]


def _own_text(el) -> str:
    return _own_text_and_errors(el)[0]


def _own_text_and_errors(el):
    """Own text of a paragraph (nested <p> excluded) and every <error> in its own scope.

    Returns (text, [(start, end, error_element), ...]) with logical char offsets from a
    document-order walk. Iterative, so deep nesting cannot hit Python's recursion limit.
    """
    parts = [el.text or ""]
    pos = len(parts[0])
    errors = []
    open_errors: list[tuple[int, object]] = []
    frames: list[tuple[object, object]] = [(iter(list(el)), None)]  # (children iterator, element closed on pop)
    while frames:
        it, owner = frames[-1]
        child = next(it, None)  # type: ignore[call-overload]
        if child is None:
            frames.pop()
            if owner is not None:
                if open_errors and open_errors[-1][1] is owner:
                    start, err = open_errors.pop()
                    errors.append((start, pos, err))
                tail = owner.tail or ""  # type: ignore[attr-defined]
                parts.append(tail)
                pos += len(tail)
            continue
        if _local(child.tag) == "p":
            tail = child.tail or ""
            parts.append(tail)
            pos += len(tail)
            continue
        if _local(child.tag) == "error":
            open_errors.append((pos, child))
        text = child.text or ""
        parts.append(text)
        pos += len(text)
        frames.append((iter(list(child)), child))
    errors.sort(key=lambda t: (t[0], t[1]))
    return "".join(parts), errors


# --------------------------------------------------------------------------------------------
# Minimal expat event recorder (byte positions; independent of the renderer and of S1)
# --------------------------------------------------------------------------------------------


@dataclass
class _ErrorEl:
    start: tuple[int, int]
    end: tuple[int, int]
    start_index: int
    end_index: int
    paragraph: int | None
    nested: bool
    unprefixed: bool
    declares_namespace: bool


@dataclass
class _Recording:
    events: list  # (byte_index, kind, payload)
    paragraphs: list[tuple[int, int]]  # byte range of each <p> element (start of start tag, end of end tag)
    errors: list[_ErrorEl]
    declared_encoding: str | None = None  # encoding="..." of the XML declaration, as written; None if absent


def _record(data: bytes) -> _Recording | None:
    parser = expat.ParserCreate(encoding="UTF-8", namespace_separator=_NS)
    parser.ordered_attributes = True
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    parser.ExternalEntityRefHandler = lambda *args: 1  # never fetch anything
    events: list = []
    declaration: list[str | None] = []

    def on_declaration(_version, encoding, _standalone):
        declaration.append(encoding)

    parser.XmlDeclHandler = on_declaration

    def rec(kind):
        return lambda *args: events.append((parser.CurrentByteIndex, kind, args))

    def on_text(_text):
        if not events or events[-1][1] != "T" or events[-1][0] < 0:
            events.append((parser.CurrentByteIndex, "T", ()))

    parser.StartElementHandler = rec("S")
    parser.EndElementHandler = rec("E")
    parser.CommentHandler = rec("C")
    parser.ProcessingInstructionHandler = rec("P")
    parser.StartNamespaceDeclHandler = rec("N")
    parser.EndNamespaceDeclHandler = rec("n")
    parser.StartCdataSectionHandler = rec("[")
    parser.EndCdataSectionHandler = rec("]")
    parser.CharacterDataHandler = on_text
    try:
        parser.Parse(data, True)
    except expat.ExpatError:
        return None

    n = len(events)
    nxt = [events[i + 1][0] if i + 1 < n else len(data) for i in range(n)]
    paragraphs: list[tuple[int, int]] = []
    errors: list[_ErrorEl] = []
    stack: list[tuple[str, int]] = []  # (local name, start event index)
    p_stack: list[int] = []
    open_errs: list[dict] = []
    for i, (idx, kind, args) in enumerate(events):
        if kind == "S":
            local = args[0].rsplit(_NS, 1)[-1]
            if local == "p":
                p_stack.append(len(paragraphs))
                paragraphs.append((idx, -1))
            if local == "error":
                declares = i > 0 and events[i - 1][1] == "N" and events[i - 1][0] == idx
                raw = data[idx : idx + 7]
                open_errs.append({
                    "start": (idx, nxt[i]), "start_index": i, "paragraph": p_stack[-1] if p_stack else None,
                    "nested": bool(open_errs), "unprefixed": raw[:6] == b"<error" and raw[6:7] in b" \t\r\n/>",
                    "declares": declares,
                })
            stack.append((local, i))
        elif kind == "E":
            local, _ = stack.pop()
            if local == "error":
                info = open_errs.pop()
                errors.append(_ErrorEl(info["start"], (idx, nxt[i]), info["start_index"], i, info["paragraph"],
                                       info["nested"], info["unprefixed"], info["declares"]))
            if local == "p":
                pid = p_stack.pop()
                paragraphs[pid] = (paragraphs[pid][0], nxt[i])
    errors.sort(key=lambda e: e.start)
    return _Recording(events, paragraphs, errors, declaration[0] if declaration else None)


def _structure(events, skip_indices: set[int]):
    """Structural stream (V5): tags with namespace URIs and attributes, comments, PIs, ns declarations."""
    out = []
    for i, (idx, kind, args) in enumerate(events):
        if kind == "T" or i in skip_indices:
            continue
        out.append((idx, kind, args))  # idx kept only to locate a difference; compared without it
    return out


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _raw_encoding_problem(original: bytes) -> str | None:
    """Reject UTF-16/UTF-32 (with or without BOM) on the raw bytes, before any parser sees them.

    A NUL byte can never occur in well-formed UTF-8 or US-ASCII XML (U+0000 is not an XML
    character), whereas every UTF-16/UTF-32 document has one: its markup contains '<'.
    """
    for bom, name in _FOREIGN_BOMS:
        if original.startswith(bom):
            return f"{name} byte-order mark; only UTF-8 and US-ASCII can be verified"
    if b"\x00" in original:
        return "NUL bytes present (UTF-16/UTF-32 or binary data); only UTF-8 and US-ASCII can be verified"
    return None


def _declaration_encoding(data: bytes) -> tuple[bool, str | None]:
    """``(well_formed, encoding)`` of the XML declaration at the start of ``data`` (no-op path).

    Cheap and independent of S1: without a declaration nothing is parsed. With one, only the
    declaration itself (up to the first ``?>``; its values cannot contain ``?``) is fed to this
    module's own expat parser, which reports ``encoding`` exactly as written (None if absent).
    """
    start = len(_UTF8_BOM) if data.startswith(_UTF8_BOM) else 0
    if not (data.startswith(b"<?xml", start) and data[start + 5 : start + 6] in _DECL_SPACE):
        return True, None  # no declaration ("<?xml-stylesheet" etc. are processing instructions)
    end = data.find(b"?>", start)
    if end < 0:
        return False, None
    parser = expat.ParserCreate(encoding="UTF-8")
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    seen: list[str | None] = []
    parser.XmlDeclHandler = lambda _version, encoding, _standalone: seen.append(encoding)
    try:
        parser.Parse(data[: end + 2], False)
    except expat.ExpatError:
        return False, None
    return (True, seen[0]) if seen else (False, None)


def _document_encoding(original: bytes, declared: str | None) -> tuple[str, str | None]:
    """The original's encoding ("utf-8" or "us-ascii") from its BOM and declared encoding, or a problem.

    Independent of S1: ``declared`` comes from this module's own expat parser.
    """
    bom = original.startswith(_UTF8_BOM)
    if declared is None:
        return "utf-8", None
    name = declared.lower()
    if name in _UTF8_NAMES:
        return "utf-8", None
    if name in _ASCII_NAMES:
        if bom:
            return "", f"UTF-8 byte-order mark contradicts declared encoding {declared!r}"
        if not original.isascii():
            return "", f"document declares {declared!r} but contains non-ASCII bytes"
        return "us-ascii", None
    return "", f"unsupported declared encoding {declared!r}; only UTF-8 and US-ASCII can be verified"


def _canonical_tags(open_tag: bytes, close_tag: bytes, encoding: str) -> bool:
    """Exact E02 bytes for the document's encoding.

    * US-ASCII: only ASCII bytes; non-ASCII characters appear as decimal references.
    * UTF-8: non-ASCII characters appear as raw UTF-8; references only for TAB, LF and CR.
    In both, references never spell ASCII characters other than TAB/LF/CR, have no leading zeros
    (one spelling per character) and are length-capped before int() is called.
    """
    if close_tag != b"</error>" or not _CANONICAL_OPEN.match(open_tag):
        return False
    ascii_doc = encoding == "us-ascii"
    if ascii_doc and not open_tag.isascii():
        return False
    for n in _CHAR_REF.findall(open_tag):
        if len(n) > 7 or n.startswith(b"0"):
            return False
        value = int(n)
        if value in _ESCAPED_CONTROLS:
            continue
        if not (ascii_doc and 128 <= value <= 0x10FFFF):
            return False
    return True


def _remove(data: bytes, ranges: list[tuple[int, int]]) -> bytes:
    kept, prev = bytearray(), 0
    for s, e in ranges:
        if s < prev:
            return b""  # overlapping ranges: cannot be a valid insertion set
        kept += data[prev:s]
        prev = e
    return bytes(kept + data[prev:])


def _first_difference(a: bytes, b: bytes) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _paragraph_at(recording: _Recording, offset: int) -> int | None:
    """Innermost paragraph of the ORIGINAL document containing ``offset`` (for issue attribution)."""
    best = None
    for pid, (start, end) in enumerate(recording.paragraphs):
        if start <= offset < end and (best is None or start >= recording.paragraphs[best][0]):
            best = pid
    return best


def _report(doc, para) -> VerificationReport:
    return VerificationReport(tuple(doc), tuple((pid, tuple(issues)) for pid, issues in sorted(para.items())))

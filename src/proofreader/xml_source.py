"""Expat source mapping: paragraphs, atoms, ownership, barriers and the XML security policy (S1).

``scan(data)`` never renders or rewrites XML. It maps every piece of ``<p>`` text to the exact
source bytes it came from, so later stages can insert markup without touching anything else.

Mapping rules (validated by the S1 Expat experiment):

* R1  An event's raw byte range is ``[CurrentByteIndex, next event's CurrentByteIndex)``.
* R2  Consecutive character-data callbacks that share one ``CurrentByteIndex`` come from a single
      entity expansion and form ONE atomic atom (Expat may split one expansion into several
      callbacks, all reported at the reference's position).
* R3  Atom kinds: CDATA (inside a CDATA section); BAD (a non-text event shares the atom's index,
      i.e. an entity expanded to markup, or the range is empty); REF (raw bytes are exactly one
      ``&...;`` reference whose expansion checks out); NL (raw ``\\r\\n``/``\\r`` normalised to
      ``\\n``); LIT (raw bytes decode exactly to the text); anything else BAD.
* R4  Skipped/external entity references and BAD atoms make the owning paragraph unmappable.
* R5  Independent self-checks per paragraph (tiling, lexical scan, ElementTree own-text) can only
      veto a paragraph, never change boundaries. If ElementTree cannot parse the document at all,
      every paragraph is DOCUMENT_UNVERIFIABLE (decision C2).

Skip-reason precedence: DOCUMENT_UNVERIFIABLE > PARA_HAS_ERROR > PARA_UNMAPPABLE > EMPTY > CDATA_ONLY.

Security (AR1): external DTDs and external entities are never fetched; parameter entities are never
parsed; Expat >= 2.4.0 (amplification protection) is required; an additional early guard limits the
amount of entity-expanded text; the input size cap and XML 1.0 policy come from ``encoding.check``.
"""

from __future__ import annotations

import bisect
import re
import xml.etree.ElementTree as ET
import xml.parsers.expat as expat
from dataclasses import dataclass, field

from proofreader import encoding as encoding_check
from proofreader.models import (
    Atom,
    AtomKind,
    ElementInfo,
    EncodingInfo,
    MalformedXMLError,
    Paragraph,
    ProofreaderError,
    ScanResult,
    SkipReason,
)

MIN_EXPAT_VERSION = (2, 4, 0)
# Early guard against entity amplification: total delivered character data may exceed the input
# size by at most this many characters. Expat's own protection (>= 2.4) still applies on top.
MAX_ENTITY_EXPANSION_CHARS = 1 << 20

_NS_SEP = " "
_PREDEFINED = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'"}
_REF_RE = re.compile(rb"&(#[0-9]+|#x[0-9A-Fa-f]+|[^\x00-\x20&;<>\"'#/=][^\x00-\x20&;<>\"'/=]*);")

# event kinds
_START, _END, _TEXT, _COMMENT, _PI, _CDATA_START, _CDATA_END, _SKIPPED, _EXTREF = range(9)


class MappingRejected(Exception):
    """A character position cannot be mapped to a safe byte offset. ``code`` names the reason."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


class _ExpansionLimitExceeded(Exception):
    pass


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------


def check_expat_version() -> None:
    """Fail unless the linked Expat library provides amplification protection (>= 2.4.0)."""
    if tuple(expat.version_info) < MIN_EXPAT_VERSION:
        found = ".".join(map(str, expat.version_info))
        required = ".".join(map(str, MIN_EXPAT_VERSION))
        raise ProofreaderError(f"Expat {required} or newer is required (found {found})")


def scan(
    data: bytes,
    *,
    max_bytes: int = encoding_check.MAX_INPUT_BYTES,
    expansion_limit: int | None = MAX_ENTITY_EXPANSION_CHARS,
) -> ScanResult:
    """Map all ``<p>`` paragraphs of ``data`` to source bytes. Raises ProofreaderError subclasses."""
    check_expat_version()
    enc = encoding_check.check(data, max_bytes=max_bytes)
    events, entity_values = _parse(data, expansion_limit)
    return _build(data, enc, events, entity_values)


def open_byte(paragraph: Paragraph, char_start: int, codec: str) -> int:
    """Byte offset at which an opening tag for a span starting at ``char_start`` would be inserted.

    The opening tag binds to the character at ``char_start``: it goes immediately before that
    character's source bytes (i.e. after any preceding markup).
    """
    atom = _atom_at(paragraph, char_start)
    k = char_start - atom.char_start
    return _offset_in_atom(atom, k, codec)


def close_byte(paragraph: Paragraph, char_end: int, codec: str) -> int:
    """Byte offset at which a closing tag for a span ending at ``char_end`` (exclusive) would go.

    The closing tag binds to the character at ``char_end - 1``: it goes immediately after that
    character's source bytes (i.e. before any following markup).
    """
    if char_end <= 0:
        raise ValueError("char_end must be > 0")
    atom = _atom_at(paragraph, char_end - 1)
    k = char_end - atom.char_start
    return _offset_in_atom(atom, k, codec)


# --------------------------------------------------------------------------------------------
# Parsing (events only)
# --------------------------------------------------------------------------------------------


def _parse(data: bytes, expansion_limit: int | None):
    # encoding="UTF-8" overrides the declaration: encoding.check() has already restricted input to
    # UTF-8 or pure US-ASCII (identical bytes), so Expat never interprets encoding aliases itself.
    parser = expat.ParserCreate(encoding="UTF-8", namespace_separator=_NS_SEP)
    parser.buffer_text = False  # True would merge chunks and break CurrentByteIndex
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    parser.UseForeignDTD(False)

    events: list[tuple[int, int, object]] = []
    entity_values: dict[str, str] = {}
    budget = None if expansion_limit is None else len(data) + expansion_limit
    delivered = 0

    def record(kind, payload=None):
        events.append((parser.CurrentByteIndex, kind, payload))

    def on_text(text):
        nonlocal delivered
        delivered += len(text)
        if budget is not None and delivered > budget:
            raise _ExpansionLimitExceeded
        record(_TEXT, text)

    def on_entity_decl(name, is_parameter, value, base, system_id, public_id, notation):
        if not is_parameter and value is not None:
            entity_values[name] = value

    def on_external_ref(context, base, system_id, public_id):
        record(_EXTREF, system_id)
        return 1  # report success WITHOUT loading anything: external resources are never fetched

    parser.StartElementHandler = lambda name, attrs: record(_START, name)
    parser.EndElementHandler = lambda name: record(_END, name)
    parser.CharacterDataHandler = on_text
    parser.CommentHandler = lambda text: record(_COMMENT)
    parser.ProcessingInstructionHandler = lambda target, text: record(_PI)
    parser.StartCdataSectionHandler = lambda: record(_CDATA_START)
    parser.EndCdataSectionHandler = lambda: record(_CDATA_END)
    parser.SkippedEntityHandler = lambda name, is_parameter: record(_SKIPPED, name)
    parser.EntityDeclHandler = on_entity_decl
    parser.ExternalEntityRefHandler = on_external_ref

    try:
        parser.Parse(data, True)
    except _ExpansionLimitExceeded:
        raise MalformedXMLError(
            "entity expansion limit exceeded (possible amplification attack)",
            byte_offset=parser.CurrentByteIndex,
            line=parser.CurrentLineNumber,
            column=parser.CurrentColumnNumber + 1,
        ) from None
    except expat.ExpatError as exc:
        raise MalformedXMLError(
            f"malformed XML: {expat.ErrorString(exc.code)} (line {exc.lineno}, column {exc.offset + 1})",
            byte_offset=parser.ErrorByteIndex,
            line=exc.lineno,
            column=exc.offset + 1,
        ) from None
    return events, entity_values


# --------------------------------------------------------------------------------------------
# Building the model
# --------------------------------------------------------------------------------------------


@dataclass
class _ElementB:
    id: int
    namespace: str | None
    local_name: str
    start_tag: tuple[int, int]
    parent_id: int | None
    start_i: int
    end_tag: tuple[int, int] = (0, 0)
    end_i: int = -1


@dataclass
class _Group:
    first_i: int
    last_i: int
    raw_start: int
    parts: list[str]
    cdata: bool
    container_id: int
    path: tuple[int, ...]


@dataclass
class _ParaB:
    id: int
    element_id: int
    groups: list[_Group] = field(default_factory=list)
    barrier_groups: list[int] = field(default_factory=list)  # number of groups before each nested <p>
    nested_p_elements: list[int] = field(default_factory=list)  # element ids of directly nested <p>
    has_error: bool = False
    unmappable: bool = False


def _split_name(name: str) -> tuple[str | None, str]:
    if _NS_SEP in name:
        uri, local = name.rsplit(_NS_SEP, 1)
        return uri, local
    return None, name


def _build(data: bytes, enc: EncodingInfo, events, entity_values) -> ScanResult:
    n = len(events)
    nxt = [events[i + 1][0] if i + 1 < n else len(data) for i in range(n)]
    # Byte indices at which some non-text event occurs (used by R3 to detect entity-produced markup).
    markup_at: set[int] = set()
    for i, (idx, kind, _) in enumerate(events):
        if kind == _TEXT or (kind == _END and _closes_empty_element_tag(data, events, i)):
            continue  # a zero-width end of a real "<x/>" tag shares its index with the next event
        markup_at.add(idx)
    document_unverifiable = any(idx < 0 for idx, _, _ in events)

    elements: list[_ElementB] = []
    paras: list[_ParaB] = []
    para_of_element: dict[int, int] = {}
    stack: list[int] = []  # element ids
    pstack: list[int] = []  # paragraph ids
    in_cdata = False
    open_group: _Group | None = None
    open_group_para: int | None = None

    for i, (idx, kind, payload) in enumerate(events):
        if kind != _TEXT:
            open_group = None
        if kind == _START:
            ns, local = _split_name(payload)  # type: ignore[arg-type]
            eid = len(elements)
            elements.append(_ElementB(eid, ns, local, (idx, nxt[i]), stack[-1] if stack else None, i))
            if local == "error":
                for pid in pstack:
                    paras[pid].has_error = True
            if local == "p":
                if pstack:
                    outer = paras[pstack[-1]]
                    outer.barrier_groups.append(len(outer.groups))
                    outer.nested_p_elements.append(eid)
                pid = len(paras)
                paras.append(_ParaB(pid, eid))
                para_of_element[eid] = pid
                pstack.append(pid)
            stack.append(eid)
        elif kind == _END:
            eid = stack.pop()
            elements[eid].end_tag = (idx, nxt[i])
            elements[eid].end_i = i
            if para_of_element.get(eid) is not None:
                pstack.pop()
        elif kind == _CDATA_START:
            in_cdata = True
        elif kind == _CDATA_END:
            in_cdata = False
        elif kind in (_SKIPPED, _EXTREF):
            if pstack:
                paras[pstack[-1]].unmappable = True
        elif kind == _TEXT and pstack:
            pid = pstack[-1]
            if (
                open_group is not None
                and open_group_para == pid
                and open_group.raw_start == idx
                and open_group.cdata == in_cdata
            ):
                open_group.parts.append(payload)  # type: ignore[arg-type]
                open_group.last_i = i
            else:
                open_group = _Group(i, i, idx, [payload], in_cdata, stack[-1], tuple(stack))  # type: ignore[list-item]
                open_group_para = pid
                paras[pid].groups.append(open_group)

    element_infos = tuple(
        ElementInfo(e.id, e.namespace, e.local_name, e.start_tag, e.end_tag, e.parent_id) for e in elements
    )

    built: list[tuple[_ParaB, list[Atom], str, tuple[int, ...]]] = []
    for pb in paras:
        atoms: list[Atom] = []
        pos = 0
        group_char_starts = []
        for g in pb.groups:
            text = "".join(g.parts)
            raw_end = nxt[g.last_i]
            kind = _classify(data, enc.codec, g, raw_end, text, markup_at, entity_values)
            if kind is AtomKind.BAD:
                pb.unmappable = True
            group_char_starts.append(pos)
            atoms.append(Atom(g.raw_start, raw_end, text, kind, pos, pos + len(text), g.container_id, g.path))
            pos += len(text)
        text = "".join(a.text for a in atoms)
        barriers = tuple(sorted({group_char_starts[k] if k < len(atoms) else pos for k in pb.barrier_groups}))
        if not pb.unmappable and not pb.has_error:
            if not _tiles(data, events, nxt, elements, pb, atoms) or not _lexically_consistent(
                data, elements, pb, atoms
            ):
                pb.unmappable = True
        built.append((pb, atoms, text, barriers))

    et_texts = None if document_unverifiable else _elementtree_own_texts(data)
    if et_texts is None or len(et_texts) != len(built):
        document_unverifiable = True

    paragraphs = []
    for k, (pb, atoms, text, barriers) in enumerate(built):
        if not document_unverifiable and et_texts[k] != text:  # type: ignore[index]
            pb.unmappable = True
        if document_unverifiable:
            reason: SkipReason | None = SkipReason.DOCUMENT_UNVERIFIABLE
        elif pb.has_error:
            reason = SkipReason.PARA_HAS_ERROR
        elif pb.unmappable:
            reason = SkipReason.PARA_UNMAPPABLE
        elif not text.strip():
            reason = SkipReason.EMPTY
        elif _cdata_only(atoms):
            reason = SkipReason.CDATA_ONLY
        else:
            reason = None
        paragraphs.append(Paragraph(pb.id, pb.element_id, tuple(atoms), text, barriers, reason))

    return ScanResult(enc, element_infos, tuple(paragraphs))


def _closes_empty_element_tag(data: bytes, events, i: int) -> bool:
    """True if END event ``i`` closes an empty-element tag written in the source (``<x/>``).

    Expat reports such an END with zero width at the position after the tag, i.e. at the same
    index as whatever follows. Markup produced by an entity expansion has a zero-width START
    as well, so it is never matched here and still marks the shared index as unsafe.
    """
    if i == 0 or events[i - 1][1] != _START:
        return False
    start_idx, end_idx = events[i - 1][0], events[i][0]
    return start_idx < end_idx and data[start_idx:end_idx].endswith(b"/>")


def _cdata_only(atoms: list[Atom]) -> bool:
    """True if all non-whitespace text is inside CDATA sections (nothing annotatable, X5)."""
    has_cdata_text = any(a.kind is AtomKind.CDATA and a.text.strip() for a in atoms)
    return has_cdata_text and all(a.kind is AtomKind.CDATA or not a.text.strip() for a in atoms)


def _classify(data, codec, g: _Group, raw_end, text, markup_at, entity_values) -> AtomKind:
    raw = data[g.raw_start:raw_end]
    if g.cdata:
        return AtomKind.CDATA
    if raw_end <= g.raw_start or g.raw_start in markup_at:
        return AtomKind.BAD  # zero-width, or an entity expansion that produced markup
    m = _REF_RE.fullmatch(raw)
    if m is not None:
        expected = _expected_expansion(m.group(1).decode(codec, "replace"), entity_values)
        if expected is not None and expected != text:
            return AtomKind.BAD
        return AtomKind.REF
    if raw in (b"\r\n", b"\r"):
        return AtomKind.NL if text == "\n" else AtomKind.BAD
    try:
        return AtomKind.LIT if raw.decode(codec) == text else AtomKind.BAD
    except UnicodeDecodeError:
        return AtomKind.BAD


def _expected_expansion(name: str, entity_values: dict[str, str]) -> str | None:
    try:
        if name.startswith("#x"):
            return chr(int(name[2:], 16))
        if name.startswith("#"):
            return chr(int(name[1:]))
    except (ValueError, OverflowError):
        return "\x00invalid"  # never equals real text -> BAD
    if name in _PREDEFINED:
        return _PREDEFINED[name]
    value = entity_values.get(name)
    if value is not None and not any(c in value for c in "&<\r\n"):
        return value  # plain-text internal entity: expansion must equal the declared value
    return None  # nested/complex internal entity: accepted on the lexical + own-text checks


# --------------------------------------------------------------------------------------------
# Self-checks (veto only)
# --------------------------------------------------------------------------------------------


def _content_region(elements, pb: _ParaB) -> tuple[int, int]:
    el = elements[pb.element_id]
    return el.start_tag[1], el.end_tag[0]


def _nested_p_blocks(elements, pb: _ParaB) -> list[tuple[int, int, int, int]]:
    """(raw_start, raw_end, start_i, end_i) of each <p> directly nested in this paragraph.

    "Directly" = this paragraph is the nearest enclosing <p>; deeper paragraphs lie inside these blocks.
    """
    return [
        (e.start_tag[0], e.end_tag[1], e.start_i, e.end_i)
        for e in (elements[eid] for eid in pb.nested_p_elements)
    ]


def _tiles(data, events, nxt, elements, pb: _ParaB, atoms: list[Atom]) -> bool:
    """Atoms + markup events + nested <p> blocks must cover the content region exactly."""
    start, end = _content_region(elements, pb)
    el = elements[pb.element_id]
    nested = {b[2]: b for b in _nested_p_blocks(elements, pb)}
    ranges = [(a.raw_start, a.raw_end) for a in atoms]
    i = el.start_i + 1
    while i < el.end_i:
        if i in nested:
            s, e, _, end_i = nested[i]
            ranges.append((s, e))
            i = end_i + 1
            continue
        idx, kind, _ = events[i]
        if kind != _TEXT and nxt[i] > idx:
            ranges.append((idx, nxt[i]))
        i += 1
    pos = start
    for s, e in sorted(ranges):
        if s != pos:
            return False
        pos = e
    return pos == end


def _lex(data: bytes, start: int, end: int):
    """Independent byte-level tokenisation of paragraph content (no nested <p> inside the range)."""
    toks = []
    i = start
    while i < end:
        if data.startswith(b"<!--", i, end):
            j = data.find(b"-->", i + 4, end)
            if j < 0:
                return None
            toks.append(("markup", i, j + 3))
            i = j + 3
        elif data.startswith(b"<![CDATA[", i, end):
            j = data.find(b"]]>", i + 9, end)
            if j < 0:
                return None
            toks += [("markup", i, i + 9), ("cdata", i + 9, j), ("markup", j, j + 3)]
            i = j + 3
        elif data.startswith(b"<?", i, end):
            j = data.find(b"?>", i + 2, end)
            if j < 0:
                return None
            toks.append(("markup", i, j + 2))
            i = j + 2
        elif data[i] == 0x3C:  # '<' tag, honouring quoted attribute values
            j, quote = i + 1, None
            while j < end:
                c = data[j]
                if quote is not None:
                    if c == quote:
                        quote = None
                elif c in (0x22, 0x27):
                    quote = c
                elif c == 0x3E:
                    break
                j += 1
            if j >= end:
                return None
            toks.append(("markup", i, j + 1))
            i = j + 1
        elif data[i] == 0x26:  # '&'
            m = _REF_RE.match(data, i, end)
            if m is None:
                return None
            toks.append(("ref", i, m.end()))
            i = m.end()
        elif data.startswith(b"\r\n", i, end):
            toks.append(("nl", i, i + 2))
            i += 2
        elif data[i] == 0x0D:
            toks.append(("nl", i, i + 1))
            i += 1
        else:
            j = i
            while j < end and data[j] not in (0x3C, 0x26, 0x0D):
                j += 1
            toks.append(("text", i, j))
            i = j
    return toks


def _lexically_consistent(data, elements, pb: _ParaB, atoms: list[Atom]) -> bool:
    start, end = _content_region(elements, pb)
    toks = []
    pos = start
    for s, e, _, _ in _nested_p_blocks(elements, pb):
        part = _lex(data, pos, s)
        if part is None:
            return False
        toks += part
        pos = e
    part = _lex(data, pos, end)
    if part is None:
        return False
    toks += part

    refs = {(s, e) for k, s, e in toks if k == "ref"}
    nls = {(s, e) for k, s, e in toks if k == "nl"}

    def inside(kind, a):
        return any(k == kind and s <= a.raw_start and a.raw_end <= e for k, s, e in toks)

    for a in atoms:
        if a.kind is AtomKind.REF and (a.raw_start, a.raw_end) not in refs:
            return False
        if a.kind is AtomKind.NL and (a.raw_start, a.raw_end) not in nls:
            return False
        if a.kind is AtomKind.LIT and not inside("text", a):
            return False
        if a.kind is AtomKind.CDATA and not inside("cdata", a):
            return False
    atom_ranges = {(a.raw_start, a.raw_end) for a in atoms}
    return refs <= atom_ranges  # every reference in the source produced exactly one text atom


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _own_text(el: ET.Element) -> str:
    """Recursive own text in document order, excluding nested <p> (iterative, no recursion limit)."""
    parts = [el.text or ""]
    frames: list[tuple[object, str | None]] = [(iter(list(el)), None)]
    while frames:
        it, _ = frames[-1]
        child = next(it, None)  # type: ignore[call-overload]
        if child is None:
            _, tail = frames.pop()
            if tail is not None:
                parts.append(tail)
            continue
        if _local(child.tag) == "p":
            parts.append(child.tail or "")
        else:
            parts.append(child.text or "")
            frames.append((iter(list(child)), child.tail or ""))
    return "".join(parts)


def _elementtree_own_texts(data: bytes) -> list[str] | None:
    """Own texts of all <p> in document order via ElementTree (a separate parser front end)."""
    try:
        root = ET.fromstring(data, parser=ET.XMLParser(encoding="utf-8"))
    except ET.ParseError:
        return None
    return [_own_text(el) for el in root.iter() if isinstance(el.tag, str) and _local(el.tag) == "p"]


# --------------------------------------------------------------------------------------------
# Char -> byte mapping (02 section 3)
# --------------------------------------------------------------------------------------------


def _atom_at(paragraph: Paragraph, char_index: int) -> Atom:
    atoms = paragraph.atoms
    if not atoms or char_index < 0 or char_index >= atoms[-1].char_end:
        raise ValueError(f"character index {char_index} outside paragraph text")
    k = bisect.bisect_right([a.char_start for a in atoms], char_index) - 1
    return atoms[k]


def _offset_in_atom(atom: Atom, k: int, codec: str) -> int:
    if atom.kind is AtomKind.LIT:
        return atom.raw_start + len(atom.text[:k].encode(codec))
    if atom.kind is AtomKind.CDATA:
        raise MappingRejected("IN_CDATA", "endpoint inside a CDATA section")
    if atom.kind is AtomKind.BAD:
        raise MappingRejected("PARA_UNMAPPABLE", "endpoint in unmappable content")
    if k == 0:
        return atom.raw_start
    if k == len(atom.text):
        return atom.raw_end
    raise MappingRejected("SPLITS_ATOMIC", f"endpoint inside atomic {atom.kind.value} atom")

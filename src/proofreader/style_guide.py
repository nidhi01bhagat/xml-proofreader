"""DOCX style guide -> sections -> verbatim text for the system prompt (S4.2a; standard library only).

Pipeline: ``load(path)`` / ``parse_docx(data)`` -> :class:`StyleGuide` (sections with exact
SHA-256 fingerprints) -> ``select(guide, scope, manifest)`` -> :class:`SelectedGuide` (text sent
to the model). Approved decisions: docs/04, rows "S4 locale rules" and "S4.2".

Reading (fail closed, every failure -> :class:`StyleGuideError`, exit 1):
  * The file, every zip part and the total are capped on the bytes ACTUALLY read / decompressed
    (zip metadata is not trusted); encrypted or duplicate (case-insensitive) entries are rejected.
  * The main document and the styles part are found through the package relationships.
  * XML is parsed with Expat (version checked by S1's ``check_expat_version``); any DOCTYPE or
    entity declaration is rejected. Only transitional WordprocessingML is supported.

Extraction ("as shown", wording exact):
  * Run text is concatenated verbatim (run splitting never matters). ``w:tab``/``w:ptab`` -> TAB,
    ``w:br``/``w:cr`` -> LF (page / column breaks ignored), ``w:noBreakHyphen`` -> U+2011,
    ``w:softHyphen`` -> U+00AD, ``w:sym`` -> its character. Hyperlinks, content controls, smart
    tags and custom XML contribute their text; fields contribute their displayed result only
    (``w:instrText`` outside a field's instruction region is malformed and rejected).
    Tables: one line per row, cells joined by " | "; a cell containing a literal "|" is rejected
    (reported by table / row / cell number), so the separator is always unambiguous.
  * Proofing marks (``w:proofErr``), bookmarks, comment anchors and note references carry no text
    and are ignored. Headers, footers, footnotes, endnotes and comments are not part of the guide.
  * Unsupported -> rejected: tracked changes (insertions, deletions, moves, property changes),
    hidden text (a run with text that ANY level marks hidden - direct ``vanish``/``specVanish``,
    character or paragraph style chain, document defaults; an explicit false never un-hides,
    because ``vanish`` is a toggle property; a table style that hides text rejects documents with
    tables), text boxes or drawings containing text, embedded objects, imported content
    (``altChunk``, ``subDoc``), symbol-font private-use characters, alternate content with text,
    nested tables, generated-text elements (page numbers, dates) and any other element that holds
    text the extractor does not understand.
  * Empty / whitespace-only paragraphs are dropped; formatting (bold, italic, ...) is not kept.

Sections and fingerprints:
  * Every heading (outline level 0-8, from the paragraph or its style chain) starts a section;
    paragraphs before the first heading form the preamble section (level None).
  * ``sha256`` of UTF-8 ``json.dumps(["proofreader-section-v1", path, paragraphs])`` where ``path``
    lists the ancestor headings as ``[level, text]`` and ``paragraphs`` lists ``[kind, level,
    text]``. No Unicode or whitespace normalisation: any wording change changes the hash.

Limits use the same character measure everywhere: Unicode code points (``len`` of ``str``).
Error messages name parts, elements and limits only, never guide wording.
"""

from __future__ import annotations

import hashlib
import io
import json
import posixpath
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from xml.parsers import expat

from .config import LocaleScope
from .models import ProofreaderError
from .xml_source import check_expat_version

# --------------------------------------------------------------------------------------------
# Limits (approved S4.2 decision 6). Bytes are counted as read / decompressed.
# --------------------------------------------------------------------------------------------

MAX_DOCX_BYTES = 10 * 1024 * 1024
MAX_ZIP_ENTRIES = 1000
MAX_PART_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 50 * 1024 * 1024
MAX_GUIDE_CHARS = 200_000  # Unicode code points of the extracted text
MAX_SECTIONS = 2000
_CHUNK = 64 * 1024
_MAX_STYLE_DEPTH = 64

FINGERPRINT_VERSION = "proofreader-section-v1"
MANIFEST_VERSION = 1
MANIFEST_RESOURCE = "neutral_rules.json"
SCOPE_EN_GB = "en-GB"
SCOPE_CROSS_LOCALE = "cross-locale"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W_STRICT = "http://purl.oclc.org/ooxml/wordprocessingml/main"
PKG_RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
REL_OFFICE_DOCUMENT = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
REL_OFFICE_DOCUMENT_STRICT = "http://purl.oclc.org/ooxml/officeDocument/relationships/officeDocument"
REL_STYLES = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"
_MC_ALTERNATE = "{http://schemas.openxmlformats.org/markup-compatibility/2006}AlternateContent"


def _w(local: str) -> str:
    return f"{{{W}}}{local}"


# Revision (tracked-change) markup. Proofing marks (w:proofErr) are NOT revisions.
_TRACKED = frozenset(_w(n) for n in (
    "ins", "del", "moveFrom", "moveTo", "moveFromRangeStart", "moveFromRangeEnd", "moveToRangeStart",
    "moveToRangeEnd", "delText", "delInstrText", "cellIns", "cellDel", "cellMerge", "rPrChange", "pPrChange",
    "sectPrChange", "tblPrChange", "tblPrExChange", "tblGridChange", "trPrChange", "tcPrChange", "numberingChange",
    "customXmlInsRangeStart", "customXmlInsRangeEnd", "customXmlDelRangeStart", "customXmlDelRangeEnd",
    "customXmlMoveFromRangeStart", "customXmlMoveFromRangeEnd", "customXmlMoveToRangeStart",
    "customXmlMoveToRangeEnd",
))
# Elements without text that are safely ignored (proofing marks, anchors, references, layout).
_IGNORED = frozenset(_w(n) for n in (
    "proofErr", "bookmarkStart", "bookmarkEnd", "permStart", "permEnd", "commentRangeStart", "commentRangeEnd",
    "commentReference", "footnoteReference", "endnoteReference", "annotationRef", "footnoteRef", "endnoteRef",
    "separator", "continuationSeparator", "lastRenderedPageBreak", "rPr", "pPr", "sectPr", "sdtPr", "sdtEndPr",
    "customXmlPr", "smartTagPr", "tblPr", "tblGrid", "trPr", "tcPr", "tblPrEx",
))
# Elements whose displayed text is generated (no text nodes): unsupported.
_GENERATED = frozenset(_w(n) for n in (
    "pgNum", "dayShort", "dayLong", "monthShort", "monthLong", "yearShort", "yearLong",
))
_INLINE_CONTAINERS = frozenset(_w(n) for n in ("hyperlink", "smartTag", "customXml", "fldSimple", "dir", "bdo"))
_TEXT_LOCAL_NAMES = frozenset({"t", "delText", "instrText", "delInstrText"})
_FALSE_VALUES = frozenset({"false", "0", "off"})


# --------------------------------------------------------------------------------------------
# Errors and data types
# --------------------------------------------------------------------------------------------


class StyleGuideError(ProofreaderError):
    """Unreadable, unsafe or unsupported style guide (exit 1). Never quotes guide wording."""

    exit_code = 1


class ManifestError(ProofreaderError):
    """The packaged neutral-rules manifest is invalid (an internal error, exit 1)."""

    exit_code = 1


@dataclass(frozen=True)
class GuideParagraph:
    kind: str  # "heading" | "paragraph" | "list" | "table"
    level: int | None  # heading outline level 0-8, else None
    text: str


@dataclass(frozen=True)
class Section:
    index: int
    level: int | None  # None for the preamble
    path: tuple[tuple[int, str], ...]  # ancestor headings (level, text), outermost first
    paragraphs: tuple[GuideParagraph, ...]  # the first one is the heading (except in the preamble)
    sha256: str

    @property
    def heading(self) -> str | None:
        return self.paragraphs[0].text if self.level is not None else None


@dataclass(frozen=True)
class StyleGuide:
    sections: tuple[Section, ...]
    total_chars: int  # code points of all extracted paragraph text


@dataclass(frozen=True)
class Manifest:
    en_gb: frozenset[str]
    cross_locale: frozenset[str]
    labels: Mapping[str, str]


@dataclass(frozen=True)
class SelectedGuide:
    text: str
    scope: LocaleScope
    included: int
    total: int
    included_hashes: tuple[str, ...]


# --------------------------------------------------------------------------------------------
# Zip / package reading (actual bytes are counted)
# --------------------------------------------------------------------------------------------


def load(path: str | Path) -> StyleGuide:
    """Read and parse a DOCX style guide. Raises StyleGuideError (exit 1)."""
    path = Path(path)
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_DOCX_BYTES + 1)
    except OSError:
        raise StyleGuideError(f"style guide {path.name!r} cannot be read") from None
    if len(data) > MAX_DOCX_BYTES:
        raise StyleGuideError(f"style guide is larger than {MAX_DOCX_BYTES} bytes")
    return parse_docx(data)


class _Package:
    def __init__(self, data: bytes) -> None:
        if len(data) > MAX_DOCX_BYTES:
            raise StyleGuideError(f"style guide is larger than {MAX_DOCX_BYTES} bytes")
        try:
            self._zip = zipfile.ZipFile(io.BytesIO(data))
            infos = self._zip.infolist()
        except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError, EOFError, NotImplementedError):
            raise StyleGuideError("style guide is not a readable .docx (zip) file") from None
        if len(infos) > MAX_ZIP_ENTRIES:
            raise StyleGuideError(f"style guide has more than {MAX_ZIP_ENTRIES} zip entries")
        self._by_name: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            if info.flag_bits & 0x1:
                raise StyleGuideError("style guide contains encrypted zip entries")
            key = info.filename.casefold()
            if key in self._by_name:
                raise StyleGuideError("style guide contains duplicate zip entries")
            self._by_name[key] = info
        self._total = 0

    def has(self, name: str) -> bool:
        return name.casefold() in self._by_name

    def read(self, name: str) -> bytes:
        info = self._by_name.get(name.casefold())
        if info is None:
            raise StyleGuideError(f"style guide is missing the part {name!r}")
        chunks: list[bytes] = []
        size = 0
        try:
            with self._zip.open(info) as handle:
                while True:
                    chunk = handle.read(_CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    self._total += len(chunk)
                    if size > MAX_PART_BYTES:
                        raise StyleGuideError(f"style guide part {name!r} is larger than {MAX_PART_BYTES} bytes")
                    if self._total > MAX_TOTAL_BYTES:
                        raise StyleGuideError(f"style guide parts exceed {MAX_TOTAL_BYTES} bytes in total")
                    chunks.append(chunk)
        except StyleGuideError:
            raise
        except (zipfile.BadZipFile, OSError, ValueError, EOFError, NotImplementedError, RuntimeError, zlib.error):
            raise StyleGuideError(f"style guide part {name!r} is corrupt or unsupported") from None
        return b"".join(chunks)


def _relationships(package: _Package, rels_name: str, source_dir: str) -> list[tuple[str, str]]:
    """(type, resolved part name) for internal relationships of a .rels part (missing part -> [])."""
    if not package.has(rels_name):
        return []
    root = _parse_xml(package.read(rels_name), rels_name)
    out = []
    for rel in root:
        if rel.tag != f"{{{PKG_RELS}}}Relationship":
            continue
        rel_type, target = rel.get("Type", ""), rel.get("Target", "")
        if rel.get("TargetMode", "Internal") != "Internal":
            if rel_type in (REL_OFFICE_DOCUMENT, REL_STYLES):
                raise StyleGuideError("style guide references an external main document or styles part")
            continue
        out.append((rel_type, _resolve(source_dir, target)))
    return out


def _resolve(source_dir: str, target: str) -> str:
    if not target or "\\" in target or "\x00" in target:
        raise StyleGuideError("style guide has an invalid relationship target")
    joined = target.lstrip("/") if target.startswith("/") else posixpath.join(source_dir, target)
    normal = posixpath.normpath(joined)
    if normal.startswith("../") or normal in ("..", ".", "") or normal.startswith("/"):
        raise StyleGuideError("style guide has a relationship target outside the package")
    return normal


# --------------------------------------------------------------------------------------------
# XML (Expat; no DTDs or entities)
# --------------------------------------------------------------------------------------------


class _Refused(Exception):
    pass


def _parse_xml(data: bytes, part: str) -> ET.Element:
    check_expat_version()
    parser = expat.ParserCreate(namespace_separator="}")
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    builder = ET.TreeBuilder()

    def refuse(*_args) -> None:
        raise _Refused()

    def name(raw: str) -> str:
        return "{" + raw if "}" in raw else raw

    parser.StartDoctypeDeclHandler = refuse
    parser.EntityDeclHandler = refuse
    parser.UnparsedEntityDeclHandler = refuse
    parser.NotationDeclHandler = refuse
    parser.ExternalEntityRefHandler = refuse
    parser.StartElementHandler = lambda tag, attrs: builder.start(name(tag), {name(k): v for k, v in attrs.items()})
    parser.EndElementHandler = lambda tag: builder.end(name(tag))
    parser.CharacterDataHandler = builder.data
    try:
        parser.Parse(data, True)
        return builder.close()
    except _Refused:
        raise StyleGuideError(f"style guide part {part!r} contains a DTD or entity declaration") from None
    except (expat.ExpatError, ValueError, TypeError):
        raise StyleGuideError(f"style guide part {part!r} is not well-formed XML") from None


def _on(element: ET.Element | None) -> bool:
    """OOXML on/off property: present and not explicitly false."""
    return element is not None and element.get(_w("val"), "true").lower() not in _FALSE_VALUES


def _has_text(element: ET.Element) -> bool:
    for node in element.iter():
        local = node.tag.rsplit("}", 1)[-1] if isinstance(node.tag, str) else ""
        if local in _TEXT_LOCAL_NAMES or (node.text and node.text.strip()):
            return True
    return False


# --------------------------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Style:
    based_on: str | None
    outline: int | None
    num_id: str | None
    vanish: bool


class _Styles:
    def __init__(self, root: ET.Element | None) -> None:
        self._styles: dict[str, _Style] = {}
        self.default_paragraph: str | None = None
        self.table_style_hides = False  # some table style (incl. conditional formats) hides text
        if root is None:
            return
        if root.tag != _w("styles"):
            raise StyleGuideError("style guide styles part has an unexpected root element")
        rpr_default = root.find(f"{_w('docDefaults')}/{_w('rPrDefault')}/{_w('rPr')}")
        if rpr_default is not None and (_on(rpr_default.find(_w("vanish"))) or _on(rpr_default.find(_w("specVanish")))):
            raise StyleGuideError("style guide hides all text by default (hidden text is not supported)")
        for style in root.iter(_w("style")):
            sid = style.get(_w("styleId"))
            if not sid:
                continue
            based = style.find(_w("basedOn"))
            ppr = style.find(_w("pPr"))
            rpr = style.find(_w("rPr"))
            outline = ppr.find(_w("outlineLvl")) if ppr is not None else None
            num = ppr.find(f"{_w('numPr')}/{_w('numId')}") if ppr is not None else None
            self._styles[sid] = _Style(
                based_on=based.get(_w("val")) if based is not None else None,
                outline=_outline_value(outline),
                num_id=num.get(_w("val")) if num is not None else None,
                vanish=rpr is not None and (_on(rpr.find(_w("vanish"))) or _on(rpr.find(_w("specVanish")))),
            )
            if style.get(_w("type"), "paragraph") == "paragraph" and _on_attr(style.get(_w("default"))):
                self.default_paragraph = sid
            if style.get(_w("type")) == "table" and any(
                _on(e) for name in ("vanish", "specVanish") for e in style.iter(_w(name))
            ):
                self.table_style_hides = True

    def chain(self, sid: str | None) -> Iterator[_Style]:
        seen: set[str] = set()
        while sid is not None and sid in self._styles:
            if sid in seen or len(seen) >= _MAX_STYLE_DEPTH:
                raise StyleGuideError("style guide styles have a basedOn cycle or are nested too deeply")
            seen.add(sid)
            style = self._styles[sid]
            yield style
            sid = style.based_on

    def outline(self, sid: str | None) -> int | None:
        return next((s.outline for s in self.chain(sid) if s.outline is not None), None)

    def numbered(self, sid: str | None) -> bool:
        num = next((s.num_id for s in self.chain(sid) if s.num_id is not None), None)
        return num is not None and num != "0"

    def hidden(self, sid: str | None) -> bool:
        return any(s.vanish for s in self.chain(sid))


def _on_attr(value: str | None) -> bool:
    return value is not None and value.lower() not in _FALSE_VALUES


def _outline_value(element: ET.Element | None) -> int | None:
    if element is None:
        return None
    raw = element.get(_w("val"), "")
    if not raw.isdigit():
        raise StyleGuideError("style guide has an invalid outline level")
    return int(raw)


# --------------------------------------------------------------------------------------------
# Document extraction
# --------------------------------------------------------------------------------------------


class _Extractor:
    def __init__(self, styles: _Styles) -> None:
        self.styles = styles
        self.fields: list[str] = []  # complex-field states: "instr" or "result"
        self.paragraphs: list[GuideParagraph] = []
        self.chars = 0
        self.tables = 0  # tables seen so far (1-based numbers in error messages)

    # block level ------------------------------------------------------------------------------

    def body(self, body: ET.Element) -> None:
        for element in body:
            self.block(element)
        if self.fields:
            raise StyleGuideError("style guide has an unterminated field")

    def block(self, element: ET.Element) -> None:
        tag = element.tag
        if tag == _w("p"):
            self.paragraph(element)
        elif tag == _w("tbl"):
            self.table(element)
        elif tag == _w("sdt"):
            for child in element.findall(_w("sdtContent")):
                for inner in child:
                    self.block(inner)
        elif tag == _w("customXml"):
            for inner in element:
                if inner.tag != _w("customXmlPr"):
                    self.block(inner)
        else:
            self.other(element)

    def table(self, table: ET.Element) -> None:
        if self.styles.table_style_hides:
            raise StyleGuideError("style guide has a table style that hides text (hidden text is not supported)")
        self.tables += 1
        for row_number, row in enumerate(self._unwrap(table, _w("tr")), 1):
            cells = []
            for cell_number, cell in enumerate(self._unwrap(row, _w("tc")), 1):
                texts = []
                for element in self._cell_blocks(cell):
                    if element.tag == _w("tbl"):
                        raise StyleGuideError("style guide has a nested table (not supported)")
                    if element.tag == _w("p"):
                        text = self.paragraph_text(element)
                        if text.strip():
                            texts.append(text)
                    else:
                        self.other(element)
                cell_text = "\n".join(texts)
                if "|" in cell_text:  # approved S4.2a decision (option A): " | " must stay an unambiguous separator
                    raise StyleGuideError(
                        f"style guide table {self.tables}, row {row_number}, cell {cell_number} contains a literal '|' "
                        "(not supported: it would make the table row ambiguous)"
                    )
                cells.append(cell_text)
            if any(cell.strip() for cell in cells):  # a row with no text in any cell is dropped
                self._emit(GuideParagraph("table", None, " | ".join(cells)))

    def _unwrap(self, parent: ET.Element, wanted: str) -> Iterator[ET.Element]:
        for child in parent:
            if child.tag == wanted:
                yield child
            elif child.tag == _w("sdt"):
                for content in child.findall(_w("sdtContent")):
                    yield from self._unwrap(content, wanted)
            elif child.tag == _w("customXml"):
                yield from self._unwrap(child, wanted)
            else:
                self.other(child)

    def _cell_blocks(self, cell: ET.Element) -> Iterator[ET.Element]:
        for child in cell:
            if child.tag == _w("sdt"):
                for content in child.findall(_w("sdtContent")):
                    yield from self._cell_blocks(content)
            elif child.tag == _w("customXml"):
                yield from (c for c in self._cell_blocks(child))
            else:
                yield child

    def other(self, element: ET.Element) -> None:
        tag = element.tag
        if tag in _TRACKED:
            raise StyleGuideError("style guide contains tracked changes; accept or reject all changes first")
        if tag in _IGNORED:
            return
        if tag in _GENERATED:
            raise StyleGuideError(f"style guide uses an unsupported generated-text element ({_local(tag)})")
        if tag == _w("object"):
            raise StyleGuideError("style guide contains an embedded object (not supported)")
        if tag in (_w("altChunk"), _w("subDoc")):
            raise StyleGuideError(f"style guide contains imported content ({_local(tag)}; not supported)")
        if _has_text(element):
            raise StyleGuideError(f"style guide contains text in an unsupported element ({_local(tag)})")

    # paragraph level -------------------------------------------------------------------------

    def paragraph(self, p: ET.Element) -> None:
        ppr = p.find(_w("pPr"))
        sid = None
        direct_outline = None
        direct_num = None
        if ppr is not None:
            style = ppr.find(_w("pStyle"))
            sid = style.get(_w("val")) if style is not None else None
            direct_outline = _outline_value(ppr.find(_w("outlineLvl")))
            num = ppr.find(f"{_w('numPr')}/{_w('numId')}")
            direct_num = num.get(_w("val")) if num is not None else None
        sid = sid if sid is not None else self.styles.default_paragraph
        text = self.paragraph_text(p, sid)
        outline = direct_outline if direct_outline is not None else self.styles.outline(sid)
        if outline is not None and 0 <= outline <= 8:
            kind, level = "heading", outline
        elif (direct_num is not None and direct_num != "0") or (direct_num is None and self.styles.numbered(sid)):
            kind, level = "list", None
        else:
            kind, level = "paragraph", None
        self._emit(GuideParagraph(kind, level, text))

    def paragraph_text(self, p: ET.Element, sid: str | None = None) -> str:
        if sid is None:
            ppr = p.find(_w("pPr"))
            style = ppr.find(_w("pStyle")) if ppr is not None else None
            sid = style.get(_w("val")) if style is not None else self.styles.default_paragraph
        parts: list[str] = []
        for child in p:
            self.inline(child, sid, parts)
        return "".join(parts)

    def inline(self, element: ET.Element, sid: str | None, parts: list[str]) -> None:
        tag = element.tag
        if tag == _w("r"):
            self.run(element, sid, parts)
        elif tag in _INLINE_CONTAINERS:
            for child in element:
                self.inline(child, sid, parts)
        elif tag == _w("sdt"):
            for content in element.findall(_w("sdtContent")):
                for child in content:
                    self.inline(child, sid, parts)
        else:
            self.other(element)

    def _run_hidden(self, rpr: ET.Element | None, sid: str | None) -> bool:
        """Fail closed: hidden if ANY level marks it hidden - direct ``vanish``/``specVanish``, the
        character style chain or the paragraph style chain. ``vanish`` is a toggle property whose
        combination across levels is not modelled, so an explicit false never un-hides."""
        if rpr is not None:
            if _on(rpr.find(_w("vanish"))) or _on(rpr.find(_w("specVanish"))):
                return True
            rstyle = rpr.find(_w("rStyle"))
            if rstyle is not None and self.styles.hidden(rstyle.get(_w("val"))):
                return True
        return self.styles.hidden(sid)

    def _displaying(self) -> bool:
        return all(state == "result" for state in self.fields)

    def run(self, run: ET.Element, sid: str | None, parts: list[str]) -> None:
        hidden = self._run_hidden(run.find(_w("rPr")), sid)
        produced: list[str] = []
        for child in run:
            tag = child.tag
            piece: str | None = None
            if tag == _w("t"):
                piece = child.text or ""
            elif tag in (_w("tab"), _w("ptab")):
                piece = "\t"
            elif tag == _w("br"):
                if child.get(_w("type"), "textWrapping") not in ("page", "column"):
                    piece = "\n"
            elif tag == _w("cr"):
                piece = "\n"
            elif tag == _w("noBreakHyphen"):
                piece = "\u2011"
            elif tag == _w("softHyphen"):
                piece = "\u00ad"
            elif tag == _w("sym"):
                piece = _symbol(child)
            elif tag == _w("fldChar"):
                self.field_char(child)
            elif tag == _w("instrText"):
                # Field code, never displayed. Valid only inside the instruction region of the innermost
                # open complex field; anywhere else the markup is malformed -> fail closed.
                if not self.fields or self.fields[-1] != "instr":
                    raise StyleGuideError("style guide has field code text outside a field instruction")
                continue
            elif tag in (_w("drawing"), _w("pict"), _MC_ALTERNATE):
                if _has_text(child):
                    raise StyleGuideError("style guide contains a text box or drawing with text (not supported)")
            else:
                self.other(child)
            if piece is not None and self._displaying():
                produced.append(piece)
        text = "".join(produced)
        if text and hidden:
            raise StyleGuideError("style guide contains hidden text (not supported)")
        parts.append(text)

    def field_char(self, element: ET.Element) -> None:
        kind = element.get(_w("fldCharType"))
        if kind == "begin":
            self.fields.append("instr")
        elif kind == "separate":
            if not self.fields or self.fields[-1] != "instr":
                raise StyleGuideError("style guide has a malformed field")
            self.fields[-1] = "result"
        elif kind == "end":
            if not self.fields:
                raise StyleGuideError("style guide has a malformed field")
            self.fields.pop()
        else:
            raise StyleGuideError("style guide has a malformed field")

    def _emit(self, paragraph: GuideParagraph) -> None:
        if not paragraph.text.strip():
            return
        self.chars += len(paragraph.text)
        if self.chars > MAX_GUIDE_CHARS:
            raise StyleGuideError(f"style guide text is longer than {MAX_GUIDE_CHARS} characters")
        self.paragraphs.append(paragraph)


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else "?"


def _symbol(element: ET.Element) -> str:
    raw = element.get(_w("char"), "")
    if not re.fullmatch(r"[0-9A-Fa-f]{1,6}", raw):
        raise StyleGuideError("style guide has an invalid symbol character")
    code = int(raw, 16)
    if 0xE000 <= code <= 0xF8FF or code > 0x10FFFF or 0xD800 <= code <= 0xDFFF:
        raise StyleGuideError("style guide uses a symbol-font private-use character (not supported)")
    ch = chr(code)
    if unicodedata.category(ch) in ("Cc", "Cs", "Co", "Cn"):
        raise StyleGuideError("style guide uses an unsupported symbol character")
    return ch


# --------------------------------------------------------------------------------------------
# Sections and fingerprints
# --------------------------------------------------------------------------------------------


def fingerprint(path: Iterable[tuple[int, str]], paragraphs: Iterable[GuideParagraph]) -> str:
    """Exact SHA-256 of a section (no normalisation; formatting not included)."""
    payload = json.dumps(
        [FINGERPRINT_VERSION, [[level, text] for level, text in path],
         [[p.kind, p.level, p.text] for p in paragraphs]],
        ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_sections(paragraphs: Iterable[GuideParagraph]) -> tuple[Section, ...]:
    groups: list[tuple[int | None, tuple[tuple[int, str], ...], list[GuideParagraph]]] = []
    stack: list[tuple[int, str]] = []
    for paragraph in paragraphs:
        if paragraph.kind == "heading":
            assert paragraph.level is not None
            while stack and stack[-1][0] >= paragraph.level:
                stack.pop()
            groups.append((paragraph.level, tuple(stack), [paragraph]))
            stack.append((paragraph.level, paragraph.text))
        else:
            if not groups:
                groups.append((None, (), []))
            groups[-1][2].append(paragraph)
    if len(groups) > MAX_SECTIONS:
        raise StyleGuideError(f"style guide has more than {MAX_SECTIONS} sections")
    return tuple(
        Section(i, level, path, tuple(paras), fingerprint(path, paras)) for i, (level, path, paras) in enumerate(groups)
    )


def parse_docx(data: bytes) -> StyleGuide:
    package = _Package(data)
    package_rels = _relationships(package, "_rels/.rels", "")
    if any(rel_type == REL_OFFICE_DOCUMENT_STRICT for rel_type, _ in package_rels):
        raise StyleGuideError("strict OOXML documents are not supported")
    main = [target for rel_type, target in package_rels if rel_type == REL_OFFICE_DOCUMENT]
    if len(main) != 1:
        raise StyleGuideError("style guide does not have exactly one main document part")
    document_name = main[0]
    doc_dir, doc_base = posixpath.split(document_name)
    rels_name = posixpath.join(doc_dir, "_rels", doc_base + ".rels")
    styles_targets = [t for rel_type, t in _relationships(package, rels_name, doc_dir) if rel_type == REL_STYLES]
    if len(styles_targets) > 1:
        raise StyleGuideError("style guide has more than one styles part")

    root = _parse_xml(package.read(document_name), document_name)
    if root.tag == f"{{{W_STRICT}}}document":
        raise StyleGuideError("strict OOXML documents are not supported")
    if root.tag != _w("document"):
        raise StyleGuideError("style guide main part is not a WordprocessingML document")
    for element in root.iter():
        if element.tag in _TRACKED:
            raise StyleGuideError("style guide contains tracked changes; accept or reject all changes first")
    styles_root = _parse_xml(package.read(styles_targets[0]), styles_targets[0]) if styles_targets else None
    body = root.find(_w("body"))
    if body is None:
        raise StyleGuideError("style guide document has no body")
    extractor = _Extractor(_Styles(styles_root))
    extractor.body(body)
    sections = build_sections(extractor.paragraphs)
    if not sections:
        raise StyleGuideError("style guide contains no text")
    return StyleGuide(sections, extractor.chars)


# --------------------------------------------------------------------------------------------
# Rendering and selection
# --------------------------------------------------------------------------------------------


def _line(paragraph: GuideParagraph) -> str:
    if paragraph.kind == "heading":
        assert paragraph.level is not None
        return "#" * (paragraph.level + 1) + " " + paragraph.text
    if paragraph.kind == "list":
        return "- " + paragraph.text
    return paragraph.text


def render(sections: Iterable[Section]) -> str:
    """Verbatim text with structural markers only; an included section is preceded by any of its
    ancestor headings not already shown (they are covered by its fingerprint)."""
    blocks: list[str] = []
    shown: list[tuple[int, str]] = []
    for section in sections:
        for depth, (level, text) in enumerate(section.path):
            if depth < len(shown) and shown[depth] == (level, text):
                continue
            del shown[depth:]
            blocks.append(_line(GuideParagraph("heading", level, text)))
            shown.append((level, text))
        del shown[len(section.path):]
        if section.level is not None:
            shown.append((section.level, section.paragraphs[0].text))
        blocks.extend(_line(p) for p in section.paragraphs)
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def select(guide: StyleGuide, scope: LocaleScope, manifest: Manifest) -> SelectedGuide:
    """D8 selection: FULL -> every section; EN_GB -> en-GB + cross-locale hashes; CROSS_LOCALE ->
    cross-locale hashes. Unlisted or changed sections are excluded."""
    if scope is LocaleScope.FULL:
        chosen = guide.sections
    elif scope is LocaleScope.EN_GB:
        allowed = manifest.en_gb | manifest.cross_locale
        chosen = tuple(s for s in guide.sections if s.sha256 in allowed)
    elif scope is LocaleScope.CROSS_LOCALE:
        chosen = tuple(s for s in guide.sections if s.sha256 in manifest.cross_locale)
    else:
        raise ValueError("unknown locale scope")
    return SelectedGuide(render(chosen), scope, len(chosen), len(guide.sections), tuple(s.sha256 for s in chosen))


# --------------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------------

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ,;:()/&'-]{0,59}\Z")


def parse_manifest(data: bytes | str) -> Manifest:
    try:
        raw = json.loads(data)
    except (ValueError, TypeError):
        raise ManifestError("neutral-rules manifest is not valid JSON") from None
    if not isinstance(raw, dict) or set(raw) != {"version", "fingerprint", "scopes"}:
        raise ManifestError("neutral-rules manifest must have exactly: version, fingerprint, scopes")
    if raw["version"] != MANIFEST_VERSION or isinstance(raw["version"], bool):
        raise ManifestError("neutral-rules manifest has an unsupported version")
    if raw["fingerprint"] != FINGERPRINT_VERSION:
        raise ManifestError("neutral-rules manifest uses an unsupported fingerprint version")
    scopes = raw["scopes"]
    if not isinstance(scopes, dict) or set(scopes) != {SCOPE_EN_GB, SCOPE_CROSS_LOCALE}:
        raise ManifestError(f"neutral-rules manifest scopes must be exactly {SCOPE_EN_GB!r} and {SCOPE_CROSS_LOCALE!r}")
    found: dict[str, set[str]] = {}
    labels: dict[str, str] = {}
    for scope_name in (SCOPE_EN_GB, SCOPE_CROSS_LOCALE):
        entries = scopes[scope_name]
        if not isinstance(entries, list):
            raise ManifestError(f"neutral-rules manifest scope {scope_name!r} must be a list")
        found[scope_name] = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"sha256", "label"}:
                raise ManifestError("neutral-rules manifest entries must have exactly: sha256, label")
            digest, label = entry["sha256"], entry["label"]
            if not isinstance(digest, str) or not _HASH.match(digest):
                raise ManifestError("neutral-rules manifest has an invalid sha256 value")
            if not isinstance(label, str) or not _LABEL.match(label):
                raise ManifestError("neutral-rules manifest has an invalid label")
            if digest in labels:
                raise ManifestError("neutral-rules manifest lists a hash more than once")
            labels[digest] = label
            found[scope_name].add(digest)
    return Manifest(frozenset(found[SCOPE_EN_GB]), frozenset(found[SCOPE_CROSS_LOCALE]), labels)


def load_manifest() -> Manifest:
    """The packaged manifest (src/proofreader/prompts/neutral_rules.json)."""
    try:
        data = resources.files("proofreader").joinpath("prompts", MANIFEST_RESOURCE).read_bytes()
    except OSError:
        raise ManifestError("neutral-rules manifest is missing from the package") from None
    return parse_manifest(data)

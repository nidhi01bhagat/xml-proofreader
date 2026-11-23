"""Internal data model and error hierarchy (S1).

Plain, immutable dataclasses shared by ``encoding``, ``xml_source`` and ``tokens``.
The Pydantic finding contract and the resolver/renderer types (``ResolvedSpan``,
``Insertion``, ``Rejection``) are added in later stages (S2-S4), not here.

Offsets:
* ``raw_*`` values are byte offsets into the original input (BOM included).
* ``char_*`` values are code-point offsets into a paragraph's logical text.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------------------------
# Errors. ``exit_code`` follows docs/03 section 6: 1 = fatal input, 2 = usage/unsupported input.
# --------------------------------------------------------------------------------------------


class ProofreaderError(Exception):
    """Base class for all expected, user-facing failures."""

    exit_code: int = 1

    def __init__(
        self,
        message: str,
        *,
        byte_offset: int | None = None,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.byte_offset = byte_offset
        self.line = line
        self.column = column


class UnsupportedInputError(ProofreaderError):
    """Input is well-formed enough to classify but outside the supported scope (exit 2)."""

    exit_code = 2


class UnsupportedEncodingError(UnsupportedInputError):
    """UTF-16/UTF-32, any declared encoding other than UTF-8/US-ASCII, or BOM/declaration conflict."""


class UnsupportedXMLVersionError(UnsupportedInputError):
    """XML declaration names a version other than 1.0 (e.g. 1.1)."""


class InputTooLargeError(UnsupportedInputError):
    """Input exceeds the configured size cap."""


class InvalidInputError(ProofreaderError):
    """Input is broken: it cannot be processed safely (exit 1)."""

    exit_code = 1


class EmptyInputError(InvalidInputError):
    """Input contains no bytes (or only a byte-order mark)."""


class InvalidEncodingError(InvalidInputError):
    """Bytes are not valid in the detected/declared encoding; ``byte_offset`` locates the first bad byte."""


class MalformedXMLError(InvalidInputError):
    """XML is not well-formed (including a malformed XML declaration)."""


# --------------------------------------------------------------------------------------------
# Encoding pre-check result
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodingInfo:
    """Result of the pre-parse encoding check. Describes the input; never changes it."""

    encoding: str  # canonical: "utf-8" or "us-ascii"
    codec: str  # Python codec name used to decode raw byte ranges: "utf-8" or "ascii"
    bom_length: int  # 3 for a UTF-8 BOM, else 0
    has_declaration: bool
    declaration_end: int  # byte offset just after "?>" of the XML declaration, 0 if none
    xml_version: str | None  # as written, e.g. "1.0"
    declared_encoding: str | None  # as written, e.g. "utf8"; None if not declared
    standalone: str | None  # "yes" / "no" / None


# --------------------------------------------------------------------------------------------
# Source-mapping model (filled by xml_source in later S1 steps)
# --------------------------------------------------------------------------------------------


class AtomKind(str, Enum):
    LIT = "LIT"  # literal character data; raw bytes decode exactly to the text
    REF = "REF"  # one entity or character reference (atomic)
    NL = "NL"  # a CRLF or CR line break normalised to "\n" (atomic)
    CDATA = "CDATA"  # text inside a CDATA section (preservation boundary, never annotated)
    BAD = "BAD"  # anything not safely mappable -> paragraph not annotatable


class SkipReason(str, Enum):
    EMPTY = "EMPTY"  # no or whitespace-only text
    PARA_HAS_ERROR = "PARA_HAS_ERROR"  # already contains an <error> element (X6, D11)
    PARA_UNMAPPABLE = "PARA_UNMAPPABLE"  # BAD atom, skipped/external entity, failed self-check (X10)
    DOCUMENT_UNVERIFIABLE = "DOCUMENT_UNVERIFIABLE"  # independent cross-check impossible (C2)
    CDATA_ONLY = "CDATA_ONLY"  # only CDATA text (plus whitespace): nothing annotatable (X5)


@dataclass(frozen=True)
class Atom:
    raw_start: int
    raw_end: int
    text: str
    kind: AtomKind
    char_start: int
    char_end: int
    container_id: int  # id of the innermost element containing the atom
    path: tuple[int, ...]  # element ids from root to container


@dataclass(frozen=True)
class ElementInfo:
    id: int  # document order of the start tag
    namespace: str | None
    local_name: str
    start_tag: tuple[int, int]  # raw byte range of the start tag (or the whole empty-element tag)
    end_tag: tuple[int, int]  # raw byte range of the end tag ((n, n) for an empty-element tag)
    parent_id: int | None


@dataclass(frozen=True)
class Paragraph:
    id: int  # document order among <p> elements
    element_id: int
    atoms: tuple[Atom, ...]
    text: str  # logical own text (nested <p> excluded)
    barriers: tuple[int, ...]  # char offsets in ``text`` where a nested <p> sits
    skip_reason: SkipReason | None

    @property
    def annotatable(self) -> bool:
        return self.skip_reason is None


@dataclass(frozen=True)
class ScanResult:
    encoding: EncodingInfo
    elements: tuple[ElementInfo, ...]
    paragraphs: tuple[Paragraph, ...]


# --------------------------------------------------------------------------------------------
# Annotation (S2): resolved spans, rejections, insertions, render result
# --------------------------------------------------------------------------------------------


class Category(str, Enum):
    """Error categories; the value is the lowercase ``type`` attribute written to ``<error>`` (E01)."""

    GRAMMAR = "grammar"
    SPELLING = "spelling"
    PUNCTUATION = "punctuation"
    CAPITALIZATION = "capitalization"
    CLARITY = "clarity"
    STYLEGUIDE = "styleguide"


class RejectCode(str, Enum):
    """Why an annotation was not applied (S2 codes only; resolver/quotation codes arrive in S3)."""

    INVALID_FIELD = "INVALID_FIELD"  # wrong type/value of a field, correction == original, bad reason
    EMPTY_SPAN = "EMPTY_SPAN"  # start == end
    OUT_OF_RANGE = "OUT_OF_RANGE"  # span outside the paragraph text, or start > end
    PARA_NOT_ANNOTATABLE = "PARA_NOT_ANNOTATABLE"  # target paragraph has a skip reason
    SPLITS_ATOMIC = "SPLITS_ATOMIC"  # endpoint inside a reference or CRLF/CR atom
    CROSSES_ELEMENT = "CROSSES_ELEMENT"  # span would touch inline markup (D4 OFF)
    CROSSES_PARAGRAPH = "CROSSES_PARAGRAPH"  # span crosses a nested <p> barrier
    IN_CDATA = "IN_CDATA"  # endpoint inside, or span containing part of, a CDATA section (X5)
    OVERLAP = "OVERLAP"  # nested or partially overlapping spans: all involved are rejected (no D15 in S2)
    CONFLICT = "CONFLICT"  # same span and category, different corrections: all involved are rejected
    VERIFY_FAILED = "VERIFY_FAILED"  # paragraph rolled back after independent verification failed
    # --- S3 (finding resolution and quotation protection; decision B) ---
    NOT_FOUND = "NOT_FOUND"  # no token-aligned occurrence of the original in any matching tier
    ANCHOR_MISMATCH = "ANCHOR_MISMATCH"  # occurrences exist, but none has matching before/after anchors
    AMBIGUOUS = "AMBIGUOUS"  # several anchor-valid candidates and the word hint does not single one out
    SPAN_TOO_LONG = "SPAN_TOO_LONG"  # original longer than 6 words (20 for clarity)
    TOO_MANY_FINDINGS = "TOO_MANY_FINDINGS"  # more than 30 raw findings for one paragraph (whole paragraph)
    SCHEMA_INVALID = "SCHEMA_INVALID"  # response envelope malformed (whole paragraph)
    IN_QUOTATION = "IN_QUOTATION"  # span overlaps protected quotation content
    TOUCHES_QUOTATION = "TOUCHES_QUOTATION"  # original contains a quote mark, or span adjacent to a quotation
    QUOTATION_IN_CORRECTION = "QUOTATION_IN_CORRECTION"  # correction contains or would create a quotation delimiter


@dataclass(frozen=True)
class Annotation:
    """A resolved span to be marked up: ``Paragraph.text[start:end]`` gets ``<error type correction [reason]>``."""

    paragraph_id: int
    start: int  # code-point offset in Paragraph.text (inclusive)
    end: int  # code-point offset in Paragraph.text (exclusive)
    category: Category
    correction: str  # may be empty (deletion)
    reason: str | None = None  # required for STYLEGUIDE, optional otherwise


@dataclass(frozen=True)
class Rejection:
    annotation: Annotation
    code: RejectCode
    detail: str = ""


@dataclass(frozen=True)
class Insertion:
    """Bytes to insert into the original document at ``offset`` (never replacing anything)."""

    offset: int
    order: int  # 0 = closing tag, 1 = opening tag: closing tags go first at equal offsets
    data: bytes
    paragraph_id: int


@dataclass(frozen=True)
class RenderResult:
    output: bytes
    applied: tuple[Annotation, ...]
    rejected: tuple[Rejection, ...]
    rolled_back: tuple[int, ...]  # paragraph ids whose annotations were rolled back by verification


@dataclass(frozen=True)
class AnnotationOutcome:
    """Result of annotate -> verify -> per-paragraph rollback (S2.5). Never carries unverified output.

    ``ok`` is True only if ``output`` passed independent verification; otherwise ``output`` is None
    and ``failures`` explains why (fail closed: nothing may be written).
    """

    ok: bool
    output: bytes | None
    applied: tuple[Annotation, ...]  # annotations present in the verified output (retained)
    rolled_back: tuple[int, ...]  # paragraph ids whose annotations were removed after verification failed
    rejected: tuple[Rejection, ...]  # span-check/overlap rejections plus VERIFY_FAILED for rolled-back ones
    failures: tuple[str, ...]  # human-readable reasons when ok is False (and rollback reasons when ok is True)
    rounds: int  # render+verify rounds performed


# --------------------------------------------------------------------------------------------
# Tokens (tokens.py)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Word:
    index: int
    start: int
    end: int
    text: str


# --------------------------------------------------------------------------------------------
# S3: model finding contract (Pydantic, strict) and internal resolution / quotation types
# --------------------------------------------------------------------------------------------

MAX_FINDINGS_PER_PARAGRAPH = 30
MAX_ORIGINAL_CHARS = 300
MAX_FINDING_CORRECTION_CHARS = 300
MAX_FINDING_REASON_CHARS = 200
MAX_ANCHOR_WORDS = 4

FindingType = Literal["grammar", "spelling", "punctuation", "capitalization", "clarity", "styleguide"]


def _first_xml_illegal(text: str) -> int | None:
    """Index of the first character XML 1.0 forbids (same set as S2 annotation validation), or None."""
    for i, ch in enumerate(text):
        cp = ord(ch)
        if cp <= 0x08 or cp in (0x0B, 0x0C) or 0x0E <= cp <= 0x1F or 0xD800 <= cp <= 0xDFFF or cp in (0xFFFE, 0xFFFF):
            return i
    return None


class Finding(BaseModel):
    """One model-proposed finding (docs/03 section 3). Strict: no type coercion, no extra keys.

    Word indices are 0-based and inclusive into the resolver's barrier-aware word grid; they are
    hints only (exact tie-break). The 30-findings limit is enforced on the raw payload by the
    resolver, before this model is used, so it can report TOO_MANY_FINDINGS.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    original: str = Field(min_length=1, max_length=MAX_ORIGINAL_CHARS)
    before: str
    after: str
    word_start: int = Field(ge=0)
    word_end: int = Field(ge=0)
    correction: str = Field(max_length=MAX_FINDING_CORRECTION_CHARS)
    type: FindingType
    reason: str = Field(min_length=1, max_length=MAX_FINDING_REASON_CHARS)

    @field_validator("original", "before", "after", "correction", "reason")
    @classmethod
    def _no_xml_illegal_characters(cls, value: str) -> str:
        bad = _first_xml_illegal(value)
        if bad is not None:
            raise ValueError(f"contains the XML-illegal character U+{ord(value[bad]):04X}")
        return value

    @field_validator("original", "reason")
    @classmethod
    def _not_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace-only")
        return value

    @field_validator("before", "after")
    @classmethod
    def _anchor_word_limit(cls, value: str) -> str:
        if len(value.split()) > MAX_ANCHOR_WORDS:
            raise ValueError(f"anchor longer than {MAX_ANCHOR_WORDS} words")
        return value

    @model_validator(mode="after")
    def _word_range_ordered(self) -> "Finding":
        if self.word_end < self.word_start:
            raise ValueError("word_end must be >= word_start")
        return self


class FindingList(BaseModel):
    """Response envelope. Source of the JSON schema given to the model (S4); the resolver validates
    findings one by one so that one invalid finding never blocks the others."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    findings: list[Finding] = Field(max_length=MAX_FINDINGS_PER_PARAGRAPH)


class MatchMode(str, Enum):
    """Tier that resolved a finding (decision A: evaluated strictly in this order)."""

    EXACT = "exact"
    WHITESPACE_GAP = "whitespace_gap"
    NFC = "nfc"


class QuoteFamily(str, Enum):
    DOUBLE = "double"
    SINGLE = "single"
    CORNER = "corner"  # 「 」
    WHITE_CORNER = "white_corner"  # 『 』
    OTHER = "other"  # unlisted Unicode Pi/Pf characters


class QuoteRole(str, Enum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    OPEN_CANDIDATE = "OPEN_CANDIDATE"
    CLOSE_OR_APOSTROPHE = "CLOSE_OR_APOSTROPHE"
    APOSTROPHE = "APOSTROPHE"
    AMBIGUOUS = "AMBIGUOUS"


class RangeOrigin(str, Enum):
    PAIR = "pair"
    UNCLOSED = "unclosed"
    EMPTY_STACK_CLOSE = "empty_stack_close"
    AMBIGUOUS = "ambiguous"
    GUARD = "guard"
    INHERITED = "inherited"


class TraceStatus(str, Enum):
    APPLIED = "APPLIED"
    REJECTED_S3 = "REJECTED_S3"
    REJECTED_S2 = "REJECTED_S2"
    ROLLED_BACK = "ROLLED_BACK"
    DOCUMENT_FAILED = "DOCUMENT_FAILED"


@dataclass(frozen=True)
class CheckedFinding:
    """A validated finding, converted to an immutable internal form (no Pydantic dependency)."""

    index: int  # position in the raw findings array
    original: str  # edge whitespace stripped
    before: str
    after: str
    word_start: int
    word_end: int
    correction: str
    category: Category
    reason: str
    original_stripped: bool  # True if edge whitespace was removed from the original


@dataclass(frozen=True)
class FindingRejection:
    """A finding (or a whole paragraph's response) rejected before it became an Annotation."""

    paragraph_id: int
    finding_index: int | None  # None: applies to the whole paragraph (no readable findings list)
    code: RejectCode
    detail: str  # field and rule only, never document text


@dataclass(frozen=True)
class ResolvedFinding:
    finding_index: int
    annotation: Annotation
    match_mode: MatchMode
    hint_drift: bool  # informational: the word hint differed from the resolved span (not a rejection)


@dataclass(frozen=True)
class ResolutionResult:
    paragraph_id: int
    raw_count: int | None  # length of the raw findings array, None if unreadable
    resolved: tuple[ResolvedFinding, ...]
    rejections: tuple[FindingRejection, ...]


@dataclass(frozen=True)
class GridWord:
    """One word of the barrier-aware word grid (decision U2); offsets are paragraph-global."""

    index: int  # 0-based, continuous across barrier-delimited segments
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class QuoteMark:
    offset: int
    char: str
    family: QuoteFamily
    role: QuoteRole


@dataclass(frozen=True)
class ProtectedRange:
    start: int
    end: int  # exclusive; marks are included in the range
    open_start: bool  # reaches the paragraph start because an opening mark is missing
    open_end: bool  # reaches the paragraph end because a closing mark is missing
    origin: RangeOrigin


@dataclass(frozen=True)
class QuoteAnalysis:
    paragraph_id: int
    marks: tuple[QuoteMark, ...]
    ranges: tuple[ProtectedRange, ...]
    unsafe_reason: str | None  # set when the adversarial guard protected the whole paragraph
    inherited_from: int | None  # outer paragraph id when protection was propagated (U3)


@dataclass(frozen=True)
class TraceEntry:
    """Final status of one raw finding (or of a whole paragraph response when finding_index is None)."""

    paragraph_id: int
    finding_index: int | None
    status: TraceStatus
    code: RejectCode | None


@dataclass(frozen=True)
class TraceRecord:
    """Audit record of one raw finding (S3.6): the TraceEntry core plus how it was resolved.

    ``match_mode``/``start``/``end``/``hint_drift`` are set only for findings that resolved to a span.
    ``detail`` is the S3 rejection detail (field/rule names and counts only); it is empty for S2
    outcomes, whose messages may quote document text.
    """

    paragraph_id: int
    finding_index: int | None
    status: TraceStatus
    code: RejectCode | None
    match_mode: MatchMode | None
    start: int | None
    end: int | None
    hint_drift: bool | None
    detail: str

    @property
    def entry(self) -> TraceEntry:
        return TraceEntry(self.paragraph_id, self.finding_index, self.status, self.code)

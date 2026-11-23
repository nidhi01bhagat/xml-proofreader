"""Pre-parse encoding policy (docs/02 section 7, D12, C3, C4).

``check(data)`` inspects the raw input bytes *before* any XML parsing, API call or output file
and either returns an :class:`EncodingInfo` or raises a typed :class:`ProofreaderError`.
It never decodes-and-re-encodes, strips a BOM, or rewrites the declaration: the input is
only read.

Supported: UTF-8 (optional BOM) and US-ASCII. Everything else is rejected explicitly, even
encodings that Expat itself would accept (UTF-16, ISO-8859-1, windows-1252, ...).

Order of checks (first failure wins):
1. type, empty input, size cap
2. UTF-32 / UTF-16 byte-order marks, then BOM-less UTF-16/UTF-32 (NUL in the first 4 bytes)
3. optional UTF-8 BOM (input consisting only of a BOM counts as empty)
4. XML declaration syntax (only if the content starts with ``<?xml`` + whitespace or ``?``)
5. XML version (only 1.0)
6. declared encoding vs BOM
7. byte validity for the effective encoding (first offending byte offset reported)

Not checked here (left to the XML parser in xml_source): well-formedness beyond the
declaration, a declaration that is not at the very start, NUL bytes after the first four.
"""

from __future__ import annotations

import re

from proofreader.models import (
    EmptyInputError,
    EncodingInfo,
    InputTooLargeError,
    InvalidEncodingError,
    MalformedXMLError,
    UnsupportedEncodingError,
    UnsupportedXMLVersionError,
)

MAX_INPUT_BYTES = 10 * 1024 * 1024  # 10 MiB (C3); configurable from S4 via the max_bytes argument

UTF8_BOM = b"\xef\xbb\xbf"
_UNSUPPORTED_BOMS = (  # UTF-32 first: its LE BOM starts with the UTF-16 LE BOM
    (b"\x00\x00\xfe\xff", "UTF-32 (big-endian BOM)"),
    (b"\xff\xfe\x00\x00", "UTF-32 (little-endian BOM)"),
    (b"\xfe\xff", "UTF-16 (big-endian BOM)"),
    (b"\xff\xfe", "UTF-16 (little-endian BOM)"),
)

_UTF8_NAMES = {"utf-8", "utf8"}
_ASCII_NAMES = {"us-ascii", "ascii"}

_S = rb"[ \t\r\n]+"
_EQ = rb"[ \t\r\n]*=[ \t\r\n]*"
_DECL_START = re.compile(rb"<\?xml(?=[ \t\r\n?])")
_DECL = re.compile(
    rb"<\?xml"
    + _S + rb"version" + _EQ + rb"(?P<vq>['\"])(?P<version>[^'\"]*)(?P=vq)"
    + rb"(?:" + _S + rb"encoding" + _EQ + rb"(?P<eq>['\"])(?P<encoding>[^'\"]*)(?P=eq))?"
    + rb"(?:" + _S + rb"standalone" + _EQ + rb"(?P<sq>['\"])(?P<standalone>[^'\"]*)(?P=sq))?"
    + rb"[ \t\r\n]*\?>"
)
_VERSION_NUM = re.compile(rb"1\.[0-9]+")
_ENC_NAME = re.compile(rb"[A-Za-z][A-Za-z0-9._-]*")


def check(data: bytes, *, max_bytes: int = MAX_INPUT_BYTES) -> EncodingInfo:
    """Validate the encoding of ``data`` and describe it. Raises a ProofreaderError subclass."""
    if not isinstance(data, bytes):
        raise TypeError(f"check() expects bytes, got {type(data).__name__}")
    if not data:
        raise EmptyInputError("input is empty")
    if len(data) > max_bytes:
        raise InputTooLargeError(f"input is {len(data)} bytes; the limit is {max_bytes} bytes")

    for bom, name in _UNSUPPORTED_BOMS:
        if data.startswith(bom):
            raise UnsupportedEncodingError(f"unsupported encoding: {name}; only UTF-8 and US-ASCII are supported")
    if b"\x00" in data[:4]:
        raise UnsupportedEncodingError(
            "unsupported encoding: UTF-16/UTF-32 without BOM (NUL byte at the start); "
            "only UTF-8 and US-ASCII are supported",
            byte_offset=data.index(b"\x00"),
        )

    bom_length = len(UTF8_BOM) if data.startswith(UTF8_BOM) else 0
    if bom_length and len(data) == bom_length:
        raise EmptyInputError("input contains only a UTF-8 byte-order mark")

    version = declared = standalone = None
    declaration_end = 0
    if _DECL_START.match(data, bom_length):
        m = _DECL.match(data, bom_length)
        if m is None:
            raise MalformedXMLError("malformed XML declaration", byte_offset=bom_length)
        declaration_end = m.end()
        version = _ascii_field(m, "version")
        declared = _ascii_field(m, "encoding")
        standalone = _ascii_field(m, "standalone")
        if not _VERSION_NUM.fullmatch(m.group("version")):
            raise MalformedXMLError(f"malformed XML version {version!r}", byte_offset=m.start("version"))
        if version != "1.0":
            raise UnsupportedXMLVersionError(f"unsupported XML version {version!r}; only XML 1.0 is supported")
        if declared is not None and not _ENC_NAME.fullmatch(m.group("encoding")):
            raise MalformedXMLError(f"malformed encoding name {declared!r}", byte_offset=m.start("encoding"))
        if standalone is not None and standalone not in ("yes", "no"):
            raise MalformedXMLError(f"malformed standalone value {standalone!r}", byte_offset=m.start("standalone"))

    encoding = _effective_encoding(declared, bom_length)
    codec = "utf-8" if encoding == "utf-8" else "ascii"
    _validate_bytes(data, bom_length, encoding)

    return EncodingInfo(
        encoding=encoding,
        codec=codec,
        bom_length=bom_length,
        has_declaration=declaration_end > 0,
        declaration_end=declaration_end,
        xml_version=version,
        declared_encoding=declared,
        standalone=standalone,
    )


def _ascii_field(m: re.Match[bytes], name: str) -> str | None:
    value = m.group(name)
    if value is None:
        return None
    try:
        return value.decode("ascii")
    except UnicodeDecodeError:
        raise MalformedXMLError(f"non-ASCII characters in XML declaration {name}", byte_offset=m.start(name)) from None


def _effective_encoding(declared: str | None, bom_length: int) -> str:
    if declared is None:
        return "utf-8"
    name = declared.lower()
    if name in _UTF8_NAMES:
        return "utf-8"
    if bom_length:
        raise UnsupportedEncodingError(f"byte-order mark (UTF-8) contradicts declared encoding {declared!r}")
    if name in _ASCII_NAMES:
        return "us-ascii"
    raise UnsupportedEncodingError(
        f"unsupported declared encoding {declared!r}; only UTF-8 and US-ASCII are supported"
    )


def _validate_bytes(data: bytes, start: int, encoding: str) -> None:
    codec, label = ("ascii", "US-ASCII") if encoding == "us-ascii" else ("utf-8", "UTF-8")
    try:
        data[start:].decode(codec, "strict")  # validation only; the decoded text is discarded
    except UnicodeDecodeError as exc:
        offset = start + exc.start
        raise InvalidEncodingError(
            f"invalid {label} at byte offset {offset} (0x{data[offset]:02X}): {exc.reason}", byte_offset=offset
        ) from None

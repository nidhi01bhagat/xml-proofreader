"""S1 step 2: pre-parse encoding policy (docs/02 section 7, D12, C3, C4).

S1 asserts the encoding-layer outcome (EncodingInfo or a typed error with exit code);
wrapping/rendering and process exit codes are asserted in S2/S4 (decision C1).
All inputs are invented; nothing here reads the ignored assessment directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from proofreader import encoding
from proofreader.models import (
    EmptyInputError,
    EncodingInfo,
    InputTooLargeError,
    InvalidEncodingError,
    MalformedXMLError,
    ProofreaderError,
    UnsupportedEncodingError,
    UnsupportedXMLVersionError,
)

XML = Path(__file__).resolve().parent / "fixtures" / "xml"
BOM = b"\xef\xbb\xbf"


def fx(name: str) -> bytes:
    return (XML / name).read_bytes()


def info(data: bytes, **kw) -> EncodingInfo:
    before = bytes(data)
    result = encoding.check(data, **kw)
    assert data == before  # input never modified
    return result


# --------------------------------------------------------------------------------------------
# Accepted inputs
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "declared"),
    [
        ("enc-06a-decl-UTF-8.xml", "UTF-8"),
        ("enc-06b-decl-utf-8.xml", "utf-8"),
        ("enc-06c-decl-utf8.xml", "utf8"),
        ("enc-06d-decl-no-encoding.xml", None),
    ],
)
def test_enc06_utf8_declarations_accepted_and_reported_as_written(name: str, declared: str | None) -> None:
    data = fx(name)
    i = info(data)
    assert (i.encoding, i.codec, i.bom_length) == ("utf-8", "utf-8", 0)
    assert i.has_declaration and i.xml_version == "1.0"
    assert i.declared_encoding == declared  # recorded exactly as written, never normalised in place
    assert data[: i.declaration_end].endswith(b"?>") and data[i.declaration_end :] == b"<p>teh</p>"


def test_enc16_no_declaration() -> None:
    i = info(fx("enc-16-no-declaration.xml"))
    assert i == EncodingInfo("utf-8", "utf-8", 0, False, 0, None, None, None)


def test_enc05_utf8_bom_without_declaration() -> None:
    i = info(fx("enc-05-utf8-bom.xml"))
    assert (i.encoding, i.bom_length, i.has_declaration) == ("utf-8", 3, False)


@pytest.mark.parametrize("decl", [b"UTF-8", b"utf-8", b"utf8", b"Utf8"])
def test_utf8_bom_with_matching_declaration(decl: bytes) -> None:
    data = BOM + b'<?xml version="1.0" encoding="' + decl + b'"?><p>teh</p>'
    i = info(data)
    assert (i.encoding, i.bom_length) == ("utf-8", 3)
    assert i.declaration_end == data.index(b"?>") + 2  # offsets include the BOM


def test_enc07_us_ascii_declaration() -> None:
    i = info(fx("enc-07-us-ascii.xml"))
    assert (i.encoding, i.codec, i.declared_encoding) == ("us-ascii", "ascii", "US-ASCII")


@pytest.mark.parametrize("decl", [b"us-ascii", b"ASCII", b"ascii"])
def test_ascii_aliases(decl: bytes) -> None:
    assert info(b'<?xml version="1.0" encoding="' + decl + b'"?><p>x</p>').encoding == "us-ascii"


@pytest.mark.parametrize(
    "name",
    ["enc-01-crlf.xml", "enc-02-crlf-gap.xml", "enc-03-lone-cr.xml", "enc-04-mixed-eol.xml", "enc-14-multibyte.xml"],
)
def test_line_endings_and_multibyte_are_plain_utf8(name: str) -> None:
    # ENC-01..04, ENC-14: the encoding layer accepts them unchanged; mapping is tested in xml_source.
    assert info(fx(name)).encoding == "utf-8"


def test_enc15_prolog_with_pi_comment_doctype() -> None:
    data = fx("enc-15-prolog.xml")
    i = info(data)
    assert i.encoding == "utf-8" and i.has_declaration
    assert data[i.declaration_end : i.declaration_end + 1] == b"\n"


def test_standalone_and_whitespace_variants() -> None:
    i = info(b"<?xml version = '1.0'\tencoding\n=\r\n'UTF-8'   standalone='yes' ?><p/>")
    assert (i.xml_version, i.declared_encoding, i.standalone) == ("1.0", "UTF-8", "yes")


def test_processing_instruction_named_like_xml_is_not_a_declaration() -> None:
    i = info(b'<?xml-stylesheet href="s.css"?><p>teh</p>')
    assert not i.has_declaration and i.encoding == "utf-8"


# --------------------------------------------------------------------------------------------
# Unsupported encodings -> exit 2
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "enc-09a-utf16le-bom.xml",
        "enc-09b-utf16be-bom.xml",
        "enc-09c-utf16le-no-bom.xml",
        "enc-09d-utf16be-no-bom.xml",
        "enc-10a-utf32le-bom.xml",
        "enc-10b-utf32be-bom.xml",
        "enc-11a-decl-iso-8859-1.xml",
        "enc-11b-decl-windows-1252.xml",
        "enc-12-bom-decl-utf16.xml",
    ],
)
def test_enc09_to_12_unsupported_encodings_exit_2(name: str) -> None:
    with pytest.raises(UnsupportedEncodingError) as exc:
        encoding.check(fx(name))
    assert exc.value.exit_code == 2


def test_utf32_le_bom_not_misreported_as_utf16() -> None:
    with pytest.raises(UnsupportedEncodingError, match="UTF-32"):
        encoding.check(fx("enc-10a-utf32le-bom.xml"))


@pytest.mark.parametrize("decl", [b"us-ascii", b"ISO-8859-1", b"UTF-16"])
def test_bom_contradicting_declaration_exit_2(decl: bytes) -> None:
    with pytest.raises(UnsupportedEncodingError, match="contradicts") as exc:
        encoding.check(BOM + b'<?xml version="1.0" encoding="' + decl + b'"?><p/>')
    assert exc.value.exit_code == 2


def test_declared_utf16_without_bom_is_unsupported() -> None:
    with pytest.raises(UnsupportedEncodingError, match="UTF-16"):
        encoding.check(b'<?xml version="1.0" encoding="UTF-16"?><p/>')


# --------------------------------------------------------------------------------------------
# Invalid bytes -> exit 1 with byte offset
# --------------------------------------------------------------------------------------------


def test_enc13_invalid_utf8_reports_offset() -> None:
    data = fx("enc-13-invalid-utf8.xml")
    with pytest.raises(InvalidEncodingError) as exc:
        encoding.check(data)
    assert exc.value.exit_code == 1
    assert exc.value.byte_offset == data.index(b"\xe9")
    assert str(exc.value.byte_offset) in exc.value.message


def test_enc08_non_ascii_byte_under_us_ascii_reports_offset() -> None:
    data = fx("enc-08-ascii-high-byte.xml")
    with pytest.raises(InvalidEncodingError) as exc:
        encoding.check(data)
    assert exc.value.exit_code == 1
    assert exc.value.byte_offset == data.index(b"\xc3")


@pytest.mark.parametrize(
    ("body", "bad_index_in_body"),
    [
        (b"ab\x80cd", 2),  # lone continuation byte
        (b"ab\xc3", 2),  # truncated 2-byte sequence at end of input
        (b"ab\xe2\x82", 2),  # truncated 3-byte sequence at end of input
        (b"ab\xe2\x28\xa1", 2),  # invalid continuation
        (b"ab\xc0\xaf", 2),  # overlong encoding of '/'
        (b"ab\xed\xa0\x80", 2),  # UTF-16 surrogate encoded in UTF-8
        (b"ab\xf4\x90\x80\x80", 2),  # code point above U+10FFFF
        (b"ab\xff", 2),  # never valid in UTF-8
        ("é😀".encode() + b"\x80", 6),  # offset counts bytes, not characters
    ],
)
def test_invalid_utf8_byte_offsets(body: bytes, bad_index_in_body: int) -> None:
    prefix = b"<p>"
    with pytest.raises(InvalidEncodingError) as exc:
        encoding.check(prefix + body + b"</p>")
    assert exc.value.byte_offset == len(prefix) + bad_index_in_body


def test_invalid_utf8_offset_includes_bom() -> None:
    data = BOM + b"<p>\x80</p>"
    with pytest.raises(InvalidEncodingError) as exc:
        encoding.check(data)
    assert exc.value.byte_offset == 6


# --------------------------------------------------------------------------------------------
# Empty input, size cap
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("data", [b"", BOM])
def test_empty_input_exit_1(data: bytes) -> None:
    with pytest.raises(EmptyInputError) as exc:
        encoding.check(data)
    assert exc.value.exit_code == 1


def test_size_cap_default_is_10_mib() -> None:
    assert encoding.MAX_INPUT_BYTES == 10 * 1024 * 1024


def test_size_cap_boundary_exact_limit_accepted_one_over_rejected() -> None:
    body = b"<p>" + b"x" * (encoding.MAX_INPUT_BYTES - 7) + b"</p>"
    assert len(body) == encoding.MAX_INPUT_BYTES
    assert info(body).encoding == "utf-8"
    with pytest.raises(InputTooLargeError) as exc:
        encoding.check(body + b" ")
    assert exc.value.exit_code == 2


def test_size_cap_is_checked_before_decoding() -> None:
    # Invalid UTF-8 in an oversized input still reports the size problem (cheap check first).
    with pytest.raises(InputTooLargeError):
        encoding.check(b"\xff" * 33, max_bytes=32)


# --------------------------------------------------------------------------------------------
# XML version and malformed declarations
# --------------------------------------------------------------------------------------------


def test_xml_1_1_rejected_exit_2() -> None:
    with pytest.raises(UnsupportedXMLVersionError) as exc:
        encoding.check(fx("ver-xml-1.1.xml"))
    assert exc.value.exit_code == 2


@pytest.mark.parametrize(
    "decl",
    [
        b"<?xml?>",  # no version
        b'<?xml encoding="UTF-8"?>',  # version missing
        b"<?xml version=1.0?>",  # unquoted
        b"<?xml version=\"1.0'?>",  # mismatched quotes
        b'<?xml version="1.0" encoding="UTF-8"',  # unterminated
        b'<?xml version="1.0"encoding="UTF-8"?>',  # missing whitespace between pseudo-attributes
        b'<?xml version="1.0" standalone="yes" encoding="UTF-8"?>',  # wrong order
        b'<?xml version="1.0" foo="bar"?>',  # unknown pseudo-attribute
        b'<?xml version="1.0" encoding="UTF 8"?>',  # invalid encoding name
        b'<?xml version="1.0" encoding=""?>',  # empty encoding name
        b'<?xml version="1.0" standalone="maybe"?>',  # invalid standalone
        b'<?xml version="one"?>',  # invalid version number
        b'<?xml version="2.0"?>',  # not 1.x
    ],
)
def test_malformed_declarations_exit_1(decl: bytes) -> None:
    with pytest.raises(MalformedXMLError) as exc:
        encoding.check(decl + b"<p/>")
    assert exc.value.exit_code == 1


def test_non_ascii_in_declaration_is_malformed() -> None:
    with pytest.raises(MalformedXMLError):
        encoding.check('<?xml version="1.0" encoding="utf-8é"?><p/>'.encode())


@pytest.mark.parametrize("version", [b"1.1", b"1.5", b"1.10"])
def test_any_xml_1_x_other_than_1_0_unsupported(version: bytes) -> None:
    with pytest.raises(UnsupportedXMLVersionError):
        encoding.check(b'<?xml version="' + version + b'"?><p/>')


# --------------------------------------------------------------------------------------------
# API contract
# --------------------------------------------------------------------------------------------


def test_rejects_non_bytes_input() -> None:
    with pytest.raises(TypeError):
        encoding.check("<p>teh</p>")  # type: ignore[arg-type]


def test_all_failures_are_proofreader_errors() -> None:
    for data in (b"", fx("enc-09a-utf16le-bom.xml"), fx("enc-13-invalid-utf8.xml"), fx("ver-xml-1.1.xml")):
        with pytest.raises(ProofreaderError):
            encoding.check(data)

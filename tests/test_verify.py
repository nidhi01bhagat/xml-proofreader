"""S2.4: independent verification of rendered output (no rollback yet; no LLM).

Correct renders must verify cleanly; every kind of corruption must be detected with an
actionable issue code. Outputs are produced by the S2.3 renderer and then corrupted by hand,
so the verifier is tested against outputs it did not help produce.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from proofreader import annotation as an
from proofreader import verify as v
from proofreader import xml_source as xs
from proofreader.models import Annotation, Category
from proofreader.verify import IssueCode as I

XML = Path(__file__).resolve().parent / "fixtures" / "xml"
SP, GR, PU, CA, CL, ST = (Category.SPELLING, Category.GRAMMAR, Category.PUNCTUATION,
                          Category.CAPITALIZATION, Category.CLARITY, Category.STYLEGUIDE)
DOC = b"<r><p>The firm recieved teh parcel.</p><p>Second <b>bold</b> line.</p></r>"
ANNS = [Annotation(0, 9, 17, SP, "received"), Annotation(0, 18, 21, GR, "the")]


def rendered(data: bytes, annotations):
    scan = xs.scan(data)
    accepted, rejected = an.prepare(data, scan, annotations)
    assert rejected == [], rejected
    output, _ = an.render(data, accepted, scan.encoding.codec)
    return output, [s.annotation for s in accepted]


def codes(report):
    return [i.code for i in report.document_issues], {pid: [i.code for i in iss] for pid, iss in report.paragraph_issues}


OUT, APPLIED = rendered(DOC, ANNS)


# --------------------------------------------------------------------------------------------
# Correct renders verify cleanly
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "annotations"),
    [
        ("m01-plain.xml", [Annotation(0, 3, 11, SP, "received")]),
        ("m03-amp.xml", [Annotation(0, 0, 12, SP, "Salt & pepper"), Annotation(0, 21, 24, GR, "the")]),
        ("m10-crlf-gap.xml", [Annotation(0, 0, 9, SP, "ice cream")]),
        ("m12-custom-entity.xml", [Annotation(0, 4, 13, CA, "ACME Corp")]),
        ("nest-01.xml", [Annotation(0, 6, 9, SP, "the"), Annotation(1, 6, 9, SP, "the"), Annotation(0, 16, 19, SP, "the")]),
        ("ns-02-prefixed.xml", [Annotation(0, 0, 3, SP, "the")]),
        ("m13-default-ns.xml", [Annotation(0, 0, 3, SP, "the")]),
        ("order-01-mixed-ns.xml", [Annotation(0, 0, 3, CA, "One"), Annotation(1, 0, 3, CA, "Two"), Annotation(2, 0, 5, CA, "Three")]),
        ("enc-14-multibyte.xml", [Annotation(0, 0, 4, SP, "cafe"), Annotation(0, 9, 12, SP, "the")]),
        ("decl-01-bom-decl-crlf.xml", [Annotation(0, 0, 3, SP, "teh"), Annotation(0, 4, 5, PU, "and")]),
        ("enc-07-us-ascii.xml", [Annotation(0, 0, 4, PU, "don’t", "apostrophe ’ \U0001F600")]),
        ("err-02-error-in-nested.xml", [Annotation(2, 0, 5, CA, "Clean")]),  # existing errors elsewhere
        ("cd-04-cdata-inside.xml", [Annotation(0, 0, 1, CA, "A"), Annotation(0, 6, 7, CA, "B")]),
        ("m09-comment-pi.xml", [Annotation(0, 0, 4, CL, "really"), Annotation(0, 5, 9, CL, "great")]),
    ],
)
def test_correct_renders_verify(name: str, annotations) -> None:
    data = (XML / name).read_bytes()
    output, applied = rendered(data, annotations)
    report = v.verify(data, output, applied)
    assert report.ok, codes(report)
    assert report.failed_paragraphs == () and report.document_issues == ()


def test_adjacent_annotations_and_every_category() -> None:
    data = b"<p>The firm recieved teh parcel on monday.</p>"
    annotations = [Annotation(0, 0, 3, CA, "THE"), Annotation(0, 4, 8, CL, "company"), Annotation(0, 9, 17, SP, "received"),
                   Annotation(0, 17, 18, PU, ", "), Annotation(0, 18, 21, GR, "the"),
                   Annotation(0, 32, 38, ST, 'Monday', 'Days: "capitalised" & <named>')]
    output, applied = rendered(data, annotations)
    assert b"</error><error" in output
    assert v.verify(data, output, applied).ok


def test_default_namespace_is_inherited_by_inserted_error() -> None:
    data = b'<r xmlns="urn:d"><p>teh cat</p></r>'
    output, applied = rendered(data, [Annotation(0, 0, 3, SP, "the")])
    assert b'<error type="spelling"' in output  # unprefixed, no xmlns (S2.3 decision)
    assert v.verify(data, output, applied).ok


def test_unicode_attribute_values_round_trip() -> None:
    value = "café — \U0001F600 & \"q\" <t>\n\t"
    output, applied = rendered(b"<p>teh cat</p>", [Annotation(0, 0, 3, ST, value, value)])
    assert v.verify(b"<p>teh cat</p>", output, applied).ok


def test_no_annotations_requires_identical_output() -> None:
    assert v.verify(DOC, DOC, []).ok
    report = v.verify(DOC, OUT, [])
    assert codes(report) == ([I.UNCHANGED_EXPECTED], {})


def test_duplicate_applied_annotations_are_tolerated() -> None:
    assert v.verify(DOC, OUT, APPLIED + APPLIED).ok


# --------------------------------------------------------------------------------------------
# Corruptions are detected, with actionable codes
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "output", "applied", "doc_codes", "para_codes"),
    [
        ("missing tag", OUT, APPLIED + [Annotation(0, 22, 28, SP, "parcels")], [], [I.MISSING_ERROR]),
        ("extra tag (no annotation)", OUT, APPLIED[:1], [], [I.EXTRA_ERROR]),
        ("type changed", OUT.replace(b'type="grammar"', b'type="spelling"'), APPLIED, [], [I.ATTRIBUTES_MISMATCH]),
        ("correction changed", OUT.replace(b'correction="the"', b'correction="teh"'), APPLIED, [], [I.ATTRIBUTES_MISMATCH]),
        ("reason added", OUT.replace(b'correction="the"', b'correction="the" reason="x"'), APPLIED, [], [I.ATTRIBUTES_MISMATCH]),
        ("attribute order swapped", OUT.replace(b'type="grammar" correction="the"', b'correction="the" type="grammar"'),
         APPLIED, [], [I.TAG_NOT_CANONICAL, I.ATTRIBUTES_MISMATCH]),
        ("original text altered", OUT.replace(b"parcel", b"parcal"), APPLIED, [], [I.BYTES_NOT_RESTORED, I.TEXT_CHANGED]),
        ("text inside error altered", OUT.replace(b">teh</error>", b">tea</error>"), APPLIED, [],
         [I.BYTES_NOT_RESTORED, I.TEXT_CHANGED, I.SPAN_MISMATCH]),
        ("closing tag misplaced", OUT.replace(b">teh</error>", b">te</error>h"), APPLIED, [], [I.MISSING_ERROR, I.EXTRA_ERROR]),
        ("prefixed error element", OUT.replace(b'<error type="grammar"', b'<x:error xmlns:x="urn:x" type="grammar"').replace(
            b"teh</error>", b"teh</x:error>"), APPLIED, [], [I.NAMESPACE_MISMATCH, I.TAG_NOT_CANONICAL, I.BYTES_NOT_RESTORED]),
        ("xmlns on inserted error", OUT.replace(b'<error type="grammar"', b'<error xmlns="urn:z" type="grammar"'),
         APPLIED, [], [I.NAMESPACE_MISMATCH, I.TAG_NOT_CANONICAL, I.BYTES_NOT_RESTORED]),
        ("nested error", OUT.replace(b">teh</error>", b'><error type="grammar" correction="x">teh</error></error>'),
         APPLIED, [], [I.NESTED_ERROR, I.EXTRA_ERROR, I.ATTRIBUTES_MISMATCH]),
        ("markup changed in annotated paragraph", OUT.replace(b"parcel.", b"<i>parcel</i>."), APPLIED, [],
         [I.BYTES_NOT_RESTORED, I.STRUCTURE_CHANGED]),
    ],
)
def test_paragraph_level_corruptions(label, output, applied, doc_codes, para_codes) -> None:
    report = v.verify(DOC, output, applied)
    assert not report.ok
    assert codes(report) == (doc_codes, {0: para_codes}), label
    assert report.failed_paragraphs == (0,)
    assert all(i.message for i in report.issues_for(0))


@pytest.mark.parametrize(
    ("label", "old", "new"),
    [
        ("tab before '=' (found by Hypothesis)", b'<error type="grammar"', b'<error type\t="grammar"'),
        ("double space between attributes", b'type="grammar" correction', b'type="grammar"  correction'),
        ("space before '>'", b'correction="the">', b'correction="the" >'),
        ("single quotes", b'correction="the"', b"correction='the'"),
        ("space in closing tag", b"teh</error>", b"teh</error >"),
        ("LF instead of space", b'<error type="grammar"', b'<error\ntype="grammar"'),
    ],
)
def test_semantically_equal_but_non_canonical_tags_are_detected(label, old, new) -> None:
    output = OUT.replace(old, new, 1)
    assert output != OUT, label
    report = v.verify(DOC, output, APPLIED)
    assert codes(report) == ([], {0: [I.TAG_NOT_CANONICAL]}), label  # ElementTree sees identical XML


@pytest.mark.parametrize(
    ("value", "reference", "spelled"),
    [
        ("A & B", b"&amp;", b"&#38;"),  # '&' must be &amp;, not a numeric reference
        ("a\tb", b"&#9;", b"&#09;"),  # no leading zeros
        ("caf" + chr(0xE9), b"&#233;", b"&#x00E9;"),  # decimal only
    ],
)
def test_non_canonical_reference_spellings_are_detected(value, reference, spelled) -> None:
    data = b'<?xml version="1.0" encoding="US-ASCII"?><p>teh</p>'
    out, applied = rendered(data, [Annotation(0, 0, 3, SP, value)])
    assert reference in out and v.verify(data, out, applied).ok
    report = v.verify(data, out.replace(reference, spelled, 1), applied)
    assert codes(report) == ([], {0: [I.TAG_NOT_CANONICAL]})


# --------------------------------------------------------------------------------------------
# Encoding-aware canonical tags (S2.6): the verifier derives the encoding itself (BOM + declaration)
# --------------------------------------------------------------------------------------------

E_ACUTE, SNOWMAN, GRIN = chr(0xE9), chr(0x2603), chr(0x1F600)
NON_ASCII = "caf" + E_ACUTE + " " + SNOWMAN + " " + GRIN
NON_ASCII_REFS = (b"&#233;", b"&#9731;", b"&#128512;")


def _doc(declaration: bytes | None, bom: bool = False) -> bytes:
    decl = b"" if declaration is None else b'<?xml version="1.0" encoding="' + declaration + b'"?>'
    return (b"\xef\xbb\xbf" if bom else b"") + decl + b"<p>teh</p>"


@pytest.mark.parametrize("name", [b"US-ASCII", b"us-ascii", b"Us-Ascii", b"ASCII", b"ascii", b"aScIi"])
def test_us_ascii_documents_accept_decimal_references_and_reject_raw_bytes(name) -> None:
    data = _doc(name)
    out, applied = rendered(data, [Annotation(0, 0, 3, SP, NON_ASCII)])
    assert out.isascii() and all(ref in out for ref in NON_ASCII_REFS)
    assert v.verify(data, out, applied).ok
    for ch, ref in zip((E_ACUTE, SNOWMAN, GRIN), NON_ASCII_REFS):  # same character, raw UTF-8 bytes
        report = v.verify(data, out.replace(ref, ch.encode("utf-8"), 1), applied)
        assert codes(report) == ([], {0: [I.TAG_NOT_CANONICAL]}), ref


@pytest.mark.parametrize(
    ("declaration", "bom"),
    [(b"UTF-8", False), (b"utf-8", False), (b"Utf-8", False), (b"UTF8", False), (b"utf8", False),
     (None, False), (b"UTF-8", True), (b"utf8", True), (None, True)],
)
def test_utf8_documents_accept_raw_characters_and_reject_their_references(declaration, bom) -> None:
    data = _doc(declaration, bom)
    out, applied = rendered(data, [Annotation(0, 0, 3, SP, NON_ASCII)])
    assert NON_ASCII.encode("utf-8") in out and b"&#" not in out
    assert v.verify(data, out, applied).ok
    for ch, ref in zip((E_ACUTE, SNOWMAN, GRIN), NON_ASCII_REFS):  # same character as a reference
        report = v.verify(data, out.replace(ch.encode("utf-8"), ref, 1), applied)
        assert codes(report) == ([], {0: [I.TAG_NOT_CANONICAL]}), ref


@pytest.mark.parametrize(("declaration", "bom"), [(b"US-ASCII", False), (b"UTF-8", False), (None, False), (None, True)])
def test_tab_lf_cr_references_are_canonical_in_both_encodings(declaration, bom) -> None:
    data = _doc(declaration, bom)
    out, applied = rendered(data, [Annotation(0, 0, 3, ST, "a\tb\nc\rd", "x\ty\nz\r")])
    assert out.count(b"&#9;") == 2 and out.count(b"&#10;") == 2 and out.count(b"&#13;") == 2
    assert v.verify(data, out, applied).ok
    for ref, bad in ((b"&#9;", b"&#09;"), (b"&#10;", b"&#xA;"), (b"&#13;", b"&#013;")):
        report = v.verify(data, out.replace(ref, bad, 1), applied)
        assert codes(report) == ([], {0: [I.TAG_NOT_CANONICAL]}), bad


def test_ascii_characters_are_never_spelled_as_references_in_either_encoding() -> None:
    for declaration in (b"US-ASCII", b"UTF-8"):
        data = _doc(declaration)
        out, applied = rendered(data, [Annotation(0, 0, 3, SP, "the")])
        report = v.verify(data, out.replace(b'correction="the"', b'correction="t&#104;e"'), applied)
        assert codes(report) == ([], {0: [I.TAG_NOT_CANONICAL]}), declaration


@pytest.mark.parametrize(
    ("original", "label"),
    [
        (_doc(b"ISO-8859-1"), "unsupported name"),
        (_doc(b"UTF-16"), "UTF-16 declared"),
        (_doc(b"windows-1252"), "another unsupported name"),
        (_doc(b"US-ASCII", bom=True), "BOM contradicts US-ASCII"),
        (_doc(b"ascii", bom=True), "BOM contradicts ascii alias"),
        (_doc(b"US-ASCII").replace(b"teh", b"teh caf\xc3\xa9"), "US-ASCII declared, non-ASCII bytes present"),
    ],
)
def test_unsupported_or_inconsistent_encodings_fail_closed(original: bytes, label: str) -> None:
    applied = [Annotation(0, 0, 3, SP, "the")]
    output = original.replace(b"<p>teh", b'<p><error type="spelling" correction="the">teh</error>', 1)
    report = v.verify(original, output, applied)
    assert codes(report) == ([I.ENCODING_UNSUPPORTED], {}), label  # never treated as UTF-8
    message = report.document_issues[0].message
    assert ("byte-order mark" in message) == label.startswith("BOM"), message  # the precise reason is named


def _utf16_forms():
    """All UTF-16 forms Python's Expat accepts (verified experimentally, even with a UTF-8 override):
    LE/BE x with/without BOM x with/without XML declaration."""
    body, out_body = "<p>teh</p>", '<p><error type="spelling" correction="the">teh</error></p>'
    decl = '<?xml version="1.0" encoding="UTF-16"?>'
    for codec, bom, order in (("utf-16-le", b"\xff\xfe", "LE"), ("utf-16-be", b"\xfe\xff", "BE")):
        for with_bom in (True, False):
            for with_decl in (True, False):
                def enc(text: str) -> bytes:
                    return (bom if with_bom else b"") + ((decl if with_decl else "") + text).encode(codec)
                label = f"UTF-16 {order} {'BOM' if with_bom else 'no BOM'}, {'declaration' if with_decl else 'no declaration'}"
                yield pytest.param(enc(body), enc(out_body), with_bom, id=label)
    for codec, bom, order in (("utf-32-le", b"\xff\xfe\x00\x00", "LE"), ("utf-32-be", b"\x00\x00\xfe\xff", "BE")):
        yield pytest.param(bom + body.encode(codec), bom + out_body.encode(codec), True, id=f"UTF-32 {order} BOM")
        # BOM-less UTF-32: Expat rejects these (invalid token); the raw check names the real reason first
        for with_decl in (True, False):
            d = '<?xml version="1.0" encoding="UTF-32"?>' if with_decl else ""
            yield pytest.param((d + body).encode(codec), (d + out_body).encode(codec), False,
                               id=f"UTF-32 {order} no BOM, {'declaration' if with_decl else 'no declaration'}")


@pytest.mark.parametrize(("original", "output", "has_bom"), list(_utf16_forms()))
def test_utf16_and_utf32_documents_fail_closed_with_or_without_bom_or_declaration(original, output, has_bom) -> None:
    applied = [Annotation(0, 0, 3, SP, "the")]
    report = v.verify(original, output, applied)
    assert codes(report) == ([I.ENCODING_UNSUPPORTED], {})  # document level: never classified as UTF-8
    message = report.document_issues[0].message
    assert ("byte-order mark" if has_bom else "NUL bytes") in message, message
    if has_bom:  # the BOM is named correctly (a UTF-32 LE BOM starts with the UTF-16 LE BOM)
        utf32 = original.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"))
        assert message.startswith("UTF-32" if utf32 else "UTF-16"), message
    assert v.verify(original, original, applied).document_issues[0].code == I.ENCODING_UNSUPPORTED  # even unchanged


@pytest.mark.parametrize(("original", "output", "has_bom"), list(_utf16_forms()))
def test_unsupported_encodings_fail_closed_even_with_no_annotations_and_identical_output(original, output, has_bom) -> None:
    report = v.verify(original, original, [])  # nothing applied, output byte-identical to the input
    assert codes(report) == ([I.ENCODING_UNSUPPORTED], {})
    assert ("byte-order mark" if has_bom else "NUL bytes") in report.document_issues[0].message


@pytest.mark.parametrize("data", [DOC, b'<?xml version="1.0" encoding="US-ASCII"?><p>teh</p>', b"\xef\xbb\xbf<p>teh</p>"])
def test_no_annotation_fast_path_does_not_parse_supported_documents(data, monkeypatch) -> None:
    def refuse(_data):
        raise AssertionError("fast path must not parse")

    monkeypatch.setattr(v, "_parse_tree", refuse)
    monkeypatch.setattr(v, "_record", refuse)
    assert v.verify(data, data, []).ok
    assert codes(v.verify(data, data + b" ", [])) == ([I.UNCHANGED_EXPECTED], {})


@pytest.mark.parametrize(
    "original",
    [
        pytest.param(_doc(b"ISO-8859-1"), id="ISO-8859-1"),
        pytest.param(_doc(b"iso-8859-1"), id="iso-8859-1 lower case"),
        pytest.param(_doc(b"latin1"), id="latin1"),
        pytest.param(_doc(b"Windows-1252"), id="Windows-1252"),
        pytest.param(_doc(b"UTF-16"), id="UTF-16 declared, ASCII-compatible bytes"),
        pytest.param(_doc(b"utf-16le"), id="utf-16le declared"),
        pytest.param(_doc(b"ISO-8859-1", bom=True), id="BOM + ISO-8859-1"),
        pytest.param(_doc(b"US-ASCII", bom=True), id="BOM contradicts US-ASCII"),
        pytest.param(_doc(b"US-ASCII").replace(b"teh", b"caf\xc3\xa9"), id="US-ASCII declared, non-ASCII bytes"),
        pytest.param(b'<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?><p>teh</p>', id="with standalone"),
        pytest.param(b'<?xml\tversion="1.0"\nencoding="Windows-1252"?>\n<p>teh</p>', id="tab/LF inside declaration"),
    ],
)
def test_noop_path_rejects_unsupported_or_inconsistent_declarations(original) -> None:
    report = v.verify(original, original, [])  # nothing applied, output byte-identical
    assert codes(report) == ([I.ENCODING_UNSUPPORTED], {})
    assert report.document_issues[0].message


@pytest.mark.parametrize(
    "original",
    [
        pytest.param(_doc(b"UTF-8"), id="UTF-8"),
        pytest.param(_doc(b"utf8"), id="utf8"),
        pytest.param(_doc(b"UTF-8", bom=True), id="BOM + UTF-8"),
        pytest.param(_doc(b"US-ASCII"), id="US-ASCII"),
        pytest.param(_doc(b"ascii"), id="ascii"),
        pytest.param(b'<?xml version="1.0"?><p>teh</p>', id="declaration without encoding"),
        pytest.param(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?><p>teh</p>', id="with standalone"),
    ],
)
def test_noop_path_accepts_supported_declarations(original) -> None:
    assert v.verify(original, original, []).ok
    assert codes(v.verify(original, original + b"\n", [])) == ([I.UNCHANGED_EXPECTED], {})


@pytest.mark.parametrize(
    "original",
    [
        pytest.param(DOC, id="plain"),
        pytest.param(b"\xef\xbb\xbf<p>teh</p>", id="BOM only"),
        pytest.param(b'<?xml-stylesheet href="a.css"?><p>teh</p>', id="leading xml-stylesheet PI"),
        pytest.param(b"<!-- c --><p>teh</p>", id="leading comment"),
    ],
)
def test_noop_path_without_declaration_parses_nothing(original, monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("the no-op path must not parse a document without a declaration")

    monkeypatch.setattr(v, "_parse_tree", refuse)
    monkeypatch.setattr(v, "_record", refuse)
    monkeypatch.setattr(v.expat, "ParserCreate", refuse)
    assert v.verify(original, original, []).ok
    assert codes(v.verify(original, original + b" ", [])) == ([I.UNCHANGED_EXPECTED], {})


@pytest.mark.parametrize(
    "original",
    [
        pytest.param(b'<?xml version="1.0" encodin="ISO-8859-1"?><p>teh</p>', id="misspelt pseudo-attribute"),
        pytest.param(b'<?xml version="1.0" encoding=""?><p>teh</p>', id="empty encoding"),
        pytest.param(b'<?xml version="1.0" encoding="UTF-8" <p>teh</p>', id="unterminated"),
    ],
)
def test_noop_path_rejects_malformed_declarations(original) -> None:
    assert codes(v.verify(original, original, [])) == ([I.ORIGINAL_UNPARSEABLE], {})


def test_annotate_fails_closed_when_the_verifier_rejects_the_encoding() -> None:
    data = b"<p>teh</p>"
    utf16 = "<p>teh</p>".encode("utf-16-le")
    scan = xs.scan(data)
    outcome = an.annotate(utf16, scan, [Annotation(0, 0, 3, SP, "the")],
                          render_fn=lambda d, spans, codec: (d, ()))  # S1 would reject UTF-16 first; force it through
    assert not outcome.ok and outcome.output is None and outcome.rolled_back == ()
    assert any("ENCODING_UNSUPPORTED (document)" in f for f in outcome.failures)


@pytest.mark.parametrize(
    ("label", "output", "doc_codes"),
    [
        ("not well-formed", OUT.replace(b"</error> parcel", b" parcel"), [I.OUTPUT_NOT_WELL_FORMED]),
        ("paragraph removed", OUT.replace(b"<p>Second <b>bold</b> line.</p>", b""), [I.PARAGRAPH_COUNT_CHANGED]),
        ("paragraph added", OUT.replace(b"</r>", b"<p>new</p></r>"), [I.PARAGRAPH_COUNT_CHANGED]),
        ("other paragraph changed", OUT.replace(b"<b>bold</b>", b"<i>bold</i>"), [I.BYTES_NOT_RESTORED, I.STRUCTURE_CHANGED]),
        ("error outside paragraphs", OUT.replace(b"</p><p>Second", b'</p><error type="spelling" correction="x">z</error><p>Second'),
         [I.ERROR_OUTSIDE_PARAGRAPH, I.BYTES_NOT_RESTORED, I.STRUCTURE_CHANGED]),
    ],
)
def test_document_level_corruptions(label, output, doc_codes) -> None:
    report = v.verify(DOC, output, APPLIED)
    assert not report.ok and codes(report)[0] == doc_codes, label
    assert codes(report)[1] == {} or label == "error outside paragraphs"


def test_unannotated_paragraph_with_text_change_is_reported() -> None:
    report = v.verify(DOC, OUT.replace(b"Second", b"Secnod"), APPLIED)
    doc_codes, para_codes = codes(report)
    assert I.BYTES_NOT_RESTORED in doc_codes and para_codes == {1: [I.TEXT_CHANGED]}


@pytest.mark.parametrize(
    ("original", "annotation", "corrupt"),
    [
        # invisible to ElementTree text comparison; caught by independent byte restoration (G1)
        (b"<p>Salt &amp; pepper teh</p>", Annotation(0, 14, 17, SP, "the"), (b"&amp;", b"&#38;")),
        (b"<p>a<![CDATA[b]]>c teh</p>", Annotation(0, 4, 7, SP, "the"), (b"<![CDATA[b]]>", b"b")),
        (b"<p>teh\r\ncat</p>", Annotation(0, 0, 3, SP, "the"), (b"\r\n", b"\n")),
        (b'<p a="1"  b="2">teh</p>', Annotation(0, 0, 3, SP, "the"), (b'"1"  b', b'"1" b')),
    ],
)
def test_corruptions_invisible_to_elementtree_are_caught_by_byte_restoration(original, annotation, corrupt) -> None:
    output, applied = rendered(original, [annotation])
    assert v.verify(original, output, applied).ok
    report = v.verify(original, output.replace(*corrupt), applied)
    assert not report.ok
    assert I.BYTES_NOT_RESTORED in codes(report)[0] + sum(codes(report)[1].values(), [])


@pytest.mark.parametrize(
    "bad",
    [Annotation(5, 0, 3, SP, "x"), Annotation(-1, 0, 3, SP, "x"), Annotation(True, 0, 3, SP, "x"),  # type: ignore[arg-type]
     Annotation(0, 3, 3, SP, "x"), Annotation(0, 5, 3, SP, "x"), Annotation(0, "a", 3, SP, "x")],  # type: ignore[arg-type]
)
def test_invalid_annotations_are_reported_as_such(bad) -> None:
    report = v.verify(DOC, OUT, [*APPLIED, bad])
    assert codes(report) == ([I.INVALID_ANNOTATION], {})


def test_unhashable_annotation_values_do_not_crash() -> None:
    bad = Annotation(0, 9, 17, SP, ["unhashable"])  # type: ignore[arg-type]
    report = v.verify(DOC, OUT, [*APPLIED, bad, bad])
    assert not report.ok  # an <error> whose correction cannot match a list is reported, not a crash


def test_prefix_bound_on_an_ancestor_is_still_a_namespace_mismatch() -> None:
    data = b'<r xmlns:x="urn:x"><p>teh cat</p></r>'
    output, applied = rendered(data, [Annotation(0, 0, 3, SP, "the")])
    corrupted = output.replace(b"<error ", b"<x:error ").replace(b"</error>", b"</x:error>")  # no xmlns on the tag
    report = v.verify(data, corrupted, applied)
    assert I.NAMESPACE_MISMATCH in codes(report)[1][0]


def test_pre_existing_error_outside_paragraphs_is_not_a_false_positive() -> None:
    data = b'<r><error type="x" correction="y">old</error><p>teh cat</p></r>'
    output, applied = rendered(data, [Annotation(0, 0, 3, SP, "the")])
    assert v.verify(data, output, applied).ok


def test_unverifiable_original() -> None:
    data = (XML / "ent-08-external-dtd-skipped.xml").read_bytes()
    report = v.verify(data, data + b" ", [Annotation(1, 0, 5, SP, "Clean")])
    assert codes(report) == ([I.ORIGINAL_UNPARSEABLE], {})


def test_verifier_is_independent_of_the_renderer_and_source_map() -> None:
    tree = ast.parse(inspect.getsource(v))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported |= {f"{node.module}.{alias.name}" for alias in node.names}
    proofreader_imports = {name for name in imported if name.startswith("proofreader")}
    assert proofreader_imports == {"proofreader.models", "proofreader.models.Annotation", "proofreader.models.Category"}
    assert list(inspect.signature(v.verify).parameters) == ["original", "output", "applied"]  # no ranges, no scan


# --------------------------------------------------------------------------------------------
# Property-based: correct renders verify; any single-byte change is detected
# --------------------------------------------------------------------------------------------

_SKIP = ("bad-", "sec-", "ver-", "ns-03", "enc-08", "enc-09", "enc-10", "enc-11", "enc-12", "enc-13", "ent-08", "ent-09")
_DOCS = [((XML / p.name).read_bytes(), xs.scan((XML / p.name).read_bytes())) for p in sorted(XML.iterdir())
         if not p.name.startswith(_SKIP)]
_DOCS = [(d, s) for d, s in _DOCS if any(p.annotatable for p in s.paragraphs)]
_VALUE = st.text(st.characters(blacklist_categories=("Cs", "Cc")) | st.sampled_from('&<>"\'\n\t'), max_size=8)


@st.composite
def rendered_case(draw):
    data, scan = draw(st.sampled_from(_DOCS))
    targets = [p for p in scan.paragraphs if p.annotatable]
    annotations = []
    for _ in range(draw(st.integers(1, 5))):
        p = draw(st.sampled_from(targets))
        start = draw(st.integers(0, len(p.text) - 1))
        end = draw(st.integers(start + 1, len(p.text)))
        category = draw(st.sampled_from(list(Category)))
        reason = draw(st.one_of(st.none(), _VALUE.filter(lambda x: x.strip())))
        if category is ST and reason is None:
            reason = "Rule"
        annotations.append(Annotation(p.id, start, end, category, draw(_VALUE), reason))
    accepted, _ = an.prepare(data, scan, annotations)
    output, _ = an.render(data, accepted, scan.encoding.codec)
    return data, output, [s.annotation for s in accepted]


SETTINGS = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@SETTINGS
@given(rendered_case())
def test_correct_renders_always_verify(case) -> None:
    data, output, applied = case
    report = v.verify(data, output, applied)
    assert report.ok, codes(report)


@SETTINGS
@given(rendered_case(), st.data())
def test_any_single_byte_change_is_detected(case, draw) -> None:
    data, output, applied = case
    pos = draw.draw(st.integers(0, len(output) - 1))
    op = draw.draw(st.sampled_from(["replace", "delete", "insert"]))
    byte = draw.draw(st.integers(0, 255))
    mutated = bytearray(output)
    if op == "replace":
        if mutated[pos] == byte:
            byte = (byte + 1) % 256
        mutated[pos] = byte
    elif op == "delete":
        del mutated[pos]
    else:
        mutated.insert(pos, byte)
    assert not v.verify(data, bytes(mutated), applied).ok

"""S1 step 3: Expat source mapping, ownership, skip reasons and XML security (xml_source.scan).

All inputs are invented committed fixtures (tests/fixtures/xml) or built inline; nothing here reads
the ignored assessment directory. Rendering/annotation is S2; S1 asserts atom tables, offsets,
skip reasons and typed errors (decision C1).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from proofreader import xml_source as xs
from proofreader.models import (
    AtomKind,
    InputTooLargeError,
    MalformedXMLError,
    ProofreaderError,
    SkipReason,
    UnsupportedEncodingError,
    UnsupportedXMLVersionError,
)

XML = Path(__file__).resolve().parent / "fixtures" / "xml"
LIT, REF, NL, CDATA, BAD = AtomKind.LIT, AtomKind.REF, AtomKind.NL, AtomKind.CDATA, AtomKind.BAD


def fx(name: str) -> bytes:
    return (XML / name).read_bytes()


def scan(data: bytes, **kw):
    before = bytes(data)
    result = xs.scan(data, **kw)
    assert data == before  # source bytes are never modified
    return result


def table(data: bytes, paragraph) -> list[tuple[AtomKind, bytes, str]]:
    return [(a.kind, data[a.raw_start : a.raw_end], a.text) for a in paragraph.atoms]


def only(result):
    assert len(result.paragraphs) == 1
    return result.paragraphs[0]


# --------------------------------------------------------------------------------------------
# Worked examples M1-M13: atom tables (docs/02 section 5)
# --------------------------------------------------------------------------------------------


def test_m01_plain_text_and_byte_offsets() -> None:
    data = fx("m01-plain.xml")
    p = only(scan(data))
    assert table(data, p) == [(LIT, b"We recieved teh parcel.", "We recieved teh parcel.")]
    assert p.skip_reason is None
    assert xs.open_byte(p, 3, "utf-8") == 6 and xs.close_byte(p, 11, "utf-8") == 14


def test_m02_inline_element_splits_atoms_by_container() -> None:
    data = fx("m02-em-hyphen.xml")
    r = scan(data)
    p = only(r)
    assert table(data, p) == [(LIT, b"We saw a ", "We saw a "), (LIT, b"bright", "bright"), (LIT, b"-red kite.", "-red kite.")]
    em = next(e for e in r.elements if e.local_name == "em")
    assert p.atoms[1].container_id == em.id and p.atoms[0].container_id == p.element_id
    assert p.atoms[1].path == (p.element_id, em.id)


def test_m03_predefined_reference_is_one_atomic_ref() -> None:
    data = fx("m03-amp.xml")
    p = only(scan(data))
    assert table(data, p) == [(LIT, b"Salt ", "Salt "), (REF, b"&amp;", "&"), (LIT, b" peper are on teh table.", " peper are on teh table.")]
    assert xs.open_byte(p, 5, "utf-8") == 8 and xs.close_byte(p, 6, "utf-8") == 13  # whole reference


def test_m04_character_references() -> None:
    data = fx("m04-quote-refs.xml")
    p = only(scan(data))
    assert [(k, raw) for k, raw, _ in table(data, p)] == [
        (LIT, b"She wrote "), (REF, b"&#8217;"), (LIT, b"back soon"), (REF, b"&#8217;"), (LIT, b" on teh door.")
    ]
    assert p.text == "She wrote ’back soon’ on teh door."


def test_m05_cdata_atoms() -> None:
    data = fx("m05-cdata.xml")
    p = only(scan(data))
    assert table(data, p) == [(CDATA, b"Teh quick fox", "Teh quick fox")]
    with pytest.raises(xs.MappingRejected) as exc:
        xs.open_byte(p, 0, "utf-8")
    assert exc.value.code == "IN_CDATA"


def test_m06_m07_m08_nested_inline_containers() -> None:
    for name, expected in [
        ("m06-em-compound.xml", [b"Ask the ", b"night-shift", b" porter."]),
        ("m07-end-bold.xml", [b"my pet ", b"dgo"]),
        ("m08-two-level.xml", [b"a ", b"teh", b" cat"]),
    ]:
        data = fx(name)
        p = only(scan(data))
        assert [raw for _, raw, _ in table(data, p)] == expected
        assert all(a.kind is LIT for a in p.atoms) and p.skip_reason is None
    data = fx("m08-two-level.xml")
    p = only(scan(data))
    assert len(p.atoms[1].path) == 3  # p > b > i


def test_m09_comment_and_pi_are_markup_not_text() -> None:
    data = fx("m09-comment-pi.xml")
    p = only(scan(data))
    assert table(data, p) == [(LIT, b"very", "very"), (LIT, b" ", " "), (LIT, b"good", "good")]
    assert p.text == "very good"


def test_m10_crlf_is_an_atomic_nl_atom() -> None:
    data = fx("m10-crlf-gap.xml")
    p = only(scan(data))
    assert table(data, p) == [(LIT, b"ice", "ice"), (NL, b"\r\n", "\n"), (LIT, b"cream cone", "cream cone")]
    assert xs.open_byte(p, 3, "utf-8") == 6 and xs.close_byte(p, 4, "utf-8") == 8  # the whole CRLF


def test_m11_empty_element_inside_paragraph() -> None:
    data = fx("m11-br.xml")
    r = scan(data)
    p = only(r)
    assert table(data, p) == [(LIT, b"ice", "ice"), (LIT, b"cream", "cream")]
    assert p.skip_reason is None  # regression: the zero-width </br> must not look like entity markup
    br = next(e for e in r.elements if e.local_name == "br")
    assert br.start_tag == (6, 11) and br.end_tag == (11, 11)  # zero-width end of <br/>


@pytest.mark.parametrize(
    "data",
    [b"<p><br/>0</p>", b"<p>a<br/>b<hr />c</p>", b"<p><x:img xmlns:x='urn:x' src='a/>'/>text</p>", b"<p>a<br/></p>"],
)
def test_empty_element_tags_do_not_make_paragraph_unmappable(data: bytes) -> None:
    p = only(scan(data))
    assert p.skip_reason is None and all(a.kind is LIT for a in p.atoms)


def test_m12_custom_internal_entity_is_atomic() -> None:
    data = fx("m12-custom-entity.xml")
    p = only(scan(data))
    assert table(data, p) == [(LIT, b"See ", "See "), (REF, b"&co;", "Acme corp"), (LIT, b" now", " now")]
    with pytest.raises(xs.MappingRejected) as exc:
        xs.close_byte(p, 8, "utf-8")  # "Acme" ends inside the reference
    assert exc.value.code == "SPLITS_ATOMIC"
    assert xs.close_byte(p, 13, "utf-8") == data.index(b"&co;") + 4


def test_m13_default_namespace_paragraph() -> None:
    r = scan(fx("m13-default-ns.xml"))
    p = only(r)
    el = r.elements[p.element_id]
    assert (el.namespace, el.local_name, p.text) == ("urn:x", "p", "teh")


# --------------------------------------------------------------------------------------------
# References, entities, multi-callback grouping (R2)
# --------------------------------------------------------------------------------------------


def test_predefined_entities_all_atomic() -> None:
    data = fx("ent-01-predefined.xml")
    p = only(scan(data))
    refs = [(raw, text) for k, raw, text in table(data, p) if k is REF]
    assert refs == [(b"&amp;", "&"), (b"&lt;", "<"), (b"&gt;", ">"), (b"&quot;", '"'), (b"&apos;", "'")]
    assert p.text == "& < > \" ' teh"


def test_decimal_and_hex_references_with_multibyte_results() -> None:
    data = fx("ent-04-char-refs.xml")
    p = only(scan(data))
    refs = [(raw, text) for k, raw, text in table(data, p) if k is REF]
    assert refs == [(b"&#8217;", "’"), (b"&#x2019;", "’")]


def test_astral_and_cr_references_are_not_normalised() -> None:
    data = fx("ent-10-astral-and-cr-refs.xml")
    p = only(scan(data))
    assert table(data, p) == [
        (LIT, b"a", "a"),
        (REF, b"&#128512;", "\U0001F600"),
        (LIT, b"b", "b"),
        (REF, b"&#13;", "\r"),  # a character reference to CR is NOT newline-normalised
        (LIT, b"c", "c"),
        (REF, b"&#xD;", "\r"),
        (REF, b"&#xA;", "\n"),
        (LIT, b"d", "d"),
        (REF, b"&#9;", "\t"),
        (LIT, b"e", "e"),
    ]


@pytest.mark.parametrize(
    ("name", "raw", "text"),
    [
        ("multi-01-lf-in-entity.xml", b"&e;", "l1\nl2"),
        ("multi-02-nested-entity.xml", b"&a;", "x z y"),
        ("multi-03-escaped-lt-entity.xml", b"&e;", "a <b> c"),
    ],
)
def test_callbacks_sharing_one_byte_index_form_one_ref_atom(name: str, raw: bytes, text: str) -> None:
    data = fx(name)
    p = only(scan(data))
    assert table(data, p) == [(LIT, b"[", "["), (REF, raw, text), (LIT, b"]", "]")]
    assert p.skip_reason is None
    # the atom is atomic: no endpoint may fall inside it
    with pytest.raises(xs.MappingRejected) as exc:
        xs.close_byte(p, 2, "utf-8")
    assert exc.value.code == "SPLITS_ATOMIC"


def test_markup_bearing_entity_makes_paragraph_unmappable() -> None:
    data = fx("ent-07-markup-entity.xml")
    p = only(scan(data))
    assert [a.kind for a in p.atoms] == [LIT, BAD, LIT]
    assert p.atoms[1].raw_start == p.atoms[1].raw_end  # zero-width: no source bytes of its own
    assert p.skip_reason is SkipReason.PARA_UNMAPPABLE
    with pytest.raises(xs.MappingRejected) as exc:
        xs.open_byte(p, 4, "utf-8")
    assert exc.value.code == "PARA_UNMAPPABLE"


def test_entity_with_markup_before_text_is_not_mistaken_for_a_ref() -> None:
    # The text callback is the LAST event at the reference's index, so its raw range is the full
    # "&m;" and would look like a clean REF; the shared-index markup events must still make it BAD.
    data = fx("ent-11-markup-then-text-entity.xml")
    p = only(scan(data))
    bad = [a for a in p.atoms if a.kind is BAD]
    assert len(bad) == 1 and data[bad[0].raw_start : bad[0].raw_end] == b"&m;"
    assert p.skip_reason is SkipReason.PARA_UNMAPPABLE


@pytest.mark.parametrize("name", ["ent-08-external-dtd-skipped.xml", "ent-09-external-general.xml"])
def test_skipped_or_external_entities_make_document_unverifiable(name: str) -> None:
    r = scan(fx(name))
    assert [p.skip_reason for p in r.paragraphs] == [SkipReason.DOCUMENT_UNVERIFIABLE] * 2
    assert r.paragraphs[0].text == "teh  end"  # the reference produced no text


def test_external_dtd_is_never_fetched(tmp_path: Path) -> None:
    dtd = tmp_path / "leak.dtd"
    dtd.write_bytes(b'<!ENTITY u "LEAKED-FROM-DTD">')
    data = b'<!DOCTYPE r SYSTEM "' + dtd.as_uri().encode() + b'"><r><p>teh &u; end</p></r>'
    r = scan(data)
    assert "LEAKED" not in r.paragraphs[0].text
    assert r.paragraphs[0].skip_reason is SkipReason.DOCUMENT_UNVERIFIABLE


def test_external_general_entity_is_never_fetched(tmp_path: Path) -> None:
    target = tmp_path / "leak.txt"
    target.write_bytes(b"LEAKED-FROM-FILE")
    data = b'<!DOCTYPE r [<!ENTITY x SYSTEM "' + target.as_uri().encode() + b'">]><r><p>teh &x; end</p></r>'
    r = scan(data)
    assert "LEAKED" not in r.paragraphs[0].text
    assert r.paragraphs[0].skip_reason is SkipReason.DOCUMENT_UNVERIFIABLE


def test_undeclared_entity_without_dtd_is_malformed() -> None:
    with pytest.raises(MalformedXMLError) as exc:
        scan(fx("bad-03-undeclared-entity.xml"))
    assert (exc.value.exit_code, exc.value.line, exc.value.column) == (1, 1, 8)


# --------------------------------------------------------------------------------------------
# CDATA, comments, processing instructions
# --------------------------------------------------------------------------------------------


def test_text_inside_comment_is_not_paragraph_text() -> None:
    data = fx("cmt-02-text-in-comment.xml")
    p = only(scan(data))
    assert p.text == "cat  dog"
    assert b"teh" not in b"".join(raw for _, raw, _ in table(data, p))


def test_processing_instruction_between_text() -> None:
    data = fx("pi-01.xml")
    p = only(scan(data))
    assert table(data, p) == [(LIT, b"a ", "a "), (LIT, b"teh", "teh")]
    assert xs.open_byte(p, 2, "utf-8") == data.index(b"teh")  # after the PI


def test_cdata_boundaries_crlf_and_literal_markup_inside() -> None:
    data = fx("cd-06-cdata-crlf-markup.xml")
    p = only(scan(data))
    assert table(data, p) == [
        (LIT, b"a", "a"),
        (CDATA, b"x&amp;<y>", "x&amp;<y>"),  # CDATA content is literal, not decoded
        (CDATA, b"\r\n", "\n"),
        (CDATA, b"z", "z"),
        (LIT, b"b", "b"),
    ]


def test_cdata_adjacent_and_inside_paragraph_text() -> None:
    p3 = only(scan(fx("cd-03-after-cdata.xml")))
    assert [a.kind for a in p3.atoms] == [CDATA, LIT] and p3.text == "x teh"
    p4 = only(scan(fx("cd-04-cdata-inside.xml")))
    assert [a.kind for a in p4.atoms] == [LIT, CDATA, LIT] and p4.text == "a teh b"


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("m05-cdata.xml", SkipReason.CDATA_ONLY),
        ("cd-07-cdata-only-with-whitespace.xml", SkipReason.CDATA_ONLY),  # whitespace/NL outside CDATA
        ("cd-08-whitespace-only-cdata.xml", SkipReason.EMPTY),  # EMPTY and CDATA_ONLY are exclusive
        ("cd-09-cdata-only-with-error.xml", SkipReason.PARA_HAS_ERROR),  # higher priority wins
        ("cd-04-cdata-inside.xml", None),  # mixed content with literal text stays annotatable
        ("cd-03-after-cdata.xml", None),
        ("cd-10-cdata-plus-reference.xml", None),  # the reference is annotatable text
    ],
)
def test_cdata_only_skip_reason_and_precedence(name: str, reason: SkipReason | None) -> None:
    assert only(scan(fx(name))).skip_reason is reason


def test_cdata_outside_paragraph_is_ignored() -> None:
    data = fx("cd-05-cdata-outside-p.xml")
    p = only(scan(data))
    assert table(data, p) == [(LIT, b"teh", "teh")]


# --------------------------------------------------------------------------------------------
# Line endings, multibyte text, declaration/encoding interaction
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("enc-01-crlf.xml", [(LIT, b"teh", "teh"), (NL, b"\r\n", "\n"), (LIT, b"cat", "cat")]),
        ("enc-03-lone-cr.xml", [(LIT, b"a", "a"), (NL, b"\r", "\n"), (LIT, b"b teh", "b teh")]),
        (
            "enc-04-mixed-eol.xml",
            [
                (LIT, b"one", "one"), (NL, b"\r\n", "\n"), (LIT, b"two", "two"), (LIT, b"\n", "\n"),
                (LIT, b"three", "three"), (NL, b"\r", "\n"), (LIT, b"teh", "teh"),
            ],
        ),
    ],
)
def test_line_ending_atoms(name: str, expected) -> None:
    data = fx(name)
    assert table(data, only(scan(data))) == expected


def test_multibyte_offsets() -> None:
    data = fx("enc-14-multibyte.xml")
    p = only(scan(data))
    assert p.text == "café — 😀 teh"
    teh = p.text.index("teh")
    assert xs.open_byte(p, teh, "utf-8") == data.index(b"teh")
    assert xs.close_byte(p, teh + 3, "utf-8") == data.index(b"teh") + 3
    emoji = p.text.index("😀")
    assert xs.close_byte(p, emoji + 1, "utf-8") - xs.open_byte(p, emoji, "utf-8") == 4


def test_bom_declaration_and_prolog_offsets() -> None:
    data = fx("decl-01-bom-decl-crlf.xml")
    r = scan(data)
    p = only(r)
    assert r.encoding.bom_length == 3
    assert p.atoms[0].raw_start == data.index(b"t\xc3\xa9h")  # offsets count the BOM
    assert table(data, p)[:2] == [(LIT, b"t\xc3\xa9h ", "téh "), (REF, b"&amp;", "&")]


def test_us_ascii_document_with_references() -> None:
    data = fx("decl-02-ascii-refs.xml")
    r = scan(data)
    p = only(r)
    assert r.encoding.codec == "ascii" and p.text == "café & teh"
    assert xs.open_byte(p, p.text.index("teh"), r.encoding.codec) == data.index(b"teh")


def test_utf8_alias_declaration() -> None:
    p = only(scan(fx("decl-03-utf8-alias.xml")))
    assert p.text == "naïve teh" and p.skip_reason is None


@pytest.mark.parametrize(
    ("name", "error"),
    [
        ("enc-09a-utf16le-bom.xml", UnsupportedEncodingError),
        ("enc-11a-decl-iso-8859-1.xml", UnsupportedEncodingError),
        ("ver-xml-1.1.xml", UnsupportedXMLVersionError),
    ],
)
def test_encoding_policy_applies_before_parsing(name: str, error: type[ProofreaderError]) -> None:
    with pytest.raises(error) as exc:
        scan(fx(name))
    assert exc.value.exit_code == 2


def test_size_cap_applies() -> None:
    with pytest.raises(InputTooLargeError):
        scan(b"<p>" + b"x" * 64 + b"</p>", max_bytes=32)


def test_long_paragraph_maps_exactly() -> None:
    body = ("é" * 70000 + "&amp;" + "x" * 70000).encode()
    data = b"<p>" + body + b"</p>"
    p = only(scan(data))
    assert p.skip_reason is None and len(p.text) == 140001
    assert p.atoms[0].raw_start == 3 and p.atoms[-1].raw_end == len(data) - 4
    assert any(a.kind is REF for a in p.atoms)


# --------------------------------------------------------------------------------------------
# Paragraph discovery, namespaces, nesting, existing errors, empty paragraphs
# --------------------------------------------------------------------------------------------


def test_document_order_and_namespaces() -> None:
    r = scan(fx("order-01-mixed-ns.xml"))
    assert [p.text for p in r.paragraphs] == ["one", "two", "three"]
    assert [r.elements[p.element_id].namespace for p in r.paragraphs] == [None, "urn:a", None]
    assert [p.id for p in r.paragraphs] == [0, 1, 2]


def test_prefixed_paragraph_matched_by_local_name() -> None:
    r = scan(fx("ns-02-prefixed.xml"))
    p = only(r)
    assert (r.elements[p.element_id].namespace, p.text) == ("urn:x", "teh")


def test_unbound_prefix_is_malformed() -> None:
    with pytest.raises(MalformedXMLError) as exc:
        scan(fx("ns-03-unbound-prefix.xml"))
    assert exc.value.exit_code == 1


def test_nested_paragraph_ownership_and_barrier() -> None:
    data = fx("nest-01.xml")
    outer, inner = scan(data).paragraphs
    assert outer.text == "Outer teh  more teh" and inner.text == "Inner teh"
    assert table(data, outer) == [(LIT, b"Outer teh ", "Outer teh "), (LIT, b" more teh", " more teh")]
    assert outer.barriers == (10,)
    assert (outer.atoms[0].raw_start, outer.atoms[0].raw_end, outer.atoms[1].raw_start) == (8, 18, 34)
    assert (inner.atoms[0].raw_start, inner.atoms[0].raw_end) == (21, 30)
    assert outer.skip_reason is None and inner.skip_reason is None


def test_nested_paragraph_inside_inline_element() -> None:
    outer, inner = scan(fx("nest-03-inside-inline.xml")).paragraphs
    assert (outer.text, outer.barriers, inner.text) == ("a x  y b", (4,), "teh")


def test_deeply_nested_paragraphs_each_own_their_text() -> None:
    a, b, c = scan(fx("nest-04-deep.xml")).paragraphs
    assert (a.text, b.text, c.text) == ("A  a", "B  b", "C")
    assert (a.barriers, b.barriers) == ((2,), (2,))
    assert all(p.skip_reason is None for p in (a, b, c))


def test_existing_error_marks_paragraph() -> None:
    p = only(scan(fx("err-01-existing-error.xml")))
    assert p.skip_reason is SkipReason.PARA_HAS_ERROR


def test_error_in_nested_paragraph_marks_all_enclosing_paragraphs() -> None:
    outer, inner, clean = scan(fx("err-02-error-in-nested.xml")).paragraphs
    assert outer.skip_reason is SkipReason.PARA_HAS_ERROR
    assert inner.skip_reason is SkipReason.PARA_HAS_ERROR
    assert clean.skip_reason is None


def test_empty_and_whitespace_only_paragraphs() -> None:
    r = scan(fx("empty-01.xml"))
    assert [p.skip_reason for p in r.paragraphs] == [SkipReason.EMPTY] * 3 + [None]
    empty_el = r.elements[r.paragraphs[0].element_id]
    assert r.paragraphs[0].atoms == ()
    assert empty_el.start_tag == (3, 7) and empty_el.end_tag == (7, 7)  # <p/>: zero-width end after the tag


# --------------------------------------------------------------------------------------------
# Malformed XML, security
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "line", "column", "byte"),
    [
        ("bad-01-mismatched.xml", 3, 9, 22),  # Expat reports the position after "</"
        ("bad-02-two-roots.xml", 1, 11, 10),
    ],
)
def test_malformed_xml_reports_line_and_column(name: str, line: int, column: int, byte: int) -> None:
    with pytest.raises(MalformedXMLError) as exc:
        scan(fx(name))
    err = exc.value
    assert (err.exit_code, err.line, err.column, err.byte_offset) == (1, line, column, byte)
    assert f"line {line}" in err.message


def _entity_doc(value_len: int, refs: int) -> bytes:
    return b'<!DOCTYPE p [<!ENTITY e "' + b"x" * value_len + b'">]><p>' + b"&e;" * refs + b"</p>"


def test_expansion_guard_default_value() -> None:
    assert xs.MAX_ENTITY_EXPANSION_CHARS == 1 << 20


def test_legitimate_entity_expansion_under_default_limit_is_accepted() -> None:
    data = _entity_doc(1000, 900)  # 900,000 expanded characters < input size + 1 MiB
    p = only(scan(data))
    assert p.skip_reason is None and len(p.text) == 900_000
    assert sum(a.kind is REF for a in p.atoms) == 900


def test_expansion_guard_exact_boundary() -> None:
    data = _entity_doc(100, 50)  # delivers exactly 5,000 characters
    limit_at_boundary = 5000 - len(data)  # budget = len(data) + limit == delivered -> allowed
    assert only(scan(data, expansion_limit=limit_at_boundary)).skip_reason is None
    with pytest.raises(MalformedXMLError, match="expansion limit") as exc:
        scan(data, expansion_limit=limit_at_boundary - 1)
    assert exc.value.exit_code == 1


def test_billion_laughs_stopped_by_early_guard() -> None:
    with pytest.raises(MalformedXMLError, match="expansion limit"):
        scan(fx("sec-01-billion-laughs.xml"))


def test_billion_laughs_also_stopped_by_expat_protection() -> None:
    with pytest.raises(MalformedXMLError, match="amplification"):
        scan(fx("sec-01-billion-laughs.xml"), expansion_limit=None)


def test_expat_version_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    xs.check_expat_version()  # the installed Expat passes
    monkeypatch.setattr(xs.expat, "version_info", (2, 3, 9))
    with pytest.raises(ProofreaderError, match="2.4.0"):
        xs.scan(b"<p>teh</p>")


# --------------------------------------------------------------------------------------------
# Independent self-checks veto paragraphs / documents (R5)
# --------------------------------------------------------------------------------------------


def test_own_text_mismatch_vetoes_paragraph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xs, "_elementtree_own_texts", lambda data: ["different"])
    assert only(xs.scan(b"<p>teh</p>")).skip_reason is SkipReason.PARA_UNMAPPABLE


@pytest.mark.parametrize("fake", [None, [], ["teh", "extra"]])
def test_unavailable_or_inconsistent_cross_check_makes_document_unverifiable(monkeypatch, fake) -> None:
    monkeypatch.setattr(xs, "_elementtree_own_texts", lambda data: fake)
    assert only(xs.scan(b"<p>teh</p>")).skip_reason is SkipReason.DOCUMENT_UNVERIFIABLE


def test_lexical_disagreement_vetoes_paragraph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xs, "_lex", lambda data, start, end: None)
    assert only(xs.scan(b"<p>teh</p>")).skip_reason is SkipReason.PARA_UNMAPPABLE


def test_tiling_gap_vetoes_paragraph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xs, "_tiles", lambda *a: False)
    assert only(xs.scan(b"<p>teh</p>")).skip_reason is SkipReason.PARA_UNMAPPABLE


# --------------------------------------------------------------------------------------------
# Invariants over every parseable committed fixture
# --------------------------------------------------------------------------------------------

PARSEABLE = sorted(
    n
    for n in (p.name for p in XML.iterdir())
    if not n.startswith(("bad-", "sec-", "ver-", "ns-03", "enc-08", "enc-09", "enc-10", "enc-11", "enc-12", "enc-13"))
)


@pytest.mark.parametrize("name", PARSEABLE)
def test_atom_invariants(name: str) -> None:
    data = fx(name)
    r = scan(data)
    for p in r.paragraphs:
        pos = 0
        last_end = -1
        for a in p.atoms:
            assert a.char_start == pos and a.char_end == pos + len(a.text)
            pos = a.char_end
            assert a.raw_start >= last_end  # ordered, non-overlapping source ranges
            last_end = a.raw_end
            raw = data[a.raw_start : a.raw_end]
            if a.kind is LIT:
                assert raw.decode(r.encoding.codec) == a.text
            elif a.kind is NL:
                assert raw in (b"\r\n", b"\r") and a.text == "\n"
            elif a.kind is REF:
                assert raw.startswith(b"&") and raw.endswith(b";")
        assert "".join(a.text for a in p.atoms) == p.text
        assert all(0 <= b <= len(p.text) for b in p.barriers)

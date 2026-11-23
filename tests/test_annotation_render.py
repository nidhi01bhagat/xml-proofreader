"""S2.3: deterministic rendering of validated spans (byte insertion only; no LLM, no verification yet).

Rendered output is checked byte-for-byte against the input outside the inserted tags, and parsed
independently with ElementTree. Special characters are built with chr() (no invisible literals).
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from proofreader import annotation as an
from proofreader import xml_source as xs
from proofreader.models import Annotation, Category

XML = Path(__file__).resolve().parent / "fixtures" / "xml"
SP, GR, PU, CA, CL, ST = (Category.SPELLING, Category.GRAMMAR, Category.PUNCTUATION,
                          Category.CAPITALIZATION, Category.CLARITY, Category.STYLEGUIDE)
RSQUO = chr(0x2019)
DOC = b"<p>The firm recieved teh parcel on monday.</p>"  # recieved [9,17) teh [18,21) monday [32,38)


def strip(output: bytes, ranges) -> bytes:
    """Remove inserted tag bytes (G1): must give back the original input exactly."""
    kept, prev = bytearray(), 0
    for s, e in ranges:
        kept += output[prev:s]
        prev = e
    return bytes(kept + output[prev:])


def render_doc(data: bytes, annotations):
    scan = xs.scan(data)
    accepted, rejected = an.prepare(data, scan, annotations)
    assert rejected == [], rejected
    output, ranges = an.render(data, accepted, scan.encoding.codec)
    assert strip(output, ranges) == data  # every original byte kept, in order, unchanged
    return output, ranges, scan, accepted


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse(output: bytes) -> ET.Element:
    # Explicit UTF-8 override, as in xml_source: inputs are UTF-8 or pure ASCII (encoding.check), and
    # plain ElementTree cannot decode non-ASCII text under an encoding="utf8" alias declaration.
    return ET.fromstring(output, parser=ET.XMLParser(encoding="utf-8"))


def errors_of(output: bytes):
    return [e for e in parse(output).iter() if local(e.tag) == "error"]


# --------------------------------------------------------------------------------------------
# Single, multiple, adjacent annotations; multiple paragraphs
# --------------------------------------------------------------------------------------------


def test_single_annotation_exact_bytes() -> None:
    output, ranges, _, _ = render_doc(DOC, [Annotation(0, 9, 17, SP, "received")])
    assert output == b'<p>The firm <error type="spelling" correction="received">recieved</error> teh parcel on monday.</p>'
    assert ranges == ((12, 57), (65, 73))


def test_multiple_annotations_in_one_paragraph() -> None:
    output, _, _, _ = render_doc(DOC, [Annotation(0, 18, 21, GR, "the"), Annotation(0, 9, 17, SP, "received")])
    assert output == (
        b'<p>The firm <error type="spelling" correction="received">recieved</error> '
        b'<error type="grammar" correction="the">teh</error> parcel on monday.</p>'
    )


def test_adjacent_annotations_close_before_open() -> None:
    output, _, _, _ = render_doc(DOC, [Annotation(0, 9, 17, SP, "received"), Annotation(0, 17, 18, PU, ", "),
                                       Annotation(0, 18, 21, GR, "the")])
    assert b'recieved</error><error type="punctuation" correction=", "> </error><error type="grammar"' in output
    assert [e.text for e in errors_of(output)] == ["recieved", " ", "teh"]


def test_annotations_at_paragraph_edges() -> None:
    output, _, _, _ = render_doc(DOC, [Annotation(0, 0, 3, CA, "THE"), Annotation(0, 38, 39, PU, "!")])
    assert output.startswith(b'<p><error type="capitalization" correction="THE">The</error> firm')
    assert output.endswith(b'monday<error type="punctuation" correction="!">.</error></p>')


def test_multiple_paragraphs_and_namespaces() -> None:
    data = (XML / "order-01-mixed-ns.xml").read_bytes()
    output, _, _, _ = render_doc(data, [Annotation(2, 0, 5, SP, "Three"), Annotation(0, 0, 3, SP, "One"),
                                        Annotation(1, 0, 3, SP, "Two")])
    assert output == data.replace(b"<p>one</p>", b'<p><error type="spelling" correction="One">one</error></p>').replace(
        b"<a:p>two</a:p>", b'<a:p><error type="spelling" correction="Two">two</error></a:p>').replace(
        b"<p>three</p>", b'<p><error type="spelling" correction="Three">three</error></p>')
    assert b'xmlns:a="urn:a"' in output and b"<h>no</h>" in output


def test_nested_paragraphs_are_annotated_independently() -> None:
    data = (XML / "nest-01.xml").read_bytes()
    output, _, _, _ = render_doc(data, [Annotation(1, 6, 9, SP, "the"), Annotation(0, 6, 9, SP, "the"),
                                        Annotation(0, 16, 19, SP, "the")])
    assert output == (
        b'<doc><p>Outer <error type="spelling" correction="the">teh</error> <p>Inner '
        b'<error type="spelling" correction="the">teh</error></p> more '
        b'<error type="spelling" correction="the">teh</error></p></doc>'
    )


# --------------------------------------------------------------------------------------------
# Categories, reasons, attribute escaping, deletions
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("category", list(Category))
def test_every_category_renders_its_lowercase_type(category: Category) -> None:
    reason = "Rule 4.2" if category is ST else None
    output, _, _, _ = render_doc(DOC, [Annotation(0, 9, 17, category, "received", reason)])
    tag = f'<error type="{category.value}" correction="received"'.encode()
    assert tag in output
    assert (b'reason="Rule 4.2"' in output) is (category is ST)


def test_reason_included_only_when_supplied() -> None:
    with_reason, _, _, _ = render_doc(DOC, [Annotation(0, 9, 17, SP, "received", "Common misspelling")])
    without, _, _, _ = render_doc(DOC, [Annotation(0, 9, 17, SP, "received")])
    assert b'<error type="spelling" correction="received" reason="Common misspelling">' in with_reason
    assert b"reason=" not in without


@pytest.mark.parametrize(
    ("value", "escaped"),
    [
        ('say "hi"', b"say &quot;hi&quot;"),
        ("R&D", b"R&amp;D"),
        ("a <b> c", b"a &lt;b&gt; c"),
        ("it's", b"it's"),  # apostrophe is legal inside a double-quoted attribute
        ("tab\tnl\ncr\r", b"tab&#9;nl&#10;cr&#13;"),
        ("&amp;", b"&amp;amp;"),  # already-escaped-looking text is escaped again, never interpreted
    ],
)
def test_attribute_escaping(value: str, escaped: bytes) -> None:
    output, _, _, _ = render_doc(DOC, [Annotation(0, 9, 17, ST, value, value)])
    assert b'correction="' + escaped + b'" reason="' + escaped + b'">' in output
    (error,) = errors_of(output)
    assert error.get("correction") == value and error.get("reason") == value  # exact round-trip


def test_unicode_attribute_values_in_utf8_document() -> None:
    value = "caf" + chr(0xE9) + " " + RSQUO + " " + chr(0x1F600) + " " + chr(0x65E5)
    output, _, _, _ = render_doc(DOC, [Annotation(0, 9, 17, CL, value, value)])
    assert ('correction="' + value + '"').encode("utf-8") in output  # literal UTF-8, no references
    assert errors_of(output)[0].get("reason") == value


def test_unicode_attribute_values_in_us_ascii_document_become_references() -> None:
    data = (XML / "enc-07-us-ascii.xml").read_bytes()
    value = "don" + RSQUO + "t " + chr(0x1F600)
    output, _, _, _ = render_doc(data, [Annotation(0, 0, 4, PU, value, value)])
    assert output.isascii()
    assert b'correction="don&#8217;t &#128512;" reason="don&#8217;t &#128512;">dont</error>' in output
    assert errors_of(output)[0].get("correction") == value


def test_empty_correction_for_deletion() -> None:
    output, _, _, _ = render_doc(DOC, [Annotation(0, 17, 21, GR, "")])
    assert b'<error type="grammar" correction=""> teh</error>' in output


# --------------------------------------------------------------------------------------------
# Encodings, BOM, declarations, prolog, mixed content, CDATA, comments, PIs
# --------------------------------------------------------------------------------------------


def test_utf8_bom_declaration_prolog_and_crlf_preserved() -> None:
    data = (XML / "decl-01-bom-decl-crlf.xml").read_bytes()
    output, ranges, scan, _ = render_doc(data, [Annotation(0, 0, 3, SP, "teh")])
    assert output.startswith(b"\xef\xbb\xbf<?xml version=\"1.0\" encoding=\"UTF-8\"?>\r\n<!--c-->\r\n<p>")
    assert ranges[0][0] == data.index(b"t\xc3\xa9h")  # insertion point counts the BOM


def test_multibyte_text_and_references_inside_spans() -> None:
    data = (XML / "enc-14-multibyte.xml").read_bytes()  # <p>café — 😀 teh</p>
    text = xs.scan(data).paragraphs[0].text
    output, _, _, _ = render_doc(data, [Annotation(0, 0, 4, SP, "cafe"), Annotation(0, text.index("teh"), len(text), SP, "the")])
    assert errors_of(output)[0].text == "caf" + chr(0xE9) and errors_of(output)[1].text == "teh"
    data = (XML / "m03-amp.xml").read_bytes()
    output, _, _, _ = render_doc(data, [Annotation(0, 0, 12, SP, "Salt & pepper")])
    assert b'correction="Salt &amp; pepper">Salt &amp; peper</error>' in output  # reference bytes untouched


def test_crlf_inside_span_is_preserved() -> None:
    data = (XML / "m10-crlf-gap.xml").read_bytes()
    output, _, _, _ = render_doc(data, [Annotation(0, 0, 9, SP, "ice cream")])
    assert output == b'<p><error type="spelling" correction="ice cream">ice\r\ncream</error> cone</p>'


def test_mixed_content_namespaces_comments_pis_and_cdata_untouched() -> None:
    data = (
        b'<?xml version="1.0"?>\n<!-- head --><?app cfg?>\n'
        b'<r xmlns="urn:d" xmlns:x="urn:x" x:lang="en">'
        b'<p x:id="1">A <em class="k">teh</em> <!--c--> word<?pi y?> here <![CDATA[raw & <cdata>]]> end</p>'
        b"<x:note>keep</x:note></r>"
    )
    scan = xs.scan(data)
    text = scan.paragraphs[0].text
    output, ranges, _, _ = render_doc(data, [Annotation(0, text.index("teh"), text.index("teh") + 3, SP, "the"),
                                             Annotation(0, text.index("word"), text.index("word") + 4, GR, "words"),
                                             Annotation(0, text.index("end"), text.index("end") + 3, CA, "End")])
    assert b'<em class="k"><error type="spelling" correction="the">teh</error></em>' in output
    assert b"<![CDATA[raw & <cdata>]]>" in output and b"<!--c-->" in output and b"<?pi y?>" in output
    assert output.startswith(data[: data.index(b"<p ")]) and output.endswith(b"<x:note>keep</x:note></r>")
    root = parse(output)
    assert [local(e.tag) for e in root.iter() if local(e.tag) == "error"] == ["error"] * 3
    # Note: an un-prefixed <error> inside a default namespace inherits it (documented risk)
    assert all(e.tag == "{urn:d}error" for e in errors_of(output))


# --------------------------------------------------------------------------------------------
# Byte-for-byte preservation, determinism
# --------------------------------------------------------------------------------------------


def test_untouched_regions_are_byte_identical() -> None:
    annotations = [Annotation(0, 9, 17, SP, "received"), Annotation(0, 32, 38, ST, "Monday", "Capitalise days")]
    output, ranges, _, accepted = render_doc(DOC, annotations)
    offsets = sorted({o for s in accepted for o in (s.open_offset, s.close_offset)})
    pieces = [DOC[a:b] for a, b in zip([0, *offsets], [*offsets, len(DOC)])]
    stripped_pieces, prev = [], 0
    for s, e in ranges:
        stripped_pieces.append(output[prev:s])
        prev = e
    stripped_pieces.append(output[prev:])
    assert [p for p in stripped_pieces if p] == [p for p in pieces if p]


def test_rendering_is_independent_of_input_order() -> None:
    pool = [Annotation(0, 0, 3, CA, "THE"), Annotation(0, 9, 17, SP, "received"), Annotation(0, 17, 18, PU, ","),
            Annotation(0, 18, 21, GR, "the"), Annotation(0, 32, 38, ST, "Monday", "Rule"), Annotation(0, 38, 39, PU, "!")]
    scan = xs.scan(DOC)
    accepted, _ = an.prepare(DOC, scan, pool)
    expected = an.render(DOC, accepted, "utf-8")
    rng = random.Random(42)
    for _ in range(50):
        shuffled = accepted[:]
        rng.shuffle(shuffled)
        assert an.render(DOC, shuffled, "utf-8") == expected
        pool_shuffled = pool[:]
        rng.shuffle(pool_shuffled)
        assert an.render(DOC, an.prepare(DOC, scan, pool_shuffled)[0], "utf-8") == expected


def test_no_spans_returns_input_unchanged() -> None:
    assert an.render(DOC, [], "utf-8") == (DOC, ())


def test_exact_duplicate_safe_spans_render_once() -> None:
    scan = xs.scan(DOC)
    (span,), _ = an.prepare(DOC, scan, [Annotation(0, 9, 17, SP, "received")])
    assert an.render(DOC, [span, span], "utf-8") == an.render(DOC, [span], "utf-8")


# --------------------------------------------------------------------------------------------
# Invalid input is refused, never repaired
# --------------------------------------------------------------------------------------------


def _safe(a: Annotation, open_offset: int, close_offset: int) -> an.SafeSpan:
    return an.SafeSpan(a, open_offset, close_offset)


GOOD = Annotation(0, 9, 17, SP, "received")


@pytest.mark.parametrize(
    "spans",
    [
        [GOOD],  # a bare Annotation, not a SafeSpan
        [_safe(GOOD, 12, 20), _safe(Annotation(0, 13, 17, GR, "x"), 16, 20)],  # overlapping
        [_safe(GOOD, 12, 20), _safe(Annotation(0, 9, 17, GR, "x"), 12, 20)],  # same range, different annotation
        [_safe(GOOD, 12, 20), _safe(Annotation(0, 9, 21, GR, "x"), 12, 24)],  # nested (same start)
        [_safe(GOOD, 20, 12)],  # reversed
        [_safe(GOOD, 12, 12)],  # empty
        [_safe(GOOD, -1, 12)],  # negative
        [_safe(GOOD, 12, len(DOC) + 1)],  # past the end
        [_safe(GOOD, 12.0, 20)],  # non-int offset
        [_safe(Annotation(0, 9, 17, SP, "re" + chr(0x01) + "ceived"), 12, 20)],  # XML-illegal correction
        [_safe(Annotation(0, 9, 17, SP, "received", "why" + chr(0xFFFE)), 12, 20)],  # XML-illegal reason
        [_safe(Annotation(0, 9, 17, SP, "received", "   "), 12, 20)],  # whitespace-only reason
        [_safe(Annotation(0, 9, 17, ST, "received"), 12, 20)],  # styleguide without reason
        [_safe(Annotation(0, 9, 17, "spelling", "received"), 12, 20)],  # category not a Category
        [_safe(Annotation(0, 9, 17, SP, ["received"]), 12, 20)],  # correction not a str
    ],
)
def test_invalid_render_input_is_refused(spans) -> None:
    with pytest.raises(an.RenderInputError):
        an.render(DOC, spans, "utf-8")


def test_same_byte_range_for_different_annotations_has_a_precise_error() -> None:
    spans = [_safe(GOOD, 12, 20), _safe(Annotation(0, 9, 17, GR, "x"), 12, 20)]
    with pytest.raises(an.RenderInputError, match="share one byte range"):
        an.render(DOC, spans, "utf-8")


def test_one_byte_overlap_is_refused_but_adjacent_ranges_render() -> None:
    other = Annotation(0, 18, 21, GR, "the")
    with pytest.raises(an.RenderInputError, match="overlapping byte ranges"):
        an.render(DOC, [_safe(GOOD, 12, 20), _safe(other, 19, 24)], "utf-8")
    output, ranges = an.render(DOC, [_safe(GOOD, 12, 20), _safe(other, 20, 24)], "utf-8")
    assert len(ranges) == 4 and output.count(b"</error><error") == 1


def test_existing_error_paragraph_untouched_while_clean_paragraph_is_annotated() -> None:
    data = (XML / "err-02-error-in-nested.xml").read_bytes()
    output, _, scan, _ = render_doc(data, [Annotation(2, 0, 5, SP, "Clean")])
    assert output.replace(b'<error type="spelling" correction="Clean">clean</error>', b"clean") == data
    assert output.count(b"<error") == data.count(b"<error") + 1


def test_render_requires_bytes() -> None:
    with pytest.raises(an.RenderInputError):
        an.render(DOC.decode(), [], "utf-8")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------
# Property-based: random valid documents/spans -> render -> independent parse
# --------------------------------------------------------------------------------------------

_SKIP = ("bad-", "sec-", "ver-", "ns-03", "enc-08", "enc-09", "enc-10", "enc-11", "enc-12", "enc-13", "ent-08", "ent-09")
_DOCS = [((XML / p.name).read_bytes(), xs.scan((XML / p.name).read_bytes())) for p in sorted(XML.iterdir())
         if not p.name.startswith(_SKIP)]
# Documents with pre-existing <error> elements are excluded here (their skipped paragraphs are
# covered by dedicated tests) so every <error> in the output is one this test inserted.
_DOCS = [(d, s) for d, s in _DOCS if any(p.annotatable for p in s.paragraphs) and b"<error" not in d]
_VALUE_CHARS = st.characters(blacklist_categories=("Cs",), blacklist_characters="".join(
    chr(c) for c in list(range(0x00, 0x09)) + [0x0B, 0x0C] + list(range(0x0E, 0x20)) + [0xFFFE, 0xFFFF]))
_VALUES = st.one_of(st.text(_VALUE_CHARS, max_size=12), st.sampled_from(['"', "&", "<>", "'", "\n\t\r", "&amp;"]))


@st.composite
def doc_with_annotations(draw):
    data, scan = draw(st.sampled_from(_DOCS))
    targets = [p for p in scan.paragraphs if p.annotatable]
    annotations = []
    for _ in range(draw(st.integers(1, 6))):
        p = draw(st.sampled_from(targets))
        start = draw(st.integers(0, len(p.text) - 1))
        end = draw(st.integers(start + 1, len(p.text)))
        category = draw(st.sampled_from(list(Category)))
        reason = draw(st.one_of(st.none(), _VALUES.filter(lambda v: v.strip())))
        if category is ST and reason is None:
            reason = "Rule"
        annotations.append(Annotation(p.id, start, end, category, draw(_VALUES), reason))
    return data, scan, annotations


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(doc_with_annotations(), st.randoms(use_true_random=False))
def test_render_properties(case, rnd) -> None:
    data, scan, annotations = case
    accepted, _ = an.prepare(data, scan, annotations)
    output, ranges = an.render(data, accepted, scan.encoding.codec)
    assert strip(output, ranges) == data  # G1
    assert len(ranges) == 2 * len(accepted)
    shuffled = accepted[:]
    rnd.shuffle(shuffled)
    assert an.render(data, shuffled, scan.encoding.codec) == (output, ranges)  # deterministic
    if not accepted:
        assert output == data
        return
    errors = errors_of(output)  # independent parse of the rendered document
    by_bytes = sorted(accepted, key=lambda s: s.open_offset)
    assert len(errors) == len(by_bytes)
    for e, s in zip(errors, by_bytes):
        a = s.annotation
        assert "".join(e.itertext()) == scan.paragraphs[a.paragraph_id].text[a.start : a.end]
        assert list(e.attrib) == ["type", "correction"] + (["reason"] if a.reason is not None else [])
        assert e.get("type") == a.category.value and e.get("correction") == a.correction
        assert e.get("reason") == a.reason

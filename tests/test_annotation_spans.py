"""S2.2: span safety (X1-X8, X10 with D4 OFF) and deterministic overlap handling.

Synthetic, resolved annotations on invented fixtures only; no rendering (S2.3) and no LLM.
Spike cases M1-M13 are ported with the approved D4-off expectations (docs/02 section 5).
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
from proofreader.models import Annotation, AtomKind, Category, RejectCode, SkipReason

XML = Path(__file__).resolve().parent / "fixtures" / "xml"
SP, GR, ST = Category.SPELLING, Category.GRAMMAR, Category.STYLEGUIDE
OK = "OK"


def load(name: str):
    data = (XML / name).read_bytes()
    return data, xs.scan(data)


def check(data, scan, pid, start, end, category=SP, correction="X", reason=None):
    return an.check_span(data, scan, Annotation(pid, start, end, category, correction, reason))


def outcome(name: str, pid: int, start: int, end: int):
    data, scan = load(name)
    result = check(data, scan, pid, start, end)
    if isinstance(result, an.SafeSpan):
        return OK, data[result.open_offset : result.close_offset]
    return result.code, None


# --------------------------------------------------------------------------------------------
# Spike cases M1-M13 with D4-off expectations (docs/02 section 5)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "pid", "start", "end", "expected", "raw"),
    [
        ("m01-plain.xml", 0, 3, 11, OK, b"recieved"),  # X1
        ("m02-em-hyphen.xml", 0, 9, 19, RejectCode.CROSSES_ELEMENT, None),  # M2a: needs expansion
        ("m02-em-hyphen.xml", 0, 9, 15, OK, b"bright"),  # M2b: inside <em>
        ("m03-amp.xml", 0, 0, 12, OK, b"Salt &amp; peper"),  # M3a: X2 reference covered whole
        ("m03-amp.xml", 0, 5, 6, OK, b"&amp;"),  # M3b: the reference alone
        ("m04-quote-refs.xml", 0, 4, 9, OK, b"wrote"),  # M4 (quotation policy is S3)
        ("m05-cdata.xml", 0, 0, 3, RejectCode.PARA_NOT_ANNOTATABLE, None),  # M5: CDATA_ONLY paragraph
        ("m06-em-compound.xml", 0, 8, 19, OK, b"night-shift"),  # M6a
        ("m06-em-compound.xml", 0, 8, 26, RejectCode.CROSSES_ELEMENT, None),  # M6b
        ("m06-em-compound.xml", 0, 4, 13, RejectCode.CROSSES_ELEMENT, None),  # M6c
        ("m07-end-bold.xml", 0, 3, 10, RejectCode.CROSSES_ELEMENT, None),  # M7
        ("m08-two-level.xml", 0, 2, 9, RejectCode.CROSSES_ELEMENT, None),  # M8
        ("m09-comment-pi.xml", 0, 0, 9, RejectCode.CROSSES_ELEMENT, None),  # M9: X8 comment + PI
        ("m10-crlf-gap.xml", 0, 0, 9, OK, b"ice\r\ncream"),  # M10: CRLF kept inside the span
        ("m11-br.xml", 0, 0, 8, RejectCode.CROSSES_ELEMENT, None),  # M11: empty element <br/>
        ("m12-custom-entity.xml", 0, 4, 8, RejectCode.SPLITS_ATOMIC, None),  # M12: "Acme"
        ("m12-custom-entity.xml", 0, 4, 13, OK, b"&co;"),  # M12: "Acme corp" = whole reference
        ("m13-default-ns.xml", 0, 0, 3, OK, b"teh"),  # M13: default namespace
    ],
)
def test_spike_cases_with_d4_off(name, pid, start, end, expected, raw) -> None:
    assert outcome(name, pid, start, end) == (expected, raw)


def test_us_ascii_document_maps_with_ascii_codec() -> None:
    data, scan = load("enc-07-us-ascii.xml")
    result = check(data, scan, 0, 0, 4, Category.PUNCTUATION, "don’t")
    assert isinstance(result, an.SafeSpan) and data[result.open_offset : result.close_offset] == b"dont"


# --------------------------------------------------------------------------------------------
# PARA_NOT_ANNOTATABLE: unknown ids and every skip reason (X6, X10, C2, EMPTY, CDATA_ONLY)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "pid", "reason"),
    [
        ("empty-01.xml", 2, SkipReason.EMPTY),
        ("err-01-existing-error.xml", 0, SkipReason.PARA_HAS_ERROR),
        ("ent-07-markup-entity.xml", 0, SkipReason.PARA_UNMAPPABLE),
        ("ent-08-external-dtd-skipped.xml", 1, SkipReason.DOCUMENT_UNVERIFIABLE),
        ("m05-cdata.xml", 0, SkipReason.CDATA_ONLY),
    ],
)
def test_skipped_paragraphs_are_not_annotatable(name: str, pid: int, reason: SkipReason) -> None:
    data, scan = load(name)
    assert scan.paragraphs[pid].skip_reason is reason
    text = scan.paragraphs[pid].text
    result = check(data, scan, pid, 0, max(1, len(text)))
    assert result.code is RejectCode.PARA_NOT_ANNOTATABLE and reason.value in result.detail


@pytest.mark.parametrize("pid", [1, 2, 99])
def test_unknown_paragraph_id(pid: int) -> None:
    data, scan = load("m01-plain.xml")
    result = check(data, scan, pid, 0, 2)
    assert result.code is RejectCode.PARA_NOT_ANNOTATABLE and "unknown" in result.detail


@pytest.mark.parametrize(
    "pid",
    [-1, -99, True, False, 0.0, 1.5, float("nan"), "0", "abc", "", None, [0], {"id": 0}],
)
def test_malformed_paragraph_id_is_invalid_field(pid) -> None:
    data, scan = load("m01-plain.xml")
    result = check(data, scan, pid, 0, 2)
    assert result.code is RejectCode.INVALID_FIELD and "paragraph_id" in result.detail


def test_malformed_paragraph_id_is_reported_even_if_other_fields_are_also_bad() -> None:
    data, scan = load("m01-plain.xml")
    result = an.check_span(data, scan, Annotation(-1, 5, 5, "typo", None))  # type: ignore[arg-type]
    assert result.code is RejectCode.INVALID_FIELD and "paragraph_id" in result.detail


def test_existing_error_paragraph_rejects_even_a_valid_span_but_clean_neighbour_accepts() -> None:
    data, scan = load("err-02-error-in-nested.xml")
    assert check(data, scan, 0, 0, 1).code is RejectCode.PARA_NOT_ANNOTATABLE  # outer (contains error)
    assert check(data, scan, 1, 0, 1).code is RejectCode.PARA_NOT_ANNOTATABLE  # inner (contains error)
    assert isinstance(check(data, scan, 2, 0, 5), an.SafeSpan)  # "clean"


# --------------------------------------------------------------------------------------------
# Field validation is applied with the paragraph's own text
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "correction", "category", "reason", "expected"),
    [
        (3, 3, "X", SP, None, RejectCode.EMPTY_SPAN),
        (3, 40, "X", SP, None, RejectCode.OUT_OF_RANGE),
        (11, 3, "X", SP, None, RejectCode.OUT_OF_RANGE),
        (3, 11, "recieved", SP, None, RejectCode.INVALID_FIELD),  # identical to the original span
        (3, 11, "received", ST, None, RejectCode.INVALID_FIELD),  # styleguide without reason
        (3, 11, "received", ST, "House style", OK),
    ],
)
def test_field_validation_uses_paragraph_text(start, end, correction, category, reason, expected) -> None:
    data, scan = load("m01-plain.xml")
    result = check(data, scan, 0, start, end, category, correction, reason)
    assert (OK if isinstance(result, an.SafeSpan) else result.code) == expected


# --------------------------------------------------------------------------------------------
# X2 atomic atoms: references and normalised newlines cannot be split
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "start", "end", "expected"),
    [
        ("m12-custom-entity.xml", 5, 13, RejectCode.SPLITS_ATOMIC),  # starts inside "&co;"
        ("m12-custom-entity.xml", 4, 8, RejectCode.SPLITS_ATOMIC),  # ends inside "&co;"
        ("m12-custom-entity.xml", 0, 13, OK),  # "See Acme corp"
        ("multi-01-lf-in-entity.xml", 1, 3, RejectCode.SPLITS_ATOMIC),  # inside a multi-callback entity
        ("multi-01-lf-in-entity.xml", 1, 6, OK),  # the whole multi-callback entity
        ("multi-02-nested-entity.xml", 2, 7, RejectCode.SPLITS_ATOMIC),
        ("ent-10-astral-and-cr-refs.xml", 1, 2, OK),  # &#128512; is one code point and one atom
        ("ent-04-char-refs.xml", 7, 8, OK),  # &#8217;
    ],
)
def test_atomic_references(name: str, start: int, end: int, expected) -> None:
    assert outcome(name, 0, start, end)[0] == expected


@pytest.mark.parametrize(
    ("name", "start", "end", "raw"),
    [
        ("enc-01-crlf.xml", 0, 7, b"teh\r\ncat"),
        ("enc-03-lone-cr.xml", 0, 3, b"a\rb"),
        ("enc-04-mixed-eol.xml", 3, 4, b"\r\n"),  # the newline atom alone
        ("enc-04-mixed-eol.xml", 7, 8, b"\n"),  # a literal LF is ordinary text
    ],
)
def test_newline_atoms_are_covered_whole(name: str, start: int, end: int, raw: bytes) -> None:
    assert outcome(name, 0, start, end) == (OK, raw)


# --------------------------------------------------------------------------------------------
# X3/X4/X8: markup inside a span (D4 OFF)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "start", "end"),
    [
        (b"<p>a <?pi x?>teh</p>", 0, 5),  # processing instruction in the gap
        (b"<p>cat <!--teh--> dog</p>", 0, 8),  # comment in the gap
        (b"<p>ice<br/>cream</p>", 0, 8),  # empty element in the gap
        (b"<p>ice<hr />cream</p>", 0, 8),
        (b"<p>a<b>c</b>d</p>", 0, 3),  # encloses a whole inline element
        (b"<p>a<b>c</b>d</p>", 0, 2),  # partially crosses it
    ],
)
def test_markup_inside_span_is_crosses_element(data: bytes, start: int, end: int) -> None:
    scan = xs.scan(data)
    assert check(data, scan, 0, start, end).code is RejectCode.CROSSES_ELEMENT


def test_spans_right_next_to_markup_are_fine() -> None:
    data = b"<p>a <?pi x?>teh <b>x</b> end</p>"
    scan = xs.scan(data)
    assert isinstance(check(data, scan, 0, 0, 2), an.SafeSpan)  # "a " ends at the PI
    assert isinstance(check(data, scan, 0, 2, 5), an.SafeSpan)  # "teh" starts after the PI
    assert isinstance(check(data, scan, 0, 6, 7), an.SafeSpan)  # "x" inside <b>


# --------------------------------------------------------------------------------------------
# X5 CDATA and X7 nested paragraphs
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "start", "end", "expected"),
    [
        ("cd-04-cdata-inside.xml", 2, 5, RejectCode.IN_CDATA),  # endpoint inside CDATA
        ("cd-04-cdata-inside.xml", 0, 7, RejectCode.IN_CDATA),  # span contains a CDATA section
        ("cd-04-cdata-inside.xml", 0, 1, OK),  # ends right before the section
        ("cd-04-cdata-inside.xml", 6, 7, OK),  # starts right after the section
        ("cd-06-cdata-crlf-markup.xml", 0, 2, RejectCode.IN_CDATA),
        ("cd-03-after-cdata.xml", 2, 5, OK),
    ],
)
def test_cdata_preservation_boundary(name: str, start: int, end: int, expected) -> None:
    assert outcome(name, 0, start, end)[0] == expected


def test_check_order_cdata_text_reported_before_element_crossing() -> None:
    # The span crosses an inline element AND covers CDATA text: the documented order reports IN_CDATA.
    data = b"<p>a<b>x</b><![CDATA[y]]>z</p>"
    scan = xs.scan(data)
    assert check(data, scan, 0, 0, 4).code is RejectCode.IN_CDATA


def test_check_order_container_reported_before_gap_classification() -> None:
    # Different containers AND an empty CDATA section in the gap: the container check comes first.
    data = b"<p>a<![CDATA[]]><b>c</b></p>"
    scan = xs.scan(data)
    assert check(data, scan, 0, 0, 2).code is RejectCode.CROSSES_ELEMENT


def test_empty_cdata_section_inside_span_is_in_cdata() -> None:
    data = b"<p>ab<![CDATA[]]>cd</p>"
    scan = xs.scan(data)
    assert check(data, scan, 0, 0, 4).code is RejectCode.IN_CDATA
    assert isinstance(check(data, scan, 0, 0, 2), an.SafeSpan)


@pytest.mark.parametrize(
    ("name", "pid", "start", "end", "expected"),
    [
        ("nest-01.xml", 0, 6, 15, RejectCode.CROSSES_PARAGRAPH),  # "teh  more"
        ("nest-01.xml", 0, 0, 9, OK),  # "Outer teh"
        ("nest-01.xml", 0, 10, 19, OK),  # " more teh" starts exactly at the barrier
        ("nest-01.xml", 1, 6, 9, OK),  # inner paragraph is independent
        ("nest-03-inside-inline.xml", 0, 2, 6, RejectCode.CROSSES_PARAGRAPH),  # "x  y" through <em><p>
        ("nest-04-deep.xml", 1, 0, 4, RejectCode.CROSSES_PARAGRAPH),
        ("nest-04-deep.xml", 2, 0, 1, OK),
    ],
)
def test_nested_paragraph_barriers(name: str, pid: int, start: int, end: int, expected) -> None:
    assert outcome(name, pid, start, end)[0] == expected


def test_namespaced_and_ordered_paragraphs() -> None:
    assert outcome("ns-02-prefixed.xml", 0, 0, 3) == (OK, b"teh")
    data, scan = load("order-01-mixed-ns.xml")
    result = check(data, scan, 1, 0, 3)
    assert isinstance(result, an.SafeSpan) and data[result.open_offset : result.close_offset] == b"two"


# --------------------------------------------------------------------------------------------
# Overlap policy (S2 review decisions 3, 5, 6)
# --------------------------------------------------------------------------------------------

DOC = b"<p>The firm recieved teh parcel on monday.</p>"
DOC_SCAN = xs.scan(DOC)


def ann(start, end, category=SP, correction="X", reason=None, pid=0) -> Annotation:
    return Annotation(pid, start, end, category, correction, reason)


def run(*annotations):
    accepted, rejected = an.prepare(DOC, DOC_SCAN, annotations)
    return [s.annotation for s in accepted], [(r.annotation, r.code) for r in rejected]


def test_exact_duplicates_are_deduplicated() -> None:
    a = ann(9, 17, correction="received")
    accepted, rejected = run(a, a, a)
    assert accepted == [a] and rejected == []


def test_same_span_same_category_different_corrections_conflict() -> None:
    a, b = ann(9, 17, correction="received"), ann(9, 17, correction="recieves")
    accepted, rejected = run(a, b)
    assert accepted == [] and {code for _, code in rejected} == {RejectCode.CONFLICT} and len(rejected) == 2


def test_same_span_same_correction_different_reason_conflict() -> None:
    a, b = ann(9, 17, correction="received", reason="r1"), ann(9, 17, correction="received", reason="r2")
    assert run(a, b) == ([], [(a, RejectCode.CONFLICT), (b, RejectCode.CONFLICT)])


def test_same_span_different_categories_overlap() -> None:
    a, b = ann(9, 17, SP, "received"), ann(9, 17, GR, "receives")
    accepted, rejected = run(a, b)
    assert accepted == [] and sorted(code.value for _, code in rejected) == ["OVERLAP", "OVERLAP"]


@pytest.mark.parametrize(
    ("first", "second"),
    [((9, 17), (13, 21)), ((9, 21), (9, 17)), ((9, 21), (13, 17)), ((4, 21), (18, 21))],
)
def test_partial_and_nested_overlaps_reject_all_involved(first, second) -> None:
    a, b = ann(*first), ann(*second)
    accepted, rejected = run(a, b)
    assert accepted == [] and {x for x, _ in rejected} == {a, b}
    assert {code for _, code in rejected} == {RejectCode.OVERLAP}


def test_adjacent_spans_are_accepted() -> None:
    a, b, c = ann(4, 8), ann(8, 9, Category.PUNCTUATION, ","), ann(9, 17, correction="received")
    accepted, rejected = run(c, a, b)
    assert accepted == [a, b, c] and rejected == []


def test_overlap_chain_rejects_every_member_but_spares_disjoint_ones() -> None:
    a, b, c, d = ann(0, 8), ann(4, 12), ann(10, 17), ann(18, 21, correction="the")
    accepted, rejected = run(a, b, c, d)
    assert accepted == [d] and {x for x, _ in rejected} == {a, b, c}


def test_conflicted_span_still_rejects_an_overlapping_span() -> None:
    a, b = ann(9, 17, correction="received"), ann(9, 17, correction="recieves")
    c = ann(13, 21)
    accepted, rejected = run(a, b, c)
    assert accepted == [] and dict(rejected) == {a: RejectCode.CONFLICT, b: RejectCode.CONFLICT, c: RejectCode.OVERLAP}


def test_unsafe_annotation_does_not_knock_out_a_valid_overlapping_one() -> None:
    good = ann(9, 17, correction="received")
    out_of_range = ann(12, 999)
    accepted, rejected = run(good, out_of_range)
    assert accepted == [good] and rejected == [(out_of_range, RejectCode.OUT_OF_RANGE)]


def test_duplicate_rejections_reported_once_and_malformed_values_do_not_crash() -> None:
    bad = Annotation(0, 9, 17, "typo", "x")  # type: ignore[arg-type]
    unhashable = Annotation(0, 9, 17, SP, ["list"])  # type: ignore[arg-type]
    accepted, rejected = run(bad, bad, unhashable, unhashable)
    assert accepted == [] and [code for _, code in rejected] == [RejectCode.INVALID_FIELD] * 2


# Malformed annotations carrying unhashable (list/dict) values in every field position.
UNHASHABLE = [
    Annotation(0, 9, 17, SP, ["received"]),  # type: ignore[arg-type]
    Annotation(0, 9, 17, SP, {"c": "received"}),  # type: ignore[arg-type]
    Annotation(0, 9, 17, SP, "received", ["why"]),  # type: ignore[arg-type]
    Annotation(0, 9, 17, SP, "received", {"r": "why"}),  # type: ignore[arg-type]
    Annotation(0, [9], 17, SP, "received"),  # type: ignore[arg-type]
    Annotation(0, 9, {"end": 17}, SP, "received"),  # type: ignore[arg-type]
    Annotation([0], 9, 17, SP, "received"),  # type: ignore[arg-type]
    Annotation(0, 9, 17, ["spelling"], "received"),  # type: ignore[arg-type]
    Annotation(0, 9, 17, {"c": 1}, "received"),  # type: ignore[arg-type]
]


@pytest.mark.parametrize("malformed", UNHASHABLE)
def test_unhashable_malformed_annotation_is_rejected_not_crashing(malformed: Annotation) -> None:
    accepted, rejected = run(malformed)
    assert accepted == [] and rejected == [(malformed, RejectCode.INVALID_FIELD)]


def test_all_unhashable_variants_together_with_valid_annotations() -> None:
    valid_a = ann(9, 17, correction="received")
    valid_b = ann(18, 21, correction="the")
    same_span_invalid = ann(9, 17, correction="recieved")  # identical correction -> INVALID_FIELD
    accepted, rejected = run(*UNHASHABLE, same_span_invalid, valid_b, valid_a)
    assert accepted == [valid_a, valid_b]  # malformed annotations never knock out valid ones
    assert len(rejected) == len(UNHASHABLE) + 1
    assert {code for _, code in rejected} == {RejectCode.INVALID_FIELD}


def test_equal_unhashable_duplicates_are_reported_once() -> None:
    a1 = Annotation(0, 9, 17, SP, ["x", {"y": 1}])  # type: ignore[arg-type]
    a2 = Annotation(0, 9, 17, SP, ["x", {"y": 1}])  # equal but a distinct object
    a3 = Annotation(0, 9, 17, SP, ["x", {"y": 2}])  # different value
    accepted, rejected = run(a1, a2, a3, a1)
    assert accepted == [] and len(rejected) == 2 and {id(x) for x, _ in rejected} <= {id(a1), id(a2), id(a3)}


def test_rejection_order_is_total_and_input_order_independent_with_malformed_values() -> None:
    pool = [*UNHASHABLE, ann(9, 17, correction="recieved"), ann(9, 17, correction="received"),
            ann(9, 17, Category.CLARITY, "received"), ann(0, 999), Annotation(-1, 0, 1, SP, "x")]
    expected = run(*pool)
    rng = random.Random(7)
    for _ in range(30):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        assert run(*shuffled) == expected
    # well-formed rejections come first in canonical order, malformed ones after them
    codes_in_order = [x for x, _ in expected[1]]
    first_malformed = next(i for i, x in enumerate(codes_in_order) if x in UNHASHABLE or x.paragraph_id == -1)
    assert all(x not in UNHASHABLE for x in codes_in_order[:first_malformed])


def test_spans_in_different_paragraphs_never_overlap() -> None:
    data, scan = load("nest-01.xml")
    outer, inner = Annotation(0, 6, 9, SP, "the"), Annotation(1, 6, 9, SP, "the")
    accepted, rejected = an.prepare(data, scan, [inner, outer])
    assert [s.annotation for s in accepted] == [outer, inner] and rejected == []


def test_result_is_independent_of_input_order() -> None:
    pool = [ann(0, 3), ann(4, 8), ann(4, 8, correction="Y"), ann(9, 17, correction="received"),
            ann(13, 21), ann(22, 28), ann(9, 17, GR, "z"), ann(29, 31, correction="in"), ann(40, 99)]
    expected = an.prepare(DOC, DOC_SCAN, pool)
    rng = random.Random(1234)
    for _ in range(50):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        assert an.prepare(DOC, DOC_SCAN, shuffled) == expected


# --------------------------------------------------------------------------------------------
# Property-based tests over every annotatable paragraph in the committed fixtures
# --------------------------------------------------------------------------------------------

_SKIP_PREFIX = ("bad-", "sec-", "ver-", "ns-03", "enc-08", "enc-09", "enc-10", "enc-11", "enc-12", "enc-13")
_DOCS = [load(p.name) for p in sorted(XML.iterdir()) if not p.name.startswith(_SKIP_PREFIX)]
_TARGETS = [(data, scan, p) for data, scan in _DOCS for p in scan.paragraphs if p.annotatable and len(p.text) > 0]
SETTINGS = settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def target_span(draw):
    data, scan, p = draw(st.sampled_from(_TARGETS))
    start = draw(st.integers(0, len(p.text) - 1))
    end = draw(st.integers(start + 1, len(p.text)))
    return data, scan, p, start, end


def _independent_text(data: bytes, scan, raw: bytes) -> str | None:
    """Parse the raw span bytes with ElementTree inside a wrapper that keeps the document's DTD."""
    root_start = scan.elements[0].start_tag[0]
    prolog = data[scan.encoding.bom_length : root_start]
    prolog = prolog.split(b"?>", 1)[1] if prolog.startswith(b"<?xml") else prolog
    root_name = data[root_start + 1 :].split(b" ", 1)[0].split(b">", 1)[0].split(b"/", 1)[0]
    if b"<!DOCTYPE" in prolog:
        prolog = prolog.replace(b"<!DOCTYPE " + root_name, b"<!DOCTYPE w", 1)
    try:
        return ET.fromstring(prolog + b"<w>" + raw + b"</w>").text or ""
    except ET.ParseError:
        return None


@SETTINGS
@given(target_span())
def test_accepted_spans_cover_exactly_their_text_and_no_markup(case) -> None:
    data, scan, p, start, end = case
    result = an.check_span(data, scan, Annotation(p.id, start, end, Category.CLARITY, "Z"))
    if p.text[start:end] == "Z":
        return
    if isinstance(result, an.SafeSpan):
        raw = data[result.open_offset : result.close_offset]
        assert b"<" not in raw  # no element, comment, PI or CDATA markup inside an accepted span
        assert _independent_text(data, scan, raw) == p.text[start:end]
        assert all(a.kind in (AtomKind.LIT, AtomKind.REF, AtomKind.NL)
                   for a in p.atoms if a.char_start < end and a.char_end > start)
    else:
        assert result.code in {
            RejectCode.SPLITS_ATOMIC, RejectCode.CROSSES_ELEMENT, RejectCode.CROSSES_PARAGRAPH, RejectCode.IN_CDATA
        }


@SETTINGS
@given(st.lists(target_span(), min_size=1, max_size=8), st.randoms(use_true_random=False))
def test_prepare_is_order_independent_and_keeps_only_disjoint_spans(cases, rnd) -> None:
    data, scan = cases[0][0], cases[0][1]
    annotations = [Annotation(p.id, s, e, SP, "Z") for d, sc, p, s, e in cases if d is data]
    accepted, rejected = an.prepare(data, scan, annotations)
    shuffled = annotations[:]
    rnd.shuffle(shuffled)
    assert an.prepare(data, scan, shuffled) == (accepted, rejected)
    for i, x in enumerate(accepted):
        for y in accepted[i + 1 :]:
            ax, ay = x.annotation, y.annotation
            assert ax.paragraph_id != ay.paragraph_id or ax.end <= ay.start or ay.end <= ax.start
            assert x.close_offset <= y.open_offset or y.close_offset <= x.open_offset  # disjoint bytes
    assert len(accepted) + len({repr(r.annotation) for r in rejected}) == len(set(map(repr, annotations)))

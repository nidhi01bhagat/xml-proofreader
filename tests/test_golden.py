"""S2.6: golden tests through the full S2 path (prepare -> render -> verify -> rollback).

* AC-03 (a): committed invented triple; the expected output is hand-written (not renderer output).
* M1-M13: accepted spike cases render to hand-written bytes; rejected ones leave the input untouched.
* AC-03 (b): the supplied example pair, LOCAL ONLY - annotations are derived from the example output
  at run time and must reproduce it byte-for-byte; skipped when the files are absent.
* AC-31: re-running on our own output adds nothing.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from proofreader import annotation as an
from proofreader import verify as v
from proofreader import xml_source as xs
from proofreader.models import Annotation, Category, RejectCode, SkipReason

XML = Path(__file__).resolve().parent / "fixtures" / "xml"
SP, GR, CA = Category.SPELLING, Category.GRAMMAR, Category.CAPITALIZATION

GOLDEN_A_ANNOTATIONS = [
    Annotation(0, 14, 21, SP, "receive"),
    Annotation(0, 22, 25, SP, "the"),
    Annotation(0, 42, 48, CA, "Monday", "Days of the week are capitalized"),
    Annotation(1, 0, 12, SP, "Salt & pepper", 'Spelling of "pepper"'),
    Annotation(1, 24, 29, SP, "table"),
    Annotation(2, 27, 31, GR, "here"),  # outer paragraph, after the nested <p>
    Annotation(3, 6, 14, SP, "paragraph"),  # nested inner paragraph
]


def run(data: bytes, annotations):
    return an.annotate(data, xs.scan(data), annotations)


# --------------------------------------------------------------------------------------------
# AC-03 (a): invented golden triple
# --------------------------------------------------------------------------------------------


def test_golden_a_reproduces_hand_written_output_byte_for_byte() -> None:
    data = (XML / "golden-a-input.xml").read_bytes()
    expected = (XML / "golden-a-expected.xml").read_bytes()
    outcome = run(data, GOLDEN_A_ANNOTATIONS)
    assert outcome.ok and outcome.rejected == () and outcome.rolled_back == ()
    assert outcome.output == expected
    assert v.verify(data, expected, GOLDEN_A_ANNOTATIONS).ok  # the hand-written file itself verifies


def test_golden_a_scan_matches_the_annotation_assumptions() -> None:
    scan = xs.scan((XML / "golden-a-input.xml").read_bytes())
    texts = [p.text for p in scan.paragraphs]
    assert [texts[a.paragraph_id][a.start : a.end] for a in GOLDEN_A_ANNOTATIONS] == [
        "recieve", "teh", "monday", "Salt & peper", "tabel", "hear", "paragrph"]
    assert scan.paragraphs[4].skip_reason is None and scan.paragraphs[5].skip_reason is SkipReason.PARA_HAS_ERROR


def test_golden_a_is_independent_of_annotation_order() -> None:
    data = (XML / "golden-a-input.xml").read_bytes()
    assert run(data, list(reversed(GOLDEN_A_ANNOTATIONS))) == run(data, GOLDEN_A_ANNOTATIONS)


def test_golden_a_rejects_annotations_on_the_existing_error_paragraph() -> None:
    data = (XML / "golden-a-input.xml").read_bytes()
    outcome = run(data, [*GOLDEN_A_ANNOTATIONS, Annotation(5, 4, 12, SP, "existed")])
    assert outcome.ok and outcome.output == (XML / "golden-a-expected.xml").read_bytes()
    assert [r.code for r in outcome.rejected] == [RejectCode.PARA_NOT_ANNOTATABLE]


# --------------------------------------------------------------------------------------------
# M1-M13 through the full path (D4 off)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "annotation", "expected"),
    [
        ("m01-plain.xml", Annotation(0, 3, 11, SP, "received"),
         b'<p>We <error type="spelling" correction="received">recieved</error> teh parcel.</p>'),
        ("m02-em-hyphen.xml", Annotation(0, 9, 15, SP, "vivid"),
         b'<p>We saw a <em><error type="spelling" correction="vivid">bright</error></em>-red kite.</p>'),
        ("m03-amp.xml", Annotation(0, 0, 12, SP, "Salt & pepper"),
         b'<p><error type="spelling" correction="Salt &amp; pepper">Salt &amp; peper</error> are on teh table.</p>'),
        ("m04-quote-refs.xml", Annotation(0, 4, 9, GR, "writes"),
         b'<p>She <error type="grammar" correction="writes">wrote</error> &#8217;back soon&#8217; on teh door.</p>'),
        ("m06-em-compound.xml", Annotation(0, 8, 19, SP, "night shift"),
         b'<p>Ask the <em><error type="spelling" correction="night shift">night-shift</error></em> porter.</p>'),
        ("m10-crlf-gap.xml", Annotation(0, 0, 9, SP, "ice cream"),
         b'<p><error type="spelling" correction="ice cream">ice\r\ncream</error> cone</p>'),
        ("m12-custom-entity.xml", Annotation(0, 4, 13, CA, "Acme Corp"),
         b'<!DOCTYPE p [<!ENTITY co "Acme corp">]><p>See <error type="capitalization" correction="Acme Corp">&co;</error> now</p>'),
        ("m13-default-ns.xml", Annotation(0, 0, 3, SP, "the"),
         b'<r xmlns="urn:x"><p><error type="spelling" correction="the">teh</error></p></r>'),
    ],
)
def test_accepted_spike_cases_render_to_hand_written_bytes(name: str, annotation: Annotation, expected: bytes) -> None:
    outcome = run((XML / name).read_bytes(), [annotation])
    assert outcome.ok and outcome.rejected == () and outcome.output == expected


@pytest.mark.parametrize(
    ("name", "start", "end", "code"),
    [
        ("m02-em-hyphen.xml", 9, 19, RejectCode.CROSSES_ELEMENT),  # M2a
        ("m05-cdata.xml", 0, 3, RejectCode.PARA_NOT_ANNOTATABLE),  # M5 (CDATA_ONLY)
        ("m06-em-compound.xml", 8, 26, RejectCode.CROSSES_ELEMENT),  # M6b
        ("m06-em-compound.xml", 4, 13, RejectCode.CROSSES_ELEMENT),  # M6c
        ("m07-end-bold.xml", 3, 10, RejectCode.CROSSES_ELEMENT),  # M7
        ("m08-two-level.xml", 2, 9, RejectCode.CROSSES_ELEMENT),  # M8
        ("m09-comment-pi.xml", 0, 9, RejectCode.CROSSES_ELEMENT),  # M9
        ("m11-br.xml", 0, 8, RejectCode.CROSSES_ELEMENT),  # M11
        ("m12-custom-entity.xml", 4, 8, RejectCode.SPLITS_ATOMIC),  # M12 partial entity
    ],
)
def test_rejected_spike_cases_leave_the_document_untouched(name: str, start: int, end: int, code: RejectCode) -> None:
    data = (XML / name).read_bytes()
    outcome = run(data, [Annotation(0, start, end, SP, "Z")])
    assert outcome.ok and outcome.output == data and [r.code for r in outcome.rejected] == [code]


BARRIER_DOC = b"<r><p>Outer teh<p>inner</p> tial</p></r>"  # outer own text "Outer teh tial", barrier at 9


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (Annotation(0, 6, 9, SP, "the"),  # ends exactly at the barrier
         b'<r><p>Outer <error type="spelling" correction="the">teh</error><p>inner</p> tial</p></r>'),
        (Annotation(0, 9, 14, SP, " tail"),  # starts exactly at the barrier
         b'<r><p>Outer teh<p>inner</p><error type="spelling" correction=" tail"> tial</error></p></r>'),
    ],
)
def test_spans_touching_a_nested_paragraph_barrier_are_allowed(annotation: Annotation, expected: bytes) -> None:
    assert xs.scan(BARRIER_DOC).paragraphs[0].barriers == (9,)
    outcome = run(BARRIER_DOC, [annotation])
    assert outcome.ok and outcome.rejected == () and outcome.output == expected


def test_span_across_a_nested_paragraph_barrier_is_rejected() -> None:
    outcome = run(BARRIER_DOC, [Annotation(0, 6, 14, SP, "the tail")])
    assert outcome.output == BARRIER_DOC and [r.code for r in outcome.rejected] == [RejectCode.CROSSES_PARAGRAPH]


# --------------------------------------------------------------------------------------------
# AC-31: re-running on our own output adds nothing
# --------------------------------------------------------------------------------------------


def test_rerun_on_own_output_adds_nothing() -> None:
    expected = (XML / "golden-a-expected.xml").read_bytes()
    outcome = run(expected, GOLDEN_A_ANNOTATIONS)
    assert outcome.ok and outcome.output == expected
    assert {r.code for r in outcome.rejected} == {RejectCode.PARA_NOT_ANNOTATABLE}


# --------------------------------------------------------------------------------------------
# AC-03 (b): supplied example pair (local only)
# --------------------------------------------------------------------------------------------

ASSESSMENT = Path(__file__).resolve().parents[1] / "SSE Assignment"
_EXAMPLE = ("example_input.xml", "example_output.xml")


@pytest.mark.skipif(
    not all((ASSESSMENT / n).is_file() for n in _EXAMPLE),
    reason="local assessment files (SSE Assignment/example_input.xml, example_output.xml) absent",
)
def test_golden_b_supplied_example_pair_byte_for_byte() -> None:
    before = {n: hashlib.sha256((ASSESSMENT / n).read_bytes()).hexdigest() for n in _EXAMPLE}
    data = (ASSESSMENT / "example_input.xml").read_bytes()
    expected = (ASSESSMENT / "example_output.xml").read_bytes()
    root = ET.fromstring(expected, parser=ET.XMLParser(encoding="utf-8"))
    paragraphs = [e for e in root.iter() if isinstance(e.tag, str) and e.tag.rsplit("}", 1)[-1] == "p"]
    annotations = []
    for pid, p in enumerate(paragraphs):  # derive annotations from the example output at run time
        _, errors = v._own_text_and_errors(p)
        for start, end, el in errors:
            annotations.append(Annotation(pid, start, end, Category(el.get("type")), el.get("correction"), el.get("reason")))
    assert annotations, "expected <error> elements in the example output"
    outcome = run(data, annotations)
    assert outcome.ok and outcome.rejected == () and outcome.rolled_back == ()
    assert outcome.output == expected
    after = {n: hashlib.sha256((ASSESSMENT / n).read_bytes()).hexdigest() for n in _EXAMPLE}
    assert after == before

"""S2.5: annotate -> independent verification -> per-paragraph rollback (fail closed).

Faults are injected through a renderer wrapper that really produces wrong output for chosen
paragraphs; the real verifier must catch them and rollback must recover. No LLM, no pipeline.
"""

from __future__ import annotations

import dataclasses
import random
import re

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from proofreader import annotation as an
from proofreader import verify as v
from proofreader import xml_source as xs
from proofreader.models import Annotation, Category, RejectCode

SP, GR = Category.SPELLING, Category.GRAMMAR
NESTED = b"<doc><p>Outer teh <p>Inner teh</p> more teh</p><p>Third teh</p><p>Fourth teh</p></doc>"
#          paragraph ids:   0 = outer, 1 = inner (nested in 0), 2 = third, 3 = fourth
NESTED_SCAN = xs.scan(NESTED)
ANNS = [Annotation(0, 6, 9, SP, "the"), Annotation(1, 6, 9, SP, "the"), Annotation(0, 16, 19, GR, "teh!"),
        Annotation(2, 6, 9, SP, "the"), Annotation(3, 7, 10, SP, "the")]


def wrong_attributes(bad_pids):
    """Renderer fault: spans of ``bad_pids`` are rendered with a different correction than annotated."""
    def render_fn(data, spans, codec):
        spans = [dataclasses.replace(s, annotation=dataclasses.replace(s.annotation, correction=s.annotation.correction + "!"))
                 if s.annotation.paragraph_id in bad_pids else s for s in spans]
        return an.render(data, spans, codec)
    return render_fn


def altered_text(bad_pids):
    """Renderer fault: the original text inside each error of ``bad_pids`` is changed (G1 + V6 + V1)."""
    def render_fn(data, spans, codec):
        output, ranges = an.render(data, spans, codec)
        for s in sorted(spans, key=lambda s: -s.open_offset):
            if s.annotation.paragraph_id in bad_pids:
                marker = b"</error>"
                # locate this span's closing tag: the n-th "</error>" in byte order
                order = sorted(spans, key=lambda x: x.open_offset).index(s)
                pos = [m.start() for m in re.finditer(re.escape(marker), output)][order]
                output = output[: pos - 1] + b"#" + output[pos:]  # replace the last original char inside the error
        return output, ranges
    return render_fn


def strip_errors(output: bytes) -> bytes:
    return re.sub(rb"<error [^>]*>|</error>", b"", output)


def pids(annotations):
    return sorted({a.paragraph_id for a in annotations})


def check_success(outcome, expected_applied_pids, expected_rolled_back):
    assert outcome.ok and outcome.output is not None
    assert pids(outcome.applied) == expected_applied_pids
    assert outcome.rolled_back == tuple(expected_rolled_back)
    assert strip_errors(outcome.output) == NESTED  # exact byte restoration
    assert v.verify(NESTED, outcome.output, outcome.applied).ok  # independently re-verified
    retained = [s for s in an.prepare(NESTED, NESTED_SCAN, outcome.applied)[0]]
    assert outcome.output == an.render(NESTED, retained, "utf-8")[0]  # == clean render of retained only
    rolled = [r.annotation for r in outcome.rejected if r.code is RejectCode.VERIFY_FAILED]
    assert pids(rolled) == sorted(expected_rolled_back)


# --------------------------------------------------------------------------------------------
# Success paths
# --------------------------------------------------------------------------------------------


def test_all_paragraphs_pass_in_one_round() -> None:
    outcome = an.annotate(NESTED, NESTED_SCAN, ANNS)
    check_success(outcome, [0, 1, 2, 3], [])
    assert outcome.rounds == 1 and outcome.failures == ()


def test_no_annotations_returns_original_bytes() -> None:
    outcome = an.annotate(NESTED, NESTED_SCAN, [])
    assert outcome.ok and outcome.output == NESTED and outcome.applied == () and outcome.rounds == 1


def test_all_annotations_rejected_returns_original_bytes() -> None:
    outcome = an.annotate(NESTED, NESTED_SCAN, [Annotation(9, 0, 1, SP, "x"), Annotation(0, 0, 99, SP, "x")])
    assert outcome.ok and outcome.output == NESTED and len(outcome.rejected) == 2


# --------------------------------------------------------------------------------------------
# Rollback paths
# --------------------------------------------------------------------------------------------


def test_one_failed_paragraph_and_successful_others() -> None:
    outcome = an.annotate(NESTED, NESTED_SCAN, ANNS, render_fn=wrong_attributes({2}))
    check_success(outcome, [0, 1, 3], [2])
    assert outcome.rounds == 2 and "ATTRIBUTES_MISMATCH" in outcome.failures[0]


def test_multiple_failed_paragraphs_in_one_round() -> None:
    outcome = an.annotate(NESTED, NESTED_SCAN, ANNS, render_fn=wrong_attributes({2, 3}))
    check_success(outcome, [0, 1], [2, 3])


def test_multiple_text_failures_are_all_caught_in_one_round() -> None:
    # V6/V1 report every paragraph whose text changed, so one rollback round suffices.
    outcome = an.annotate(NESTED, NESTED_SCAN, ANNS, render_fn=altered_text({2, 3}))
    check_success(outcome, [0, 1], [2, 3])
    assert outcome.rounds == 2


def test_g1_only_failures_in_several_paragraphs_need_several_rounds() -> None:
    # A reference respelled (&amp; -> &#38;) is invisible to ElementTree text and structure checks;
    # only byte restoration (G1) sees it, and G1 reports the first difference only, so each round
    # rolls back one more paragraph until the output verifies.
    data = b"<r><p>a &amp; teh</p><p>b &amp; teh</p><p>c &amp; teh</p></r>"
    scan = xs.scan(data)
    annotations = [Annotation(pid, 4, 7, SP, "the") for pid in range(3)]

    def respell(bad_pids):
        def render_fn(d, spans, codec):
            output, ranges = an.render(d, spans, codec)
            rendered_pids = {s.annotation.paragraph_id for s in spans}
            for pid, letter in enumerate(b"abc"):
                if pid in bad_pids and pid in rendered_pids:  # the fault only hits paragraphs being annotated
                    output = output.replace(b"<p>" + bytes([letter]) + b" &amp;", b"<p>" + bytes([letter]) + b" &#38;")
            return output, ranges
        return render_fn

    outcome = an.annotate(data, scan, annotations, render_fn=respell({0, 2}))
    assert outcome.ok and outcome.rolled_back == (0, 2) and pids(outcome.applied) == [1]
    assert outcome.rounds == 3
    assert strip_errors(outcome.output) == data and v.verify(data, outcome.output, outcome.applied).ok


def test_fault_persisting_in_a_rolled_back_paragraph_fails_closed() -> None:
    # If corruption remains after a paragraph's annotations were removed, that paragraph is now
    # untouched yet changed: a document-level problem, so nothing is output.
    data = b"<r><p>a &amp; teh</p><p>b &amp; teh</p></r>"
    scan = xs.scan(data)

    def sticky(d, spans, codec):
        output, ranges = an.render(d, spans, codec)
        return output.replace(b"<p>a &amp;", b"<p>a &#38;"), ranges

    outcome = an.annotate(data, scan, [Annotation(0, 4, 7, SP, "the"), Annotation(1, 4, 7, SP, "the")], render_fn=sticky)
    assert not outcome.ok and outcome.output is None and outcome.rolled_back == (0,)
    assert "BYTES_NOT_RESTORED (document)" in outcome.failures[-1]


def test_outer_fails_nested_succeeds() -> None:
    outcome = an.annotate(NESTED, NESTED_SCAN, ANNS, render_fn=wrong_attributes({0}))
    check_success(outcome, [1, 2, 3], [0])
    assert b"<p>Inner <error" in outcome.output and b"<p>Outer teh <p>" in outcome.output


def test_nested_fails_outer_succeeds() -> None:
    outcome = an.annotate(NESTED, NESTED_SCAN, ANNS, render_fn=wrong_attributes({1}))
    check_success(outcome, [0, 2, 3], [1])
    assert b"<p>Inner teh</p>" in outcome.output and b"<p>Outer <error" in outcome.output


def test_outer_and_nested_both_fail() -> None:
    outcome = an.annotate(NESTED, NESTED_SCAN, ANNS, render_fn=wrong_attributes({0, 1}))
    check_success(outcome, [2, 3], [0, 1])


@pytest.mark.parametrize("bad", [{0}, {1}, {0, 1}])
def test_nested_byte_failures_are_isolated(bad) -> None:
    outcome = an.annotate(NESTED, NESTED_SCAN, ANNS, render_fn=altered_text(bad))
    check_success(outcome, sorted({0, 1, 2, 3} - bad), sorted(bad))


def test_every_paragraph_fails_returns_original_bytes() -> None:
    outcome = an.annotate(NESTED, NESTED_SCAN, ANNS, render_fn=wrong_attributes({0, 1, 2, 3}))
    assert outcome.ok and outcome.output == NESTED and outcome.applied == () and outcome.rolled_back == (0, 1, 2, 3)


# --------------------------------------------------------------------------------------------
# Fail closed
# --------------------------------------------------------------------------------------------


def _outside_paragraph_fault(data, spans, codec):
    output, ranges = an.render(data, spans, codec)
    return output.replace(b"<doc>", b'<doc a="1">'), ranges


def test_document_level_failure_produces_no_output() -> None:
    outcome = an.annotate(NESTED, NESTED_SCAN, ANNS, render_fn=_outside_paragraph_fault)
    assert not outcome.ok and outcome.output is None and outcome.applied == ()
    assert any("(document)" in f for f in outcome.failures)


def test_malformed_output_produces_no_output() -> None:
    def broken(data, spans, codec):
        output, ranges = an.render(data, spans, codec)
        return output.replace(b"</doc>", b""), ranges
    outcome = an.annotate(NESTED, NESTED_SCAN, ANNS, render_fn=broken)
    assert not outcome.ok and outcome.output is None and "OUTPUT_NOT_WELL_FORMED" in outcome.failures[-1]


def test_final_verification_failure_produces_no_output() -> None:
    calls = []

    def stateful(data, spans, codec):  # round 1: paragraph fault; round 2: document fault
        calls.append(len(spans))
        if len(calls) == 1:
            return wrong_attributes({2})(data, spans, codec)
        return _outside_paragraph_fault(data, spans, codec)

    outcome = an.annotate(NESTED, NESTED_SCAN, ANNS, render_fn=stateful)
    assert not outcome.ok and outcome.output is None and outcome.rounds == 2
    assert outcome.rolled_back == (2,)  # the first rollback is still reported, but nothing is output


def test_verifier_blaming_an_untouched_paragraph_fails_closed() -> None:
    def blames_untouched(original, output, applied):
        issue = v.VerificationIssue(v.IssueCode.TEXT_CHANGED, "synthetic", paragraph_id=3)
        return v.VerificationReport((), ((3, (issue,)),))

    outcome = an.annotate(NESTED, NESTED_SCAN, ANNS[:2], verify_fn=blames_untouched)  # paragraph 3 not annotated
    assert not outcome.ok and outcome.output is None and "paragraph 3" in outcome.failures[-1]


def test_renderer_refusal_fails_closed() -> None:
    def refuses(data, spans, codec):
        raise an.RenderInputError("synthetic refusal")
    outcome = an.annotate(NESTED, NESTED_SCAN, ANNS, render_fn=refuses)
    assert not outcome.ok and outcome.output is None and "RENDER_REFUSED" in outcome.failures[-1]


def test_unverified_output_is_never_reported_as_success() -> None:
    def lies(original, output, applied):  # a verifier that never passes anything
        issue = v.VerificationIssue(v.IssueCode.BYTES_NOT_RESTORED, "synthetic")
        return v.VerificationReport((issue,), ())
    outcome = an.annotate(NESTED, NESTED_SCAN, ANNS, verify_fn=lies)
    assert not outcome.ok and outcome.output is None


# --------------------------------------------------------------------------------------------
# Preservation and determinism
# --------------------------------------------------------------------------------------------


def test_markup_declarations_and_namespaces_preserved_after_rollback() -> None:
    data = (b'\xef\xbb\xbf<?xml version="1.0" encoding="UTF-8"?>\r\n<!-- c --><?pi x?>'
            b'<r xmlns="urn:d" xmlns:x="urn:x"><p x:id="1">teh <![CDATA[raw]]> cat</p><x:p>dgo &amp; teh</x:p></r>')
    scan = xs.scan(data)
    annotations = [Annotation(0, 0, 3, SP, "the"), Annotation(1, 0, 3, SP, "dog"), Annotation(1, 6, 9, SP, "the")]
    outcome = an.annotate(data, scan, annotations, render_fn=wrong_attributes({1}))
    assert outcome.ok and outcome.rolled_back == (1,) and pids(outcome.applied) == [0]
    assert strip_errors(outcome.output) == data
    assert outcome.output.startswith(b'\xef\xbb\xbf<?xml version="1.0" encoding="UTF-8"?>\r\n<!-- c --><?pi x?>')
    assert b"<![CDATA[raw]]>" in outcome.output and b"<x:p>dgo &amp; teh</x:p>" in outcome.output


def test_rejections_are_in_canonical_order_across_prepare_and_rollback() -> None:
    out_of_range = Annotation(3, 0, 99, SP, "x")  # rejected by prepare (paragraph 3)
    outcome = an.annotate(NESTED, NESTED_SCAN, [*ANNS, out_of_range], render_fn=wrong_attributes({0}))
    assert outcome.ok and outcome.rolled_back == (0,)
    assert [r.annotation.paragraph_id for r in outcome.rejected] == [0, 0, 3]  # rollback (para 0) before para 3
    assert list(outcome.rejected) == sorted(outcome.rejected, key=an._rejection_sort_key)


def test_deterministic_under_shuffled_annotation_order() -> None:
    expected = an.annotate(NESTED, NESTED_SCAN, ANNS, render_fn=altered_text({0, 3}))
    rng = random.Random(99)
    for _ in range(25):
        shuffled = ANNS[:]
        rng.shuffle(shuffled)
        assert an.annotate(NESTED, NESTED_SCAN, shuffled, render_fn=altered_text({0, 3})) == expected


# --------------------------------------------------------------------------------------------
# Property-based: random faults in random paragraph sets are rolled back exactly
# --------------------------------------------------------------------------------------------

DOCS = [
    NESTED,
    b"<r><p>alpha teh beta</p><p>gamma teh</p><p>a <p>b teh c</p> d teh</p><p>last teh</p></r>",
    b'<r xmlns="urn:d"><p>one teh</p><s><p>two teh <b>bold</b> teh</p></s><p>three &amp; teh</p></r>',
]


@st.composite
def faulty_case(draw):
    data = draw(st.sampled_from(DOCS))
    scan = xs.scan(data)
    annotations = []
    for p in scan.paragraphs:
        for m in re.finditer("teh", p.text):
            if draw(st.booleans()):
                annotations.append(Annotation(p.id, m.start(), m.end(), SP, "the"))
    bad = set(draw(st.sets(st.sampled_from([p.id for p in scan.paragraphs]))))
    fault = draw(st.sampled_from([wrong_attributes, altered_text]))
    return data, scan, annotations, bad, fault


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(faulty_case())
def test_rollback_properties(case) -> None:
    data, scan, annotations, bad, fault = case
    outcome = an.annotate(data, scan, annotations, render_fn=fault(bad))
    annotated = {a.paragraph_id for a in annotations}
    assert outcome.ok
    assert set(outcome.rolled_back) == bad & annotated
    assert {a.paragraph_id for a in outcome.applied} == annotated - bad
    assert len(outcome.applied) == sum(1 for a in annotations if a.paragraph_id not in bad)
    assert strip_errors(outcome.output) == data
    assert v.verify(data, outcome.output, outcome.applied).ok

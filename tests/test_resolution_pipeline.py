"""S3.6: quotation integration, document resolution and traceability (resolver -> annotate -> trace).

All texts are invented. Expected output bytes are written by hand.
"""

from __future__ import annotations

import dataclasses
import random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from proofreader import annotation as an
from proofreader import quotes as q
from proofreader import resolver as r
from proofreader import xml_source as xs
from proofreader.models import (
    FindingRejection,
    MatchMode,
    ProtectedRange,
    QuoteAnalysis,
    RangeOrigin,
    RejectCode,
    ResolvedFinding,
    TraceEntry,
    TraceRecord,
    TraceStatus,
)
from proofreader.verify import VerificationIssue, VerificationReport

LDQ, RDQ, LSQ, RSQ, SDQ, SSQ, LAQ = chr(0x201C), chr(0x201D), chr(0x2018), chr(0x2019), chr(0x22), chr(0x27), chr(0xAB)
LDQ_B, RDQ_B, RSQ_B = LDQ.encode(), RDQ.encode(), RSQ.encode()
IN, TOUCH, IN_CORR = RejectCode.IN_QUOTATION, RejectCode.TOUCHES_QUOTATION, RejectCode.QUOTATION_IN_CORRECTION
SECRET = "UNIQUE-DOCUMENT-TEXT"


def item(original, before="", after="", correction="FIX", type_="spelling", ws=0, we=0, reason="why") -> dict:
    return {"original": original, "before": before, "after": after, "word_start": ws, "word_end": we,
            "correction": correction, "type": type_, "reason": reason}


def run(markup: str, payloads: dict, **annotate_kwargs):
    data = markup.encode("utf-8")
    scan = xs.scan(data)
    results = r.resolve_document(scan, payloads)
    outcome = an.annotate(data, scan, r.annotations(results), **annotate_kwargs)
    return data, scan, results, outcome, r.trace(results, outcome)


def one(markup: str, finding: dict):
    """Resolve a single finding for paragraph 0; returns the ResolvedFinding or FindingRejection."""
    scan = xs.scan(markup.encode("utf-8"))
    (result,) = r.resolve_document(scan, {0: {"findings": [finding]}})
    return (result.resolved or result.rejections)[0]


def code(x) -> RejectCode | None:
    return x.code if isinstance(x, FindingRejection) else None


def statuses(records) -> list[tuple]:
    return [(t.paragraph_id, t.finding_index, t.status, t.code) for t in records]


# --------------------------------------------------------------------------------------------
# Complete pipeline: valid findings alongside rejected ones
# --------------------------------------------------------------------------------------------


def test_pipeline_with_mixed_findings_end_to_end() -> None:
    markup = f"<r><p>He said {LDQ}teh end{RDQ} and teh dogs bark.</p><p>the boys bikes are red</p></r>"
    payloads = {
        1: {"findings": [item("boys", "the", "bikes", "boy" + RSQ + "s"), item("bikes", "zzz", "are", "bike")]},
        0: {"findings": [
            item("teh", "said " + LDQ, "end", "the"),  # inside the quotation
            item("teh", "and", "dogs", "the"),  # applied
            item("zzz", "", "", "x"),  # not found
            {"original": "x"},  # invalid item
        ]},
    }
    data, scan, results, outcome, records = run(markup, payloads)
    assert outcome.ok
    expected = (b"<r><p>He said " + LDQ_B + b"teh end" + RDQ_B + b' and <error type="spelling" correction="the" '
                b'reason="why">teh</error> dogs bark.</p><p>the <error type="spelling" correction="boy' + RSQ_B
                + b's" reason="why">boys</error> bikes are red</p></r>')
    assert outcome.output == expected
    assert statuses(records) == [
        (0, 0, TraceStatus.REJECTED_S3, IN),
        (0, 1, TraceStatus.APPLIED, None),
        (0, 2, TraceStatus.REJECTED_S3, RejectCode.NOT_FOUND),
        (0, 3, TraceStatus.REJECTED_S3, RejectCode.INVALID_FIELD),
        (1, 0, TraceStatus.APPLIED, None),
        (1, 1, TraceStatus.REJECTED_S3, RejectCode.ANCHOR_MISMATCH),
    ]
    assert [t.detail for t in records if t.status is TraceStatus.REJECTED_S3] == [
        "span overlaps protected quotation content", "no token-aligned occurrence of the original",
        "before: missing; after: missing; word_start: missing; word_end: missing; "
        "correction: missing; type: missing; reason: missing",  # the item {"original": "x"}
        "1 occurrence(s), none with matching anchors"]  # S3 details carried into the trace
    applied = [t for t in records if t.status is TraceStatus.APPLIED]
    assert [(t.match_mode, t.start, t.end, t.hint_drift) for t in applied] == [
        (MatchMode.EXACT, 22, 25, True), (MatchMode.EXACT, 4, 8, True)]


# --------------------------------------------------------------------------------------------
# Every quotation rejection code
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "finding", "expected"),
    [
        (f"He said {LDQ}teh end{RDQ} today.", item("teh", "said " + LDQ, "end"), IN),
        (f"He said {SDQ}yes{SDQ}.", item(f"{SDQ}yes{SDQ}", "said", ".", LDQ + "yes" + RDQ), TOUCH),  # Q-17
        (f"a {LDQ}b{RDQ}c d", item("c", LDQ + "b" + RDQ, "d"), TOUCH),  # adjacent to the range end
        (f"a b{LDQ}c{RDQ} d", item("b", "a", LDQ + "c"), TOUCH),  # adjacent to the range start
        ("a teh b", item("teh", "a", "b", LDQ + "the" + RDQ), IN_CORR),
        ("a teh b", item("teh", "a", "b", "the" + LAQ), IN_CORR),
        ("a teh b", item("teh", "a", "b", LSQ + "the"), IN_CORR),  # U+2018 is never an apostrophe
        ("a teh b", item("teh", "a", "b", SDQ + "the"), IN_CORR),
        ("a teh b", item("teh", "a", "b", chr(0x2E02) + "the"), IN_CORR),  # other Pi/Pf character
        ("a tis day", item("tis", "a", "day", RSQ + "tis"), IN_CORR),  # opens a quotation (E7 shape)
        ("a teh b", item("teh", "a", "b", "the " + RSQ + " x"), IN_CORR),  # ambiguous mark
        (f"the cats{RSQ} toys", item("cats", "the", RSQ + " toys", "cats "), IN_CORR),  # E8: neighbour stability
    ],
)
def test_quotation_rejection_codes(text: str, finding: dict, expected: RejectCode) -> None:
    assert code(one(f"<p>{text}</p>", finding)) is expected


def test_quotation_rejections_are_text_free() -> None:
    result = one(f"<p>{SECRET} said {LDQ}teh end{RDQ}</p>", item("teh", "said " + LDQ, "end"))
    assert code(result) is IN and SECRET not in result.detail


# --------------------------------------------------------------------------------------------
# Apostrophe exceptions (allowed) and revision-3 examples
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "finding", "span"),
    [
        ("I dont know", item("dont", "I", "know", "don" + RSQ + "t"), (2, 6)),
        ("I dont know", item("dont", "I", "know", "don't"), (2, 6)),  # straight apostrophe
        ("the boys bikes", item("boys", "the", "bikes", "boy" + RSQ + "s"), (4, 8)),  # Q-10 shape
        ("the boys bikes", item("boys", "the", "bikes", "boys" + RSQ), (4, 8)),  # possessive, no opener
        (f"{LSQ}dogs{RSQ} and cats{RSQ} met the boys bikes.", item("boys", "the", "bikes.", "boys" + RSQ), (25, 29)),  # E2
        (f"The boys bikes and {LSQ}dogs{RSQ} toys.", item("boys", "The", "bikes", "boys" + RSQ), (4, 8)),  # E4
        (f"{LSQ}dogs{RSQ} and cats{RSQ} met teh boys", item("boys", "teh", "", "boys" + RSQ), (25, 29)),  # E6
        (f"{LSQ}dogs{RSQ} and cats{RSQ} met the boys bikes.", item("bikes", "boys", ".", "bikes" + RSQ), (30, 35)),  # E10
        ("in the 90s it", item("90s", "the", "it", RSQ + "90s"), (7, 10)),  # boundary-digit apostrophe
        (f"I cann{RSQ}t go", item("cann", "I", RSQ + "t go", "can"), (2, 6)),  # E9b: neighbour keeps its role
    ],
)
def test_apostrophe_corrections_are_allowed(text: str, finding: dict, span) -> None:
    result = one(f"<p>{text}</p>", finding)
    assert isinstance(result, ResolvedFinding), result
    assert (result.annotation.start, result.annotation.end) == span


@pytest.mark.parametrize(
    ("text", "finding", "expected"),
    [
        (f"{LSQ}dogs{RSQ} and cats{RSQ} were fed.", item("and", "dogs" + RSQ, "cats" + RSQ, "or"), IN),  # E1
        (f"{LSQ}dogs{RSQ} and the boys bikes and cats{RSQ} fun.", item("boys", "the", "bikes", "boys" + RSQ), IN),  # E3
        (f"{LSQ}dogs and the boys bikes", item("boys", "the", "bikes", "boys" + RSQ), IN),  # E5 / P-7
        (f"{LSQ}dogs{RSQ} and cats{RSQ} met tis day.", item("tis", "met", "day", RSQ + "tis"), IN_CORR),  # E7
        (f"a {LDQ}b{RDQ}boys x", item("boys", LDQ + "b" + RDQ, "x", "boys" + RSQ), TOUCH),  # P-8: adjacency first
    ],
)
def test_revision_3_rejections_and_precedence(text: str, finding: dict, expected: RejectCode) -> None:
    assert code(one(f"<p>{text}</p>", finding)) is expected


def test_e1_to_e3_source_protection_uses_the_last_candidate_closer() -> None:
    analysis = q.analyse_text(f"{LSQ}dogs{RSQ} and cats{RSQ} were fed.")
    assert [(x.start, x.end) for x in analysis.ranges] == [(0, 16)]  # ends after cats', not dogs'


def test_defensive_branch_rejects_an_apostrophe_next_to_protected_text() -> None:
    # never reached through the pipeline (span checks run first); checked directly
    text = "a bc d"
    guarded = QuoteAnalysis(0, (), (ProtectedRange(0, 2, False, False, RangeOrigin.PAIR),), None, None)
    assert r._introduced_apostrophe_verdict(guarded, text, (), 2, 4, "bc" + RSQ) is not None
    open_ = QuoteAnalysis(0, (), (), None, None)
    assert r._introduced_apostrophe_verdict(open_, text, (), 2, 4, "bc" + RSQ) is None


# --------------------------------------------------------------------------------------------
# Q-01 ... Q-21 through the resolver (docs/03 section 8 texts and findings)
# --------------------------------------------------------------------------------------------

Q = {
    "Q-01": (f"He said {LDQ}teh end{RDQ} today.", "teh", "FIX", IN),
    "Q-02": (f"He said {LDQ}teh end{RDQ} today.", "today", "FIX", None),
    "Q-03": (f"He said {SDQ}teh end{SDQ} today.", "teh", "FIX", IN),
    "Q-04a": (f"{LDQ}He said {LSQ}teh{RSQ} loudly,{RDQ} she noted.", "teh", "FIX", IN),
    "Q-04b": (f"{LDQ}He said {LSQ}teh{RSQ} loudly,{RDQ} she noted.", "loudly", "FIX", IN),
    "Q-05": (f"{LDQ}He said {LSQ}teh{RSQ} loudly,{RDQ} she noted.", "noted", "FIX", None),
    "Q-06": (f"She wrote {SDQ}teh end. Then left.", "Then", "FIX", IN),
    "Q-07": (f"Teh clerk wrote {SDQ}done.", "Teh", "FIX", None),
    "Q-08a": (f"The note read {RSQ}keep left{RSQ}.", "keep", "FIX", IN),
    "Q-08b": (f"The note read {RSQ}keep left{RSQ}.", "read", "FIX", None),
    "Q-09": (f"Don{SSQ}t move teh box.", f"Don{SSQ}t", "Do not", None),
    "Q-10": (f"The boys{RSQ} bikes are red.", f"boys{RSQ}", f"boy{RSQ}s", None),
    "Q-11": (f"The {LSQ}dogs{RSQ} bowls{RSQ} were teh issue.", "bowls", "FIX", IN),
    "Q-12": (f"The {LSQ}dogs{RSQ} bowls{RSQ} were teh issue.", "teh", "FIX", None),
    "Q-13": (f"{SSQ}Tis teh season.", "teh", "FIX", IN),
    "Q-14a": (f"teh end,{RDQ} he said.", "teh", "FIX", IN),
    "Q-14b": (f"teh end,{RDQ} he said.", "he said", "FIX", None),
    "Q-15": (f"In the {RSQ}90s teh market grew.", "teh", "FIX", None),
    "Q-16": (f"{LAQ} teh fin {chr(0xBB)} dit-il.", "teh", "FIX", IN),
    "Q-17": (f"He said {SDQ}yes{SDQ}.", f"{SDQ}yes{SDQ}", f"{LDQ}yes{RDQ}", TOUCH),
    "Q-18": (f"{LDQ}Don{SSQ}t move teh box,{RDQ} she said.", "teh", "FIX", IN),
    "Q-19": (f"A 6{SDQ} pipe and teh valve.", "teh", "FIX", IN),
    "Q-20": (f"He wrote {LDQ}teh{RDQ} and {LDQ}then{RDQ}.", "then", "FIX", IN),
    "Q-21a": (f"Prices rose in the 1950{RSQ}s, then time-frames changed.", f"1950{RSQ}s", "1950s", None),
    "Q-21b": (f"Prices rose in the 1950{RSQ}s, then time-frames changed.", "time-frames", "time frames", None),
}


def _true_anchors(text: str, start: int, end: int) -> tuple[str, str]:
    before_words = text[:start].split()[-2:]
    after_words = text[end:].split()[:2]
    before = text[:start].rstrip()
    before = before[len(before) - len(" ".join(before_words)):] if before_words else ""
    return before.strip(), " ".join(after_words)


@pytest.mark.parametrize("qid", sorted(Q))
def test_q_cases_through_the_resolver(qid: str) -> None:
    text, original, correction, expected = Q[qid]
    start = text.index(original)
    before, after = _true_anchors(text, start, start + len(original))
    result = one(f"<p>{text}</p>", item(original, before, after, correction))
    if expected is None:
        assert isinstance(result, ResolvedFinding), (qid, result)
        assert (result.annotation.start, result.annotation.end) == (start, start + len(original))
    else:
        assert code(result) is expected, (qid, result)


# --------------------------------------------------------------------------------------------
# Nested paragraphs (U3), unmatched / ambiguous quotes, guard
# --------------------------------------------------------------------------------------------


def test_nested_inheritance_and_sibling_isolation() -> None:
    markup = f"<r><p>He said {LDQ}a <p>inner teh</p> b{RDQ} ok <p>other teh</p> end.</p></r>"
    payloads = {1: {"findings": [item("teh", "inner", "", "the")]}, 2: {"findings": [item("teh", "other", "", "the")]}}
    data, scan, results, outcome, records = run(markup, payloads)
    assert statuses(records) == [(1, 0, TraceStatus.REJECTED_S3, IN), (2, 0, TraceStatus.APPLIED, None)]
    assert outcome.output == data.replace(b"other teh", b'other <error type="spelling" correction="the" reason="why">teh</error>')


def test_unmatched_closer_and_ambiguous_straight_quote() -> None:
    assert code(one(f"<p>teh end,{RDQ} he said.</p>", item("teh", "", "end," + RDQ))) is IN
    assert isinstance(one(f"<p>teh end,{RDQ} he said.</p>", item("he", "end," + RDQ, "said.")), ResolvedFinding)
    assert code(one(f"<p>a {SDQ} b teh c</p>", item("teh", "b", "c"))) is IN  # ambiguous: protect to the end
    assert code(one(f"<p>a {SSQ} b teh c</p>", item("teh", "b", "c"))) is IN  # ambiguous single mark (E)
    assert isinstance(one(f"<p>teh c {SDQ}x</p>", item("teh", "", "c")), ResolvedFinding)  # before the mark


def test_guarded_paragraph_rejects_every_finding() -> None:
    marks = " ".join(f"{LDQ}w{RDQ}" for _ in range(33))
    assert code(one(f"<p>teh {marks}</p>", item("teh", "", LDQ + "w" + RDQ))) is IN


def test_production_entry_points_cannot_omit_quotation_protection() -> None:
    import inspect

    assert list(inspect.signature(r.resolve_document).parameters) == ["scan", "payloads"]  # no switch to turn it off
    quotes_param = inspect.signature(r.resolve_paragraph).parameters["quotes"]
    assert quotes_param.default is inspect.Parameter.empty  # required, never defaulted to None


def test_resolve_document_always_passes_the_document_analysis(monkeypatch) -> None:
    markup = (f"<r><p>He said {LDQ}a <p>inner teh</p> b{RDQ} ok</p><p>plain teh here</p>"
              f"<p>{' '.join(f'{LDQ}w{RDQ}' for _ in range(33))} teh</p></r>")
    scan = xs.scan(markup.encode())
    expected = q.analyse_document(scan)
    seen = []
    real = r.resolve_findings

    def spy(paragraph, findings, *, quotes=None):
        seen.append((paragraph.id, quotes))
        return real(paragraph, findings, quotes=quotes)

    monkeypatch.setattr(r, "resolve_findings", spy)
    payloads = {pid: {"findings": [item("teh", "", "", "the")]} for pid in range(len(scan.paragraphs))}
    results = r.resolve_document(scan, payloads)
    assert sorted(pid for pid, _ in seen) == [0, 1, 2, 3]  # every annotatable paragraph is resolved
    assert all(quotes is not None and quotes == expected[pid] for pid, quotes in seen)
    assert len(results) == 4
    assert expected[1].inherited_from == 0 and expected[3].unsafe_reason is not None


def test_quoted_findings_are_rejected_on_every_production_path() -> None:
    markup = (f"<r><p>He said {LDQ}teh{RDQ} now <p>child {LDQ}x{RDQ}</p></p>"
              f"<p>{LDQ}a <p>inner teh</p> b{RDQ}</p><p>{' '.join(f'{LDQ}w{RDQ}' for _ in range(33))} teh</p></r>")
    payloads = {0: {"findings": [item("teh", "said " + LDQ, RDQ + " now", "the")]},  # inside a pair
                3: {"findings": [item("teh", "inner", "", "the")]},  # inherited from the outer quotation
                4: {"findings": [item("teh", RDQ, "", "the")]}}  # guarded paragraph
    scan = xs.scan(markup.encode())
    assert [p.text for p in scan.paragraphs][3] == "inner teh"
    _, _, results, outcome, records = run(markup, payloads)
    assert [(t.paragraph_id, t.status, t.code) for t in records] == [
        (0, TraceStatus.REJECTED_S3, IN), (3, TraceStatus.REJECTED_S3, IN), (4, TraceStatus.REJECTED_S3, IN)]
    assert outcome.output == markup.encode()  # nothing was annotated
    # the same findings WOULD resolve through the legacy building block without an analysis
    checked, _, _ = r.parse_findings(3, payloads[3])
    assert isinstance(r.resolve_findings(scan.paragraphs[3], checked)[0], ResolvedFinding)


def test_resolve_findings_without_quotes_keeps_s3_5_behaviour() -> None:
    p = xs.scan(f"<p>He said {LDQ}teh end{RDQ}</p>".encode()).paragraphs[0]
    checked, _, _ = r.parse_findings(0, {"findings": [item("teh", "said " + LDQ, "end", "the")]})
    assert isinstance(r.resolve_findings(p, checked)[0], ResolvedFinding)
    analysis = q.analyse_paragraph(p)
    assert code(r.resolve_findings(p, checked, quotes=analysis)[0]) is IN


def test_barrier_after_the_span_is_shifted_in_the_corrected_text() -> None:
    # "the boys<p>x</p>bikes": outer text "the boysbikes", barrier at 8 right after "boys".
    # In "the boys'bikes" the barrier moves to 9, after the new apostrophe: possessive, allowed.
    p = xs.scan(b"<r><p>the boys<p>x</p>bikes</p></r>").paragraphs[0]
    assert p.text == "the boysbikes" and p.barriers == (8,)
    result = one("<r><p>the boys<p>x</p>bikes</p></r>", item("boys", "the", "", "boys" + RSQ))
    assert isinstance(result, ResolvedFinding) and (result.annotation.start, result.annotation.end) == (4, 8)


def test_resolve_paragraph_itself_rejects_a_skipped_paragraph() -> None:
    scan = xs.scan(b'<r><p><error type="spelling" correction="x">a</error> b</p></r>')
    p = scan.paragraphs[0]
    result = r.resolve_paragraph(p, {"findings": [item("b"), item("a")]}, q.analyse_paragraph(p))
    assert result.raw_count == 2 and result.resolved == ()
    assert [(x.finding_index, x.code) for x in result.rejections] == [(None, RejectCode.PARA_NOT_ANNOTATABLE)]


def test_resolve_paragraph_refuses_another_paragraphs_analysis() -> None:
    scan = xs.scan(b"<r><p>a</p><p>b</p></r>")
    with pytest.raises(r.ResolverInputError):
        r.resolve_paragraph(scan.paragraphs[0], {"findings": []}, q.analyse_paragraph(scan.paragraphs[1]))


# --------------------------------------------------------------------------------------------
# S3.4 / S3.5 behaviour unchanged through the pipeline
# --------------------------------------------------------------------------------------------

PIPE = "<r><p>Our team will recieve teh final report</p><p>a teh b a teh b</p><p>teh<p>x</p> tial</p></r>"


@pytest.mark.parametrize(
    ("pid", "payload", "expected"),
    [
        (0, {"findings": [item("recieve", "will", "teh")] * 31}, [RejectCode.TOO_MANY_FINDINGS] * 31),
        (0, {"findings": [], "extra": 1}, [RejectCode.SCHEMA_INVALID]),
        (0, None, [RejectCode.SCHEMA_INVALID]),
        (0, {"findings": [{"original": "x"}]}, [RejectCode.INVALID_FIELD]),
        (0, {"findings": [item("a b c d e f g", type_="grammar")]}, [RejectCode.SPAN_TOO_LONG]),
        (0, {"findings": [item("zzz")]}, [RejectCode.NOT_FOUND]),
        (0, {"findings": [item("recieve", "zzz", "teh")]}, [RejectCode.ANCHOR_MISMATCH]),
        (1, {"findings": [item("teh", "a", "b")]}, [RejectCode.AMBIGUOUS]),
        (2, {"findings": [item("teh tial", "", "")]}, [RejectCode.CROSSES_PARAGRAPH]),
        (0, {"findings": [item("recieve", "will", "teh", "recieve")]}, [RejectCode.INVALID_FIELD]),  # no-op
        (4, {"findings": [item("x"), item("y")]}, [RejectCode.PARA_NOT_ANNOTATABLE] * 2),  # unknown id (3 is the nested "x")
        (99, {"findings": [item("x")]}, [RejectCode.PARA_NOT_ANNOTATABLE]),
        (-1, {"findings": [item("x")]}, [RejectCode.INVALID_FIELD]),
    ],
)
def test_s3_4_and_s3_5_codes_are_unchanged_through_the_pipeline(pid: int, payload, expected) -> None:
    _, scan, results, outcome, records = run(PIPE, {pid: payload})
    assert [t.code for t in records] == expected
    assert all(t.status is TraceStatus.REJECTED_S3 and t.paragraph_id == pid for t in records)


# --------------------------------------------------------------------------------------------
# End to end: S2 byte-level safety on spans the resolver hands over (S2 behaviour unchanged)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "finding", "code", "span"),
    [
        (b"<p>my pet <b>dgo</b> here</p>", item("pet dgo", "my", "here", "pet dog"), RejectCode.CROSSES_ELEMENT, (3, 10)),
        (b"<p>very<!--c--> good day</p>", item("very good", "", "day", "verygood"), RejectCode.CROSSES_ELEMENT, (0, 9)),
        (b"<p>a <![CDATA[teh]]> b c</p>", item("a teh b", "", "c", "a the b"), RejectCode.IN_CDATA, (0, 7)),
        (b'<!DOCTYPE p [<!ENTITY co "Acme corp">]><p>See &co; now</p>', item("Acme", "See", "corp now", "ACME"),
         RejectCode.SPLITS_ATOMIC, (4, 8)),
    ],
    ids=["inline-element", "comment", "cdata", "partial-entity"],
)
def test_s2_span_safety_rejects_resolved_spans_and_is_traced(data: bytes, finding: dict, code: RejectCode, span) -> None:
    scan = xs.scan(data)
    results = r.resolve_document(scan, {0: {"findings": [finding]}})
    assert [(rf.annotation.start, rf.annotation.end) for rf in results[0].resolved] == [span]  # S3 resolved it
    outcome = an.annotate(data, scan, r.annotations(results))
    assert outcome.ok and outcome.output == data  # S2 refused it: document byte-identical
    (record,) = r.trace(results, outcome)
    assert (record.status, record.code, record.match_mode, (record.start, record.end), record.detail) == (
        TraceStatus.REJECTED_S2, code, MatchMode.EXACT, span, "")


@pytest.mark.parametrize(
    ("data", "finding", "mode", "expected"),
    [
        (b'<!DOCTYPE p [<!ENTITY co "Acme corp">]><p>See &co; now</p>', item("Acme corp", "See", "now", "Acme Corp"),
         MatchMode.EXACT,
         b'<!DOCTYPE p [<!ENTITY co "Acme corp">]><p>See <error type="spelling" correction="Acme Corp" reason="why">'
         b"&co;</error> now</p>"),
        (b"<p>Salt &amp; peper are here</p>", item("Salt & peper", "", "are", "Salt & pepper"), MatchMode.EXACT,
         b'<p><error type="spelling" correction="Salt &amp; pepper" reason="why">Salt &amp; peper</error> are here</p>'),
        (b"<p>ice\r\ncream cone</p>", item("ice cream", "", "cone", "icecream"), MatchMode.WHITESPACE_GAP,
         b'<p><error type="spelling" correction="icecream" reason="why">ice\r\ncream</error> cone</p>'),
        (b'<r xmlns="urn:x"><p>teh cat</p></r>', item("teh", "", "cat", "the"), MatchMode.EXACT,
         b'<r xmlns="urn:x"><p><error type="spelling" correction="the" reason="why">teh</error> cat</p></r>'),
        (b'<?xml version="1.0" encoding="US-ASCII"?><p>dont go</p>', item("dont", "", "go", "don" + RSQ + "t"),
         MatchMode.EXACT,
         b'<?xml version="1.0" encoding="US-ASCII"?><p><error type="spelling" correction="don&#8217;t" '
         b'reason="why">dont</error> go</p>'),
    ],
    ids=["whole-entity", "amp-reference", "crlf-gap", "default-namespace", "us-ascii-correction"],
)
def test_resolved_spans_render_to_hand_written_bytes(data: bytes, finding: dict, mode: MatchMode, expected: bytes) -> None:
    scan = xs.scan(data)
    results = r.resolve_document(scan, {0: {"findings": [finding]}})
    outcome = an.annotate(data, scan, r.annotations(results))
    assert outcome.ok and outcome.output == expected
    (record,) = r.trace(results, outcome)
    assert (record.status, record.code, record.match_mode) == (TraceStatus.APPLIED, None, mode)


def test_existing_error_paragraph_is_not_annotatable() -> None:
    markup = '<r><p><error type="spelling" correction="x">teh</error> b</p></r>'
    _, _, _, _, records = run(markup, {0: {"findings": [item("b"), item("teh")]}})
    assert statuses(records) == [(0, 0, TraceStatus.REJECTED_S3, RejectCode.PARA_NOT_ANNOTATABLE),
                                 (0, 1, TraceStatus.REJECTED_S3, RejectCode.PARA_NOT_ANNOTATABLE)]


def test_non_int_payload_keys_are_refused() -> None:
    scan = xs.scan(b"<p>a</p>")
    for bad in ("0", True, 0.0):
        with pytest.raises(r.ResolverInputError):
            r.resolve_document(scan, {bad: {"findings": []}})


# --------------------------------------------------------------------------------------------
# Trace schema and statuses
# --------------------------------------------------------------------------------------------


def test_trace_record_contract() -> None:
    assert [f.name for f in dataclasses.fields(TraceRecord)] == [
        "paragraph_id", "finding_index", "status", "code", "match_mode", "start", "end", "hint_drift", "detail"]
    record = TraceRecord(0, 1, TraceStatus.APPLIED, None, MatchMode.EXACT, 2, 5, False, "")
    assert record.entry == TraceEntry(0, 1, TraceStatus.APPLIED, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.status = TraceStatus.REJECTED_S3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"findings": [item("x")] * 3, "extra": 1}, [0, 1, 2]),  # readable list: one record per raw index
        ({"findings": "not a list"}, [None]),  # unreadable: a single record
        (None, [None]),
        ({"findings": [], "extra": 1}, [None]),  # readable but empty: still traceable
    ],
)
def test_paragraph_level_rejections_expand_per_raw_index(payload, expected) -> None:
    _, _, _, _, records = run(PIPE, {0: payload})
    assert [t.finding_index for t in records] == expected
    assert all(t.code is RejectCode.SCHEMA_INVALID for t in records)


def test_skipped_paragraph_expands_to_every_raw_finding() -> None:
    markup = '<r><p><error type="spelling" correction="x">a</error></p></r>'
    _, _, _, _, records = run(markup, {0: {"findings": [item("a"), {"bad": 1}, "x"]}})
    assert [(t.finding_index, t.code) for t in records] == [(i, RejectCode.PARA_NOT_ANNOTATABLE) for i in range(3)]


def test_duplicates_and_conflicts_are_traced_to_s2() -> None:
    markup = "<r><p>a teh b</p><p>c teh d</p></r>"
    payloads = {0: {"findings": [item("teh", "a", "b", "the"), item("teh", "a", "b", "the")]},
                1: {"findings": [item("teh", "c", "d", "the"), item("teh", "c", "d", "tea")]}}
    _, _, _, outcome, records = run(markup, payloads)
    assert statuses(records) == [
        (0, 0, TraceStatus.APPLIED, None), (0, 1, TraceStatus.APPLIED, None),  # duplicates collapse, both applied
        (1, 0, TraceStatus.REJECTED_S2, RejectCode.CONFLICT), (1, 1, TraceStatus.REJECTED_S2, RejectCode.CONFLICT),
    ]
    assert all(t.detail == "" for t in records if t.status is TraceStatus.REJECTED_S2)  # S2 messages not copied


def test_overlap_is_traced_to_s2() -> None:
    payloads = {0: {"findings": [item("teh cat", "the", "sat"), item("cat", "teh", "sat")]}}
    _, _, _, _, records = run("<r><p>the teh cat sat</p></r>", payloads)
    assert [(t.status, t.code) for t in records] == [(TraceStatus.REJECTED_S2, RejectCode.OVERLAP)] * 2


def test_rolled_back_and_document_failed_statuses() -> None:
    markup = "<r><p>a teh b</p><p>c teh d</p></r>"
    payloads = {0: {"findings": [item("teh", "a", "b", "the")]}, 1: {"findings": [item("teh", "c", "d", "the")]}}

    def blame_paragraph_1(original, output, applied):
        real = an.verify(original, output, applied)
        if any(a.paragraph_id == 1 for a in applied):
            return VerificationReport((), ((1, (VerificationIssue("TEXT_CHANGED", "injected", 1),)),))
        return real

    _, _, _, outcome, records = run(markup, payloads, verify_fn=blame_paragraph_1)
    assert outcome.ok and statuses(records) == [(0, 0, TraceStatus.APPLIED, None),
                                                 (1, 0, TraceStatus.ROLLED_BACK, RejectCode.VERIFY_FAILED)]

    def document_failure(original, output, applied):
        return VerificationReport((VerificationIssue("OUTPUT_NOT_WELL_FORMED", "injected"),), ())

    _, _, _, outcome, records = run(markup, payloads, verify_fn=document_failure)
    assert not outcome.ok and [t.status for t in records] == [TraceStatus.DOCUMENT_FAILED] * 2


def test_trace_refuses_an_unrelated_outcome() -> None:
    _, scan, results, _, _ = run("<r><p>a teh b</p></r>", {0: {"findings": [item("teh", "a", "b", "the")]}})
    empty = an.annotate(b"<r><p>a teh b</p></r>", scan, [])
    with pytest.raises(ValueError):
        r.trace(results, empty)


def test_trace_details_never_contain_document_text() -> None:
    markup = f"<r><p>{SECRET} {LDQ}teh{RDQ} {SECRET} x</p></r>"
    payloads = {0: {"findings": [item("teh", SECRET + " " + LDQ, RDQ), item(SECRET), item("x", SECRET, "", SECRET),
                                 {"original": SECRET, SECRET: 1}, item("x", "", "", "x")]}}
    _, _, _, _, records = run(markup, payloads)
    assert len(records) == 5 and all(SECRET not in t.detail for t in records)


def test_trace_is_sorted_and_independent_of_results_order() -> None:
    markup = "<r><p>a teh b</p><p>c teh d</p></r>"
    payloads = {1: {"findings": [item("zzz"), item("teh", "c", "d", "the")]}, 0: {"findings": [item("teh", "a", "b", "the")]}}
    data, scan, results, outcome, records = run(markup, payloads)
    assert [res.paragraph_id for res in results] == [0, 1]  # resolve_document sorts by paragraph id
    assert [(t.paragraph_id, t.finding_index) for t in records] == [(0, 0), (1, 0), (1, 1)]
    assert r.trace(tuple(reversed(results)), outcome) == records


def test_response_order_does_not_change_any_findings_outcome() -> None:
    markup = f"<r><p>He said {LDQ}teh end{RDQ} and teh dogs bark teh cat.</p></r>"
    findings = [item("teh", "said " + LDQ, "end", "the"), item("teh", "and", "dogs", "the"), item("zzz"),
                {"bad": 1}, item("teh", "bark", "cat.", "the"), item("dogs", "teh", "bark", "dog")]
    baseline = run(markup, {0: {"findings": findings}})
    by_finding = {repr(f): (t.status, t.code, t.start, t.end) for f, t in zip(findings, baseline[4])}
    for seed in range(5):
        order = list(range(len(findings)))
        random.Random(seed).shuffle(order)
        shuffled = [findings[i] for i in order]
        data, _, _, outcome, records = run(markup, {0: {"findings": shuffled}})
        assert outcome.output == baseline[3].output
        assert {repr(f): (t.status, t.code, t.start, t.end) for f, t in zip(shuffled, records)} == by_finding


# --------------------------------------------------------------------------------------------
# Adversarial
# --------------------------------------------------------------------------------------------


def test_adversarial_mark_heavy_paragraph_and_many_findings() -> None:
    marks = (LDQ + RSQ + SDQ + SSQ) * 40
    markup = f"<r><p>teh {marks} teh</p><p>{' '.join(['teh'] * 500)}</p></r>"
    payloads = {0: {"findings": [item("teh", "", marks[:4])] * 30},
                1: {"findings": [item("teh", "teh", "teh", "the", ws=k, we=k) for k in range(1, 31)]}}
    _, _, _, outcome, records = run(markup, payloads)
    assert outcome.ok and len(records) == 60
    assert all(t.code is IN for t in records[:30])  # guard (more than 64 marks)
    assert sum(t.status is TraceStatus.APPLIED for t in records[30:]) == 30  # exact hints separate them


# --------------------------------------------------------------------------------------------
# Properties: trace completeness, ordering, determinism, safety
# --------------------------------------------------------------------------------------------

WORDS = ["teh", "cat", "a", "the", "dogs" + RSQ, "x,", "end."]
MARKS = [LDQ, RDQ, SDQ, LSQ, RSQ, SSQ] + [""] * 30  # quotation marks on a minority of tokens
CORRECTIONS = ["the", "dog" + RSQ + "s", "dogs" + RSQ, LDQ + "x" + RDQ, "teh", "x", RSQ + "x"]
token_st = st.tuples(st.sampled_from(MARKS), st.sampled_from(WORDS), st.sampled_from(MARKS))
documents = st.lists(st.lists(token_st, min_size=1, max_size=8), min_size=1, max_size=3)


def _text(tokens) -> str:
    return " ".join(a + w + b for a, w, b in tokens)


def _true_finding(tokens, i: int, correction: str) -> dict:
    """The inner word of token i with its verbatim anchors and exact word hint (a genuine finding)."""
    a, w, b = tokens[i]
    before = (" ".join(x + y + z for x, y, z in tokens[max(0, i - 1):i]) + " " + a).strip()
    after = (b + " " + " ".join(x + y + z for x, y, z in tokens[i + 1:i + 2])).strip()
    return item(w, before, after, correction, ws=i, we=i)


def _findings_for(tokens, data) -> list:
    words = sorted({w for _, w, _ in tokens} | {"zzz"})
    true = st.builds(lambda i, c: _true_finding(tokens, i, c), st.integers(0, len(tokens) - 1), st.sampled_from(CORRECTIONS))
    random_ = st.builds(lambda o, b, c: item(o, b, "", c), st.sampled_from(words), st.sampled_from(words + [""]),
                        st.sampled_from(CORRECTIONS))
    bad = st.sampled_from([{"bad": 1}, "text", item("", "", "")])
    findings = data.draw(st.lists(st.one_of(true, true, true, random_, bad), max_size=8))
    if data.draw(st.integers(0, 9)) == 0:
        findings = findings * 31 or [{"bad": 1}] * 31  # occasionally more than 30 raw findings
    return findings


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(documents, st.data())
def test_property_trace_is_complete_ordered_deterministic_and_safe(docs, data) -> None:
    markup = "<r>" + "".join(f"<p>{_text(tokens)}</p>" for tokens in docs) + "</r>"
    payloads = {pid: {"findings": _findings_for(tokens, data)} for pid, tokens in enumerate(docs)}
    data_bytes, scan, results, outcome, records = run(markup, payloads)
    assert outcome.ok  # no fault injection: verification always passes

    # completeness: one record per raw finding (or one per unreadable paragraph), sorted
    expected = [(pid, i) for pid in sorted(payloads) for i in (range(len(payloads[pid]["findings"])) or [None])]
    if any(len(p["findings"]) > 30 for p in payloads.values()):
        assert all(t.code is RejectCode.TOO_MANY_FINDINGS for t in records
                   if len(payloads[t.paragraph_id]["findings"]) > 30)
    assert [(t.paragraph_id, t.finding_index) for t in records] == [
        (pid, i) for pid, i in expected if not (i is None and len(payloads[pid]["findings"]) == 0)]

    # consistency with the annotation outcome
    applied = set(outcome.applied)
    analyses = q.analyse_document(scan)
    # no bypass: EVERY finding resolve_document resolved respects document-level quotation protection
    for res in results:
        for rf in res.resolved:
            assert q.span_verdict(analyses[res.paragraph_id], rf.annotation.start, rf.annotation.end) is None
    for t in records:
        if t.status is TraceStatus.APPLIED:
            (rf,) = [x for res in results if res.paragraph_id == t.paragraph_id for x in res.resolved
                     if x.finding_index == t.finding_index]
            assert rf.annotation in applied
            # safety: applied spans never touch or overlap protected quotation content
            assert q.span_verdict(analyses[t.paragraph_id], t.start, t.end) is None
            assert not any(q.is_quote_character(ch) and ch not in (RSQ, SSQ) for ch in rf.annotation.correction)
        else:
            assert t.status is not TraceStatus.DOCUMENT_FAILED

    # determinism
    assert run(markup, payloads)[4] == records

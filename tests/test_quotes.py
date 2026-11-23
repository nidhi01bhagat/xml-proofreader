"""S3.2: quotation-mark classification, pairing (rule S), protected ranges and guards.

All texts are invented. Q-01...Q-21 are the cases of docs/03 section 8; a test checks that the
data below still matches that table, so the expectations are the documented ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from proofreader import quotes as q
from proofreader import xml_source as xs
from proofreader.models import ProtectedRange, QuoteFamily, QuoteRole, RangeOrigin, RejectCode

LDQ, RDQ, LOWD, REVD = chr(0x201C), chr(0x201D), chr(0x201E), chr(0x201F)
SDQ, FWDQ, LAQ, RAQ = chr(0x22), chr(0xFF02), chr(0xAB), chr(0xBB)
LSAQ, RSAQ = chr(0x2039), chr(0x203A)
LSQ, RSQ, SSQ, LOWS, REVS = chr(0x2018), chr(0x2019), chr(0x27), chr(0x201A), chr(0x201B)
LC, RC, LWC, RWC = chr(0x300C), chr(0x300D), chr(0x300E), chr(0x300F)
DP_OPEN, DP_CLOSE, DP_LOW = chr(0x301D), chr(0x301E), chr(0x301F)
PRIME, DPRIME = chr(0x2032), chr(0x2033)
OTHER_PI = chr(0x2E02)  # LEFT SUBSTITUTION BRACKET (category Pi), not in the table
COMBINING_ACUTE = chr(0x0301)

IN, TOUCH, OK = RejectCode.IN_QUOTATION, RejectCode.TOUCHES_QUOTATION, None


def verdict(text: str, finding: str, occurrence: int = 0):
    analysis = q.analyse_text(text)
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(finding, start + 1)
    return q.span_verdict(analysis, start, start + len(finding))


def roles(text: str, barriers=()) -> list[tuple[str, QuoteRole]]:
    return [(m.char, m.role) for m in q.classify(text, barriers)]


# --------------------------------------------------------------------------------------------
# Q-01 ... Q-20 (docs/03 section 8), executed
# --------------------------------------------------------------------------------------------

Q_TEXT = {
    "Q-01": f"He said {LDQ}teh end{RDQ} today.",
    "Q-03": f"He said {SDQ}teh end{SDQ} today.",
    "Q-04": f"{LDQ}He said {LSQ}teh{RSQ} loudly,{RDQ} she noted.",
    "Q-06": f"She wrote {SDQ}teh end. Then left.",
    "Q-07": f"Teh clerk wrote {SDQ}done.",
    "Q-08": f"The note read {RSQ}keep left{RSQ}.",
    "Q-09": f"Don{SSQ}t move teh box.",
    "Q-10": f"The boys{RSQ} bikes are red.",
    "Q-11": f"The {LSQ}dogs{RSQ} bowls{RSQ} were teh issue.",
    "Q-13": f"{SSQ}Tis teh season.",
    "Q-14": f"teh end,{RDQ} he said.",
    "Q-15": f"In the {RSQ}90s teh market grew.",
    "Q-16": f"{LAQ} teh fin {RAQ} dit-il.",
    "Q-17": f"He said {SDQ}yes{SDQ}.",
    "Q-18": f"{LDQ}Don{SSQ}t move teh box,{RDQ} she said.",
    "Q-19": f"A 6{SDQ} pipe and teh valve.",
    "Q-20": f"He wrote {LDQ}teh{RDQ} and {LDQ}then{RDQ}.",
    "Q-21": f"Prices rose in the 1950{RSQ}s, then time-frames changed.",  # S4.8e digit-letter apostrophe
}
Q_TEXT["Q-02"], Q_TEXT["Q-05"], Q_TEXT["Q-12"] = Q_TEXT["Q-01"], Q_TEXT["Q-04"], Q_TEXT["Q-11"]

# (id, findings with expected verdicts) - findings exactly as in the table's "Finding" column
Q_CASES = [
    ("Q-01", [("teh", IN)]),
    ("Q-02", [("today", OK)]),
    ("Q-03", [("teh", IN)]),
    ("Q-04", [("teh", IN), ("loudly", IN)]),
    ("Q-05", [("noted", OK)]),
    ("Q-06", [("Then", IN)]),
    ("Q-07", [("Teh", OK)]),
    ("Q-08", [("keep", IN), ("read", OK)]),
    ("Q-09", [(f"Don{SSQ}t", OK)]),
    ("Q-10", [(f"boys{RSQ}", OK)]),
    ("Q-11", [("bowls", IN)]),
    ("Q-12", [("teh", OK)]),
    ("Q-13", [("teh", IN)]),
    ("Q-14", [("teh", IN), ("he said", OK)]),
    ("Q-15", [("teh", OK)]),
    ("Q-16", [("teh", IN)]),
    ("Q-17", [(f"{SDQ}yes{SDQ}", TOUCH)]),
    ("Q-18", [("teh", IN)]),
    ("Q-19", [("teh", IN)]),
    ("Q-20", [("then", IN)]),
    ("Q-21", [(f"1950{RSQ}s", OK), ("time-frames", OK)]),
]


@pytest.mark.parametrize(("qid", "findings"), Q_CASES, ids=[c[0] for c in Q_CASES])
def test_q_cases(qid: str, findings) -> None:
    text = Q_TEXT[qid]
    for finding, expected in findings:
        assert verdict(text, finding) == expected, (qid, finding)


def test_q_case_data_matches_the_documented_table() -> None:
    doc = Path(__file__).resolve().parents[1] / "docs" / "03-final-architecture.md"
    rows = {}
    for line in doc.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Q-"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows[cells[0]] = cells
    assert sorted(rows) == sorted(c[0] for c in Q_CASES) == [f"Q-{i:02d}" for i in range(1, 22)]
    word = {"IN_QUOTATION": IN, "TOUCHES_QUOTATION": TOUCH, "accept": OK}
    for qid, findings in Q_CASES:
        qid_, text_cell, finding_cell, expected_cell = rows[qid][:4]
        if text_cell.startswith("same as "):
            assert Q_TEXT[qid] == Q_TEXT[text_cell.removeprefix("same as ")]
        else:
            assert text_cell == f"`{Q_TEXT[qid]}`", qid
        documented = [part.split("`")[1] for part in finding_cell.split(" / ")]
        assert documented == [f for f, _ in findings], qid
        expected = [word[e.split(" (")[0].strip()] for e in expected_cell.split(" / ")]
        assert expected == [e for _, e in findings], qid


# --------------------------------------------------------------------------------------------
# Classification table (decision E), one test per row
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("char", "family", "role"),
    [
        (LDQ, QuoteFamily.DOUBLE, QuoteRole.OPEN),
        (RDQ, QuoteFamily.DOUBLE, QuoteRole.CLOSE),
        (LOWD, QuoteFamily.DOUBLE, QuoteRole.OPEN),
        (REVD, QuoteFamily.DOUBLE, QuoteRole.OPEN),
        (LAQ, QuoteFamily.DOUBLE, QuoteRole.OPEN),
        (RAQ, QuoteFamily.DOUBLE, QuoteRole.CLOSE),
        (LSAQ, QuoteFamily.SINGLE, QuoteRole.OPEN),
        (RSAQ, QuoteFamily.SINGLE, QuoteRole.CLOSE),
        (LOWS, QuoteFamily.SINGLE, QuoteRole.OPEN),
        (REVS, QuoteFamily.SINGLE, QuoteRole.OPEN),
        (LC, QuoteFamily.CORNER, QuoteRole.OPEN),
        (RC, QuoteFamily.CORNER, QuoteRole.CLOSE),
        (LWC, QuoteFamily.WHITE_CORNER, QuoteRole.OPEN),
        (RWC, QuoteFamily.WHITE_CORNER, QuoteRole.CLOSE),
        (DP_OPEN, QuoteFamily.DOUBLE, QuoteRole.OPEN),
        (DP_CLOSE, QuoteFamily.DOUBLE, QuoteRole.CLOSE),
        (DP_LOW, QuoteFamily.DOUBLE, QuoteRole.CLOSE),
        (OTHER_PI, QuoteFamily.OTHER, QuoteRole.AMBIGUOUS),
    ],
)
def test_fixed_role_marks(char: str, family: QuoteFamily, role: QuoteRole) -> None:
    (mark,) = q.classify(f"a {char}b")
    assert (mark.char, mark.family, mark.role) == (char, family, role)


@pytest.mark.parametrize("straight", [SDQ, FWDQ])
@pytest.mark.parametrize(
    ("template", "role"),
    [
        ("say {q}hi", QuoteRole.OPEN),  # boundary before, non-space after
        ("({q}hi", QuoteRole.OPEN),  # ( is a boundary
        ("{q}hi", QuoteRole.OPEN),  # paragraph start
        ("hi{q} there", QuoteRole.CLOSE),  # non-space before, boundary after
        ("hi{q}.", QuoteRole.CLOSE),  # punctuation after
        ("hi{q}", QuoteRole.CLOSE),  # paragraph end
        ("a {q} b", QuoteRole.AMBIGUOUS),  # spaces on both sides
        ("a{q}b", QuoteRole.AMBIGUOUS),  # letters on both sides
    ],
)
def test_straight_double_context_rule(straight: str, template: str, role: QuoteRole) -> None:
    assert roles(template.format(q=straight)) == [(straight, role)]


@pytest.mark.parametrize("apos", [RSQ, SSQ])
@pytest.mark.parametrize(
    ("template", "role"),
    [
        ("don{a}t", QuoteRole.APOSTROPHE),  # letter-letter
        ("the {a}90s", QuoteRole.APOSTROPHE),  # boundary-digit
        ("boys{a} bikes", QuoteRole.CLOSE_OR_APOSTROPHE),  # letter-boundary
        ("boys{a}.", QuoteRole.CLOSE_OR_APOSTROPHE),  # letter-punct
        ("the 90{a} line", QuoteRole.CLOSE_OR_APOSTROPHE),  # number-boundary
        ("a {a}tis", QuoteRole.OPEN_CANDIDATE),  # boundary-letter
        ("a {a} b", QuoteRole.AMBIGUOUS),  # boundary-boundary
        ("19{a}s", QuoteRole.APOSTROPHE),  # digit-letter (plural year): apostrophe since S4.8e (was AMBIGUOUS)
        (f"see{{a}}{RDQ}", QuoteRole.AMBIGUOUS),  # a quotation mark is not "punct"
    ],
)
def test_apostrophe_like_context_rule(apos: str, template: str, role: QuoteRole) -> None:
    marks = [r for r in roles(template.format(a=apos)) if r[0] == apos]
    assert marks == [(apos, role)]


@pytest.mark.parametrize(
    ("text", "role"),
    [("a " + LSQ + "word", QuoteRole.OPEN), ("don" + LSQ + "t", QuoteRole.AMBIGUOUS), (LSQ, QuoteRole.OPEN)],
)
def test_left_single_quote(text: str, role: QuoteRole) -> None:
    assert roles(text) == [(LSQ, role)]


def test_primes_and_ordinary_punctuation_are_not_quotation_marks() -> None:
    assert q.classify(f"a 6{PRIME} by 4{DPRIME} board, (yes) - ok!") == ()
    assert not q.is_quote_character(PRIME) and not q.is_quote_character(DPRIME)


def test_combining_marks_are_skipped_to_the_base_letter() -> None:
    # "cafe" + combining acute, then an apostrophe and a letter: letter-letter -> apostrophe
    assert roles(f"cafe{COMBINING_ACUTE}{RSQ}s") == [(RSQ, QuoteRole.APOSTROPHE)]


def test_barriers_count_as_boundaries() -> None:
    text = f"x{SSQ}y"  # letter-letter without a barrier
    assert roles(text) == [(SSQ, QuoteRole.APOSTROPHE)]
    assert roles(text, barriers=(1,)) == [(SSQ, QuoteRole.OPEN_CANDIDATE)]  # barrier before the mark
    assert roles(text, barriers=(2,)) == [(SSQ, QuoteRole.CLOSE_OR_APOSTROPHE)]  # barrier after it


# --------------------------------------------------------------------------------------------
# Pairing and ranges
# --------------------------------------------------------------------------------------------


def ranges(text: str) -> list[tuple[int, int, bool, bool, RangeOrigin]]:
    return [(r.start, r.end, r.open_start, r.open_end, r.origin) for r in q.analyse_text(text).ranges]


def test_pair_includes_both_marks() -> None:
    text = f"a {LDQ}b{RDQ} c"
    assert ranges(text) == [(2, 5, False, False, RangeOrigin.PAIR)]


def test_closer_without_opener_protects_from_paragraph_start() -> None:
    assert ranges(f"ab{RDQ} c") == [(0, 3, True, False, RangeOrigin.EMPTY_STACK_CLOSE)]


def test_unclosed_opener_protects_to_paragraph_end() -> None:
    assert ranges(f"a {LAQ}b c") == [(2, 6, False, True, RangeOrigin.UNCLOSED)]


def test_mismatched_closer_does_not_close_another_family_member() -> None:
    # a guillemet closer cannot close a curly double opener: both get their own ranges
    text = f"a {LDQ}b{RAQ} c"  # closer at 4 -> [0, 5); unclosed opener at 2 -> [2, 7); merged
    assert len(text) == 7
    assert ranges(text) == [(0, 7, True, True, RangeOrigin.EMPTY_STACK_CLOSE)]


def test_nearest_matching_opener_is_used_and_others_stay_open() -> None:
    text = f"{LAQ}a {LDQ}b{RAQ} c"  # the guillemet pair encloses an unclosed curly opener
    analysis = q.analyse_text(text)
    assert [(r.start, r.end) for r in analysis.ranges] == [(0, len(text))]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # rule S: protect to the LAST close-or-apostrophe after the opener
        (f"x {LSQ}dogs{RSQ} and cats{RSQ} y", [(2, 18, False, False, RangeOrigin.PAIR)]),  # last mark at 17
        (f"x {LSQ}dogs y", [(2, 9, False, True, RangeOrigin.UNCLOSED)]),
        (f"a {LOWS}b c", [(2, 6, False, True, RangeOrigin.UNCLOSED)]),
        (f"a {REVS}b c{RSQ} d", [(2, 7, False, False, RangeOrigin.PAIR)]),  # closing mark at 6
        # single guillemets are a definite pair, not rule S
        (f"a {LSAQ}b{RSAQ} c", [(2, 5, False, False, RangeOrigin.PAIR)]),
        (f"a {LC}b{RC} c", [(2, 5, False, False, RangeOrigin.PAIR)]),
        (f"a {LWC}b{RWC} c", [(2, 5, False, False, RangeOrigin.PAIR)]),
        (f"a {DP_OPEN}b{DP_LOW} c", [(2, 5, False, False, RangeOrigin.PAIR)]),
        # ambiguous marks protect to the end
        (f"a {SSQ} b c", [(2, 7, False, True, RangeOrigin.AMBIGUOUS)]),
        (f"a {OTHER_PI}b c", [(2, 6, False, True, RangeOrigin.AMBIGUOUS)]),
    ],
)
def test_pairing_rules(text: str, expected) -> None:
    assert ranges(text) == expected


def test_closer_pairs_with_the_nearest_matching_opener() -> None:
    # outer and inner curly openers, one closer: the INNER opener is paired, the outer stays unclosed
    text = f"x {LDQ}a {LDQ}b{RDQ} c d"
    assert ranges(text) == [(2, len(text), False, True, RangeOrigin.UNCLOSED)]


def test_merged_range_keeps_open_start_of_a_shorter_member() -> None:
    # unclosed guillemet at 0 -> [0, n); unmatched curly closer -> [0, 4) reaching the start
    text = f"{LAQ}a {RDQ} b"
    assert ranges(text) == [(0, len(text), True, True, RangeOrigin.UNCLOSED)]


def test_depth_counts_half_open_ranges_touching_ranges_do_not_nest() -> None:
    # 8 rule-S ranges end exactly where an unclosed opener's range starts (a barrier separates them)
    body = f"{LSQ}a " * 8 + f"end{RSQ}"
    text = body + f"{LDQ}x y"
    analysis = q.analyse_text(text, barriers=(len(body),))
    assert analysis.unsafe_reason is None  # depth is 8, not 9
    assert [(r.start, r.end) for r in analysis.ranges] == [(0, len(text))]


def test_german_low_high_quotes_never_close_and_protect_to_the_paragraph_end() -> None:
    # „…“ : the table classifies both „ and “ as OPEN (no German pairing rule), so nothing closes and
    # protection runs to the paragraph end - the documented conservative loss (02 §6.1)
    text = f"Er sagte {LOWD}teh{LDQ} und ging."
    assert roles(text) == [(LOWD, QuoteRole.OPEN), (LDQ, QuoteRole.OPEN)]
    assert ranges(text) == [(9, len(text), False, True, RangeOrigin.UNCLOSED)]
    assert verdict(text, "ging") == IN  # after the quotation, still protected
    assert verdict(text, "sagte") is OK  # before it, editable


def test_close_or_apostrophe_outside_ranges_becomes_an_apostrophe() -> None:
    analysis = q.analyse_text(f"The boys{RSQ} bikes.")
    assert [m.role for m in analysis.marks] == [QuoteRole.APOSTROPHE] and analysis.ranges == ()
    inside = q.analyse_text(f"The {LSQ}dogs{RSQ} bowls{RSQ} were.")
    assert [m.role for m in inside.marks] == [QuoteRole.OPEN, QuoteRole.CLOSE_OR_APOSTROPHE, QuoteRole.CLOSE_OR_APOSTROPHE]


def test_odd_straight_double_count_protects_from_the_first_unmatched_mark() -> None:
    text = f"a {SDQ}b{SDQ} c {SDQ}d e"
    assert ranges(text) == [(2, 5, False, False, RangeOrigin.PAIR), (8, 12, False, True, RangeOrigin.UNCLOSED)]


def test_span_verdict_adjacency_and_apostrophes() -> None:
    text = f"x {LDQ}y{RDQ}z w"
    analysis = q.analyse_text(text)  # protected [2, 5)
    assert q.span_verdict(analysis, 0, 1) is None
    assert q.span_verdict(analysis, 5, 6) == TOUCH  # starts where the range ends (decision G)
    assert q.span_verdict(analysis, 1, 2) == TOUCH  # ends where the range starts
    assert q.span_verdict(analysis, 3, 4) == IN
    assert q.span_verdict(analysis, 2, 5) == TOUCH  # contains quotation marks: checked first
    plain = q.analyse_text(f"don{RSQ}t go")
    assert q.span_verdict(plain, 0, 5) is None  # apostrophes never block


def test_analyse_paragraph_uses_the_scanned_paragraph() -> None:
    data = f"<r><p>He said {LDQ}teh end{RDQ} today.</p></r>".encode()
    paragraph = xs.scan(data).paragraphs[0]
    analysis = q.analyse_paragraph(paragraph)
    assert analysis.paragraph_id == paragraph.id and analysis.inherited_from is None
    assert analysis == q.analyse_text(paragraph.text, paragraph.barriers, paragraph.id)


def test_analyse_paragraph_sees_reference_and_cdata_text() -> None:
    data = b"<p>He said &#8220;teh <![CDATA[end]]>&#8221; today.</p>"
    paragraph = xs.scan(data).paragraphs[0]
    analysis = q.analyse_paragraph(paragraph)
    start = paragraph.text.index("teh")
    assert q.span_verdict(analysis, start, start + 3) == IN


def test_analyse_paragraph_treats_barriers_as_boundaries() -> None:
    data = f"<r><p>x{SSQ}<p>inner</p>y</p></r>".encode()  # x'|y with a nested <p> after the mark
    outer = xs.scan(data).paragraphs[0]
    assert outer.text == f"x{SSQ}y" and outer.barriers == (2,)
    assert roles(outer.text, outer.barriers) == [(SSQ, QuoteRole.CLOSE_OR_APOSTROPHE)]  # local role
    final = q.analyse_paragraph(outer)
    assert final.ranges == () and [m.role for m in final.marks] == [QuoteRole.APOSTROPHE]  # nothing opened


# --------------------------------------------------------------------------------------------
# Guard (decision F)
# --------------------------------------------------------------------------------------------


def _pairs(k: int) -> str:
    return " ".join(f"{LDQ}w{RDQ}" for _ in range(k))  # 2k marks, depth 1


def test_guard_triggers_above_64_marks_only() -> None:
    at_limit = q.analyse_text(_pairs(32))  # 64 marks
    assert at_limit.unsafe_reason is None and len(at_limit.ranges) == 32
    over = q.analyse_text(_pairs(32) + f" {LDQ}")  # 65 marks
    assert over.unsafe_reason and "64" in over.unsafe_reason
    assert [(r.start, r.end, r.open_start, r.open_end, r.origin) for r in over.ranges] == [
        (0, len(_pairs(32)) + 2, True, True, RangeOrigin.GUARD)]


def test_apostrophes_do_not_count_towards_the_mark_guard() -> None:
    text = " ".join(f"don{RSQ}t" for _ in range(100))
    analysis = q.analyse_text(text)
    assert analysis.unsafe_reason is None and analysis.ranges == ()


def test_guard_triggers_above_depth_8_only() -> None:
    def nested(k: int) -> str:
        return f"{LDQ}a " * k + "core" + f" b{RDQ}" * k

    assert q.analyse_text(nested(8)).unsafe_reason is None
    deep = q.analyse_text(nested(9))
    assert deep.unsafe_reason and "8" in deep.unsafe_reason
    assert deep.ranges[0].origin is RangeOrigin.GUARD


def test_empty_text_has_no_ranges_even_when_guarded() -> None:
    assert q.analyse_text("").ranges == ()


# --------------------------------------------------------------------------------------------
# Adversarial inputs
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        SDQ * 10_000,
        (LDQ + RDQ) * 5_000,
        (LSQ + "a" + RSQ + " ") * 3_000,
        "".join(sorted(q.QUOTE_CHARACTERS)) * 500,
        (RDQ + "x") * 4_000,
        "a" * 50_000 + LDQ,
    ],
    ids=["straight", "alternating-curly", "rule-s-heavy", "every-mark", "closers", "long-tail"],
)
def test_adversarial_inputs_are_handled_conservatively(text: str) -> None:
    analysis = q.analyse_text(text)
    # Deterministic instead of a wall-clock bound: mark-heavy inputs must trip the guard, which skips
    # pairing entirely (that is what keeps them linear); timing is measured, not asserted (docs/04 risk 10).
    counted = sum(1 for m in q.classify(text) if m.role is not QuoteRole.APOSTROPHE)
    assert (analysis.unsafe_reason is not None) is (counted > q.MAX_QUOTE_MARKS)
    if any(m.role is not QuoteRole.APOSTROPHE for m in analysis.marks):
        assert analysis.ranges  # something is always protected
    _check_invariants(text, analysis)


# --------------------------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------------------------

ALPHABET = list("abcXY 09.,;!?-()\n") + [LDQ, RDQ, LOWD, SDQ, FWDQ, LAQ, RAQ, LSAQ, RSAQ, LSQ, RSQ, SSQ,
                                         LOWS, REVS, LC, RC, LWC, RWC, DP_OPEN, DP_CLOSE, PRIME, OTHER_PI,
                                         COMBINING_ACUTE, chr(0x2014)]
TEXTS = st.text(alphabet=st.sampled_from(ALPHABET), max_size=80)
SETTINGS = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _check_invariants(text: str, analysis) -> None:
    n = len(text)
    previous_end = -1
    for r in analysis.ranges:
        assert 0 <= r.start < r.end <= n
        assert r.start > previous_end  # sorted, disjoint and not touching (merged)
        previous_end = r.end
        assert not r.open_start or r.start == 0
        assert not r.open_end or r.end == n
    for m in analysis.marks:
        assert text[m.offset] == m.char and q.is_quote_character(m.char)
        if m.role is not QuoteRole.APOSTROPHE:  # every quotation mark is protected
            assert any(r.start <= m.offset < r.end for r in analysis.ranges)
    if analysis.unsafe_reason is not None:
        assert [(r.start, r.end) for r in analysis.ranges] == ([(0, n)] if n else [])


@SETTINGS
@given(TEXTS)
def test_property_invariants(text: str) -> None:
    analysis = q.analyse_text(text)
    _check_invariants(text, analysis)
    assert q.analyse_text(text) == analysis  # deterministic


@SETTINGS
@given(TEXTS)
def test_property_every_quotation_character_is_found(text: str) -> None:
    offsets = [m.offset for m in q.classify(text)]
    assert offsets == [i for i, ch in enumerate(text) if q.is_quote_character(ch)]
    assert all(text[i] not in (PRIME, chr(0x2033)) for i in offsets)


@SETTINGS
@given(TEXTS, st.data())
def test_property_spans_never_touch_protected_text_when_accepted(text: str, data) -> None:
    analysis = q.analyse_text(text)
    if not text:
        return
    start = data.draw(st.integers(0, len(text) - 1))
    end = data.draw(st.integers(start + 1, len(text)))
    if q.span_verdict(analysis, start, end) is None:
        for r in analysis.ranges:
            assert end < r.start or start > r.end  # no overlap, no adjacency
        assert all(m.role is QuoteRole.APOSTROPHE for m in analysis.marks if start <= m.offset < end)


def test_scanned_fixtures_analyse_without_error() -> None:
    fixtures = Path(__file__).resolve().parent / "fixtures" / "xml"
    for path in sorted(fixtures.iterdir()):
        try:
            scan = xs.scan(path.read_bytes())
        except Exception:
            continue  # inputs S1 rejects never reach S3
        for paragraph in scan.paragraphs:
            analysis = q.analyse_paragraph(paragraph)
            _check_invariants(paragraph.text, analysis)


# --------------------------------------------------------------------------------------------
# S4.8e: digit-letter apostrophe (plural years) is an APOSTROPHE, not an unpaired quotation mark
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("apos", [RSQ, SSQ])
def test_plural_year_apostrophe_does_not_open_a_quotation(apos: str) -> None:
    text = f"Prices rose in the 1950{apos}s, and later time-frames changed."
    analysis = q.analyse_text(text)
    assert [(m.char, m.role) for m in analysis.marks] == [(apos, QuoteRole.APOSTROPHE)]
    assert analysis.ranges == () and analysis.unsafe_reason is None  # nothing protected


@pytest.mark.parametrize("apos", [RSQ, SSQ])
def test_text_after_a_plural_year_is_not_protected(apos: str) -> None:
    text = f"Prices rose in the 1950{apos}s, and later time-frames changed."
    assert verdict(text, f"1950{apos}s") is OK  # the plural-year correction itself is allowed
    assert verdict(text, "time-frames") is OK  # a later finding in the same paragraph is allowed
    assert verdict(text, "Prices") is OK


@pytest.mark.parametrize("apos", [RSQ, SSQ])
def test_quoted_material_containing_a_plural_year_stays_protected(apos: str) -> None:
    text = f"He wrote {LDQ}the 1990{apos}s were teh best{RDQ} and left."
    assert verdict(text, "teh") is IN  # inside the genuine quotation
    assert verdict(text, f"1990{apos}s") is IN
    assert verdict(text, "left") is OK  # after the closing mark
    assert [r[:2] for r in ranges(text)] == [(9, 9 + len(f"{LDQ}the 1990{apos}s were teh best{RDQ}"))]


def test_unmatched_and_other_apostrophe_rules_are_unchanged() -> None:
    assert ranges(f"She wrote {LDQ}teh end. Then left.")[0][3] is True  # unclosed opener: to paragraph end
    assert verdict(f"The note read {RSQ}keep left{RSQ}.", "keep") is IN  # Q-08 open-candidate pairing
    assert roles(f"the {RSQ}90s") == [(RSQ, QuoteRole.APOSTROPHE)]  # boundary-digit unchanged
    assert roles(f"the 90{RSQ} line") == [(RSQ, QuoteRole.CLOSE_OR_APOSTROPHE)]  # digit-boundary unchanged
    assert roles(f"a {RSQ}tis") == [(RSQ, QuoteRole.OPEN_CANDIDATE)]  # boundary-letter unchanged
    assert roles(f"x{LSQ}s") == [(LSQ, QuoteRole.AMBIGUOUS)]  # a LEFT single quote between characters: unchanged
    assert roles(f"19{LSQ}s") == [(LSQ, QuoteRole.OPEN)]  # only apostrophe-like marks gained the digit-letter rule

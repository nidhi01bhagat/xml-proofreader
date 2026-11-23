"""S3.4: parsing and validation of a paragraph's raw model response (resolver steps 1-7).

Matching, anchors and quotation checks are not part of this step. All data is invented.
"""

from __future__ import annotations

import copy
import dataclasses
import random
import unicodedata
from pathlib import Path
from xml.sax.saxutils import escape

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from proofreader import resolver as r
from proofreader import tokens
from proofreader import xml_source as xs
from proofreader.models import (
    Annotation,
    Category,
    CheckedFinding,
    FindingRejection,
    MatchMode,
    Paragraph,
    RejectCode,
    ResolvedFinding,
)

PID = 3
SECRET = "UNIQUE-DOCUMENT-TEXT"


def item(**overrides) -> dict:
    data = {
        "original": "recieve",
        "before": "Our team will",
        "after": "the final report",
        "word_start": 3,
        "word_end": 3,
        "correction": "receive",
        "type": "spelling",
        "reason": "Misspelling",
    }
    data.update(overrides)
    return data


def parse(payload):
    return r.parse_findings(PID, payload)


def codes(rejections) -> list[tuple[int | None, RejectCode]]:
    return [(x.finding_index, x.code) for x in rejections]


# --------------------------------------------------------------------------------------------
# Step 1: paragraph check
# --------------------------------------------------------------------------------------------

SCAN = xs.scan(b'<r><p><error type="spelling" correction="x">teh</error> b</p><p>ok teh</p><p> </p></r>')


@pytest.mark.parametrize(
    ("paragraph_id", "code"),
    [
        (1, None),  # annotatable
        (0, RejectCode.PARA_NOT_ANNOTATABLE),  # already contains <error> (D11)
        (2, RejectCode.PARA_NOT_ANNOTATABLE),  # empty
        (3, RejectCode.PARA_NOT_ANNOTATABLE),  # out of range: unknown id
        (10**9, RejectCode.PARA_NOT_ANNOTATABLE),
        (-1, RejectCode.INVALID_FIELD),  # negative (S2 precedent)
    ],
)
def test_paragraph_rejection(paragraph_id: int, code) -> None:
    result = r.paragraph_rejection(SCAN, paragraph_id)
    if code is None:
        assert result is None
    else:
        assert isinstance(result, FindingRejection) and result.code is code
        assert result.finding_index is None and result.detail
        assert type(result.paragraph_id) is int and result.paragraph_id == paragraph_id  # type-safe contract


@pytest.mark.parametrize("bad_id", [True, False, "1", 1.0, None, b"1", [1]])
def test_non_int_paragraph_ids_are_refused_not_stored(bad_id) -> None:
    with pytest.raises(r.ResolverInputError):
        r.paragraph_rejection(SCAN, bad_id)
    with pytest.raises(r.ResolverInputError):
        r.parse_findings(bad_id, {"findings": [item()]})
    assert issubclass(r.ResolverInputError, TypeError)


def test_skipped_paragraph_detail_names_the_skip_reason() -> None:
    assert "PARA_HAS_ERROR" in r.paragraph_rejection(SCAN, 0).detail


def test_s1_paragraph_ids_are_contiguous_zero_based_positions_on_every_fixture() -> None:
    fixtures = Path(__file__).resolve().parent / "fixtures" / "xml"
    scanned = 0
    for path in sorted(fixtures.iterdir()):
        try:
            scan = xs.scan(path.read_bytes())
        except Exception:
            continue
        assert [p.id for p in scan.paragraphs] == list(range(len(scan.paragraphs))), path.name
        for p in scan.paragraphs:
            assert r.find_paragraph(scan, p.id) is p
        scanned += 1
    assert scanned > 50


def test_lookup_uses_the_paragraph_id_field_when_ids_are_not_positions() -> None:
    # S1 never produces this, but a hand-built ScanResult may: ids 10, 11, 12 stored in reverse order
    shifted = tuple(dataclasses.replace(p, id=p.id + 10) for p in reversed(SCAN.paragraphs))
    scan = dataclasses.replace(SCAN, paragraphs=shifted)
    assert r.find_paragraph(scan, 11) is shifted[1] and shifted[1].id == 11
    assert r.paragraph_rejection(scan, 11) is None  # annotatable (was id 1)
    assert r.paragraph_rejection(scan, 10).code is RejectCode.PARA_NOT_ANNOTATABLE  # has <error>
    for positional in (0, 1, 2):  # positions are NOT ids here
        assert r.find_paragraph(scan, positional) is None
        assert r.paragraph_rejection(scan, positional).code is RejectCode.PARA_NOT_ANNOTATABLE


def test_paragraph_rejection_composes_with_parsing_as_s3_6_will_use_it() -> None:
    # S3.6 order: check the paragraph first; parse the payload only when the paragraph is resolvable
    payload = {"findings": [item(), item(type="x")]}
    blocked = r.paragraph_rejection(SCAN, 0)
    assert blocked is not None and blocked.finding_index is None  # whole paragraph, nothing parsed
    assert r.paragraph_rejection(SCAN, 1) is None
    checked, rejected, raw_count = r.parse_findings(1, payload)
    assert (len(checked), codes(rejected), raw_count) == (1, [(1, RejectCode.INVALID_FIELD)], 2)


# --------------------------------------------------------------------------------------------
# Steps 2-4: envelope and raw count
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [None, [], [item()], "findings", b"{}", 5, {}, {"results": [item()]}, {"findings": None},
     {"findings": {"0": item()}}, {"findings": "text"}, {"findings": (item(),)}],
)
def test_malformed_envelopes_are_schema_invalid_for_the_whole_paragraph(payload) -> None:
    checked, rejected, raw_count = parse(payload)
    assert checked == () and raw_count is None
    assert codes(rejected) == [(None, RejectCode.SCHEMA_INVALID)]


def test_empty_findings_list_is_valid() -> None:
    assert parse({"findings": []}) == ((), (), 0)


@pytest.mark.parametrize("count", [1, 29, 30])
def test_up_to_thirty_findings_are_processed(count: int) -> None:
    checked, rejected, raw_count = parse({"findings": [item() for _ in range(count)]})
    assert raw_count == count and len(checked) == count and rejected == ()


@pytest.mark.parametrize(
    "findings",
    [
        [item()] * 31,  # valid duplicates still count
        [{"bad": True}] * 31,  # invalid items count too (raw count, before validation)
        [item()] * 30 + ["not an object"],
        [item(type="nope")] * 40,
    ],
)
def test_more_than_thirty_raw_findings_rejects_the_whole_paragraph(findings) -> None:
    checked, rejected, raw_count = parse({"findings": findings})
    assert checked == () and raw_count == len(findings)
    assert codes(rejected) == [(None, RejectCode.TOO_MANY_FINDINGS)]
    assert str(len(findings)) in rejected[0].detail


def test_too_many_findings_is_reported_before_extra_keys() -> None:
    checked, rejected, _ = parse({"findings": [item()] * 31, "notes": "x"})
    assert codes(rejected) == [(None, RejectCode.TOO_MANY_FINDINGS)]


def test_too_many_findings_does_not_depend_on_order() -> None:
    findings = [item(original=f"w{i}") for i in range(20)] + [{"bad": i} for i in range(15)]
    for seed in range(5):
        shuffled = findings[:]
        random.Random(seed).shuffle(shuffled)
        assert codes(parse({"findings": shuffled})[1]) == [(None, RejectCode.TOO_MANY_FINDINGS)]


def test_extra_top_level_keys_are_schema_invalid() -> None:
    checked, rejected, raw_count = parse({"findings": [item()], "confidence": 0.9})
    assert checked == () and raw_count == 1
    assert codes(rejected) == [(None, RejectCode.SCHEMA_INVALID)]
    assert "confidence" not in rejected[0].detail  # unknown key names are model text: not echoed


# --------------------------------------------------------------------------------------------
# Step 5: per-item validation, isolated
# --------------------------------------------------------------------------------------------


def test_one_invalid_finding_does_not_block_the_others() -> None:
    findings = [item(original="a"), item(type="typo"), item(original="b"), "text", item(original="c", word_end=-1)]
    checked, rejected, raw_count = parse({"findings": findings})
    assert raw_count == 5
    assert [(c.index, c.original) for c in checked] == [(0, "a"), (2, "b")]
    assert codes(rejected) == [(1, RejectCode.INVALID_FIELD), (3, RejectCode.INVALID_FIELD), (4, RejectCode.INVALID_FIELD)]


@pytest.mark.parametrize(
    ("bad", "expected_detail"),
    [
        (item(type="typo"), "type: literal_error"),
        (item(word_start="3"), "word_start: int_type"),
        (item(word_start=True), "word_start: int_type"),
        (item(word_start=4, word_end=3), "item: value_error"),
        (item(original=""), "original: string_too_short"),
        (item(reason="  "), "reason: value_error"),
        (item(before="a b c d e"), "before: value_error"),
        (item(correction="x" + chr(1)), "correction: value_error"),
        (item(extra=1), "<unexpected field>: extra_forbidden"),
        ({k: v for k, v in item().items() if k != "reason"}, "reason: missing"),
        ("not an object", "item: model_type"),
        (None, "item: model_type"),
        ([item()], "item: model_type"),
    ],
)
def test_invalid_items_name_the_field_and_rule(bad, expected_detail: str) -> None:
    checked, rejected, _ = parse({"findings": [bad]})
    assert checked == ()
    assert codes(rejected) == [(0, RejectCode.INVALID_FIELD)]
    assert rejected[0].detail == expected_detail


def test_rejection_details_never_contain_document_or_model_text() -> None:
    findings = [
        item(original=SECRET * 20),  # too long
        item(before=f"{SECRET} {SECRET} {SECRET} {SECRET} {SECRET}"),  # too many words
        item(**{SECRET: 1}),  # unknown key
        item(original=" ".join([SECRET] * 7)),  # span too long
    ]
    _, rejected, _ = parse({"findings": findings})
    assert len(rejected) == 4
    assert all(SECRET not in x.detail for x in rejected)
    _, rejected, _ = parse({"findings": [item()], SECRET: 1})
    assert SECRET not in rejected[0].detail


# --------------------------------------------------------------------------------------------
# Steps 6-7: conversion and span cap
# --------------------------------------------------------------------------------------------


def test_conversion_to_checked_finding() -> None:
    (c,), rejected, _ = parse({"findings": [item(type="styleguide", reason="House style")]})
    assert rejected == ()
    assert c == CheckedFinding(0, "recieve", "Our team will", "the final report", 3, 3, "receive",
                               Category.STYLEGUIDE, "House style", False)


@pytest.mark.parametrize("category", [c.value for c in Category])
def test_every_category_keeps_its_reason(category: str) -> None:
    (c,), _, _ = parse({"findings": [item(type=category, reason=f"why {category}")]})
    assert c.category is Category(category) and c.reason == f"why {category}"


@pytest.mark.parametrize(
    ("original", "stripped", "flag"),
    [
        ("teh", "teh", False),
        (" teh ", "teh", True),
        ("\tteh\n", "teh", True),
        ("ice  cream", "ice  cream", False),  # inner whitespace is kept
        (" ice\ncream", "ice\ncream", True),
    ],
)
def test_original_edge_whitespace_is_stripped_and_flagged(original: str, stripped: str, flag: bool) -> None:
    (c,), _, _ = parse({"findings": [item(original=original)]})
    assert (c.original, c.original_stripped) == (stripped, flag)


def test_anchors_and_correction_are_not_stripped() -> None:
    (c,), _, _ = parse({"findings": [item(before=" will ", after=" the ", correction=" receive ")]})
    assert (c.before, c.after, c.correction) == (" will ", " the ", " receive ")


@pytest.mark.parametrize(
    ("category", "words", "code"),
    [
        ("spelling", 6, None),
        ("spelling", 7, RejectCode.SPAN_TOO_LONG),
        ("styleguide", 7, RejectCode.SPAN_TOO_LONG),
        ("grammar", 7, RejectCode.SPAN_TOO_LONG),
        ("clarity", 7, None),
        ("clarity", 20, None),
        ("clarity", 21, RejectCode.SPAN_TOO_LONG),
    ],
)
def test_span_caps(category: str, words: int, code) -> None:
    original = " ".join(f"w{i}" for i in range(words))
    checked, rejected, _ = parse({"findings": [item(type=category, original=original, reason="r")]})
    if code is None:
        assert len(checked) == 1 and rejected == ()
    else:
        assert checked == () and codes(rejected) == [(0, code)]
        assert f"{words} words" in rejected[0].detail


def test_span_cap_counts_words_after_stripping_and_on_any_whitespace() -> None:
    six = "a\tb\nc  d e f"
    checked, rejected, _ = parse({"findings": [item(original=f"  {six}  ")]})
    assert len(checked) == 1 and rejected == ()
    checked, rejected, _ = parse({"findings": [item(original=six + "\ng")]})
    assert codes(rejected) == [(0, RejectCode.SPAN_TOO_LONG)]


def test_duplicate_findings_are_kept_separately() -> None:
    checked, rejected, raw_count = parse({"findings": [item(), item()]})
    assert raw_count == 2 and [c.index for c in checked] == [0, 1] and rejected == ()


def test_payload_is_not_modified() -> None:
    payload = {"findings": [item(original=" teh "), item(type="x"), "bad"]}
    snapshot = copy.deepcopy(payload)
    parse(payload)
    assert payload == snapshot


def test_paragraph_id_is_carried_into_every_result() -> None:
    _, rejected, _ = r.parse_findings(7, {"findings": [item(type="x")]})
    assert rejected[0].paragraph_id == 7
    _, rejected, _ = r.parse_findings(7, None)
    assert rejected[0].paragraph_id == 7


# --------------------------------------------------------------------------------------------
# Property: each item's result depends only on that item (order and neighbours never matter)
# --------------------------------------------------------------------------------------------

VALUES = st.one_of(st.text(max_size=12), st.integers(-2, 40), st.booleans(), st.none())
GOOD_ITEMS = st.builds(
    lambda original, typ, ws, extra: item(original=original, type=typ, word_start=ws, word_end=ws + extra, reason="r"),
    st.text(alphabet="ab \t", min_size=1, max_size=30).filter(lambda s: s.strip()),
    st.sampled_from([c.value for c in Category]),
    st.integers(0, 5),
    st.integers(0, 3),
)
BAD_ITEMS = st.one_of(
    st.builds(lambda field, value: item(**{field: value}), st.sampled_from(sorted(item())), VALUES),
    VALUES,
)
ITEMS = st.lists(st.one_of(GOOD_ITEMS, BAD_ITEMS), max_size=30)


def _single(entry) -> tuple:
    checked, rejected, _ = parse({"findings": [entry]})
    if checked:
        c = checked[0]
        return ("ok", c.original, c.category, c.original_stripped)
    return ("rejected", rejected[0].code, rejected[0].detail)


def _at(result, index: int) -> tuple:
    checked, rejected, _ = result
    for c in checked:
        if c.index == index:
            return ("ok", c.original, c.category, c.original_stripped)
    (x,) = [x for x in rejected if x.finding_index == index]
    return ("rejected", x.code, x.detail)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ITEMS, st.randoms(use_true_random=False))
def test_property_each_result_depends_only_on_its_item(entries, rnd) -> None:
    result = parse({"findings": entries})
    assert result[2] == len(entries)
    indices = sorted([c.index for c in result[0]] + [x.finding_index for x in result[1]])
    assert indices == list(range(len(entries)))  # exactly one outcome per raw finding
    for i, entry in enumerate(entries):
        assert _at(result, i) == _single(entry)
    order = list(range(len(entries)))
    rnd.shuffle(order)
    shuffled = parse({"findings": [entries[i] for i in order]})
    for new_index, old_index in enumerate(order):
        assert _at(shuffled, new_index) == _at(result, old_index)


# ============================================================================================
# S3.5: candidate matching and anchor resolution
# ============================================================================================

ACUTE = chr(0x301)
E_NFC, E_NFD = chr(0xE9), "e" + ACUTE
RSQ = chr(0x2019)
FAMILY = chr(0x1F468) + chr(0x200D) + chr(0x1F469) + chr(0x200D) + chr(0x1F467)  # one ZWJ cluster
EXACT, GAP, NFC = MatchMode.EXACT, MatchMode.WHITESPACE_GAP, MatchMode.NFC


def para(text: str) -> Paragraph:
    return xs.scan(f"<p>{escape(text)}</p>".encode()).paragraphs[0]


def outer(markup: str) -> Paragraph:
    return xs.scan(markup.encode()).paragraphs[0]


def fnd(original: str, before: str = "", after: str = "", hint=(0, 0), correction: str = "FIX",
        category: Category = Category.SPELLING, index: int = 0) -> CheckedFinding:
    return CheckedFinding(index, original.strip(), before, after, hint[0], hint[1], correction, category, "why",
                          original.strip() != original)


def res(p: Paragraph, f: CheckedFinding):
    return r.resolve_finding(p, f)


def span_of(result) -> tuple[int, int]:
    assert isinstance(result, ResolvedFinding), result
    return result.annotation.start, result.annotation.end


def code_of(result) -> RejectCode:
    assert isinstance(result, FindingRejection), result
    return result.code


# ---- AC-24: repeated occurrences ----------------------------------------------------------------

REPEATS = para("teh cat and teh dog and teh cat")  # teh at 0, 12, 24


@pytest.mark.parametrize(
    ("before", "after", "span"),
    [("", "cat", (0, 3)), ("and", "dog", (12, 15)), ("and", "cat", (24, 27)), ("cat and", "dog and teh", (12, 15))],
)
def test_repeated_occurrences_are_separated_by_anchors(before: str, after: str, span) -> None:
    assert span_of(res(REPEATS, fnd("teh", before, after))) == span


@pytest.mark.parametrize(
    ("hint", "expected"),
    [((1, 1), (2, 5)), ((5, 5), (12, 15)), ((2, 2), RejectCode.AMBIGUOUS), ((0, 1), RejectCode.AMBIGUOUS),
     ((4, 4), RejectCode.AMBIGUOUS), ((6, 6), RejectCode.AMBIGUOUS)],
)
def test_word_index_breaks_exact_ties_only_without_plus_minus_one(hint, expected) -> None:
    p = para("a teh b c a teh b")  # words: 0:a 1:teh 2:b 3:c 4:a 5:teh 6:b
    result = res(p, fnd("teh", "a", "b", hint))
    if isinstance(expected, RejectCode):
        assert code_of(result) is expected
    else:
        assert span_of(result) == expected and result.hint_drift is False


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        ((1, 1), (2, 5)),  # planted first occurrence -> exactly that span
        ((3, 3), (8, 11)),  # planted second occurrence -> exactly that span, never the first
        ((2, 2), RejectCode.AMBIGUOUS),  # a hint matching neither occurrence never picks one
        ((0, 0), RejectCode.AMBIGUOUS),
        ((4, 4), RejectCode.AMBIGUOUS),
    ],
)
def test_repeated_occurrences_with_identical_true_anchors_follow_the_exact_hint(hint, expected) -> None:
    p = para("x teh x teh x")  # words 0:x 1:teh 2:x 3:teh 4:x; both "teh" have anchors "x" / "x"
    assert r._tier_candidates(r._context(p), "teh")[0] == ((2, 5), (8, 11))
    result = res(p, fnd("teh", "x", "x", hint))
    if isinstance(expected, RejectCode):
        assert code_of(result) is expected
    else:
        assert span_of(result) == expected and result.match_mode is EXACT and result.hint_drift is False


def test_repeated_occurrences_in_one_word_cannot_be_separated_by_the_hint() -> None:
    p = para("x-teh-y-teh-y z")  # both "teh" inside word 0: same word range, same anchors
    for hint in ((0, 0), (1, 1)):
        assert code_of(res(p, fnd("teh", "-", "-", hint))) is RejectCode.AMBIGUOUS


def test_hint_drift_is_recorded_but_never_overrides_a_unique_anchor_match() -> None:
    p = para("one teh two")
    good, drifted = res(p, fnd("teh", "one", "two", (1, 1))), res(p, fnd("teh", "one", "two", (7, 9)))
    assert span_of(good) == span_of(drifted) == (4, 7)
    assert (good.hint_drift, drifted.hint_drift) == (False, True)


def test_hint_never_rescues_wrong_anchors() -> None:
    assert code_of(res(REPEATS, fnd("teh", "dog", "and", (3, 3)))) is RejectCode.ANCHOR_MISMATCH


# ---- AC-24: punctuation, substrings, missing punctuation --------------------------------------


@pytest.mark.parametrize(
    ("text", "finding", "span"),
    [
        ("Stop,he said.", fnd(",", "Stop", "he said."), (4, 5)),
        ("It is the end.", fnd(".", "the end", ""), (13, 14)),
        ("It is the end.", fnd("end", "the", "."), (10, 13)),
        ("Wait...what", fnd("...", "Wait", "what"), (4, 7)),
        ("a " + chr(0x2014) + " b", fnd(chr(0x2014), "a", "b"), (2, 3)),
    ],
)
def test_punctuation_findings(text: str, finding: CheckedFinding, span) -> None:
    assert span_of(res(para(text), finding)) == span


@pytest.mark.parametrize(
    ("text", "finding", "expected"),
    [
        ("concatenate the cat", fnd("cat", "the", ""), (16, 19)),  # never inside "concatenate"
        ("there the", fnd("the", "there", ""), (6, 9)),
        ("concatenate", fnd("cat"), RejectCode.NOT_FOUND),
        ("theme", fnd("the", "", "me"), RejectCode.NOT_FOUND),
        ("cats", fnd("cat"), RejectCode.NOT_FOUND),
        ("e-mail", fnd("mail", "e-", ""), (2, 6)),  # '-' is its own fine token
    ],
)
def test_substrings_only_match_on_fine_token_boundaries(text: str, finding: CheckedFinding, expected) -> None:
    result = res(para(text), finding)
    if isinstance(expected, RejectCode):
        assert code_of(result) is expected
    else:
        assert span_of(result) == expected


def test_missing_punctuation_is_a_correction_of_a_non_empty_span() -> None:
    p = para("However the plan failed.")
    result = res(p, fnd("However", "", "the plan", correction="However,", category=Category.PUNCTUATION))
    assert span_of(result) == (0, 7) and result.annotation.correction == "However,"
    two = res(p, fnd("However the", "", "plan failed.", correction="However, the", category=Category.PUNCTUATION))
    assert span_of(two) == (0, 11)


# ---- AC-24: Unicode (NFC/NFD, combining marks, ZWJ emoji) -----------------------------------


@pytest.mark.parametrize(
    ("text", "original", "mode"),
    [
        (f"caf{E_NFD} au lait", f"caf{E_NFD}", EXACT),
        (f"caf{E_NFD} au lait", f"caf{E_NFC}", NFC),  # NFD text, NFC original
        (f"caf{E_NFC} au lait", f"caf{E_NFD}", NFC),  # NFC text, NFD original
        (f"caf{E_NFC} au lait", f"caf{E_NFC}", EXACT),
    ],
)
def test_nfc_and_nfd_forms_match(text: str, original: str, mode) -> None:
    result = res(para(text), fnd(original, "", "au lait"))
    assert span_of(result) == (0, len(text.split(" ")[0])) and result.match_mode is mode


def test_nfc_tier_with_nfd_anchor() -> None:
    p = para(f"caf{E_NFC} teh bar")
    assert span_of(res(p, fnd("teh", f"caf{E_NFD}", "bar"))) == (5, 8)


@pytest.mark.parametrize(
    ("text", "original"),
    [(f"caf{E_NFD}", "cafe"), (f"caf{E_NFD}", E_NFD), (f"caf{E_NFC}", "e"), (f"x {E_NFD} y", "e")],
)
def test_combining_marks_never_split(text: str, original: str) -> None:
    assert code_of(res(para(text), fnd(original))) is RejectCode.NOT_FOUND


def test_zwj_emoji_sequences_are_matched_whole_only() -> None:
    p = para(f"we {FAMILY} met")
    assert span_of(res(p, fnd(FAMILY, "we", "met"))) == (3, 3 + len(FAMILY))
    assert code_of(res(p, fnd(chr(0x1F468), "we", ""))) is RejectCode.NOT_FOUND


# ---- AC-24: whitespace gaps -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "original", "expected"),
    [
        ("ice\ncream cone", "ice cream", (GAP, (0, 9))),
        ("ice  cream cone", "ice cream", (GAP, (0, 10))),
        ("ice\tcream cone", "ice cream", (GAP, (0, 9))),
        ("ice cream cone", "ice\n\tcream", (GAP, (0, 9))),
        ("ice cream cone", "ice cream", (EXACT, (0, 9))),
        ("icecream cone", "ice cream", RejectCode.NOT_FOUND),  # no zero-width gaps
        ("ice cream cone", "icecream", RejectCode.NOT_FOUND),
    ],
)
def test_whitespace_gaps(text: str, original: str, expected) -> None:
    result = res(para(text), fnd(original, "", "cone"))
    if isinstance(expected, RejectCode):
        assert code_of(result) is expected
    else:
        assert (result.match_mode, span_of(result)) == expected


def test_whitespace_in_anchors_is_collapsed() -> None:
    p = para("Our  team\twill recieve it")
    assert span_of(res(p, fnd("recieve", "Our team  will", "it"))) == (15, 22)


# ---- AC-24: adjacency, overlaps, anchor mismatch ------------------------------------------------


def test_adjacent_findings_both_resolve() -> None:
    p = para("teh cta end.")
    results = r.resolve_findings(p, [fnd("teh", "", "cta", index=0), fnd("cta", "teh", "end.", index=1),
                                     fnd("end", "teh cta", ".", index=2), fnd(".", "end", "", index=3)])
    assert [span_of(x) for x in results] == [(0, 3), (4, 7), (8, 11), (11, 12)]


def test_overlapping_findings_both_resolve_and_are_left_to_s2() -> None:
    p = para("the teh cat sat")
    results = r.resolve_findings(p, [fnd("teh cat", "the", "sat", index=0), fnd("cat", "the teh", "sat", index=1)])
    assert [span_of(x) for x in results] == [(4, 11), (8, 11)]


@pytest.mark.parametrize(
    ("before", "after"),
    [("Our team", "the"), ("zzz", "the final report"), ("Our team will", "a final"), ("will", "report")],
)
def test_anchor_mismatch(before: str, after: str) -> None:
    p = para("Our team will recieve the final report")
    assert code_of(res(p, fnd("recieve", before, after))) is RejectCode.ANCHOR_MISMATCH


def test_anchor_must_start_on_a_fine_token_boundary() -> None:
    p = para("swill recieve it")  # "will" is a suffix of "swill", not a token
    assert code_of(res(p, fnd("recieve", "will", "it"))) is RejectCode.ANCHOR_MISMATCH
    assert code_of(res(p, fnd("recieve", "swill", "i"))) is RejectCode.ANCHOR_MISMATCH  # "i" is a prefix of "it"
    assert span_of(res(p, fnd("recieve", "swill", "it"))) == (6, 13)


def test_empty_anchors_only_at_the_paragraph_edge() -> None:
    p = para("teh middle teh")
    assert span_of(res(p, fnd("teh", "", "middle"))) == (0, 3)
    assert span_of(res(p, fnd("teh", "middle", ""))) == (11, 14)
    assert code_of(res(p, fnd("middle", "", ""))) is RejectCode.ANCHOR_MISMATCH


def test_quote_marks_in_anchors_are_exact() -> None:
    p = para(f"He said {chr(0x201C)}teh{chr(0x201D)} today")
    assert span_of(res(p, fnd("teh", f"said {chr(0x201C)}", chr(0x201D) + " today"))) == (9, 12)
    assert code_of(res(p, fnd("teh", 'said "', '" today'))) is RejectCode.ANCHOR_MISMATCH  # straight != curly


# ---- Decision A: cumulative tiers and fall-through (R5-1 ... R5-5) -----------------------------


def test_r5_1_exact_with_wrong_anchors_falls_through_to_a_whitespace_gap_match() -> None:
    p = para("ice cream bar and ice\ncream cone")
    tiers = r._tier_candidates(r._context(p), "ice cream")
    assert tiers == (((0, 9),), ((0, 9), (18, 27)), ((0, 9), (18, 27)))  # T1 occurrence kept in T2 and T3
    result = res(p, fnd("ice cream", "and", "cone"))
    assert span_of(result) == (18, 27) and result.match_mode is GAP


def test_r5_2_no_tier_with_valid_anchors_is_anchor_mismatch_not_not_found() -> None:
    p = para("ice cream bar and ice\ncream cone")
    assert code_of(res(p, fnd("ice cream", "zzz", "cone"))) is RejectCode.ANCHOR_MISMATCH


def test_r5_3_higher_tier_with_a_valid_candidate_decides() -> None:
    p = para("and ice cream cone and ice\ncream cone")
    result = res(p, fnd("ice cream", "and", "cone"))
    assert span_of(result) == (4, 13) and result.match_mode is EXACT  # not AMBIGUOUS across tiers


def test_r5_4_ties_within_the_deciding_tier() -> None:
    p = para("and ice\ncream cone and ice\tcream cone")  # words: 0 and 1 ice 2 cream 3 cone 4 and 5 ice 6 cream 7 cone
    assert code_of(res(p, fnd("ice cream", "and", "cone"))) is RejectCode.AMBIGUOUS
    chosen = res(p, fnd("ice cream", "and", "cone", (5, 6)))
    assert span_of(chosen) == (23, 32) and chosen.match_mode is GAP


def test_r5_5_nfc_tier_fall_through_and_ties() -> None:
    p = para(f"caf{E_NFC} bar and caf{E_NFD} cone and caf{E_NFD} cone")
    assert code_of(res(p, fnd(f"caf{E_NFD}", "and", "cone"))) is RejectCode.AMBIGUOUS  # both exact (T1)
    p2 = para(f"caf{E_NFD} bar and caf{E_NFC} cone")
    result = res(p2, fnd(f"caf{E_NFD}", "and", "cone"))  # exact at 0 fails anchors; NFC tier decides
    assert span_of(result) == (14, 18) and result.match_mode is NFC
    assert (0, 5) in r._tier_candidates(r._context(p2), f"caf{E_NFD}")[2]


def test_tiers_keep_exact_occurrences_the_normalised_view_cannot_see() -> None:
    # a barrier splits a whitespace run: the view holds two spaces, the collapsed original one
    p = outer("<r><p>teh <p>x</p> tial</p></r>")
    assert p.text == "teh  tial" and p.barriers == (4,)
    c1, c2, c3 = r._tier_candidates(r._context(p), "teh  tial")
    assert c1 == c2 == c3 == ((0, 9),)  # only the explicit union keeps T1 inside T2 and T3
    assert code_of(res(p, fnd("teh  tial"))) is RejectCode.CROSSES_PARAGRAPH


def test_several_hint_matches_inside_one_word_stay_ambiguous() -> None:
    p = para("x-teh-y-teh-y")  # one grid word; both "teh" have anchors "-" / "-"
    assert code_of(res(p, fnd("teh", "-", "-", (0, 0)))) is RejectCode.AMBIGUOUS


def test_empty_anchor_away_from_the_edge_fails_even_when_the_other_anchor_holds() -> None:
    p = para("a teh b")
    assert code_of(res(p, fnd("teh", "", "b"))) is RejectCode.ANCHOR_MISMATCH
    assert code_of(res(p, fnd("teh", "a", ""))) is RejectCode.ANCHOR_MISMATCH
    assert span_of(res(p, fnd("teh", "a", "b"))) == (2, 5)


def test_defensive_helpers_fail_closed_off_boundaries() -> None:
    ctx = r._context(para(f"a  caf{E_NFD}"))
    assert r._word_range(ctx, (1, 2)) is None  # starts in whitespace
    assert r._anchors_valid(ctx, (3, 7), "a", "") is False  # ends inside the "e + accent" cluster


def test_tiers_are_cumulative_on_a_mixed_paragraph() -> None:
    p = para(f"x ice cream y ice  cream z ic{E_NFD} cream w ice cream")
    c1, c2, c3 = r._tier_candidates(r._context(p), "ice cream")
    assert set(c1) <= set(c2) <= set(c3) and len(c1) == 2 and len(c2) == 3


# ---- E9a / E9b ---------------------------------------------------------------------------------


def test_e9a_partial_word_before_an_apostrophe_is_not_found() -> None:
    assert code_of(res(para(f"I don{RSQ}t know"), fnd("do", "I", ""))) is RejectCode.NOT_FOUND


def test_e9b_word_part_before_an_apostrophe_resolves() -> None:
    result = res(para(f"I cann{RSQ}t go"), fnd("cann", "I", f"{RSQ}t go", correction="can"))
    assert span_of(result) == (2, 6)


# ---- Barriers (decision C, U2) -----------------------------------------------------------------

BARRIER = outer("<r><p>Outer teh<p>inner</p> tial</p></r>")  # "Outer teh tial", barrier at 9


def test_word_grid_is_barrier_aware() -> None:
    p = outer("<r><p>foo<p>x</p>bar baz</p></r>")
    assert p.text == "foobar baz" and p.barriers == (3,)
    assert [(w.index, w.start, w.end, w.text) for w in r.word_grid(p)] == [(0, 0, 3, "foo"), (1, 3, 6, "bar"), (2, 7, 10, "baz")]
    assert [(w.start, w.end) for w in r.word_grid(para("a  b"))] == [(0, 1), (3, 4)]


@pytest.mark.parametrize(
    ("finding", "expected"),
    [
        (fnd("teh", "Outer", ""), (6, 9)),  # empty after-anchor valid at the barrier
        (fnd("tial", "", ""), (10, 14)),  # empty before-anchor valid at the barrier
        (fnd("teh", "Outer", "tial"), RejectCode.ANCHOR_MISMATCH),  # anchors never cross a barrier
        (fnd("tial", "teh", ""), RejectCode.ANCHOR_MISMATCH),
        (fnd("teh tial", "Outer", ""), RejectCode.CROSSES_PARAGRAPH),  # candidate crosses the barrier
    ],
)
def test_barriers_are_hard_edges(finding: CheckedFinding, expected) -> None:
    result = res(BARRIER, finding)
    if isinstance(expected, RejectCode):
        assert code_of(result) is expected
    else:
        assert span_of(result) == expected


def test_barrier_between_letters_splits_tokens() -> None:
    p = outer("<r><p>foo<p>x</p>bar</p></r>")  # "foobar", barrier at 3
    assert span_of(res(p, fnd("foo", "", ""))) == (0, 3)
    assert span_of(res(p, fnd("bar", "", ""))) == (3, 6)


# ---- Correction checks (decision D) and result contract ---------------------------------------


@pytest.mark.parametrize(
    ("text", "original", "correction"),
    [("a teh b", "teh", "teh"), (f"a caf{E_NFD} b", f"caf{E_NFD}", f"caf{E_NFC}"), (f"a caf{E_NFC} b", f"caf{E_NFD}", f"caf{E_NFD}")],
)
def test_canonically_equivalent_correction_is_a_no_op(text: str, original: str, correction: str) -> None:
    result = res(para(text), fnd(original, "a", "b", correction=correction))
    assert code_of(result) is RejectCode.INVALID_FIELD and "no-op" in result.detail


def test_resolved_annotation_carries_category_correction_reason_and_index() -> None:
    p = para("our teh cat")
    result = res(p, fnd("teh", "our", "cat", (1, 1), correction="the", category=Category.GRAMMAR, index=7))
    assert result.finding_index == 7
    assert result.annotation == Annotation(p.id, 4, 7, Category.GRAMMAR, "the", "why")


def test_results_follow_input_order_and_keep_raw_indices() -> None:
    p = para("teh cat teh")
    findings = [fnd("zzz", index=9), fnd("teh", "", "cat", index=2), fnd("cat", "teh", "teh", index=5)]
    results = r.resolve_findings(p, findings)
    assert [x.finding_index for x in results] == [9, 2, 5]
    assert code_of(results[0]) is RejectCode.NOT_FOUND


def test_rejection_details_contain_no_text() -> None:
    p = para(f"{SECRET} teh {SECRET}")
    results = r.resolve_findings(p, [fnd("teh", "zzz", ""), fnd("nothing"), fnd("teh", SECRET, SECRET, correction="teh")])
    assert all(SECRET not in x.detail for x in results)


# ---- Adversarial ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "finding", "expected"),
    [
        ("a" * 2000, fnd("a" * 3), RejectCode.NOT_FOUND),  # inside one long token
        (" ".join(["teh"] * 3000), fnd("teh", "teh", "teh"), RejectCode.AMBIGUOUS),
        (" ".join(["teh"] * 3000), fnd("teh", "", "teh"), (0, 3)),
        ("x " * 3000 + "end", fnd("end", "x", ""), (6000, 6003)),
        ((ACUTE * 50) + " teh", fnd("teh", ACUTE * 50, ""), (51, 54)),  # a mark-only cluster is a valid anchor
        ((ACUTE * 50) + " teh", fnd("teh", ACUTE * 49, ""), RejectCode.ANCHOR_MISMATCH),  # never part of a cluster
    ],
)
def test_adversarial_inputs(text: str, finding: CheckedFinding, expected) -> None:
    result = res(para(text), finding)
    if isinstance(expected, RejectCode):
        assert code_of(result) is expected
    else:
        assert span_of(result) == expected


# ---- Properties (checked against an independent oracle written with plain string operations) ----


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", " ".join(s.split()))


def _segment(p: Paragraph, s: int, e: int) -> tuple[int, int]:
    bounds = sorted({0, len(p.text), *p.barriers})
    return max(b for b in bounds if b <= s), min(b for b in bounds if b >= e)


def _fine(p: Paragraph) -> set[int]:
    bounds = sorted({0, len(p.text), *p.barriers})
    out: set[int] = set()
    for a, b in zip(bounds, bounds[1:]):
        out |= {a + x for x in tokens.fine_boundaries(p.text[a:b])}
    return out


def _oracle_anchor(region: str, anchor: str, at_end: bool) -> bool:
    region_n, anchor_n = _norm(region), _norm(anchor)
    if not anchor_n:
        return region_n == ""
    if at_end:
        if not region_n.endswith(anchor_n):
            return False
        k = len(region_n) - len(anchor_n)
        return k == 0 or k in tokens.fine_boundaries(region_n)
    if not region_n.startswith(anchor_n):
        return False
    k = len(anchor_n)
    return k == len(region_n) or k in tokens.fine_boundaries(region_n)


def _oracle_words(p: Paragraph) -> list[tuple[int, int]]:
    """Barrier-aware word grid, independently: whitespace-split words of each barrier segment."""
    bounds = sorted({0, len(p.text), *p.barriers})
    out = []
    for a, b in zip(bounds, bounds[1:]):
        i = a
        while i < b:
            if p.text[i].isspace():
                i += 1
                continue
            j = i
            while j < b and not p.text[j].isspace():
                j += 1
            out.append((i, j))
            i = j
    return out


def _oracle_tokens(p: Paragraph) -> list[tuple[int, int]]:
    """Fine tokens of each barrier segment (tokens.fine_tokens, independent of the resolver)."""
    bounds = sorted({0, len(p.text), *p.barriers})
    return [(a + t.start, a + t.end) for a, b in zip(bounds, bounds[1:]) for t in tokens.fine_tokens(p.text[a:b])]


def _oracle_word_range(words: list[tuple[int, int]], s: int, e: int) -> tuple[int, int]:
    first = next(k for k, (ws, we) in enumerate(words) if ws <= s < we)
    last = next(k for k, (ws, we) in enumerate(words) if ws <= e - 1 < we)
    return first, last


def _oracle_exact_candidates(p: Paragraph, original: str) -> list[tuple[int, int]]:
    fine, out, i = _fine(p), [], p.text.find(original)
    while i != -1:
        if i in fine and i + len(original) in fine:
            out.append((i, i + len(original)))
        i = p.text.find(original, i + 1)
    return out


def _oracle_valid(p: Paragraph, span: tuple[int, int], f: CheckedFinding) -> bool:
    s, e = span
    a, b = _segment(p, s, e)
    return (s in _fine(p) and e in _fine(p) and _norm(p.text[s:e]) == _norm(f.original)
            and _oracle_anchor(p.text[a:s], f.before, True) and _oracle_anchor(p.text[e:b], f.after, False))


WORDS = ["teh", "cat", "ice", "cream", "caf" + E_NFC, "caf" + E_NFD, "a", "the", "end.", "x,", RSQ + "t"]
SEPARATORS = [" ", "  ", "\n", "\t"]
segments_st = st.lists(
    st.lists(st.tuples(st.sampled_from(WORDS), st.sampled_from(SEPARATORS)), min_size=1, max_size=8).map(
        lambda pairs: "".join(w + s for w, s in pairs).strip()),
    min_size=1, max_size=3)


def _paragraph_of(segments: list[str]) -> Paragraph:
    markup = "<r><p>" + "<p>n</p>".join(escape(s) for s in segments) + "</p></r>"
    return xs.scan(markup.encode()).paragraphs[0]


PROPERTY_SETTINGS = settings(max_examples=250, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _check_planted_span(p: Paragraph, data) -> str:
    """Plant an aligned span with its true anchors and exact word hint; check the result against
    the independent oracle. Returns which branch was taken (for coverage checks)."""
    ctx = r._context(p)
    fine = sorted(ctx.fine)
    s = data.draw(st.sampled_from([x for x in fine if x < len(p.text) and not p.text[x].isspace()]))
    ends = [x for x in fine if x > s and not p.text[x - 1].isspace()]
    e = data.draw(st.sampled_from(ends))
    if any(s < bar < e for bar in p.barriers):
        return "skipped"  # crossing spans are covered by test_barriers_are_hard_edges
    a, b = _segment(p, s, e)
    words = _oracle_words(p)
    if data.draw(st.sampled_from(["words", "token"])) == "words":
        # true anchors: from the start of up to two words before s, to the end of up to two words after e
        previous = [w for w in words if a <= w[0] < s]
        following = [w for w in words if e < w[1] <= b]
    else:
        # true minimal anchors: just the adjacent fine token on each side (within the segment)
        previous = [t for t in _oracle_tokens(p) if a <= t[0] and t[1] <= s][-1:]
        following = [t for t in _oracle_tokens(p) if e <= t[0] and t[1] <= b][:1]
    before = p.text[previous[-2:][0][0] if previous else s:s]
    after = p.text[e:following[:2][-1][1] if following else e]
    hint = _oracle_word_range(words, s, e)
    f = fnd(p.text[s:e], before.strip(), after.strip(), hint)
    if len(f.original.split()) > 6 or len(f.before.split()) > 4 or len(f.after.split()) > 4:
        return "skipped"
    assert _oracle_valid(p, (s, e), f)  # the planted finding is genuinely valid

    # Independent expectation (decision A + D9): the planted span is an anchor-valid EXACT occurrence,
    # so T1 decides. Its hint is its own word range, so another span can only matter if it is also an
    # anchor-valid exact occurrence in the SAME words -> AMBIGUOUS. Nothing can select a different span.
    rivals = [c for c in _oracle_exact_candidates(p, f.original)
              if c != (s, e) and _oracle_valid(p, c, f) and _oracle_word_range(words, *c) == hint]
    result = res(p, f)
    if rivals:
        assert isinstance(result, FindingRejection) and result.code is RejectCode.AMBIGUOUS, result
        return "ambiguous"
    else:
        assert isinstance(result, ResolvedFinding), result
        assert span_of(result) == (s, e) and result.match_mode is EXACT and not result.hint_drift
    return "tie-broken" if len([c for c in _oracle_exact_candidates(p, f.original) if _oracle_valid(p, c, f)]) > 1 else "unique"


@PROPERTY_SETTINGS
@given(segments_st, st.data())
def test_property_planted_span_with_true_anchors_resolves_to_itself(segments, data) -> None:
    _check_planted_span(_paragraph_of(segments), data)


REPEAT_SEPARATORS = [" ", "-", " ", "\n"]
repetitive_segments_st = st.tuples(
    st.lists(st.tuples(st.sampled_from(["teh", "cat", "x", "a"]), st.sampled_from(REPEAT_SEPARATORS)), min_size=1, max_size=4),
    st.integers(2, 3),
).map(lambda t: ["".join(w + sep for w, sep in t[0]) * t[1]])


@PROPERTY_SETTINGS
@given(repetitive_segments_st, st.data())
def test_property_planted_span_in_repetitive_text(segments, data) -> None:
    # repeated blocks: identical anchors in different words (exact hint must pick the planted span)
    # and, with '-' joins, rivals inside the same word (must be AMBIGUOUS)
    p = _paragraph_of([segments[0].strip(" -\n") or "x"])
    _check_planted_span(p, data)


@PROPERTY_SETTINGS
@given(segments_st, st.sampled_from(WORDS + ["cat cream", "ice cream", "teh,", "zzz"]), st.sampled_from(WORDS + ["", "x"]),
       st.sampled_from(WORDS + ["", "a the"]), st.integers(0, 6), st.integers(0, 2))
def test_property_every_resolution_is_sound(segments, original, before, after, ws, extra) -> None:
    p = _paragraph_of(segments)
    f = fnd(original, before, after, (ws, ws + extra))
    result = res(p, f)
    if isinstance(result, ResolvedFinding):
        span = span_of(result)
        assert _oracle_valid(p, span, f)  # aligned, equal after normalisation, anchors hold
        assert not any(span[0] < bar < span[1] for bar in p.barriers)
        assert _norm(f.correction) != _norm(p.text[span[0]:span[1]])
    elif result.code is RejectCode.NOT_FOUND:
        assert not any(_norm(p.text[s:e]) == _norm(f.original) for s in _fine(p) for e in _fine(p) if s < e)


@PROPERTY_SETTINGS
@given(segments_st, st.sampled_from(WORDS + ["ice cream", "cat cream"]))
def test_property_tiers_are_cumulative(segments, original) -> None:
    c1, c2, c3 = r._tier_candidates(r._context(_paragraph_of(segments)), original)
    assert set(c1) <= set(c2) <= set(c3)


@PROPERTY_SETTINGS
@given(segments_st, st.lists(st.tuples(st.sampled_from(WORDS), st.sampled_from(WORDS + [""])), min_size=1, max_size=6),
       st.randoms(use_true_random=False))
def test_property_results_do_not_depend_on_other_findings_or_order(segments, specs, rnd) -> None:
    p = _paragraph_of(segments)
    findings = [fnd(o, b, "", index=i) for i, (o, b) in enumerate(specs)]
    together = {x.finding_index: x for x in r.resolve_findings(p, findings)}
    shuffled = findings[:]
    rnd.shuffle(shuffled)
    assert {x.finding_index: x for x in r.resolve_findings(p, shuffled)} == together
    for f in findings:
        assert res(p, f) == together[f.index]

"""S3.3: nested-paragraph barriers and conditional quotation propagation (decision U3).

A nested paragraph inherits whole-paragraph protection only when a protected range of its parent
actually spans its barrier, or when its parent is itself entirely protected (transitive).
All documents are invented.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from proofreader import quotes as q
from proofreader import xml_source as xs
from proofreader.models import RangeOrigin, RejectCode

LDQ, RDQ, LAQ, SDQ, LSQ, RSQ = chr(0x201C), chr(0x201D), chr(0xAB), chr(0x22), chr(0x2018), chr(0x2019)
XML = Path(__file__).resolve().parent / "fixtures" / "xml"


def scan_of(markup: str):
    return xs.scan(markup.encode("utf-8"))


def inherited(markup: str) -> dict[int, int | None]:
    """paragraph id -> id it inherited protection from (None if it did not)."""
    return {pid: a.inherited_from for pid, a in q.analyse_document(scan_of(markup)).items()}


def assert_entirely_protected(analysis, text: str) -> None:
    n = len(text)
    expected = [(0, n, True, True, RangeOrigin.INHERITED)] if n else []  # reaches both edges of the child
    assert [(r.start, r.end, r.open_start, r.open_end, r.origin) for r in analysis.ranges] == expected
    if n:
        assert q.span_verdict(analysis, 0, n) in (RejectCode.IN_QUOTATION, RejectCode.TOUCHES_QUOTATION)


# --------------------------------------------------------------------------------------------
# nested_barriers: derived from S1 data, equal to paragraph.barriers
# --------------------------------------------------------------------------------------------


def test_nested_barriers_lists_direct_children_with_offsets() -> None:
    scan = scan_of("<r><p>ab <p>one</p>cd<p>two</p><p>three</p></p></r>")
    outer = scan.paragraphs[0]
    assert outer.text == "ab cd" and outer.barriers == (3, 5)
    assert q.nested_barriers(scan, 0) == ((1, 3), (2, 5), (3, 5))  # two children share offset 5
    assert q.nested_barriers(scan, 1) == ()


def test_nested_barriers_only_direct_children_through_inline_elements() -> None:
    scan = scan_of("<r><p>a <em>x <p>mid <p>deep</p> m</p> y</em> b</p></r>")
    assert [p.text for p in scan.paragraphs] == ["a x  y b", "mid  m", "deep"]
    assert q.nested_barriers(scan, 0) == ((1, 4),)  # the grandchild is not a direct child
    assert q.nested_barriers(scan, 1) == ((2, 4),)


def test_nested_barriers_at_paragraph_edges_and_with_namespaces() -> None:
    scan = scan_of('<r xmlns:x="urn:x"><x:p><x:p>first</x:p>mid<p>last</p></x:p></r>')
    assert scan.paragraphs[0].text == "mid" and scan.paragraphs[0].barriers == (0, 3)
    assert q.nested_barriers(scan, 0) == ((1, 0), (2, 3))


def _committed_scans():
    for path in sorted(XML.iterdir()):
        try:
            yield path.name, xs.scan(path.read_bytes())
        except Exception:
            continue  # inputs S1 rejects never reach S3


def test_derived_barriers_equal_s1_barriers_on_every_fixture() -> None:
    checked = with_children = 0
    for name, scan in _committed_scans():
        for paragraph in scan.paragraphs:
            derived = q.nested_barriers(scan, paragraph.id)
            assert tuple(sorted({offset for _, offset in derived})) == paragraph.barriers, (name, paragraph.id)
            checked += 1
            with_children += bool(derived)
    assert checked > 50 and with_children >= 3  # the fixtures do exercise nesting


# --------------------------------------------------------------------------------------------
# U3 propagation cases
# --------------------------------------------------------------------------------------------


def test_child_inside_a_quotation_inherits() -> None:
    markup = f"<r><p>He said {LDQ}a <p>inner teh</p> b{RDQ} ok.</p></r>"
    result = q.analyse_document(scan_of(markup))
    assert result[0].inherited_from is None and result[1].inherited_from == 0
    assert_entirely_protected(result[1], "inner teh")


def test_child_outside_a_quotation_keeps_its_own_analysis() -> None:
    markup = f"<r><p>He said {LDQ}a b{RDQ} then <p>inner {LDQ}x{RDQ} teh</p> done.</p></r>"
    scan = scan_of(markup)
    result = q.analyse_document(scan)
    assert result[1].inherited_from is None
    assert result[1] == q.analyse_paragraph(scan.paragraphs[1])  # its own quotation only


@pytest.mark.parametrize(
    "markup",
    [
        f"<r><p>x <p>c</p>{LDQ}a{RDQ} y</p></r>",  # barrier exactly at the range start
        f"<r><p>x {LDQ}a{RDQ}<p>c</p> y</p></r>",  # barrier exactly at the range end
    ],
)
def test_child_at_a_closed_range_edge_does_not_inherit(markup: str) -> None:
    assert inherited(markup) == {0: None, 1: None}


def test_unclosed_quotation_reaching_a_child_at_the_paragraph_end() -> None:
    markup = f"<r><p>He said {LDQ}a b <p>inner</p></p></r>"
    scan = scan_of(markup)
    outer = scan.paragraphs[0]
    assert outer.barriers == (len(outer.text),)  # the child sits at the very end
    assert inherited(markup) == {0: None, 1: 0}


def test_unclosed_quotation_reaching_a_child_at_the_end_via_an_ambiguous_mark() -> None:
    assert inherited(f"<r><p>a {SDQ} b <p>inner</p></p></r>") == {0: None, 1: 0}


def test_unmatched_closing_quote_with_a_child_at_offset_zero() -> None:
    markup = f"<r><p><p>inner</p>a b{RDQ} c</p></r>"
    scan = scan_of(markup)
    assert scan.paragraphs[0].barriers == (0,)
    assert inherited(markup) == {0: None, 1: 0}


def test_paired_quotation_starting_at_offset_zero_does_not_reach_a_child_before_it() -> None:
    # range [0, 3) is a pair, not open_start: a child at barrier 0 is before the opening mark
    assert inherited(f"<r><p><p>inner</p>{LDQ}a{RDQ} c</p></r>") == {0: None, 1: None}


def test_guard_range_propagates_to_every_child_including_edges() -> None:
    marks = " ".join(f"{LDQ}w{RDQ}" for _ in range(33))  # 66 marks: guard
    markup = f"<r><p><p>first</p>{marks}<p>middle</p> tail<p>last</p></p></r>"
    result = q.analyse_document(scan_of(markup))
    assert result[0].unsafe_reason is not None
    assert [result[i].inherited_from for i in (1, 2, 3)] == [0, 0, 0]


def test_propagation_is_transitive_to_grandchildren() -> None:
    markup = f"<r><p>{LDQ}a <p>mid <p>deep</p> m</p> b{RDQ}</p></r>"
    result = q.analyse_document(scan_of(markup))
    assert [result[i].inherited_from for i in (0, 1, 2)] == [None, 0, 1]
    assert_entirely_protected(result[2], "deep")


def test_grandchild_at_the_edge_of_an_entirely_protected_child_inherits() -> None:
    markup = f"<r><p>{LDQ}a <p><p>deep</p>mid</p> b{RDQ}</p></r>"  # deep sits at offset 0 of mid
    assert inherited(markup) == {0: None, 1: 0, 2: 1}


def test_grandchild_inherits_from_its_own_parent_quotation_only() -> None:
    markup = f"<r><p>a <p>mid {LAQ}x <p>deep</p> y</p> b</p></r>"  # mid's own unclosed guillemet
    assert inherited(markup) == {0: None, 1: None, 2: 1}


def test_sibling_outside_the_quotation_is_unaffected() -> None:
    markup = f"<r><p>{LDQ}a <p>one</p> b{RDQ} c <p>two</p> d</p></r>"
    assert inherited(markup) == {0: None, 1: 0, 2: None}


# ---- touching quotations: separate quotations that meet at a barrier never protect the child ----

LAQ_, RAQ_ = chr(0xAB), chr(0xBB)


@pytest.mark.parametrize(
    ("markup", "expected"),
    [
        (f"<r><p>{LDQ}a{RDQ}<p>child teh</p>{LDQ}b{RDQ} end</p></r>", {0: None, 1: None}),  # touching pairs
        (f"<r><p>{LAQ_}a{RAQ_}<p>child</p>{LDQ}b{RDQ}</p></r>", {0: None, 1: None}),  # touching, mixed families
        (f"<r><p>a b{RDQ}<p>child</p>{LDQ}x y</p></r>", {0: None, 1: None}),  # open_start range meets unclosed
        (f"<r><p>{LDQ}a <p>child teh</p> b{RDQ} end</p></r>", {0: None, 1: 0}),  # ONE quotation spans it
        (f"<r><p>{LAQ_}z {LDQ}a{RDQ}<p>child</p>{LDQ}b{RDQ} e</p></r>", {0: None, 1: 0}),  # unclosed outer spans it
    ],
    ids=["touching-pairs", "touching-mixed", "touching-open-edges", "single-spanning", "touching-inside-another"],
)
def test_touching_quotations_do_not_protect_the_child_but_a_spanning_one_does(markup: str, expected) -> None:
    assert inherited(markup) == expected


def test_touching_quotations_keep_the_merged_parent_ranges_unchanged() -> None:
    scan = scan_of(f"<r><p>{LDQ}a{RDQ}<p>child</p>{LDQ}b{RDQ} end</p></r>")
    outer = q.analyse_document(scan)[0]
    assert [(r.start, r.end) for r in outer.ranges] == [(0, 6)]  # final merging behaviour is unchanged
    assert outer == q.analyse_paragraph(scan.paragraphs[0])


@pytest.mark.parametrize(
    ("markup", "expected"),
    [
        # touching quotations around a child: neither the child nor its own child inherits
        (f"<r><p>{LDQ}a{RDQ}<p>mid <p>deep</p> m</p>{LDQ}b{RDQ}</p></r>", {0: None, 1: None, 2: None}),
        # ... but the child's own quotation still protects the grandchild
        (f"<r><p>{LDQ}a{RDQ}<p>mid {LAQ_}x <p>deep</p> y</p>{LDQ}b{RDQ}</p></r>", {0: None, 1: None, 2: 1}),
        # one spanning quotation: transitive to the grandchild
        (f"<r><p>{LDQ}a <p>mid <p>deep</p> m</p> b{RDQ}</p></r>", {0: None, 1: 0, 2: 1}),
    ],
    ids=["touching-transitive-none", "touching-child-own-quote", "spanning-transitive"],
)
def test_touching_versus_spanning_through_nested_levels(markup: str, expected) -> None:
    assert inherited(markup) == expected


def test_finding_in_a_child_between_touching_quotations_is_applied() -> None:
    from proofreader import annotation as an
    from proofreader import resolver as r

    markup = f"<r><p>{LDQ}a{RDQ}<p>child teh</p>{LDQ}b{RDQ} end</p></r>"
    data = markup.encode()
    scan = xs.scan(data)
    payload = {"findings": [{"original": "teh", "before": "child", "after": "", "word_start": 1, "word_end": 1,
                             "correction": "the", "type": "spelling", "reason": "why"}]}
    results = r.resolve_document(scan, {1: payload})
    outcome = an.annotate(data, scan, r.annotations(results))
    assert outcome.ok and outcome.output == data.replace(
        b"child teh", b'child <error type="spelling" correction="the" reason="why">teh</error>')


def test_children_sharing_a_barrier_inside_a_quotation_all_inherit() -> None:
    markup = f"<r><p>{LDQ}a <p>x</p><p>y</p> b{RDQ}</p></r>"
    assert inherited(markup) == {0: None, 1: 0, 2: 0}


def test_separate_top_level_paragraphs_never_inherit_from_each_other() -> None:
    markup = f"<r><p>{LDQ}open never closed</p><p>next paragraph</p></r>"
    assert inherited(markup) == {0: None, 1: None}


def test_empty_inherited_child_has_no_ranges() -> None:
    result = q.analyse_document(scan_of(f"<r><p>{LDQ}a <p></p> b{RDQ}</p></r>"))
    assert result[1].inherited_from == 0 and result[1].ranges == ()


def test_inherited_child_keeps_its_own_marks_and_guard_reason() -> None:
    inner = " ".join(f"{LDQ}w{RDQ}" for _ in range(33))
    result = q.analyse_document(scan_of(f"<r><p>{LDQ}a <p>{inner}</p> b{RDQ}</p></r>"))
    assert result[1].inherited_from == 0 and result[1].unsafe_reason is not None
    assert len(result[1].marks) == 66


def test_rule_s_and_apostrophes_in_the_parent() -> None:
    # rule S: the opener protects to the LAST close-or-apostrophe, which lies after the child
    assert inherited(f"<r><p>{LSQ}dogs{RSQ} and <p>child</p> cats{RSQ} x</p></r>") == {0: None, 1: 0}
    # a plain apostrophe protects nothing
    assert inherited(f"<r><p>the boys{RSQ} <p>child</p> bikes</p></r>") == {0: None, 1: None}


def test_covers_barrier_edge_rule_applies_only_at_true_paragraph_edges() -> None:
    from proofreader.models import ProtectedRange

    n = 20
    pair = ProtectedRange(5, 10, False, False, RangeOrigin.PAIR)
    assert [q._covers_barrier(pair, b, n) for b in (4, 5, 6, 9, 10, 11)] == [False, False, True, True, False, False]
    start_edge = ProtectedRange(0, 10, True, False, RangeOrigin.EMPTY_STACK_CLOSE)
    end_edge = ProtectedRange(5, n, False, True, RangeOrigin.UNCLOSED)
    assert q._covers_barrier(start_edge, 0, n) and q._covers_barrier(end_edge, n, n)
    # defensive: the edge rule never applies away from the real paragraph edges, even for a
    # (never produced) range whose open flags disagree with its position
    assert not q._covers_barrier(ProtectedRange(5, 10, True, False, RangeOrigin.EMPTY_STACK_CLOSE), 5, n)
    assert not q._covers_barrier(ProtectedRange(5, 10, False, True, RangeOrigin.UNCLOSED), 10, n)


def test_documents_without_nesting_equal_paragraph_analysis() -> None:
    for name, scan in _committed_scans():
        result = q.analyse_document(scan)
        assert sorted(result) == [p.id for p in scan.paragraphs], name
        for p in scan.paragraphs:
            if result[p.id].inherited_from is None:
                assert result[p.id] == q.analyse_paragraph(p), (name, p.id)


# --------------------------------------------------------------------------------------------
# Property: random nested documents
# --------------------------------------------------------------------------------------------

TEXT_ALPHABET = list("ab Z.,-") + [LDQ, RDQ, LAQ, SDQ, LSQ, RSQ]
texts = st.text(alphabet=st.sampled_from(TEXT_ALPHABET), max_size=8)


def nodes(depth: int):
    leaf = texts.map(escape)
    if depth == 0:
        return st.lists(leaf, max_size=3).map("".join)
    child = st.one_of(
        leaf,
        nodes(depth - 1).map(lambda inner: f"<p>{inner}</p>"),
        nodes(depth - 1).map(lambda inner: f"<em>{inner}</em>"),
    )
    return st.lists(child, max_size=4).map("".join)


documents = st.lists(nodes(3).map(lambda body: f"<p>{body}</p>"), min_size=1, max_size=2).map(
    lambda ps: "<r>" + "".join(ps) + "</r>")
SETTINGS = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _space_oracle_spans(parent_text: str, barriers: tuple[int, ...], b: int) -> bool:
    """Independent oracle (no U3 edge rule): replace the barrier at ``b`` by a space - classification
    treats a barrier exactly like whitespace, so every mark keeps its role - and ask the plain
    analysis whether that position is inside a protected range."""
    text = parent_text[:b] + " " + parent_text[b:]
    others = tuple(x if x < b else x + 1 for x in barriers if x != b)
    return any(r.start <= b < r.end for r in q.analyse_text(text, others).ranges)


@SETTINGS
@given(documents)
def test_property_barriers_and_propagation(markup: str) -> None:
    scan = scan_of(markup)
    result = q.analyse_document(scan)
    parent_of = {}
    for p in scan.paragraphs:
        derived = q.nested_barriers(scan, p.id)
        assert tuple(sorted({o for _, o in derived})) == p.barriers
        for child, offset in derived:
            parent_of[child] = (p.id, offset)
    for p in scan.paragraphs:
        a = result[p.id]
        if p.id not in parent_of:
            assert a.inherited_from is None and a == q.analyse_paragraph(p)
            continue
        parent, offset = parent_of[p.id]
        parent_p = scan.paragraphs[parent]
        # exact expectation from the independent oracle (no restatement of the U3 edge rule)
        expected = result[parent].inherited_from is not None or _space_oracle_spans(
            parent_p.text, parent_p.barriers, offset)
        assert (a.inherited_from == parent) is expected
        if expected:
            assert_entirely_protected(a, p.text)
        else:
            assert a.inherited_from is None and a == q.analyse_paragraph(p)

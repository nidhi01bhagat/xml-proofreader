"""S2.1: annotation data models and field validation (synthetic data only, no LLM calls)."""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import pytest

from proofreader import annotation as ann
from proofreader.models import Annotation, Category, Insertion, RejectCode, Rejection, RenderResult

TEXT = "Our team will recieve the final report."  # "recieve" = [14, 21)
G, SP, ST = Category.GRAMMAR, Category.SPELLING, Category.STYLEGUIDE


def a(start=14, end=21, category=SP, correction="receive", reason=None, paragraph_id=0) -> Annotation:
    return Annotation(paragraph_id, start, end, category, correction, reason)


def code(annotation: Annotation, text: str = TEXT) -> RejectCode | None:
    rejection = ann.validate_annotation(annotation, text)
    if rejection is not None:
        assert rejection.annotation is annotation and rejection.detail  # always explains itself
    return None if rejection is None else rejection.code


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------


def test_category_values_are_the_six_lowercase_types() -> None:
    assert [c.value for c in Category] == [
        "grammar", "spelling", "punctuation", "capitalization", "clarity", "styleguide"
    ]


S2_REJECT_CODES = [
    "INVALID_FIELD", "EMPTY_SPAN", "OUT_OF_RANGE", "PARA_NOT_ANNOTATABLE", "SPLITS_ATOMIC",
    "CROSSES_ELEMENT", "CROSSES_PARAGRAPH", "IN_CDATA", "OVERLAP", "CONFLICT", "VERIFY_FAILED",
]


def test_s2_reject_codes_unchanged() -> None:
    # S3 appends its codes after these (exact full list: tests/test_models_contract.py).
    members = list(RejectCode)[: len(S2_REJECT_CODES)]
    assert [c.name for c in members] == S2_REJECT_CODES
    assert [c.value for c in members] == S2_REJECT_CODES


def test_annotation_module_uses_only_s2_reject_codes() -> None:
    tree = ast.parse(Path(inspect.getsourcefile(ann)).read_text(encoding="utf-8"))
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "RejectCode":
            used.add(node.attr)
        # no dynamic lookup (RejectCode(...) / RejectCode[...]) that could yield a non-S2 code
        if isinstance(node, (ast.Call, ast.Subscript)):
            target = node.func if isinstance(node, ast.Call) else node.value
            assert not (isinstance(target, ast.Name) and target.id == "RejectCode")
    assert used and used <= set(S2_REJECT_CODES), sorted(used - set(S2_REJECT_CODES))


@pytest.mark.parametrize(
    ("obj", "field"),
    [
        (a(), "start"),
        (Rejection(a(), RejectCode.EMPTY_SPAN, "x"), "code"),
        (Insertion(3, 1, b"<error>", 0), "offset"),
        (RenderResult(b"<p/>", (), (), ()), "output"),
    ],
)
def test_models_are_immutable(obj, field: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, field, None)


def test_annotation_equality_and_hashing() -> None:
    assert a() == a() and hash(a()) == hash(a())
    assert len({a(), a(), a(correction="received")}) == 2  # usable for deterministic deduplication


def test_reason_defaults_to_none() -> None:
    assert Annotation(0, 14, 21, SP, "receive").reason is None


# --------------------------------------------------------------------------------------------
# Valid annotations
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "annotation",
    [
        a(),  # non-styleguide without reason
        a(reason="Common misspelling"),  # non-styleguide with optional reason
        a(category=ST, correction="receives", reason="House style requires the -s form"),
        a(start=0, end=3, category=Category.CAPITALIZATION, correction="OUR"),  # paragraph start
        a(start=38, end=39, category=Category.PUNCTUATION, correction="!"),  # paragraph end
        a(start=0, end=len(TEXT), category=Category.CLARITY, correction="The report will arrive."),
        a(start=13, end=21, correction=""),  # empty correction = deletion
        a(correction="x" * ann.MAX_CORRECTION_CHARS, reason="r" * ann.MAX_REASON_CHARS),  # at the limits
        a(correction="receive\t\n\r"),  # TAB/LF/CR are legal XML characters
    ],
)
def test_valid_annotations(annotation: Annotation) -> None:
    assert code(annotation) is None


def test_unicode_text_and_corrections() -> None:
    accent = chr(0x0301)
    emoji = chr(0x1F600)
    text = "Cafe" + accent + " teh " + emoji + " niave"  # code-point offsets: "teh" = [6, 9)
    assert code(a(start=6, end=9, correction="the"), text) is None
    assert code(a(start=12, end=17, correction="na" + chr(0x00EF) + "ve"), text) is None
    assert code(a(start=0, end=5, correction="Caf" + chr(0x00E9)), text) is None  # NFD -> NFC differs: allowed
    assert code(a(start=10, end=11, category=Category.PUNCTUATION, correction=emoji + chr(0xFE0F)), text) is None
    assert code(a(start=6, end=9, correction="teh"), text) is RejectCode.INVALID_FIELD


# --------------------------------------------------------------------------------------------
# Invalid annotations
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("category", ["spelling", "Spelling", "typo", None, 3])
def test_invalid_category(category) -> None:
    assert code(a(category=category)) is RejectCode.INVALID_FIELD


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (5, 5, RejectCode.EMPTY_SPAN),
        (0, 0, RejectCode.EMPTY_SPAN),
        (len(TEXT), len(TEXT), RejectCode.EMPTY_SPAN),
        (-1, 3, RejectCode.OUT_OF_RANGE),
        (30, len(TEXT) + 1, RejectCode.OUT_OF_RANGE),
        (21, 14, RejectCode.OUT_OF_RANGE),  # reversed
        (100, 200, RejectCode.OUT_OF_RANGE),
    ],
)
def test_empty_and_out_of_range_spans(start: int, end: int, expected: RejectCode) -> None:
    assert code(a(start=start, end=end, correction="zzz")) is expected


def test_span_on_empty_paragraph_text_is_out_of_range() -> None:
    assert code(a(start=0, end=1), "") is RejectCode.OUT_OF_RANGE


@pytest.mark.parametrize(
    "fields",
    [
        {"start": 14.0},
        {"end": "21"},
        {"start": True},  # bool is not an int here
        {"paragraph_id": None},
        {"paragraph_id": -1},
    ],
)
def test_invalid_numeric_fields(fields: dict) -> None:
    assert code(dataclasses.replace(a(), **fields)) is RejectCode.INVALID_FIELD


@pytest.mark.parametrize(
    "correction",
    [None, 7, b"receive", "x" * (ann.MAX_CORRECTION_CHARS + 1)],
)
def test_invalid_correction(correction) -> None:
    assert code(a(correction=correction)) is RejectCode.INVALID_FIELD


def test_correction_identical_to_original_span() -> None:
    assert code(a(correction="recieve")) is RejectCode.INVALID_FIELD
    assert code(a(start=9, end=13, correction="will")) is RejectCode.INVALID_FIELD


def test_missing_styleguide_reason() -> None:
    assert code(a(category=ST, reason=None)) is RejectCode.INVALID_FIELD


@pytest.mark.parametrize("category", [c for c in Category if c is not Category.STYLEGUIDE])
def test_reason_is_optional_for_other_categories(category: Category) -> None:
    assert code(a(category=category, reason=None)) is None


@pytest.mark.parametrize("reason", ["", "   ", "\t\n", 5, b"rule", "r" * (ann.MAX_REASON_CHARS + 1)])
@pytest.mark.parametrize("category", [ST, G])
def test_invalid_reason_when_present(reason, category: Category) -> None:
    assert code(a(category=category, reason=reason)) is RejectCode.INVALID_FIELD


@pytest.mark.parametrize(
    "cp",
    [0x00, 0x01, 0x08, 0x0B, 0x0C, 0x0E, 0x1F, 0xD800, 0xDFFF, 0xFFFE, 0xFFFF],
)
@pytest.mark.parametrize("field", ["correction", "reason"])
def test_xml_illegal_characters_rejected(cp: int, field: str) -> None:
    bad = "re" + chr(cp) + "ceive"
    fields = {"correction": "receive", "reason": "Misspelling"}
    fields[field] = bad
    rejection = ann.validate_annotation(a(**fields), TEXT)
    assert rejection is not None and rejection.code is RejectCode.INVALID_FIELD
    assert f"U+{cp:04X}" in rejection.detail


@pytest.mark.parametrize("cp", [0x09, 0x0A, 0x0D, 0x20, 0xD7FF, 0xE000, 0xFFFD, 0x10000, 0x10FFFF])
def test_xml_legal_boundary_characters_accepted(cp: int) -> None:
    assert code(a(correction="re" + chr(cp) + "ceive", reason="r" + chr(cp))) is None


def test_validation_order_is_deterministic() -> None:
    # several problems at once: the first rule in the documented order wins
    assert code(a(category="typo", start=5, end=5, correction=None)) is RejectCode.INVALID_FIELD
    assert code(a(start=5, end=5, correction=None)) is RejectCode.EMPTY_SPAN
    assert code(a(start=-1, end=3, correction=None)) is RejectCode.OUT_OF_RANGE


def test_validation_does_not_mutate_inputs() -> None:
    annotation, text = a(reason="why"), str(TEXT)
    before = (dataclasses.astuple(annotation), text)
    ann.validate_annotation(annotation, text)
    assert (dataclasses.astuple(annotation), text) == before

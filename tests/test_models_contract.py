"""S3.1: the model finding contract (Pydantic) and the internal S3 types.

Only the contract and type definitions are tested here; quotation analysis and resolution
arrive in later S3 steps.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from enum import Enum
from pathlib import Path

import pytest
from pydantic import ValidationError

from proofreader import models as m
from proofreader.models import (
    Annotation,
    Category,
    CheckedFinding,
    Finding,
    FindingList,
    FindingRejection,
    GridWord,
    MatchMode,
    ProtectedRange,
    QuoteAnalysis,
    QuoteFamily,
    QuoteMark,
    QuoteRole,
    RangeOrigin,
    RejectCode,
    ResolutionResult,
    ResolvedFinding,
    TraceEntry,
    TraceStatus,
)


def valid(**overrides) -> dict:
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


def error_types(data: dict) -> list[str]:
    with pytest.raises(ValidationError) as exc:
        Finding.model_validate(data)
    return [e["type"] for e in exc.value.errors()]


# --------------------------------------------------------------------------------------------
# Finding: accepted inputs
# --------------------------------------------------------------------------------------------


def test_valid_finding_is_accepted_frozen_and_hashable() -> None:
    f = Finding.model_validate(valid())
    assert f.original == "recieve" and f.word_start == 3 and f.type == "spelling"
    with pytest.raises(ValidationError):
        f.original = "x"  # type: ignore[misc]
    assert hash(f) == hash(Finding.model_validate(valid()))


@pytest.mark.parametrize(
    "overrides",
    [
        {"before": "", "after": ""},  # empty anchors (paragraph edges) are allowed by the contract
        {"before": "a b c d", "after": "e f g h"},  # exactly 4 words
        {"correction": ""},  # deletion
        {"correction": "x" * m.MAX_FINDING_CORRECTION_CHARS},
        {"original": "x" * m.MAX_ORIGINAL_CHARS},
        {"reason": "r" * m.MAX_FINDING_REASON_CHARS},
        {"original": " recieve "},  # edge whitespace is stripped later by the resolver, not rejected here
        {"word_start": 0, "word_end": 0},
        {"word_start": 2, "word_end": 7},
        {"original": "ice\ncream", "correction": "ice cream"},  # newline/tab are legal XML characters
        {"before": "a\tb"},
        {"original": "caf" + chr(0xE9), "correction": "cafe"},
        {"correction": "don" + chr(0x2019) + "t"},  # the contract itself does not police quote marks
    ],
)
def test_boundary_values_are_accepted(overrides) -> None:
    Finding.model_validate(valid(**overrides))


@pytest.mark.parametrize("category", sorted(c.value for c in Category))
def test_every_category_is_accepted(category: str) -> None:
    assert Finding.model_validate(valid(type=category)).type == category


def test_finding_types_match_the_category_enum_exactly() -> None:
    assert set(m.FindingType.__args__) == {c.value for c in Category}


# --------------------------------------------------------------------------------------------
# Finding: rejected inputs (strict typing, limits, rules)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("word_start", True, "int_type"),  # bool is not an int
        ("word_start", "3", "int_type"),  # no coercion from strings
        ("word_start", 3.0, "int_type"),  # no coercion from floats
        ("word_end", None, "int_type"),
        ("original", 7, "string_type"),
        ("before", None, "string_type"),
        ("after", ["the"], "string_type"),
        ("correction", b"receive", "string_type"),
        ("reason", 1, "string_type"),
        ("type", "Spelling", "literal_error"),  # case-sensitive
        ("type", "typo", "literal_error"),
        ("type", None, "literal_error"),
        ("word_start", -1, "greater_than_equal"),
        ("word_end", -1, "greater_than_equal"),
        ("original", "", "string_too_short"),
        ("original", "x" * (m.MAX_ORIGINAL_CHARS + 1), "string_too_long"),
        ("correction", "x" * (m.MAX_FINDING_CORRECTION_CHARS + 1), "string_too_long"),
        ("reason", "", "string_too_short"),
        ("reason", "r" * (m.MAX_FINDING_REASON_CHARS + 1), "string_too_long"),
    ],
)
def test_field_types_and_limits_are_enforced(field: str, value, expected: str) -> None:
    assert error_types(valid(**{field: value})) == [expected]


@pytest.mark.parametrize("field", ["original", "reason"])
@pytest.mark.parametrize("blank", [" ", "\t\n", "   "])
def test_whitespace_only_original_and_reason_are_rejected(field: str, blank: str) -> None:
    assert error_types(valid(**{field: blank})) == ["value_error"]


@pytest.mark.parametrize("field", ["before", "after"])
def test_anchors_longer_than_four_words_are_rejected(field: str) -> None:
    assert error_types(valid(**{field: "a b c d e"})) == ["value_error"]


def test_word_end_before_word_start_is_rejected() -> None:
    assert error_types(valid(word_start=4, word_end=3)) == ["value_error"]


@pytest.mark.parametrize("field", ["original", "before", "after", "correction", "reason"])
@pytest.mark.parametrize("cp", [0x00, 0x08, 0x0B, 0x0C, 0x1F, 0xD800, 0xDFFF, 0xFFFE, 0xFFFF])
def test_xml_illegal_characters_are_rejected_in_every_string_field(field: str, cp: int) -> None:
    base = valid()
    value = (base[field] or "x") + chr(cp)
    types = error_types({**base, field: value})
    if 0xD800 <= cp <= 0xDFFF:  # lone surrogates: Pydantic's own Unicode check may reject them first
        assert types in (["value_error"], ["string_unicode"])
    else:
        assert types == ["value_error"]


def test_xml_illegal_check_matches_the_s2_annotation_rule() -> None:
    from proofreader.annotation import _xml_illegal

    for cp in list(range(0x00, 0x110)) + list(range(0xD7F0, 0xE010)) + [0xFFFD, 0xFFFE, 0xFFFF, 0x10000, 0x10FFFF]:
        ch = chr(cp)
        assert (m._first_xml_illegal(ch) is None) == (_xml_illegal(ch) is None), hex(cp)


@pytest.mark.parametrize("missing", ["original", "before", "after", "word_start", "word_end", "correction", "type", "reason"])
def test_every_field_is_required(missing: str) -> None:
    data = valid()
    del data[missing]
    assert error_types(data) == ["missing"]


def test_extra_keys_are_rejected() -> None:
    assert error_types(valid(confidence=0.9)) == ["extra_forbidden"]


def test_error_messages_do_not_echo_long_document_text() -> None:
    secret = "UNIQUE-PARAGRAPH-TEXT " * 20
    with pytest.raises(ValidationError) as exc:
        Finding.model_validate(valid(before=secret))  # anchor too long
    messages = " ".join(e["msg"] for e in exc.value.errors())
    assert "UNIQUE-PARAGRAPH-TEXT" not in messages


# --------------------------------------------------------------------------------------------
# FindingList envelope and JSON schema (S4 uses the schema; the resolver checks the raw count)
# --------------------------------------------------------------------------------------------


def test_finding_list_accepts_up_to_thirty_findings() -> None:
    assert len(FindingList.model_validate({"findings": [valid()] * 30}).findings) == 30
    assert FindingList.model_validate({"findings": []}).findings == []


def test_finding_list_rejects_more_than_thirty_and_extra_keys() -> None:
    with pytest.raises(ValidationError) as exc:
        FindingList.model_validate({"findings": [valid()] * 31})
    assert [e["type"] for e in exc.value.errors()] == ["too_long"]
    with pytest.raises(ValidationError) as exc:
        FindingList.model_validate({"findings": [], "notes": "x"})
    assert [e["type"] for e in exc.value.errors()] == ["extra_forbidden"]
    with pytest.raises(ValidationError):
        FindingList.model_validate({"findings": "not a list"})


def test_json_schema_is_strict_and_capped() -> None:
    schema = FindingList.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["findings"]
    assert schema["properties"]["findings"]["maxItems"] == m.MAX_FINDINGS_PER_PARAGRAPH == 30
    finding = schema["$defs"]["Finding"]
    assert finding["additionalProperties"] is False
    assert set(finding["required"]) == {"original", "before", "after", "word_start", "word_end", "correction", "type", "reason"}
    assert set(finding["properties"]["type"]["enum"]) == {c.value for c in Category}
    assert finding["properties"]["original"]["maxLength"] == 300
    assert finding["properties"]["correction"]["maxLength"] == 300
    assert finding["properties"]["reason"]["maxLength"] == 200


# --------------------------------------------------------------------------------------------
# RejectCode: S3 additions, S2 codes unchanged
# --------------------------------------------------------------------------------------------

S2_CODES = [
    "INVALID_FIELD", "EMPTY_SPAN", "OUT_OF_RANGE", "PARA_NOT_ANNOTATABLE", "SPLITS_ATOMIC", "CROSSES_ELEMENT",
    "CROSSES_PARAGRAPH", "IN_CDATA", "OVERLAP", "CONFLICT", "VERIFY_FAILED",
]
S3_CODES = [
    "NOT_FOUND", "ANCHOR_MISMATCH", "AMBIGUOUS", "SPAN_TOO_LONG", "TOO_MANY_FINDINGS", "SCHEMA_INVALID",
    "IN_QUOTATION", "TOUCHES_QUOTATION", "QUOTATION_IN_CORRECTION",
]


def test_reject_codes_are_s2_codes_unchanged_followed_by_the_approved_s3_codes() -> None:
    assert [c.name for c in RejectCode] == S2_CODES + S3_CODES
    assert all(c.value == c.name for c in RejectCode)
    assert "HINT_DRIFT" not in RejectCode.__members__  # informational only (decision B)


# --------------------------------------------------------------------------------------------
# Internal S3 types: immutable, exact fields
# --------------------------------------------------------------------------------------------

ANN = Annotation(0, 0, 3, Category.SPELLING, "the", "Misspelling")
INSTANCES = [
    CheckedFinding(0, "teh", "", "", 0, 0, "the", Category.SPELLING, "Misspelling", False),
    FindingRejection(0, None, RejectCode.TOO_MANY_FINDINGS, "31 findings"),
    ResolvedFinding(0, ANN, MatchMode.EXACT, False),
    ResolutionResult(0, 1, (ResolvedFinding(0, ANN, MatchMode.EXACT, False),), ()),
    GridWord(0, 0, 3, "teh"),
    QuoteMark(4, chr(0x201C), QuoteFamily.DOUBLE, QuoteRole.OPEN),
    ProtectedRange(4, 9, False, False, RangeOrigin.PAIR),
    QuoteAnalysis(0, (), (), None, None),
    TraceEntry(0, 0, TraceStatus.APPLIED, None),
]
FIELDS = {
    CheckedFinding: ["index", "original", "before", "after", "word_start", "word_end", "correction", "category", "reason", "original_stripped"],
    FindingRejection: ["paragraph_id", "finding_index", "code", "detail"],
    ResolvedFinding: ["finding_index", "annotation", "match_mode", "hint_drift"],
    ResolutionResult: ["paragraph_id", "raw_count", "resolved", "rejections"],
    GridWord: ["index", "start", "end", "text"],
    QuoteMark: ["offset", "char", "family", "role"],
    ProtectedRange: ["start", "end", "open_start", "open_end", "origin"],
    QuoteAnalysis: ["paragraph_id", "marks", "ranges", "unsafe_reason", "inherited_from"],
    TraceEntry: ["paragraph_id", "finding_index", "status", "code"],
}


@pytest.mark.parametrize("obj", INSTANCES, ids=lambda o: type(o).__name__)
def test_internal_types_are_frozen_with_the_approved_fields(obj) -> None:
    assert [f.name for f in dataclasses.fields(obj)] == FIELDS[type(obj)]
    first = dataclasses.fields(obj)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, first, None)


@pytest.mark.parametrize(
    ("enum", "values"),
    [
        (MatchMode, ["exact", "whitespace_gap", "nfc"]),  # decision A order
        (QuoteFamily, ["double", "single", "corner", "white_corner", "other"]),
        (QuoteRole, ["OPEN", "CLOSE", "OPEN_CANDIDATE", "CLOSE_OR_APOSTROPHE", "APOSTROPHE", "AMBIGUOUS"]),
        (RangeOrigin, ["pair", "unclosed", "empty_stack_close", "ambiguous", "guard", "inherited"]),
        (TraceStatus, ["APPLIED", "REJECTED_S3", "REJECTED_S2", "ROLLED_BACK", "DOCUMENT_FAILED"]),
    ],
)
def test_internal_enums_have_exactly_the_planned_values(enum: type[Enum], values: list[str]) -> None:
    assert [e.value for e in enum] == values


# --------------------------------------------------------------------------------------------
# Boundaries: models stays a leaf module; S2 types untouched
# --------------------------------------------------------------------------------------------


def test_models_imports_only_stdlib_and_pydantic() -> None:
    tree = ast.parse(Path(inspect.getsourcefile(m)).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported == {"__future__", "dataclasses", "enum", "typing", "pydantic"}


def test_s2_annotation_and_rejection_fields_are_unchanged() -> None:
    assert [f.name for f in dataclasses.fields(Annotation)] == ["paragraph_id", "start", "end", "category", "correction", "reason"]
    assert [f.name for f in dataclasses.fields(m.Rejection)] == ["annotation", "code", "detail"]
    assert [f.name for f in dataclasses.fields(m.AnnotationOutcome)] == [
        "ok", "output", "applied", "rolled_back", "rejected", "failures", "rounds"]

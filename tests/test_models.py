"""S1 step 1: data model and error hierarchy."""

from __future__ import annotations

import dataclasses

import pytest

from proofreader import models as m


@pytest.mark.parametrize(
    ("cls", "code"),
    [
        (m.ProofreaderError, 1),
        (m.InvalidInputError, 1),
        (m.EmptyInputError, 1),
        (m.InvalidEncodingError, 1),
        (m.MalformedXMLError, 1),
        (m.UnsupportedInputError, 2),
        (m.UnsupportedEncodingError, 2),
        (m.UnsupportedXMLVersionError, 2),
        (m.InputTooLargeError, 2),
    ],
)
def test_exit_codes(cls: type[m.ProofreaderError], code: int) -> None:
    err = cls("msg")
    assert err.exit_code == code
    assert isinstance(err, m.ProofreaderError)


def test_error_carries_location() -> None:
    err = m.InvalidEncodingError("bad byte", byte_offset=7, line=1, column=4)
    assert (err.message, err.byte_offset, err.line, err.column) == ("bad byte", 7, 1, 4)
    assert str(err) == "bad byte"


def test_dataclasses_are_frozen() -> None:
    tok = m.Token(0, 3, "teh")
    with pytest.raises(dataclasses.FrozenInstanceError):
        tok.start = 1  # type: ignore[misc]


def test_paragraph_annotatable_follows_skip_reason() -> None:
    p = m.Paragraph(id=0, element_id=1, atoms=(), text="", barriers=(), skip_reason=m.SkipReason.EMPTY)
    assert not p.annotatable
    q = dataclasses.replace(p, skip_reason=None)
    assert q.annotatable

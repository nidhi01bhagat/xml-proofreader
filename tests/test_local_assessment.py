"""Local-only checks against the supplied assessment files (golden-test policy).

* The files live in the git-ignored ``SSE Assignment/`` directory; every test here is skipped
  when they are absent (clean clone). Only these tests skip.
* Files are opened read-only; each test verifies their bytes are unchanged afterwards.
* No assessment content is stored in this repository: expectations are derived from the files
  themselves at run time (e.g. the example output must have the example input's paragraph text).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from proofreader import tokens, xml_source
from proofreader.models import AtomKind, SkipReason

ASSESSMENT = Path(__file__).resolve().parents[1] / "SSE Assignment"
NAMES = ("example_input.xml", "example_output.xml", "sample_input.xml")
pytestmark = pytest.mark.skipif(
    not all((ASSESSMENT / n).is_file() for n in NAMES),
    reason="local assessment files (SSE Assignment/example_input.xml, example_output.xml, sample_input.xml) absent",
)


@pytest.fixture
def read_only():
    """Read assessment files and assert they are byte-identical after the test."""
    before = {n: hashlib.sha256((ASSESSMENT / n).read_bytes()).hexdigest() for n in NAMES}
    yield lambda name: (ASSESSMENT / name).read_bytes()
    after = {n: hashlib.sha256((ASSESSMENT / n).read_bytes()).hexdigest() for n in NAMES}
    assert after == before, "assessment files changed during the test"


@pytest.mark.parametrize("name", ["example_input.xml", "sample_input.xml"])
def test_supplied_inputs_are_fully_mappable(read_only, name: str) -> None:
    data = read_only(name)
    r = xml_source.scan(data)
    assert r.paragraphs, "expected at least one <p>"
    assert all(p.skip_reason is None for p in r.paragraphs)
    for p in r.paragraphs:
        for a in p.atoms:
            if a.kind is AtomKind.LIT:
                assert data[a.raw_start : a.raw_end].decode(r.encoding.codec) == a.text
        # every character maps to its own source bytes
        for c in range(len(p.text)):
            s = xml_source.open_byte(p, c, r.encoding.codec)
            e = xml_source.close_byte(p, c + 1, r.encoding.codec)
            assert data[s:e].decode(r.encoding.codec) == p.text[c]


def test_example_output_paragraph_text_equals_example_input(read_only) -> None:
    inp = xml_source.scan(read_only("example_input.xml"))
    out = xml_source.scan(read_only("example_output.xml"))
    assert [p.text for p in out.paragraphs] == [p.text for p in inp.paragraphs]


def test_example_output_paragraphs_with_errors_are_skipped(read_only) -> None:
    data = read_only("example_output.xml")
    out = xml_source.scan(data)
    for p in out.paragraphs:
        el = out.elements[p.element_id]
        has_error = b"<error" in data[el.start_tag[0] : el.end_tag[1]]
        assert p.skip_reason is (SkipReason.PARA_HAS_ERROR if has_error else None)
    assert any(p.skip_reason is SkipReason.PARA_HAS_ERROR for p in out.paragraphs)
    assert any(p.skip_reason is None for p in out.paragraphs)


@pytest.mark.parametrize("name", NAMES)
def test_token_grids_on_supplied_text(read_only, name: str) -> None:
    for p in xml_source.scan(read_only(name)).paragraphs:
        assert [w.text for w in tokens.words(p.text)] == p.text.split()
        for tok in tokens.fine_tokens(p.text):
            assert p.text[tok.start : tok.end] == tok.text and not tok.text.isspace()

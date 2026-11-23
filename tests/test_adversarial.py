"""S4.8c: adversarial and edge-case tests through the command line (cli.main) and the pipeline.

Every case is invented. Each asserts the exit code, whether a corrected file was written (and its exact bytes
or byte-identity with the input), the exact paragraphs offered to the model (strict Fake), and - where output
exists - the independent verifier (V1-V6). The session network guard (tests/conftest.py) fails any test that
attempts non-loopback network access, so "external DTD / entity never fetched" is enforced, not assumed.
Unit-level coverage of the same rules lives in the S1-S3 suites; these cases add the end-to-end path.
"""

from __future__ import annotations

import dataclasses
import io
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import docx_factory as f
import pytest

from proofreader import annotation as an
from proofreader import cli
from proofreader import config as c
from proofreader import encoding as enc
from proofreader import pipeline as pl
from proofreader import resolver as r
from proofreader import style_guide
from proofreader import verify as v
from proofreader import xml_source as xs
from proofreader.llm import CallStatus, FakeOk, FakeProofreader
from proofreader.models import Annotation, Category, RejectCode, TraceStatus

NO_DOTENV = {"PYTHON_DOTENV_DISABLED": "1"}
GUIDE_DOCX = f.make_docx(f.heading("House rules", 0) + f.para("Write per cent in running text, never the % sign."))
EMPTY = {"findings": []}


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def cli_run(tmp_path: Path, data: bytes, fake, *extra: str, guide: bytes = GUIDE_DOCX, lang: str = "en"):
    (tmp_path / "doc.xml").write_bytes(data)
    (tmp_path / "g.docx").write_bytes(guide)
    out, err = io.StringIO(), io.StringIO()
    code = cli.main([str(tmp_path / "doc.xml"), "--lang", lang, "--style-guide", str(tmp_path / "g.docx"), *extra],
                    environ=NO_DOTENV, cwd=tmp_path, proofreader_factory=lambda cfg, rec: fake,
                    stdout=out, stderr=err)
    target = tmp_path / "doc.corrected.xml"
    return code, (target.read_bytes() if target.exists() else None), err.getvalue()


def empty_fake() -> FakeProofreader:
    return FakeProofreader(default=FakeOk(EMPTY))


def finding(data: bytes, pid: int, original: str, correction: str, type_: str = "spelling", reason: str = "Fix",
            *, before: str | None = None, after: str | None = None) -> dict:
    """Anchors are the 3 whitespace-separated words around the span unless given explicitly (needed where a
    nested-<p> barrier ends the segment: anchors never cross a barrier)."""
    para = next(p for p in xs.scan(data).paragraphs if p.id == pid)
    text = para.text
    start = text.index(original)
    end = start + len(original)
    grid = r.word_grid(para)
    first = next((w.index for w in grid if w.end > start), 0)
    last = max((w.index for w in grid if w.start < end), default=first)
    return {"original": original,
            "before": " ".join(text[:start].split()[-3:]) if before is None else before,
            "after": " ".join(text[end:].split()[:3]) if after is None else after,
            "word_start": first, "word_end": last, "correction": correction, "type": type_, "reason": reason}


def env(*items: dict, **extra) -> dict:
    return {"findings": list(items), **extra}


def make_cfg(tmp_path: Path, lang: str = "en") -> c.RunConfig:
    return c.build_config({"input": "a.xml", "lang": lang, "style_guide": "g.docx"}, NO_DOTENV, tmp_path)


def pipeline_run(tmp_path: Path, data: bytes, script: dict[int, dict], lang: str = "en"):
    cfg = make_cfg(tmp_path, lang)
    guide = style_guide.select(style_guide.parse_docx(GUIDE_DOCX), cfg.locale_scope, style_guide.load_manifest())
    fake = FakeProofreader({pid: [FakeOk(p)] for pid, p in script.items()}, default=FakeOk(EMPTY))
    return pl.run(data, cfg, fake, guide)


def codes(result) -> list[tuple[int, int | None, TraceStatus, RejectCode | None]]:
    return [(t.paragraph_id, t.finding_index, t.status, t.code) for t in result.trace]


def errors_in(output: bytes) -> list[Annotation]:
    root = ET.fromstring(output, parser=ET.XMLParser(encoding="utf-8"))
    paragraphs = [e for e in root.iter() if isinstance(e.tag, str) and e.tag.rsplit("}", 1)[-1] == "p"]
    found = []
    for pid, para in enumerate(paragraphs):
        _, errs = v._own_text_and_errors(para)
        found += [Annotation(pid, s, e, Category(el.get("type")), el.get("correction"), el.get("reason"))
                  for s, e, el in errs]
    return found


def assert_verified(original: bytes, output: bytes) -> None:
    """Independent check of any produced output: bytes outside <error> markup unchanged, V1-V6 pass."""
    strip = lambda b: re.sub(rb"<error [^>]*>|</error>", b"", b)  # noqa: E731
    assert strip(output) == strip(original)
    added = [a for a in errors_in(output) if a not in errors_in(original)]
    assert v.verify(original, output, added).ok


# --------------------------------------------------------------------------------------------
# A. Hostile XML (AR1) through the CLI
# --------------------------------------------------------------------------------------------

LOL = (b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaaaaaaaa">'
       + b"".join(b'<!ENTITY %s "%s">' % (bytes([98 + i]), (b"&%s;" % bytes([97 + i])) * 10) for i in range(8))
       + b"]><r><p>&i;</p></r>")

HOSTILE = {
    # name: (input, exit code, output identical to input (None = no output), paragraphs sent)
    "billion_laughs": (LOL, 1, None, []),
    # ElementTree cannot verify these documents: every paragraph is skipped (C2) and reported as not proofread,
    # so the run is degraded (exit 3; S4.8c fix, formerly a silent exit 0)
    "external_dtd_not_fetched": (b'<!DOCTYPE r SYSTEM "http://192.0.2.1/x.dtd"><r><p>teh &u;</p><p>ok text</p></r>',
                                 3, True, []),
    "external_entity_not_fetched": (b'<!DOCTYPE r [<!ENTITY e SYSTEM "http://192.0.2.1/e">]><r><p>see &e; here</p>'
                                    b"<p>ok text</p></r>", 3, True, []),
    "parameter_entity_not_parsed": (b'<!DOCTYPE r [<!ENTITY % p SYSTEM "http://192.0.2.1/p.dtd"> %p;]>'
                                    b"<r><p>plain text</p></r>", 0, True, [0]),
    "undeclared_entity": (b"<r><p>teh &nope; cat</p></r>", 1, None, []),
    "markup_bearing_entity": (b'<!DOCTYPE r [<!ENTITY m "<b>x</b>">]><r><p>teh &m;</p><p>plain words</p></r>',
                              3, True, [1]),  # ENT-07: that paragraph passes through, reported as not proofread
    "malformed": (b"<r><p>unclosed</r>", 1, None, []),
    "two_roots": (b"<r/><r/>", 1, None, []),
    "empty_file": (b"", 1, None, []),
    "bom_only": (b"\xef\xbb\xbf", 1, None, []),
    "whitespace_only_document": (b"   \n", 1, None, []),
    "xml_1_1": (b'<?xml version="1.1"?><r><p>x</p></r>', 2, None, []),
    "nul_character_reference": (b"<r><p>a &#0; b</p></r>", 1, None, []),
    "deep_nesting_2000": (b"<r>" + b"<d>" * 2000 + b"<p>deep text</p>" + b"</d>" * 2000 + b"</r>", 0, True, [0]),
    "attribute_of_1_mb": (b'<r a="' + b"x" * 1_000_000 + b'"><p>text here</p></r>', 0, True, [0]),
}


GUARD_MESSAGE = "entity expansion limit exceeded (possible amplification attack)"


def repeated_entity(value_chars: int, uses: int) -> bytes:
    return (b'<!DOCTYPE r [<!ENTITY e "' + b"x" * value_chars + b'">]><r><p>'
            + b"&e;" * uses + b"</p></r>")


# The S1 early guard limits text delivered BEYOND the input's own size: budget = len(input) + (1 << 20) characters
# (it measures expansion, not the size of an entity written literally in the input).
@pytest.mark.parametrize("data", [
    LOL,  # high amplification: Expat's own protection would also refuse it, but our guard fires first
    repeated_entity(120_000, 10),  # ~1.08 MiB beyond the input at 10x amplification: below Expat's 8 MiB
], ids=["billion_laughs", "moderate_amplification_over_budget"])  # activation threshold, so only our guard fires
def test_entity_expansion_guard_fires(tmp_path, data) -> None:
    assert len(data) < 1 << 20
    fake = empty_fake()
    code, output, err = cli_run(tmp_path, data, fake)
    assert code == 1 and output is None and fake.calls == ()
    assert GUARD_MESSAGE in err


@pytest.mark.parametrize("data", [repeated_entity(100_000, 10),  # ~0.9 MiB beyond the input: within budget
                                  repeated_entity(1_100_000, 1)],  # large but written literally: no expansion
                         ids=["moderate_amplification_within_budget", "large_literal_entity_used_once"])
def test_entity_expansion_within_budget_is_accepted(tmp_path, data) -> None:
    fake = empty_fake()
    code, output, err = cli_run(tmp_path, data, fake)
    assert code == 3 and output == data and fake.calls == ()  # accepted; the huge paragraph is size-guarded
    assert GUARD_MESSAGE not in err


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_hostile_xml(tmp_path, name, blocked_attempts) -> None:
    data, exit_code, identical, sent = HOSTILE[name]
    fake = empty_fake()
    code, output, err = cli_run(tmp_path, data, fake)
    assert code == exit_code, err
    assert sorted(fake.call_counts()) == sent
    if identical is None:
        assert output is None and "no corrected output was written" in err
    else:
        assert output == data
    assert blocked_attempts == []  # nothing was fetched over the network
    assert (tmp_path / "doc.xml").read_bytes() == data


# --------------------------------------------------------------------------------------------
# B. Encodings
# --------------------------------------------------------------------------------------------

ENCODINGS = {
    "utf16le_bom": ("<r><p>x</p></r>".encode("utf-16"), 2, "UTF-16 (little-endian BOM)"),
    "utf16be_no_bom": ("<r><p>x</p></r>".encode("utf-16-be"), 2, "UTF-16/UTF-32 without BOM (NUL byte at the start)"),
    "utf32_bom": ("<r><p>x</p></r>".encode("utf-32"), 2, "UTF-32 (little-endian BOM)"),
    "utf8_bom_with_utf16_declaration": (b'\xef\xbb\xbf<?xml version="1.0" encoding="UTF-16"?><r><p>x</p></r>', 2,
                                        "contradicts"),
    "latin1_declaration": (b'<?xml version="1.0" encoding="ISO-8859-1"?><r><p>x</p></r>', 2, "ISO-8859-1"),
    "invalid_utf8_byte": (b"<r><p>caf\xe9 x</p></r>", 1, "byte offset 9"),
    "us_ascii_with_high_byte": (b'<?xml version="1.0" encoding="US-ASCII"?><r><p>caf\xc3\xa9</p></r>', 1,
                                "byte offset 50"),
}


@pytest.mark.parametrize("name", sorted(ENCODINGS))
def test_unsupported_or_invalid_encodings_write_nothing(tmp_path, name) -> None:
    data, exit_code, message = ENCODINGS[name]
    fake = empty_fake()
    code, output, err = cli_run(tmp_path, data, fake)
    assert code == exit_code and output is None and fake.calls == ()
    assert message in err


def test_input_size_cap_boundary(tmp_path) -> None:
    cap = enc.MAX_INPUT_BYTES
    body = b"x " * ((cap - len(b"<r><p></p></r>")) // 2)
    at_cap = b"<r><p>" + body + b"</p></r>"
    assert len(at_cap) <= cap
    fake = empty_fake()
    code, output, err = cli_run(tmp_path, at_cap, fake)
    assert code == 3 and output == at_cap and fake.calls == ()  # accepted; the one huge paragraph is size-guarded
    assert "passthrough" not in err and "not_called (request_too_large)" in err
    over = b"<r><p>" + b"x" * (cap - len(b"<r><p></p></r>") + 1) + b"</p></r>"
    assert len(over) == cap + 1
    (tmp_path / "doc.corrected.xml").unlink()
    code, output, err = cli_run(tmp_path, over, empty_fake())
    assert code == 2 and output is None and f"the limit is {cap} bytes" in err


# --------------------------------------------------------------------------------------------
# C. Paragraph limits and structure through the CLI
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("text,sent", [(" ".join(f"w{i}" for i in range(50)), True),
                                       (" ".join(f"w{i}" for i in range(51)), False),
                                       ("y" * 5000, True), ("y" * 5001, False)])
def test_paragraph_size_limits(tmp_path, text, sent) -> None:
    data = f"<r><p>{text}</p></r>".encode("utf-8")
    fake = empty_fake()
    code, output, err = cli_run(tmp_path, data, fake)
    assert output == data and (sorted(fake.call_counts()) == ([0] if sent else []))
    assert code == (0 if sent else 3)
    if not sent:
        assert "WARNING paragraph 0: not_called (request_too_large), attempts 0" in err


def test_eligibility_across_unusual_paragraph_structures(tmp_path) -> None:
    data = ('<r xmlns:n="urn:example:n">'
            "<p>  </p>"  # 0 whitespace only
            "<p> </p>"  # 1 no-break space only
            "<p><![CDATA[only cdata]]></p>"  # 2 CDATA only
            "<p>Outer <p>Middle <p>Inner</p> m</p> o</p>"  # 3, 4, 5 nested three levels
            "<n:p>Namespaced text</n:p>"  # 6 prefixed paragraph
            "<p>A <!-- c --> b <?pi x?> c</p>"  # 7 comment and PI inside
            '<p>Top <p>Has <error type="spelling" correction="x">y</error></p> rest</p>'  # 8, 9: the error skips both
            "</r>").encode("utf-8")
    fake = FakeProofreader({pid: [FakeOk(EMPTY)] for pid in (3, 4, 5, 6, 7)})  # strict: any other call fails
    code, output, err = cli_run(tmp_path, data, fake)
    assert code == 0, err
    assert output == data and sorted(fake.call_counts()) == [3, 4, 5, 6, 7]


def test_nested_paragraphs_annotated_independently(tmp_path) -> None:
    data = b"<r><p>One teh <p>Two teh <p>Three teh</p> b</p> c</p></r>"
    script = {0: env(finding(data, 0, "teh", "the", after="")),  # segment ends at the nested <p>: empty anchor
              1: env(finding(data, 1, "teh", "the", after="")),
              2: env(finding(data, 2, "teh", "the"))}
    result = pipeline_run(tmp_path, data, script)
    expected = (b'<r><p>One <error type="spelling" correction="the" reason="Fix">teh</error> <p>Two '
                b'<error type="spelling" correction="the" reason="Fix">teh</error> <p>Three '
                b'<error type="spelling" correction="the" reason="Fix">teh</error></p> b</p> c</p></r>')  # by hand
    assert result.output == expected
    assert [s for _, _, s, _ in codes(result)] == [TraceStatus.APPLIED] * 3
    assert_verified(data, result.output)


def test_span_across_a_nested_paragraph_barrier_is_never_applied(tmp_path) -> None:
    data = b"<r><p>One teh <p>Two</p> c</p></r>"
    result = pipeline_run(tmp_path, data, {0: env({"original": "teh c", "before": "One", "after": "",
                                                   "word_start": 1, "word_end": 2, "correction": "the c",
                                                   "type": "spelling", "reason": "x"})})
    assert result.output == data and codes(result) == [(0, 0, TraceStatus.REJECTED_S3, RejectCode.NOT_FOUND)]


# --------------------------------------------------------------------------------------------
# D. Quotations and Unicode
# --------------------------------------------------------------------------------------------

U = lambda s: s.encode("utf-8")  # noqa: E731
QUOTE_CASES = {
    # name: (input, finding builder, expected (status, code), expected output or None = unchanged)
    "german_quotation": (U("<r><p>Er sagte „teh Haus“ und ging.</p></r>"),
                         lambda d: finding(d, 0, "teh", "the"), (TraceStatus.REJECTED_S3, RejectCode.IN_QUOTATION)),
    "french_guillemets": (U("<r><p>Il a dit « teh maison » puis partit.</p></r>"),
                          lambda d: finding(d, 0, "teh", "the"), (TraceStatus.REJECTED_S3, RejectCode.IN_QUOTATION)),
    "english_curly_quotation": (U("<r><p>He said “teh cat” and left.</p></r>"),
                                lambda d: finding(d, 0, "teh", "the"),
                                (TraceStatus.REJECTED_S3, RejectCode.IN_QUOTATION)),
    "original_contains_quote_mark": (U('<r><p>She wrote "teh" twice.</p></r>'),
                                     lambda d: finding(d, 0, 'wrote "teh', 'wrote "the'),
                                     (TraceStatus.REJECTED_S3, RejectCode.TOUCHES_QUOTATION)),
    "quotation_mark_in_correction": (U("<r><p>The cat sat down.</p></r>"), lambda d: finding(d, 0, "cat", '"cat"'),
                                     (TraceStatus.REJECTED_S3, RejectCode.QUOTATION_IN_CORRECTION)),
    "split_combining_mark": (U("<r><p>The café is cosy.</p></r>"),
                             lambda d: {"original": "cafe", "before": "The", "after": "is cosy.", "word_start": 1,
                                        "word_end": 1, "correction": "coffee", "type": "spelling", "reason": "x"},
                             (TraceStatus.REJECTED_S3, RejectCode.NOT_FOUND)),
}


@pytest.mark.parametrize("name", sorted(QUOTE_CASES))
def test_protected_or_unsafe_spans_are_rejected(tmp_path, name) -> None:
    data, build, expected = QUOTE_CASES[name]
    result = pipeline_run(tmp_path, data, {0: env(build(data))})
    assert result.output == data and codes(result) == [(0, 0, *expected)]


@pytest.mark.parametrize("data,build,expected", [
    (U("<r><p>The café is cosy.</p></r>"),  # NFD text, NFC original (tier 3)
     lambda d: dict(finding(d, 0, "café", "cafés"), original="café"),
     U('<r><p>The <error type="spelling" correction="cafés" reason="Fix">café</error> is cosy.</p></r>')),
    (U("<r><p>Family \U0001F468‍\U0001F469 teh trip.</p></r>"),  # ZWJ emoji neighbour
     lambda d: finding(d, 0, "teh", "the"),
     U('<r><p>Family \U0001F468‍\U0001F469 <error type="spelling" correction="the" reason="Fix">teh</error>'
       " trip.</p></r>")),
    (b"<r><p>We ate ice\r\n  cream today.</p></r>",  # whitespace gap (tier 2); CRLF kept inside the span
     lambda d: {"original": "ice cream", "before": "We ate", "after": "today.", "word_start": 2, "word_end": 3,
                "correction": "ice-cream", "type": "spelling", "reason": "x"},
     b'<r><p>We ate <error type="spelling" correction="ice-cream" reason="x">ice\r\n  cream</error> today.</p></r>'),
    (b"<r><p>Wait ,, then go.</p></r>",  # punctuation-only span
     lambda d: finding(d, 0, ",,", ",", "punctuation"),
     b'<r><p>Wait <error type="punctuation" correction="," reason="Fix">,,</error> then go.</p></r>'),
], ids=["nfd_text_nfc_original", "zwj_emoji_neighbour", "whitespace_gap_crlf", "punctuation_only"])
def test_unicode_and_whitespace_matching(tmp_path, data, build, expected) -> None:
    result = pipeline_run(tmp_path, data, {0: env(build(data))})
    assert result.output == expected  # hand-written expected bytes
    assert_verified(data, result.output)


# --------------------------------------------------------------------------------------------
# E. Malformed, invalid and overlapping findings
# --------------------------------------------------------------------------------------------

BASE = b"<r><p>The cat sat on teh mat today.</p></r>"
GOOD = finding(BASE, 0, "teh", "the")
APPLIED_BASE = b'<r><p>The cat sat on <error type="spelling" correction="the" reason="Fix">teh</error> mat today.</p></r>'
REJECTED = {
    "envelope_extra_key": (env(GOOD, extra=1), [(0, 0, TraceStatus.REJECTED_S3, RejectCode.SCHEMA_INVALID)]),
    "findings_not_a_list": ({"findings": "teh"}, [(0, None, TraceStatus.REJECTED_S3, RejectCode.SCHEMA_INVALID)]),
    "wrong_field_type": (env(dict(GOOD, word_start="3")), [(0, 0, TraceStatus.REJECTED_S3, RejectCode.INVALID_FIELD)]),
    "negative_word_index": (env(dict(GOOD, word_start=-1)),
                            [(0, 0, TraceStatus.REJECTED_S3, RejectCode.INVALID_FIELD)]),
    "correction_equals_original": (env(dict(GOOD, correction="teh")),
                                   [(0, 0, TraceStatus.REJECTED_S3, RejectCode.INVALID_FIELD)]),
    "nul_in_correction": (env(dict(GOOD, correction="t\u0000he")),
                          [(0, 0, TraceStatus.REJECTED_S3, RejectCode.INVALID_FIELD)]),
    "noncharacter_in_correction": (env(dict(GOOD, correction="t￾he")),
                                   [(0, 0, TraceStatus.REJECTED_S3, RejectCode.INVALID_FIELD)]),
    "correction_301_chars": (env(dict(GOOD, correction="x" * 301)),
                             [(0, 0, TraceStatus.REJECTED_S3, RejectCode.INVALID_FIELD)]),
    "reason_201_chars": (env(dict(GOOD, reason="r" * 201)), [(0, 0, TraceStatus.REJECTED_S3, RejectCode.INVALID_FIELD)]),
    "styleguide_without_reason": (env(dict(GOOD, type="styleguide", reason="")),
                                  [(0, 0, TraceStatus.REJECTED_S3, RejectCode.INVALID_FIELD)]),
    "conflicting_corrections_same_span": (env(GOOD, dict(GOOD, correction="thee")),
                                          [(0, 0, TraceStatus.REJECTED_S2, RejectCode.CONFLICT),
                                           (0, 1, TraceStatus.REJECTED_S2, RejectCode.CONFLICT)]),
}


@pytest.mark.parametrize("name", sorted(REJECTED))
def test_invalid_findings_are_rejected_and_traced(tmp_path, name) -> None:
    payload, expected = REJECTED[name]
    result = pipeline_run(tmp_path, BASE, {0: payload})
    assert result.output == BASE and codes(result) == expected
    assert not result.degraded  # rejected findings are not a paragraph failure


def test_31_findings_reject_the_whole_paragraph_30_are_accepted(tmp_path) -> None:
    data = ("<r><p>" + " ".join(f"w{i}" for i in range(40)) + "</p></r>").encode("utf-8")
    many = [finding(data, 0, f"w{i}", f"W{i}") for i in range(31)]
    result = pipeline_run(tmp_path, data, {0: env(*many)})
    assert result.output == data
    assert {(s, cd) for _, _, s, cd in codes(result)} == {(TraceStatus.REJECTED_S3, RejectCode.TOO_MANY_FINDINGS)}
    result = pipeline_run(tmp_path, data, {0: env(*many[:30])})
    assert result.output.count(b"<error ") == 30
    assert_verified(data, result.output)


@pytest.mark.parametrize("payload", [env(dict(GOOD, correction="x" * 300)), env(dict(GOOD, reason="r" * 200)),
                                     env(dict(GOOD, word_start=10**9, word_end=10**9)), env(GOOD, dict(GOOD))],
                         ids=["correction_300_chars", "reason_200_chars", "huge_word_hint", "exact_duplicate"])
def test_boundary_values_that_are_accepted(tmp_path, payload) -> None:
    result = pipeline_run(tmp_path, BASE, {0: payload})
    assert result.output.count(b"<error ") == 1
    assert_verified(BASE, result.output)


def test_special_characters_round_trip_exactly_through_the_attribute(tmp_path) -> None:
    correction = "a&b<c>d\ng\rh\ti"
    result = pipeline_run(tmp_path, BASE, {0: env(dict(GOOD, correction=correction))})
    assert b'correction="a&amp;b&lt;c&gt;d&#10;g&#13;h&#9;i"' in result.output
    (error,) = errors_in(result.output)
    assert error.correction == correction  # an XML parser reads back exactly the original value
    assert_verified(BASE, result.output)


def test_thirty_mutually_overlapping_findings_are_all_rejected(tmp_path) -> None:
    data = b"<r><p>alpha beta gamma delta epsilon zeta eta theta iota kappa</p></r>"
    words = data[6:-8].decode().split()
    many = [finding(data, 0, " ".join(words[i % 5:i % 5 + 3]), f"X{i}", reason=f"r{i}") for i in range(30)]
    result = pipeline_run(tmp_path, data, {0: env(*many)})
    assert result.output == data and len(result.trace) == 30
    assert all(s is TraceStatus.REJECTED_S2 and cd in {RejectCode.CONFLICT, RejectCode.OVERLAP}
               for _, _, s, cd in codes(result))


# --------------------------------------------------------------------------------------------
# F. Style-guide failures through the CLI
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("guide,message", [
    (b"not a zip archive", "style guide"),
    (f.make_docx(f.p(f.t("x"), f'<w:ins w:id="1" w:author="a">{f.t("y")}</w:ins>')), "tracked changes"),
], ids=["not_a_docx", "tracked_changes"])
def test_unusable_style_guide_is_fatal_before_any_call(tmp_path, guide, message) -> None:
    fake = empty_fake()
    code, output, err = cli_run(tmp_path, BASE, fake, guide=guide)
    assert code == 1 and output is None and fake.calls == () and message in err


def test_styleguide_finding_rejected_when_the_locale_selects_no_rules(tmp_path) -> None:
    data = b"<r><p>Sales rose by 12% this year.</p></r>"
    fake = FakeProofreader({0: [FakeOk(env(finding(data, 0, "12%", "12 per cent", "styleguide", "House rule")))]})
    code, output, err = cli_run(tmp_path, data, fake, lang="de")
    assert code == 0 and output == data and "findings by code: INVALID_FIELD 1" in err


# --------------------------------------------------------------------------------------------
# G. Rollback through the CLI
# --------------------------------------------------------------------------------------------


def test_verification_failure_rolls_back_only_that_paragraph(tmp_path, monkeypatch) -> None:
    data = b"<r><p>Our team will recieve it.</p><p>They will recieve it later.</p></r>"
    real = an.annotate

    def faulty_render(d, spans, codec):  # paragraph 1 rendered with a different correction than annotated
        spans = [dataclasses.replace(s, annotation=dataclasses.replace(s.annotation, correction="bogus"))
                 if s.annotation.paragraph_id == 1 else s for s in spans]
        return an.render(d, spans, codec)

    monkeypatch.setattr(an, "annotate", lambda d, s, a: real(d, s, a, render_fn=faulty_render))
    fake = FakeProofreader({pid: [FakeOk(env(finding(data, pid, "recieve", "receive")))] for pid in (0, 1)})
    code, output, err = cli_run(tmp_path, data, fake)
    assert code == 3
    assert output == (b'<r><p>Our team will <error type="spelling" correction="receive" reason="Fix">recieve</error>'
                      b" it.</p><p>They will recieve it later.</p></r>")  # hand-written
    assert "WARNING paragraphs rolled back after verification: 1" in err
    assert_verified(data, output)


# --------------------------------------------------------------------------------------------
# H. Regressions for known risks (docs/04 "Unresolved technical risks"; S4.6 / S4.7 limitations)
# --------------------------------------------------------------------------------------------


def test_inl_07_link_target_bytes_unchanged(tmp_path) -> None:
    data = b'<r><p>See <a href="x?a=1&amp;b=2">teh</a> site</p></r>'
    result = pipeline_run(tmp_path, data, {0: env(finding(data, 0, "teh", "the"))})
    assert result.output == (b'<r><p>See <a href="x?a=1&amp;b=2"><error type="spelling" correction="the" '
                             b'reason="Fix">teh</error></a> site</p></r>')  # hand-written (03 section 8, INL-07)
    assert_verified(data, result.output)


def test_inl_08_span_equal_to_element_content(tmp_path) -> None:
    data = b"<r><p><em>teh</em></p></r>"
    result = pipeline_run(tmp_path, data, {0: env(finding(data, 0, "teh", "the"))})
    assert result.output == b'<r><p><em><error type="spelling" correction="the" reason="Fix">teh</error></em></p></r>'
    assert_verified(data, result.output)


def test_many_empty_nested_paragraphs_in_one_paragraph_complete(tmp_path) -> None:
    """docs/04 risk 12 (S1 scan cost with very many nested <p> in ONE paragraph): a moderate size must finish
    correctly. No wall-clock assertion (decision D6)."""
    data = b"<r><p>Outer teh" + b"<p/>" * 2000 + b" end</p></r>"
    fake = empty_fake()
    code, output, err = cli_run(tmp_path, data, fake)
    assert code == 0 and output == data and sorted(fake.call_counts()) == [0]


def test_max_calls_counts_paragraphs_not_http_requests(tmp_path) -> None:
    data = b"<r><p>First words here.</p><p>Second words here.</p></r>"
    fake = FakeProofreader(default=FakeOk(EMPTY, attempts=3))  # each call reports 3 HTTP requests (retries)
    code, output, err = cli_run(tmp_path, data, fake, "--max-calls", "2", "--metrics-json",
                                str(tmp_path / "m.json"))
    assert code == 0 and output == data
    metrics = json.loads((tmp_path / "m.json").read_text())
    assert metrics["requests_sent"] == 6 > 2  # documented: --max-calls limits paragraphs, not requests


def test_whitespace_only_guide_text_never_yields_styleguide_markup(tmp_path) -> None:
    """S4.6 limitation: the prompt builder still sends whitespace-only guide text; the pipeline must reject any
    resulting styleguide finding."""
    data = b"<r><p>Sales rose by 12% this year.</p></r>"
    cfg = make_cfg(tmp_path)
    guide = style_guide.SelectedGuide("  \n\t ", cfg.locale_scope, 1, 1, ())
    fake = FakeProofreader({0: [FakeOk(env(finding(data, 0, "12%", "12 per cent", "styleguide", "House rule")))]})
    result = pl.run(data, cfg, fake, guide)
    assert result.output == data
    assert codes(result) == [(0, 0, TraceStatus.REJECTED_S3, RejectCode.INVALID_FIELD)]
    assert all(rec.status is CallStatus.OK for rec in result.call_records)


# --------------------------------------------------------------------------------------------
# I. Paragraphs that could not be proofread are reported, never silent (S4.8c fix, Option A)
# --------------------------------------------------------------------------------------------

from proofreader.models import SkipReason  # noqa: E402

UNVERIFIABLE = SkipReason.DOCUMENT_UNVERIFIABLE.value
UNMAPPABLE = SkipReason.PARA_UNMAPPABLE.value
NOT_PROOFREAD_CASES = {
    # name: (input, paragraphs sent, expected not_proofread, expected WARNING lines)
    "D1_external_dtd_undeclared_entity": (
        b'<!DOCTYPE r SYSTEM "http://192.0.2.1/x.dtd"><r><p>teh &u;</p><p>ok text</p></r>', [],
        ((0, SkipReason.DOCUMENT_UNVERIFIABLE), (1, SkipReason.DOCUMENT_UNVERIFIABLE)),
        [f"WARNING paragraph 0: not proofread ({UNVERIFIABLE})", f"WARNING paragraph 1: not proofread ({UNVERIFIABLE})"]),
    "D4_external_parameter_entity_undeclared_entity": (
        b'<!DOCTYPE r [<!ENTITY % p SYSTEM "http://192.0.2.1/p.dtd"> %p;]><r><p>teh &u;</p><p>ok text</p></r>', [],
        ((0, SkipReason.DOCUMENT_UNVERIFIABLE), (1, SkipReason.DOCUMENT_UNVERIFIABLE)),
        [f"WARNING paragraph 0: not proofread ({UNVERIFIABLE})", f"WARNING paragraph 1: not proofread ({UNVERIFIABLE})"]),
    "D6_markup_bearing_internal_entity": (
        b'<!DOCTYPE r [<!ENTITY m "<b>x</b>">]><r><p>teh &m;</p><p>ok text</p></r>', [1],
        ((0, SkipReason.PARA_UNMAPPABLE),),
        [f"WARNING paragraph 0: not proofread ({UNMAPPABLE})"]),
}


@pytest.mark.parametrize("name", sorted(NOT_PROOFREAD_CASES))
def test_not_proofread_paragraphs_make_the_run_degraded(tmp_path, name, blocked_attempts) -> None:
    data, sent, expected, warnings = NOT_PROOFREAD_CASES[name]
    fake = FakeProofreader({pid: [FakeOk(EMPTY)] for pid in sent})  # strict: no other paragraph may be sent
    code, output, err = cli_run(tmp_path, data, fake, "--metrics-json", str(tmp_path / "m.json"),
                                "--log-level", "DEBUG")
    assert code == 3  # formerly 0: the text was never proofread, so this is not a full success
    assert output == data  # original bytes preserved exactly
    assert sorted(fake.call_counts()) == sent  # no model call for the affected paragraphs
    assert blocked_attempts == []  # the external DTD / entity was never fetched
    assert [ln for ln in err.splitlines() if "not proofread" in ln] == warnings
    assert "some paragraphs passed through unchanged" in err
    for fragment in ("teh", "ok text", "<b>x</b>", "192.0.2.1"):  # ids and reasons only, never XML text
        assert fragment not in err
    metrics = json.loads((tmp_path / "m.json").read_text())
    expected_counts: dict[str, int] = {}
    for _, reason in expected:
        expected_counts[reason.value] = expected_counts.get(reason.value, 0) + 1
    assert metrics["paragraphs"]["not_proofread"] == expected_counts and metrics["exit_code"] == 3
    result = pipeline_run(tmp_path, data, {pid: EMPTY for pid in sent})
    assert result.not_proofread == expected and result.degraded and result.output == data
    assert "teh" not in repr(result) and "ok text" not in repr(result)  # the new field is text-free


def test_whitespace_only_unverifiable_paragraph_is_not_counted(tmp_path) -> None:
    data = b'<!DOCTYPE r SYSTEM "http://192.0.2.1/x.dtd"><r><p>teh &u;</p><p>  </p></r>'
    result = pipeline_run(tmp_path, data, {})
    assert result.not_proofread == ((0, SkipReason.DOCUMENT_UNVERIFIABLE),)  # paragraph 1 has no text to lose


@pytest.mark.parametrize("data", [
    b'<r><p><error type="spelling" correction="cat">cta</error> sat.</p></r>',  # already annotated (AC-11)
    b"<r><p>   </p><p/></r>",  # empty
    b"<r><p><![CDATA[only cdata]]></p></r>",  # CDATA only (preservation boundary)
    b'<!DOCTYPE r SYSTEM "http://192.0.2.1/x.dtd"><r><p>teh cat</p></r>',  # external DTD, nothing unresolved
], ids=["already_annotated", "empty", "cdata_only", "external_dtd_without_undeclared_entities"])
def test_intentional_skips_still_exit_0(tmp_path, data, blocked_attempts) -> None:
    fake = empty_fake()
    code, output, err = cli_run(tmp_path, data, fake, "--metrics-json", str(tmp_path / "m.json"))
    assert code == 0 and output == data and "not proofread" not in err
    assert json.loads((tmp_path / "m.json").read_text())["paragraphs"]["not_proofread"] == {}
    assert blocked_attempts == []


# --------------------------------------------------------------------------------------------
# J. Plural years with an apostrophe (S4.8e fix: digit-letter apostrophe is not a quotation mark)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("apos", ["'", "’"])
def test_plural_year_fix_and_later_findings_in_the_paragraph_are_applied(tmp_path, apos) -> None:
    data = f"<r><p>Prices rose in the 1950{apos}s, and later time-frames changed.</p></r>".encode("utf-8")
    script = {0: env(finding(data, 0, f"1950{apos}s", "1950s", "styleguide", "No apostrophe in plural years"),
                     finding(data, 0, "time-frames", "time frames", "styleguide", "Preferred usage: time frame"))}
    result = pipeline_run(tmp_path, data, script)
    expected = (f'<r><p>Prices rose in the <error type="styleguide" correction="1950s" '
                f'reason="No apostrophe in plural years">1950{apos}s</error>, and later '
                f'<error type="styleguide" correction="time frames" reason="Preferred usage: time frame">'
                f"time-frames</error> changed.</p></r>").encode("utf-8")  # hand-written
    assert result.output == expected
    assert [s for _, _, s, _ in codes(result)] == [TraceStatus.APPLIED, TraceStatus.APPLIED]
    assert_verified(data, result.output)


def test_plural_year_inside_a_genuine_quotation_stays_protected(tmp_path) -> None:
    data = "<r><p>He wrote “the 1990's were teh best” and left.</p></r>".encode("utf-8")
    script = {0: env(finding(data, 0, "teh", "the"), finding(data, 0, "1990's", "1990s", "styleguide", "Plural years"))}
    result = pipeline_run(tmp_path, data, script)
    assert result.output == data
    assert [(s, cd) for _, _, s, cd in codes(result)] == [(TraceStatus.REJECTED_S3, RejectCode.IN_QUOTATION)] * 2

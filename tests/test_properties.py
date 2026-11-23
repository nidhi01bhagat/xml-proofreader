"""S1 step 7: Hypothesis property-based tests for tokens, encoding and xml_source.

The xml_source generator builds random *valid* documents together with an independent oracle
of every paragraph's expected own text, so ``scan`` is checked against something it did not
compute itself. Special characters are built with chr() (no invisible literals in the source).
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from proofreader import encoding, tokens, xml_source
from proofreader.models import AtomKind, ProofreaderError, SkipReason

SETTINGS = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
XML_DIR = Path(__file__).resolve().parent / "fixtures" / "xml"

# --------------------------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------------------------

_INTERESTING = [chr(c) for c in (0x0301, 0x0308, 0x200D, 0xFE0F, 0x20E3, 0x1F3FD, 0x1F1EB, 0x1F1F7, 0x1F468, 0x00A0)]
token_text = st.text(
    alphabet=st.one_of(st.characters(blacklist_categories=("Cs",)), st.sampled_from(_INTERESTING)), max_size=60
)


@SETTINGS
@given(token_text)
def test_clusters_cover_text_exactly(s: str) -> None:
    pos = 0
    for a, b in tokens.clusters(s):
        assert a == pos and b > a
        pos = b
    assert pos == len(s)


@SETTINGS
@given(token_text)
def test_fine_tokens_respect_clusters_and_cover_all_non_space_clusters(s: str) -> None:
    cluster_bounds = {0, len(s)} | {b for _, b in tokens.clusters(s)}
    toks = tokens.fine_tokens(s)
    last = 0
    for tok in toks:
        assert tok.start >= last and tok.end > tok.start
        assert tok.start in cluster_bounds and tok.end in cluster_bounds
        assert s[tok.start : tok.end] == tok.text
        last = tok.end
    for a, b in tokens.clusters(s):
        inside = [tok for tok in toks if tok.start <= a and b <= tok.end]
        assert len(inside) == (0 if s[a].isspace() else 1)


@SETTINGS
@given(token_text)
def test_word_grid_matches_str_split(s: str) -> None:
    ws = tokens.words(s)
    assert [w.text for w in ws] == s.split()
    assert all(s[w.start : w.end] == w.text and w.index == i for i, w in enumerate(ws))


# --------------------------------------------------------------------------------------------
# encoding / xml_source robustness
# --------------------------------------------------------------------------------------------


@SETTINGS
@given(st.binary(max_size=300))
def test_encoding_check_only_raises_typed_errors(data: bytes) -> None:
    try:
        encoding.check(data)
    except ProofreaderError:
        pass


def _check_result_consistency(data: bytes, result) -> None:
    for p in result.paragraphs:
        assert "".join(a.text for a in p.atoms) == p.text
        for a in p.atoms:
            if a.kind is AtomKind.LIT:
                assert data[a.raw_start : a.raw_end].decode(result.encoding.codec) == a.text


@SETTINGS
@given(st.binary(max_size=300))
def test_scan_on_random_bytes_only_raises_typed_errors(data: bytes) -> None:
    try:
        result = xml_source.scan(data)
    except ProofreaderError:
        return
    _check_result_consistency(data, result)


_FUZZ_SEEDS = sorted(p.name for p in XML_DIR.iterdir() if p.name.startswith(("m", "ent-", "multi-", "cd-", "nest-")))
_EDIT_BYTES = list(b"<>&;/!?[]-'\" \r\nxp=#") + [0xC3, 0xA9, 0x80, 0x00]


@st.composite
def mutated_fixture(draw):
    data = bytearray((XML_DIR / draw(st.sampled_from(_FUZZ_SEEDS))).read_bytes())
    for _ in range(draw(st.integers(1, 4))):
        op = draw(st.sampled_from(("insert", "delete", "replace")))
        pos = draw(st.integers(0, len(data)))
        if op == "insert":
            data[pos:pos] = bytes([draw(st.sampled_from(_EDIT_BYTES))])
        elif data and pos < len(data):
            if op == "delete":
                del data[pos]
            else:
                data[pos] = draw(st.sampled_from(_EDIT_BYTES))
    return bytes(data)


@SETTINGS
@given(mutated_fixture())
def test_scan_on_mutated_fixtures_only_raises_typed_errors(data: bytes) -> None:
    try:
        result = xml_source.scan(data)
    except ProofreaderError:
        return
    _check_result_consistency(data, result)


# --------------------------------------------------------------------------------------------
# xml_source against an independent oracle
# --------------------------------------------------------------------------------------------

_LITERAL_CHARS = list("abcdefghijklmnopqrstuvwxyzABC 0123456789.,;:!?'\"()-/=+*#%$@~`|{}") + [
    chr(c) for c in (0x00E9, 0x00FC, 0x2014, 0x2019, 0x65E5, 0x1F600, 0x00A0, 0x0301)
]
_ENTITY_DECLS = (
    b'<!ENTITY e1 "Acme corp">'
    b'<!ENTITY e2 "l1&#10;l2">'  # expands to several callbacks at one byte index
    b'<!ENTITY e3 "x &e1; y">'  # nested entity
)
_ENTITY_TEXT = {"e1": "Acme corp", "e2": "l1\nl2", "e3": "x Acme corp y"}
_PREDEFINED = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'"}
_REF_CODEPOINTS = st.one_of(
    st.sampled_from([0x9, 0xA, 0xD, 0x26, 0x3C, 0x20AC]),
    st.integers(0x20, 0xD7FF),
    st.integers(0xE000, 0xFFFD),
    st.integers(0x10000, 0x10FFFF),
)


@st.composite
def literal(draw):
    """Literal text with LF / CRLF / CR line breaks; never ends with CR (avoids cross-piece CRLF)."""
    parts = draw(st.lists(st.one_of(st.text(_LITERAL_CHARS, min_size=1, max_size=6), st.sampled_from(["\n", "\r\n", "\rq"])), min_size=1, max_size=4))
    raw = "".join(parts)
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    return raw.encode("utf-8"), text, []


@st.composite
def reference(draw):
    kind = draw(st.sampled_from(("predef", "dec", "hex", "entity")))
    if kind == "predef":
        name = draw(st.sampled_from(sorted(_PREDEFINED)))
        return f"&{name};".encode(), _PREDEFINED[name], []
    if kind == "entity":
        name = draw(st.sampled_from(sorted(_ENTITY_TEXT)))
        return f"&{name};".encode(), _ENTITY_TEXT[name], []
    cp = draw(_REF_CODEPOINTS)
    raw = f"&#{cp};" if kind == "dec" else f"&#x{cp:X};"
    return raw.encode(), chr(cp), []  # character references are not newline-normalised


def markup_piece():
    return st.sampled_from(
        [(b"<!-- note -->", "", []), (b"<?pi data?>", "", []), (b"<br/>", "", []), (b"<!---->", "", [])]
    )


@st.composite
def cdata(draw):
    body = draw(st.text(_LITERAL_CHARS + ["<", "&", ">"], max_size=6)).replace("]]>", "")
    return b"<![CDATA[" + body.encode() + b"]]>", body, []


def content(depth: int):
    leaves = [literal(), reference(), markup_piece(), cdata()]
    if depth <= 0:
        return st.lists(st.one_of(*leaves), max_size=6)
    return st.lists(st.one_of(*leaves, inline(depth - 1), paragraph(depth - 1)), max_size=6)


@st.composite
def inline(draw, depth):
    tag = draw(st.sampled_from(["em", "b", "x:i"]))
    attr = draw(st.sampled_from(["", ' class="a&amp;b"', " n='>'"]))
    pieces = draw(content(depth))
    xml = b"".join(p[0] for p in pieces)
    text = "".join(p[1] for p in pieces)
    paras = [q for p in pieces for q in p[2]]
    return f"<{tag}{attr}>".encode() + xml + f"</{tag}>".encode(), text, paras


@st.composite
def paragraph(draw, depth):
    pieces = draw(content(depth))
    xml = b"".join(p[0] for p in pieces)
    own = "".join(p[1] for p in pieces)
    inner = [q for p in pieces for q in p[2]]
    return b"<p>" + xml + b"</p>", "", [own] + inner  # contributes no text to its parent


@st.composite
def document(draw):
    bom = draw(st.sampled_from([b"", b"\xef\xbb\xbf"]))
    decl = draw(st.sampled_from([b"", b'<?xml version="1.0" encoding="UTF-8"?>\n']))
    body = draw(st.lists(st.one_of(paragraph(2), inline(1)), min_size=1, max_size=4))
    xml = (
        bom + decl + b"<!DOCTYPE r [" + _ENTITY_DECLS + b"]>"
        + b'<r xmlns:x="urn:x">' + b"".join(p[0] for p in body) + b"</r>"
    )
    expected = [q for p in body for q in p[2]]
    return xml, expected


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(document())
def test_scan_matches_independent_oracle(doc) -> None:
    data, expected = doc
    result = xml_source.scan(data)
    assert [p.text for p in result.paragraphs] == expected
    codec = result.encoding.codec
    for p in result.paragraphs:
        # valid generated content is always mappable: no fail-closed outcome
        assert p.skip_reason in (None, SkipReason.EMPTY, SkipReason.CDATA_ONLY), p.skip_reason
        assert (p.skip_reason is SkipReason.EMPTY) == (not p.text.strip())
        for a in p.atoms:
            if a.kind is AtomKind.LIT:
                for k in range(len(a.text)):
                    c = a.char_start + k
                    start = xml_source.open_byte(p, c, codec)
                    end = xml_source.close_byte(p, c + 1, codec)
                    assert data[start:end].decode(codec) == p.text[c]
            elif a.kind in (AtomKind.REF, AtomKind.NL):
                assert xml_source.open_byte(p, a.char_start, codec) == a.raw_start
                assert xml_source.close_byte(p, a.char_end, codec) == a.raw_end

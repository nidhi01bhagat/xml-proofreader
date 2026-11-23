"""S1 stress tests: deep nesting and larger inputs (all generated, invented content).

Sizes are chosen to keep the default suite fast (seconds). A near-cap (10 MiB) run is measured
manually (S1 review notes) rather than run on every test invocation.
"""

from __future__ import annotations

from proofreader import xml_source as xs
from proofreader.models import AtomKind, SkipReason


def _nested_paragraphs(depth: int) -> bytes:
    opens = b"".join(b"<p>t%d " % i for i in range(depth))
    closes = b"".join(b" e%d</p>" % i for i in reversed(range(depth)))
    return b"<r>" + opens + closes + b"</r>"


def test_deeply_nested_paragraphs_each_own_their_text() -> None:
    depth = 1000
    r = xs.scan(_nested_paragraphs(depth))
    assert len(r.paragraphs) == depth
    for i, p in enumerate(r.paragraphs):
        if i < depth - 1:
            assert p.text == f"t{i}  e{i}"
            assert p.barriers == (len(f"t{i} "),)  # exactly one directly nested <p>
        else:
            assert p.text == f"t{i}  e{i}" and p.barriers == ()
        assert p.skip_reason is None


def test_very_deep_inline_nesting_has_no_recursion_limit() -> None:
    depth = 5000
    data = b"<p>start " + b"<em>" * depth + b"deep" + b"</em>" * depth + b" end</p>"
    p = xs.scan(data).paragraphs[0]
    assert p.text == "start deep end" and p.skip_reason is None
    deep_atom = p.atoms[1]
    assert deep_atom.text == "deep" and len(deep_atom.path) == depth + 1


def test_many_mixed_paragraphs() -> None:
    para = "Our team will recieve the parcel &amp; café <em>today</em>.\r\nNext &#8217;ok&#8217; <!--c--> done. ".encode()
    count = 20_000
    data = b"<r>" + (b"<p>" + para + b"</p>\n") * count + b"</r>"
    assert len(data) > 2 * 1024 * 1024
    r = xs.scan(data)
    assert len(r.paragraphs) == count
    assert all(p.skip_reason is None for p in r.paragraphs)
    expected = "Our team will recieve the parcel & café today.\nNext ’ok’  done. "
    assert {p.text for p in r.paragraphs} == {expected}
    # spot-check exact byte mapping in the last paragraph
    last = r.paragraphs[-1]
    k = last.text.index("caf")
    start = xs.open_byte(last, k, "utf-8")
    end = xs.close_byte(last, k + 4, "utf-8")
    assert data[start:end] == "café".encode()
    assert sum(a.kind is AtomKind.REF for a in last.atoms) == 3


def test_wide_document_with_nested_and_flat_paragraphs() -> None:
    block = b"<sec><p>a <p>b</p> c</p><p>flat</p><h>no</h></sec>"
    data = b"<r>" + block * 5000 + b"</r>"
    r = xs.scan(data)
    assert len(r.paragraphs) == 15000
    assert [p.text for p in r.paragraphs[:3]] == ["a  c", "b", "flat"]
    assert not any(p.skip_reason is SkipReason.PARA_UNMAPPABLE for p in r.paragraphs)

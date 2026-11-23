"""Single source of truth for the committed XML byte fixtures (all invented).

Run from the repository root to (re)write the files in tests/fixtures/xml/:

    python -I tests/fixtures/generate_xml_fixtures.py

tests/test_fixtures.py fails if any committed file differs from FIXTURES (e.g. after an
accidental line-ending conversion), so the files and this table cannot drift apart.
Fixtures are byte-exact test data; .gitattributes marks them -text.
"""

from __future__ import annotations

from pathlib import Path

BOM = b"\xef\xbb\xbf"
DECL_UTF8 = b'<?xml version="1.0" encoding="UTF-8"?>'
DECL_ASCII = b'<?xml version="1.0" encoding="US-ASCII"?>'

FIXTURES: dict[str, bytes] = {
    # --- line endings (ENC-01..04) ---
    "enc-01-crlf.xml": b"<p>teh\r\ncat</p>",
    "enc-02-crlf-gap.xml": b"<p>ice\r\ncream cone</p>",
    "enc-03-lone-cr.xml": b"<p>a\rb teh</p>",
    "enc-04-mixed-eol.xml": b"<p>one\r\ntwo\nthree\rteh</p>",
    # --- BOM / declarations (ENC-05, ENC-06, ENC-16) ---
    "enc-05-utf8-bom.xml": BOM + b"<p>teh</p>",
    "enc-06a-decl-UTF-8.xml": DECL_UTF8 + b"<p>teh</p>",
    "enc-06b-decl-utf-8.xml": b'<?xml version="1.0" encoding="utf-8"?><p>teh</p>',
    "enc-06c-decl-utf8.xml": b"<?xml version='1.0' encoding='utf8'?><p>teh</p>",
    "enc-06d-decl-no-encoding.xml": b'<?xml version="1.0"?><p>teh</p>',
    "enc-16-no-declaration.xml": b"<p>teh</p>",
    # --- US-ASCII (ENC-07, ENC-08) ---
    "enc-07-us-ascii.xml": DECL_ASCII + b"<p>dont go</p>",
    "enc-08-ascii-high-byte.xml": DECL_ASCII + b"<p>caf\xc3\xa9</p>",
    # --- unsupported encodings (ENC-09..12) ---
    "enc-09a-utf16le-bom.xml": "\ufeff<p>teh</p>".encode("utf-16-le"),
    "enc-09b-utf16be-bom.xml": "\ufeff<p>teh</p>".encode("utf-16-be"),
    "enc-09c-utf16le-no-bom.xml": "<p>teh</p>".encode("utf-16-le"),
    "enc-09d-utf16be-no-bom.xml": "<p>teh</p>".encode("utf-16-be"),
    "enc-10a-utf32le-bom.xml": "\ufeff<p>teh</p>".encode("utf-32-le"),
    "enc-10b-utf32be-bom.xml": "\ufeff<p>teh</p>".encode("utf-32-be"),
    "enc-11a-decl-iso-8859-1.xml": b'<?xml version="1.0" encoding="ISO-8859-1"?><p>teh</p>',
    "enc-11b-decl-windows-1252.xml": b'<?xml version="1.0" encoding="windows-1252"?><p>teh</p>',
    "enc-12-bom-decl-utf16.xml": BOM + b'<?xml version="1.0" encoding="UTF-16"?><p>teh</p>',
    # --- invalid UTF-8 (ENC-13) ---
    "enc-13-invalid-utf8.xml": DECL_UTF8 + b"<p>caf\xe9 teh</p>",
    # --- multibyte offsets (ENC-14) ---
    "enc-14-multibyte.xml": "<p>café — 😀 teh</p>".encode("utf-8"),
    # --- prolog (ENC-15) ---
    "enc-15-prolog.xml": (
        DECL_UTF8
        + b"\n<?app-config mode=\"x\"?>\n<!-- prolog comment -->\n"
        + b'<!DOCTYPE p [<!ENTITY co "Acme corp">]>\n'
        + b"<p>teh</p>"
    ),
    # --- XML version (C4) ---
    "ver-xml-1.1.xml": b'<?xml version="1.1" encoding="UTF-8"?><p>teh</p>',
    # ============================ xml_source (S1 step 3) ============================
    # --- worked examples M1-M13 (docs/02 section 5) ---
    "m01-plain.xml": b"<p>We recieved teh parcel.</p>",
    "m02-em-hyphen.xml": b"<p>We saw a <em>bright</em>-red kite.</p>",
    "m03-amp.xml": b"<p>Salt &amp; peper are on teh table.</p>",
    "m04-quote-refs.xml": b"<p>She wrote &#8217;back soon&#8217; on teh door.</p>",
    "m05-cdata.xml": b"<p><![CDATA[Teh quick fox]]></p>",
    "m06-em-compound.xml": b"<p>Ask the <em>night-shift</em> porter.</p>",
    "m07-end-bold.xml": b"<p>my pet <b>dgo</b></p>",
    "m08-two-level.xml": b"<p>a <b><i>teh</i></b> cat</p>",
    "m09-comment-pi.xml": b"<p>very<!--c--> <?pi x?>good</p>",
    "m10-crlf-gap.xml": b"<p>ice\r\ncream cone</p>",
    "m11-br.xml": b"<p>ice<br/>cream</p>",
    "m12-custom-entity.xml": b'<!DOCTYPE p [<!ENTITY co "Acme corp">]><p>See &co; now</p>',
    "m13-default-ns.xml": b'<r xmlns="urn:x"><p>teh</p></r>',
    # --- references and entities ---
    "ent-01-predefined.xml": b"<p>&amp; &lt; &gt; &quot; &apos; teh</p>",
    "ent-04-char-refs.xml": b"<p>The cat&#8217;s bowl and the dog&#x2019;s bed.</p>",
    "ent-07-markup-entity.xml": b'<!DOCTYPE p [<!ENTITY m "<b>x</b>">]><p>teh &m; end</p>',
    "ent-08-external-dtd-skipped.xml": (
        b'<!DOCTYPE r SYSTEM "never-fetched.dtd"><r><p>teh &u; end</p><p>clean text</p></r>'
    ),
    "ent-09-external-general.xml": (
        b'<!DOCTYPE r [<!ENTITY x SYSTEM "never-fetched.txt">]><r><p>teh &x; end</p><p>clean text</p></r>'
    ),
    "ent-10-astral-and-cr-refs.xml": b"<p>a&#128512;b&#13;c&#xD;&#xA;d&#9;e</p>",
    "ent-11-markup-then-text-entity.xml": b'<!DOCTYPE p [<!ENTITY m "<b/>x">]><p>teh &m; end</p>',
    # --- several Expat callbacks at one CurrentByteIndex (R2) ---
    "multi-01-lf-in-entity.xml": b'<!DOCTYPE p [<!ENTITY e "l1&#10;l2">]><p>[&e;]</p>',
    "multi-02-nested-entity.xml": b'<!DOCTYPE p [<!ENTITY b "z"><!ENTITY a "x &b; y">]><p>[&a;]</p>',
    "multi-03-escaped-lt-entity.xml": b'<!DOCTYPE p [<!ENTITY e "a &#38;lt;b&#62; c">]><p>[&e;]</p>',
    # --- comments, PIs, CDATA ---
    "cmt-02-text-in-comment.xml": b"<p>cat <!--teh--> dog</p>",
    "pi-01.xml": b"<p>a <?pi x?>teh</p>",
    "cd-03-after-cdata.xml": b"<p><![CDATA[x]]> teh</p>",
    "cd-04-cdata-inside.xml": b"<p>a <![CDATA[teh]]> b</p>",
    "cd-05-cdata-outside-p.xml": b"<r><![CDATA[teh]]><p>teh</p></r>",
    "cd-06-cdata-crlf-markup.xml": b"<p>a<![CDATA[x&amp;<y>\r\nz]]>b</p>",
    "cd-07-cdata-only-with-whitespace.xml": b"<p> <![CDATA[Teh fox]]>\r\n<![CDATA[jumps]]> </p>",
    "cd-08-whitespace-only-cdata.xml": b"<p><![CDATA[ \t ]]></p>",
    "cd-09-cdata-only-with-error.xml": b'<p><![CDATA[teh]]><error type="spelling" correction="x"/></p>',
    "cd-10-cdata-plus-reference.xml": b"<p><![CDATA[teh]]>&amp;</p>",
    # --- nesting, namespaces, order, existing errors, empty paragraphs ---
    "nest-01.xml": b"<doc><p>Outer teh <p>Inner teh</p> more teh</p></doc>",
    "nest-03-inside-inline.xml": b"<p>a <em>x <p>teh</p> y</em> b</p>",
    "nest-04-deep.xml": b"<r><p>A <p>B <p>C</p> b</p> a</p></r>",
    "ns-02-prefixed.xml": b'<r xmlns:x="urn:x"><x:p x:id="1">teh</x:p></r>',
    "ns-03-unbound-prefix.xml": b"<x:p>teh</x:p>",
    "order-01-mixed-ns.xml": b'<r xmlns:a="urn:a"><p>one</p><a:p>two</a:p><s><p>three</p></s><h>no</h></r>',
    "err-01-existing-error.xml": b'<p><error type="spelling" correction="the">teh</error> cat teh</p>',
    "err-02-error-in-nested.xml": b"<r><p>a <p>b <error>c</error></p> d</p><p>clean</p></r>",
    "empty-01.xml": b"<r><p/><p></p><p> \r\n\t</p><p>x</p></r>",
    # --- malformed XML ---
    "bad-01-mismatched.xml": b"<r>\n<p>ok</p>\n<p>bad</q>\n</r>",
    "bad-02-two-roots.xml": b"<p>one</p><p>two</p>",
    "bad-03-undeclared-entity.xml": b"<p>teh &nope; end</p>",
    # --- amplification ("billion laughs") ---
    "sec-01-billion-laughs.xml": (
        b'<!DOCTYPE p [<!ENTITY a "lol">'
        + b"".join(b'<!ENTITY %c "%s">' % (98 + i, (b"&%c;" % (97 + i)) * 10) for i in range(9))
        + b"]><p>&j;</p>"
    ),
    # --- declaration / encoding interaction ---
    "decl-01-bom-decl-crlf.xml": BOM + DECL_UTF8 + b"\r\n<!--c-->\r\n<p>t\xc3\xa9h &amp; ok</p>",
    "decl-02-ascii-refs.xml": DECL_ASCII + b"<p>caf&#233; &amp; teh</p>",
    "decl-03-utf8-alias.xml": b"<?xml version='1.0' encoding='utf8'?><p>na\xc3\xafve teh</p>",
    # ============================ S2 golden triple (AC-03 a) ============================
    # Input and expected output are both written by hand; the expected output is NOT produced by the
    # renderer. The annotations live in tests/test_golden.py.
    "golden-a-input.xml": (
        DECL_UTF8 + b"\r\n<!-- invented golden document -->\r\n"
        b'<book xmlns="urn:example:book" xmlns:m="urn:example:meta" m:id="g1">\r\n'
        b"  <p>Our team will recieve teh final report on monday.</p>\r\n"
        b'  <p m:lang="en">Salt &amp; peper are on the <em>tabel</em>; <![CDATA[keep <this>]]> intact.</p>\r\n'
        b"  <section><p>Outer paragraph <p>Inner paragrph here</p> continues hear.</p></section>\r\n"
        b"  <p>Clean paragraph with nothing to fix.</p>\r\n"
        b'  <p><error type="spelling" correction="x">old</error> existing error kept.</p>\r\n'
        b"</book>\r\n"
    ),
    "golden-a-expected.xml": (
        DECL_UTF8 + b"\r\n<!-- invented golden document -->\r\n"
        b'<book xmlns="urn:example:book" xmlns:m="urn:example:meta" m:id="g1">\r\n'
        b'  <p>Our team will <error type="spelling" correction="receive">recieve</error> '
        b'<error type="spelling" correction="the">teh</error> final report on '
        b'<error type="capitalization" correction="Monday" reason="Days of the week are capitalized">monday</error>.</p>\r\n'
        b'  <p m:lang="en"><error type="spelling" correction="Salt &amp; pepper" reason="Spelling of &quot;pepper&quot;">'
        b"Salt &amp; peper</error> are on the <em><error type=\"spelling\" correction=\"table\">tabel</error></em>; "
        b"<![CDATA[keep <this>]]> intact.</p>\r\n"
        b'  <section><p>Outer paragraph <p>Inner <error type="spelling" correction="paragraph">paragrph</error> here</p> '
        b'continues <error type="grammar" correction="here">hear</error>.</p></section>\r\n'
        b"  <p>Clean paragraph with nothing to fix.</p>\r\n"
        b'  <p><error type="spelling" correction="x">old</error> existing error kept.</p>\r\n'
        b"</book>\r\n"
    ),
    # ============================ S4.8a end-to-end golden documents ============================
    # Inputs AND expected outputs are written by hand (never produced by the renderer under test). The model
    # findings that lead from one to the other are hand-written in tests/test_e2e_golden.py. Paragraph ids
    # (document order of <p> / <x:p>): 0 office, 1 tea, 2 cafe (prefixed), 3 sales, 4 outer, 5 inner (nested),
    # 6 clean, 7 existing <error> (skipped), 8 CDATA + text, 9 whitespace only (skipped).
    "golden-e2e-1-input.xml": (
        BOM + DECL_UTF8 + b"\r\n<!-- invented end-to-end golden document (S4.8a) -->\r\n"
        b'<?build stage="e2e"?>\r\n'
        b'<report xmlns="urn:example:report" xmlns:x="urn:example:extra" x:version="2">\r\n'
        b"  <title>Quarterly notes</title>\r\n"
        b"  <p>Our office will recieve the new chairs on tuesday morning.</p>\r\n"
        b'  <p x:tone="calm">Tea &amp; cofee are served in the <em>lobbby</em> daily.</p>\r\n'
        b"  <x:p>The caf\xc3\xa9 opened earlier then usual this week.</x:p>\r\n"
        b"  <p>Sales grew by 12% in the spring.</p>\r\n"
        b"  <section><p>Outer text <p>Inner text has a extra word</p> ends here.</p></section>\r\n"
        b"  <p>Everything here is correct.</p>\r\n"
        b'  <p><error type="spelling" correction="x">old</error> stays as it is.</p>\r\n'
        b"  <p>Data <![CDATA[<raw> & stuff]]> follows the the rules.</p>\r\n"
        b"  <p>   </p>\r\n"
        b"</report>\r\n"
    ),
    "golden-e2e-1-expected.xml": (
        BOM + DECL_UTF8 + b"\r\n<!-- invented end-to-end golden document (S4.8a) -->\r\n"
        b'<?build stage="e2e"?>\r\n'
        b'<report xmlns="urn:example:report" xmlns:x="urn:example:extra" x:version="2">\r\n'
        b"  <title>Quarterly notes</title>\r\n"
        b'  <p>Our office will <error type="spelling" correction="receive" reason="Misspelling">recieve</error> '
        b'the new chairs on <error type="capitalization" correction="Tuesday" '
        b'reason="Days of the week are capitalized">tuesday</error> morning.</p>\r\n'
        b'  <p x:tone="calm">Tea &amp; <error type="spelling" correction="coffee" reason="Misspelling">cofee</error> '
        b'are served in the <em><error type="spelling" correction="lobby" reason="Misspelling">lobbby</error></em> '
        b"daily.</p>\r\n"
        b'  <x:p>The caf\xc3\xa9 opened earlier <error type="grammar" correction="than" '
        b'reason="Use &quot;than&quot; for comparisons">then</error> usual this week.</x:p>\r\n'
        b'  <p>Sales grew by <error type="styleguide" correction="12 per cent" '
        b'reason="House style: write per cent">12%</error> in the spring.</p>\r\n'
        b'  <section><p>Outer text <p>Inner text has <error type="grammar" correction="an extra" '
        b'reason="Use an before a vowel sound">a extra</error> word</p> ends here.</p></section>\r\n'
        b"  <p>Everything here is correct.</p>\r\n"
        b'  <p><error type="spelling" correction="x">old</error> stays as it is.</p>\r\n'
        b'  <p>Data <![CDATA[<raw> & stuff]]> follows <error type="grammar" correction="the" '
        b'reason="Repeated word">the the</error> rules.</p>\r\n'
        b"  <p>   </p>\r\n"
        b"</report>\r\n"
    ),
    # US-ASCII document: non-ASCII corrections must become decimal character references (hand-written).
    "golden-e2e-2-input.xml": (
        DECL_ASCII + b"\n<doc>\n  <p>The naive cafe served creme brulee today.</p>\n</doc>\n"
    ),
    "golden-e2e-2-expected.xml": (
        DECL_ASCII + b"\n<doc>\n"
        b'  <p>The <error type="spelling" correction="na&#239;ve" reason="Diaeresis">naive</error> cafe served '
        b'<error type="spelling" correction="cr&#232;me br&#251;l&#233;e" reason="French spelling">creme brulee</error>'
        b" today.</p>\n</doc>\n"
    ),
}

XML_DIR = Path(__file__).resolve().parent / "xml"


def main() -> None:
    XML_DIR.mkdir(exist_ok=True)
    for name, data in FIXTURES.items():
        (XML_DIR / name).write_bytes(data)
    print(f"wrote {len(FIXTURES)} fixtures to {XML_DIR}")


if __name__ == "__main__":
    main()

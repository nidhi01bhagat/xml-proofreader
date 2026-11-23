"""S4.2a: DOCX style-guide extraction, sections, exact fingerprints, rendering, manifest and scopes.

Every document is generated at test time from invented text (tests/docx_factory.py). The marker
word QZXWV appears in invented guide text so tests can check that errors never quote the guide.
"""

from __future__ import annotations

import io
import json
import struct
import unicodedata
import warnings
import zipfile

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import docx_factory as f
from proofreader import style_guide as sg
from proofreader.config import LocaleScope

MARK = "QZXWV"
EMPTY = sg.parse_manifest(json.dumps({"version": 1, "fingerprint": "proofreader-section-v1",
                                      "scopes": {"en-GB": [], "cross-locale": []}}))


def guide(body: str, **kwargs) -> sg.StyleGuide:
    return sg.parse_docx(f.make_docx(body, **kwargs))


def texts(g: sg.StyleGuide) -> list[str]:
    return [p.text for s in g.sections for p in s.paragraphs]


def rejected(data: bytes, needle: str | None = None) -> sg.StyleGuideError:
    with pytest.raises(sg.StyleGuideError) as info:
        sg.parse_docx(data)
    assert info.value.exit_code == 1
    assert MARK not in str(info.value)
    if needle:
        assert needle in str(info.value)
    return info.value


def manifest(en_gb=(), cross=()) -> sg.Manifest:
    def entries(hashes):
        return [{"sha256": h, "label": f"Area {i}"} for i, h in enumerate(hashes)]
    return sg.parse_manifest(json.dumps({"version": 1, "fingerprint": "proofreader-section-v1",
                                         "scopes": {"en-GB": entries(en_gb), "cross-locale": entries(cross)}}))


# --------------------------------------------------------------------------------------------
# Structure, sections and paths
# --------------------------------------------------------------------------------------------


def test_preamble_headings_and_paths() -> None:
    g = guide(f.para("Preamble text.") + f.heading("Alpha", 0) + f.para("A body.") + f.heading("Beta", 1)
              + f.para("B body.") + f.heading("Gamma", 2) + f.para("C body.") + f.heading("Delta", 1)
              + f.heading("Omega", 0) + f.para("O body."))
    shape = [(s.level, s.path, s.heading, [p.text for p in s.paragraphs]) for s in g.sections]
    assert shape == [
        (None, (), None, ["Preamble text."]),
        (0, (), "Alpha", ["Alpha", "A body."]),
        (1, ((0, "Alpha"),), "Beta", ["Beta", "B body."]),
        (2, ((0, "Alpha"), (1, "Beta")), "Gamma", ["Gamma", "C body."]),
        (1, ((0, "Alpha"),), "Delta", ["Delta"]),
        (0, (), "Omega", ["Omega", "O body."]),
    ]
    assert [s.index for s in g.sections] == list(range(6))
    assert g.total_chars == sum(len(t) for t in texts(g))


def test_no_preamble_section_when_the_guide_starts_with_a_heading() -> None:
    g = guide(f.heading("Alpha", 0) + f.para("Body."))
    assert [s.level for s in g.sections] == [0]


def test_skipped_levels_keep_the_right_ancestors() -> None:
    g = guide(f.heading("Top", 0) + f.heading("Deep", 2) + f.heading("Mid", 1) + f.heading("Deeper", 2))
    assert [s.path for s in g.sections] == [(), ((0, "Top"),), ((0, "Top"),), ((0, "Top"), (1, "Mid"))]


def test_empty_and_whitespace_paragraphs_are_dropped() -> None:
    g = guide(f.para("One.") + f.p() + f.para("   ") + f.para("\t") + f.p(f.t(" "), f.t("  ")) + f.para("Two."))
    assert texts(g) == ["One.", "Two."]


def test_a_guide_without_text_is_rejected() -> None:
    rejected(f.make_docx(f.p() + f.para("  ")), "no text")


# --------------------------------------------------------------------------------------------
# Headings, lists and styles
# --------------------------------------------------------------------------------------------


def test_heading_level_from_style_chain_and_direct_property() -> None:
    styles = f.styles_xml(f.DEFAULT_STYLES + f.style("Derived", based_on="Head2") + f.style("DerivedBody", based_on="Derived", outline=9))
    g = guide(f.para("Chain", style_id="Derived") + f.para("Direct", outline=3) + f.para("Override", style_id="Head1", outline=4)
              + f.para("Body level", style_id="DerivedBody") + f.para("Nine", outline=9) + f.para("Plain"),
              styles=styles)
    kinds = [(p.kind, p.level, p.text) for p in g.sections[0].paragraphs] + \
            [(p.kind, p.level, p.text) for s in g.sections[1:] for p in s.paragraphs]
    assert ("heading", 1, "Chain") in kinds and ("heading", 3, "Direct") in kinds
    assert ("heading", 4, "Override") in kinds
    assert ("paragraph", None, "Body level") in kinds and ("paragraph", None, "Nine") in kinds
    assert ("paragraph", None, "Plain") in kinds


@pytest.mark.parametrize("level", range(9))
def test_outline_levels_0_to_8_are_headings(level: int) -> None:
    g = guide(f.para("Head", outline=level) + f.para("Body"))
    assert g.sections[0].level == level and g.sections[0].paragraphs[0].kind == "heading"


@pytest.mark.parametrize("value", ["x", "-1", "", "1.5"])
def test_invalid_outline_value_is_rejected(value: str) -> None:
    rejected(f.make_docx(f.p(f.t("Text"), ppr_extra=f'<w:outlineLvl w:val="{value}"/>')), "outline level")


def test_outline_level_above_9_is_body_text() -> None:
    assert guide(f.para("Text", outline=12)).sections[0].paragraphs[0].kind == "paragraph"


def test_lists_from_direct_numbering_or_style_and_num_zero_override() -> None:
    g = guide(f.para("Direct item", num_id="5") + f.para("Style item", style_id="ListItem")
              + f.para("Not a list", style_id="ListItem", num_id="0") + f.para("Zero", num_id="0"))
    assert [(p.kind, p.text) for p in g.sections[0].paragraphs] == [
        ("list", "Direct item"), ("list", "Style item"), ("paragraph", "Not a list"), ("paragraph", "Zero")]


def test_style_with_numbering_switched_off_is_not_a_list() -> None:
    styles = f.styles_xml(f.DEFAULT_STYLES + f.style("NoNumbers", num_id="0", based_on="ListItem"))
    g = guide(f.para("Plain again", style_id="NoNumbers") + f.para("Inherited item", style_id="ListItem"), styles=styles)
    assert [p.kind for p in g.sections[0].paragraphs] == ["paragraph", "list"]


def test_numbered_heading_stays_a_heading() -> None:
    g = guide(f.para("Numbered head", style_id="Head1", num_id="3"))
    assert g.sections[0].paragraphs[0].kind == "heading"


def test_default_paragraph_style_applies_without_pstyle() -> None:
    styles = f.styles_xml(f.style("Normal", default=True, outline=1))
    assert guide(f.para("Default heading"), styles=styles).sections[0].level == 1


def test_missing_styles_part_still_supports_direct_outline() -> None:
    g = guide(f.para("Head", outline=0) + f.para("Styled but unknown", style_id="Head1"), with_styles=False)
    assert [(p.kind, p.text) for p in g.sections[0].paragraphs] == [("heading", "Head"), ("paragraph", "Styled but unknown")]


def test_document_without_styles_relationship_parses() -> None:
    g = guide(f.para("Only body"), document_rels=None)
    assert texts(g) == ["Only body"]


def test_style_cycle_is_rejected() -> None:
    styles = f.styles_xml(f.style("A", based_on="B") + f.style("B", based_on="A"))
    rejected(f.make_docx(f.para("x", style_id="A"), styles=styles), "basedOn")


def test_styles_part_with_wrong_root_is_rejected() -> None:
    rejected(f.make_docx(f.para("x"), styles=f'<w:other xmlns:w="{f.W_NS}"/>'), "styles part")


# --------------------------------------------------------------------------------------------
# Run content (verbatim)
# --------------------------------------------------------------------------------------------


def test_run_splitting_does_not_matter() -> None:
    one = guide(f.heading("Head", 0) + f.para("Exact wording here."))
    split = guide(f.p(f.t("He"), f.t("ad"), style_id="Head1") + f.p(f.t("Exact "), f.t("word"), f.t("ing here.")))
    assert [s.sha256 for s in one.sections] == [s.sha256 for s in split.sections]


def test_edge_spaces_and_inner_spacing_are_preserved_exactly() -> None:
    assert texts(guide(f.p(f.t("  two  spaces "), f.t(" end"))))[0] == "  two  spaces  end"


def test_special_run_elements() -> None:
    body = f.p(
        f.t("a"), "<w:r><w:tab/></w:r>", f.t("b"), "<w:r><w:ptab/></w:r>", f.t("c"), "<w:r><w:br/></w:r>",
        f.t("d"), '<w:r><w:br w:type="textWrapping"/></w:r>', f.t("e"), "<w:r><w:cr/></w:r>", f.t("f"),
        '<w:r><w:br w:type="page"/></w:r>', f.t("g"), '<w:r><w:br w:type="column"/></w:r>', f.t("h"),
        "<w:r><w:noBreakHyphen/></w:r>", f.t("i"), "<w:r><w:softHyphen/></w:r>", f.t("j"),
        '<w:r><w:sym w:font="X" w:char="2192"/></w:r>', f.t("k"),
    )
    assert texts(guide(body)) == ["a\tb\tc\nd\ne\nfgh\u2011i\u00adj\u2192k"]


def test_multiple_text_elements_in_one_run() -> None:
    body = '<w:p><w:r><w:t>one</w:t><w:t xml:space="preserve"> two</w:t><w:tab/><w:t>three</w:t></w:r></w:p>'
    assert texts(guide(body)) == ["one two\tthree"]


def test_xml_character_references_are_decoded() -> None:
    assert texts(guide("<w:p><w:r><w:t>caf&#233; &amp; &#x2014; &lt;x&gt;</w:t></w:r></w:p>")) == ["café & \u2014 <x>"]


@pytest.mark.parametrize("char", ["F0E0", "E000", "F8FF", "zz", "", "110000", "0007", "D800"])
def test_unsupported_symbols_are_rejected(char: str) -> None:
    rejected(f.make_docx(f.p(f.t(MARK), f'<w:r><w:sym w:font="Wingdings" w:char="{char}"/></w:r>')), "symbol")


def test_inline_containers_contribute_their_text() -> None:
    body = f.p(
        f.t("A "), f'<w:hyperlink w:anchor="x">{f.t("link")}</w:hyperlink>', f.t(" "),
        f'<w:smartTag w:uri="u" w:element="e">{f.t("tag")}</w:smartTag>', f.t(" "),
        f'<w:customXml w:element="c"><w:customXmlPr/>{f.t("custom")}</w:customXml>', f.t(" "),
        f'<w:sdt><w:sdtPr><w:alias w:val="z"/></w:sdtPr><w:sdtContent>{f.t("control")}</w:sdtContent></w:sdt>', f.t(" "),
        f'<w:fldSimple w:instr="HYPERLINK x">{f.t("simple")}</w:fldSimple>',
    )
    assert texts(guide(body)) == ["A link tag custom control simple"]


def test_block_level_content_controls_and_custom_xml() -> None:
    body = (f'<w:sdt><w:sdtPr/><w:sdtContent>{f.para("In control")}</w:sdtContent></w:sdt>'
            f'<w:customXml w:element="c">{f.para("In custom")}</w:customXml>')
    assert texts(guide(body)) == ["In control", "In custom"]


def test_complex_fields_show_only_the_result() -> None:
    body = f.p(
        f.t("See "), '<w:r><w:fldChar w:fldCharType="begin"/></w:r>', f'<w:r><w:instrText>REF {MARK} \\h</w:instrText></w:r>',
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>', f.t("result"),
        '<w:r><w:fldChar w:fldCharType="end"/></w:r>', f.t(" now"),
    )
    assert texts(guide(body)) == ["See result now"]


def test_nested_fields_and_fields_across_paragraphs() -> None:
    nested = f.p(
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>', '<w:r><w:instrText>IF </w:instrText></w:r>',
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>', f'<w:r><w:instrText>{MARK}</w:instrText></w:r>',
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>', f.t(MARK), '<w:r><w:fldChar w:fldCharType="end"/></w:r>',
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>', f.t("outer"), '<w:r><w:fldChar w:fldCharType="end"/></w:r>',
    )
    across = (f.p(f.t("x "), '<w:r><w:fldChar w:fldCharType="begin"/></w:r>', '<w:r><w:instrText>TOC</w:instrText></w:r>',
                  '<w:r><w:fldChar w:fldCharType="separate"/></w:r>', f.t("first"))
              + f.p(f.t("second"), '<w:r><w:fldChar w:fldCharType="end"/></w:r>', f.t(" y")))
    assert texts(guide(nested)) == ["outer"]
    assert texts(guide(across)) == ["x first", "second y"]


def test_field_char_in_the_same_run_as_text() -> None:
    body = ('<w:p><w:r><w:t>a</w:t><w:fldChar w:fldCharType="begin"/><w:instrText>X</w:instrText>'
            f'<w:t>{MARK}</w:t><w:fldChar w:fldCharType="separate"/><w:t>b</w:t><w:fldChar w:fldCharType="end"/>'
            '<w:t>c</w:t></w:r></w:p>')
    assert texts(guide(body)) == ["abc"]


@pytest.mark.parametrize(
    "runs",
    [
        ['<w:r><w:fldChar w:fldCharType="separate"/></w:r>'],
        ['<w:r><w:fldChar w:fldCharType="end"/></w:r>'],
        ['<w:r><w:fldChar w:fldCharType="begin"/></w:r>'],
        ['<w:r><w:fldChar w:fldCharType="odd"/></w:r>'],
        ['<w:r><w:fldChar w:fldCharType="begin"/></w:r>', '<w:r><w:fldChar w:fldCharType="separate"/></w:r>',
         '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'],
        ['<w:r><w:fldChar w:fldCharType="begin"/></w:r>', '<w:r><w:fldChar w:fldCharType="separate"/></w:r>',
         '<w:r><w:fldChar w:fldCharType="separate"/></w:r>', '<w:r><w:fldChar w:fldCharType="end"/></w:r>'],
        ['<w:r><w:fldChar/></w:r>'],
    ],
)
def test_malformed_fields_are_rejected(runs) -> None:
    rejected(f.make_docx(f.p(f.t(MARK), *runs)), "field")


# --------------------------------------------------------------------------------------------
# Supported markers vs fail-closed constructs
# --------------------------------------------------------------------------------------------


def test_proofing_marks_bookmarks_and_references_are_supported_and_ignored() -> None:
    body = f.p(
        '<w:proofErr w:type="spellStart"/>', f.t("Speling"), '<w:proofErr w:type="spellEnd"/>', f.t(" "),
        '<w:proofErr w:type="gramStart"/>', f.t("text"), '<w:proofErr w:type="gramEnd"/>',
        '<w:bookmarkStart w:id="0" w:name="b"/>', '<w:bookmarkEnd w:id="0"/>',
        '<w:commentRangeStart w:id="1"/>', '<w:commentRangeEnd w:id="1"/>',
        '<w:r><w:commentReference w:id="1"/></w:r>', '<w:r><w:footnoteReference w:id="2"/></w:r>',
        '<w:r><w:lastRenderedPageBreak/></w:r>', '<w:permStart w:id="3"/>', '<w:permEnd w:id="3"/>',
    )
    assert texts(guide(body)) == ["Speling text"]


@pytest.mark.parametrize(
    "snippet",
    [
        f'<w:ins w:id="1" w:author="a">{f.t(MARK)}</w:ins>',
        f'<w:del w:id="1" w:author="a"><w:r><w:delText>{MARK}</w:delText></w:r></w:del>',
        f'<w:moveFrom w:id="1" w:author="a">{f.t(MARK)}</w:moveFrom>',
        f'<w:moveTo w:id="1" w:author="a">{f.t(MARK)}</w:moveTo>',
        '<w:moveFromRangeStart w:id="1" w:name="m" w:author="a"/>',
        f'<w:r><w:rPr><w:b/><w:rPrChange w:id="1" w:author="a"><w:rPr/></w:rPrChange></w:rPr>{f.t(MARK)[5:-6]}</w:r>',
        '<w:customXmlInsRangeStart w:id="1" w:author="a"/>',
    ],
)
def test_tracked_changes_in_runs_are_rejected(snippet: str) -> None:
    rejected(f.make_docx(f.p(f.t("x"), snippet)), "tracked changes")


def test_tracked_paragraph_properties_and_marks_are_rejected() -> None:
    rejected(f.make_docx(f.p(f.t(MARK), ppr_extra='<w:pPrChange w:id="1" w:author="a"><w:pPr/></w:pPrChange>')),
             "tracked changes")
    rejected(f.make_docx(f.p(f.t(MARK), ppr_extra='<w:rPr><w:ins w:id="2" w:author="a"/></w:rPr>')), "tracked changes")


def test_tracked_table_cell_insertion_is_rejected() -> None:
    table = (f'<w:tbl><w:tr><w:tc><w:tcPr><w:cellIns w:id="1" w:author="a"/></w:tcPr>{f.para(MARK)}</w:tc></w:tr></w:tbl>')
    rejected(f.make_docx(table), "tracked changes")


@pytest.mark.parametrize(
    "run",
    [
        f'<w:r><w:rPr><w:vanish/></w:rPr>{f.t(MARK)[5:-6]}</w:r>',
        f'<w:r><w:rPr><w:vanish w:val="true"/></w:rPr>{f.t(MARK)[5:-6]}</w:r>',
        f'<w:r><w:rPr><w:specVanish/></w:rPr>{f.t(MARK)[5:-6]}</w:r>',
        f'<w:r><w:rPr><w:rStyle w:val="HiddenChar"/></w:rPr>{f.t(MARK)[5:-6]}</w:r>',
    ],
)
def test_hidden_text_is_rejected(run: str) -> None:
    styles = f.styles_xml(f.DEFAULT_STYLES + f.style("HiddenChar", kind="character", vanish=True))
    rejected(f.make_docx(f.p(f.t("x"), run), styles=styles), "hidden text")


def test_hidden_paragraph_style_and_doc_defaults_are_rejected() -> None:
    styles = f.styles_xml(f.DEFAULT_STYLES + f.style("HiddenPara", vanish=True))
    rejected(f.make_docx(f.para(MARK, style_id="HiddenPara"), styles=styles), "hidden text")
    defaults = '<w:docDefaults><w:rPrDefault><w:rPr><w:vanish/></w:rPr></w:rPrDefault></w:docDefaults>'
    rejected(f.make_docx(f.para(MARK), styles=f.styles_xml(doc_defaults=defaults)), "hidden")


def test_explicit_false_never_unhides_but_hidden_runs_without_text_are_fine() -> None:
    styles = f.styles_xml(f.DEFAULT_STYLES + f.style("HiddenPara", vanish=True))
    body = f.p(f'<w:r><w:rPr><w:vanish w:val="false"/></w:rPr><w:t>{MARK}</w:t></w:r>', style_id="HiddenPara")
    rejected(f.make_docx(body, styles=styles), "hidden text")  # vanish is a toggle property: fail closed
    body = f.p(f.t("text"), '<w:r><w:rPr><w:vanish/></w:rPr><w:footnoteReference w:id="1"/></w:r>')
    assert texts(guide(body)) == ["text"]
    vanish_mark = f.p(f.t("mark hidden only"), ppr_extra="<w:rPr><w:vanish/></w:rPr>")
    assert texts(guide(vanish_mark)) == ["mark hidden only"]


TEXTBOX = (f'<w:r><w:drawing><wp:inline xmlns:wp="urn:wp"><a:graphic xmlns:a="urn:a"><w:txbxContent>{f.para(MARK)}'
           "</w:txbxContent></a:graphic></wp:inline></w:drawing></w:r>")


@pytest.mark.parametrize(
    "snippet, needle",
    [
        (TEXTBOX, "text box"),
        (f'<w:r><w:pict><v:shape xmlns:v="urn:v"><w:txbxContent>{f.para(MARK)}</w:txbxContent></v:shape></w:pict></w:r>', "text box"),
        ('<w:r><mc:AlternateContent xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006">'
         f'<mc:Choice Requires="x"><w:t>{MARK}</w:t></mc:Choice></mc:AlternateContent></w:r>', "text box"),
        ('<w:r><w:object w:dxaOrig="1"/></w:r>', "embedded object"),
        ("<w:r><w:pgNum/></w:r>", "generated-text"),
        ("<w:r><w:monthLong/></w:r>", "generated-text"),
        (f'<m:oMath xmlns:m="urn:math"><m:r><m:t>{MARK}</m:t></m:r></m:oMath>', "unsupported element"),
        (f'<w:ruby><w:rubyBase>{f.t(MARK)}</w:rubyBase></w:ruby>', "unsupported element"),
    ],
)
def test_unsupported_inline_constructs_fail_closed(snippet: str, needle: str) -> None:
    rejected(f.make_docx(f.p(f.t("x"), snippet)), needle)


def test_drawings_and_unknown_elements_without_text_are_ignored() -> None:
    body = f.p(f.t("before"), '<w:r><w:drawing><wp:inline xmlns:wp="urn:wp"/></w:drawing></w:r>',
               '<w:r><w14:shadow xmlns:w14="urn:w14"/></w:r>', '<x:empty xmlns:x="urn:x"/>', f.t(" after"))
    assert texts(guide(body)) == ["before after"]


def test_unknown_element_holding_only_whitespace_text_is_rejected() -> None:
    """Dropping even a space would change the wording."""
    rejected(f.make_docx(f.p(f.t("a"), '<x:wrap xmlns:x="urn:x"><w:r><w:t xml:space="preserve"> </w:t></w:r></x:wrap>',
                             f.t("b"))), "unsupported element")


def test_block_level_unsupported_content_is_rejected() -> None:
    rejected(f.make_docx(f.para("x") + '<w:altChunk r:id="r9" xmlns:r="urn:r"/>'), "altChunk")
    rejected(f.make_docx(f.para("x") + f'<x:block xmlns:x="urn:x"><x:t>{MARK}</x:t></x:block>'), "unsupported element")


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------


def table(*rows: list[str]) -> str:
    out = "<w:tbl><w:tblPr/><w:tblGrid/>"
    for row in rows:
        out += "<w:tr><w:trPr/>" + "".join(f"<w:tc><w:tcPr/>{cell}</w:tc>" for cell in row) + "</w:tr>"
    return out + "</w:tbl>"


def test_table_rows_become_lines_with_cells_joined() -> None:
    body = f.heading("Head", 0) + table([f.para("Term"), f.para("Use")], [f.para("one") + f.para("two"), f.p()],
                                        [f.p(), f.p()])
    g = guide(body)
    assert [(p.kind, p.text) for p in g.sections[0].paragraphs[1:]] == [("table", "Term | Use"), ("table", "one\ntwo | ")]


def test_table_rows_wrapped_in_content_controls() -> None:
    body = ("<w:tbl><w:sdt><w:sdtContent><w:tr><w:tc>" + f.para("wrapped") + "</w:tc></w:tr></w:sdtContent></w:sdt></w:tbl>")
    assert texts(guide(body)) == ["wrapped"]


def test_nested_tables_are_rejected() -> None:
    rejected(f.make_docx(table([table([f.para(MARK)])])), "nested table")


def test_headings_inside_tables_are_table_text() -> None:
    g = guide(table([f.para("Looks like a heading", style_id="Head1")]))
    assert [(s.level, p.kind) for s in g.sections for p in s.paragraphs] == [(None, "table")]


# --------------------------------------------------------------------------------------------
# Parts outside the body
# --------------------------------------------------------------------------------------------


def test_headers_footers_footnotes_and_comments_are_not_part_of_the_guide() -> None:
    other = (f'<?xml version="1.0" encoding="UTF-8"?><w:hdr xmlns:w="{f.W_NS}">{f.para(MARK)}</w:hdr>')
    data = f.make_docx(f.para("Body only"), extra={"word/header1.xml": other, "word/footer1.xml": other,
                                                   "word/footnotes.xml": other, "word/comments.xml": other})
    g = sg.parse_docx(data)
    assert texts(g) == ["Body only"]


# --------------------------------------------------------------------------------------------
# Package and XML safety
# --------------------------------------------------------------------------------------------


def test_not_a_zip_truncated_and_corrupt_packages_are_rejected() -> None:
    rejected(b"not a zip at all", "zip")
    good = f.make_docx(f.para("x"))
    rejected(good[: len(good) // 2])
    stored = bytearray(f.make_docx(f.para("CRC check text"), compression=zipfile.ZIP_STORED))
    position = bytes(stored).find(b"CRC check text")
    stored[position] ^= 0x01
    rejected(bytes(stored), "corrupt")


def test_lying_zip_size_metadata_does_not_bypass_reading_checks() -> None:
    data = bytearray(f.make_docx(f.para("Some text " * 50), compression=zipfile.ZIP_STORED))
    name = b"word/document.xml"
    for signature, size_offset in ((b"PK\x01\x02", 24), (b"PK\x03\x04", 22)):
        start = 0
        while (start := bytes(data).find(signature, start)) != -1:
            name_len_offset = 28 if signature == b"PK\x01\x02" else 26
            name_len = struct.unpack_from("<H", data, start + name_len_offset)[0]
            header_len = 46 if signature == b"PK\x01\x02" else 30
            if bytes(data[start + header_len:start + header_len + name_len]) == name:
                struct.pack_into("<I", data, start + size_offset, 10)  # claim 10 bytes uncompressed
            start += 4
    rejected(bytes(data))


def test_encrypted_entries_are_rejected() -> None:
    data = bytearray(f.make_docx(f.para("x")))
    start = 0
    while (start := bytes(data).find(b"PK\x01\x02", start)) != -1:
        data[start + 8] |= 0x01  # general-purpose flag bit 0 in the central directory
        start += 4
    rejected(bytes(data), "encrypted")


def test_duplicate_entries_are_rejected() -> None:
    rejected(f.make_docx(f.para("x"), extra={"WORD/Document.xml": "dup"}), "duplicate")
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, content in zipfile.ZipFile(io.BytesIO(f.make_docx(f.para("x")))).read.__self__.namelist() and [
                ("[Content_Types].xml", f.CONTENT_TYPES), ("_rels/.rels", f.PACKAGE_RELS),
                ("word/document.xml", f.document_xml(f.para("x"))), ("word/document.xml", f.document_xml(f.para("y")))]:
                archive.writestr(name, content)
    rejected(buffer.getvalue(), "duplicate")


@pytest.mark.parametrize(
    "prolog",
    [
        '<!DOCTYPE w:document [<!ENTITY e "QZXWV">]>',
        '<!DOCTYPE w:document SYSTEM "http://example.invalid/x.dtd">',
        "<!DOCTYPE w:document>",
    ],
)
def test_dtds_and_entities_are_rejected(prolog: str) -> None:
    document = f.document_xml(f.para("x")).replace("?>", "?>" + prolog, 1)
    rejected(f.make_docx(document=document), "DTD or entity")


def test_undefined_entities_and_malformed_xml_are_rejected() -> None:
    rejected(f.make_docx(document=f.document_xml("<w:p><w:r><w:t>&undefined;</w:t></w:r></w:p>")), "well-formed")
    rejected(f.make_docx(document=f.document_xml("<w:p><w:r><w:t>open")), "well-formed")
    rejected(f.make_docx(f.para("x"), styles="<w:styles"), "well-formed")


def test_strict_ooxml_is_rejected() -> None:
    strict_rel = f.PACKAGE_RELS.replace(f.OFFICE_DOC, "http://purl.oclc.org/ooxml/officeDocument/relationships/officeDocument")
    rejected(f.make_docx(f.para("x"), package_rels=strict_rel), "strict")
    strict_doc = f.document_xml(f.para("x")).replace(f.W_NS, "http://purl.oclc.org/ooxml/wordprocessingml/main")
    rejected(f.make_docx(document=strict_doc), "strict")


@pytest.mark.parametrize(
    "rels, needle",
    [
        (f'<Relationships xmlns="{f.REL_NS}"/>', "exactly one main document"),
        (f.PACKAGE_RELS.replace("</Relationships>", f'<Relationship Id="r2" Type="{f.OFFICE_DOC}" Target="word/b.xml"/></Relationships>'),
         "exactly one main document"),
        (f.PACKAGE_RELS.replace('Target="word/document.xml"', 'Target="https://example.invalid/d.xml" TargetMode="External"'),
         "external"),
        (f.PACKAGE_RELS.replace('Target="word/document.xml"', 'Target="../outside.xml"'), "outside the package"),
        (f.PACKAGE_RELS.replace('Target="word/document.xml"', 'Target="word/missing.xml"'), "missing the part"),
        (f.PACKAGE_RELS.replace('Target="word/document.xml"', 'Target="word\\document.xml"'), "invalid relationship"),
    ],
)
def test_bad_package_relationships_are_rejected(rels: str, needle: str) -> None:
    rejected(f.make_docx(f.para("x"), package_rels=rels), needle)


def test_absolute_relationship_target_is_resolved() -> None:
    rels = f.PACKAGE_RELS.replace('Target="word/document.xml"', 'Target="/word/document.xml"')
    assert texts(guide(f.para("absolute ok"), package_rels=rels)) == ["absolute ok"]


def test_missing_package_relationships_are_rejected() -> None:
    rejected(f.make_docx(f.para("x"), omit=("_rels/.rels",)), "exactly one main document")


def test_bad_styles_relationships_are_rejected() -> None:
    two = f.DOCUMENT_RELS.replace("</Relationships>", f'<Relationship Id="r2" Type="{f.STYLES_REL}" Target="styles.xml"/></Relationships>')
    rejected(f.make_docx(f.para("x"), document_rels=two), "more than one styles")
    external = f.DOCUMENT_RELS.replace('Target="styles.xml"', 'Target="https://example.invalid/s.xml" TargetMode="External"')
    rejected(f.make_docx(f.para("x"), document_rels=external), "external")


def test_wrong_document_root_or_missing_body_is_rejected() -> None:
    rejected(f.make_docx(document=f.document_xml(f.para("x"), root="w:other")), "not a WordprocessingML")
    rejected(f.make_docx(document=f'<w:document xmlns:w="{f.W_NS}"/>'), "no body")


# --------------------------------------------------------------------------------------------
# Limits (actual bytes; code points)
# --------------------------------------------------------------------------------------------


def test_docx_size_limit_on_bytes(monkeypatch, tmp_path) -> None:
    data = f.make_docx(f.para("x"))
    monkeypatch.setattr(sg, "MAX_DOCX_BYTES", len(data) - 1)
    rejected(data, "larger than")
    path = tmp_path / "guide.docx"
    path.write_bytes(data)
    with pytest.raises(sg.StyleGuideError, match="larger than"):
        sg.load(path)
    monkeypatch.setattr(sg, "MAX_DOCX_BYTES", len(data))
    assert texts(sg.load(path)) == ["x"]


def test_part_size_limit_counts_decompressed_bytes(monkeypatch) -> None:
    data = f.make_docx(f.para("a" * 5000))  # highly compressible: compressed far below 5000 bytes
    size = len(f.document_xml(f.para("a" * 5000)).encode())
    monkeypatch.setattr(sg, "MAX_PART_BYTES", size - 1)
    rejected(data, "document.xml")
    monkeypatch.setattr(sg, "MAX_PART_BYTES", size)
    assert texts(sg.parse_docx(data)) == ["a" * 5000]


def test_total_decompressed_limit_spans_parts(monkeypatch) -> None:
    data = f.make_docx(f.para("text"))
    parts = zipfile.ZipFile(io.BytesIO(data))
    read = sum(parts.getinfo(n).file_size for n in ("_rels/.rels", "word/document.xml", "word/_rels/document.xml.rels",
                                                    "word/styles.xml"))
    monkeypatch.setattr(sg, "MAX_TOTAL_BYTES", read - 1)
    rejected(data, "in total")
    monkeypatch.setattr(sg, "MAX_TOTAL_BYTES", read)
    assert texts(sg.parse_docx(data)) == ["text"]


def test_zip_entry_limit(monkeypatch) -> None:
    data = f.make_docx(f.para("x"), extra={f"custom/{i}.xml": "<a/>" for i in range(5)})
    count = len(zipfile.ZipFile(io.BytesIO(data)).infolist())
    monkeypatch.setattr(sg, "MAX_ZIP_ENTRIES", count - 1)
    rejected(data, "zip entries")
    monkeypatch.setattr(sg, "MAX_ZIP_ENTRIES", count)
    assert texts(sg.parse_docx(data)) == ["x"]


@pytest.mark.parametrize("unit", ["a", "\U0001F600", "e\u0301", "\u00e9"])
def test_character_limit_counts_code_points(monkeypatch, unit: str) -> None:
    text = unit * 10
    limit = len(text)  # code points, whatever the UTF-8 / UTF-16 size
    data = f.make_docx(f.para(text))
    monkeypatch.setattr(sg, "MAX_GUIDE_CHARS", limit)
    assert sg.parse_docx(data).total_chars == limit
    monkeypatch.setattr(sg, "MAX_GUIDE_CHARS", limit - 1)
    rejected(data, "characters")


def test_character_limit_counts_every_kept_paragraph(monkeypatch) -> None:
    data = f.make_docx(f.heading("Head", 0) + f.para("abc") + table([f.para("de"), f.para("f")]))
    monkeypatch.setattr(sg, "MAX_GUIDE_CHARS", 4 + 3 + len("de | f"))
    assert sg.parse_docx(data).total_chars == 13
    monkeypatch.setattr(sg, "MAX_GUIDE_CHARS", 12)
    rejected(data, "characters")


def test_section_limit(monkeypatch) -> None:
    data = f.make_docx("".join(f.heading(f"H{i}", 0) for i in range(5)))
    monkeypatch.setattr(sg, "MAX_SECTIONS", 4)
    rejected(data, "sections")
    monkeypatch.setattr(sg, "MAX_SECTIONS", 5)
    assert len(sg.parse_docx(data).sections) == 5


def test_load_errors_are_exit_1_without_wording(tmp_path) -> None:
    for target in (tmp_path / "missing.docx", tmp_path):
        with pytest.raises(sg.StyleGuideError) as info:
            sg.load(target)
        assert info.value.exit_code == 1


def test_error_classes() -> None:
    from proofreader.models import ProofreaderError

    assert issubclass(sg.StyleGuideError, ProofreaderError) and sg.StyleGuideError("x").exit_code == 1
    assert issubclass(sg.ManifestError, ProofreaderError) and sg.ManifestError("x").exit_code == 1


# --------------------------------------------------------------------------------------------
# Fingerprints (exact)
# --------------------------------------------------------------------------------------------


def test_fingerprint_definition() -> None:
    paragraphs = (sg.GuideParagraph("heading", 1, "Head"), sg.GuideParagraph("list", None, 'Item "q"\nline café —'))
    payload = json.dumps(["proofreader-section-v1", [[0, "Top"]],
                          [["heading", 1, "Head"], ["list", None, 'Item "q"\nline café —']]],
                         ensure_ascii=False, separators=(",", ":"))
    import hashlib

    assert sg.fingerprint(((0, "Top"),), paragraphs) == hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert "é".encode("utf-8") in payload.encode("utf-8")  # raw UTF-8, not \\u escapes


def section_hashes(body: str) -> list[str]:
    return [s.sha256 for s in guide(body).sections]


BASE = f.heading("Parent", 0) + f.para("Intro.") + f.heading("Child", 1) + f.para("Use the café rule.")


@pytest.mark.parametrize(
    "variant, changed",
    [
        (BASE.replace("café", unicodedata.normalize("NFD", "café")), [1]),  # NFC vs NFD
        (BASE.replace("Use the", "Use  the"), [1]),  # whitespace
        (BASE.replace("Parent", "Parent!"), [0, 1]),  # parent rename changes the child (path)
        (BASE.replace("Intro.", "Intro!"), [0]),  # sibling body: only its own section
        (BASE.replace(f.para("Use the café rule."), f.para("Use the café rule.", num_id="1")), [1]),  # kind
        (BASE.replace(f.heading("Child", 1), f.heading("Child", 2)), [1]),  # level
    ],
)
def test_any_wording_or_structure_change_changes_the_hash(variant: str, changed: list[int]) -> None:
    before, after = section_hashes(BASE), section_hashes(variant)
    assert [i for i, (a, b) in enumerate(zip(before, after)) if a != b] == changed


def test_formatting_does_not_change_the_hash() -> None:
    styled = BASE.replace(f.t("Use the café rule."),
                          '<w:r><w:rPr><w:b/><w:i/></w:rPr><w:t>Use the </w:t></w:r><w:r><w:t>café rule.</w:t></w:r>')
    assert section_hashes(styled) == section_hashes(BASE)


def test_identical_sections_have_identical_hashes_and_parsing_is_deterministic() -> None:
    body = f.heading("Same", 0) + f.para("Text") + f.heading("Same", 0) + f.para("Text")
    hashes = section_hashes(body)
    assert hashes[0] == hashes[1] and section_hashes(body) == hashes


# --------------------------------------------------------------------------------------------
# Rendering and selection
# --------------------------------------------------------------------------------------------

RENDER_BODY = (f.para("Preamble.") + f.heading("Dates", 0) + f.para("Line one\u2011a") + f.para("Item", num_id="1")
               + table([f.para("A"), f.para("B")]) + f.heading("Sub", 1) + f.p(f.t("x"), "<w:r><w:br/></w:r>", f.t("y"))
               + f.heading("Other", 0) + f.para("*not emphasis* # not heading"))


def test_full_rendering_is_verbatim_with_structural_markers_only() -> None:
    selected = sg.select(guide(RENDER_BODY), LocaleScope.FULL, EMPTY)
    assert selected.text == ("Preamble.\n\n# Dates\n\nLine one\u2011a\n\n- Item\n\nA | B\n\n## Sub\n\nx\ny\n\n"
                             "# Other\n\n*not emphasis* # not heading\n")
    assert (selected.included, selected.total, selected.scope) == (4, 4, LocaleScope.FULL)


def test_empty_manifest_selects_nothing_for_en_gb_and_other_locales() -> None:
    g = guide(RENDER_BODY)
    for scope in (LocaleScope.EN_GB, LocaleScope.CROSS_LOCALE):
        selected = sg.select(g, scope, EMPTY)
        assert (selected.text, selected.included, selected.total, selected.included_hashes) == ("", 0, 4, ())


def test_scopes_en_gb_gets_both_lists_others_get_cross_locale_only() -> None:
    g = guide(RENDER_BODY)
    pre, dates, sub, other = (s.sha256 for s in g.sections)
    m = manifest(en_gb=[dates], cross=[other])
    assert sg.select(g, LocaleScope.EN_GB, m).included_hashes == (dates, other)
    assert sg.select(g, LocaleScope.CROSS_LOCALE, m).included_hashes == (other,)
    assert sg.select(g, LocaleScope.FULL, m).included_hashes == (pre, dates, sub, other)


def test_approved_child_brings_its_heading_path_but_not_the_parent_body() -> None:
    g = guide(RENDER_BODY)
    sub = g.sections[2].sha256
    text = sg.select(g, LocaleScope.CROSS_LOCALE, manifest(cross=[sub])).text
    assert text == "# Dates\n\n## Sub\n\nx\ny\n"


def test_parent_heading_is_not_repeated_when_parent_and_child_are_included() -> None:
    g = guide(RENDER_BODY)
    text = sg.select(g, LocaleScope.CROSS_LOCALE, manifest(cross=[g.sections[1].sha256, g.sections[2].sha256])).text
    assert text.count("# Dates") == 1 and text.startswith("# Dates\n\nLine one")


def test_changed_section_is_excluded_even_if_its_old_hash_was_approved() -> None:
    old = guide(RENDER_BODY)
    approved = manifest(cross=[old.sections[1].sha256, old.sections[3].sha256])
    new = guide(RENDER_BODY.replace("Line one", "Line 1"))
    assert sg.select(new, LocaleScope.CROSS_LOCALE, approved).included_hashes == (new.sections[3].sha256,)


def test_selection_is_byte_stable() -> None:
    g = guide(RENDER_BODY)
    m = manifest(cross=[s.sha256 for s in g.sections[1:]])
    assert sg.select(g, LocaleScope.CROSS_LOCALE, m) == sg.select(guide(RENDER_BODY), LocaleScope.CROSS_LOCALE, m)


def test_unknown_scope_is_rejected() -> None:
    with pytest.raises(ValueError):
        sg.select(guide(f.para("x")), "full", EMPTY)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------------


def test_packaged_manifest_is_valid_and_empty_in_s4_2a() -> None:
    m = sg.load_manifest()
    assert m.en_gb == frozenset() and m.cross_locale == frozenset() and dict(m.labels) == {}


H1, H2 = "a" * 64, "b" * 64


def raw_manifest(**overrides):
    base = {"version": 1, "fingerprint": "proofreader-section-v1",
            "scopes": {"en-GB": [{"sha256": H1, "label": "Time formats"}], "cross-locale": [{"sha256": H2, "label": "Numbers"}]}}
    base.update(overrides)
    return base


def test_valid_manifest() -> None:
    m = sg.parse_manifest(json.dumps(raw_manifest()))
    assert m.en_gb == {H1} and m.cross_locale == {H2} and m.labels == {H1: "Time formats", H2: "Numbers"}


@pytest.mark.parametrize(
    "data",
    [
        "not json", "[]", json.dumps({**raw_manifest(), "extra": 1}), json.dumps(raw_manifest(version=2)),
        json.dumps(raw_manifest(version=True)), json.dumps(raw_manifest(version="1")),
        json.dumps(raw_manifest(fingerprint="proofreader-section-v2")),
        json.dumps(raw_manifest(scopes={"en-GB": []})), json.dumps(raw_manifest(scopes={"en-GB": [], "cross-locale": [], "fr": []})),
        json.dumps(raw_manifest(scopes={"en-GB": {}, "cross-locale": []})),
        json.dumps(raw_manifest(scopes={"en-GB": [{"sha256": H1}], "cross-locale": []})),
        json.dumps(raw_manifest(scopes={"en-GB": [{"sha256": H1, "label": "x", "note": "y"}], "cross-locale": []})),
        json.dumps(raw_manifest(scopes={"en-GB": [{"sha256": H1.upper(), "label": "x"}], "cross-locale": []})),
        json.dumps(raw_manifest(scopes={"en-GB": [{"sha256": "a" * 63, "label": "x"}], "cross-locale": []})),
        json.dumps(raw_manifest(scopes={"en-GB": [{"sha256": 5, "label": "x"}], "cross-locale": []})),
        json.dumps(raw_manifest(scopes={"en-GB": [{"sha256": H1, "label": ""}], "cross-locale": []})),
        json.dumps(raw_manifest(scopes={"en-GB": [{"sha256": H1, "label": "x" * 61}], "cross-locale": []})),
        json.dumps(raw_manifest(scopes={"en-GB": [{"sha256": H1, "label": "Caf\u00e9"}], "cross-locale": []})),
        json.dumps(raw_manifest(scopes={"en-GB": [{"sha256": H1, "label": "two\nlines"}], "cross-locale": []})),
        json.dumps(raw_manifest(scopes={"en-GB": [{"sha256": H1, "label": "a"}, {"sha256": H1, "label": "b"}], "cross-locale": []})),
        json.dumps(raw_manifest(scopes={"en-GB": [{"sha256": H1, "label": "a"}], "cross-locale": [{"sha256": H1, "label": "b"}]})),
    ],
)
def test_invalid_manifests_are_rejected(data: str) -> None:
    with pytest.raises(sg.ManifestError):
        sg.parse_manifest(data)


# --------------------------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------------------------

_WORD = st.text(alphabet="abcdefghijklmnopqrstuvwxyz\u00e9\u2019-", min_size=1, max_size=8)
_SENTENCE = st.lists(_WORD, min_size=1, max_size=5).map(" ".join)
_LAYOUT = st.lists(st.tuples(st.integers(0, 3), st.lists(_SENTENCE, min_size=0, max_size=3)), min_size=1, max_size=6)


def _build(layout, split_runs=False):
    body = ""
    expected = []
    for i, (level, sentences) in enumerate(layout):
        head = f"H{i}x"
        body += f.para(head, outline=level)
        expected.append(head)
        for j, sentence in enumerate(sentences):
            text = f"S{i}p{j} {sentence}"
            if split_runs:
                cut = max(1, len(text) // 2)
                body += f.p(f.t(text[:cut]), f.t(text[cut:]))
            else:
                body += f.para(text)
            expected.append(text)
    return body, expected


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_LAYOUT)
def test_property_round_trip_and_run_split_invariance(layout) -> None:
    body, expected = _build(layout)
    g = guide(body)
    assert texts(g) == expected
    split, _ = _build(layout, split_runs=True)
    assert [s.sha256 for s in guide(split).sections] == [s.sha256 for s in g.sections]


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_LAYOUT, st.data())
def test_property_selection_includes_exactly_the_approved_sections(layout, data) -> None:
    body, _ = _build(layout)
    g = guide(body)
    approved = [s for s in g.sections if data.draw(st.booleans())]
    cross = [s for s in approved if data.draw(st.booleans())]
    en_gb_only = [s for s in approved if s not in cross]
    m = manifest(en_gb=sorted({s.sha256 for s in en_gb_only} - {s.sha256 for s in cross}),
                 cross=sorted({s.sha256 for s in cross}))
    for scope, allowed in ((LocaleScope.EN_GB, m.en_gb | m.cross_locale), (LocaleScope.CROSS_LOCALE, m.cross_locale)):
        selected = sg.select(g, scope, m)
        for section in g.sections:
            body_texts = [p.text for p in section.paragraphs[1:]]
            present = [t in selected.text for t in body_texts]
            if section.sha256 in allowed:
                assert all(present)
                assert section.paragraphs[0].text in selected.text
            else:
                assert not any(present)
        assert selected.included == sum(1 for s in g.sections if s.sha256 in allowed)
    full = sg.select(g, LocaleScope.FULL, m)
    assert full.text == sg.render(g.sections) and full.included == len(g.sections)


# --------------------------------------------------------------------------------------------
# S4.2a review regressions: exact extraction against hand-written expected output
# --------------------------------------------------------------------------------------------

INSTR = "INSTRNOISE"  # must never appear in extracted text
BEGIN = '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
SEPARATE = '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
END = '<w:r><w:fldChar w:fldCharType="end"/></w:r>'


def instr(text: str = INSTR) -> str:
    return f'<w:r><w:instrText xml:space="preserve">{text}</w:instrText></w:r>'


def test_text_split_across_many_runs_is_extracted_exactly_once() -> None:
    runs = [f.t(piece) for piece in ("Un", "iq", "ue", "Tok", "en", " ", "s", "plit")]
    g = guide(f.p(*runs) + table([f.p(f.t("Ce"), f.t("ll"))]) +
              f'<w:sdt><w:sdtContent>{f.p(f.t("Con"), f.t("trol"))}</w:sdtContent></w:sdt>')
    assert texts(g) == ["UniqueToken split", "Cell", "Control"]
    rendered = sg.render(g.sections)
    assert rendered.count("UniqueToken") == 1 and rendered.count("Cell") == 1 and rendered.count("Control") == 1


def test_simple_field_shows_its_result_and_never_its_instruction() -> None:
    body = f.p(f.t("A "), f'<w:fldSimple w:instr=" REF {INSTR} \\h ">{f.t("Dis")}{f.t("played")}</w:fldSimple>', f.t(" B"))
    assert texts(guide(body)) == ["A Displayed B"]


def test_complex_field_with_instruction_and_result_split_across_runs() -> None:
    body = f.p(f.t("x "), BEGIN, instr(" REF "), instr(INSTR), instr(" \\h "), SEPARATE, f.t("res"), f.t("ult"),
               END, f.t(" y"))
    assert texts(guide(body)) == ["x result y"]


def test_complex_field_spanning_three_paragraphs() -> None:
    body = (f.p(f.t("one "), BEGIN, instr("TOC "))
            + f.p(instr(INSTR), SEPARATE, f.t("two"))
            + f.p(f.t("three"), END, f.t(" four")))
    assert texts(guide(body)) == ["one ", "two", "three four"]


def test_instruction_only_paragraph_is_dropped() -> None:
    body = f.p(f.t("before "), BEGIN) + f.p(instr()) + f.p(SEPARATE, f.t("after"), END)
    assert texts(guide(body)) == ["before ", "after"]


def test_nested_field_inside_a_displayed_result_is_shown() -> None:
    body = f.p(BEGIN, instr("IF "), SEPARATE, f.t("A"), BEGIN, instr(INSTR), SEPARATE, f.t("B"), END, f.t("C"), END)
    assert texts(guide(body)) == ["ABC"]


def test_nested_field_inside_an_instruction_is_hidden() -> None:
    body = f.p(f.t("<"), BEGIN, instr("IF "), BEGIN, instr(INSTR), SEPARATE, f.t(INSTR), END, instr(" = 1 "),
               SEPARATE, f.t("shown"), END, f.t(">"))
    assert texts(guide(body)) == ["<shown>"]


def test_field_boundaries_inside_one_run() -> None:
    body = (f'<w:p><w:r><w:t xml:space="preserve">pre </w:t><w:fldChar w:fldCharType="begin"/>'
            f'<w:instrText>{INSTR}</w:instrText><w:t>{INSTR}</w:t><w:fldChar w:fldCharType="separate"/>'
            '<w:t>mid</w:t><w:tab/><w:t>dle</w:t><w:fldChar w:fldCharType="end"/><w:t xml:space="preserve"> post</w:t></w:r></w:p>')
    assert texts(guide(body)) == ["pre mid\tdle post"]


def test_field_without_a_result_shows_nothing() -> None:
    body = f.p(f.t("before "), BEGIN, instr(), f.t(INSTR), END, f.t("after"))
    assert texts(guide(body)) == ["before after"]


def test_simple_field_inside_a_complex_instruction_is_hidden() -> None:
    body = f.p(f.t("a"), BEGIN, f'<w:fldSimple w:instr="X">{f.t(INSTR)}</w:fldSimple>', SEPARATE, f.t("b"), END)
    assert texts(guide(body)) == ["ab"]


def test_field_result_inside_supported_containers() -> None:
    body = f.p(BEGIN, instr("HYPERLINK x"), SEPARATE, f'<w:hyperlink w:anchor="a">{f.t("link")}</w:hyperlink>',
               f'<w:sdt><w:sdtContent>{f.t(" text")}</w:sdtContent></w:sdt>', END)
    assert texts(guide(body)) == ["link text"]


def test_hidden_field_instruction_is_fine_but_hidden_result_is_rejected() -> None:
    hidden_instr = f'<w:r><w:rPr><w:vanish/></w:rPr><w:instrText>{INSTR}</w:instrText></w:r>'
    assert texts(guide(f.p(BEGIN, hidden_instr, SEPARATE, f.t("ok"), END))) == ["ok"]
    hidden_result = f'<w:r><w:rPr><w:vanish/></w:rPr><w:t>{MARK}</w:t></w:r>'
    rejected(f.make_docx(f.p(BEGIN, instr(), SEPARATE, hidden_result, END)), "hidden text")


def test_hyphen_elements_produce_exact_code_points() -> None:
    nb, soft = texts(guide(f.p("<w:r><w:noBreakHyphen/></w:r>", f.t("x")) + f.p("<w:r><w:softHyphen/></w:r>", f.t("y"))))
    assert [ord(c) for c in nb] == [0x2011, ord("x")]
    assert [ord(c) for c in soft] == [0x00AD, ord("y")]


@pytest.mark.parametrize("para_hidden", [False, True])
@pytest.mark.parametrize("char_hidden", [False, True])
@pytest.mark.parametrize("direct", [None, "true", "false", "on", "off"])
@pytest.mark.parametrize("spec", [False, True])
def test_hidden_text_matrix(para_hidden: bool, char_hidden: bool, direct: str | None, spec: bool) -> None:
    """Independent rule: reject if ANY level marks the text hidden; an explicit false never un-hides."""
    styles = f.styles_xml(f.DEFAULT_STYLES + f.style("P", vanish=para_hidden) +
                          f.style("C", kind="character", vanish=char_hidden))
    rpr = '<w:rStyle w:val="C"/>'
    rpr += f'<w:vanish w:val="{direct}"/>' if direct is not None else ""
    rpr += "<w:specVanish/>" if spec else ""
    body = f.p(f.t("visible "), f"<w:r><w:rPr>{rpr}</w:rPr><w:t>{MARK}</w:t></w:r>", style_id="P")
    expect_reject = para_hidden or char_hidden or direct in ("true", "on") or spec
    if expect_reject:
        rejected(f.make_docx(body, styles=styles), "hidden text")
    else:
        assert texts(guide(body, styles=styles)) == [f"visible {MARK}"]


def test_hidden_base_style_is_not_unhidden_by_a_derived_style() -> None:
    derived = '<w:style w:type="paragraph" w:styleId="D"><w:basedOn w:val="Base"/><w:rPr><w:vanish w:val="false"/></w:rPr></w:style>'
    styles = f.styles_xml(f.DEFAULT_STYLES + f.style("Base", vanish=True) + derived)
    rejected(f.make_docx(f.para(MARK, style_id="D"), styles=styles), "hidden text")


def test_table_style_that_hides_text_rejects_documents_with_tables() -> None:
    hiding = ('<w:style w:type="table" w:styleId="T"><w:tblStylePr w:type="firstRow"><w:rPr><w:vanish/></w:rPr>'
              "</w:tblStylePr></w:style>")
    styles = f.styles_xml(f.DEFAULT_STYLES + hiding)
    rejected(f.make_docx(table([f.para(MARK)]), styles=styles), "table style")
    assert texts(guide(f.para("no tables here"), styles=styles)) == ["no tables here"]
    plain_table_style = '<w:style w:type="table" w:styleId="T2"><w:rPr><w:b/></w:rPr></w:style>'
    assert texts(guide(table([f.para("cell")]), styles=f.styles_xml(f.DEFAULT_STYLES + plain_table_style))) == ["cell"]


def test_subdocuments_are_rejected() -> None:
    rejected(f.make_docx(f.para("x") + f.p('<w:subDoc r:id="r5" xmlns:r="urn:r"/>')), "subDoc")


def test_kitchen_sink_extraction_matches_hand_written_output() -> None:
    body = (
        f.p(f.t("Pre"), f.t("amble"))
        + f.heading("Head", 0)
        + f.p(f.t("See "), f'<w:hyperlink w:anchor="a">{f.t("the ")}{f.t("link")}</w:hyperlink>', f.t(", "),
              f'<w:smartTag w:uri="u" w:element="e"><w:customXml w:element="c">{f.t("nested")}</w:customXml></w:smartTag>',
              f'<w:sdt><w:sdtContent><w:hyperlink w:anchor="b">{f.t(" deep")}</w:hyperlink></w:sdtContent></w:sdt>',
              f'<w:fldSimple w:instr="{INSTR}">{f.t(" field")}</w:fldSimple>', f.t("."))
        + table([f.para("R1C1"), f.para("R1C2a") + f.para("R1C2b")])
        + f'<w:sdt><w:sdtContent>{f.para("In block control")}{table([f.para("T2")])}</w:sdtContent></w:sdt>'
        + f.heading("Sub", 1)
        + f'<w:customXml w:element="c"><w:customXmlPr/>{f.para("Item", num_id="1")}</w:customXml>'
        + f.p(f.t("Tail"), BEGIN, instr(), SEPARATE, f.t(" result"), END)
    )
    g = guide(body)
    expected = [
        [("paragraph", None, "Preamble")],
        [("heading", 0, "Head"), ("paragraph", None, "See the link, nested deep field."),
         ("table", None, "R1C1 | R1C2a\nR1C2b"), ("paragraph", None, "In block control"), ("table", None, "T2")],
        [("heading", 1, "Sub"), ("list", None, "Item"), ("paragraph", None, "Tail result")],
    ]
    assert [[(p.kind, p.level, p.text) for p in s.paragraphs] for s in g.sections] == expected
    assert sg.render(g.sections) == ("Preamble\n\n# Head\n\nSee the link, nested deep field.\n\nR1C1 | R1C2a\nR1C2b\n\n"
                                     "In block control\n\nT2\n\n## Sub\n\n- Item\n\nTail result\n")
    assert INSTR not in sg.render(g.sections)


_TOKEN = st.text(alphabet="abcdefghijklmnopqrstuvwxyz é’", min_size=1, max_size=6)
_WRAPPER = st.sampled_from(["plain", "hyperlink", "smartTag", "customXml", "sdt", "fldSimple", "complex", "sdtLink", "split"])


def _wrap(kind: str, token: str) -> str:
    if kind == "plain":
        return f.t(token)
    if kind == "hyperlink":
        return f'<w:hyperlink w:anchor="a">{f.t(token)}</w:hyperlink>'
    if kind == "smartTag":
        return f'<w:smartTag w:uri="u" w:element="e">{f.t(token)}</w:smartTag>'
    if kind == "customXml":
        return f'<w:customXml w:element="c">{f.t(token)}</w:customXml>'
    if kind == "sdt":
        return f'<w:sdt><w:sdtPr/><w:sdtContent>{f.t(token)}</w:sdtContent></w:sdt>'
    if kind == "fldSimple":
        return f'<w:fldSimple w:instr="{INSTR}">{f.t(token)}</w:fldSimple>'
    if kind == "complex":
        return BEGIN + instr() + SEPARATE + f.t(token) + END
    if kind == "sdtLink":
        return f'<w:sdt><w:sdtContent><w:hyperlink w:anchor="z">{f.t(token)}</w:hyperlink></w:sdtContent></w:sdt>'
    cut = max(1, len(token) // 2)
    return f.t(token[:cut]) + f.t(token[cut:])


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.tuples(_TOKEN, _WRAPPER), min_size=1, max_size=12))
def test_property_containers_never_duplicate_drop_or_reorder_text(items) -> None:
    expected = "".join(token for token, _ in items)
    body = f.p(*(_wrap(kind, token) for token, kind in items))
    if not expected.strip():
        rejected(f.make_docx(body), "no text")
        return
    extracted = texts(guide(body))
    assert extracted == [expected]
    assert INSTR not in extracted[0]


# --------------------------------------------------------------------------------------------
# S4.2a review rounds 3-4: literal "|" in table cells (rejected, option A), private-use message
# --------------------------------------------------------------------------------------------


def _row(*cells: str) -> str:
    return "<w:tbl><w:tr>" + "".join(f"<w:tc>{f.para(c)}</w:tc>" for c in cells) + "</w:tr></w:tbl>"


PIPE_SECRET = f"{MARK}-cell"  # invented cell text that must never appear in an error message


def _cells(*cells: str) -> str:
    return "<w:tr>" + "".join(f"<w:tc>{cell}</w:tc>" for cell in cells) + "</w:tr>"


@pytest.mark.parametrize(
    "rows, location",
    [
        ([_cells(f.para(f"{PIPE_SECRET}|x"), f.para("b"))], "table 1, row 1, cell 1"),
        ([_cells(f.para("a"), f.para(f"{PIPE_SECRET} | x"), f.para("c"))], "table 1, row 1, cell 2"),
        ([_cells(f.para("a"), f.para("|"))], "table 1, row 1, cell 2"),
        ([_cells(f.para("a"), f.para("b")), _cells(f.para("c"), f.para(PIPE_SECRET) + f.para("|"))], "table 1, row 2, cell 2"),
        ([_cells(f.para("a"), f.p(f.t(PIPE_SECRET), '<w:r><w:sym w:font="X" w:char="007C"/></w:r>'))], "table 1, row 1, cell 2"),
        ([_cells(f.para("a"), f.p(f'<w:hyperlink w:anchor="h">{f.t(PIPE_SECRET + "|")}</w:hyperlink>'))], "table 1, row 1, cell 2"),
        ([_cells(f.para("a"), f.p(f.t(PIPE_SECRET), f.t("|")))], "table 1, row 1, cell 2"),  # pipe in its own run
    ],
)
def test_literal_pipe_in_a_table_cell_is_rejected_with_a_safe_location(rows, location: str) -> None:
    data = f.make_docx(f.heading("H", 0) + "<w:tbl>" + "".join(rows) + "</w:tbl>")
    error = rejected(data, location)
    assert "literal '|'" in str(error) and PIPE_SECRET not in str(error)


def test_pipe_location_counts_tables_rows_and_cells_in_document_order() -> None:
    body = (table([f.para("a"), f.para("b")])
            + f'<w:sdt><w:sdtContent>{table([f.para("c")], [f.para("d"), f.para("e"), f.para(PIPE_SECRET + "|")])}'
              "</w:sdtContent></w:sdt>")
    error = rejected(f.make_docx(body), "table 2, row 2, cell 3")
    assert PIPE_SECRET not in str(error)


def test_pipe_outside_tables_is_still_verbatim_text() -> None:
    assert texts(guide(f.para("Use a | b here") + f.p(f.t("x"), f.t("|"), f.t("y")))) == ["Use a | b here", "x|y"]


def test_ordinary_tables_keep_the_unchanged_rendering_contract() -> None:
    g = guide(f.heading("H", 0) + table([f.para("Term"), f.para("Use")], [f.para("one") + f.para("two"), f.p()],
                                        [f.p(), f.para("tail ")]))
    assert [(p.kind, p.text) for p in g.sections[0].paragraphs] == [
        ("heading", "H"), ("table", "Term | Use"), ("table", "one\ntwo | "), ("table", " | tail ")]
    assert sg.render(g.sections) == "# H\n\nTerm | Use\n\none\ntwo | \n\n | tail \n"


def test_ordinary_table_fingerprint_is_unchanged() -> None:
    import hashlib

    g = guide(f.heading("H", 0) + table([f.para("Term"), f.para("Use")]))
    payload = json.dumps(["proofreader-section-v1", [], [["heading", 0, "H"], ["table", None, "Term | Use"]]],
                         ensure_ascii=False, separators=(",", ":"))
    assert g.sections[0].sha256 == hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_former_ambiguity_is_now_a_rejection() -> None:
    """A 2-cell row whose first cell contains ' | ' can no longer collide with a 3-cell row."""
    rejected(f.make_docx(f.heading("H", 0) + _row(f"{PIPE_SECRET} | b", "c")), "literal '|'")
    assert texts(guide(f.heading("H", 0) + _row("a", "b", "c"))) == ["H", "a | b | c"]


# --------------------------------------------------------------------------------------------
# w:instrText outside a complex field's instruction region (round 4)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "runs",
    [
        [instr()],  # no field at all
        [BEGIN, instr("REF "), SEPARATE, instr()],  # in the result region
        [BEGIN, instr("REF "), SEPARATE, f.t("r"), END, instr()],  # after the field ended
        [BEGIN, instr("IF "), BEGIN, instr("X"), SEPARATE, instr(), END, SEPARATE, f.t("r"), END],  # inner result
    ],
)
def test_instr_text_outside_an_instruction_region_is_rejected(runs) -> None:
    error = rejected(f.make_docx(f.p(f.t(MARK), *runs)), "field code text outside a field instruction")
    assert INSTR not in str(error)


def test_instr_text_inside_instruction_regions_is_excluded_not_rejected() -> None:
    body = f.p(f.t("a"), BEGIN, instr("RE"), instr("F "), instr(INSTR), BEGIN, instr("X"), SEPARATE, f.t(INSTR), END,
               instr(" more"), SEPARATE, f.t("b"), END, f.t("c"))
    assert texts(guide(body)) == ["abc"]


def test_rows_without_a_literal_pipe_split_back_into_their_exact_cells() -> None:
    """Without a literal '|', every '|' is a separator, so " | " recovers the cells exactly
    (edge spaces and empty cells included)."""
    cases = {  # cell XML -> expected cell texts
        (f.para("a"), f.para("b")): ["a", "b"],
        (f.para("a "), f.para(" b")): ["a ", " b"],
        (f.p(), f.para("b"), f.para("c")): ["", "b", "c"],
        (f.para("one two"), f.para("x") + f.para("y")): ["one two", "x\ny"],
    }
    for cells, expected in cases.items():
        body = "<w:tbl><w:tr>" + "".join(f"<w:tc>{cell}</w:tc>" for cell in cells) + "</w:tr></w:tbl>"
        assert guide(body).sections[0].paragraphs[0].text.split(" | ") == expected


@pytest.mark.parametrize("char", ["E000", "F0E0", "F8FF"])
def test_private_use_symbols_report_the_private_use_rule(char: str) -> None:
    rejected(f.make_docx(f.p(f.t(MARK), f'<w:r><w:sym w:font="Wingdings" w:char="{char}"/></w:r>')), "private-use")

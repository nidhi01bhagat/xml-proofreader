"""Builds small, invented DOCX packages in memory for the style-guide tests (S4.2a).

No binary fixtures are committed: every document is generated at test time from invented text.
Output is deterministic (fixed timestamps, fixed entry order).
"""

from __future__ import annotations

import io
import zipfile
from xml.sax.saxutils import escape

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE_DOC = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
STYLES_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"
DATE = (2026, 1, 1, 0, 0, 0)

CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    "</Types>"
)
PACKAGE_RELS = (
    f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="{REL_NS}">'
    f'<Relationship Id="rId1" Type="{OFFICE_DOC}" Target="word/document.xml"/></Relationships>'
)
DOCUMENT_RELS = (
    f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="{REL_NS}">'
    f'<Relationship Id="rId1" Type="{STYLES_REL}" Target="styles.xml"/></Relationships>'
)


def style(style_id: str, *, outline: int | None = None, based_on: str | None = None, num_id: str | None = None,
          vanish: bool = False, kind: str = "paragraph", default: bool = False) -> str:
    ppr = ""
    if outline is not None or num_id is not None:
        inner = (f'<w:outlineLvl w:val="{outline}"/>' if outline is not None else "")
        inner += (f'<w:numPr><w:numId w:val="{num_id}"/></w:numPr>' if num_id is not None else "")
        ppr = f"<w:pPr>{inner}</w:pPr>"
    default_attr = ' w:default="1"' if default else ""
    return (f'<w:style w:type="{kind}" w:styleId="{style_id}"{default_attr}>'
            f'<w:name w:val="{style_id}"/>'
            + (f'<w:basedOn w:val="{based_on}"/>' if based_on else "") + ppr
            + ("<w:rPr><w:vanish/></w:rPr>" if vanish else "") + "</w:style>")


DEFAULT_STYLES = "".join([
    style("Normal", default=True),
    style("Head1", outline=0, based_on="Normal"),
    style("Head2", outline=1, based_on="Normal"),
    style("Head3", outline=2, based_on="Normal"),
    style("BodyNote", based_on="Normal"),
    style("ListItem", num_id="1", based_on="Normal"),
])


def styles_xml(styles: str = DEFAULT_STYLES, doc_defaults: str = "") -> str:
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="{W_NS}">'
            f"{doc_defaults}{styles}</w:styles>")


def t(text: str) -> str:
    """A run with one text element (xml:space=preserve keeps edge spaces)."""
    return f'<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def p(*runs: str, style_id: str | None = None, outline: int | None = None, num_id: str | None = None,
      ppr_extra: str = "") -> str:
    props = ""
    if style_id or outline is not None or num_id is not None or ppr_extra:
        props = "<w:pPr>"
        props += f'<w:pStyle w:val="{style_id}"/>' if style_id else ""
        props += f'<w:numPr><w:numId w:val="{num_id}"/></w:numPr>' if num_id is not None else ""
        props += f'<w:outlineLvl w:val="{outline}"/>' if outline is not None else ""
        props += ppr_extra + "</w:pPr>"
    return f"<w:p>{props}{''.join(runs)}</w:p>"


def para(text: str, **kwargs) -> str:
    return p(t(text), **kwargs)


def heading(text: str, level: int) -> str:
    return para(text, style_id=f"Head{level + 1}")


def document_xml(body: str, *, extra_ns: str = "", root: str = "w:document") -> str:
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><{root} xmlns:w="{W_NS}"{extra_ns}>'
            f'<w:body>{body}<w:sectPr/></w:body></{root}>')


def make_docx(body: str = "", *, document: str | None = None, styles: str | None = None,
              with_styles: bool = True, extra: dict[str, bytes | str] | None = None,
              package_rels: str = PACKAGE_RELS, document_rels: str | None = DOCUMENT_RELS,
              compression: int = zipfile.ZIP_DEFLATED, omit: tuple[str, ...] = ()) -> bytes:
    parts: dict[str, bytes | str] = {
        "[Content_Types].xml": CONTENT_TYPES,
        "_rels/.rels": package_rels,
        "word/document.xml": document if document is not None else document_xml(body),
    }
    if with_styles:
        parts["word/styles.xml"] = styles if styles is not None else styles_xml()
        if document_rels is not None:
            parts["word/_rels/document.xml.rels"] = document_rels
    parts.update(extra or {})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in parts.items():
            if name in omit:
                continue
            info = zipfile.ZipInfo(name, DATE)
            info.compress_type = compression
            archive.writestr(info, content.encode("utf-8") if isinstance(content, str) else content)
    return buffer.getvalue()

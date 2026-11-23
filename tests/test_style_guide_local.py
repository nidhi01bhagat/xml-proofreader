"""S4.2a, LOCAL ONLY: the supplied style guide parses as expected (skipped when it is absent).

Privacy: nothing here prints, logs or asserts on guide wording. Extracted text is compared with an
independent regex-based oracle through SHA-256 digests and booleans only, so neither a pass nor a
failure can reveal the guide's content. No facts about the guide are hard-coded.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

import pytest

from proofreader import style_guide as sg
from proofreader.config import LocaleScope

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "SSE Assignment" / "Style Guide.docx"

pytestmark = pytest.mark.skipif(not GUIDE.is_file(), reason="local style guide (SSE Assignment/Style Guide.docx) absent")


def digest(items) -> str:
    return hashlib.sha256("\x1f".join(items).encode("utf-8")).hexdigest()


def _oracle() -> tuple[list[str], list[int | None], str]:
    """Independent extraction: regexes over the raw parts (no shared code with style_guide.py)."""
    __tracebackhide__ = True
    archive = zipfile.ZipFile(GUIDE)
    document = archive.read("word/document.xml").decode("utf-8")
    styles = archive.read("word/styles.xml").decode("utf-8")
    outline, based = {}, {}
    for match in re.finditer(r'<w:style\b[^>]*w:styleId="([^"]+)".*?</w:style>', styles, re.S):
        body = match.group(0)
        level = re.search(r'<w:outlineLvl w:val="(\d+)"', body)
        parent = re.search(r'<w:basedOn w:val="([^"]+)"', body)
        outline[match.group(1)] = int(level.group(1)) if level else None
        based[match.group(1)] = parent.group(1) if parent else None

    def style_level(sid):
        for _ in range(64):
            if sid is None:
                return None
            if outline.get(sid) is not None:
                return outline[sid]
            sid = based.get(sid)
        return None

    texts, levels = [], []
    body = document[document.index("<w:body>"):]
    for paragraph in re.findall(r"<w:p\b(?:[^>]*/>|.*?</w:p>)", body, re.S):
        pieces = re.findall(r"<w:t(?:\s[^>]*)?>([^<]*)</w:t>|<w:r>.*?</w:r>", paragraph, re.S)
        runs = re.findall(r"<w:r\b.*?</w:r>", paragraph, re.S)
        text = ""
        for run in runs:
            for token in re.finditer(r"<w:t(?:\s[^>]*)?>([^<]*)</w:t>|<w:tab/>|<w:br/>", run):
                text += token.group(1) if token.group(1) is not None else ("\t" if token.group(0) == "<w:tab/>" else "\n")
        del pieces
        text = (text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                .replace("&quot;", '"').replace("&apos;", "'"))
        if not text.strip():
            continue
        direct = re.search(r"<w:pPr>.*?<w:outlineLvl w:val=\"(\d+)\"", paragraph, re.S)
        sid = re.search(r'<w:pStyle w:val="([^"]+)"', paragraph)
        level = int(direct.group(1)) if direct else style_level(sid.group(1) if sid else None)
        texts.append(text)
        levels.append(level if level is not None and level <= 8 else None)
    header = "".join(re.findall(r"<w:t(?:\s[^>]*)?>([^<]*)</w:t>", archive.read("word/header1.xml").decode("utf-8"))) \
        if "word/header1.xml" in archive.namelist() else ""
    return texts, levels, header


def _facts() -> dict:
    """Digests, integers and booleans only. Wording stays inside this hidden frame, so even
    ``pytest --showlocals`` cannot print it from a failing test."""
    __tracebackhide__ = True
    guide = sg.load(GUIDE)
    oracle_texts, oracle_levels, header = _oracle()
    ours = [p.text for s in guide.sections for p in s.paragraphs]
    full = sg.select(guide, LocaleScope.FULL, sg.load_manifest())
    needle = header.strip()
    return {
        "same_text": digest(ours) == digest(oracle_texts),
        "our_levels": [p.level if p.kind == "heading" else None for s in guide.sections for p in s.paragraphs],
        "oracle_levels": oracle_levels,
        "total_chars": guide.total_chars,
        "oracle_chars": sum(len(t) for t in oracle_texts),
        "headings": sum(1 for s in guide.sections if s.level is not None),
        "sections": len(guide.sections),
        "full_included": full.included,
        "full_is_render": full.text == sg.render(guide.sections),
        "header_in_output": full.text.count(needle) if needle else 0,
        "header_in_body": sum(t.count(needle) for t in oracle_texts) if needle else 0,
        "scoped_included": [sg.select(guide, scope, sg.load_manifest()).included
                            for scope in (LocaleScope.EN_GB, LocaleScope.CROSS_LOCALE)],
        "hashes": [s.sha256 for s in guide.sections],
        "hashes_again": [s.sha256 for s in sg.load(GUIDE).sections],
    }


def test_local_guide_extraction_matches_an_independent_oracle() -> None:
    facts = _facts()
    assert facts["same_text"] is True, "extracted text differs from the oracle (content deliberately not shown)"
    assert facts["our_levels"] == facts["oracle_levels"]  # integers only
    assert facts["total_chars"] == facts["oracle_chars"]


def test_local_guide_sections_and_selection() -> None:
    facts = _facts()
    assert facts["headings"] == sum(1 for level in facts["oracle_levels"] if level is not None)
    assert facts["full_included"] == facts["sections"] and facts["full_is_render"] is True
    # The page-header part is not included: its text occurs in the output exactly as often as in
    # the body itself (a header may legitimately repeat body wording such as a title).
    assert facts["header_in_output"] == facts["header_in_body"]  # integers only
    assert facts["scoped_included"] == [0, 0]  # S4.2a: the packaged manifest is empty


def test_local_guide_parsing_is_deterministic() -> None:
    facts = _facts()
    assert facts["hashes"] == facts["hashes_again"]  # hex digests only

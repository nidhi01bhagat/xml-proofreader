"""S4.8a: end-to-end golden tests (G1-G6) through pipeline.run AND the command line (cli.main).

XML input -> paragraph extraction -> Fake model responses (hand-written findings in the model's contract form:
original / before / after / word indices) -> S3 resolution -> S2 annotation and verification -> CLI publication.
Inputs and expected outputs are hand-written fixtures (tests/fixtures/generate_xml_fixtures.py); the expected
bytes are never produced by the renderer under test. Offline only: FakeProofreader, no key, no network.
"""

from __future__ import annotations

import dataclasses
import io
import itertools
import json
import random
import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import docx_factory as f
import pytest

from proofreader import cli
from proofreader import config as c
from proofreader import llm
from proofreader import pipeline as pl
from proofreader import resolver as r
from proofreader import style_guide
from proofreader import verify as v
from proofreader import xml_source as xs
from proofreader.config import LocaleScope
from proofreader.llm import CallStatus, FakeBlock, FakeOk, FakeProofreader
from proofreader.models import Annotation, Category, RejectCode, TraceStatus

XML = Path(__file__).resolve().parent / "fixtures" / "xml"
G1_IN = (XML / "golden-e2e-1-input.xml").read_bytes()
G1_OUT = (XML / "golden-e2e-1-expected.xml").read_bytes()
G2_IN = (XML / "golden-e2e-2-input.xml").read_bytes()
G2_OUT = (XML / "golden-e2e-2-expected.xml").read_bytes()
NO_DOTENV = {"PYTHON_DOTENV_DISABLED": "1"}
GUIDE_BODY = f.heading("House rules", 0) + f.para("Write per cent in running text, never the % sign.")
WAIT_S = 5.0


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def paragraph(data: bytes, pid: int):
    return next(p for p in xs.scan(data).paragraphs if p.id == pid)


def finding(data: bytes, pid: int, original: str, correction: str, type_: str, reason: str,
            occurrence: int = 0) -> dict:
    """A finding exactly as the model is told to write it: the original text, up to 3 whitespace-separated
    words of context on each side (taken from the paragraph text itself), and word-grid hints."""
    para = paragraph(data, pid)
    text = para.text
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(original, start + 1)
    end = start + len(original)
    grid = r.word_grid(para)
    first = next(w.index for w in grid if w.end > start)
    last = max(w.index for w in grid if w.start < end)
    return {"original": original, "before": " ".join(text[:start].split()[-3:]),
            "after": " ".join(text[end:].split()[:3]), "word_start": first, "word_end": last,
            "correction": correction, "type": type_, "reason": reason}


def env(*findings: dict) -> dict:
    return {"findings": list(findings)}


EMPTY = env()

G1_FINDINGS = {
    0: env(finding(G1_IN, 0, "recieve", "receive", "spelling", "Misspelling"),
           finding(G1_IN, 0, "tuesday", "Tuesday", "capitalization", "Days of the week are capitalized")),
    1: env(finding(G1_IN, 1, "cofee", "coffee", "spelling", "Misspelling"),
           finding(G1_IN, 1, "lobbby", "lobby", "spelling", "Misspelling")),
    2: env(finding(G1_IN, 2, "then", "than", "grammar", 'Use "than" for comparisons')),
    3: env(finding(G1_IN, 3, "12%", "12 per cent", "styleguide", "House style: write per cent")),
    4: EMPTY,  # outer paragraph: nothing to fix in its own text
    5: env(finding(G1_IN, 5, "a extra", "an extra", "grammar", "Use an before a vowel sound")),
    6: EMPTY,  # clean paragraph
    8: env(finding(G1_IN, 8, "the the", "the", "grammar", "Repeated word")),
}
G1_SENT = [0, 1, 2, 3, 4, 5, 6, 8]  # 7 has an <error> already, 9 is whitespace only
G2_FINDINGS = {0: env(finding(G2_IN, 0, "naive", "naïve", "spelling", "Diaeresis"),
                      finding(G2_IN, 0, "creme brulee", "crème brûlée", "spelling", "French spelling"))}


def strict_fake(script: dict[int, dict]) -> FakeProofreader:
    """Strict: a call for any paragraph not in ``script`` (or a second call) raises FakeScriptError."""
    return FakeProofreader({pid: [FakeOk(payload)] for pid, payload in script.items()})


def make_cfg(tmp_path: Path, lang: str = "en", **cli_values) -> c.RunConfig:
    values = {"input": str(tmp_path / "doc.xml"), "lang": lang, "style_guide": str(tmp_path / "g.docx")}
    values.update(cli_values)
    return c.build_config(values, NO_DOTENV, tmp_path)


def selected_guide(cfg: c.RunConfig) -> style_guide.SelectedGuide:
    return style_guide.select(style_guide.parse_docx(f.make_docx(GUIDE_BODY)), cfg.locale_scope,
                              style_guide.load_manifest())


def run_cli(tmp_path: Path, data: bytes, client, *extra: str, lang: str = "en"):
    (tmp_path / "doc.xml").write_bytes(data)
    (tmp_path / "g.docx").write_bytes(f.make_docx(GUIDE_BODY))
    out, err = io.StringIO(), io.StringIO()
    code = cli.main([str(tmp_path / "doc.xml"), "--lang", lang, "--style-guide", str(tmp_path / "g.docx"), *extra],
                    environ=NO_DOTENV, cwd=tmp_path, proofreader_factory=lambda cfg, rec: client,
                    stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def annotations_from_output(output: bytes) -> list[Annotation]:
    """The <error> elements actually present in ``output`` (the verifier's own reader), for independent checks."""
    root = ET.fromstring(output, parser=ET.XMLParser(encoding="utf-8"))
    paragraphs = [e for e in root.iter() if isinstance(e.tag, str) and e.tag.rsplit("}", 1)[-1] == "p"]
    found = []
    for pid, para in enumerate(paragraphs):
        _, errors = v._own_text_and_errors(para)
        for start, end, el in errors:
            found.append(Annotation(pid, start, end, Category(el.get("type")), el.get("correction"), el.get("reason")))
    return found


def strip_errors(data: bytes) -> bytes:
    return re.sub(rb"<error [^>]*>|</error>", b"", data)


def assert_independently_valid(original: bytes, output: bytes, newly_added: int) -> None:
    """G1 byte invariant, V1-V6 from the independent verifier, well-formedness, AC-14."""
    assert strip_errors(output) == strip_errors(original)  # every byte outside the inserted tags is preserved
    before = annotations_from_output(original)
    after = annotations_from_output(output)
    added = [a for a in after if a not in before]
    assert len(added) == newly_added
    report = v.verify(original, output, added)
    assert report.ok, report
    for a in after:  # AC-14: every styleguide error carries a non-empty reason
        if a.category is Category.STYLEGUIDE:
            assert a.reason and a.reason.strip()


# --------------------------------------------------------------------------------------------
# Fixture sanity (hand-written files are consistent with their own claims)
# --------------------------------------------------------------------------------------------


def test_golden_fixtures_are_well_formed_and_differ_only_by_error_markup() -> None:
    for data in (G1_IN, G1_OUT, G2_IN, G2_OUT):
        ET.fromstring(data, parser=ET.XMLParser(encoding="utf-8"))
    assert strip_errors(G1_OUT) == strip_errors(G1_IN)
    assert strip_errors(G2_OUT) == strip_errors(G2_IN)
    assert G1_OUT.startswith(b"\xef\xbb\xbf") and b"\r\n" in G1_OUT and b"\n<" not in G1_OUT.replace(b"\r\n", b"")
    assert all(b < 0x80 for b in G2_OUT)  # the US-ASCII document stays ASCII


def test_g1_paragraph_map_matches_the_fixture_comment() -> None:
    paras = {p.id: p for p in xs.scan(G1_IN).paragraphs}
    assert len(paras) == 10
    assert paras[2].text.startswith("The café")  # prefixed <x:p> is a paragraph by local name
    assert paras[5].text == "Inner text has a extra word" and paras[4].barriers  # nested inner / outer
    assert [pid for pid, p in paras.items() if not pl.is_eligible(p)] == [7, 9]


# --------------------------------------------------------------------------------------------
# G1: full document through pipeline.run and the CLI
# --------------------------------------------------------------------------------------------


def test_g1_pipeline_reproduces_the_hand_written_output(tmp_path) -> None:
    cfg = make_cfg(tmp_path)
    fake = strict_fake(G1_FINDINGS)
    result = pl.run(G1_IN, cfg, fake, selected_guide(cfg))
    assert result.output == G1_OUT
    assert sorted(fake.call_counts()) == G1_SENT and all(n == 1 for n in fake.call_counts().values())  # AC-25
    assert [rec.paragraph_id for rec in result.call_records] == G1_SENT
    assert all(rec.status is CallStatus.OK for rec in result.call_records) and not result.degraded
    assert [t.status for t in result.trace] == [TraceStatus.APPLIED] * 8 and result.rolled_back == ()
    assert_independently_valid(G1_IN, result.output, newly_added=8)


def test_g1_cli_publishes_the_same_bytes_and_leaves_the_input_untouched(tmp_path) -> None:
    fake = strict_fake(G1_FINDINGS)
    code, out, err = run_cli(tmp_path, G1_IN, fake)
    assert code == 0, err
    output = tmp_path / "doc.corrected.xml"
    assert output.read_bytes() == G1_OUT
    assert (tmp_path / "doc.xml").read_bytes() == G1_IN  # AC-16
    assert out.strip() == str(output)
    assert sorted(fake.call_counts()) == G1_SENT
    assert "findings by status: APPLIED 8" in err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["doc.corrected.xml", "doc.xml", "g.docx"]


def test_clean_paragraphs_stay_byte_identical_ac04(tmp_path) -> None:
    fake = FakeProofreader(default=FakeOk(EMPTY))
    code, _, err = run_cli(tmp_path, G1_IN, fake)
    assert code == 0, err
    assert (tmp_path / "doc.corrected.xml").read_bytes() == G1_IN
    assert sorted(fake.call_counts()) == G1_SENT


def test_g1_failed_call_passes_its_paragraph_through_unchanged_exit_3(tmp_path) -> None:
    """Paragraph 0's call fails (its payload even carries valid findings): that paragraph is published exactly as
    in the input, every other paragraph exactly as in the hand-written expected file, and the run exits 3."""
    script = {pid: [FakeOk(payload)] for pid, payload in G1_FINDINGS.items()}
    script[0] = [llm.FakeFailure(llm.FailureClass.SCHEMA_INVALID, payload=G1_FINDINGS[0])]
    code, _, err = run_cli(tmp_path, G1_IN, FakeProofreader(script))
    assert code == 3, err
    line_in = next(ln for ln in G1_IN.split(b"\r\n") if b"Our office" in ln)
    line_out = next(ln for ln in G1_OUT.split(b"\r\n") if b"Our office" in ln)
    assert (tmp_path / "doc.corrected.xml").read_bytes() == G1_OUT.replace(line_out, line_in)
    assert "WARNING paragraph 0: passthrough (schema_invalid)" in err


# --------------------------------------------------------------------------------------------
# G2: US-ASCII document
# --------------------------------------------------------------------------------------------


def test_g2_us_ascii_output_uses_character_references(tmp_path) -> None:
    code, _, err = run_cli(tmp_path, G2_IN, strict_fake(G2_FINDINGS))
    assert code == 0, err
    output = (tmp_path / "doc.corrected.xml").read_bytes()
    assert output == G2_OUT
    assert_independently_valid(G2_IN, output, newly_added=2)


# --------------------------------------------------------------------------------------------
# G3: re-running on our own output adds nothing (AC-31, AC-11)
# --------------------------------------------------------------------------------------------


def test_g3_rerun_on_the_corrected_output_adds_nothing(tmp_path) -> None:
    # Every paragraph that contains an <error> (also through a nested <p>) is skipped; only the clean paragraph
    # 6 is offered to the model. Even a model that "finds" something there must not touch the error-bearing ones.
    fake = strict_fake({6: EMPTY})
    code, _, err = run_cli(tmp_path, G1_OUT, fake)
    assert code == 0, err
    assert (tmp_path / "doc.corrected.xml").read_bytes() == G1_OUT
    assert sorted(fake.call_counts()) == [6]


# --------------------------------------------------------------------------------------------
# G4: provider completion order does not matter (AC-20), through the CLI
# --------------------------------------------------------------------------------------------


class Handshake:
    """Event handshake around each provider call (no polling, no sleeps): ``started[pid]`` is set by FakeBlock
    when the call is in flight; ``done[pid]`` is set when the call has returned to the pipeline, and
    ``completed`` records the order in which calls returned."""

    def __init__(self, fake: FakeProofreader, started: dict[int, threading.Event]) -> None:
        self.fake, self.started = fake, started
        self.done = {pid: threading.Event() for pid in started}
        self.completed: list[int] = []
        self._lock = threading.Lock()

    def proofread(self, request: llm.ProofreadRequest) -> llm.ProofreadResult:
        try:
            return self.fake.proofread(request)
        finally:
            with self._lock:
                self.completed.append(request.paragraph_id)
            self.done[request.paragraph_id].set()


def ordered_cli_run(tmp_path: Path, release_order: list[int]):
    gates = {pid: threading.Event() for pid in G1_SENT}
    started = {pid: threading.Event() for pid in G1_SENT}
    fake = FakeProofreader({pid: [FakeBlock(gates[pid], FakeOk(G1_FINDINGS[pid]), started=started[pid])]
                            for pid in G1_SENT})
    client = Handshake(fake, started)
    box: dict = {}

    def target():
        box["result"] = run_cli(tmp_path, G1_IN, client, "--concurrency", str(len(G1_SENT)))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    for pid in G1_SENT:  # every call is in flight before any is released
        assert started[pid].wait(WAIT_S), f"call {pid} did not start"
    for pid in release_order:
        gates[pid].set()
        assert client.done[pid].wait(WAIT_S), f"call {pid} did not complete"  # returned before the next release
    thread.join(WAIT_S)
    assert not thread.is_alive()
    assert client.completed == release_order  # the scenario really happened in this order
    code, out, err = box["result"]
    output = (tmp_path / "doc.corrected.xml").read_bytes()
    (tmp_path / "doc.corrected.xml").unlink()
    return code, out, err, output


def test_g4_completion_order_never_changes_output_or_log(tmp_path) -> None:
    orders = [list(G1_SENT), list(reversed(G1_SENT))]
    for seed in range(6):
        order = list(G1_SENT)
        random.Random(seed).shuffle(order)
        orders.append(order)
    results = [ordered_cli_run(tmp_path, order) for order in orders]

    def without_timing(err: str) -> list[str]:
        # The B21 runtime summary (S4.8e) carries wall-clock times by design: exactly one such line per run,
        # excluded from the comparison; every other log line must be identical.
        timing = [ln for ln in err.splitlines() if ln.startswith("INFO run time ")]
        assert len(timing) == 1
        return [ln for ln in err.splitlines() if not ln.startswith("INFO run time ")]

    for code, out, err, output in results:
        assert code == 0 and output == G1_OUT
        assert out == results[0][1]  # stdout identical
        assert without_timing(err) == without_timing(results[0][2])  # the log, line for line, except timing


# --------------------------------------------------------------------------------------------
# G5: valid and invalid findings mixed; style-guide rejection by locale
# --------------------------------------------------------------------------------------------

G5_IN = ('<doc><p>He said “teh cat” and wrote recieve twice then <b>very</b> good indeed today.</p>'
         "<p>Costs fell by 5% last year.</p></doc>").encode("utf-8")
G5_OUT = ('<doc><p>He said “teh cat” and wrote <error type="spelling" correction="receive" '
          'reason="Misspelling">recieve</error> twice then <b>very</b> good indeed today.</p>'
          "<p>Costs fell by 5% last year.</p></doc>").encode("utf-8")  # hand-written


def g5_script() -> dict[int, dict]:
    return {
        0: env(finding(G5_IN, 0, "recieve", "receive", "spelling", "Misspelling"),  # 0 applied
               finding(G5_IN, 0, "teh", "the", "spelling", "Misspelling"),  # 1 inside the quotation
               {"original": "zebra", "before": "", "after": "", "word_start": 0, "word_end": 0,
                "correction": "zebras", "type": "spelling", "reason": "Plural"},  # 2 not in the text
               finding(G5_IN, 0, "very good", "very well", "grammar", "Adverb"),  # 3 crosses </b>
               finding(G5_IN, 0, "twice then", "twice, then", "punctuation", "Comma"),  # 4 overlaps 5
               finding(G5_IN, 0, "then", "than", "grammar", "Comparison"),  # 5 overlaps 4
               {**finding(G5_IN, 0, "indeed", "in deed", "spelling", "Split"),
                "before": "wrong context words"}),  # 6 real text, anchors that do not match
        1: env(finding(G5_IN, 1, "5%", "5 per cent", "styleguide", "House style: write per cent")),
    }


def g5_trace(tmp_path: Path, lang: str):
    cfg = make_cfg(tmp_path, lang=lang)
    return pl.run(G5_IN, cfg, strict_fake(g5_script()), selected_guide(cfg))


def test_g5_only_valid_findings_are_applied_and_every_rejection_is_traced(tmp_path) -> None:
    result = g5_trace(tmp_path, "en")
    expected = G5_OUT.replace(b"by 5% last", b'by <error type="styleguide" correction="5 per cent" '
                              b'reason="House style: write per cent">5%</error> last')  # en: the guide applies
    assert result.output == expected
    codes = {(t.paragraph_id, t.finding_index): (t.status, t.code) for t in result.trace}
    assert codes == {
        (0, 0): (TraceStatus.APPLIED, None),
        (0, 1): (TraceStatus.REJECTED_S3, RejectCode.IN_QUOTATION),
        (0, 2): (TraceStatus.REJECTED_S3, RejectCode.NOT_FOUND),
        (0, 3): (TraceStatus.REJECTED_S2, RejectCode.CROSSES_ELEMENT),
        (0, 4): (TraceStatus.REJECTED_S2, RejectCode.OVERLAP),
        (0, 5): (TraceStatus.REJECTED_S2, RejectCode.OVERLAP),
        (0, 6): (TraceStatus.REJECTED_S3, RejectCode.ANCHOR_MISMATCH),
        (1, 0): (TraceStatus.APPLIED, None),
    }
    assert_independently_valid(G5_IN, result.output, newly_added=2)


@pytest.mark.parametrize("lang", ["fr", "en-GB"])
def test_g5_styleguide_findings_rejected_without_selected_guide_text(tmp_path, lang) -> None:
    result = g5_trace(tmp_path, lang)
    assert result.output == G5_OUT  # the spelling fix still applies; the per-cent rule does not
    styled = [t for t in result.trace if t.paragraph_id == 1]
    assert [(t.status, t.code) for t in styled] == [(TraceStatus.REJECTED_S3, RejectCode.INVALID_FIELD)]


def test_g5_through_the_cli_logs_codes_but_no_text(tmp_path) -> None:
    code, _, err = run_cli(tmp_path, G5_IN, strict_fake(g5_script()), "--log-level", "DEBUG",
                           "--metrics-json", str(tmp_path / "m.json"), lang="fr")
    assert code == 0, err
    assert (tmp_path / "doc.corrected.xml").read_bytes() == G5_OUT
    metrics = json.loads((tmp_path / "m.json").read_text())
    assert metrics["findings"]["rejected_by_code"] == {"ANCHOR_MISMATCH": 1, "CROSSES_ELEMENT": 1, "INVALID_FIELD": 1,
                                                       "IN_QUOTATION": 1, "NOT_FOUND": 1, "OVERLAP": 2}
    for word in ("teh", "zebra", "recieve", "per cent", "twice"):
        assert word not in err


# --------------------------------------------------------------------------------------------
# G6: requests seen by the model, end to end (AC-27, partial)
# --------------------------------------------------------------------------------------------


class Capturing:
    """Records every request, then answers like the strict G1 Fake."""

    def __init__(self) -> None:
        self.fake = FakeProofreader(default=FakeOk(EMPTY))
        self.requests: list[llm.ProofreadRequest] = []
        self.lock = threading.Lock()

    def proofread(self, request: llm.ProofreadRequest) -> llm.ProofreadResult:
        with self.lock:
            self.requests.append(request)
        return self.fake.proofread(request)


@pytest.mark.parametrize("lang,scope,has_guide", [("en", LocaleScope.FULL, True), ("en-us", LocaleScope.FULL, True),
                                                  ("en-gb", LocaleScope.EN_GB, False),
                                                  ("fr", LocaleScope.CROSS_LOCALE, False)])
def test_g6_requests_carry_one_system_prompt_and_the_locale_selection(tmp_path, lang, scope, has_guide) -> None:
    client = Capturing()
    code, _, err = run_cli(tmp_path, G1_IN, client, lang=lang)
    assert code == 0, err
    assert sorted(req.paragraph_id for req in client.requests) == G1_SENT
    assert len({req.system for req in client.requests}) == 1  # byte-identical system prompt for the whole run
    users = [json.loads(req.user) for req in client.requests]
    normalised = c.normalise_lang(lang)
    assert {u["lang"] for u in users} == {normalised} and {u["scope"] for u in users} == {scope.value}
    guides = {json.dumps(u["style_guide"], sort_keys=True) for u in users}
    assert len(guides) == 1  # the same guide block for every paragraph
    guide_block = users[0]["style_guide"]
    if has_guide:
        assert guide_block is not None and "per cent" in guide_block["text"]
    else:
        assert guide_block is None  # en-GB / other locales: the supplied guide's rules are not sent (D8)
    fingerprints = [req.fingerprint for req in client.requests]
    assert len(set(fingerprints)) == len(fingerprints)  # one distinct request per paragraph


def test_g6_system_prompt_is_identical_across_locales(tmp_path) -> None:
    systems = set()
    for lang in ("en", "en-GB", "fr"):
        client = Capturing()
        sub = tmp_path / lang
        sub.mkdir()
        assert run_cli(sub, G1_IN, client, lang=lang)[0] == 0
        systems |= {req.system for req in client.requests}
    assert len(systems) == 1

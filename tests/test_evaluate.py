"""S4.8e: scripts/evaluate.py (AC-15) on invented data only (the S4.8a golden document and variants of it)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "tests" / "fixtures" / "xml"
G1_IN = (XML / "golden-e2e-1-input.xml").read_bytes()
G1_OUT = (XML / "golden-e2e-1-expected.xml").read_bytes()


def _load():
    spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["evaluate"] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


EV = _load()
RECIEVE = b'<error type="spelling" correction="receive" reason="Misspelling">recieve</error>'


def files(tmp_path: Path, predicted: bytes, expected: bytes = G1_OUT, original: bytes = G1_IN):
    (tmp_path / "in.xml").write_bytes(original)
    (tmp_path / "gold.xml").write_bytes(expected)
    (tmp_path / "pred.xml").write_bytes(predicted)
    return tmp_path / "in.xml", tmp_path / "gold.xml", tmp_path / "pred.xml"


def scores(tmp_path, predicted: bytes, **kw) -> dict:
    return EV.evaluate(*files(tmp_path, predicted, **kw))


def level(report, name):
    s = report["levels"][name]
    return s["tp"], s["fp"], s["fn"], s["precision"], s["recall"]


def test_identical_output_scores_1_at_every_level(tmp_path) -> None:
    report = scores(tmp_path, G1_OUT)
    assert report["paragraphs"] == 10 and report["expected"] == 8 and report["predicted"] == 8  # existing <error> ignored
    for name in EV.LEVELS:
        assert level(report, name) == (8, 0, 0, 1.0, 1.0)
    assert report["styleguide_reason_coverage"] == 1.0
    assert report["by_type"]["grammar"] == {"expected": 3, "predicted": 3, "matched": 3, "precision": 1.0,
                                            "recall": 1.0}


def test_missing_annotation_lowers_recall_only(tmp_path) -> None:
    report = scores(tmp_path, G1_OUT.replace(RECIEVE, b"recieve"))
    assert level(report, "strict") == (7, 0, 1, 1.0, 0.875)


def test_extra_annotation_lowers_precision_only(tmp_path) -> None:
    extra = G1_OUT.replace(b"new chairs", b'new <error type="spelling" correction="chair">chairs</error>')
    assert level(scores(tmp_path, extra), "strict") == (8, 1, 0, 0.8889, 1.0)


def test_wrong_type_matches_on_span_but_not_type(tmp_path) -> None:
    report = scores(tmp_path, G1_OUT.replace(RECIEVE, RECIEVE.replace(b'"spelling"', b'"grammar"')))
    assert level(report, "span")[0] == 8 and level(report, "type")[0] == 7 and level(report, "strict")[0] == 7
    assert report["by_type"]["spelling"]["recall"] == 0.6667  # 2 of the 3 expected spelling errors matched by type
    assert report["by_type"]["grammar"]["precision"] == 0.75  # 3 of the 4 predicted grammar errors are right


def test_wrong_correction_fails_only_the_strict_level(tmp_path) -> None:
    report = scores(tmp_path, G1_OUT.replace(RECIEVE, RECIEVE.replace(b'"receive"', b'"receives"')))
    assert level(report, "type")[0] == 8 and level(report, "strict")[0] == 7


def test_multi_word_span_difference_matches_only_by_overlap(tmp_path) -> None:
    wider = G1_OUT.replace(b"will " + RECIEVE, b'<error type="spelling" correction="will receive" '
                                                b'reason="Misspelling">will recieve</error>')
    report = scores(tmp_path, wider)
    assert level(report, "span")[0] == 7 and level(report, "overlap")[0] == 8


def test_one_span_covering_two_expected_errors_matches_only_one(tmp_path) -> None:
    expected = (b'<r><p>The <error type="spelling" correction="cat">cta</error> '
                b'<error type="spelling" correction="sat">sta</error> down.</p></r>')
    predicted = b'<r><p>The <error type="spelling" correction="cat sat">cta sta</error> down.</p></r>'
    report = scores(tmp_path, predicted, expected=expected, original=b"<r><p>The cta sta down.</p></r>")
    assert level(report, "overlap") == (1, 0, 1, 1.0, 0.5)  # one-to-one pairing, no double counting


def test_nested_paragraph_offsets_are_per_paragraph(tmp_path) -> None:
    report = scores(tmp_path, G1_OUT)
    inner = [a for a in EV.read(tmp_path / "pred.xml")[1] if a.paragraph == 5]
    assert [(a.start, a.end, a.correction) for a in inner] == [(15, 22, "an extra")]
    assert report["levels"]["strict"]["tp"] == 8


@pytest.mark.parametrize("bad,message", [
    (G1_OUT.replace(b"Everything here is correct.", b"Everything here is correct!"), "differs from the original"),
    (G1_OUT.replace(b"  <p>   </p>\r\n", b""), "paragraphs, the original has 10"),
    (b"<r><p>unclosed</r>", "cannot be read as XML"),
])
def test_inputs_that_are_not_the_same_document_are_refused(tmp_path, bad, message) -> None:
    with pytest.raises(EV.EvaluationError, match=message):
        scores(tmp_path, bad)


def test_no_annotations_anywhere_reports_na(tmp_path) -> None:
    data = b"<r><p>Nothing to fix.</p></r>"
    report = scores(tmp_path, data, expected=data, original=data)
    assert level(report, "strict") == (0, 0, 0, "n/a", "n/a") and report["levels"]["strict"]["f1"] == "n/a"
    assert report["styleguide_reason_coverage"] == "n/a"


def run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", "-B", "-X", "utf8", str(ROOT / "scripts" / "evaluate.py"), *args],
                          capture_output=True, text=True, timeout=60)


def test_command_line_report_json_and_exit_codes(tmp_path) -> None:
    original, gold, _ = files(tmp_path, G1_OUT.replace(RECIEVE, b"recieve"))
    ok = run_cli(tmp_path, "--original", str(original), "--expected", str(gold), "--predicted",
                 str(tmp_path / "pred.xml"), "--json", str(tmp_path / "r.json"), "--details")
    assert ok.returncode == 0, ok.stderr
    assert "strict" in ok.stdout and "0.875" in ok.stdout
    for word in ("recieve", "receive", "Misspelling", "office"):  # ids and offsets only, never document text
        assert word not in ok.stdout and word not in (tmp_path / "r.json").read_text()
    report = json.loads((tmp_path / "r.json").read_text())
    assert set(report) == {"paragraphs", "expected", "predicted", "levels", "by_type", "styleguide_reason_coverage",
                           "unmatched"}
    assert report["unmatched"]["strict"]["expected"] == [{"paragraph": 0, "start": 16, "end": 23,
                                                          "type": "spelling"}]
    (tmp_path / "pred.xml").write_bytes(b"<r><p>other</p></r>")
    assert run_cli(tmp_path, "--original", str(original), "--expected", str(gold), "--predicted",
                   str(tmp_path / "pred.xml")).returncode == 1
    assert run_cli(tmp_path, "--original", str(original)).returncode == 2

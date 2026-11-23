"""S4.7: command-line interface.

Stage 1: output naming, artifact path validation (before any API call) and atomic writes.
Offline only; every file lives in pytest's tmp_path; ./.env loading is disabled.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from proofreader import cli
from proofreader import config as c
from proofreader.config import UsageError

NO_DOTENV = {"PYTHON_DOTENV_DISABLED": "1"}


def make_cfg(tmp_path: Path, input_name: str = "doc.xml", **cli_values) -> c.RunConfig:
    values = {"input": str(tmp_path / input_name), "lang": "en", "style_guide": str(tmp_path / "g.docx")}
    values.update({k: (str(v) if isinstance(v, Path) else v) for k, v in cli_values.items()})
    return c.build_config(values, NO_DOTENV, tmp_path)


def leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.startswith(".tmp-"))


def hard_link_or_skip(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        pytest.skip("hard links are not supported here")


def symlink_or_skip(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not permitted here")


# --------------------------------------------------------------------------------------------
# Output naming (AC-16)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("x.xml", "x.corrected.xml"),
    ("x", "x.corrected.xml"),
    ("a.b.xml", "a.b.corrected.xml"),
    ("X.XML", "X.corrected.xml"),
    ("report.corrected.xml", "report.corrected.corrected.xml"),
    ("notes.txt", "notes.corrected.xml"),
])
def test_output_name(name, expected) -> None:
    assert cli.output_path_for(Path(name)) == Path(expected)


def test_output_stays_in_the_input_directory(tmp_path) -> None:
    assert cli.output_path_for(tmp_path / "sub" / "x.xml") == tmp_path / "sub" / "x.corrected.xml"


# --------------------------------------------------------------------------------------------
# same_file (D1)
# --------------------------------------------------------------------------------------------


def test_same_file_detects_spelling_variants_and_hard_links(tmp_path) -> None:
    a = tmp_path / "a.xml"
    a.write_bytes(b"<p/>")
    (tmp_path / "sub").mkdir()
    assert cli.same_file(a, tmp_path / "sub" / ".." / "a.xml")
    assert not cli.same_file(a, tmp_path / "b.xml")
    hard_link_or_skip(a, tmp_path / "alias.xml")
    assert cli.same_file(a, tmp_path / "alias.xml")


def test_same_file_detects_symlinks(tmp_path) -> None:
    a = tmp_path / "a.xml"
    a.write_bytes(b"<p/>")
    symlink_or_skip(a, tmp_path / "link.xml")
    assert cli.same_file(a, tmp_path / "link.xml")


@pytest.mark.skipif(sys.platform != "win32", reason="case-insensitive filesystem behaviour (Windows)")
def test_same_file_respects_case_insensitive_filesystems(tmp_path) -> None:
    a = tmp_path / "Doc.xml"
    a.write_bytes(b"<p/>")
    assert cli.same_file(a, tmp_path / "DOC.XML")  # existing file
    assert cli.same_file(tmp_path / "New.xml", tmp_path / "NEW.XML")  # neither exists yet


# --------------------------------------------------------------------------------------------
# plan_artifacts: validation before any API call (D1, D3, D7)
# --------------------------------------------------------------------------------------------


def test_plan_accepts_fresh_targets(tmp_path) -> None:
    cfg = make_cfg(tmp_path, record=tmp_path / "r.recording.json", metrics_json=tmp_path / "m.json",
                   log_file=tmp_path / "run.log")
    paths = cli.plan_artifacts(cfg)
    assert paths.output == tmp_path / "doc.corrected.xml"
    assert [label for label, _ in paths.auxiliary()] == ["--record", "--metrics-json", "--log-file"]
    assert leftovers(tmp_path) == []  # the directory probe leaves nothing behind


@pytest.mark.parametrize("which", ["output", "record", "metrics_json", "log_file"])
def test_existing_target_refused_without_force_and_left_untouched(tmp_path, which) -> None:
    targets = {"output": tmp_path / "doc.corrected.xml", "record": tmp_path / "r.recording.json",
               "metrics_json": tmp_path / "m.json", "log_file": tmp_path / "run.log"}
    targets[which].write_bytes(b"ORIGINAL")
    extra = {k: v for k, v in targets.items() if k != "output"}
    with pytest.raises(UsageError) as info:
        cli.plan_artifacts(make_cfg(tmp_path, **extra))
    assert info.value.exit_code == 2 and "already exists" in info.value.message
    assert targets[which].read_bytes() == b"ORIGINAL"
    assert cli.plan_artifacts(make_cfg(tmp_path, force=True, **extra)).output == targets["output"]


def test_record_name_must_end_with_recording_suffix(tmp_path) -> None:
    with pytest.raises(UsageError) as info:
        cli.plan_artifacts(make_cfg(tmp_path, record=tmp_path / "r.json"))
    assert info.value.exit_code == 2 and ".recording.json" in info.value.message


@pytest.mark.parametrize("which", ["record", "metrics_json", "log_file"])
def test_auxiliary_target_may_not_be_the_input(tmp_path, which) -> None:
    input_name = "doc.recording.json" if which == "record" else "doc.xml"
    with pytest.raises(UsageError) as info:
        cli.plan_artifacts(make_cfg(tmp_path, input_name=input_name, force=True, **{which: tmp_path / input_name}))
    assert "never overwritten" in info.value.message


def test_hard_link_alias_of_the_input_is_refused_even_with_force(tmp_path) -> None:
    source = tmp_path / "doc.xml"
    source.write_bytes(b"<doc><p>original</p></doc>")
    hard_link_or_skip(source, tmp_path / "doc.corrected.xml")  # the output name is an alias of the input
    with pytest.raises(UsageError) as info:
        cli.plan_artifacts(make_cfg(tmp_path, force=True))
    assert "input file" in info.value.message
    assert source.read_bytes() == b"<doc><p>original</p></doc>"


def test_symlink_alias_of_the_input_is_refused_even_with_force(tmp_path) -> None:
    source = tmp_path / "doc.xml"
    source.write_bytes(b"<doc/>")
    symlink_or_skip(source, tmp_path / "m.json")
    with pytest.raises(UsageError):
        cli.plan_artifacts(make_cfg(tmp_path, force=True, metrics_json=tmp_path / "m.json"))


@pytest.mark.skipif(sys.platform != "win32", reason="case-insensitive filesystem behaviour (Windows)")
def test_case_variant_of_the_input_is_refused_on_windows(tmp_path) -> None:
    (tmp_path / "doc.xml").write_bytes(b"<doc/>")
    with pytest.raises(UsageError):
        cli.plan_artifacts(make_cfg(tmp_path, force=True, log_file=tmp_path / "DOC.XML"))


@pytest.mark.parametrize("a,b", [("metrics_json", "log_file"), ("record", "log_file"), ("record", "metrics_json")])
def test_auxiliary_destinations_may_not_collide(tmp_path, a, b) -> None:
    shared = tmp_path / "same.recording.json"
    with pytest.raises(UsageError) as info:
        cli.plan_artifacts(make_cfg(tmp_path, **{a: shared, b: shared}))
    assert "same file" in info.value.message


def test_auxiliary_destination_may_not_be_the_corrected_output(tmp_path) -> None:
    with pytest.raises(UsageError) as info:
        cli.plan_artifacts(make_cfg(tmp_path, metrics_json=tmp_path / "doc.corrected.xml"))
    assert "same file" in info.value.message


def test_target_that_is_a_directory_is_refused(tmp_path) -> None:
    (tmp_path / "out.json").mkdir()
    with pytest.raises(UsageError):
        cli.plan_artifacts(make_cfg(tmp_path, force=True, metrics_json=tmp_path / "out.json"))


def test_missing_destination_directory_is_fatal(tmp_path) -> None:
    with pytest.raises(cli.OutputError) as info:
        cli.plan_artifacts(make_cfg(tmp_path, log_file=tmp_path / "missing" / "run.log"))
    assert info.value.exit_code == 1


def test_unwritable_destination_directory_is_fatal(tmp_path, monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(cli.tempfile, "mkstemp", refuse)
    with pytest.raises(cli.OutputError) as info:
        cli.plan_artifacts(make_cfg(tmp_path))
    assert info.value.exit_code == 1 and "cannot create files" in info.value.message


# --------------------------------------------------------------------------------------------
# write_atomic (AC-16)
# --------------------------------------------------------------------------------------------


def test_atomic_write_publishes_exact_bytes(tmp_path) -> None:
    target = tmp_path / "doc.corrected.xml"
    cli.write_atomic(target, b"<doc>\r\n<p>caf\xc3\xa9</p></doc>", force=False)
    assert target.read_bytes() == b"<doc>\r\n<p>caf\xc3\xa9</p></doc>"
    assert leftovers(tmp_path) == []


def test_atomic_write_refuses_existing_target_without_force(tmp_path) -> None:
    target = tmp_path / "doc.corrected.xml"
    target.write_bytes(b"OLD")
    with pytest.raises(UsageError) as info:
        cli.write_atomic(target, b"NEW", force=False)
    assert "already exists" in info.value.message  # refused up front, not reported as a race
    assert target.read_bytes() == b"OLD" and leftovers(tmp_path) == []
    cli.write_atomic(target, b"NEW", force=True)
    assert target.read_bytes() == b"NEW" and leftovers(tmp_path) == []


def test_target_created_during_the_run_is_not_replaced(tmp_path, monkeypatch) -> None:
    """The existence check passed, then the file appeared before publication: the exclusive publish refuses."""
    target = tmp_path / "doc.corrected.xml"
    real_fsync = os.fsync

    def fsync_then_race(fd):
        real_fsync(fd)
        target.write_bytes(b"RACER")

    monkeypatch.setattr(cli.os, "fsync", fsync_then_race)
    with pytest.raises(UsageError) as info:
        cli.write_atomic(target, b"NEW", force=False)
    assert "appeared during the run" in info.value.message
    assert target.read_bytes() == b"RACER" and leftovers(tmp_path) == []


@pytest.mark.parametrize("stage", ["fsync", "publish"])
def test_failure_publishes_nothing_and_leaves_no_temporary_file(tmp_path, monkeypatch, stage) -> None:
    def boom(*args, **kwargs):
        raise OSError("disk error")

    if stage == "fsync":
        monkeypatch.setattr(cli.os, "fsync", boom)
    else:
        for name in ("rename", "link", "replace"):
            monkeypatch.setattr(cli.os, name, boom)
    target = tmp_path / "doc.corrected.xml"
    with pytest.raises(cli.OutputError) as info:
        cli.write_atomic(target, b"NEW", force=False)
    assert info.value.exit_code == 1 and "nothing was published" in info.value.message
    assert not target.exists() and leftovers(tmp_path) == []


def test_forced_write_failure_keeps_the_previous_file(tmp_path, monkeypatch) -> None:
    target = tmp_path / "doc.corrected.xml"
    target.write_bytes(b"OLD")

    def boom(*args, **kwargs):
        raise OSError("disk error")

    monkeypatch.setattr(cli.os, "replace", boom)
    with pytest.raises(cli.OutputError):
        cli.write_atomic(target, b"NEW", force=True)
    assert target.read_bytes() == b"OLD" and leftovers(tmp_path) == []


def test_protected_input_is_never_overwritten_even_with_force(tmp_path) -> None:
    source = tmp_path / "doc.xml"
    source.write_bytes(b"INPUT")
    with pytest.raises(UsageError):
        cli.write_atomic(source, b"NEW", force=True, protected=[source])
    hard_link_or_skip(source, tmp_path / "alias.xml")
    with pytest.raises(UsageError):
        cli.write_atomic(tmp_path / "alias.xml", b"NEW", force=True, protected=[source])
    assert source.read_bytes() == b"INPUT" and leftovers(tmp_path) == []


def test_write_into_a_directory_target_is_refused(tmp_path) -> None:
    (tmp_path / "out").mkdir()
    with pytest.raises(UsageError):
        cli.write_atomic(tmp_path / "out", b"x", force=True)


def test_existing_parser_and_parse_config_are_unchanged(tmp_path) -> None:
    cfg = cli.parse_config([str(tmp_path / "a.xml"), "--lang", "en-gb", "--style-guide", "g.docx"],
                           environ=NO_DOTENV, cwd=tmp_path)
    assert cfg.lang == "en-GB" and cfg.force is False and cfg.record_path is None


# ============================================================================================
# Stage 2: main() orchestration, publication, logging, metrics, exit codes
# ============================================================================================

import io  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import subprocess  # noqa: E402

import docx_factory as f  # noqa: E402
import httpx2  # noqa: E402

from proofreader import llm  # noqa: E402
from proofreader import prompt as pr  # noqa: E402
from proofreader import xml_source as xs  # noqa: E402
from proofreader.llm import FailureClass, FakeFailure, FakeOk, FakeProofreader, FakeRaise  # noqa: E402

DUMMY_KEY = "dummy-test-key-not-real-SENTINELKEY42"
WORD = "Zqxsentinel"  # a document word that must never appear in logs or metrics
DOC = (f"<doc><p>Our team will recieve the {WORD} report.</p><p>Nothing to fix here.</p>"
       f"<p>Sales rose by 12% this year.</p></doc>").encode("utf-8")
GUIDE_BODY = f.heading("House rules", 0) + f.para("Write per cent in running text, never the % sign.")


def finding(data: bytes, pid: int, original: str, correction: str, type_: str = "spelling",
            reason: str = "Misspelling") -> dict:
    paragraph = next(p for p in xs.scan(data).paragraphs if p.id == pid)
    words = [w for group in pr.segment_words(paragraph) for _, w in group]
    n = len(original.split(" "))
    (i,) = [k for k in range(len(words) - n + 1) if " ".join(words[k:k + n]) == original]
    j = i + n - 1
    return {"original": original, "before": " ".join(words[max(0, i - 3):i]), "after": " ".join(words[j + 1:j + 4]),
            "word_start": i, "word_end": j, "correction": correction, "type": type_, "reason": reason}


FINDINGS = {0: {"findings": [finding(DOC, 0, "recieve", "receive", reason=f"Misspelling near {WORD}")]},
            1: {"findings": []},
            2: {"findings": [finding(DOC, 2, "12%", "12 per cent", "styleguide", "House rule: per cent")]}}


class Run:
    def __init__(self, code, out, err, tmp):
        self.code, self.out, self.err, self.tmp = code, out, err, tmp

    @property
    def output(self) -> Path:
        return self.tmp / "doc.corrected.xml"


def setup_files(tmp_path: Path, data: bytes = DOC) -> None:
    (tmp_path / "doc.xml").write_bytes(data)
    (tmp_path / "g.docx").write_bytes(f.make_docx(GUIDE_BODY))


def run_main(tmp_path: Path, *extra: str, factory=None, environ=None, data: bytes | None = DOC,
             lang: str = "en") -> Run:
    if data is not None:
        setup_files(tmp_path, data)
    out, err = io.StringIO(), io.StringIO()
    argv = [str(tmp_path / "doc.xml"), "--lang", lang, "--style-guide", str(tmp_path / "g.docx"), *extra]
    code = cli.main(argv, environ=NO_DOTENV if environ is None else environ, cwd=tmp_path,
                    proofreader_factory=factory, stdout=out, stderr=err)
    return Run(code, out.getvalue(), err.getvalue(), tmp_path)


def fake_factory(fake):
    return lambda cfg, recorder: fake


def ok_fake():
    return FakeProofreader({pid: [FakeOk(payload)] for pid, payload in FINDINGS.items()})


def mock_anthropic_factory(statuses: dict[int, int] | None = None, seen: list | None = None):
    """The REAL AnthropicProofreader and SDK over an in-memory transport (no network)."""
    statuses = statuses or {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        user = json.loads(body["messages"][0]["content"])
        text = "".join(user["paragraph"]["segments"])
        pid = next(i for i, word in ((0, "recieve"), (1, "Nothing"), (2, "Sales")) if word in text)
        if seen is not None:
            seen.append(pid)
        if statuses.get(pid):
            return httpx2.Response(statuses[pid], json={"type": "error", "error": {"type": "x", "message": "m"}},
                                   headers={"request-id": f"req_{statuses[pid]}"})
        message = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                   "content": [{"type": "text", "text": json.dumps(FINDINGS[pid])}], "stop_reason": "end_turn",
                   "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5,
                                                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}
        return httpx2.Response(200, json=message, headers={"request-id": f"req_{pid}"})

    def factory(cfg, recorder):
        http = httpx2.Client(transport=httpx2.MockTransport(handler), trust_env=False)
        return llm.AnthropicProofreader(cfg, http_client=http, recorder=recorder, sleep=lambda s: None)

    return factory


KEY_ENV = {**NO_DOTENV, "ANTHROPIC_API_KEY": DUMMY_KEY}


def artifacts(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in tmp_path.iterdir() if p.name not in {"doc.xml", "g.docx"})


# ---------------------------------------------------------------- success and passthrough


def test_success_writes_the_corrected_output_and_exits_0(tmp_path) -> None:
    fake = ok_fake()
    run = run_main(tmp_path, factory=fake_factory(fake))
    assert run.code == 0, run.err
    out = run.output.read_bytes()
    assert b'<error type="spelling" correction="receive"' in out and b'type="styleguide"' in out
    assert run.out.strip() == str(run.output)
    assert "corrected output written: doc.corrected.xml" in run.err
    assert artifacts(tmp_path) == ["doc.corrected.xml"]  # no temporary or probe files left
    assert (tmp_path / "doc.xml").read_bytes() == DOC  # AC-16: input unchanged


def test_already_annotated_input_makes_no_calls_and_is_identical_ac11(tmp_path) -> None:
    data = b'<doc><p>A <error type="spelling" correction="cat">cta</error> sat.</p><p/></doc>'
    fake = FakeProofreader()
    run = run_main(tmp_path, factory=fake_factory(fake), data=data)
    assert run.code == 0 and fake.calls == () and run.output.read_bytes() == data


@pytest.mark.parametrize("failure", list(FailureClass))
def test_paragraph_failure_writes_output_logs_the_class_and_exits_3_ac13(tmp_path, failure) -> None:
    fake = FakeProofreader({0: [FakeOk(FINDINGS[0])], 1: [FakeFailure(failure)], 2: [FakeOk(FINDINGS[2])]})
    run = run_main(tmp_path, factory=fake_factory(fake))
    assert run.code == 3 and run.output.exists()
    assert f"WARNING paragraph 1: passthrough ({failure.value})" in run.err


def test_http_429_through_the_real_client_exits_3_ac13(tmp_path) -> None:
    run = run_main(tmp_path, factory=mock_anthropic_factory({1: 429}), environ=KEY_ENV)
    assert run.code == 3 and run.output.exists()
    assert "paragraph 1: passthrough (rate_limited), attempts 3" in run.err


def test_existing_output_refused_before_any_call_then_forced(tmp_path) -> None:
    setup_files(tmp_path)
    (tmp_path / "doc.corrected.xml").write_bytes(b"OLD")
    fake = ok_fake()
    run = run_main(tmp_path, factory=fake_factory(fake), data=None)
    assert run.code == 2 and fake.calls == () and run.output.read_bytes() == b"OLD"
    run = run_main(tmp_path, "--force", factory=fake_factory(ok_fake()), data=None)
    assert run.code == 0 and run.output.read_bytes() != b"OLD"


# ---------------------------------------------------------------- fatal before any call (AC-22, AC-18)


@pytest.mark.parametrize("case,expected", [
    ("malformed", 1), ("utf16", 2), ("missing_input", 1), ("missing_guide", 1), ("bad_guide", 1),
    ("bad_lang", 2), ("cache", 2), ("max_calls", 2), ("record_suffix", 2), ("replay_missing_file", 1),
    ("replay_bad_file", 1), ("existing_log", 2),
])
def test_fatal_and_usage_errors_write_nothing_and_make_no_call_ac22(tmp_path, case, expected) -> None:
    setup_files(tmp_path)
    extra, lang = [], "en"
    if case == "malformed":
        (tmp_path / "doc.xml").write_bytes(b"<doc><p>unclosed</doc>")
    elif case == "utf16":
        (tmp_path / "doc.xml").write_bytes("<doc><p>x</p></doc>".encode("utf-16"))
    elif case == "missing_input":
        (tmp_path / "doc.xml").unlink()
    elif case == "missing_guide":
        (tmp_path / "g.docx").unlink()
    elif case == "bad_guide":
        (tmp_path / "g.docx").write_bytes(b"not a zip")
    elif case == "bad_lang":
        lang = "en_US_x_invalid!"
    elif case == "cache":
        extra = ["--cache"]
    elif case == "max_calls":
        extra = ["--max-calls", "1"]
    elif case == "record_suffix":
        extra = ["--record", str(tmp_path / "r.json")]
    elif case == "replay_missing_file":
        extra = ["--replay", str(tmp_path / "none.recording.json")]
    elif case == "replay_bad_file":
        (tmp_path / "bad.recording.json").write_text("{}", encoding="utf-8")
        extra = ["--replay", str(tmp_path / "bad.recording.json")]
    elif case == "existing_log":
        (tmp_path / "run.log").write_text("OLD", encoding="utf-8")
        extra = ["--log-file", str(tmp_path / "run.log")]
    fake = ok_fake()
    factory = None if case.startswith("replay") else fake_factory(fake)
    before = set(artifacts(tmp_path))
    run = run_main(tmp_path, *extra, factory=factory, data=None, lang=lang)
    assert run.code == expected, run.err
    assert fake.calls == ()
    assert not run.output.exists() and set(artifacts(tmp_path)) == before
    assert "error" in run.err.lower()


def test_missing_api_key_in_live_mode_is_fatal_before_any_request(tmp_path) -> None:
    run = run_main(tmp_path, factory=None, environ=NO_DOTENV)  # live mode, no key: no client is ever built
    assert run.code == 1 and "ANTHROPIC_API_KEY is not set" in run.err and not run.output.exists()


def test_incomplete_replay_is_fatal_before_any_call(tmp_path) -> None:
    rec = tmp_path / "r.recording.json"
    assert run_main(tmp_path, "--record", str(rec), factory=mock_anthropic_factory(), environ=KEY_ENV).code == 0
    (tmp_path / "doc.corrected.xml").unlink()
    document = json.loads(rec.read_text(encoding="utf-8"))
    document["entries"] = document["entries"][:1]
    partial = tmp_path / "partial.recording.json"
    partial.write_text(json.dumps(document), encoding="utf-8")
    run = run_main(tmp_path, "--replay", str(partial), data=None)
    assert run.code == 1 and "no response for 2 of 3" in run.err and not run.output.exists()


@pytest.mark.parametrize("fails", [False, True])
def test_client_is_closed_after_the_run(tmp_path, fails) -> None:
    class Closing(FakeProofreader):
        closed = 0

        def close(self):
            Closing.closed += 1

    script = {0: [FakeRaise(llm.ProviderFatalError("HTTP 401"))]} if fails else {}
    fake = Closing(script, default=FakeOk({"findings": []}))
    run = run_main(tmp_path, factory=fake_factory(fake))
    assert run.code == (1 if fails else 0) and Closing.closed == 1


# ---------------------------------------------------------------- record and replay (AC-30 supporting)


def test_record_then_replay_reproduces_the_output_byte_for_byte(tmp_path) -> None:
    rec = tmp_path / "r.recording.json"
    seen: list[int] = []
    live = run_main(tmp_path, "--record", str(rec), factory=mock_anthropic_factory(seen=seen), environ=KEY_ENV)
    assert live.code == 0 and sorted(seen) == [0, 1, 2] and rec.exists()
    first = live.output.read_bytes()
    live.output.unlink()
    replay = run_main(tmp_path, "--replay", str(rec), data=None)  # no factory, no key: the Replay client
    assert replay.code == 0 and replay.output.read_bytes() == first
    assert "mode replay" in replay.err


def test_degraded_run_still_writes_the_recording_d4(tmp_path) -> None:
    rec = tmp_path / "r.recording.json"
    run = run_main(tmp_path, "--record", str(rec), factory=mock_anthropic_factory({1: 503}), environ=KEY_ENV)
    assert run.code == 3 and rec.exists()
    assert any(e["result"] == "failure" for e in json.loads(rec.read_text(encoding="utf-8"))["entries"])


def test_existing_recording_refused_before_any_call_d3(tmp_path) -> None:
    setup_files(tmp_path)
    rec = tmp_path / "r.recording.json"
    rec.write_text("OLD", encoding="utf-8")
    seen: list[int] = []
    run = run_main(tmp_path, "--record", str(rec), factory=mock_anthropic_factory(seen=seen), environ=KEY_ENV,
                   data=None)
    assert run.code == 2 and seen == [] and rec.read_text(encoding="utf-8") == "OLD"


def test_fatal_pipeline_failure_writes_no_recording_d4(tmp_path) -> None:
    rec = tmp_path / "r.recording.json"
    fake = FakeProofreader({0: [FakeRaise(llm.ProviderFatalError("HTTP 401"))]}, default=FakeOk({"findings": []}))
    run = run_main(tmp_path, "--record", str(rec), "--metrics-json", str(tmp_path / "m.json"),
                   factory=fake_factory(fake))
    assert run.code == 1 and not rec.exists() and not (tmp_path / "m.json").exists() and not run.output.exists()


def test_fatal_pipeline_failure_still_publishes_the_log_file(tmp_path) -> None:
    fake = FakeProofreader({0: [FakeRaise(llm.ProviderFatalError("HTTP 401"))]}, default=FakeOk({"findings": []}))
    run = run_main(tmp_path, "--log-file", str(tmp_path / "run.log"), factory=fake_factory(fake))
    assert run.code == 1 and not run.output.exists()
    log_text = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "HTTP 401" in log_text and "no corrected output was written" in log_text


# ---------------------------------------------------------------- publication order and failure semantics


def test_publication_order_is_output_recording_metrics_log(tmp_path, monkeypatch) -> None:
    order: list[str] = []
    real_write, real_record = cli.write_atomic, llm.RecordingWriter.write
    monkeypatch.setattr(cli, "write_atomic", lambda p, d, **k: (order.append(k["label"]), real_write(p, d, **k))[1])
    monkeypatch.setattr(llm.RecordingWriter, "write",
                        lambda self, p, **k: (order.append("--record"), real_record(self, p, **k))[1])
    run = run_main(tmp_path, "--record", str(tmp_path / "r.recording.json"), "--metrics-json",
                   str(tmp_path / "m.json"), "--log-file", str(tmp_path / "run.log"),
                   factory=mock_anthropic_factory(), environ=KEY_ENV)
    assert run.code == 0
    assert order == ["corrected output", "--record", "--metrics-json", "--log-file"]


def test_output_publication_failure_writes_no_recording_or_metrics(tmp_path, monkeypatch) -> None:
    real_write = cli.write_atomic

    def failing(path, data, **kwargs):
        if kwargs["label"] == "corrected output":
            raise cli.OutputError("corrected output: writing failed; nothing was published")
        return real_write(path, data, **kwargs)

    monkeypatch.setattr(cli, "write_atomic", failing)
    run = run_main(tmp_path, "--record", str(tmp_path / "r.recording.json"), "--metrics-json",
                   str(tmp_path / "m.json"), "--log-file", str(tmp_path / "run.log"),
                   factory=mock_anthropic_factory(), environ=KEY_ENV)
    assert run.code == 1 and not run.output.exists()
    assert not (tmp_path / "r.recording.json").exists() and not (tmp_path / "m.json").exists()
    assert "no corrected output was written" in (tmp_path / "run.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("failing_label", ["--record", "--metrics-json", "--log-file"])
def test_auxiliary_failure_keeps_the_output_continues_and_exits_3(tmp_path, monkeypatch, failing_label) -> None:
    real_write, real_record = cli.write_atomic, llm.RecordingWriter.write

    def write(path, data, **kwargs):
        if kwargs["label"] == failing_label:
            raise cli.OutputError(f"{failing_label}: disk full")
        return real_write(path, data, **kwargs)

    def record(self, path, **kwargs):
        if failing_label == "--record":
            raise llm.RecordingError("disk full")
        return real_record(self, path, **kwargs)

    monkeypatch.setattr(cli, "write_atomic", write)
    monkeypatch.setattr(llm.RecordingWriter, "write", record)
    paths = {"--record": tmp_path / "r.recording.json", "--metrics-json": tmp_path / "m.json",
             "--log-file": tmp_path / "run.log"}
    run = run_main(tmp_path, *[x for k, v in paths.items() for x in (k, str(v))], factory=mock_anthropic_factory(),
                   environ=KEY_ENV)
    assert run.code == 3
    assert run.output.exists()  # never deleted or rolled back
    if failing_label == "--log-file":
        assert "--log-file could not be written" in run.err
    else:
        assert f"corrected output written: doc.corrected.xml; {failing_label} could not be written" in run.err
    for label, path in paths.items():
        assert path.exists() == (label != failing_label)  # the remaining artifacts were still attempted
    if failing_label == "--record":
        metrics_data = json.loads(paths["--metrics-json"].read_text())
        assert metrics_data["auxiliary_failures"] == ["--record"] and metrics_data["exit_code"] == 3


# ---------------------------------------------------------------- interrupts and internal errors


def test_keyboard_interrupt_exits_130_and_writes_nothing(tmp_path) -> None:
    fake = FakeProofreader({0: [FakeRaise(KeyboardInterrupt())]}, default=FakeOk({"findings": []}))
    run = run_main(tmp_path, factory=fake_factory(fake))
    assert run.code == 130 and not run.output.exists() and "interrupted" in run.err


def test_keyboard_interrupt_after_publication_reports_the_written_output(tmp_path, monkeypatch) -> None:
    def interrupt(self, path, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(llm.RecordingWriter, "write", interrupt)
    run = run_main(tmp_path, "--record", str(tmp_path / "r.recording.json"), factory=mock_anthropic_factory(),
                   environ=KEY_ENV)
    assert run.code == 130 and run.output.exists()
    assert "interrupted after the corrected output was written: doc.corrected.xml" in run.err


def test_internal_error_exits_1_without_its_message(tmp_path) -> None:
    fake = FakeProofreader({0: [FakeRaise(RuntimeError(f"leaks {WORD}"))]}, default=FakeOk({"findings": []}))
    run = run_main(tmp_path, "--log-level", "DEBUG", factory=fake_factory(fake))
    assert run.code == 1 and "internal error (RuntimeError)" in run.err
    assert WORD not in run.err and "traceback (frames only)" in run.err and not run.output.exists()


# ---------------------------------------------------------------- logging: redaction, no text, SDK pinning


def test_api_key_never_appears_anywhere_ac26(tmp_path) -> None:
    def factory(cfg, recorder):
        logging.getLogger(cli.LOGGER_NAME).warning(f"probe {cfg.api_key.reveal()}")  # a careless log line
        return mock_anthropic_factory()(cfg, recorder)

    run = run_main(tmp_path, "--record", str(tmp_path / "r.recording.json"), "--metrics-json",
                   str(tmp_path / "m.json"), "--log-file", str(tmp_path / "run.log"), "--log-level", "DEBUG",
                   factory=factory, environ=KEY_ENV)
    assert run.code == 0 and "probe ***" in run.err
    blobs = [run.out, run.err] + [p.read_bytes().decode("utf-8", "replace") for p in tmp_path.iterdir() if p.is_file()]
    assert all(DUMMY_KEY not in blob and "SENTINELKEY42" not in blob for blob in blobs)


def test_document_text_never_logged_at_any_level_d2(tmp_path) -> None:
    run = run_main(tmp_path, "--metrics-json", str(tmp_path / "m.json"), "--log-file", str(tmp_path / "run.log"),
                   "--log-level", "DEBUG", factory=fake_factory(ok_fake()))
    assert run.code == 0
    texts = [run.err, (tmp_path / "run.log").read_text(encoding="utf-8"), (tmp_path / "m.json").read_text()]
    for forbidden in (WORD, "recieve", "receive", "per cent", "House rule", "Write per cent"):
        assert all(forbidden not in t for t in texts), forbidden


def test_sdk_loggers_pinned_to_warning_during_the_run_and_restored(tmp_path) -> None:
    sdk = logging.getLogger("anthropic")
    saved = sdk.level
    sdk.setLevel(logging.DEBUG)  # as ANTHROPIC_LOG=debug would at SDK import
    levels: list[int] = []
    try:
        def factory(cfg, recorder):
            levels.append(logging.getLogger("anthropic._base_client").getEffectiveLevel())
            return mock_anthropic_factory()(cfg, recorder)

        run = run_main(tmp_path, "--log-level", "DEBUG", factory=factory, environ=KEY_ENV)
        # the S4.5 DEBUG guard would have stopped the run if the SDK logger were still at DEBUG
        assert run.code == 0 and levels == [logging.WARNING]
        assert sdk.level == logging.DEBUG  # restored afterwards
    finally:
        sdk.setLevel(saved)


def test_log_file_is_published_last_with_the_exit_code(tmp_path) -> None:
    run = run_main(tmp_path, "--log-file", str(tmp_path / "run.log"), factory=fake_factory(ok_fake()))
    log_text = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert run.code == 0 and log_text.rstrip().endswith("INFO exit code 0")
    assert "corrected output written: doc.corrected.xml" in log_text


# ---------------------------------------------------------------- metrics (AC-17)


@pytest.mark.parametrize("heap_flag,expect_heap", [("--heap-metrics", True), ("--no-heap-metrics", False)])
def test_metrics_json_contents_ac17(tmp_path, heap_flag, expect_heap) -> None:
    run = run_main(tmp_path, "--metrics-json", str(tmp_path / "m.json"), heap_flag, factory=mock_anthropic_factory(),
                   environ=KEY_ENV)
    m = json.loads((tmp_path / "m.json").read_text())
    assert run.code == 0 and m["exit_code"] == 0
    assert {"total_ms", "stages_ms", "pipeline_stages_ms", "latency_ms", "memory", "tokens", "requests_sent",
            "findings", "paragraphs"} <= set(m)
    assert {"read_input", "load_style_guide", "pipeline", "write_output"} <= set(m["stages_ms"])
    assert set(m["latency_ms"]["by_paragraph"]) == {"0", "1", "2"}
    assert m["tokens"] == {"input": 30, "output": 15, "cache_creation_input": 0, "cache_read_input": 0}
    assert m["requests_sent"] == 3 and m["findings"]["by_status"] == {"APPLIED": 2}
    assert m["memory"]["heap_tracing"] is expect_heap
    assert isinstance(m["memory"]["peak_heap_bytes"], int) is expect_heap
    assert m["memory"]["peak_heap_bytes"] == "n/a" or m["memory"]["peak_heap_bytes"] > 0
    assert m["memory"]["peak_process_bytes"] == "n/a" or m["memory"]["peak_process_bytes"] > 0
    assert m["input"] == "doc.xml" and m["output"] == "doc.corrected.xml"  # names only, never directories
    assert str(tmp_path) not in (tmp_path / "m.json").read_text()


def test_heap_is_na_when_disabled_even_if_something_else_traces(tmp_path) -> None:
    import tracemalloc

    tracemalloc.start()
    try:
        run = run_main(tmp_path, "--metrics-json", str(tmp_path / "m.json"), "--no-heap-metrics",
                       factory=fake_factory(ok_fake()))
        memory = json.loads((tmp_path / "m.json").read_text())["memory"]
        assert run.code == 0 and memory["heap_tracing"] is False and memory["peak_heap_bytes"] == "n/a"
        assert tracemalloc.is_tracing()  # tracing we did not start is left running
    finally:
        tracemalloc.stop()


def test_process_memory_is_reported_separately_or_na() -> None:
    from proofreader import metrics

    value = metrics.peak_process_memory_bytes()
    assert value is None or value > 0


# ---------------------------------------------------------------- subprocess exit codes (AC-32)


def subprocess_run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "PROOFREADER_"))}
    env["PYTHON_DOTENV_DISABLED"] = "1"
    return subprocess.run([sys.executable, "-I", "-B", "-X", "utf8", "-m", "proofreader", *args], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=120)


def test_exit_codes_through_python_m_proofreader_ac32(tmp_path) -> None:
    rec_ok, rec_bad = tmp_path / "ok.recording.json", tmp_path / "bad.recording.json"
    assert run_main(tmp_path, "--record", str(rec_ok), factory=mock_anthropic_factory(), environ=KEY_ENV).code == 0
    (tmp_path / "doc.corrected.xml").unlink()
    assert run_main(tmp_path, "--record", str(rec_bad), factory=mock_anthropic_factory({2: 500}),
                    environ=KEY_ENV, data=None).code == 3
    (tmp_path / "doc.corrected.xml").unlink()
    common = ["doc.xml", "--lang", "en", "--style-guide", "g.docx"]
    ok = subprocess_run(tmp_path, *common, "--replay", rec_ok.name)
    assert ok.returncode == 0, ok.stderr
    (tmp_path / "doc.corrected.xml").unlink()
    degraded = subprocess_run(tmp_path, *common, "--replay", rec_bad.name)
    assert degraded.returncode == 3, degraded.stderr
    assert subprocess_run(tmp_path, *common, "--replay", rec_bad.name).returncode == 2  # output exists now
    (tmp_path / "broken.xml").write_bytes(b"<doc><p>x</doc>")
    assert subprocess_run(tmp_path, "broken.xml", "--lang", "en", "--style-guide", "g.docx",
                          "--replay", rec_ok.name).returncode == 1
    assert subprocess_run(tmp_path, *common[:1], "--lang", "!!", "--style-guide", "g.docx").returncode == 2
    assert subprocess_run(tmp_path).returncode == 2  # argparse usage error


# ---------------------------------------------------------------- B21 runtime summary in the default log (S4.8e)

import re  # noqa: E402

SUMMARY_RE = re.compile(
    r"INFO run time \d+\.\d{3} s \(read_input \d+\.\d{3} s, load_style_guide \d+\.\d{3} s, pipeline \d+\.\d{3} s "
    r"\[prepare \d+\.\d{3} s, calls \d+\.\d{3} s, resolve \d+\.\d{3} s, annotate \d+\.\d{3} s\], "
    r"write_output \d+\.\d{3} s\); peak heap (?P<heap>\d+\.\d MiB|n/a); peak process memory (?P<rss>\d+\.\d MiB|n/a)$")


def summary_lines(err: str) -> list[str]:
    return [ln for ln in err.splitlines() if ln.startswith("INFO run time ")]


def test_default_log_reports_run_time_stages_and_memory_b21(tmp_path) -> None:
    run = run_main(tmp_path, factory=fake_factory(ok_fake()))  # no --metrics-json: the default log alone
    assert run.code == 0
    (line,) = summary_lines(run.err)
    match = SUMMARY_RE.match(line)
    assert match, line
    assert match["heap"] != "n/a"  # heap tracing is automatic for small inputs
    lines = run.err.splitlines()
    assert lines.index(line) < lines.index("INFO exit code 0")  # logged before the run ends (and in --log-file)


def test_runtime_summary_reports_na_without_heap_tracing_b21(tmp_path) -> None:
    run = run_main(tmp_path, "--no-heap-metrics", factory=fake_factory(ok_fake()))
    (line,) = summary_lines(run.err)
    assert SUMMARY_RE.match(line)["heap"] == "n/a"


def test_runtime_summary_also_on_degraded_runs_and_in_the_log_file_b21(tmp_path) -> None:
    fake = FakeProofreader({0: [FakeOk(FINDINGS[0])], 1: [FakeFailure(FailureClass.TIMEOUT)], 2: [FakeOk(FINDINGS[2])]})
    run = run_main(tmp_path, "--log-file", str(tmp_path / "run.log"), factory=fake_factory(fake))
    assert run.code == 3 and len(summary_lines(run.err)) == 1
    assert len(summary_lines((tmp_path / "run.log").read_text(encoding="utf-8"))) == 1


def test_runtime_summary_formatting_exact_b21() -> None:
    summary = {"total_ms": 1234.5, "stages_ms": {"read_input": 1.0, "pipeline": 1200.25},
               "pipeline_stages_ms": {"calls": 1100.0}, "memory": {"peak_heap_bytes": 3 * 1024 * 1024 // 2,
                                                                  "peak_process_bytes": "n/a"}}
    assert cli._runtime_summary(summary) == ("run time 1.234 s (read_input 0.001 s, pipeline 1.200 s [calls 1.100 s]); "
                                             "peak heap 1.5 MiB; peak process memory n/a")
    summary["pipeline_stages_ms"] = {}
    summary["memory"] = {"peak_heap_bytes": "n/a", "peak_process_bytes": 64 * 1024 * 1024}
    assert cli._runtime_summary(summary).endswith("pipeline 1.200 s); peak heap n/a; peak process memory 64.0 MiB")


def test_no_runtime_summary_when_no_output_was_produced_b21(tmp_path) -> None:
    run = run_main(tmp_path, data=b"<doc><p>unclosed</doc>", factory=fake_factory(ok_fake()))
    assert run.code == 1 and summary_lines(run.err) == []


# ============================================================================================
# S5: Gemini provider through the command line (real GeminiProofreader over an in-memory transport)
# ============================================================================================

from proofreader import gemini  # noqa: E402

GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_KEY = "dummy-gemini-key-not-real-SENTINELGEM7"
GEMINI_ENV = {**NO_DOTENV, "GEMINI_API_KEY": GEMINI_KEY}


def mock_gemini_factory(statuses: dict[int, int] | None = None, seen: list | None = None):
    """The REAL GeminiProofreader over an in-memory transport (no network); same findings as the Anthropic mock."""
    statuses = statuses or {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.host == "generativelanguage.googleapis.com" and GEMINI_KEY not in str(request.url)
        body = json.loads(request.content)
        user = json.loads(body["contents"][0]["parts"][0]["text"])
        text = "".join(user["paragraph"]["segments"])
        pid = next(i for i, word in ((0, "recieve"), (1, "Nothing"), (2, "Sales")) if word in text)
        if seen is not None:
            seen.append(pid)
        if statuses.get(pid):
            return httpx2.Response(statuses[pid], json={"error": {"code": statuses[pid], "message": "m"}})
        return httpx2.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(FINDINGS[pid])}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}, "responseId": f"resp_{pid}"})

    def factory(cfg, recorder):
        http = httpx2.Client(transport=httpx2.MockTransport(handler), trust_env=False)
        return gemini.GeminiProofreader(cfg, http_client=http, recorder=recorder, sleep=lambda s: None)

    return factory


class _StubSession:
    def __init__(self):
        self.pinned = 0

    def pin_sdk_loggers(self):
        self.pinned += 1


def test_live_client_choice_follows_the_model_id(tmp_path) -> None:
    gem_cfg = c.build_config({"input": "a.xml", "lang": "en", "style_guide": "g.docx", "model": GEMINI_MODEL},
                             GEMINI_ENV, tmp_path)
    session = _StubSession()
    client, mode = cli._choose_client(gem_cfg, None, None, session)
    try:
        assert isinstance(client, gemini.GeminiProofreader) and mode == "live" and session.pinned == 1
    finally:
        client.close()
    anth_cfg = c.build_config({"input": "a.xml", "lang": "en", "style_guide": "g.docx"}, KEY_ENV, tmp_path)
    client, mode = cli._choose_client(anth_cfg, None, None, _StubSession())
    try:
        assert isinstance(client, llm.AnthropicProofreader) and mode == "live"
    finally:
        client.close()


def test_replay_takes_precedence_over_the_gemini_client(tmp_path) -> None:
    rec = tmp_path / "r.recording.json"
    rec.write_text(llm.RecordingWriter().render(), encoding="utf-8")
    cfg = c.build_config({"input": "a.xml", "lang": "en", "style_guide": "g.docx", "model": GEMINI_MODEL,
                          "replay": str(rec)}, NO_DOTENV, tmp_path)  # no key needed
    client, mode = cli._choose_client(cfg, None, None, _StubSession())
    assert isinstance(client, llm.ReplayProofreader) and mode == "replay"


def test_gemini_run_produces_the_same_output_as_the_anthropic_run(tmp_path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "g").mkdir()
    anth = run_main(tmp_path / "a", factory=mock_anthropic_factory(), environ=KEY_ENV)
    gem = run_main(tmp_path / "g", "--model", GEMINI_MODEL, factory=mock_gemini_factory(), environ=GEMINI_ENV)
    assert anth.code == 0 == gem.code, gem.err
    assert gem.output.read_bytes() == anth.output.read_bytes()
    assert b"<error" in gem.output.read_bytes()


def test_gemini_record_then_replay_reproduces_the_output_byte_for_byte(tmp_path) -> None:
    rec = tmp_path / "r.recording.json"
    seen: list[int] = []
    live = run_main(tmp_path, "--model", GEMINI_MODEL, "--record", str(rec), factory=mock_gemini_factory(seen=seen),
                    environ=GEMINI_ENV)
    assert live.code == 0 and sorted(seen) == [0, 1, 2] and rec.exists()
    first = live.output.read_bytes()
    live.output.unlink()
    replay = run_main(tmp_path, "--model", GEMINI_MODEL, "--replay", str(rec), data=None)  # no key, no network
    assert replay.code == 0 and replay.output.read_bytes() == first and "mode replay" in replay.err
    assert GEMINI_KEY not in rec.read_text(encoding="utf-8")


def test_gemini_degraded_paragraph_exits_3_and_leaves_it_unchanged(tmp_path) -> None:
    run = run_main(tmp_path, "--model", GEMINI_MODEL, factory=mock_gemini_factory({1: 503}), environ=GEMINI_ENV)
    assert run.code == 3 and run.output.exists()


def test_gemini_run_fatal_status_exits_1_without_output(tmp_path) -> None:
    run = run_main(tmp_path, "--model", GEMINI_MODEL, factory=mock_gemini_factory({0: 403, 1: 403, 2: 403}),
                   environ=GEMINI_ENV)
    assert run.code == 1 and not run.output.exists() and "HTTP 403" in run.err and GEMINI_KEY not in run.err


def test_missing_gemini_key_is_fatal_before_any_request(tmp_path) -> None:
    run = run_main(tmp_path, "--model", GEMINI_MODEL, factory=None, environ={**NO_DOTENV, "ANTHROPIC_API_KEY": DUMMY_KEY})
    assert run.code == 1 and "GEMINI_API_KEY is not set" in run.err and not run.output.exists()


def test_gemini_key_and_text_never_reach_the_logs(tmp_path) -> None:
    run = run_main(tmp_path, "--model", GEMINI_MODEL, "--log-level", "DEBUG", factory=mock_gemini_factory(),
                   environ=GEMINI_ENV)
    assert run.code == 0 and GEMINI_KEY not in run.err + run.out and WORD not in run.err


@pytest.mark.parametrize("model, expected", [("Gemini-2.5-flash-lite", "ambiguous"),
                                             ("gemini-2.5-", "Gemini model id is invalid")])
def test_ambiguous_or_invalid_gemini_ids_exit_2_before_any_request(tmp_path, model, expected) -> None:
    seen: list[int] = []
    run = run_main(tmp_path, "--model", model, factory=mock_gemini_factory(seen=seen), environ=GEMINI_ENV)
    assert run.code == 2 and expected in run.err and seen == [] and not run.output.exists()

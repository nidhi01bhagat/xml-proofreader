"""S4.4: Proofreader protocol, immutable requests, FakeProofreader, ReplayProofreader, recordings.

Offline only: no network, no API key, no private files. All documents and payloads are invented.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import inspect
import json
import subprocess
import threading
import typing
from pathlib import Path

import pytest

from proofreader import config as c
from proofreader import llm
from proofreader import prompt as pr
from proofreader import resolver as r
from proofreader import xml_source as xs
from proofreader.llm import (CallRecord, CallStatus, FailureClass, FakeBlock, FakeFailure, FakeOk, FakeProofreader,
                             FakeRaise, FakeScriptError, ProofreadRequest, ProofreadResult, RecordingError,
                             RecordingWriter, ReplayMissError, ReplayProofreader, Usage)
from proofreader.models import ProofreaderError, RejectCode

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_SENTINEL = "SYSTEM-SENTINEL-TEXT"
USER_SENTINEL = "USER-SENTINEL-PARAGRAPH"
PAYLOAD_SENTINEL = "PAYLOAD-SENTINEL-FINDING"
SCHEMA = {"type": "object", "properties": {"findings": {"type": "array"}}, "required": ["findings"]}
GOOD = {"findings": [{"original": "recieve", "before": "Our team will", "after": "the final report",
                      "word_start": 3, "word_end": 3, "correction": "receive", "type": "spelling",
                      "reason": "Misspelling"}]}


@pytest.fixture
def cfg(tmp_path) -> c.RunConfig:
    return c.build_config({"input": "a.xml", "lang": "en", "style_guide": "g.docx"}, {}, tmp_path)


def request(cfg, paragraph_id: int = 0, user: str = USER_SENTINEL, schema: dict | None = None) -> ProofreadRequest:
    return llm.make_request(cfg, paragraph_id, system=SYSTEM_SENTINEL, user=user,
                            schema=SCHEMA if schema is None else schema)


def ok(payload=None, **kw) -> ProofreadResult:
    return ProofreadResult(GOOD if payload is None else payload, None, Usage(5, 7), 12.5, 1, "req_1", "end_turn", **kw)


# --------------------------------------------------------------------------------------------
# Protocol conformance (runtime check AND real signature inspection)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("client", [FakeProofreader(), ReplayProofreader({})])
def test_clients_conform_to_the_protocol_at_runtime(client) -> None:
    assert isinstance(client, llm.Proofreader)


@pytest.mark.parametrize("cls", [FakeProofreader, ReplayProofreader])
def test_proofread_signature_is_inspected_not_assumed(cls) -> None:
    assert list(inspect.signature(cls.proofread).parameters) == ["self", "request"]
    hints = typing.get_type_hints(cls.proofread)
    assert hints == {"request": ProofreadRequest, "return": ProofreadResult}
    assert typing.get_type_hints(llm.Proofreader.proofread) == hints


def test_runtime_checkable_does_not_check_signatures_negative_control() -> None:
    class WrongSignature:
        def proofread(self):  # no request parameter
            return None

    assert isinstance(WrongSignature(), llm.Proofreader)  # runtime_checkable only sees the attribute
    assert list(inspect.signature(WrongSignature.proofread).parameters) == ["self"]  # inspection catches it


# --------------------------------------------------------------------------------------------
# Immutable, verified requests
# --------------------------------------------------------------------------------------------


def test_make_request_matches_the_s43_fingerprint(cfg) -> None:
    req = request(cfg)
    assert req.fingerprint == pr.fingerprint_for(cfg, system=SYSTEM_SENTINEL, user=USER_SENTINEL, schema=SCHEMA)
    assert (req.model, req.effort, req.max_tokens) == (cfg.model, "low", 8000)


def test_mutating_the_schema_after_construction_changes_nothing(cfg) -> None:
    schema = json.loads(json.dumps(SCHEMA))
    req = request(cfg, schema=schema)
    fingerprint = req.fingerprint
    schema["properties"]["findings"]["type"] = "string"
    schema["extra"] = 1
    assert req.schema == SCHEMA and req.fingerprint == fingerprint
    copy_one = req.schema
    copy_one["required"].append("x")
    assert req.schema == SCHEMA and req.schema is not req.schema


def test_requests_are_frozen(cfg) -> None:
    req = request(cfg)
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.user = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "change",
    [
        {"fingerprint": "0" * 64}, {"fingerprint": "A" * 64}, {"fingerprint": "abc"},
        {"user": "tampered user"}, {"schema_json": '{"type": "object"}'}, {"schema_json": "[]"},
        {"schema_json": "not json"}, {"paragraph_id": True}, {"paragraph_id": -1}, {"max_tokens": True},
        {"max_tokens": 0}, {"model": ""}, {"effort": ""}, {"system": None},
    ],
)
def test_tampered_or_invalid_requests_are_rejected(cfg, change) -> None:
    req = request(cfg)
    fields = {f.name: getattr(req, f.name) for f in dataclasses.fields(req)}
    with pytest.raises(ValueError):
        ProofreadRequest(**{**fields, **change})


def test_non_canonical_schema_json_is_rejected_even_with_a_matching_fingerprint(cfg) -> None:
    req = request(cfg)
    fields = {f.name: getattr(req, f.name) for f in dataclasses.fields(req)}
    pretty = json.dumps(SCHEMA, indent=1)  # same content, correct fingerprint, but not the canonical form
    assert json.loads(pretty) == req.schema
    with pytest.raises(ValueError, match="canonical"):
        ProofreadRequest(**{**fields, "schema_json": pretty})


def test_make_request_rejects_unsafe_schemas(cfg) -> None:
    for schema in ([], {"a": float("nan")}, {"a": (1, 2)}, {1: "x"}):
        with pytest.raises(ValueError):
            request(cfg, schema=schema)  # type: ignore[arg-type]


def test_request_repr_is_redacted(cfg) -> None:
    text = repr(request(cfg))
    assert SYSTEM_SENTINEL not in text and USER_SENTINEL not in text and "findings" not in text
    assert "paragraph_id=0" in text and len(request(cfg).fingerprint[:12]) == 12


# --------------------------------------------------------------------------------------------
# Error hierarchy
# --------------------------------------------------------------------------------------------


def test_error_hierarchy() -> None:
    for cls in (llm.ProviderFatalError, RecordingError, ReplayMissError):
        assert issubclass(cls, llm.RunFatalError) and issubclass(cls, ProofreaderError)
        assert cls("x").exit_code == 1
    assert issubclass(ReplayMissError, RecordingError)
    assert issubclass(FakeScriptError, AssertionError) and not issubclass(FakeScriptError, ProofreaderError)


# --------------------------------------------------------------------------------------------
# FakeProofreader
# --------------------------------------------------------------------------------------------


def test_fake_success_and_clean_paragraph(cfg) -> None:
    fake = FakeProofreader({0: [FakeOk(GOOD)], 1: [FakeOk({"findings": []})]})
    first, second = fake.proofread(request(cfg, 0)), fake.proofread(request(cfg, 1))
    assert first.payload == GOOD and first.failure is None and first.attempts == 1
    assert second.payload == {"findings": []}
    assert CallRecord.from_result(0, first).status is CallStatus.OK


@pytest.mark.parametrize("failure", list(FailureClass))
def test_every_failure_class_is_returned_not_raised(cfg, failure: FailureClass) -> None:
    fake = FakeProofreader({0: [FakeFailure(failure, stop_reason="refusal" if failure is FailureClass.REFUSAL
                                            else None)]})
    result = fake.proofread(request(cfg))
    assert result.failure is failure
    record = CallRecord.from_result(0, result)
    assert record.status is CallStatus.PASSTHROUGH and record.failure is failure


def test_schema_invalid_with_and_without_a_payload(cfg) -> None:
    fake = FakeProofreader({0: [FakeFailure(FailureClass.SCHEMA_INVALID, payload=["not", "an", "object"],
                                            attempts=2), FakeFailure(FailureClass.SCHEMA_INVALID)]})
    first, second = fake.proofread(request(cfg)), fake.proofread(request(cfg))
    assert (first.payload, first.attempts) == (["not", "an", "object"], 2) and second.payload is None


def test_fake_is_strict_about_exhausted_and_unscripted_calls(cfg) -> None:
    fake = FakeProofreader({0: [FakeOk(GOOD)]})
    fake.proofread(request(cfg, 0))
    with pytest.raises(FakeScriptError, match=r"paragraph 0: unexpected call 2 \(script has 1\)"):
        fake.proofread(request(cfg, 0))
    with pytest.raises(FakeScriptError, match="paragraph 5: unexpected call"):
        fake.proofread(request(cfg, 5))


def test_strict_fake_catches_an_accidental_duplicate_call(cfg) -> None:
    fake = FakeProofreader({0: [FakeOk(GOOD)], 1: [FakeOk(GOOD)]})
    paragraphs = [0, 1, 1]  # a buggy loop that submits paragraph 1 twice
    with pytest.raises(FakeScriptError, match="paragraph 1"):
        for pid in paragraphs:
            fake.proofread(request(cfg, pid))


def test_repeat_last_and_default_are_explicit_opt_ins(cfg) -> None:
    fake = FakeProofreader({0: [FakeOk(GOOD)]}, repeat_last=True, default=FakeOk({"findings": []}))
    assert [fake.proofread(request(cfg, 0)).payload for _ in range(3)] == [GOOD] * 3
    assert fake.proofread(request(cfg, 9)).payload == {"findings": []}
    assert fake.call_counts() == {0: 3, 9: 1}
    assert [(call.paragraph_id, call.sequence_index) for call in fake.calls] == [(0, 0), (0, 1), (0, 2), (9, 0)]


def test_fake_call_log_holds_no_text(cfg) -> None:
    fake = FakeProofreader(default=FakeOk(GOOD))
    fake.proofread(request(cfg))
    assert [f.name for f in dataclasses.fields(llm.FakeCall)] == ["paragraph_id", "fingerprint", "sequence_index"]
    assert USER_SENTINEL not in repr(fake.calls) and SYSTEM_SENTINEL not in repr(fake) and "recieve" not in repr(fake)


def test_scripted_payload_is_protected_from_caller_mutation(cfg) -> None:
    payload = {"findings": [{"original": PAYLOAD_SENTINEL}]}
    fake = FakeProofreader({0: [FakeOk(payload)]}, repeat_last=True)
    payload["findings"].clear()  # caller mutates its own dict after scripting
    first = fake.proofread(request(cfg))
    assert first.payload == {"findings": [{"original": PAYLOAD_SENTINEL}]}
    first.payload["findings"].clear()  # caller mutates a returned payload
    assert fake.proofread(request(cfg)).payload == {"findings": [{"original": PAYLOAD_SENTINEL}]}


@pytest.mark.parametrize("exc", [llm.ProviderFatalError("auth"), KeyboardInterrupt(),
                                 concurrent.futures.CancelledError(), RuntimeError("boom")])
def test_raised_outcomes_propagate_and_counters_recover(cfg, exc) -> None:
    fake = FakeProofreader({0: [FakeRaise(exc)]})
    with pytest.raises(type(exc)):
        fake.proofread(request(cfg))
    assert fake.in_flight == 0 and fake.max_in_flight == 1


def test_fake_block_counts_in_flight_calls_and_releases_reliably(cfg) -> None:
    release = threading.Event()
    started = [threading.Event() for _ in range(3)]
    fake = FakeProofreader({i: [FakeBlock(release, FakeOk(GOOD), timeout_s=5.0, started=started[i])]
                            for i in range(3)})
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fake.proofread, request(cfg, i)) for i in range(3)]
        try:
            for event in started:
                assert event.wait(5.0), "a call did not start"
            assert fake.in_flight == 3 and fake.max_in_flight == 3
        finally:
            release.set()
        results = [future.result(timeout=5.0) for future in futures]
    assert all(result.payload == GOOD for result in results)
    assert fake.in_flight == 0


def test_unreleased_block_times_out_instead_of_hanging(cfg) -> None:
    fake = FakeProofreader({0: [FakeBlock(threading.Event(), FakeOk(GOOD), timeout_s=0.05)]})
    with pytest.raises(FakeScriptError, match="not released"):
        fake.proofread(request(cfg))
    assert fake.in_flight == 0


def test_fake_block_then_can_raise_and_counters_recover(cfg) -> None:
    release = threading.Event()
    release.set()
    fake = FakeProofreader({0: [FakeBlock(release, FakeRaise(llm.ProviderFatalError("x")))]})
    with pytest.raises(llm.ProviderFatalError):
        fake.proofread(request(cfg))
    assert fake.in_flight == 0


def test_fake_is_deterministic_and_validates_its_script(cfg) -> None:
    script = {0: [FakeOk(GOOD), FakeFailure(FailureClass.TIMEOUT)]}
    runs = []
    for _ in range(2):
        fake = FakeProofreader(script, latency_ms=3.0, usage=Usage(2, 3))
        runs.append([fake.proofread(request(cfg)) for _ in range(2)])
    assert runs[0] == runs[1]
    for bad in ({True: [FakeOk(GOOD)]}, {0: []}, {0: ["not an outcome"]}, {-1: [FakeOk(GOOD)]}):
        with pytest.raises(ValueError):
            FakeProofreader(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        FakeFailure("timeout")  # type: ignore[arg-type]


def test_fake_payload_is_resolved_by_the_authoritative_resolver(cfg) -> None:
    scan = xs.scan(b"<doc><p>Our team will recieve the final report on Monday.</p></doc>")
    fake = FakeProofreader({0: [FakeOk(GOOD)], 1: [FakeOk({"findings": "not a list"})]})
    (resolved,) = r.resolve_document(scan, {0: fake.proofread(request(cfg, 0)).payload})
    assert resolved.rejections == () and resolved.resolved[0].annotation.correction == "receive"
    (rejected,) = r.resolve_document(scan, {0: fake.proofread(request(cfg, 1)).payload})
    assert [x.code for x in rejected.rejections] == [RejectCode.SCHEMA_INVALID]


# --------------------------------------------------------------------------------------------
# RecordingWriter
# --------------------------------------------------------------------------------------------


def _fp(n: int) -> str:
    return f"{n:064x}"


def test_writer_round_trip_and_deterministic_bytes(tmp_path) -> None:
    results = {_fp(1): ok(), _fp(2): ProofreadResult(None, FailureClass.REFUSAL, Usage(1), 0.0, 1, None, "refusal"),
               _fp(3): ProofreadResult(["bad"], FailureClass.SCHEMA_INVALID, Usage(), 1, 2, "req-3", None)}
    a, b = RecordingWriter(), RecordingWriter()
    for fp, result in results.items():
        a.add(fp, result)
    for fp, result in reversed(list(results.items())):
        b.add(fp, result)
    assert a.render() == b.render()
    target = tmp_path / "run.recording.json"
    a.write(target)
    assert target.read_bytes() == a.render().encode("utf-8") and target.read_text(encoding="utf-8").endswith("\n")
    assert llm.load_recording(target) == results


def test_writer_serialises_only_the_whitelisted_fields(tmp_path, cfg) -> None:
    req = request(cfg)
    writer = RecordingWriter()
    writer.add(req.fingerprint, ok())
    document = json.loads(writer.render())
    assert set(document) == {"format", "version", "fingerprint_version", "prompt_version", "entries"}
    assert set(document["entries"][0]) == {"fingerprint", "result", "failure", "payload", "usage", "attempts",
                                           "latency_ms", "request_id", "stop_reason"}
    text = writer.render()
    for forbidden in (SYSTEM_SENTINEL, USER_SENTINEL, "properties", cfg.model, "low", "8000"):
        assert forbidden not in text


def test_writer_snapshot_is_not_affected_by_later_payload_mutation() -> None:
    payload = {"findings": [{"original": "a"}]}
    writer = RecordingWriter()
    writer.add(_fp(1), ok(payload))
    payload["findings"].append({"original": "added later"})
    assert "added later" not in writer.render()


def test_writer_refuses_existing_targets_unless_forced(tmp_path) -> None:
    target = tmp_path / "run.recording.json"
    target.write_text("ORIGINAL", encoding="utf-8")
    writer = RecordingWriter()
    writer.add(_fp(1), ok())
    with pytest.raises(RecordingError, match="refusing to overwrite"):
        writer.write(target)
    assert target.read_text(encoding="utf-8") == "ORIGINAL"
    writer.write(target, force=True)
    assert llm.load_recording(target) == {_fp(1): ok()}
    assert list(tmp_path.glob(".tmp-*.recording.json")) == []


def test_writer_requires_the_git_ignored_suffix(tmp_path) -> None:
    writer = RecordingWriter()
    with pytest.raises(RecordingError, match=".recording.json"):
        writer.write(tmp_path / "run.json")
    assert list(tmp_path.iterdir()) == []


def test_writer_scans_for_secrets_without_echoing_them(tmp_path) -> None:
    secret = "sk-ant-" + "Z" * 24
    writer = RecordingWriter()
    writer.add(_fp(1), ok({"findings": [{"reason": secret}]}))
    target = tmp_path / "run.recording.json"
    with pytest.raises(RecordingError) as info:
        writer.write(target)
    assert secret not in str(info.value) and not target.exists()
    assert list(tmp_path.glob(".tmp-*.recording.json")) == []


def test_writer_rejects_duplicates_bad_fingerprints_and_unsafe_payloads() -> None:
    writer = RecordingWriter()
    writer.add(_fp(1), ok())
    with pytest.raises(RecordingError, match="duplicate"):
        writer.add(_fp(1), ok())
    for fp in ("A" * 64, "abc", 5):
        with pytest.raises(RecordingError):
            writer.add(fp, ok())  # type: ignore[arg-type]
    for payload in ({"x": float("nan")}, {"x": (1, 2)}, {1: "x"}):
        with pytest.raises(RecordingError, match="not JSON-safe"):
            writer.add(_fp(9), ok(payload))
    with pytest.raises(RecordingError):
        writer.add(_fp(8), "not a result")  # type: ignore[arg-type]


def test_writer_fails_closed_without_atomic_link_support(tmp_path, monkeypatch) -> None:
    def no_link(src, dst):
        raise OSError("hard links not supported")

    monkeypatch.setattr(llm.os, "link", no_link)
    writer = RecordingWriter()
    writer.add(_fp(1), ok())
    target = tmp_path / "run.recording.json"
    with pytest.raises(RecordingError, match="atomic exclusive publication"):
        writer.write(target)
    assert not target.exists() and list(tmp_path.glob(".tmp-*.recording.json")) == []


def test_writer_cleans_up_when_fsync_fails(tmp_path, monkeypatch) -> None:
    def broken_fsync(fd):
        raise OSError("disk error")

    monkeypatch.setattr(llm.os, "fsync", broken_fsync)
    writer = RecordingWriter()
    writer.add(_fp(1), ok())
    target = tmp_path / "run.recording.json"
    with pytest.raises(RecordingError, match="nothing was published"):
        writer.write(target)
    assert not target.exists() and list(tmp_path.glob(".tmp-*.recording.json")) == []


def test_temporary_file_is_next_to_the_target_and_git_ignored(tmp_path, monkeypatch) -> None:
    """A temp file left by a crash must not be visible to Git: its name matches *.recording.json."""
    seen = {}
    real_link = llm.os.link

    def spy_link(src, dst):
        seen["src"] = Path(src)
        assert Path(src).exists() and Path(src).read_bytes()  # complete before publication
        return real_link(src, dst)

    monkeypatch.setattr(llm.os, "link", spy_link)
    writer = RecordingWriter()
    writer.add(_fp(1), ok())
    writer.write(tmp_path / "run.recording.json")
    temp = seen["src"]
    assert temp.parent == tmp_path and temp.name.startswith(".tmp-") and temp.name.endswith(".recording.json")
    assert not temp.exists()  # removed after publication
    check = subprocess.run(["git", "check-ignore", "-q", "--no-index", temp.name], cwd=ROOT, capture_output=True)
    assert check.returncode == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.recording.json"]


def test_force_uses_atomic_replace_and_a_failed_replace_keeps_the_original(tmp_path, monkeypatch) -> None:
    target = tmp_path / "run.recording.json"
    target.write_text("ORIGINAL", encoding="utf-8")
    calls = []

    def no_link(src, dst):
        raise AssertionError("force=True must not use os.link")

    def failing_replace(src, dst):
        calls.append((Path(src).name, Path(dst).name))
        raise PermissionError("target locked")

    monkeypatch.setattr(llm.os, "link", no_link)
    monkeypatch.setattr(llm.os, "replace", failing_replace)
    writer = RecordingWriter()
    writer.add(_fp(1), ok())
    with pytest.raises(RecordingError, match="nothing was published"):
        writer.write(target, force=True)
    assert len(calls) == 1 and calls[0][1] == "run.recording.json"
    assert target.read_text(encoding="utf-8") == "ORIGINAL"  # no partial or replaced content
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.recording.json"]


def test_other_link_failures_fail_closed_without_a_destination(tmp_path, monkeypatch) -> None:
    def denied(src, dst):
        raise PermissionError("access denied")

    monkeypatch.setattr(llm.os, "link", denied)
    writer = RecordingWriter()
    writer.add(_fp(1), ok())
    with pytest.raises(RecordingError, match="nothing was written"):
        writer.write(tmp_path / "run.recording.json")
    assert list(tmp_path.iterdir()) == []


def test_interrupt_during_write_propagates_and_cleans_up(tmp_path, monkeypatch) -> None:
    def interrupted(fd):
        raise KeyboardInterrupt

    monkeypatch.setattr(llm.os, "fsync", interrupted)
    writer = RecordingWriter()
    writer.add(_fp(1), ok())
    with pytest.raises(KeyboardInterrupt):
        writer.write(tmp_path / "run.recording.json")
    assert list(tmp_path.iterdir()) == []


def test_descriptor_is_closed_when_fdopen_fails(tmp_path, monkeypatch) -> None:
    closed = []
    real_close = llm.os.close

    def failing_fdopen(fd, mode):
        raise OSError("cannot wrap descriptor")

    def tracking_close(fd):
        closed.append(fd)
        return real_close(fd)

    monkeypatch.setattr(llm.os, "fdopen", failing_fdopen)
    monkeypatch.setattr(llm.os, "close", tracking_close)
    writer = RecordingWriter()
    writer.add(_fp(1), ok())
    with pytest.raises(RecordingError, match="nothing was published"):
        writer.write(tmp_path / "run.recording.json")
    assert len(closed) == 1
    assert list(tmp_path.iterdir()) == []


def test_writer_detects_a_target_created_during_publication(tmp_path, monkeypatch) -> None:
    """The exclusive link is the real guard: even if the target appears after the pre-check, it is not
    overwritten."""
    target = tmp_path / "run.recording.json"
    real_link = llm.os.link

    def racing_link(src, dst):
        Path(dst).write_text("RACE WINNER", encoding="utf-8")
        return real_link(src, dst)

    monkeypatch.setattr(llm.os, "link", racing_link)
    writer = RecordingWriter()
    writer.add(_fp(1), ok())
    with pytest.raises(RecordingError, match="refusing to overwrite"):
        writer.write(target)
    assert target.read_text(encoding="utf-8") == "RACE WINNER"
    assert list(tmp_path.glob(".tmp-*.recording.json")) == []


# --------------------------------------------------------------------------------------------
# ReplayProofreader
# --------------------------------------------------------------------------------------------


def _recording(tmp_path, cfg, results_by_user: dict[str, ProofreadResult]) -> Path:
    writer = RecordingWriter()
    for user, result in results_by_user.items():
        writer.add(request(cfg, user=user).fingerprint, result)
    target = tmp_path / "replay.recording.json"
    writer.write(target)
    return target


def test_replay_returns_recorded_results_including_metrics(tmp_path, cfg) -> None:
    refusal = ProofreadResult(None, FailureClass.REFUSAL, Usage(3), 4.0, 1, "req_9", "refusal")
    replay = ReplayProofreader.from_file(_recording(tmp_path, cfg, {"para A": ok(), "para B": refusal}))
    assert replay.proofread(request(cfg, 0, user="para A")) == ok()
    assert replay.proofread(request(cfg, 1, user="para B")) == refusal
    assert len(replay) == 2


def test_replay_is_idempotent_and_identical_requests_share_a_response(tmp_path, cfg) -> None:
    replay = ReplayProofreader.from_file(_recording(tmp_path, cfg, {"same text": ok()}))
    results = [replay.proofread(request(cfg, pid, user="same text")) for pid in (0, 7, 0)]
    assert results == [ok()] * 3


def test_replay_payloads_are_independent_copies(tmp_path, cfg) -> None:
    source = {_fp(1): ok({"findings": [{"original": PAYLOAD_SENTINEL}]})}
    req = request(cfg)
    replay = ReplayProofreader({req.fingerprint: source[_fp(1)]})
    source[_fp(1)].payload["findings"].clear()  # mutating the source after construction
    first = replay.proofread(req)
    assert first.payload == {"findings": [{"original": PAYLOAD_SENTINEL}]}
    first.payload["findings"].clear()  # mutating a returned result
    assert replay.proofread(req).payload == {"findings": [{"original": PAYLOAD_SENTINEL}]}


def test_replay_miss_and_fingerprint_mismatch_are_fatal(tmp_path, cfg) -> None:
    replay = ReplayProofreader.from_file(_recording(tmp_path, cfg, {"recorded": ok()}))
    for changed in (request(cfg, user="different text"), request(cfg, user="recorded", schema={"type": "object"})):
        with pytest.raises(ReplayMissError) as info:
            replay.proofread(changed)
        assert info.value.exit_code == 1 and "different text" not in str(info.value)
    other_model = dataclasses.replace(cfg, model="other-model")
    with pytest.raises(ReplayMissError):
        replay.proofread(request(other_model, user="recorded"))


def test_missing_lists_absent_fingerprints_in_order(tmp_path, cfg) -> None:
    replay = ReplayProofreader.from_file(_recording(tmp_path, cfg, {"a": ok()}))
    present, absent1, absent2 = (request(cfg, user=u).fingerprint for u in ("a", "b", "c"))
    assert replay.missing([absent2, present, absent1, absent2]) == (absent2, absent1)
    assert replay.missing([present]) == ()


def test_replay_is_safe_under_concurrent_use(tmp_path, cfg) -> None:
    replay = ReplayProofreader.from_file(_recording(tmp_path, cfg, {"x": ok()}))
    req = request(cfg, user="x")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: replay.proofread(req), range(64), timeout=10))
    assert results == [ok()] * 64


def test_replay_validates_its_mapping() -> None:
    with pytest.raises(ValueError):
        ReplayProofreader({"bad": ok()})
    with pytest.raises(ValueError):
        ReplayProofreader({_fp(1): "not a result"})  # type: ignore[dict-item]


# --------------------------------------------------------------------------------------------
# Strict recording loading
# --------------------------------------------------------------------------------------------


def _valid_document() -> dict:
    writer = RecordingWriter()
    writer.add(_fp(1), ok({"findings": [{"original": PAYLOAD_SENTINEL}]}))
    writer.add(_fp(2), ProofreadResult(None, FailureClass.TIMEOUT, Usage(), 0.0, 1, None, None))
    return json.loads(writer.render())


def _entry(document: dict, **changes) -> dict:
    document["entries"][0].update(changes)
    return document


MUTATIONS = {
    "top-level list": lambda d: [d],
    "missing header key": lambda d: {k: v for k, v in d.items() if k != "prompt_version"},
    "extra header key": lambda d: {**d, "extra": 1},
    "wrong format": lambda d: {**d, "format": "other"},
    "version 2": lambda d: {**d, "version": 2},
    "version true": lambda d: {**d, "version": True},
    "stale fingerprint version": lambda d: {**d, "fingerprint_version": "proofreader-request-v0"},
    "stale prompt version": lambda d: {**d, "prompt_version": "s4.3-v0"},
    "entries not a list": lambda d: {**d, "entries": {}},
    "entry extra key": lambda d: _entry(d, extra=1),
    "entry missing key": lambda d: {**d, "entries": [{k: v for k, v in d["entries"][0].items() if k != "usage"}]},
    "uppercase fingerprint": lambda d: _entry(d, fingerprint="A" * 64),
    "short fingerprint": lambda d: _entry(d, fingerprint="abc"),
    "bad result": lambda d: _entry(d, result="maybe"),
    "ok with failure": lambda d: _entry(d, failure="timeout"),
    "unknown failure class": lambda d: _entry(d, result="failure", failure="explosion"),
    "failure without class": lambda d: _entry(d, result="failure", failure=None),
    "usage missing key": lambda d: _entry(d, usage={"input_tokens": 1}),
    "usage negative": lambda d: _entry(d, usage={**d["entries"][0]["usage"], "input_tokens": -1}),
    "usage bool": lambda d: _entry(d, usage={**d["entries"][0]["usage"], "output_tokens": True}),
    "attempts bool": lambda d: _entry(d, attempts=True),
    "ok without payload": lambda d: _entry(d, payload=None),
    "ok with zero attempts": lambda d: _entry(d, attempts=0),
    "negative latency": lambda d: _entry(d, latency_ms=-1),
    "bad request id": lambda d: _entry(d, request_id="has space"),
    "bad stop reason": lambda d: _entry(d, stop_reason="End Turn"),
    "duplicate fingerprint": lambda d: {**d, "entries": d["entries"] + [d["entries"][0]]},
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_malformed_recordings_are_rejected(tmp_path, name: str) -> None:
    target = tmp_path / "bad.recording.json"
    target.write_text(json.dumps(MUTATIONS[name](_valid_document())), encoding="utf-8")
    with pytest.raises(RecordingError) as info:
        llm.load_recording(target)
    assert PAYLOAD_SENTINEL not in str(info.value)


@pytest.mark.parametrize(
    "raw, needle",
    [
        (b"not json", "invalid JSON"),
        (b'{"format": "a", "format": "b"}', "duplicate JSON object key"),
        (b'{"version": NaN}', "non-standard JSON constant"),
        (b"{}  {}", "invalid JSON"),
        (b"\xff\xfe\x00", "UTF-8"),
    ],
)
def test_non_strict_json_is_rejected(tmp_path, raw: bytes, needle: str) -> None:
    target = tmp_path / "bad.recording.json"
    target.write_bytes(raw)
    with pytest.raises(RecordingError, match=needle):
        llm.load_recording(target)


def test_oversized_and_missing_recordings_are_rejected(tmp_path, monkeypatch) -> None:
    target = tmp_path / "big.recording.json"
    target.write_text(json.dumps(_valid_document()), encoding="utf-8")
    monkeypatch.setattr(llm, "MAX_RECORDING_BYTES", 10)
    with pytest.raises(RecordingError, match="larger than"):
        llm.load_recording(target)
    with pytest.raises(RecordingError, match="cannot be read"):
        llm.load_recording(tmp_path / "absent.recording.json")


def test_the_unmutated_document_loads_positive_control(tmp_path) -> None:
    """Positive control for the malformed-recording tests: the same document without a mutation loads."""
    target = tmp_path / "good.recording.json"
    target.write_text(json.dumps(_valid_document()), encoding="utf-8")
    loaded = llm.load_recording(target)
    assert set(loaded) == {_fp(1), _fp(2)}
    assert loaded[_fp(1)].payload == {"findings": [{"original": PAYLOAD_SENTINEL}]}
    assert loaded[_fp(2)].failure is FailureClass.TIMEOUT


# --------------------------------------------------------------------------------------------
# Privacy rule in .gitignore (D1)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, ignored",
    [("run.recording.json", True), ("local/out.recording.json", True), ("examples/a.recording.json", True),
     ("tests/fixtures/llm/invented.recording.json", False)],
)
def test_real_recordings_are_git_ignored_and_invented_fixtures_are_not(path: str, ignored: bool) -> None:
    result = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT, capture_output=True)
    assert (result.returncode == 0) is ignored


# --------------------------------------------------------------------------------------------
# S4.4 audit additions
# --------------------------------------------------------------------------------------------


def test_bool_max_tokens_is_rejected_by_the_request_itself(cfg) -> None:
    """The request's own check fires first (its message), not only the fingerprint function's."""
    req = request(cfg)
    fields = {f.name: getattr(req, f.name) for f in dataclasses.fields(req)}
    with pytest.raises(ValueError, match=r"max_tokens must be a positive int$"):
        ProofreadRequest(**{**fields, "max_tokens": True})


@pytest.mark.parametrize(
    "raw, needle",
    [
        (b'{"version": Infinity}', "non-standard JSON constant"),
        (b'{"version": -Infinity}', "non-standard JSON constant"),
        (b'{"version": 1e400}', "non-finite JSON number"),
        (b'{"version": -1e400}', "non-finite JSON number"),
    ],
)
def test_infinite_numbers_are_rejected(tmp_path, raw: bytes, needle: str) -> None:
    target = tmp_path / "bad.recording.json"
    target.write_bytes(raw)
    with pytest.raises(RecordingError, match=needle):
        llm.load_recording(target)


def test_pathologically_deep_json_fails_cleanly_not_with_a_crash(tmp_path) -> None:
    target = tmp_path / "deep.recording.json"
    target.write_bytes(b"[" * 200_000 + b"]" * 200_000)
    with pytest.raises(RecordingError, match="nested too deeply"):
        llm.load_recording(target)


def test_overflowing_number_inside_a_payload_is_rejected(tmp_path) -> None:
    document = _valid_document()
    text = json.dumps(document).replace(f'"original": "{PAYLOAD_SENTINEL}"', '"original": 1e400', 1)
    target = tmp_path / "bad.recording.json"
    target.write_text(text, encoding="utf-8")
    with pytest.raises(RecordingError, match="non-finite"):
        llm.load_recording(target)


def _nest(levels: int, key: str) -> dict:
    node: dict = {}
    for _ in range(levels):
        node = {key: node}
    return node


def test_loaded_payload_obeys_the_writer_json_rules_without_leaking_keys(tmp_path) -> None:
    document = _valid_document()
    document["entries"][0]["payload"] = _nest(70, PAYLOAD_SENTINEL)  # deeper than the 64-level limit
    target = tmp_path / "bad.recording.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RecordingError, match="payload is not JSON-safe") as info:
        llm.load_recording(target)
    assert PAYLOAD_SENTINEL not in str(info.value)


def test_writer_payload_errors_do_not_leak_payload_keys() -> None:
    writer = RecordingWriter()
    with pytest.raises(RecordingError, match="not JSON-safe") as info:
        writer.add(_fp(1), ok(_nest(70, PAYLOAD_SENTINEL)))
    assert PAYLOAD_SENTINEL not in str(info.value)
    with pytest.raises(RecordingError) as info:
        writer.add(_fp(2), ok({PAYLOAD_SENTINEL: float("inf")}))
    assert PAYLOAD_SENTINEL not in str(info.value)


@pytest.mark.parametrize(
    "result",
    [
        ProofreadResult(GOOD, None, Usage(), 0.0, 1, "sk-ant-" + "Q" * 20, None),  # request ids allow this
        ok({"findings": [{"reason": "ANTHROPIC_API_KEY=" + "q" * 20}]}),
    ],
)
def test_secret_patterns_in_any_serialised_field_are_refused(tmp_path, result) -> None:
    writer = RecordingWriter()
    writer.add(_fp(1), result)
    with pytest.raises(RecordingError, match="credential-like") as info:
        writer.write(tmp_path / "run.recording.json")
    assert "QQQQ" not in str(info.value) and "qqqq" not in str(info.value)
    assert list(tmp_path.iterdir()) == []


def test_data_is_fsynced_before_publication(tmp_path, monkeypatch) -> None:
    events = []
    real_fsync, real_link = llm.os.fsync, llm.os.link
    monkeypatch.setattr(llm.os, "fsync", lambda fd: (events.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(llm.os, "link", lambda s, d: (events.append("link"), real_link(s, d))[1])
    writer = RecordingWriter()
    writer.add(_fp(1), ok())
    writer.write(tmp_path / "run.recording.json")
    assert events == ["fsync", "link"]


def test_refusing_an_existing_target_writes_nothing_at_all(tmp_path, monkeypatch) -> None:
    target = tmp_path / "run.recording.json"
    target.write_text("ORIGINAL", encoding="utf-8")

    def no_temp(*args, **kwargs):
        raise AssertionError("no temporary file may be created when the target exists")

    monkeypatch.setattr(llm.tempfile, "mkstemp", no_temp)
    writer = RecordingWriter()
    writer.add(_fp(1), ok())
    with pytest.raises(RecordingError, match="refusing to overwrite"):
        writer.write(target)
    assert target.read_text(encoding="utf-8") == "ORIGINAL"


def test_missing_destination_directory_fails_cleanly(tmp_path) -> None:
    writer = RecordingWriter()
    writer.add(_fp(1), ok())
    with pytest.raises(RecordingError, match="temporary file"):
        writer.write(tmp_path / "no-such-dir" / "run.recording.json")
    assert list(tmp_path.iterdir()) == []


def test_loading_a_directory_fails_cleanly(tmp_path) -> None:
    folder = tmp_path / "folder.recording.json"
    folder.mkdir()
    with pytest.raises(RecordingError, match="cannot be read"):
        llm.load_recording(folder)


def test_results_loaded_from_file_are_independent_of_replay_results(tmp_path, cfg) -> None:
    path = _recording(tmp_path, cfg, {"x": ok({"findings": [{"original": PAYLOAD_SENTINEL}]})})
    loaded = llm.load_recording(path)
    replay = ReplayProofreader(loaded)
    next(iter(loaded.values())).payload["findings"].clear()  # mutate the loaded mapping afterwards
    assert replay.proofread(request(cfg, user="x")).payload == {"findings": [{"original": PAYLOAD_SENTINEL}]}


def test_replay_does_not_swallow_base_exceptions(cfg, monkeypatch) -> None:
    replay = ReplayProofreader({request(cfg).fingerprint: ok()})

    def interrupted(value, memo=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(llm.copy, "deepcopy", interrupted)
    with pytest.raises(KeyboardInterrupt):
        replay.proofread(request(cfg))

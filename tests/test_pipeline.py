"""S4.6: pipeline orchestration and bounded concurrency.

Offline only: FakeProofreader, ReplayProofreader and AnthropicProofreader over httpx2.MockTransport (dummy key).
All XML and findings are invented.
"""

from __future__ import annotations

import dataclasses
import random
import re
import threading
import time
import xml.etree.ElementTree as ET

import httpx2
import pytest

from proofreader import annotation as an
from proofreader import config as c
from proofreader import llm
from proofreader import pipeline as pl
from proofreader import prompt as pr
from proofreader import xml_source as xs
from proofreader.config import LocaleScope, UsageError
from proofreader.llm import (CallStatus, FailureClass, FakeBlock, FakeFailure, FakeOk, FakeProofreader, FakeRaise,
                             ProofreadResult, ProviderFatalError, ReplayMissError, ReplayProofreader, Usage)
from proofreader.models import AnnotationOutcome, Category, ProofreaderError, RejectCode, TraceStatus
from proofreader.style_guide import SelectedGuide

GUIDE_TEXT = "# Invented house rules\n\nWrite 'per cent' in running text, never the % sign.\n"
SENTINEL = "Zqxsentinel"
WAIT_S = 5.0


@pytest.fixture
def cfg(tmp_path) -> c.RunConfig:
    return c.build_config({"input": "a.xml", "lang": "en", "style_guide": "g.docx"}, {}, tmp_path)


def guide(text: str = GUIDE_TEXT) -> SelectedGuide:
    return SelectedGuide(text, LocaleScope.FULL, 1 if text else 0, 1, ())


def doc(*paragraphs: str, prolog: str = "") -> bytes:
    return (prolog + "<doc>" + "".join(paragraphs) + "</doc>").encode("utf-8")


def finding(data: bytes, pid: int, original: str, correction: str, type_: str = "spelling",
            reason: str = "Misspelling") -> dict:
    """A well-formed finding with real anchors (up to 3 words each side), as the model is instructed."""
    paragraph = next(p for p in xs.scan(data).paragraphs if p.id == pid)
    words = [w for group in pr.segment_words(paragraph) for _, w in group]
    n = len(original.split(" "))
    (i,) = [k for k in range(len(words) - n + 1) if " ".join(words[k:k + n]) == original]
    j = i + n - 1
    return {"original": original, "before": " ".join(words[max(0, i - 3):i]), "after": " ".join(words[j + 1:j + 4]),
            "word_start": i, "word_end": j, "correction": correction, "type": type_, "reason": reason}


def env(*findings: dict) -> dict:
    return {"findings": list(findings)}


EMPTY = FakeOk(env())
CLEAN3 = doc("<p>The committee met on Tuesday.</p>", "<p>It approved the budget.</p>", "<p>Everyone left early.</p>")
TYPO = doc("<p>Our team will recieve the report.</p>", "<p>Nothing to fix here.</p>")
TYPO_OUT = doc('<p>Our team will <error type="spelling" correction="receive" reason="Misspelling">recieve</error> the report.</p>',
               "<p>Nothing to fix here.</p>")


def run(data, cfg, fake, g=None):
    return pl.run(data, cfg, fake, g or guide())


def worker_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith(pl._THREAD_PREFIX)]


def wait_until(predicate, timeout=WAIT_S) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.002)


class Runner(threading.Thread):
    """Runs the pipeline in a background thread so the test can steer FakeBlock calls."""

    def __init__(self, *args):
        super().__init__(daemon=True)
        self.args, self.result, self.error = args, None, None

    def run(self):
        try:
            self.result = pl.run(*self.args)
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
            self.error = exc

    def finish(self):
        self.join(WAIT_S)
        assert not self.is_alive(), "pipeline did not finish"
        return self


@pytest.fixture(autouse=True)
def no_leaked_workers():
    yield
    wait_until(lambda: not worker_threads(), timeout=2.0)


# --------------------------------------------------------------------------------------------
# Basic flow, AC-04, AC-11, AC-31
# --------------------------------------------------------------------------------------------


def test_clean_document_is_byte_identical_ac04(cfg) -> None:
    fake = FakeProofreader(default=EMPTY)
    res = run(CLEAN3, cfg, fake)
    assert res.output == CLEAN3
    assert sorted(fake.call_counts()) == [0, 1, 2]
    assert [r.status for r in res.call_records] == [CallStatus.OK] * 3
    assert res.trace == () and not res.degraded and res.rolled_back == ()


def test_a_finding_is_applied_and_traced(cfg) -> None:
    fake = FakeProofreader({0: [FakeOk(env(finding(TYPO, 0, "recieve", "receive")))], 1: [EMPTY]})
    res = run(TYPO, cfg, fake)
    assert res.output == TYPO_OUT
    assert [(t.paragraph_id, t.finding_index, t.status) for t in res.trace] == [(0, 0, TraceStatus.APPLIED)]
    assert res.usage == Usage(2, 2) and res.requests_sent == 2 and not res.degraded


def test_already_annotated_input_makes_no_calls_ac11(cfg) -> None:
    data = doc('<p>A <error type="spelling" correction="cat">cta</error> sat.</p>', "<p> </p>", "<p/>")
    fake = FakeProofreader()  # strict: any call raises FakeScriptError
    res = run(data, cfg, fake)
    assert res.output == data and fake.calls == () and res.call_records == ()


def test_rerun_on_own_output_adds_nothing_ac31(cfg) -> None:
    first = run(TYPO, cfg, FakeProofreader({0: [FakeOk(env(finding(TYPO, 0, "recieve", "receive")))], 1: [EMPTY]}))
    again = FakeProofreader(default=EMPTY)
    second = run(first.output, cfg, again)
    assert second.output == first.output
    assert sorted(again.call_counts()) == [1]  # the annotated paragraph is skipped; only the clean one is asked
    assert second.output.count(b"<error ") == 1


def test_eligibility_only_annotatable_non_empty_paragraphs_are_sent_ac25(cfg) -> None:
    data = doc("<p>First real text.</p>",  # 0
               "<p/>",  # 1 empty
               "<p>  \n\t </p>",  # 2 whitespace only
               "<p><![CDATA[Teh quick fox]]></p>",  # 3 CDATA only
               '<p>Has <error type="spelling" correction="x">y</error> markup.</p>',  # 4 existing error
               "<p>Outer words <p>inner words</p> tail.</p>",  # 5 outer, 6 nested
               "<p> </p>",  # 7 no-break space only
               "<p>Last one.</p>")  # 8
    fake = FakeProofreader(default=EMPTY)
    res = run(data, cfg, fake)
    assert sorted(fake.call_counts()) == [0, 5, 6, 8]
    assert all(n == 1 for n in fake.call_counts().values())
    assert [r.paragraph_id for r in res.call_records] == [0, 5, 6, 8]
    assert res.output == data


def test_no_paragraphs_at_all(cfg) -> None:
    data = b"<doc><title>No paragraphs</title></doc>"
    res = run(data, cfg, FakeProofreader())
    assert res.output == data and res.call_records == () and res.trace == ()


# --------------------------------------------------------------------------------------------
# AC-14 and the empty style guide (Q2)
# --------------------------------------------------------------------------------------------

STYLE = doc("<p>Sales rose by 12% in the north.</p>", "<p>Costs fell by 5% in the south.</p>",
            "<p>We will recieve it.</p>")


def style_fake(data=STYLE):
    return FakeProofreader({
        0: [FakeOk(env(finding(data, 0, "12%", "12 per cent", "styleguide", "House style: write per cent")))],
        1: [FakeOk(env(finding(data, 1, "5%", "5 per cent", "styleguide", "House style: write per cent")))],
        2: [FakeOk(env(finding(data, 2, "recieve", "receive")))],
    })


def test_every_styleguide_error_has_a_non_empty_reason_ac14(cfg) -> None:
    res = run(STYLE, cfg, style_fake())
    root = ET.fromstring(res.output)
    errors = list(root.iter("error"))
    styles = [e for e in errors if e.get("type") == "styleguide"]
    assert len(errors) == 3 and len(styles) == 2
    assert all(e.get("reason") and e.get("reason").strip() for e in styles)
    assert list(styles[0].attrib) == ["type", "correction", "reason"]


@pytest.mark.parametrize("text", ["", "   \n\t"])
def test_styleguide_findings_rejected_without_guide_text_q2(cfg, text) -> None:
    res = run(STYLE, cfg, style_fake(), guide(text))
    assert res.output.count(b'type="styleguide"') == 0
    assert res.output.count(b'type="spelling"') == 1  # other findings unaffected
    rejected = [t for t in res.trace if t.status is TraceStatus.REJECTED_S3]
    assert [(t.paragraph_id, t.finding_index, t.code) for t in rejected] == [(0, 0, RejectCode.INVALID_FIELD),
                                                                            (1, 0, RejectCode.INVALID_FIELD)]
    assert all(t.detail == pl.STYLEGUIDE_WITHOUT_GUIDE_DETAIL for t in rejected)
    assert not res.degraded  # a rejected finding is not a paragraph failure


def test_styleguide_findings_kept_with_guide_text(cfg) -> None:
    res = run(STYLE, cfg, style_fake(), guide())
    assert res.output.count(b'type="styleguide"') == 2


# --------------------------------------------------------------------------------------------
# Size guard and --max-calls
# --------------------------------------------------------------------------------------------


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_size_guard_boundaries(cfg) -> None:
    data = doc(f"<p>{words(50)}</p>", f"<p>{words(51)}</p>", f"<p>{'x' * 5000}</p>", f"<p>{'y' * 5001}</p>")
    fake = FakeProofreader(default=EMPTY)
    res = run(data, cfg, fake)
    assert sorted(fake.call_counts()) == [0, 2]
    guarded = [r for r in res.call_records if r.status is CallStatus.NOT_CALLED]
    assert [(r.paragraph_id, r.failure, r.attempts) for r in guarded] == [(1, FailureClass.REQUEST_TOO_LARGE, 0),
                                                                         (3, FailureClass.REQUEST_TOO_LARGE, 0)]
    assert res.output == data and res.degraded


def test_max_calls_boundary(cfg) -> None:
    ok = FakeProofreader(default=EMPTY)
    assert run(CLEAN3, dataclasses.replace(cfg, max_calls=3), ok).output == CLEAN3
    fake = FakeProofreader(default=EMPTY)
    with pytest.raises(UsageError) as info:
        run(CLEAN3, dataclasses.replace(cfg, max_calls=2), fake)
    assert info.value.exit_code == 2 and fake.calls == ()


def test_max_calls_counts_only_paragraphs_that_will_be_sent(cfg) -> None:
    data = doc("<p>One.</p>", f"<p>{words(60)}</p>", "<p/>", "<p>Two.</p>")
    fake = FakeProofreader(default=EMPTY)
    res = run(data, dataclasses.replace(cfg, max_calls=2), fake)
    assert sorted(fake.call_counts()) == [0, 3] and res.output == data


# --------------------------------------------------------------------------------------------
# Pre-call refusals: cache flag (Q3), bad input, replay completeness
# --------------------------------------------------------------------------------------------


def test_cache_flag_is_refused_before_anything_q3(cfg) -> None:
    fake = FakeProofreader(default=EMPTY)
    with pytest.raises(UsageError) as info:
        run(b"<doc><p>broken", dataclasses.replace(cfg, cache=True), fake)  # refused before the XML is even read
    assert info.value.exit_code == 2 and "cach" in info.value.message and fake.calls == ()


def test_malformed_input_is_fatal_before_any_call(cfg) -> None:
    fake = FakeProofreader(default=EMPTY)
    with pytest.raises(ProofreaderError):
        run(b"<doc><p>unclosed</doc>", cfg, fake)
    assert fake.calls == ()


def fingerprints_for(data, cfg, g=None) -> dict[int, str]:
    fake = FakeProofreader(default=EMPTY)
    run(data, cfg, fake, g)
    return {call.paragraph_id: call.fingerprint for call in fake.calls}


def test_replay_missing_entry_is_fatal_before_any_call(cfg) -> None:
    fps = fingerprints_for(TYPO, cfg)
    ok = ProofreadResult(env(), None, Usage(1, 1), 5.0, 1, "req_a", "end_turn")
    replay = ReplayProofreader({fps[0]: ok})  # paragraph 1 is missing

    class Spy(ReplayProofreader):
        def proofread(self, request):  # pragma: no cover - must never run
            raise AssertionError("a call was made before the replay pre-check")

    with pytest.raises(ReplayMissError) as info:
        run(TYPO, cfg, replay)
    assert info.value.exit_code == 1 and fps[0] not in info.value.message and fps[1] not in info.value.message
    with pytest.raises(ReplayMissError):
        run(TYPO, cfg, Spy({fps[0]: ok}))


def test_replay_reproduces_the_fake_run_byte_for_byte(cfg) -> None:
    fake = FakeProofreader({0: [FakeOk(env(finding(TYPO, 0, "recieve", "receive")))], 1: [EMPTY]})
    expected = run(TYPO, cfg, fake)
    results = {call.fingerprint: ProofreadResult(env(finding(TYPO, 0, "recieve", "receive")) if call.paragraph_id == 0
                                                 else env(), None, Usage(3, 4), 7.0, 1, f"req_{call.paragraph_id}",
                                                 "end_turn")
               for call in fake.calls}
    replayed = run(TYPO, cfg, ReplayProofreader(results))
    assert replayed.output == expected.output == TYPO_OUT
    assert replayed.trace == expected.trace
    assert [r.request_id for r in replayed.call_records] == ["req_0", "req_1"]


# --------------------------------------------------------------------------------------------
# Paragraph-level failures and payload exclusion (Q1)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("failure", list(FailureClass))
def test_every_failure_class_passes_the_paragraph_through(cfg, failure) -> None:
    data = doc("<p>Our team will recieve the report.</p>", "<p>They will recieve it later.</p>")
    good = finding(data, 0, "recieve", "receive")
    # the failed call carries a perfectly valid finding: it must still never reach the resolver
    fake = FakeProofreader({0: [FakeOk(env(good))],
                            1: [FakeFailure(failure, payload=env(finding(data, 1, "recieve", "receive")))]})
    res = run(data, cfg, fake)
    assert res.output.count(b"<error ") == 1
    assert b"They will recieve it later." in res.output
    assert res.call_records[1].status is CallStatus.PASSTHROUGH and res.call_records[1].failure is failure
    assert [t.paragraph_id for t in res.trace] == [0]
    assert res.degraded


def test_successful_but_malformed_envelope_is_traced_not_applied(cfg) -> None:
    data = doc("<p>Our team will recieve the report.</p>")
    fake = FakeProofreader({0: [FakeOk({"findings": [finding(data, 0, "recieve", "receive")], "extra": 1})]})
    res = run(data, cfg, fake)
    assert res.output == data
    assert [(t.status, t.code) for t in res.trace] == [(TraceStatus.REJECTED_S3, RejectCode.SCHEMA_INVALID)]


def test_rollback_marks_the_run_degraded(cfg, monkeypatch) -> None:
    data = doc("<p>Our team will recieve the report.</p>", "<p>They will recieve it later.</p>")
    real = an.annotate

    def faulty_render(d, spans, codec):  # paragraph 1 is rendered with a different correction
        spans = [dataclasses.replace(s, annotation=dataclasses.replace(s.annotation, correction="bogus"))
                 if s.annotation.paragraph_id == 1 else s for s in spans]
        return an.render(d, spans, codec)

    monkeypatch.setattr(an, "annotate", lambda d, s, a: real(d, s, a, render_fn=faulty_render))
    fake = FakeProofreader({pid: [FakeOk(env(finding(data, pid, "recieve", "receive")))] for pid in (0, 1)})
    res = run(data, cfg, fake)
    assert res.rolled_back == (1,) and res.degraded
    assert res.output.count(b"<error ") == 1 and b"They will recieve it later." in res.output
    assert [t.status for t in res.trace] == [TraceStatus.APPLIED, TraceStatus.ROLLED_BACK]


def test_document_verification_failure_returns_nothing(cfg, monkeypatch) -> None:
    secret = f"issue quoting {SENTINEL}"
    monkeypatch.setattr(an, "annotate", lambda d, s, a: AnnotationOutcome(False, None, (), (), (), (secret,), 1))
    with pytest.raises(pl.DocumentVerificationError) as info:
        run(TYPO, cfg, FakeProofreader(default=EMPTY))
    assert info.value.exit_code == 1 and isinstance(info.value, llm.RunFatalError)
    assert SENTINEL not in str(info.value)


def test_non_result_return_value_is_fatal(cfg) -> None:
    class Odd:
        def proofread(self, request):
            return {"findings": []}

    with pytest.raises(TypeError):
        run(TYPO, cfg, Odd())


# --------------------------------------------------------------------------------------------
# AC-28 bounded concurrency
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("limit,n", [(1, 5), (2, 7), (3, 10), (8, 20), (8, 3), (64, 12)])
def test_in_flight_never_exceeds_and_reaches_the_limit_ac28(cfg, limit, n) -> None:
    data = doc(*(f"<p>Paragraph number {i} here.</p>" for i in range(n)))
    gate = threading.Event()
    fake = FakeProofreader(default=FakeBlock(gate, EMPTY))
    runner = Runner(data, dataclasses.replace(cfg, concurrency=limit), fake, guide())
    runner.start()
    expected_peak = min(limit, n)
    wait_until(lambda: fake.in_flight == expected_peak)
    time.sleep(0.05)  # give any excess worker time to show up
    assert fake.in_flight == expected_peak and len(fake.calls) == expected_peak
    assert len(worker_threads()) == expected_peak
    gate.set()
    runner.finish()
    assert runner.error is None and runner.result.output == data
    assert fake.max_in_flight == expected_peak and len(fake.calls) == n


# --------------------------------------------------------------------------------------------
# AC-20 completion order does not matter
# --------------------------------------------------------------------------------------------

ORDER_N = 8
ORDER_DOC = doc(*(f"<p>Item {i} will recieve a numbr today.</p>" for i in range(ORDER_N)))


def ordered_run(cfg, release_order):
    gates = {pid: threading.Event() for pid in range(ORDER_N)}
    script = {}
    for pid in range(ORDER_N):
        fs = [finding(ORDER_DOC, pid, "recieve", "receive")]
        if pid % 2:
            fs.append(finding(ORDER_DOC, pid, "numbr", "number"))
        script[pid] = [FakeBlock(gates[pid], FakeOk(env(*fs)))]
    fake = FakeProofreader(script)
    runner = Runner(ORDER_DOC, dataclasses.replace(cfg, concurrency=ORDER_N), fake, guide())
    runner.start()
    wait_until(lambda: fake.in_flight == ORDER_N)
    for k, pid in enumerate(release_order):
        gates[pid].set()
        wait_until(lambda k=k: fake.in_flight == ORDER_N - k - 1)  # pid has completed before the next release
    runner.finish()
    assert runner.error is None
    return runner.result


def test_completion_order_does_not_change_anything_ac20(cfg) -> None:
    baseline = ordered_run(cfg, list(range(ORDER_N)))
    assert baseline.output.count(b"<error ") == ORDER_N + ORDER_N // 2
    orders = [list(reversed(range(ORDER_N)))]
    for seed in range(6):
        order = list(range(ORDER_N))
        random.Random(seed).shuffle(order)
        orders.append(order)
    for order in orders:
        res = ordered_run(cfg, order)
        assert res.output == baseline.output, order
        assert res.call_records == baseline.call_records, order
        assert res.trace == baseline.trace, order


# --------------------------------------------------------------------------------------------
# Run-fatal errors, cancellation (Q4), interrupts
# --------------------------------------------------------------------------------------------

MANY = doc(*(f"<p>Line {i} of the text.</p>" for i in range(20)))


def test_fatal_error_cancels_unstarted_calls_and_waits_for_running_ones(cfg) -> None:
    g0, g1 = threading.Event(), threading.Event()
    s0, s1 = threading.Event(), threading.Event()
    script = {0: [FakeBlock(g0, FakeRaise(ProviderFatalError("HTTP 401")), started=s0)],
              1: [FakeBlock(g1, EMPTY, started=s1)]}
    fake = FakeProofreader(script, default=EMPTY)
    runner = Runner(MANY, dataclasses.replace(cfg, concurrency=2), fake, guide())
    runner.start()
    assert s0.wait(WAIT_S) and s1.wait(WAIT_S)
    g0.set()  # paragraph 0 fails while paragraph 1 is still running
    time.sleep(0.1)
    assert runner.is_alive(), "the pipeline must wait for the running call"
    assert fake.in_flight == 1
    g1.set()
    runner.finish()
    assert isinstance(runner.error, ProviderFatalError) and runner.result is None
    assert sorted(fake.call_counts()) == [0, 1], "a call started after the failure"
    assert fake.in_flight == 0


def test_lowest_observed_paragraph_id_is_raised(cfg) -> None:
    g0, g1 = threading.Event(), threading.Event()
    s0, s1 = threading.Event(), threading.Event()
    script = {0: [FakeBlock(g0, FakeRaise(ProviderFatalError("first paragraph")), started=s0)],
              1: [FakeBlock(g1, FakeRaise(ProviderFatalError("second paragraph")), started=s1)]}
    runner = Runner(MANY, dataclasses.replace(cfg, concurrency=2), FakeProofreader(script, default=EMPTY), guide())
    runner.start()
    assert s0.wait(WAIT_S) and s1.wait(WAIT_S)
    g1.set()  # paragraph 1 fails FIRST in time
    time.sleep(0.05)
    g0.set()  # paragraph 0 fails later, while the pipeline waits for it
    runner.finish()
    assert str(runner.error) == "first paragraph"


def test_base_exception_from_a_worker_takes_precedence(cfg) -> None:
    g0, g1 = threading.Event(), threading.Event()
    s0, s1 = threading.Event(), threading.Event()
    script = {0: [FakeBlock(g0, FakeRaise(ProviderFatalError("fatal")), started=s0)],
              1: [FakeBlock(g1, FakeRaise(KeyboardInterrupt()), started=s1)]}
    runner = Runner(MANY, dataclasses.replace(cfg, concurrency=2), FakeProofreader(script, default=EMPTY), guide())
    runner.start()
    assert s0.wait(WAIT_S) and s1.wait(WAIT_S)
    g0.set()
    time.sleep(0.05)
    g1.set()
    runner.finish()
    assert isinstance(runner.error, KeyboardInterrupt)


def test_unexpected_exception_propagates_and_nothing_is_returned(cfg) -> None:
    fake = FakeProofreader({3: [FakeRaise(RuntimeError("boom"))]}, default=EMPTY)
    with pytest.raises(RuntimeError, match="boom"):
        run(MANY, dataclasses.replace(cfg, concurrency=1), fake)
    assert sorted(fake.call_counts()) == [0, 1, 2, 3]  # nothing starts after the failing call


def test_keyboard_interrupt_in_the_calling_thread_cancels_and_joins(cfg, monkeypatch) -> None:
    gate = threading.Event()
    fake = FakeProofreader(default=FakeBlock(gate, EMPTY))
    real_wait = pl.wait

    def interrupted_wait(fs, return_when=None):
        wait_until(lambda: fake.in_flight == 2)
        threading.Timer(0.1, gate.set).start()  # running calls finish shortly after the interrupt
        raise KeyboardInterrupt

    monkeypatch.setattr(pl, "wait", interrupted_wait)
    with pytest.raises(KeyboardInterrupt):
        run(MANY, dataclasses.replace(cfg, concurrency=2), fake)
    assert not worker_threads()  # shutdown waited for the running calls
    assert len(fake.calls) == 2 and fake.in_flight == 0  # every queued call was cancelled
    monkeypatch.setattr(pl, "wait", real_wait)


class HookGate(pl._StartGate):
    """The real start gate, with Event handshakes around paragraph ``target``'s start attempt (no sleeps)."""

    instances: list = []
    created = threading.Event()

    def __init__(self, target: int, hold_before_start: bool):
        super().__init__()
        self.target, self.hold_before_start = target, hold_before_start
        self.reached, self.stopped_event = threading.Event(), threading.Event()
        HookGate.instances.append(self)
        HookGate.created.set()

    def try_start(self, paragraph_id):
        if paragraph_id != self.target:
            return super().try_start(paragraph_id)
        if self.hold_before_start:  # the worker is between "dequeued" and its start attempt
            self.reached.set()
            self.stopped_event.wait(WAIT_S)  # a missing stop is detected by the caller's assertions
            return super().try_start(paragraph_id)
        started = super().try_start(paragraph_id)  # marked started; still before the provider call
        self.reached.set()
        assert self.stopped_event.wait(WAIT_S)
        return started

    def stop(self):
        super().stop()
        self.stopped_event.set()


def race_run(cfg, monkeypatch, hold_before_start):
    """Paragraph 0 fails exactly while paragraph 1's worker is paused at its start transition."""
    HookGate.instances = []
    HookGate.created = threading.Event()
    monkeypatch.setattr(pl, "_StartGate", lambda: HookGate(1, hold_before_start))
    g0, s0 = threading.Event(), threading.Event()
    fake = FakeProofreader({0: [FakeBlock(g0, FakeRaise(ProviderFatalError("fatal")), started=s0)]}, default=EMPTY)
    runner = Runner(MANY, dataclasses.replace(cfg, concurrency=2), fake, guide())
    runner.start()
    assert HookGate.created.wait(WAIT_S) and s0.wait(WAIT_S)
    assert HookGate.instances[0].reached.wait(WAIT_S)  # paragraph 1's worker is paused at its start transition
    g0.set()  # paragraph 0 now fails and stops the gate
    runner.finish()
    assert isinstance(runner.error, ProviderFatalError)
    return HookGate.instances[0], fake


def test_queued_call_reaching_the_gate_after_stop_never_starts_race(cfg, monkeypatch) -> None:
    gate, fake = race_run(cfg, monkeypatch, hold_before_start=True)
    assert gate.started_at_stop == (0,) and gate.started == [0]
    assert 1 in gate.refused
    assert sorted(fake.call_counts()) == [0]  # paragraph 1 never reached the provider


def test_call_started_before_stop_finishes_and_nothing_starts_after_race(cfg, monkeypatch) -> None:
    gate, fake = race_run(cfg, monkeypatch, hold_before_start=False)
    assert gate.started_at_stop == (0, 1) and gate.started == [0, 1]  # started never grew after stop
    assert sorted(fake.call_counts()) == [0, 1]  # paragraph 1 was started first, so it may finish
    assert fake.in_flight == 0
    assert all(pid not in gate.started for pid in gate.refused)


def test_interrupt_in_calling_thread_stops_a_call_already_dequeued_by_a_worker(cfg, monkeypatch) -> None:
    """cancel_futures cannot cancel an item a worker has already taken from the queue: only the gate can."""
    HookGate.instances = []
    HookGate.created = threading.Event()
    monkeypatch.setattr(pl, "_StartGate", lambda: HookGate(1, True))
    fake = FakeProofreader(default=EMPTY)

    def interrupted_wait(fs, return_when=None):
        assert HookGate.instances[0].reached.wait(WAIT_S)  # the worker holds paragraph 1, not yet started
        raise KeyboardInterrupt

    monkeypatch.setattr(pl, "wait", interrupted_wait)
    with pytest.raises(KeyboardInterrupt):
        run(MANY, dataclasses.replace(cfg, concurrency=1), fake)
    gate = HookGate.instances[0]
    assert gate.started_at_stop == (0,) and gate.started == [0] and 1 in gate.refused
    assert sorted(fake.call_counts()) == [0] and not worker_threads()


@pytest.mark.parametrize("seed", range(25))
def test_no_call_reaches_the_provider_unless_started_before_stop(cfg, monkeypatch, seed) -> None:
    gates = []

    class Capture(pl._StartGate):
        def __init__(self):
            super().__init__()
            gates.append(self)

    monkeypatch.setattr(pl, "_StartGate", Capture)
    failing = random.Random(seed).randrange(20)
    fake = FakeProofreader({failing: [FakeRaise(ProviderFatalError("fatal"))]}, default=EMPTY)
    with pytest.raises(ProviderFatalError):
        run(MANY, dataclasses.replace(cfg, concurrency=4), fake)
    (gate,) = gates
    assert gate.started_at_stop is not None and gate.started == list(gate.started_at_stop)
    assert sorted(fake.call_counts()) == sorted(gate.started)  # every provider call passed the gate first
    assert failing in gate.started and not set(gate.refused) & set(gate.started)


# --------------------------------------------------------------------------------------------
# Records are text-free
# --------------------------------------------------------------------------------------------


def test_records_and_results_contain_no_document_text(cfg) -> None:
    data = doc(f"<p>{SENTINEL} will recieve it.</p>", f"<p>{SENTINEL} again here.</p>",
               f"<p>{' '.join([SENTINEL] * 60)}</p>")
    fake = FakeProofreader({0: [FakeOk(env(finding(data, 0, "recieve", "receive"),
                                           finding(data, 0, SENTINEL, "x" + SENTINEL)))],
                            1: [FakeFailure(FailureClass.REFUSAL, payload=env(finding(data, 1, "again", SENTINEL)))]})
    res = run(data, cfg, fake)
    for text in (repr(res), repr(res.call_records), repr(res.trace), repr(res.stage_ms)):
        assert SENTINEL not in text and "recieve" not in text
    assert all(SENTINEL not in t.detail for t in res.trace)


# --------------------------------------------------------------------------------------------
# AnthropicProofreader through the real SDK and an in-memory transport (no network)
# --------------------------------------------------------------------------------------------


def test_anthropic_client_integration_offline(tmp_path) -> None:
    import json

    cfg = c.build_config({"input": "a.xml", "lang": "en", "style_guide": "g.docx"},
                         {"ANTHROPIC_API_KEY": "dummy-test-key-not-real-0123456789"}, tmp_path)
    data = doc("<p>Our team will recieve the report.</p>", "<p>Nothing to fix.</p>", "<p>Rate limited one.</p>")
    texts = {0: json.dumps(env(finding(data, 0, "recieve", "receive"))), 1: json.dumps(env())}
    lock, seen = threading.Lock(), []

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.host == "api.anthropic.com"
        body = json.loads(request.content)
        pid = next(i for i, word in ((0, "recieve"), (1, "Nothing"), (2, "Rate")) if word in body["messages"][0]["content"])
        with lock:
            seen.append(pid)
        if pid == 2:
            return httpx2.Response(429, json={"type": "error", "error": {"type": "rate_limit_error", "message": "x"}},
                                   headers={"request-id": "req_429"})
        message = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                   "content": [{"type": "text", "text": texts[pid]}], "stop_reason": "end_turn",
                   "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5,
                                                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}
        return httpx2.Response(200, json=message, headers={"request-id": f"req_{pid}"})

    http = httpx2.Client(transport=httpx2.MockTransport(handler), trust_env=False)
    client = llm.AnthropicProofreader(cfg, http_client=http, sleep=lambda s: None)
    try:
        res = pl.run(data, cfg, client, guide())
    finally:
        client.close()
    assert res.output == data.replace(b"recieve", b'<error type="spelling" correction="receive" reason="Misspelling">recieve</error>')
    assert sorted(seen) == [0, 1, 2, 2, 2]  # 429: first request + 2 transport retries
    assert [(r.status, r.failure, r.attempts) for r in res.call_records] == [
        (CallStatus.OK, None, 1), (CallStatus.OK, None, 1), (CallStatus.PASSTHROUGH, FailureClass.RATE_LIMITED, 3)]
    assert res.degraded and res.requests_sent == 5
    assert res.usage == Usage(20, 10)

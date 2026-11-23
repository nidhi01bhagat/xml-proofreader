"""S4.1: text-free result / audit types for model calls (Usage, ProofreadResult, CallRecord)."""

from __future__ import annotations

import dataclasses

import pytest

from proofreader import llm
from proofreader.llm import CallRecord, CallStatus, FailureClass, ProofreadResult, Usage

OK_USAGE = Usage(10, 5, 0, 0)


def result(**overrides) -> ProofreadResult:
    base = dict(payload={"findings": []}, failure=None, usage=OK_USAGE, latency_ms=12.5, attempts=1,
                request_id="req_abc-1", stop_reason="end_turn")
    base.update(overrides)
    return ProofreadResult(**base)


def test_failure_classes_are_the_approved_paragraph_level_set() -> None:
    assert {f.value for f in FailureClass} == {
        "timeout", "connection", "rate_limited", "server_error", "request_too_large", "refusal", "max_tokens",
        "context_window_exceeded", "schema_invalid",
    }
    assert {s.value for s in CallStatus} == {"ok", "passthrough", "not_called"}


def test_usage_validation_and_sum() -> None:
    assert Usage() == Usage(0, 0, 0, 0)
    assert Usage(1, 2, 3, 4) + Usage(10, 20, 30, 40) == Usage(11, 22, 33, 44)
    for bad in (-1, 1.0, True, None, "3"):
        with pytest.raises(ValueError):
            Usage(input_tokens=bad)  # type: ignore[arg-type]
    assert Usage().__add__(5) is NotImplemented


def test_successful_result_needs_a_payload_and_an_attempt() -> None:
    assert result().failure is None
    with pytest.raises(ValueError):
        result(payload=None)
    with pytest.raises(ValueError):
        result(attempts=0)


def test_failed_results_may_carry_a_payload_or_none() -> None:
    assert result(failure=FailureClass.SCHEMA_INVALID, payload="not json", attempts=2).attempts == 2
    assert result(failure=FailureClass.REFUSAL, payload=None, stop_reason="refusal").payload is None
    assert result(failure=FailureClass.REQUEST_TOO_LARGE, payload=None, attempts=0, request_id=None,
                  stop_reason=None, usage=Usage(), latency_ms=0).attempts == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"attempts": -1}, {"attempts": True}, {"latency_ms": -0.1}, {"latency_ms": float("nan")},
        {"latency_ms": float("inf")}, {"request_id": "has space"}, {"request_id": "x" * 129},
        {"request_id": "<script>"}, {"stop_reason": "End_Turn"}, {"stop_reason": "free text here"},
        {"usage": None}, {"failure": "refusal"},
    ],
)
def test_result_rejects_malformed_fields(overrides) -> None:
    with pytest.raises(ValueError):
        result(**overrides)


def test_result_repr_never_shows_the_payload() -> None:
    marker = "UNIQUE-DOCUMENT-TEXT-IN-PAYLOAD"
    text = repr(result(payload={"findings": [{"original": marker}]}))
    assert marker not in text and "payload" not in text


def test_call_record_from_result_maps_status() -> None:
    ok = CallRecord.from_result(3, result())
    assert (ok.status, ok.failure, ok.attempts, ok.request_id) == (CallStatus.OK, None, 1, "req_abc-1")
    failed = CallRecord.from_result(4, result(failure=FailureClass.RATE_LIMITED, payload=None, stop_reason=None))
    assert (failed.status, failed.failure) == (CallStatus.PASSTHROUGH, FailureClass.RATE_LIMITED)
    guarded = CallRecord.from_result(5, result(failure=FailureClass.REQUEST_TOO_LARGE, payload=None, attempts=0,
                                               request_id=None, stop_reason=None, usage=Usage(), latency_ms=0))
    assert (guarded.status, guarded.attempts) == (CallStatus.NOT_CALLED, 0)


@pytest.mark.parametrize(
    "fields",
    [
        dict(status=CallStatus.OK, failure=FailureClass.REFUSAL, attempts=1),
        dict(status=CallStatus.OK, failure=None, attempts=0),
        dict(status=CallStatus.PASSTHROUGH, failure=None, attempts=1),
        dict(status=CallStatus.PASSTHROUGH, failure=FailureClass.TIMEOUT, attempts=0),
        dict(status=CallStatus.NOT_CALLED, failure=None, attempts=0),
        dict(status=CallStatus.NOT_CALLED, failure=FailureClass.TIMEOUT, attempts=0),
        dict(status=CallStatus.NOT_CALLED, failure=FailureClass.REQUEST_TOO_LARGE, attempts=1),
        dict(status=CallStatus.NOT_CALLED, failure=FailureClass.REQUEST_TOO_LARGE, attempts=0, usage=Usage(1)),
        dict(status=CallStatus.NOT_CALLED, failure=FailureClass.REQUEST_TOO_LARGE, attempts=0, request_id="r"),
        dict(status=CallStatus.NOT_CALLED, failure=FailureClass.REQUEST_TOO_LARGE, attempts=0, stop_reason="x"),
        dict(status="ok", failure=None, attempts=1),
        dict(status=CallStatus.OK, failure=None, attempts=1, paragraph_id=-1),
        dict(status=CallStatus.OK, failure=None, attempts=1, paragraph_id=True),
    ],
)
def test_call_record_invariants(fields) -> None:
    base = dict(paragraph_id=0, latency_ms=1.0, usage=Usage())
    base.update(fields)
    with pytest.raises(ValueError):
        CallRecord(**base)


def test_call_records_hold_no_text_fields() -> None:
    names = {f.name for f in dataclasses.fields(CallRecord)}
    assert names == {"paragraph_id", "status", "failure", "attempts", "latency_ms", "usage", "request_id",
                     "stop_reason"}


def test_types_are_frozen() -> None:
    record = CallRecord.from_result(0, result())
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.attempts = 2  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result().failure = FailureClass.TIMEOUT  # type: ignore[misc]


def test_s4_types_live_outside_models() -> None:
    from proofreader import models

    for name in ("ProofreadResult", "CallRecord", "Usage", "FailureClass", "CallStatus"):
        assert hasattr(llm, name) and not hasattr(models, name)

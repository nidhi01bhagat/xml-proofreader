"""Proofreader protocol with Anthropic, Fake and Replay implementations.

S4.1: the result and audit types. S4.4: the ``Proofreader`` protocol, the immutable
``ProofreadRequest``, the run-fatal error hierarchy, the deterministic ``FakeProofreader`` and the
``ReplayProofreader`` with its strict recording format and ``RecordingWriter``. The Anthropic client
(S4.5) follows later.

These types live here rather than in ``models.py`` because S4 does not modify S0-S3 files
(docs/04, S4.1 note). ``CallRecord`` is a text-free audit record (one per paragraph) kept separate
from the S3 ``TraceRecord`` (one per finding). ``ProofreadResult.payload`` is the parsed JSON handed
to the resolver; it is never logged or shown in a repr. Recordings store result payloads, which MAY
contain paragraph text (model findings quote the original wording, its anchors and corrections); they
never store request text (the paragraph data block), system prompts, style guides, credentials,
headers or configuration. Real-document recordings therefore stay local and git-ignored.

Paragraph-level failures (the nine ``FailureClass`` values) are RETURNED as ``ProofreadResult``;
run-fatal conditions are RAISED as ``RunFatalError`` subclasses; ``BaseException`` (interrupts,
cancellation) is never caught.
"""

from __future__ import annotations

import contextlib
import copy
import enum
import json
import math
import os
import re
import tempfile
import threading
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol, runtime_checkable

from . import prompt
from .config import RunConfig
from .models import ProofreaderError

_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_STOP_REASON = re.compile(r"[a-z_]{1,40}\Z")


class FailureClass(str, enum.Enum):
    """Why a paragraph passed through unchanged (paragraph level, exit 3; docs/03 section 6 / D5).

    Run-fatal conditions (HTTP 400/401/402/403/404/422 and other non-paragraph-specific 4xx,
    missing key, missing replay recording, ...) are NOT failure classes: they abort the run.
    """

    TIMEOUT = "timeout"  # after the SDK's retries
    CONNECTION = "connection"  # after the SDK's retries
    RATE_LIMITED = "rate_limited"  # HTTP 429 after the SDK's retries
    SERVER_ERROR = "server_error"  # HTTP 5xx / 529 after the SDK's retries
    REQUEST_TOO_LARGE = "request_too_large"  # HTTP 413, or the pre-call size guard (then not called)
    REFUSAL = "refusal"  # stop_reason "refusal": no retry, no fallback (D13)
    MAX_TOKENS = "max_tokens"  # stop_reason "max_tokens"
    CONTEXT_WINDOW_EXCEEDED = "context_window_exceeded"  # stop_reason "model_context_window_exceeded"
    SCHEMA_INVALID = "schema_invalid"  # malformed JSON / envelope after the one schema retry


class CallStatus(str, enum.Enum):
    OK = "ok"  # a parsed payload was obtained and handed to the resolver
    PASSTHROUGH = "passthrough"  # called, failed at paragraph level; paragraph left unchanged
    NOT_CALLED = "not_called"  # no request sent (pre-call size guard)


@dataclass(frozen=True)
class Usage:
    """Token counts reported by the API (all non-negative; zero when unknown or not called)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Usage.{name} must be a non-negative int")

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
        )


def _check_common(owner: str, attempts: object, latency_ms: object, request_id: object, stop_reason: object,
                  usage: object) -> None:
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
        raise ValueError(f"{owner}.attempts must be a non-negative int")
    if isinstance(latency_ms, bool) or not isinstance(latency_ms, (int, float)) or not math.isfinite(latency_ms) \
            or latency_ms < 0:
        raise ValueError(f"{owner}.latency_ms must be a finite non-negative number")
    if request_id is not None and (not isinstance(request_id, str) or not _REQUEST_ID.match(request_id)):
        raise ValueError(f"{owner}.request_id must be None or an identifier of [A-Za-z0-9_-]")
    if stop_reason is not None and (not isinstance(stop_reason, str) or not _STOP_REASON.match(stop_reason)):
        raise ValueError(f"{owner}.stop_reason must be None or a lowercase identifier")
    if not isinstance(usage, Usage):
        raise ValueError(f"{owner}.usage must be a Usage")


@dataclass(frozen=True)
class ProofreadResult:
    """Outcome of proofreading one paragraph (docs/03 section 2, planned S4 type).

    ``payload`` is the parsed JSON response (handed unchanged to ``resolver.resolve_document``),
    or None when nothing parseable was received. With ``failure`` None the call succeeded; with
    SCHEMA_INVALID the (unparseable or invalid) payload may still be passed on for tracing.
    """

    payload: object = field(repr=False)
    failure: FailureClass | None
    usage: Usage
    latency_ms: float
    attempts: int  # HTTP requests sent for this paragraph (transport retries and the schema retry included)
    request_id: str | None = None
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        _check_common("ProofreadResult", self.attempts, self.latency_ms, self.request_id, self.stop_reason,
                      self.usage)
        if self.failure is not None and not isinstance(self.failure, FailureClass):
            raise ValueError("ProofreadResult.failure must be a FailureClass or None")
        if self.failure is None and (self.payload is None or self.attempts < 1):
            raise ValueError("a successful ProofreadResult needs a payload and at least one attempt")


@dataclass(frozen=True)
class CallRecord:
    """Text-free audit record of the model call for one paragraph (separate from S3 TraceRecord)."""

    paragraph_id: int
    status: CallStatus
    failure: FailureClass | None
    attempts: int
    latency_ms: float
    usage: Usage
    request_id: str | None = None
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.paragraph_id, bool) or not isinstance(self.paragraph_id, int) or self.paragraph_id < 0:
            raise ValueError("CallRecord.paragraph_id must be a non-negative int")
        if not isinstance(self.status, CallStatus):
            raise ValueError("CallRecord.status must be a CallStatus")
        if self.failure is not None and not isinstance(self.failure, FailureClass):
            raise ValueError("CallRecord.failure must be a FailureClass or None")
        _check_common("CallRecord", self.attempts, self.latency_ms, self.request_id, self.stop_reason, self.usage)
        if self.status is CallStatus.OK and (self.failure is not None or self.attempts < 1):
            raise ValueError("an OK CallRecord has no failure and at least one attempt")
        if self.status is CallStatus.PASSTHROUGH and (self.failure is None or self.attempts < 1):
            raise ValueError("a PASSTHROUGH CallRecord has a failure and at least one attempt")
        if self.status is CallStatus.NOT_CALLED and (
            self.failure is not FailureClass.REQUEST_TOO_LARGE or self.attempts != 0 or self.request_id is not None
            or self.stop_reason is not None or self.usage != Usage()
        ):
            raise ValueError("a NOT_CALLED CallRecord comes only from the size guard: no attempts, no usage")

    @classmethod
    def from_result(cls, paragraph_id: int, result: ProofreadResult) -> "CallRecord":
        return cls(
            paragraph_id=paragraph_id,
            status=(CallStatus.OK if result.failure is None
                    else CallStatus.NOT_CALLED if result.attempts == 0 else CallStatus.PASSTHROUGH),
            failure=result.failure,
            attempts=result.attempts,
            latency_ms=result.latency_ms,
            usage=result.usage,
            request_id=result.request_id,
            stop_reason=result.stop_reason,
        )


# ============================================================================================
# S4.4: errors
# ============================================================================================


class RunFatalError(ProofreaderError):
    """Aborts the whole run: no output is written (exit 1). Never used for paragraph-level failures."""

    exit_code = 1


class ProviderFatalError(RunFatalError):
    """Provider/configuration failure that affects every request (S4.5: HTTP 400/401/402/403/404/422 ...)."""


class RecordingError(RunFatalError):
    """A replay recording is unreadable, invalid, unsafe to write, or would overwrite a file."""


class ReplayMissError(RecordingError):
    """No recorded outcome exists for a request fingerprint."""


class FakeScriptError(AssertionError):
    """Test-harness only: the FakeProofreader was called in a way its script does not allow."""


# ============================================================================================
# S4.4: request and protocol
# ============================================================================================

_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_CANONICAL = {"sort_keys": True, "ensure_ascii": False, "separators": (",", ":"), "allow_nan": False}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class ProofreadRequest:
    """One model request. Immutable: the schema is held as a canonical JSON string, never a live dict,
    and the fingerprint is re-verified from the stored data on construction. ``repr`` never shows the
    system prompt, user message or schema."""

    paragraph_id: int
    model: str
    effort: str
    max_tokens: int
    fingerprint: str
    system: str = field(repr=False)
    user: str = field(repr=False)
    schema_json: str = field(repr=False)

    def __post_init__(self) -> None:
        if not _is_int(self.paragraph_id) or self.paragraph_id < 0:
            raise ValueError("paragraph_id must be a non-negative int")
        if not _is_int(self.max_tokens) or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive int")
        for name in ("model", "effort"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("system", "user", "schema_json"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if not isinstance(self.fingerprint, str) or not _FINGERPRINT.match(self.fingerprint):
            raise ValueError("fingerprint must be 64 lowercase hex characters")
        try:
            schema = json.loads(self.schema_json)
        except ValueError:
            raise ValueError("schema_json is not valid JSON") from None
        if not isinstance(schema, dict) or json.dumps(schema, **_CANONICAL) != self.schema_json:
            raise ValueError("schema_json must be the canonical JSON of a schema object")
        expected = prompt.request_fingerprint(model=self.model, effort=self.effort, max_tokens=self.max_tokens,
                                              system=self.system, user=self.user, schema=schema)
        if expected != self.fingerprint:
            raise ValueError("fingerprint does not match the request data")

    @property
    def schema(self) -> dict[str, object]:
        """A fresh, independent copy of the request schema on every access."""
        return json.loads(self.schema_json)

    def __repr__(self) -> str:
        return (f"ProofreadRequest(paragraph_id={self.paragraph_id}, model={self.model!r}, effort={self.effort!r}, "
                f"max_tokens={self.max_tokens}, fingerprint={self.fingerprint[:12]}...)")


def make_request(config: RunConfig, paragraph_id: int, *, system: str, user: str,
                 schema: dict[str, object]) -> ProofreadRequest:
    """Build a request from the RunConfig (S4.1 policy). The schema is snapshotted once into canonical
    JSON; the fingerprint is computed by ``prompt.fingerprint_for`` from exactly that snapshot."""
    if not isinstance(schema, dict):
        raise ValueError("schema must be a JSON object (dict)")
    prompt.validate_json_value(schema, "schema")
    schema_json = json.dumps(schema, **_CANONICAL)
    fingerprint = prompt.fingerprint_for(config, system=system, user=user, schema=json.loads(schema_json))
    return ProofreadRequest(paragraph_id, config.model, config.effort, config.max_tokens, fingerprint,
                            system, user, schema_json)


@runtime_checkable
class Proofreader(Protocol):
    """Proofreads one paragraph request. Paragraph-level failures are returned in the result;
    run-fatal failures are raised as ``RunFatalError``."""

    def proofread(self, request: ProofreadRequest) -> ProofreadResult: ...


# ============================================================================================
# S4.4: FakeProofreader (offline tests)
# ============================================================================================


@dataclass(frozen=True)
class FakeOk:
    """Return ``payload`` (deep-copied at construction and again for every call)."""

    payload: object = field(repr=False)
    attempts: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", copy.deepcopy(self.payload))


@dataclass(frozen=True)
class FakeFailure:
    """Return a paragraph-level failure of class ``failure``."""

    failure: FailureClass
    payload: object = field(default=None, repr=False)
    stop_reason: str | None = None
    attempts: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.failure, FailureClass):
            raise ValueError("FakeFailure.failure must be a FailureClass")
        object.__setattr__(self, "payload", copy.deepcopy(self.payload))


@dataclass(frozen=True)
class FakeRaise:
    """Raise ``exception`` (run-fatal errors, cancellation, unexpected exceptions)."""

    exception: BaseException


@dataclass(frozen=True)
class FakeBlock:
    """Wait until ``release`` is set (at most ``timeout_s``), then serve ``then``. ``started`` (optional)
    is set as soon as the call is in flight. An unreleased block raises instead of hanging."""

    release: threading.Event
    then: object
    timeout_s: float = 5.0
    started: threading.Event | None = None


FakeOutcome = FakeOk | FakeFailure | FakeRaise | FakeBlock


@dataclass(frozen=True)
class FakeCall:
    paragraph_id: int
    fingerprint: str
    sequence_index: int


class FakeProofreader:
    """Deterministic scripted client. Strict by default: a call past the end of a paragraph's script,
    or for an unscripted paragraph, raises ``FakeScriptError`` (opt-ins: ``repeat_last``, ``default``)."""

    def __init__(self, script: Mapping[int, Sequence[FakeOutcome]] | None = None, *,
                 default: FakeOutcome | None = None, repeat_last: bool = False, latency_ms: float = 0.0,
                 usage: Usage | None = None) -> None:
        self._script: dict[int, tuple[FakeOutcome, ...]] = {}
        for paragraph_id, outcomes in (script or {}).items():
            if not _is_int(paragraph_id) or paragraph_id < 0:
                raise ValueError("script keys must be non-negative int paragraph ids")
            outcomes = tuple(outcomes)
            if not outcomes or not all(_is_outcome(o) for o in outcomes):
                raise ValueError("each script entry must be a non-empty sequence of Fake outcomes")
            self._script[paragraph_id] = outcomes
        if default is not None and not _is_outcome(default):
            raise ValueError("default must be a Fake outcome or None")
        self._default = default
        self._repeat_last = bool(repeat_last)
        self._latency_ms = latency_ms
        self._usage = usage if usage is not None else Usage(1, 1)
        self._lock = threading.Lock()
        self._next: dict[int, int] = {}
        self._calls: list[FakeCall] = []
        self._in_flight = 0
        self._max_in_flight = 0

    def proofread(self, request: ProofreadRequest) -> ProofreadResult:
        if not isinstance(request, ProofreadRequest):
            raise TypeError("request must be a ProofreadRequest")
        pid = request.paragraph_id
        with self._lock:
            index = self._next.get(pid, 0)
            outcomes = self._script.get(pid)
            if outcomes is not None:
                if index < len(outcomes):
                    outcome = outcomes[index]
                elif self._repeat_last:
                    outcome = outcomes[-1]
                else:
                    raise FakeScriptError(f"paragraph {pid}: unexpected call {index + 1} "
                                          f"(script has {len(outcomes)})")
            elif self._default is not None:
                outcome = self._default
            else:
                raise FakeScriptError(f"paragraph {pid}: unexpected call (paragraph not scripted)")
            self._next[pid] = index + 1
            self._calls.append(FakeCall(pid, request.fingerprint, index))
            self._in_flight += 1
            self._max_in_flight = max(self._max_in_flight, self._in_flight)
        try:
            return self._serve(outcome)
        finally:
            with self._lock:
                self._in_flight -= 1

    def _serve(self, outcome: FakeOutcome) -> ProofreadResult:
        while isinstance(outcome, FakeBlock):
            if outcome.started is not None:
                outcome.started.set()
            if not outcome.release.wait(outcome.timeout_s):
                raise FakeScriptError(f"blocked call not released within {outcome.timeout_s} s")
            outcome = outcome.then
        if isinstance(outcome, FakeRaise):
            raise outcome.exception
        if isinstance(outcome, FakeOk):
            return ProofreadResult(copy.deepcopy(outcome.payload), None, self._usage, self._latency_ms,
                                   outcome.attempts, None, "end_turn")
        if isinstance(outcome, FakeFailure):
            return ProofreadResult(copy.deepcopy(outcome.payload), outcome.failure, self._usage, self._latency_ms,
                                   outcome.attempts, None, outcome.stop_reason)
        raise FakeScriptError("FakeBlock.then must be a Fake outcome")

    @property
    def calls(self) -> tuple[FakeCall, ...]:
        with self._lock:
            return tuple(self._calls)

    def call_counts(self) -> dict[int, int]:
        with self._lock:
            return dict(Counter(call.paragraph_id for call in self._calls))

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def max_in_flight(self) -> int:
        with self._lock:
            return self._max_in_flight

    def __repr__(self) -> str:
        return f"FakeProofreader(scripted={len(self._script)}, calls={len(self.calls)})"


def _is_outcome(value: object) -> bool:
    return isinstance(value, (FakeOk, FakeFailure, FakeRaise, FakeBlock))


# ============================================================================================
# S4.4: recordings, ReplayProofreader, RecordingWriter
# ============================================================================================

RECORDING_FORMAT = "proofreader-recording"
RECORDING_VERSION = 1
RECORDING_SUFFIX = ".recording.json"
MAX_RECORDING_BYTES = 50 * 1024 * 1024
_HEADER_KEYS = frozenset({"format", "version", "fingerprint_version", "prompt_version", "entries"})
_ENTRY_KEYS = frozenset({"fingerprint", "result", "failure", "payload", "usage", "attempts", "latency_ms",
                         "request_id", "stop_reason"})
_USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
# The repository's existing secret patterns (tests/test_s0_foundation.py). Defence in depth only: not
# every possible credential format is detectable.
_SECRET_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"ANTHROPIC_API_KEY\s*=\s*['\"]?[A-Za-z0-9_\-]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),  # Google API key shape (Gemini, S5)
    re.compile(r"GEMINI_API_KEY\s*=\s*['\"]?[A-Za-z0-9_\-]{8,}"),
)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate JSON object key")
    return dict(pairs)


def _reject_constant(name: str) -> object:
    raise ValueError(f"non-standard JSON constant {name}")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):  # e.g. 1e400 overflows to inf without reaching parse_constant
        raise ValueError("non-finite JSON number")
    return value


def _strict_json(text: str) -> object:
    return json.loads(text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant,
                      parse_float=_finite_float)


def load_recording(path: str | Path) -> dict[str, ProofreadResult]:
    """Strictly load a recording: {fingerprint: ProofreadResult}. Any problem -> RecordingError.
    Messages name keys and entry positions, never values."""
    path = Path(path)
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_RECORDING_BYTES + 1)
    except OSError:
        raise RecordingError(f"recording {path.name!r} cannot be read") from None
    if len(data) > MAX_RECORDING_BYTES:
        raise RecordingError(f"recording is larger than {MAX_RECORDING_BYTES} bytes")
    try:
        document = _strict_json(data.decode("utf-8"))
    except UnicodeDecodeError:
        raise RecordingError("recording is not valid UTF-8") from None
    except RecursionError:  # pathologically deep nesting: not a ValueError, must not escape as a crash
        raise RecordingError("recording is nested too deeply") from None
    except ValueError as exc:
        reason = "duplicate JSON object key" if "duplicate" in str(exc) else \
            "non-standard JSON constant" if "constant" in str(exc) else \
            "non-finite JSON number" if "non-finite" in str(exc) else "invalid JSON"
        raise RecordingError(f"recording is not strict JSON ({reason})") from None
    if not isinstance(document, dict) or set(document) != _HEADER_KEYS:
        raise RecordingError("recording must have exactly the keys: " + ", ".join(sorted(_HEADER_KEYS)))
    if document["format"] != RECORDING_FORMAT:
        raise RecordingError("recording has an unsupported format")
    if not _is_int(document["version"]) or document["version"] != RECORDING_VERSION:
        raise RecordingError("recording has an unsupported version")
    if document["fingerprint_version"] != prompt.FINGERPRINT_VERSION:
        raise RecordingError("recording was made with a different fingerprint version (stale)")
    if document["prompt_version"] != prompt.PROMPT_VERSION:
        raise RecordingError("recording was made with a different prompt version (stale)")
    entries = document["entries"]
    if not isinstance(entries, list):
        raise RecordingError("recording entries must be a list")
    results: dict[str, ProofreadResult] = {}
    for index, entry in enumerate(entries):
        fingerprint, result = _parse_entry(index, entry)
        if fingerprint in results:
            raise RecordingError(f"recording entry {index}: duplicate fingerprint")
        results[fingerprint] = result
    return results


def _parse_entry(index: int, entry: object) -> tuple[str, ProofreadResult]:
    where = f"recording entry {index}"
    if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
        raise RecordingError(f"{where}: must have exactly the keys: " + ", ".join(sorted(_ENTRY_KEYS)))
    fingerprint = entry["fingerprint"]
    if not isinstance(fingerprint, str) or not _FINGERPRINT.match(fingerprint):
        raise RecordingError(f"{where}: fingerprint must be 64 lowercase hex characters")
    if entry["result"] == "ok":
        if entry["failure"] is not None:
            raise RecordingError(f"{where}: an ok entry must have failure null")
        failure = None
    elif entry["result"] == "failure":
        try:
            failure = FailureClass(entry["failure"])
        except ValueError:
            raise RecordingError(f"{where}: unknown failure class") from None
    else:
        raise RecordingError(f"{where}: result must be 'ok' or 'failure'")
    try:  # same JSON-safety rules as the writer (depth limit, string keys, finite numbers)
        prompt.validate_json_value(entry["payload"], "payload")
    except ValueError:  # the validator's path may contain payload keys: not echoed
        raise RecordingError(f"{where}: payload is not JSON-safe") from None
    usage = entry["usage"]
    if not isinstance(usage, dict) or set(usage) != set(_USAGE_KEYS):
        raise RecordingError(f"{where}: usage must have exactly the keys: " + ", ".join(_USAGE_KEYS))
    try:
        result = ProofreadResult(entry["payload"], failure, Usage(**usage), entry["latency_ms"], entry["attempts"],
                                 entry["request_id"], entry["stop_reason"])
    except ValueError as exc:  # S4.1 validation messages name fields only, never values
        raise RecordingError(f"{where}: {exc}") from None
    return fingerprint, result


class ReplayProofreader:
    """Serves recorded outcomes by request fingerprint. Immutable after construction; every result gets
    an independent payload copy, so mutating one result never affects later ones."""

    def __init__(self, results: Mapping[str, ProofreadResult]) -> None:
        self._results: dict[str, ProofreadResult] = {}
        for fingerprint, result in results.items():
            if not isinstance(fingerprint, str) or not _FINGERPRINT.match(fingerprint):
                raise ValueError("recording keys must be 64 lowercase hex fingerprints")
            if not isinstance(result, ProofreadResult):
                raise ValueError("recording values must be ProofreadResult")
            self._results[fingerprint] = replace(result, payload=copy.deepcopy(result.payload))

    @classmethod
    def from_file(cls, path: str | Path) -> "ReplayProofreader":
        return cls(load_recording(path))

    def proofread(self, request: ProofreadRequest) -> ProofreadResult:
        if not isinstance(request, ProofreadRequest):
            raise TypeError("request must be a ProofreadRequest")
        result = self._results.get(request.fingerprint)
        if result is None:
            raise ReplayMissError(f"no recorded response for request {request.fingerprint[:12]}... "
                                  f"(paragraph {request.paragraph_id})")
        return replace(result, payload=copy.deepcopy(result.payload))

    def missing(self, fingerprints: Iterable[str]) -> tuple[str, ...]:
        """The given fingerprints that have no recorded outcome, in first-seen order (for the S4.6 pre-check)."""
        seen: dict[str, None] = {}
        for fingerprint in fingerprints:
            if fingerprint not in self._results:
                seen.setdefault(fingerprint, None)
        return tuple(seen)

    def __len__(self) -> int:
        return len(self._results)

    def __repr__(self) -> str:
        return f"ReplayProofreader(entries={len(self._results)})"


class RecordingWriter:
    """Collects (fingerprint, ProofreadResult) pairs and publishes them atomically.

    It never receives a request, prompt, style guide, configuration, credential or header, and it
    serialises an explicit whitelist of result fields only. The model's parsed response payload IS
    recorded as-is: its finding fields (``original``, ``before``, ``after``) quote fragments of the
    source paragraph, so a recording is not text-free and must stay local (``*.recording.json`` is
    git-ignored). Before publishing, the text is scanned for
    the repository's secret patterns (defence in depth). ``write(..., force=False)`` refuses an existing
    target; publication is exclusive and atomic via ``os.link`` (fails closed if unsupported)."""

    def __init__(self) -> None:
        self._entries: dict[str, str] = {}  # fingerprint -> canonical JSON of the whitelisted entry

    def add(self, fingerprint: str, result: ProofreadResult) -> None:
        if not isinstance(fingerprint, str) or not _FINGERPRINT.match(fingerprint):
            raise RecordingError("recording fingerprint must be 64 lowercase hex characters")
        if not isinstance(result, ProofreadResult):
            raise RecordingError("only ProofreadResult values can be recorded")
        if fingerprint in self._entries:
            raise RecordingError(f"duplicate recording fingerprint {fingerprint[:12]}...")
        try:
            prompt.validate_json_value(result.payload, "payload")
        except ValueError:  # the validator's path may contain payload keys (payload content): not echoed
            raise RecordingError("recording payload is not JSON-safe") from None
        entry = {
            "fingerprint": fingerprint,
            "result": "ok" if result.failure is None else "failure",
            "failure": None if result.failure is None else result.failure.value,
            "payload": result.payload,
            "usage": {name: getattr(result.usage, name) for name in _USAGE_KEYS},
            "attempts": result.attempts,
            "latency_ms": result.latency_ms,
            "request_id": result.request_id,
            "stop_reason": result.stop_reason,
        }
        self._entries[fingerprint] = json.dumps(entry, **_CANONICAL)  # snapshot: later mutation cannot leak in

    def __len__(self) -> int:
        return len(self._entries)

    def render(self) -> str:
        """Deterministic serialisation: header constants plus entries sorted by fingerprint."""
        document = {
            "format": RECORDING_FORMAT,
            "version": RECORDING_VERSION,
            "fingerprint_version": prompt.FINGERPRINT_VERSION,
            "prompt_version": prompt.PROMPT_VERSION,
            "entries": [json.loads(self._entries[fp]) for fp in sorted(self._entries)],
        }
        return json.dumps(document, **_CANONICAL) + "\n"

    def write(self, path: str | Path, *, force: bool = False) -> None:
        path = Path(path)
        if not path.name.endswith(RECORDING_SUFFIX):
            raise RecordingError(f"recording file names must end with {RECORDING_SUFFIX!r} (git-ignored)")
        text = self.render()
        if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
            raise RecordingError("recording contains a credential-like value; nothing was written")
        data = text.encode("utf-8")
        directory = path.parent if str(path.parent) else Path(".")
        if not force and (path.exists() or path.is_symlink()):
            raise RecordingError(f"refusing to overwrite existing recording {path.name!r} (use force)")
        # The temporary file is in the destination directory (same filesystem, so link/replace are atomic)
        # and its name also ends with RECORDING_SUFFIX, so even a file left behind by a crash is git-ignored.
        try:
            descriptor, temp_name = tempfile.mkstemp(prefix=".tmp-", suffix=RECORDING_SUFFIX, dir=directory)
        except OSError:
            raise RecordingError("cannot create a temporary file next to the recording") from None
        temp = Path(temp_name)
        try:
            try:
                handle = os.fdopen(descriptor, "wb")
            except BaseException:
                os.close(descriptor)  # fdopen did not take ownership
                raise
            with handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if force:
                os.replace(temp, path)  # atomic replacement
            else:
                try:
                    os.link(temp, path)  # exclusive and atomic: fails if the target exists (also during a race)
                except FileExistsError:
                    raise RecordingError(f"refusing to overwrite existing recording {path.name!r} (use force)") \
                        from None
                except (OSError, NotImplementedError, AttributeError):
                    raise RecordingError("atomic exclusive publication failed (unsupported filesystem or I/O "
                                         "error); nothing was written") from None
        except RecordingError:
            raise
        except OSError:
            raise RecordingError("writing the recording failed; nothing was published") from None
        finally:
            with contextlib.suppress(FileNotFoundError, OSError):
                temp.unlink()


# ============================================================================================
# S4.5: Anthropic client
# ============================================================================================

MAX_TRANSPORT_RETRIES = 2  # approved: 2 retries for transient failures (408/409/429/5xx/529, timeout, connection)
MAX_SCHEMA_RETRIES = 1  # approved: one fresh, identical request when the output is unusable
BACKOFF_BASE_S = 0.5
BACKOFF_CAP_S = 8.0
RETRY_AFTER_CAP_S = 30.0
_IGNORED_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})
_RUN_FATAL_STATUSES = frozenset({400, 401, 402, 403, 404, 422})
_TRANSIENT_FAILURES = {408: FailureClass.TIMEOUT, 409: FailureClass.SERVER_ERROR, 429: FailureClass.RATE_LIMITED}
_FINAL_STOPS = {"refusal": FailureClass.REFUSAL, "max_tokens": FailureClass.MAX_TOKENS,
                "model_context_window_exceeded": FailureClass.CONTEXT_WINDOW_EXCEEDED}
_OK_STOPS = frozenset({"end_turn", "stop_sequence"})


def decode_response(text: str) -> object:
    """Strict JSON decoding of the model's output text. Never repairs: duplicate keys, NaN/Infinity,
    non-finite numbers, trailing data and pathological nesting all raise ValueError."""
    if not isinstance(text, str):
        raise ValueError("response text must be a string")
    try:
        return _strict_json(text)
    except RecursionError:
        raise ValueError("response JSON is nested too deeply") from None


def _envelope_invalid(paragraph_id: int, payload: object) -> bool:
    """True when the S3 resolver would reject the whole response as SCHEMA_INVALID (not an object, no
    ``findings`` list, extra top-level keys). Per-finding problems and TOO_MANY_FINDINGS are NOT envelope
    failures: they reach the resolver unchanged (no schema retry)."""
    from .models import RejectCode
    from .resolver import parse_findings

    _, rejections, _ = parse_findings(paragraph_id, payload)
    return any(r.finding_index is None and r.code is RejectCode.SCHEMA_INVALID for r in rejections)


def _safe_request_id(value: object) -> str | None:
    return value if isinstance(value, str) and _REQUEST_ID.match(value) else None


def _safe_stop_reason(value: object) -> str | None:
    return value if isinstance(value, str) and _STOP_REASON.match(value) else None


def _usage_of(message: object) -> Usage:
    usage = getattr(message, "usage", None)

    def count(name: str) -> int:
        value = getattr(usage, name, None)
        return value if _is_int(value) and value >= 0 else 0

    return Usage(count("input_tokens"), count("output_tokens"), count("cache_creation_input_tokens"),
                 count("cache_read_input_tokens"))


def _single_text(message: object) -> str | None:
    """The one text block of a structured-output response, or None if the content is missing, empty,
    has several text blocks or contains any block type other than (ignored) thinking blocks."""
    content = getattr(message, "content", None)
    if not isinstance(content, list) or not content:
        return None
    texts = []
    for block in content:
        kind = getattr(block, "type", None)
        if kind == "text":
            text = getattr(block, "text", None)
            if not isinstance(text, str):
                return None
            texts.append(text)
        elif kind not in _IGNORED_BLOCK_TYPES:
            return None
    return texts[0] if len(texts) == 1 else None


def _retry_delay(retry_index: int, headers: object) -> float:
    """Bounded exponential backoff; a numeric ``retry-after-ms`` / ``retry-after`` header is honoured up to
    RETRY_AFTER_CAP_S (an HTTP-date value is ignored)."""
    delay = min(BACKOFF_BASE_S * (2 ** retry_index), BACKOFF_CAP_S)
    get = getattr(headers, "get", None)
    if callable(get):
        for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
            raw = get(name)
            try:
                value = float(raw) * scale if raw is not None else None
            except (TypeError, ValueError):
                value = None
            if value is not None and math.isfinite(value) and value >= 0:
                return min(value, RETRY_AFTER_CAP_S)
    return delay


def _refuse_sdk_override_env() -> None:
    """Fail closed if ``ANTHROPIC_CUSTOM_HEADERS`` is in ``os.environ`` when the client is built. The S4.1
    check (``config.build_config``) validates the environment once; the SDK re-reads ``os.environ`` at
    construction, and an ``x-api-key`` / ``Authorization`` line there replaces the configured key. Same
    rule and exit code as S4.1 (empty means unset). The value is never echoed."""
    from .config import SDK_OVERRIDE_ENV, UsageError

    for name in SDK_OVERRIDE_ENV:
        if os.environ.get(name):
            raise UsageError(f"{name} is set; it is not supported because the Anthropic SDK would add its headers "
                             "to every request (it can replace the API key). Unset it and run again.")


def _refuse_sdk_debug_logging() -> None:
    """Fail closed (run-fatal, before any request) if the SDK would log request options: at DEBUG the
    ``anthropic._base_client`` logger writes the request body (system prompt, user message with the
    paragraph and style-guide data, schema). This is a guard, not logging configuration: pinning the SDK
    loggers to WARNING regardless of ``ANTHROPIC_LOG`` remains S4.7."""
    import logging

    if logging.getLogger("anthropic._base_client").isEnabledFor(logging.DEBUG):
        raise RunFatalError("Anthropic SDK debug logging is enabled (for example ANTHROPIC_LOG=debug); it would "
                            "log request content. Disable it and run again.")


@dataclass
class _Tally:
    attempts: int = 0
    usage: Usage = field(default_factory=Usage)


class AnthropicProofreader:
    """Live client (S4.5). Built only from the RunConfig (S4.1): explicit API key, base URL, timeout,
    telemetry off, environment proxies ignored, SDK-internal retries off (``max_retries=0``) so that every
    HTTP request is counted. Transient failures are retried here (bounded); unusable output gets one schema
    retry; paragraph-level failures are returned, run-fatal ones raised. Nothing is logged."""

    def __init__(self, config: RunConfig, *, http_client: object = None, recorder: RecordingWriter | None = None,
                 max_retries: int = MAX_TRANSPORT_RETRIES, sleep=None, clock=None) -> None:
        import time

        import anthropic
        import httpx2

        if not _is_int(max_retries) or not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be an int from 0 to 5")
        if recorder is not None and not isinstance(recorder, RecordingWriter):
            raise ValueError("recorder must be a RecordingWriter or None")
        _refuse_sdk_override_env()  # the SDK re-reads os.environ here, after config validation
        key = config.require_api_key()  # MissingApiKeyError (exit 1) before any request
        self._anthropic = anthropic
        self._model, self._effort, self._max_tokens = config.model, config.effort, config.max_tokens
        self._timeout = config.timeout_seconds
        self._host = config.base_url.host
        self._max_retries = max_retries
        self._sleep = sleep if sleep is not None else time.sleep
        self._clock = clock if clock is not None else time.monotonic
        if http_client is None:  # no environment proxies / CA overrides / netrc (S4.1 SDK-environment findings)
            http_client = httpx2.Client(trust_env=False, timeout=config.timeout_seconds)
        self._http_client = http_client
        self._client = anthropic.Anthropic(
            api_key=key.reveal(), base_url=config.base_url.url, timeout=config.timeout_seconds, max_retries=0,
            open_telemetry={"traces": False, "propagation": False}, http_client=http_client,
        )
        self._recorder = recorder
        self._recorded: set[str] = set()
        self._record_lock = threading.Lock()

    def __repr__(self) -> str:
        return f"AnthropicProofreader(model={self._model!r}, host={self._host!r}, max_retries={self._max_retries})"

    def close(self) -> None:
        self._client.close()

    # -- protocol ---------------------------------------------------------------------------

    def proofread(self, request: ProofreadRequest) -> ProofreadResult:
        if not isinstance(request, ProofreadRequest):
            raise TypeError("request must be a ProofreadRequest")
        if (request.model, request.effort, request.max_tokens) != (self._model, self._effort, self._max_tokens):
            raise ValueError("request settings do not match this client's configuration")
        _refuse_sdk_debug_logging()
        start = self._clock()
        tally = _Tally()
        payload: object = None
        request_id: str | None = None
        stop_reason: str | None = None
        for schema_round in range(MAX_SCHEMA_RETRIES + 1):
            kind, value, request_id = self._call(request, tally)
            if kind == "failure":
                return self._finish(request, None, value, tally, start, request_id, None)
            payload, stop_reason = None, None
            if kind == "message":
                tally.usage = tally.usage + _usage_of(value)
                stop_reason = _safe_stop_reason(getattr(value, "stop_reason", None))
                if stop_reason in _FINAL_STOPS:  # refusal / truncation: final, never retried
                    return self._finish(request, None, _FINAL_STOPS[stop_reason], tally, start, request_id,
                                        stop_reason)
                text = _single_text(value) if stop_reason in _OK_STOPS else None
                if text is not None:
                    try:
                        payload = decode_response(text)
                    except ValueError:
                        payload = None
                    else:
                        if not _envelope_invalid(request.paragraph_id, payload):
                            return self._finish(request, payload, None, tally, start, request_id, stop_reason)
            # unusable output: fall through to the (single) schema retry
        return self._finish(request, payload, FailureClass.SCHEMA_INVALID, tally, start, request_id, stop_reason)

    # -- internals --------------------------------------------------------------------------

    def _send(self, request: ProofreadRequest):
        return self._client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            system=request.system,
            messages=[{"role": "user", "content": request.user}],
            output_config={"effort": request.effort, "format": {"type": "json_schema", "schema": request.schema}},
            timeout=self._timeout,
        )

    def _call(self, request: ProofreadRequest, tally: _Tally) -> tuple[str, object, str | None]:
        """One logical call with bounded transport retries. Returns ("message", Message, request_id),
        ("invalid", None, request_id) for an unparseable provider response, or ("failure", FailureClass,
        request_id). Raises ProviderFatalError for run-fatal provider errors."""
        anthropic = self._anthropic
        for retry_index in range(self._max_retries + 1):
            tally.attempts += 1
            headers = None
            try:
                message = self._send(request)
            except anthropic.APIStatusError as exc:
                status = exc.status_code
                headers = getattr(exc.response, "headers", None)
                request_id = _safe_request_id(headers.get("request-id") if headers is not None else None)
                if status == 413:
                    return "failure", FailureClass.REQUEST_TOO_LARGE, request_id
                if status >= 500:
                    failure = FailureClass.SERVER_ERROR
                elif status in _TRANSIENT_FAILURES:
                    failure = _TRANSIENT_FAILURES[status]
                else:  # 400/401/402/403/404/422 and every other 4xx: run-fatal (approved S4 decisions)
                    suffix = f", request id {request_id}" if request_id else ""
                    raise ProviderFatalError(f"Anthropic API rejected the request (HTTP {status}{suffix}); "
                                             "the run was stopped") from None
            except anthropic.APITimeoutError:
                failure, request_id = FailureClass.TIMEOUT, None
            except anthropic.APIConnectionError:
                failure, request_id = FailureClass.CONNECTION, None
            except anthropic.APIResponseValidationError:
                return "invalid", None, None
            except anthropic.AnthropicError as exc:
                raise ProviderFatalError(f"Anthropic client error ({type(exc).__name__}); the run was stopped") \
                    from None
            else:
                return "message", message, _safe_request_id(getattr(message, "_request_id", None))
            if retry_index < self._max_retries:
                self._sleep(_retry_delay(retry_index, headers))
        return "failure", failure, request_id

    def _finish(self, request: ProofreadRequest, payload: object, failure: FailureClass | None, tally: _Tally,
                start: float, request_id: str | None, stop_reason: str | None) -> ProofreadResult:
        latency_ms = max(0.0, (self._clock() - start) * 1000.0)
        result = ProofreadResult(payload, failure, tally.usage, latency_ms, tally.attempts, request_id, stop_reason)
        if self._recorder is not None:
            with self._record_lock:
                if request.fingerprint not in self._recorded:  # identical requests are recorded once
                    self._recorder.add(request.fingerprint, result)
                    self._recorded.add(request.fingerprint)
        return result

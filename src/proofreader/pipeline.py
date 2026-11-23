"""Orchestration (S4.6): scan, eligibility, pre-call checks, bounded concurrent model calls, resolve,
annotate (render + verify + rollback), trace.

``run`` turns input bytes into VERIFIED output bytes. It reads no files and no environment and builds no
client: the CLI (S4.7) loads the style guide, chooses the Proofreader (Fake / Replay / Anthropic), writes the
output atomically and maps the result to an exit code.

Order of work (docs/03 section 5; decisions Q1-Q5 approved 2026-11-22):

1. ``config.cache`` must be False: prompt caching is not implemented, so the flag is refused (UsageError,
   exit 2) before anything else, never silently ignored (Q3).
2. ``xml_source.scan`` (encoding check included): fatal input errors are raised before any call.
3. Eligibility, in paragraph-id order: paragraphs with a skip reason (``<error>`` present, CDATA-only,
   unmappable, ...) and whitespace-only paragraphs are never sent and get no CallRecord (AC-11, AC-25).
   A paragraph over the pre-call size limit is not sent either; it gets a NOT_CALLED / REQUEST_TOO_LARGE
   CallRecord and passes through unchanged. A paragraph with non-whitespace text that is skipped as
   PARA_UNMAPPABLE or DOCUMENT_UNVERIFIABLE (S1 X10 / C2: it could not be proofread safely) is listed in
   ``not_proofread`` and makes the run degraded (exit 3): its bytes pass through unchanged, but it must not be
   reported as success (S4.8c fix). Already-annotated, empty and CDATA-only paragraphs are intentional skips.
4. Pre-call checks, before ANY request: ``--max-calls`` (counted in paragraphs to be sent; one paragraph may
   use up to 6 HTTP requests through retries) -> UsageError (exit 2); for a ReplayProofreader every request
   fingerprint must be recorded -> ReplayMissError (exit 1).
5. Bounded fan-out: a thread pool with ``min(concurrency, requests)`` workers, so at most ``concurrency``
   calls are ever in flight (AC-28). Results are keyed by paragraph id, never by completion order (AC-20).
6. Only SUCCESSFUL results (``failure is None``) reach the resolver; failed, refused, truncated or
   schema-invalid payloads never do (Q1). Their paragraphs pass through unchanged and the run is degraded.
7. With no applicable style-guide text, every resolved ``styleguide`` finding is rejected before annotation
   and traced as REJECTED_S3 / INVALID_FIELD (the existing "wrong value of a field" code; Q2).
8. ``annotation.annotate`` (verification and per-paragraph rollback). A document-level failure raises
   DocumentVerificationError (exit 1): no partial output is ever returned (Q5).

Fatal errors during the calls (Q4): every call must pass a start gate before it may invoke the provider.
The gate's two transitions, "queued -> started" (``try_start``) and "running -> stopped" (``stop``), happen
under ONE lock, so they are totally ordered: a call is started only if the gate was not yet stopped at that
instant, and after ``stop`` no call can ever be started. A failing call stops the gate before its worker can
take more work; queued calls are also cancelled. A call started before the stop (it may still be entering
the provider) is allowed to finish, as are running calls (they cannot be interrupted safely); no worker
thread is left running when ``run`` returns or raises. Of the failures OBSERVED by then, the one with the lowest paragraph id
is raised (a BaseException such as KeyboardInterrupt takes precedence over ordinary exceptions). Because
cancellation stops other calls, the observed set depends on timing: the choice is deterministic only for a
given set of observed failures, not globally. A KeyboardInterrupt in the calling thread cancels pending
calls the same way and propagates after running calls finish.

Records never contain document text or credentials: CallRecords hold ids, statuses, counts and request ids;
TraceRecords hold ids, codes and offsets; error messages carry counts only.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

from . import annotation as an
from . import prompt
from . import quotes as q
from . import resolver as r
from . import xml_source as xs
from .config import RunConfig, UsageError
from .llm import (
    CallRecord,
    CallStatus,
    FailureClass,
    ProofreadRequest,
    Proofreader,
    ProofreadResult,
    ReplayMissError,
    ReplayProofreader,
    RunFatalError,
    Usage,
    make_request,
)
from .models import Category, FindingRejection, RejectCode, ResolutionResult, SkipReason, TraceRecord
from .style_guide import SelectedGuide

STYLEGUIDE_WITHOUT_GUIDE_DETAIL = "type styleguide but no style-guide text was selected"
_THREAD_PREFIX = "proofreader-call"


class _Skipped(Exception):
    """A queued call that was not started because another call had already failed."""


class _StartGate:
    """Coordinates "queued -> started" with "-> stopped" under one lock (see the module docstring).

    ``started`` lists paragraph ids in start order; ``started_at_stop`` is the snapshot taken by the first
    ``stop``; ``refused`` lists ids that found the gate stopped. Invariant (by the lock): after ``stop``,
    ``started`` never grows, so ``started == started_at_stop`` once stopped."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stopped = False
        self.started: list[int] = []
        self.refused: list[int] = []
        self.started_at_stop: tuple[int, ...] | None = None

    def try_start(self, paragraph_id: int) -> bool:
        with self._lock:
            if self._stopped:
                self.refused.append(paragraph_id)
                return False
            self.started.append(paragraph_id)
            return True

    def stop(self) -> None:
        with self._lock:
            if not self._stopped:
                self._stopped = True
                self.started_at_stop = tuple(self.started)

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stopped


class DocumentVerificationError(RunFatalError):
    """The annotated document failed verification even after rollback: nothing may be written (exit 1)."""


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of one run. ``output`` is verified; everything else is text-free."""

    output: bytes = field(repr=False)
    call_records: tuple[CallRecord, ...]  # one per paragraph that was sent or size-guarded, by paragraph id
    trace: tuple[TraceRecord, ...]  # one per raw finding of every successful call
    rolled_back: tuple[int, ...]  # paragraph ids rolled back by verification
    degraded: bool  # a paragraph passed through because of a call failure, the size guard, a rollback, or it
    #                 could not be proofread at all (not_proofread)
    usage: Usage  # summed over every call
    requests_sent: int  # HTTP requests (attempts) summed over every call
    stage_ms: dict[str, float] = field(compare=False)  # wall time per stage, for S4.7 metrics
    # (paragraph id, skip reason) of non-whitespace paragraphs skipped as unmappable or document-unverifiable
    not_proofread: tuple[tuple[int, SkipReason], ...] = ()


NOT_PROOFREAD_REASONS = frozenset({SkipReason.PARA_UNMAPPABLE, SkipReason.DOCUMENT_UNVERIFIABLE})


def not_proofread(scan) -> tuple[tuple[int, SkipReason], ...]:
    """Paragraphs with text that the scanner refused to map or verify: their text was never proofread."""
    return tuple((p.id, p.skip_reason) for p in sorted(scan.paragraphs, key=lambda p: p.id)
                 if p.skip_reason in NOT_PROOFREAD_REASONS and p.text.strip())


def is_eligible(paragraph) -> bool:
    """A paragraph is offered to the model only if it is annotatable and has non-whitespace text."""
    return paragraph.skip_reason is None and bool(paragraph.text.strip())


def _check_settings(config: RunConfig) -> None:
    if config.cache:
        raise UsageError("prompt caching (--cache) is not implemented in this version; run without it")
    concurrency = config.concurrency
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise UsageError("concurrency must be a positive integer")


def _plan(scan, config: RunConfig, guide: SelectedGuide) -> tuple[list[ProofreadRequest], list[CallRecord]]:
    """Requests for eligible paragraphs (id order) and NOT_CALLED records for size-guarded ones."""
    analyses = q.analyse_document(scan)
    system = prompt.system_prompt()
    schema = prompt.output_schema()
    requests: list[ProofreadRequest] = []
    guarded: list[CallRecord] = []
    for paragraph in sorted(scan.paragraphs, key=lambda p: p.id):
        if not is_eligible(paragraph):
            continue
        if prompt.size_violation(paragraph) is not None:
            guarded.append(CallRecord(paragraph.id, CallStatus.NOT_CALLED, FailureClass.REQUEST_TOO_LARGE, 0, 0.0,
                                      Usage()))
            continue
        user = prompt.build_user_message(paragraph, analyses[paragraph.id], lang=config.lang,
                                         scope=config.locale_scope, guide=guide)
        requests.append(make_request(config, paragraph.id, system=system, user=user, schema=schema))
    return requests, guarded


def _precheck(requests: list[ProofreadRequest], config: RunConfig, proofreader: Proofreader) -> None:
    if len(requests) > config.max_calls:
        raise UsageError(f"--max-calls {config.max_calls} exceeded: {len(requests)} paragraphs need a model call")
    if isinstance(proofreader, ReplayProofreader):
        missing = proofreader.missing(req.fingerprint for req in requests)
        if missing:
            raise ReplayMissError(f"the replay recording has no response for {len(missing)} of {len(requests)} "
                                  "requests; nothing was processed")


def _select_failure(errors: dict[int, BaseException]) -> BaseException:
    """Lowest paragraph id among the observed failures; non-Exception BaseExceptions first."""
    severe = {pid: e for pid, e in errors.items() if not isinstance(e, Exception)}
    pool = severe or errors
    return pool[min(pool)]


def _execute(requests: list[ProofreadRequest], proofreader: Proofreader,
             concurrency: int) -> dict[int, ProofreadResult]:
    """Run every request with at most ``concurrency`` in flight. Returns results by paragraph id, or raises
    the selected failure after cancelling unstarted calls and waiting for running ones."""
    if not requests:
        return {}
    results: dict[int, ProofreadResult] = {}
    errors: dict[int, BaseException] = {}
    gate = _StartGate()

    def call(request: ProofreadRequest) -> ProofreadResult:
        if not gate.try_start(request.paragraph_id):  # atomic with stop(): never start after a failure
            raise _Skipped()
        try:
            value = proofreader.proofread(request)
            if not isinstance(value, ProofreadResult):
                raise TypeError("the proofreader returned something other than a ProofreadResult")
            return value
        except BaseException:
            gate.stop()  # before this worker can pick up the next queued call
            raise

    pool = ThreadPoolExecutor(max_workers=min(concurrency, len(requests)), thread_name_prefix=_THREAD_PREFIX)
    futures: dict[Future, int] = {}

    def collect(done) -> None:
        for future in done:
            if future.cancelled():
                continue
            pid = futures[future]
            exc = future.exception()
            if exc is None:
                results[pid] = future.result()
            elif not isinstance(exc, _Skipped):
                errors[pid] = exc

    try:
        for request in requests:
            futures[pool.submit(call, request)] = request.paragraph_id
        pending = set(futures)
        while pending and not errors:
            done, pending = wait(pending, return_when=FIRST_EXCEPTION)
            collect(done)
        if errors:
            for future in pending:
                future.cancel()  # not started -> never runs; running -> unaffected
            done, _ = wait(pending)  # running calls finish safely
            collect(done)
    except BaseException:
        gate.stop()  # e.g. KeyboardInterrupt in this thread: no further call may start
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)  # also on KeyboardInterrupt in this thread
    if errors:
        raise _select_failure(errors)
    return results


def _reject_unguided_styleguide(results: tuple[ResolutionResult, ...],
                                guide: SelectedGuide) -> tuple[ResolutionResult, ...]:
    """Q2: without applicable guide text, a styleguide finding cannot cite a rule -> REJECTED_S3 INVALID_FIELD."""
    if guide.text.strip():
        return results
    out = []
    for result in results:
        bad = [rf for rf in result.resolved if rf.annotation.category is Category.STYLEGUIDE]
        if not bad:
            out.append(result)
            continue
        kept = tuple(rf for rf in result.resolved if rf.annotation.category is not Category.STYLEGUIDE)
        rejected = tuple(FindingRejection(result.paragraph_id, rf.finding_index, RejectCode.INVALID_FIELD,
                                          STYLEGUIDE_WITHOUT_GUIDE_DETAIL) for rf in bad)
        out.append(ResolutionResult(result.paragraph_id, result.raw_count, kept, result.rejections + rejected))
    return tuple(out)


def run(data: bytes, config: RunConfig, proofreader: Proofreader, guide: SelectedGuide) -> PipelineResult:
    """Proofread ``data`` and return verified output bytes plus text-free records (see the module docstring)."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if not isinstance(guide, SelectedGuide):
        raise TypeError("guide must be a SelectedGuide")
    if not isinstance(proofreader, Proofreader):
        raise TypeError("proofreader must implement the Proofreader protocol")
    stage_ms: dict[str, float] = {}
    clock = time.perf_counter

    _check_settings(config)
    t = clock()
    scan = xs.scan(data)
    requests, guarded = _plan(scan, config, guide)
    stage_ms["prepare"] = (clock() - t) * 1000

    _precheck(requests, config, proofreader)

    t = clock()
    results = _execute(requests, proofreader, config.concurrency)
    stage_ms["calls"] = (clock() - t) * 1000

    records = sorted([*guarded, *(CallRecord.from_result(pid, res) for pid, res in results.items())],
                     key=lambda rec: rec.paragraph_id)
    payloads = {pid: res.payload for pid, res in results.items() if res.failure is None}  # Q1

    t = clock()
    resolved = _reject_unguided_styleguide(r.resolve_document(scan, payloads), guide)
    stage_ms["resolve"] = (clock() - t) * 1000

    t = clock()
    outcome = an.annotate(data, scan, r.annotations(resolved))
    stage_ms["annotate"] = (clock() - t) * 1000
    if not outcome.ok or outcome.output is None:
        raise DocumentVerificationError(
            f"the annotated document failed verification ({len(outcome.failures)} issue(s)); nothing was written")
    trace = r.trace(resolved, outcome)

    usage = Usage()
    for rec in records:
        usage = usage + rec.usage
    skipped = not_proofread(scan)
    degraded = (bool(outcome.rolled_back) or any(rec.status is not CallStatus.OK for rec in records)
                or bool(skipped))
    return PipelineResult(outcome.output, tuple(records), trace, tuple(outcome.rolled_back), degraded, usage,
                          sum(rec.attempts for rec in records), stage_ms, not_proofread=skipped)

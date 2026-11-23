"""Run metrics (S4.7, AC-17): total and stage times, per-paragraph latency, peak Python heap, peak process
memory (reported separately, or "n/a"), tokens including cache, requests, and findings by status and code.

The summary is text-free: no document text, corrections, prompts, style-guide content, credentials or
directory paths (file names only). Heap tracing (``tracemalloc``) is enabled only when the RunConfig says so
(automatically for inputs <= 256 KiB, overridable). Peak process memory uses the operating system (Windows
``K32GetProcessMemoryInfo`` through ctypes, POSIX ``resource.getrusage``); psutil is not used (docs/04).
"""

from __future__ import annotations

import contextlib
import statistics
import sys
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterator

from .llm import CallStatus, Usage

NA = "n/a"
METRICS_VERSION = 1


def peak_process_memory_bytes() -> int | None:
    """Peak resident memory of this process in bytes, or None if the platform does not report it."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            current = kernel32.GetCurrentProcess
            current.argtypes = []
            current.restype = wintypes.HANDLE
            query = kernel32.K32GetProcessMemoryInfo
            query.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
            query.restype = wintypes.BOOL
            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            if not query(current(), ctypes.byref(counters), counters.cb):
                return None
            return int(counters.PeakWorkingSetSize)
        import resource  # POSIX only

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak) if sys.platform == "darwin" else int(peak) * 1024  # macOS: bytes; Linux: KiB
    except Exception:  # noqa: BLE001 - a metric must never break the run
        return None


class RunMetrics:
    """Collects one run's metrics. ``close()`` stops heap tracing if this object started it."""

    def __init__(self, *, clock=time.perf_counter) -> None:
        self._clock = clock
        self._start = clock()
        self.stages_ms: dict[str, float] = {}
        self._heap_started = False
        self._heap_enabled = False

    def enable_heap(self) -> None:
        if self._heap_enabled:
            return
        self._heap_enabled = True
        if tracemalloc.is_tracing():
            tracemalloc.reset_peak()
        else:
            tracemalloc.start()
            self._heap_started = True

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = self._clock()
        try:
            yield
        finally:
            self.stages_ms[name] = round((self._clock() - start) * 1000, 3)

    def heap_peak_bytes(self) -> int | None:
        if not self._heap_enabled or not tracemalloc.is_tracing():
            return None
        return int(tracemalloc.get_traced_memory()[1])

    def close(self) -> None:
        if self._heap_started and tracemalloc.is_tracing():
            tracemalloc.stop()
        self._heap_started = False

    def summary(self, result, *, input_name: str, output_name: str | None, exit_code: int,
                auxiliary_failures: tuple[str, ...] = ()) -> dict:
        """The JSON-safe, text-free summary. ``result`` is a PipelineResult (or None when no run completed)."""
        heap = self.heap_peak_bytes()
        process = peak_process_memory_bytes()
        data: dict = {
            "version": METRICS_VERSION,
            "exit_code": exit_code,
            "input": input_name,
            "output": output_name,
            "total_ms": round((self._clock() - self._start) * 1000, 3),
            "stages_ms": dict(self.stages_ms),
            "memory": {
                "heap_tracing": self._heap_enabled,
                "peak_heap_bytes": NA if heap is None else heap,
                "peak_process_bytes": NA if process is None else process,
            },
            "auxiliary_failures": list(auxiliary_failures),
        }
        if result is None:
            return data
        records = result.call_records
        latencies = [r.latency_ms for r in records if r.attempts > 0]
        usage = result.usage if isinstance(result.usage, Usage) else Usage()
        data["pipeline_stages_ms"] = {k: round(v, 3) for k, v in result.stage_ms.items()}
        data["paragraphs"] = {
            "ok": sum(r.status is CallStatus.OK for r in records),
            "passthrough": sum(r.status is CallStatus.PASSTHROUGH for r in records),
            "not_called": sum(r.status is CallStatus.NOT_CALLED for r in records),
            "rolled_back": list(result.rolled_back),
            "failures_by_class": dict(sorted(Counter(r.failure.value for r in records if r.failure).items())),
            "not_proofread": dict(sorted(Counter(reason.value for _, reason in result.not_proofread).items())),
        }
        data["latency_ms"] = {
            "min": min(latencies) if latencies else NA,
            "median": statistics.median(latencies) if latencies else NA,
            "max": max(latencies) if latencies else NA,
            "by_paragraph": {str(r.paragraph_id): r.latency_ms for r in records if r.attempts > 0},
        }
        data["requests_sent"] = result.requests_sent
        data["tokens"] = {"input": usage.input_tokens, "output": usage.output_tokens,
                          "cache_creation_input": usage.cache_creation_input_tokens,
                          "cache_read_input": usage.cache_read_input_tokens}
        data["findings"] = {
            "by_status": dict(sorted(Counter(t.status.value for t in result.trace).items())),
            "rejected_by_code": dict(sorted(Counter(t.code.value for t in result.trace if t.code is not None).items())),
        }
        return data

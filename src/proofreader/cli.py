"""Command-line interface: argument parsing, exit codes 0/1/2/3.

S4.1: the argument parser and its translation into a :class:`proofreader.config.RunConfig`
(CLI flags > process environment > ``./.env`` > defaults). ``main`` (pipeline call, exit-code
mapping, logging) is implemented in S4.7.

argparse reports malformed command lines itself with exit status 2 (docs/03 section 6, "usage").
There is deliberately no flag for the API key (environment or ``./.env`` only), and the custom
base-URL opt-in exists only as a flag (never from the environment).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import tempfile
import traceback
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import config
from . import encoding as encoding_check
from . import gemini
from . import llm
from . import pipeline
from . import style_guide
from .llm import RECORDING_SUFFIX, CallStatus
from .metrics import RunMetrics
from .models import InvalidInputError, ProofreaderError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m proofreader",
        description="Proofread the <p> elements of an XML file and write <input>.corrected.xml.",
        allow_abbrev=False,
    )
    parser.add_argument("input", help="input XML file")
    parser.add_argument("--lang", required=True, help="BCP 47 language tag, e.g. en, en-US, en-GB, fr")
    parser.add_argument("--style-guide", required=True, dest="style_guide", help="style guide (.docx)")

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--replay", default=None, help="serve recorded responses from this file (no API calls)")
    source.add_argument("--record", default=None, help="live run that also records responses to this file")

    parser.add_argument("--model", default=None,
                        help=f"model id (default {config.DEFAULT_MODEL}; env {config.ENV_MODEL}); an id starting "
                             f"with '{config.GEMINI_MODEL_PREFIX}' uses Gemini (key {config.ENV_GEMINI_KEY})")
    parser.add_argument("--concurrency", type=int, default=None,
                        help=f"maximum requests in flight (default {config.DEFAULT_CONCURRENCY}; env {config.ENV_CONCURRENCY})")
    parser.add_argument("--timeout", type=float, default=None,
                        help=f"seconds per request (default {config.DEFAULT_TIMEOUT_SECONDS:g}; env {config.ENV_TIMEOUT})")
    parser.add_argument("--max-calls", type=int, default=None, dest="max_calls",
                        help=f"refuse to run if more paragraphs need a call (default {config.DEFAULT_MAX_CALLS})")
    parser.add_argument("--cache", action="store_true",
                        help="prompt caching is not implemented; this option is refused (exit 2)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file (never the input)")
    parser.add_argument("--base-url", default=None, dest="base_url",
                        help=f"API base URL (default {config.DEFAULT_BASE_URL}; env {config.ENV_BASE_URL})")
    parser.add_argument("--allow-custom-base-url", action="store_true", dest="allow_custom_base_url",
                        help="explicitly allow an https base URL other than the Anthropic API (http is never allowed)")

    heap = parser.add_mutually_exclusive_group()
    heap.add_argument("--heap-metrics", action="store_true", dest="heap_metrics",
                      help="always trace the Python heap (slow on large inputs)")
    heap.add_argument("--no-heap-metrics", action="store_true", dest="no_heap_metrics",
                      help="never trace the Python heap")

    parser.add_argument("--log-level", default=None, dest="log_level", type=str.upper, choices=config.LOG_LEVELS,
                        help=f"log level (default {config.DEFAULT_LOG_LEVEL})")
    parser.add_argument("--log-file", default=None, dest="log_file", help="also write the log to this file")
    parser.add_argument("--metrics-json", default=None, dest="metrics_json", help="write the metrics summary as JSON")
    return parser


def parse_config(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> config.RunConfig:
    """Parse ``argv`` and resolve the full configuration. argparse errors exit with status 2;
    invalid values raise :class:`proofreader.config.UsageError` (exit 2)."""
    namespace = build_parser().parse_args(argv)
    return config.build_config(
        vars(namespace),
        os.environ if environ is None else environ,
        Path.cwd() if cwd is None else cwd,
    )


# ============================================================================================
# S4.7 stage 1: output naming, artifact path validation, atomic writes
# ============================================================================================
#
# Artifacts: the corrected XML is the PRIMARY deliverable; the recording (--record), the metrics JSON
# (--metrics-json) and the log file (--log-file) are AUXILIARY. Every artifact path is validated before any
# API call (plan_artifacts): never the input or an alias of it (symlink, hard link, case variant), never
# another artifact, never an existing file unless --force (the input is never overwritten, even with --force),
# recording names end with .recording.json, and every destination directory must accept a new file.
# Every artifact is published with write_atomic (temporary file in the destination directory, fsync, then an
# exclusive rename/link, or os.replace with --force): no partial file is ever visible.

CORRECTED_SUFFIX = ".corrected.xml"
_TEMP_PREFIX = ".tmp-proofreader-"


class OutputError(ProofreaderError):
    """An artifact cannot be written (I/O, permissions, unsupported filesystem): exit 1."""

    exit_code = 1


def output_path_for(input_path: Path) -> Path:
    """``<stem>.corrected.xml`` next to the input; only the LAST suffix is replaced
    (``x.xml`` -> ``x.corrected.xml``, ``x`` -> ``x.corrected.xml``, ``a.b.xml`` -> ``a.b.corrected.xml``)."""
    input_path = Path(input_path)
    return input_path.with_name(input_path.stem + CORRECTED_SUFFIX)


def _canonical(path: Path) -> str:
    return os.path.normcase(os.path.realpath(path))


def same_file(a: Path, b: Path) -> bool:
    """True if ``a`` and ``b`` are, or would be, the same file: identical inode/file id when both exist
    (symlinks, hard links, case variants on case-insensitive filesystems), else the same resolved path
    (symlinked directories resolved; case folded where the platform folds case)."""
    try:
        if os.path.exists(a) and os.path.exists(b):
            return os.path.samefile(a, b)
    except OSError:
        pass
    return _canonical(a) == _canonical(b)


@dataclass(frozen=True)
class ArtifactPaths:
    input: Path
    output: Path
    recording: Path | None
    metrics: Path | None
    log: Path | None

    def auxiliary(self) -> tuple[tuple[str, Path], ...]:
        return tuple((label, path) for label, path in (("--record", self.recording),
                                                       ("--metrics-json", self.metrics),
                                                       ("--log-file", self.log)) if path is not None)


def _probe_directory(path: Path, label: str) -> None:
    """The destination directory must exist and accept a new file (checked by creating and deleting one)."""
    directory = path.parent if str(path.parent) else Path(".")
    if not directory.is_dir():
        raise OutputError(f"{label}: the directory for {path.name!r} does not exist")
    try:
        descriptor, temp_name = tempfile.mkstemp(prefix=_TEMP_PREFIX, suffix=".probe", dir=directory)
    except OSError:
        raise OutputError(f"{label}: cannot create files next to {path.name!r}") from None
    os.close(descriptor)
    with contextlib.suppress(OSError):
        os.unlink(temp_name)


def _check_target(path: Path, label: str, force: bool) -> None:
    if path.is_dir():
        raise config.UsageError(f"{label}: {path.name!r} is a directory")
    if (path.exists() or path.is_symlink()) and not force:
        raise config.UsageError(f"{label}: {path.name!r} already exists (use --force to replace it)")


def plan_artifacts(cfg: config.RunConfig) -> ArtifactPaths:
    """Validate every artifact path before any API call. UsageError (exit 2) for conflicts and existing
    targets; OutputError (exit 1) for directories that cannot take a new file."""
    input_path = Path(cfg.input_path)
    paths = ArtifactPaths(input_path, output_path_for(input_path), cfg.record_path, cfg.metrics_json, cfg.log_file)
    if paths.recording is not None and not paths.recording.name.endswith(RECORDING_SUFFIX):
        raise config.UsageError(f"--record: the file name must end with {RECORDING_SUFFIX!r} (git-ignored)")
    named = [("corrected output", paths.output), *paths.auxiliary()]
    for label, path in named:
        if same_file(path, input_path):
            raise config.UsageError(f"{label}: {path.name!r} is the input file; the input is never overwritten")
    for i, (label_a, a) in enumerate(named):
        for label_b, b in named[i + 1:]:
            if same_file(a, b):
                raise config.UsageError(f"{label_b}: {b.name!r} is the same file as {label_a}")
    for label, path in named:
        _check_target(path, label, cfg.force)
        _probe_directory(path, label)
    return paths


def write_atomic(path: Path, data: bytes, *, force: bool, protected: Sequence[Path] = (),
                 label: str = "output") -> None:
    """Publish ``data`` at ``path`` atomically: a temporary file in the same directory is written and fsynced,
    then published exclusively (Windows ``os.rename``, which fails if the target exists; elsewhere
    ``os.link``) or, with ``force``, by ``os.replace``. Never publishes over a ``protected`` file (the input),
    even with ``force``. On any failure nothing is published and the temporary file is removed.
    UsageError (exit 2) if the target appeared without ``force``; OutputError (exit 1) otherwise."""
    path = Path(path)
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    for guarded in protected:
        if same_file(path, guarded):
            raise config.UsageError(f"{label}: {path.name!r} is the input file; the input is never overwritten")
    if path.is_dir():
        raise config.UsageError(f"{label}: {path.name!r} is a directory")
    if not force and (path.exists() or path.is_symlink()):
        raise config.UsageError(f"{label}: {path.name!r} already exists (use --force to replace it)")
    directory = path.parent if str(path.parent) else Path(".")
    try:
        descriptor, temp_name = tempfile.mkstemp(prefix=_TEMP_PREFIX, suffix=".tmp", dir=directory)
    except OSError:
        raise OutputError(f"{label}: cannot create a temporary file next to {path.name!r}") from None
    temp = Path(temp_name)
    try:
        try:
            handle = os.fdopen(descriptor, "wb")
        except BaseException:
            os.close(descriptor)
            raise
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if force:
            os.replace(temp, path)
        elif sys.platform == "win32":
            os.rename(temp, path)  # atomic; raises FileExistsError if the target exists
        else:
            os.link(temp, path)  # atomic and exclusive; raises FileExistsError if the target exists
    except FileExistsError:
        raise config.UsageError(f"{label}: {path.name!r} appeared during the run; it was not replaced "
                                "(use --force)") from None
    except (OSError, NotImplementedError, AttributeError):
        raise OutputError(f"{label}: writing {path.name!r} failed; nothing was published") from None
    finally:
        with contextlib.suppress(OSError):
            temp.unlink()


# ============================================================================================
# S4.7 stage 2: main() orchestration, logging, publication
# ============================================================================================
#
# Order (every fatal or usage error happens before any API call and before any artifact exists):
#   parse -> logging -> plan_artifacts -> read input -> load style guide + manifest + select ->
#   choose client (Replay / Anthropic / injected) -> pipeline.run -> publish.
# Publication (approved): 1. corrected XML, 2. recording, 3. metrics JSON, 4. log file. If the corrected XML
# cannot be published, neither the recording nor the metrics JSON is written (the log file still is). If an
# auxiliary artifact fails after the XML was published, the XML is never deleted, the remaining auxiliary
# artifacts are still attempted, stderr says the corrected XML WAS written, and the exit code is 3.
# Exit codes: 0 success; 3 output written but degraded or an auxiliary artifact failed; 1 fatal; 2 usage /
# unsupported input; 130 interrupted (KeyboardInterrupt).
# Logging never contains document text, corrections, prompts or style-guide content at ANY level (D2); the API
# key is redacted from every line (defence in depth); the SDK loggers are pinned to WARNING for the run.

EXIT_OK, EXIT_FATAL, EXIT_USAGE, EXIT_DEGRADED, EXIT_INTERRUPTED = 0, 1, 2, 3, 130
LOGGER_NAME = "proofreader"
SDK_LOGGERS = ("anthropic", "anthropic._base_client", "httpx2", "httpcore2", "httpx", "httpcore")

ProofreaderFactory = Callable[[config.RunConfig, "llm.RecordingWriter | None"], llm.Proofreader]


class _Redactor(logging.Filter):
    """Replaces the API key (if any) in every formatted message; drops args so nothing is re-formatted."""

    def __init__(self) -> None:
        super().__init__()
        self.secret: str | None = None

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if self.secret:
            message = message.replace(self.secret, "***")
        record.msg, record.args = message, ()
        return True


class _AsciiFormatter(logging.Formatter):
    """``LEVEL message`` with non-ASCII escaped, so a cp1252 console can always print it."""

    def format(self, record: logging.LogRecord) -> str:
        text = f"{record.levelname} {record.getMessage()}"
        return text.encode("ascii", "backslashreplace").decode("ascii")


class _BufferHandler(logging.Handler):
    """Keeps formatted lines in memory; the log file is published atomically at the end of the run."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


class _Session:
    """The run's logging: a private ``proofreader`` logger (no propagation) writing to ``stderr`` and, when
    requested, to an in-memory buffer for --log-file. Everything is undone by ``close()``."""

    def __init__(self, stderr) -> None:
        self.logger = logging.getLogger(LOGGER_NAME)
        self.redactor = _Redactor()
        self.buffer: _BufferHandler | None = None
        self.published: str | None = None  # name of the corrected output once it has been published
        self.paths: ArtifactPaths | None = None  # set as soon as every artifact path has been validated
        self._saved = (self.logger.level, self.logger.propagate, list(self.logger.handlers))
        self._sdk_saved: dict[str, int] = {}
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        self.console = logging.StreamHandler(stderr)
        self._add(self.console, logging.INFO)

    def _add(self, handler: logging.Handler, level: int) -> None:
        handler.setLevel(level)
        handler.setFormatter(_AsciiFormatter())
        handler.addFilter(self.redactor)
        self.logger.addHandler(handler)

    def configure(self, cfg: config.RunConfig) -> None:
        level = getattr(logging, cfg.log_level)
        self.logger.setLevel(level)
        self.console.setLevel(level)
        if cfg.api_key is not None:
            self.redactor.secret = cfg.api_key.reveal()
        if cfg.log_file is not None:
            self.buffer = _BufferHandler()
            self._add(self.buffer, level)
        self.pin_sdk_loggers()

    def pin_sdk_loggers(self) -> None:
        """WARNING for the SDK and HTTP loggers, whatever ANTHROPIC_LOG or --log-level say. Called again after
        the SDK is imported (its import may raise the level); the original levels are restored by close()."""
        for name in SDK_LOGGERS:
            sdk_logger = logging.getLogger(name)
            self._sdk_saved.setdefault(name, sdk_logger.level)
            sdk_logger.setLevel(logging.WARNING)

    def close(self) -> None:
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
        level, propagate, handlers = self._saved
        self.logger.setLevel(level)
        self.logger.propagate = propagate
        for handler in handlers:
            self.logger.addHandler(handler)
        for name, sdk_level in self._sdk_saved.items():
            logging.getLogger(name).setLevel(sdk_level)


def _read_input(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(encoding_check.MAX_INPUT_BYTES + 1)  # the scanner rejects anything larger
    except OSError:
        raise InvalidInputError(f"input {path.name!r} cannot be read") from None


def _load_guide(cfg: config.RunConfig) -> style_guide.SelectedGuide:
    guide = style_guide.load(cfg.style_guide_path)
    return style_guide.select(guide, cfg.locale_scope, style_guide.load_manifest())


def _choose_client(cfg: config.RunConfig, factory: ProofreaderFactory | None,
                   recorder: llm.RecordingWriter | None, session: _Session) -> tuple[llm.Proofreader, str]:
    if factory is not None:  # tests only: there is no command-line flag for this
        return factory(cfg, recorder), "injected"
    if cfg.replay_path is not None:
        return llm.ReplayProofreader(llm.load_recording(cfg.replay_path)), "replay"
    if cfg.provider == config.PROVIDER_GEMINI:  # model id "gemini-..." (config.provider_for)
        client: llm.Proofreader = gemini.GeminiProofreader(cfg, recorder=recorder)  # MissingApiKeyError first
    else:
        client = llm.AnthropicProofreader(cfg, recorder=recorder)  # MissingApiKeyError before any request
    session.pin_sdk_loggers()  # the SDK import may have raised its logger levels (ANTHROPIC_LOG)
    return client, "live"


def _log_result(log: logging.Logger, result: pipeline.PipelineResult) -> None:
    for rec in result.call_records:
        line = (f"paragraph {rec.paragraph_id}: {rec.status.value}"
                + (f" ({rec.failure.value})" if rec.failure else "")
                + f", attempts {rec.attempts}, latency {rec.latency_ms:.0f} ms"
                + (f", request {rec.request_id}" if rec.request_id else ""))
        log.log(logging.INFO if rec.status is CallStatus.OK else logging.WARNING, line)
    for paragraph_id, reason in result.not_proofread:  # id and reason only, never text
        log.warning(f"paragraph {paragraph_id}: not proofread ({reason.value})")
    statuses = Counter(t.status.value for t in result.trace)
    codes = Counter(t.code.value for t in result.trace if t.code is not None)
    log.info("findings by status: " + (", ".join(f"{k} {v}" for k, v in sorted(statuses.items())) or "none"))
    if codes:
        log.info("findings by code: " + ", ".join(f"{k} {v}" for k, v in sorted(codes.items())))
    if result.rolled_back:
        log.warning("paragraphs rolled back after verification: " + ", ".join(map(str, result.rolled_back)))
    u = result.usage
    log.info(f"requests {result.requests_sent}; tokens in {u.input_tokens}, out {u.output_tokens}, "
             f"cache write {u.cache_creation_input_tokens}, cache read {u.cache_read_input_tokens}")


def _mib(value: object) -> str:
    return value if isinstance(value, str) else f"{value / (1024 * 1024):.1f} MiB"  # "n/a" passes through


def _runtime_summary(summary: dict) -> str:
    """B21 (S4.8e): one text-free line for the default log - total time, stage times (pipeline stages in
    brackets), peak Python heap and peak process memory; ``n/a`` where a measurement is unavailable."""
    inner = summary.get("pipeline_stages_ms") or {}
    detail = (" [" + ", ".join(f"{name} {ms / 1000:.3f} s" for name, ms in inner.items()) + "]") if inner else ""
    stages = ", ".join(f"{name} {ms / 1000:.3f} s" + (detail if name == "pipeline" else "")
                       for name, ms in summary["stages_ms"].items())
    memory = summary["memory"]
    return (f"run time {summary['total_ms'] / 1000:.3f} s ({stages}); "
            f"peak heap {_mib(memory['peak_heap_bytes'])}; peak process memory {_mib(memory['peak_process_bytes'])}")


def _internal_error(log: logging.Logger, exc: BaseException) -> None:
    """Type name only: an unexpected exception's message may quote data. Frames (code lines, no values) at DEBUG."""
    log.error(f"internal error ({type(exc).__name__}); nothing further was written")
    log.debug("traceback (frames only):\n" + "".join(traceback.format_tb(exc.__traceback__)).rstrip())


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None, cwd: Path | None = None,
         proofreader_factory: ProofreaderFactory | None = None, stdout=None, stderr=None) -> int:
    """Run the proofreader. Returns the exit code (see the section comment above)."""
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    session = _Session(stderr)
    try:
        return _run(argv, environ, cwd, proofreader_factory, session, stdout)
    finally:
        session.close()


def _run(argv, environ, cwd, factory, session: _Session, stdout) -> int:
    log = session.logger
    try:
        with contextlib.redirect_stderr(session.console.stream):
            cfg = parse_config(argv, environ=environ, cwd=cwd)
    except SystemExit as exc:  # argparse: usage error (2) or --help (0); it printed its own message
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    except ProofreaderError as exc:
        log.error(f"error: {exc.message}")
        return exc.exit_code
    except KeyboardInterrupt:
        log.error("interrupted; nothing was written")
        return EXIT_INTERRUPTED
    session.configure(cfg)

    metrics = RunMetrics()
    try:
        result, recorder, paths = _process(cfg, factory, session, metrics)
    except ProofreaderError as exc:
        log.error(f"error: {exc.message}")
        log.error("no corrected output was written")
        _publish_log(cfg, session, session.paths)
        metrics.close()
        return exc.exit_code
    except KeyboardInterrupt:
        log.error("interrupted; no corrected output was written")
        _publish_log(cfg, session, session.paths)
        metrics.close()
        return EXIT_INTERRUPTED
    except Exception as exc:  # noqa: BLE001 - mapped to exit 1 with the type name only
        _internal_error(log, exc)
        _publish_log(cfg, session, session.paths)
        metrics.close()
        return EXIT_FATAL
    try:
        return _publish(cfg, paths, result, recorder, metrics, session, stdout)
    except KeyboardInterrupt:
        if session.published is not None:
            log.error(f"interrupted after the corrected output was written: {session.published}")
        else:
            log.error("interrupted; no corrected output was written")
        return EXIT_INTERRUPTED
    finally:
        metrics.close()


def _process(cfg, factory, session: _Session, metrics: RunMetrics):
    log = session.logger
    paths = plan_artifacts(cfg)  # before any API call: names, aliases, collisions, existing targets, directories
    session.paths = paths
    with metrics.stage("read_input"):
        data = _read_input(paths.input)
    if cfg.heap_metrics_enabled(len(data)):
        metrics.enable_heap()
    with metrics.stage("load_style_guide"):
        guide = _load_guide(cfg)
    log.info(f"input {paths.input.name} ({len(data)} bytes); lang {cfg.lang} ({cfg.locale_scope.value}); "
             f"style guide: {guide.included} of {guide.total} sections selected")
    recorder = llm.RecordingWriter() if paths.recording is not None else None
    client, mode = _choose_client(cfg, factory, recorder, session)
    log.info(f"mode {mode}; model {cfg.model}; concurrency {cfg.concurrency}; max calls {cfg.max_calls}"
             + (f"; host {cfg.base_url.host}" if mode == "live" else ""))
    try:
        with metrics.stage("pipeline"):
            result = pipeline.run(data, cfg, client, guide)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    _log_result(log, result)
    return result, recorder, paths


def _publish(cfg, paths: ArtifactPaths, result, recorder, metrics: RunMetrics, session: _Session, stdout) -> int:
    log = session.logger
    try:
        with metrics.stage("write_output"):
            write_atomic(paths.output, result.output, force=cfg.force, protected=[paths.input],
                         label="corrected output")
    except ProofreaderError as exc:
        log.error(f"error: {exc.message}")
        log.error("no corrected output was written; the recording and metrics were not written either")
        _publish_log(cfg, session, paths)
        return exc.exit_code
    except Exception as exc:  # noqa: BLE001
        _internal_error(log, exc)
        log.error("no corrected output was written; the recording and metrics were not written either")
        _publish_log(cfg, session, paths)
        return EXIT_FATAL
    session.published = paths.output.name
    exit_code = EXIT_DEGRADED if result.degraded else EXIT_OK
    log.info(f"corrected output written: {paths.output.name}"
             + (" (some paragraphs passed through unchanged; see the warnings above)" if result.degraded else ""))
    print(str(paths.output), file=stdout)
    log.info(_runtime_summary(metrics.summary(result, input_name=paths.input.name, output_name=paths.output.name,
                                              exit_code=exit_code)))

    failures: list[str] = []

    def auxiliary(label: str, action) -> None:
        try:
            action()
        except ProofreaderError as exc:
            failures.append(label)
            log.error(f"corrected output written: {paths.output.name}; {label} could not be written: {exc.message}")
        except Exception as exc:  # noqa: BLE001
            failures.append(label)
            log.error(f"corrected output written: {paths.output.name}; {label} could not be written "
                      f"({type(exc).__name__})")

    if recorder is not None:
        auxiliary("--record", lambda: recorder.write(paths.recording, force=cfg.force))
    if paths.metrics is not None:
        def write_metrics() -> None:
            summary = metrics.summary(result, input_name=paths.input.name, output_name=paths.output.name,
                                      exit_code=EXIT_DEGRADED if (result.degraded or failures) else EXIT_OK,
                                      auxiliary_failures=tuple(failures))
            text = json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
            write_atomic(paths.metrics, text.encode("ascii"), force=cfg.force, protected=[paths.input],
                         label="--metrics-json")
        auxiliary("--metrics-json", write_metrics)
    if failures:
        exit_code = EXIT_DEGRADED
    log.info(f"exit code {exit_code}")
    if paths.log is not None and not _publish_log(cfg, session, paths):
        exit_code = EXIT_DEGRADED
    return exit_code


def _publish_log(cfg, session: _Session, paths: ArtifactPaths | None) -> bool:
    """Publish the buffered log (last). Returns False if it was requested but could not be written."""
    if session.buffer is None or cfg.log_file is None:
        return True
    if paths is None:  # planning itself failed: the destination was never validated, so it is not written
        session.logger.warning("--log-file was not written (its destination could not be validated)")
        return False
    text = "\n".join(session.buffer.lines) + "\n"
    try:
        write_atomic(paths.log, text.encode("ascii", "backslashreplace"), force=cfg.force,
                     protected=[paths.input], label="--log-file")
        return True
    except ProofreaderError as exc:
        session.logger.error(f"--log-file could not be written: {exc.message}")
        return False
    except Exception as exc:  # noqa: BLE001
        session.logger.error(f"--log-file could not be written ({type(exc).__name__})")
        return False

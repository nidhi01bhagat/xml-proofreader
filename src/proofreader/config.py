"""Run configuration (S4.1): sources, precedence, validation and secret handling.

Precedence (docs/04, S4 decisions): CLI flags > process environment > ``./.env`` > built-in defaults.

``.env`` is read ONLY from the current working directory (no search of parent directories) and
never overrides a variable that is already set (``override=False`` semantics). It is parsed with
``dotenv_values`` (no variable interpolation) and merged into a private mapping; ``os.environ`` is
NOT modified, so the Anthropic SDK can never pick up a ``.env`` value behind our back - every
client setting is passed explicitly. ``PYTHON_DOTENV_DISABLED`` (python-dotenv's switch) is honoured.

Security:
  * The API key comes only from ``ANTHROPIC_API_KEY`` - or ``GEMINI_API_KEY`` when the model id selects
    Gemini (``gemini-...``; see :func:`provider_for`) - from the environment or ``./.env``, never from a CLI
    flag. It is wrapped in :class:`ApiKey`, whose ``repr``/``str``/``format`` are redacted and which
    refuses to be pickled or copied, so it cannot leak through logs, exceptions, metrics or recordings.
  * The key is used exactly as given: a value with leading, trailing or embedded whitespace (or any
    control / non-ASCII character) is rejected (exit 2), never trimmed. An empty value means "not set".
  * The API base URL must be ``https://api.anthropic.com`` unless the CLI flag
    ``--allow-custom-base-url`` is given; the opt-in cannot come from the environment or ``.env``.
    Every endpoint must use **https** (plain http never receives the key; tests intercept requests
    with an in-memory transport instead). URLs with credentials, query strings or fragments are
    always rejected. Messages name the host only, never the full URL.
  * ``ANTHROPIC_CUSTOM_HEADERS`` is rejected (exit 2): the SDK merges it into every request even
    when the key and base URL are passed explicitly, and an ``Authorization`` line in it replaces
    the key (verified offline against anthropic 2.12.0, ``tests/test_sdk_environment.py``).

Exit codes: :class:`UsageError` (bad flag values, malformed ``--lang``, rejected base URL, unreadable
``.env``) -> 2; :class:`MissingApiKeyError` -> 1 (docs/03 section 6).
"""

from __future__ import annotations

import enum
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

from .models import ProofreaderError

# --------------------------------------------------------------------------------------------
# Defaults (approved: D1 model/effort, docs/03 section 7, S4 review decisions)
# --------------------------------------------------------------------------------------------

DEFAULT_MODEL = "claude-opus-5-5"
EFFORT = "low"  # D1; not configurable
DEFAULT_CONCURRENCY = 8
DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_TOKENS = 8000  # docs/03 section 7 ("~8k, non-streaming"); not configurable
DEFAULT_MAX_CALLS = 200
DEFAULT_LOG_LEVEL = "INFO"
HEAP_METRICS_AUTO_MAX_BYTES = 256 * 1024  # tracemalloc on automatically only up to this input size

ANTHROPIC_API_HOST = "api.anthropic.com"
DEFAULT_BASE_URL = f"https://{ANTHROPIC_API_HOST}"

# Providers (S5): chosen from the model id - "gemini-..." selects Gemini, every other id Anthropic.
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GEMINI = "gemini"
GEMINI_MODEL_PREFIX = "gemini-"
GEMINI_API_HOST = "generativelanguage.googleapis.com"
GEMINI_BASE_URL = f"https://{GEMINI_API_HOST}"
# Lowercase letters, digits, dots and hyphens after the prefix, ending in a letter or digit: the id is used
# verbatim in the request path, so nothing else is accepted.
_GEMINI_MODEL_ID = re.compile(r"gemini-[a-z0-9](?:[a-z0-9.-]{0,90}[a-z0-9])?\Z")

# Validation bounds.
MIN_CONCURRENCY, MAX_CONCURRENCY = 1, 64
MIN_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS = 1.0, 600.0
MIN_MAX_CALLS, MAX_MAX_CALLS = 1, 100_000
MAX_LANG_TAG_LENGTH = 64
MAX_ENV_FILE_BYTES = 64 * 1024
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,99}\Z")

# Environment variable names (documented in .env.example).
ENV_API_KEY = "ANTHROPIC_API_KEY"
ENV_GEMINI_KEY = "GEMINI_API_KEY"
ENV_BASE_URL = "ANTHROPIC_BASE_URL"
ENV_MODEL = "PROOFREADER_MODEL"
ENV_CONCURRENCY = "PROOFREADER_CONCURRENCY"
ENV_TIMEOUT = "PROOFREADER_TIMEOUT_SECONDS"
ENV_DOTENV_DISABLED = "PYTHON_DOTENV_DISABLED"
# Read by the SDK even when the client is configured explicitly, and able to change requests
# (headers, including Authorization): refused rather than silently honoured.
SDK_OVERRIDE_ENV = ("ANTHROPIC_CUSTOM_HEADERS",)
_TRUTHY = frozenset({"1", "true", "t", "yes", "y"})  # python-dotenv's own definition


# --------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------


class UsageError(ProofreaderError):
    """Invalid configuration or usage (exit 2). Messages never contain secrets or full URLs."""

    exit_code = 2


class MissingApiKeyError(ProofreaderError):
    """Live mode without the provider's key (``ANTHROPIC_API_KEY`` or ``GEMINI_API_KEY``; exit 1, before any
    API call)."""

    exit_code = 1


# --------------------------------------------------------------------------------------------
# Secret wrapper
# --------------------------------------------------------------------------------------------


class ApiKey:
    """An API key that never renders itself. Use :meth:`reveal` only when building the client."""

    __slots__ = ("_value",)

    def __init__(self, value: str, name: str = ENV_API_KEY) -> None:
        # ``name`` is the variable the value came from; it is used in error messages only, never stored.
        if not isinstance(value, str) or not value:
            raise UsageError(f"{name} is empty")
        if any(ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value) or not value.isascii():
            # Never echo the value (or any part of it).
            raise UsageError(f"{name} contains whitespace, control or non-ASCII characters")
        object.__setattr__(self, "_value", value)

    def reveal(self) -> str:
        return self._value

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ApiKey is immutable")

    def __repr__(self) -> str:
        return "ApiKey('***')"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return repr(self)

    def __reduce__(self):  # pickle, copy and deepcopy all go through here
        raise TypeError("ApiKey must not be serialised or copied")

    def __eq__(self, other: object) -> bool:
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]


# --------------------------------------------------------------------------------------------
# Language tags (D8; RFC 5646 well-formedness, conservative)
# --------------------------------------------------------------------------------------------


class LocaleScope(str, enum.Enum):
    """Which style-guide rules a language receives (D8 as approved; scopes per S4 review decisions)."""

    FULL = "full"  # exactly "en" or "en-US": the full style guide
    EN_GB = "en-GB"  # exactly "en-GB": en-GB neutral scope + cross-locale neutral scope
    CROSS_LOCALE = "cross-locale"  # every other tag: cross-locale neutral scope only


def normalise_lang(tag: object) -> str:
    """Validate a BCP 47 tag (RFC 5646 syntax) and return it in canonical case; ``_`` -> ``-``.

    Accepted: language (2-3 letters with up to three 3-letter extlangs, or 5-8 letters), optional
    script (4 letters), region (2 letters or 3 digits), variants, extensions and a private-use part.
    Rejected (exit 2): anything else, including 4-letter (reserved) languages, bare private-use or
    grandfathered tags, duplicate variants or extension singletons, non-ASCII and over-long input.
    Canonical case (RFC 5646 section 2.1.1): language/extlang lower, script title, region upper,
    everything else lower. No registry lookup is made (syntax only).
    """
    if not isinstance(tag, str) or not tag or len(tag) > MAX_LANG_TAG_LENGTH or not tag.isascii():
        raise UsageError(_lang_message(tag))
    parts = tag.replace("_", "-").split("-")
    if any(not p or not p.isalnum() for p in parts):
        raise UsageError(_lang_message(tag))

    n = len(parts)
    language = parts[0]
    if not (language.isalpha() and (2 <= len(language) <= 3 or 5 <= len(language) <= 8)):
        raise UsageError(_lang_message(tag))
    out = [language.lower()]
    i = 1
    if len(language) <= 3:
        extlangs = 0
        while i < n and extlangs < 3 and len(parts[i]) == 3 and parts[i].isalpha():
            out.append(parts[i].lower())
            i += 1
            extlangs += 1
    if i < n and len(parts[i]) == 4 and parts[i].isalpha():
        out.append(parts[i].title())
        i += 1
    if i < n and ((len(parts[i]) == 2 and parts[i].isalpha()) or (len(parts[i]) == 3 and parts[i].isdigit())):
        out.append(parts[i].upper())
        i += 1
    variants: set[str] = set()
    while i < n and (5 <= len(parts[i]) <= 8 or (len(parts[i]) == 4 and parts[i][0].isdigit())):
        variant = parts[i].lower()
        if variant in variants:
            raise UsageError(_lang_message(tag))
        variants.add(variant)
        out.append(variant)
        i += 1
    singletons: set[str] = set()
    while i < n and len(parts[i]) == 1 and parts[i].lower() != "x":
        singleton = parts[i].lower()
        if singleton in singletons:
            raise UsageError(_lang_message(tag))
        singletons.add(singleton)
        out.append(singleton)
        i += 1
        start = i
        while i < n and 2 <= len(parts[i]) <= 8:
            out.append(parts[i].lower())
            i += 1
        if i == start:
            raise UsageError(_lang_message(tag))
    if i < n and parts[i].lower() == "x":
        out.append("x")
        i += 1
        if i == n:
            raise UsageError(_lang_message(tag))
        while i < n and 1 <= len(parts[i]) <= 8:
            out.append(parts[i].lower())
            i += 1
    if i != n:
        raise UsageError(_lang_message(tag))
    return "-".join(out)


def locale_scope(normalised_tag: str) -> LocaleScope:
    """D8: only exactly ``en`` / ``en-US`` get the full guide; exactly ``en-GB`` gets the en-GB scope."""
    if normalised_tag in ("en", "en-US"):
        return LocaleScope.FULL
    if normalised_tag == "en-GB":
        return LocaleScope.EN_GB
    return LocaleScope.CROSS_LOCALE


def _lang_message(tag: object) -> str:
    shown = repr(tag) if isinstance(tag, str) and len(tag) <= MAX_LANG_TAG_LENGTH else "(value too long or not text)"
    return f"--lang {shown} is not a well-formed BCP 47 language tag (examples: en, en-US, en-GB, fr, pt-BR)"


# --------------------------------------------------------------------------------------------
# API base URL policy
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseUrl:
    url: str
    host: str
    custom: bool  # True only for an explicitly opted-in non-default endpoint


def resolve_base_url(candidate: str | None, *, allow_custom: bool) -> BaseUrl:
    """Apply the base-URL policy. Errors name the host only (never the full URL or credentials)."""
    if candidate is None or candidate == "":
        return BaseUrl(DEFAULT_BASE_URL, ANTHROPIC_API_HOST, False)
    try:
        parts = urlsplit(candidate)
        port = parts.port
        host = parts.hostname
    except ValueError:
        raise UsageError("API base URL is malformed") from None
    if parts.scheme != "https" or not host:
        raise UsageError("API base URL must be an absolute https URL (plain http is never allowed)")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise UsageError("API base URL must not contain credentials")
    if parts.query or parts.fragment:
        raise UsageError(f"API base URL for host {host} must not contain a query or fragment")
    is_default = host == ANTHROPIC_API_HOST and port in (None, 443) and parts.path in ("", "/")
    if is_default:
        return BaseUrl(DEFAULT_BASE_URL, ANTHROPIC_API_HOST, False)
    if not allow_custom:
        raise UsageError(
            f"API base URL host {host} is not the Anthropic API ({ANTHROPIC_API_HOST}) or is not its default "
            "endpoint; custom https endpoints require the explicit --allow-custom-base-url flag"
        )
    return BaseUrl(candidate.rstrip("/"), host, True)


def provider_for(model: str) -> str:
    """The provider for an (already well-formed) model id: ``gemini-...`` is Gemini, anything else Anthropic.

    Fails closed on ambiguity: an id that mentions Gemini but is not a well-formed lowercase ``gemini-`` id
    (``Gemini-2.5``, ``models/gemini-x``, ``gemini-``, ``my-gemini``) is a usage error, never routed to Anthropic.
    """
    if model.startswith(GEMINI_MODEL_PREFIX):
        if not _GEMINI_MODEL_ID.match(model):
            raise UsageError("Gemini model id is invalid: after 'gemini-' use only lowercase letters, digits, dots "
                             "and hyphens, ending in a letter or digit (for example gemini-2.5-flash-lite)")
        return PROVIDER_GEMINI
    if "gemini" in model.casefold():
        raise UsageError("model id is ambiguous: Gemini models must start with the lowercase prefix 'gemini-' "
                         "(for example gemini-2.5-flash-lite)")
    return PROVIDER_ANTHROPIC


# --------------------------------------------------------------------------------------------
# .env
# --------------------------------------------------------------------------------------------


def read_env_file(cwd: Path, environ: Mapping[str, str]) -> dict[str, str] | None:
    """Values from ``<cwd>/.env`` (no parent search, no interpolation), or None if absent/disabled.

    Keys without a value are dropped. The process environment is not modified.
    """
    if environ.get(ENV_DOTENV_DISABLED, "").casefold() in _TRUTHY:
        return None
    path = Path(cwd) / ".env"
    if not path.is_file():
        return None
    try:
        if path.stat().st_size > MAX_ENV_FILE_BYTES:
            raise UsageError(f".env in the current directory is larger than {MAX_ENV_FILE_BYTES} bytes")
        raw = dotenv_values(path, interpolate=False, encoding="utf-8")
    except UsageError:
        raise
    except (OSError, UnicodeDecodeError):
        raise UsageError(".env in the current directory cannot be read as UTF-8 text") from None
    return {k: v for k, v in raw.items() if v is not None}


def merged_environment(environ: Mapping[str, str], env_file: Mapping[str, str] | None) -> dict[str, str]:
    """``override=False`` semantics: a ``.env`` value is used only if the process environment lacks it."""
    merged = dict(env_file or {})
    merged.update(environ)
    return merged


# --------------------------------------------------------------------------------------------
# RunConfig
# --------------------------------------------------------------------------------------------


class HeapMetrics(str, enum.Enum):
    AUTO = "auto"  # on only for inputs <= HEAP_METRICS_AUTO_MAX_BYTES
    ON = "on"  # --heap-metrics
    OFF = "off"  # --no-heap-metrics


@dataclass(frozen=True)
class RunConfig:
    input_path: Path
    style_guide_path: Path
    lang: str  # normalised
    locale_scope: LocaleScope
    model: str
    concurrency: int
    timeout_seconds: float
    max_calls: int
    cache: bool  # D13: optional, default OFF
    force: bool
    replay_path: Path | None
    record_path: Path | None
    base_url: BaseUrl
    heap_metrics: HeapMetrics
    log_level: str
    log_file: Path | None
    metrics_json: Path | None
    env_file_loaded: bool
    api_key: ApiKey | None = field(default=None, repr=False, compare=False)
    effort: str = EFFORT
    max_tokens: int = MAX_TOKENS
    provider: str = PROVIDER_ANTHROPIC  # derived from the model id (provider_for); never set independently

    @property
    def api_key_variable(self) -> str:
        return ENV_GEMINI_KEY if self.provider == PROVIDER_GEMINI else ENV_API_KEY

    @property
    def live(self) -> bool:
        return self.replay_path is None

    def heap_metrics_enabled(self, input_size: int) -> bool:
        if self.heap_metrics is HeapMetrics.ON:
            return True
        if self.heap_metrics is HeapMetrics.OFF:
            return False
        return input_size <= HEAP_METRICS_AUTO_MAX_BYTES

    def require_api_key(self) -> ApiKey:
        """The key for live mode; raises MissingApiKeyError (exit 1) if absent."""
        if self.api_key is None:
            raise MissingApiKeyError(
                f"{self.api_key_variable} is not set (environment or ./.env); it is needed for live runs, "
                "not for --replay"
            )
        return self.api_key


def build_config(cli: Mapping[str, object], environ: Mapping[str, str], cwd: Path) -> RunConfig:
    """Resolve a RunConfig from parsed CLI values, the process environment and ``<cwd>/.env``.

    ``cli`` holds argparse destinations; a value of None means "not given on the command line".
    """
    env_file = read_env_file(cwd, environ)
    env = merged_environment(environ, env_file)
    for name in SDK_OVERRIDE_ENV:
        if env.get(name):
            raise UsageError(
                f"{name} is set; it is not supported because the Anthropic SDK would add its headers to every "
                "request (it can replace the API key). Unset it and run again."
            )

    model = _pick(cli.get("model"), env.get(ENV_MODEL), DEFAULT_MODEL)
    if not isinstance(model, str) or not _MODEL_ID.match(model):
        raise UsageError(f"model id is invalid (set via --model or {ENV_MODEL})")
    provider = provider_for(model)

    concurrency = _int_setting(cli.get("concurrency"), env.get(ENV_CONCURRENCY), DEFAULT_CONCURRENCY,
                               "--concurrency", ENV_CONCURRENCY, MIN_CONCURRENCY, MAX_CONCURRENCY)
    timeout = _float_setting(cli.get("timeout"), env.get(ENV_TIMEOUT), DEFAULT_TIMEOUT_SECONDS,
                             "--timeout", ENV_TIMEOUT, MIN_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS)
    max_calls = _int_setting(cli.get("max_calls"), None, DEFAULT_MAX_CALLS,
                             "--max-calls", None, MIN_MAX_CALLS, MAX_MAX_CALLS)

    lang = normalise_lang(cli.get("lang"))
    base_candidate = _pick(cli.get("base_url"), env.get(ENV_BASE_URL), None)
    if provider == PROVIDER_GEMINI:
        # Gemini always uses Google's endpoint; the Anthropic base-URL settings cannot redirect it.
        if base_candidate not in (None, "") or cli.get("allow_custom_base_url"):
            raise UsageError(f"--base-url, {ENV_BASE_URL} and --allow-custom-base-url apply only to Anthropic "
                             "models; Gemini models always use the Gemini API endpoint")
        base_url = BaseUrl(GEMINI_BASE_URL, GEMINI_API_HOST, False)
    else:
        base_url = resolve_base_url(base_candidate, allow_custom=bool(cli.get("allow_custom_base_url")))

    replay, record = cli.get("replay"), cli.get("record")
    if replay is not None and record is not None:
        raise UsageError("--replay and --record cannot be combined")
    heap_on, heap_off = bool(cli.get("heap_metrics")), bool(cli.get("no_heap_metrics"))
    if heap_on and heap_off:
        raise UsageError("--heap-metrics and --no-heap-metrics cannot be combined")
    heap = HeapMetrics.ON if heap_on else HeapMetrics.OFF if heap_off else HeapMetrics.AUTO

    log_level = str(_pick(cli.get("log_level"), None, DEFAULT_LOG_LEVEL)).upper()
    if log_level not in LOG_LEVELS:
        raise UsageError(f"--log-level must be one of {', '.join(LOG_LEVELS)}")

    key_variable = ENV_GEMINI_KEY if provider == PROVIDER_GEMINI else ENV_API_KEY
    raw_key = env.get(key_variable, "")
    # Used verbatim; malformed values raise, never trimmed. Only the selected provider's key is read.
    api_key = ApiKey(raw_key, key_variable) if raw_key != "" else None

    return RunConfig(
        input_path=_path(cli.get("input"), "input XML path"),
        style_guide_path=_path(cli.get("style_guide"), "--style-guide"),
        lang=lang,
        locale_scope=locale_scope(lang),
        model=model,
        concurrency=concurrency,
        timeout_seconds=timeout,
        max_calls=max_calls,
        cache=bool(cli.get("cache")),
        force=bool(cli.get("force")),
        replay_path=_optional_path(replay),
        record_path=_optional_path(record),
        base_url=base_url,
        heap_metrics=heap,
        log_level=log_level,
        log_file=_optional_path(cli.get("log_file")),
        metrics_json=_optional_path(cli.get("metrics_json")),
        env_file_loaded=env_file is not None,
        api_key=api_key,
        provider=provider,
    )


def _pick(cli_value: object, env_value: str | None, default: object) -> object:
    if cli_value is not None:
        return cli_value
    if env_value is not None and env_value != "":
        return env_value
    return default


def _int_setting(cli_value: object, env_value: str | None, default: int, flag: str, env_name: str | None,
                 low: int, high: int) -> int:
    source = flag
    value: object = default
    if cli_value is not None:
        value = cli_value
    elif env_value is not None and env_value.strip() != "":
        source = env_name or flag
        try:
            value = int(env_value.strip(), 10)
        except ValueError:
            raise UsageError(f"{source} must be an integer") from None
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise UsageError(f"{source} must be an integer from {low} to {high}")
    return value


def _float_setting(cli_value: object, env_value: str | None, default: float, flag: str, env_name: str,
                   low: float, high: float) -> float:
    source = flag
    value: object = default
    if cli_value is not None:
        value = cli_value
    elif env_value is not None and env_value.strip() != "":
        source = env_name
        try:
            value = float(env_value.strip())
        except ValueError:
            raise UsageError(f"{source} must be a number of seconds") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or not low <= value <= high:
        raise UsageError(f"{source} must be a number of seconds from {low:g} to {high:g}")
    return float(value)


def _path(value: object, what: str) -> Path:
    if not isinstance(value, (str, Path)) or str(value) == "":
        raise UsageError(f"{what} is required")
    return Path(value)


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)) or str(value) == "":
        raise UsageError("path arguments must not be empty")
    return Path(value)

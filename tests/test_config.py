"""S4.1: run configuration - precedence, ./.env loading, --lang, base-URL policy, secret handling, CLI parser.

All values are invented. No network access, no API key needed.
"""

from __future__ import annotations

import copy
import os
import pickle
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from proofreader import cli
from proofreader import config as c

SENTINEL_KEY = "test-sentinel-key-0123456789"  # invented; must never appear in any rendering
REQUIRED = ["in.xml", "--lang", "en", "--style-guide", "guide.docx"]


def build(tmp_path: Path, argv=None, environ=None, env_file: str | None = None) -> c.RunConfig:
    if env_file is not None:
        (tmp_path / ".env").write_text(env_file, encoding="utf-8")
    return cli.parse_config(REQUIRED + list(argv or []), environ=dict(environ or {}), cwd=tmp_path)


# --------------------------------------------------------------------------------------------
# Defaults and precedence (CLI > process environment > ./.env > defaults)
# --------------------------------------------------------------------------------------------


def test_defaults(tmp_path: Path) -> None:
    cfg = build(tmp_path)
    assert cfg.model == "claude-opus-5-5" and cfg.effort == "low"
    assert (cfg.concurrency, cfg.timeout_seconds, cfg.max_calls, cfg.max_tokens) == (8, 60.0, 200, 8000)
    assert cfg.cache is False and cfg.force is False and cfg.live is True
    assert cfg.base_url == c.BaseUrl("https://api.anthropic.com", "api.anthropic.com", False)
    assert cfg.heap_metrics is c.HeapMetrics.AUTO and cfg.log_level == "INFO"
    assert cfg.api_key is None and cfg.env_file_loaded is False
    assert cfg.input_path == Path("in.xml") and cfg.style_guide_path == Path("guide.docx")
    assert cfg.lang == "en" and cfg.locale_scope is c.LocaleScope.FULL


@pytest.mark.parametrize(
    "flag, env_name, cli_value, env_value, file_value, default, attr, cast",
    [
        ("--model", "PROOFREADER_MODEL", "cli-model", "env-model", "file-model", "claude-opus-5-5", "model", str),
        ("--concurrency", "PROOFREADER_CONCURRENCY", "3", "5", "7", 8, "concurrency", int),
        ("--timeout", "PROOFREADER_TIMEOUT_SECONDS", "11", "22", "33", 60.0, "timeout_seconds", float),
    ],
)
def test_precedence_cli_over_env_over_dotenv_over_default(
    tmp_path: Path, flag, env_name, cli_value, env_value, file_value, default, attr, cast
) -> None:
    dotenv = f"{env_name}={file_value}\n"
    assert getattr(build(tmp_path), attr) == default
    assert getattr(build(tmp_path, env_file=dotenv), attr) == cast(file_value)
    assert getattr(build(tmp_path, environ={env_name: env_value}, env_file=dotenv), attr) == cast(env_value)
    assert getattr(build(tmp_path, [flag, cli_value], {env_name: env_value}, dotenv), attr) == cast(cli_value)


def test_empty_environment_values_fall_through_to_the_next_source(tmp_path: Path) -> None:
    cfg = build(tmp_path, environ={"PROOFREADER_MODEL": "", "PROOFREADER_CONCURRENCY": "  "},
                env_file="PROOFREADER_CONCURRENCY=4\n")
    assert cfg.model == "claude-opus-5-5"
    assert cfg.concurrency == 8  # the (blank) process value shadows .env, then falls back to the default


def test_max_calls_comes_only_from_the_cli_or_the_default(tmp_path: Path) -> None:
    assert build(tmp_path).max_calls == 200
    assert build(tmp_path, ["--max-calls", "5"]).max_calls == 5
    assert build(tmp_path, environ={"PROOFREADER_MAX_CALLS": "5"}).max_calls == 200


@pytest.mark.parametrize(
    "argv, environ, needle",
    [
        (["--concurrency", "0"], {}, "--concurrency"),
        (["--concurrency", "65"], {}, "--concurrency"),
        ([], {"PROOFREADER_CONCURRENCY": "abc"}, "PROOFREADER_CONCURRENCY"),
        ([], {"PROOFREADER_CONCURRENCY": "0"}, "PROOFREADER_CONCURRENCY"),
        ([], {"PROOFREADER_CONCURRENCY": "2.5"}, "PROOFREADER_CONCURRENCY"),
        (["--timeout", "0.5"], {}, "--timeout"),
        (["--timeout", "601"], {}, "--timeout"),
        (["--timeout", "nan"], {}, "--timeout"),
        (["--timeout", "inf"], {}, "--timeout"),
        ([], {"PROOFREADER_TIMEOUT_SECONDS": "soon"}, "PROOFREADER_TIMEOUT_SECONDS"),
        ([], {"PROOFREADER_TIMEOUT_SECONDS": "0"}, "PROOFREADER_TIMEOUT_SECONDS"),
        (["--max-calls", "0"], {}, "--max-calls"),
        (["--max-calls", "100001"], {}, "--max-calls"),
        (["--model", "bad model"], {}, "model"),
        ([], {"PROOFREADER_MODEL": "x" * 101}, "PROOFREADER_MODEL"),
    ],
)
def test_invalid_values_are_usage_errors_naming_their_source(tmp_path: Path, argv, environ, needle) -> None:
    with pytest.raises(c.UsageError) as info:
        build(tmp_path, argv, environ)
    assert info.value.exit_code == 2
    assert needle in info.value.message


def test_bounds_are_inclusive(tmp_path: Path) -> None:
    assert build(tmp_path, ["--concurrency", "1"]).concurrency == 1
    assert build(tmp_path, ["--concurrency", "64"]).concurrency == 64
    assert build(tmp_path, ["--timeout", "1"]).timeout_seconds == 1.0
    assert build(tmp_path, ["--timeout", "600"]).timeout_seconds == 600.0
    assert build(tmp_path, ["--max-calls", "1"]).max_calls == 1
    assert build(tmp_path, ["--max-calls", "100000"]).max_calls == 100000


def test_build_config_rejects_bool_values_for_numbers(tmp_path: Path) -> None:
    base = {"input": "a.xml", "lang": "en", "style_guide": "g.docx"}
    for key in ("concurrency", "max_calls", "timeout"):
        with pytest.raises(c.UsageError):
            c.build_config({**base, key: True}, {}, tmp_path)


def test_build_config_requires_input_lang_and_style_guide(tmp_path: Path) -> None:
    with pytest.raises(c.UsageError, match="input XML path"):
        c.build_config({"lang": "en", "style_guide": "g.docx"}, {}, tmp_path)
    with pytest.raises(c.UsageError, match="--style-guide"):
        c.build_config({"input": "a.xml", "lang": "en", "style_guide": ""}, {}, tmp_path)
    with pytest.raises(c.UsageError, match="--lang"):
        c.build_config({"input": "a.xml", "style_guide": "g.docx"}, {}, tmp_path)


# --------------------------------------------------------------------------------------------
# ./.env (current directory only, override=False, no interpolation, os.environ untouched)
# --------------------------------------------------------------------------------------------


def test_dotenv_is_read_only_from_the_current_directory(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("PROOFREADER_MODEL=parent-model\n", encoding="utf-8")
    child = tmp_path / "child"
    child.mkdir()
    cfg = cli.parse_config(REQUIRED, environ={}, cwd=child)
    assert cfg.model == "claude-opus-5-5" and cfg.env_file_loaded is False


def test_dotenv_does_not_override_the_process_environment(tmp_path: Path) -> None:
    cfg = build(tmp_path, environ={"PROOFREADER_MODEL": "process-model"}, env_file="PROOFREADER_MODEL=file-model\n")
    assert cfg.model == "process-model" and cfg.env_file_loaded is True


def test_dotenv_never_modifies_os_environ(tmp_path: Path) -> None:
    before = dict(os.environ)
    (tmp_path / ".env").write_text(f"ANTHROPIC_API_KEY={SENTINEL_KEY}\nS41_UNIQUE_MARKER=1\n", encoding="utf-8")
    # A copy of the real environment minus the settings this module reads (machine-independent).
    process = {k: v for k, v in os.environ.items()
               if not k.startswith(("ANTHROPIC_", "PROOFREADER_", "PYTHON_DOTENV"))}
    cfg = cli.parse_config(REQUIRED, environ=process, cwd=tmp_path)
    assert dict(os.environ) == before
    assert "S41_UNIQUE_MARKER" not in os.environ
    assert cfg.api_key is not None and cfg.api_key.reveal() == SENTINEL_KEY


def test_dotenv_values_are_not_interpolated(tmp_path: Path) -> None:
    cfg = build(tmp_path, environ={"OTHER": "expanded-value"}, env_file="ANTHROPIC_API_KEY=${OTHER}\n")
    assert cfg.api_key is not None and cfg.api_key.reveal() == "${OTHER}"
    with pytest.raises(c.UsageError):  # "${OTHER}" taken literally is not a valid model id
        build(tmp_path, environ={"OTHER": "fine-model"}, env_file="PROOFREADER_MODEL=${OTHER}\n")


def test_dotenv_keys_without_values_are_ignored(tmp_path: Path) -> None:
    assert build(tmp_path, env_file="PROOFREADER_MODEL\n").model == "claude-opus-5-5"


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "y", "t"])
def test_python_dotenv_disabled_is_honoured(tmp_path: Path, value: str) -> None:
    cfg = build(tmp_path, environ={"PYTHON_DOTENV_DISABLED": value}, env_file="PROOFREADER_MODEL=file-model\n")
    assert cfg.model == "claude-opus-5-5" and cfg.env_file_loaded is False


def test_python_dotenv_disabled_falsy_value_still_loads(tmp_path: Path) -> None:
    cfg = build(tmp_path, environ={"PYTHON_DOTENV_DISABLED": "0"}, env_file="PROOFREADER_MODEL=file-model\n")
    assert cfg.model == "file-model"


def test_oversized_or_undecodable_dotenv_is_a_usage_error(tmp_path: Path) -> None:
    (tmp_path / ".env").write_bytes(b"A=" + b"x" * (c.MAX_ENV_FILE_BYTES + 1))
    with pytest.raises(c.UsageError, match="larger than"):
        cli.parse_config(REQUIRED, environ={}, cwd=tmp_path)
    (tmp_path / ".env").write_bytes(b"PROOFREADER_MODEL=\xff\xfe\n")
    with pytest.raises(c.UsageError, match="UTF-8"):
        cli.parse_config(REQUIRED, environ={}, cwd=tmp_path)


def test_a_directory_named_dotenv_is_ignored(tmp_path: Path) -> None:
    (tmp_path / ".env").mkdir()
    assert cli.parse_config(REQUIRED, environ={}, cwd=tmp_path).env_file_loaded is False


# --------------------------------------------------------------------------------------------
# API key: environment / .env only, redacted everywhere
# --------------------------------------------------------------------------------------------


def test_api_key_from_environment_or_dotenv(tmp_path: Path) -> None:
    assert build(tmp_path, environ={"ANTHROPIC_API_KEY": SENTINEL_KEY}).require_api_key().reveal() == SENTINEL_KEY
    cfg = build(tmp_path, env_file=f"ANTHROPIC_API_KEY={SENTINEL_KEY}\n")
    assert cfg.require_api_key().reveal() == SENTINEL_KEY


@pytest.mark.parametrize("value", [f" {SENTINEL_KEY}", f"{SENTINEL_KEY} ", f"{SENTINEL_KEY}\n", f"\t{SENTINEL_KEY}",
                                   f"{SENTINEL_KEY}\r\n", "   ", "\n"])
def test_api_key_whitespace_is_rejected_never_trimmed(tmp_path: Path, value: str) -> None:
    with pytest.raises(c.UsageError) as info:
        build(tmp_path, environ={"ANTHROPIC_API_KEY": value})
    assert info.value.exit_code == 2 and SENTINEL_KEY not in str(info.value)


def test_quoted_dotenv_key_with_spaces_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(c.UsageError):
        build(tmp_path, env_file=f'ANTHROPIC_API_KEY=" {SENTINEL_KEY} "\n')


def test_missing_api_key_is_exit_1_only_when_required(tmp_path: Path) -> None:
    cfg = build(tmp_path, environ={"ANTHROPIC_API_KEY": ""})
    assert cfg.api_key is None
    with pytest.raises(c.MissingApiKeyError) as info:
        cfg.require_api_key()
    assert info.value.exit_code == 1


def test_there_is_no_cli_flag_for_the_api_key(tmp_path: Path) -> None:
    for flag in ("--api-key", "--key", "--anthropic-api-key"):
        with pytest.raises(SystemExit) as info:
            build(tmp_path, [flag, SENTINEL_KEY])
        assert info.value.code == 2


def test_api_key_never_renders(tmp_path: Path) -> None:
    cfg = build(tmp_path, environ={"ANTHROPIC_API_KEY": SENTINEL_KEY})
    key = cfg.api_key
    for text in (repr(cfg), str(cfg), repr(key), str(key), f"{key}", f"{key!r}", "%s %r" % (key, key),
                 repr(vars(cfg) if hasattr(cfg, "__dict__") else cfg)):
        assert SENTINEL_KEY not in text
    assert "***" in repr(key)


def test_api_key_cannot_be_pickled_copied_mutated_or_compared(tmp_path: Path) -> None:
    key = build(tmp_path, environ={"ANTHROPIC_API_KEY": SENTINEL_KEY}).api_key
    for operation in (pickle.dumps, copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            operation(key)
    with pytest.raises(AttributeError):
        key._value = "other"  # type: ignore[misc]
    assert (key == c.ApiKey(SENTINEL_KEY)) is False  # identity only; no value comparison
    with pytest.raises(TypeError):
        hash(key)


def test_run_config_with_a_key_cannot_be_pickled(tmp_path: Path) -> None:
    cfg = build(tmp_path, environ={"ANTHROPIC_API_KEY": SENTINEL_KEY})
    with pytest.raises(TypeError):
        pickle.dumps(cfg)


@pytest.mark.parametrize("bad", ["two words", "tab\there", "ctl\x01x", "café", "nul\x00x"])
def test_malformed_api_key_is_rejected_without_echoing_it(tmp_path: Path, bad: str) -> None:
    with pytest.raises(c.UsageError) as info:
        build(tmp_path, environ={"ANTHROPIC_API_KEY": bad})
    assert bad.strip() not in info.value.message and "ANTHROPIC_API_KEY" in info.value.message


def test_errors_raised_while_a_key_is_present_do_not_contain_it(tmp_path: Path) -> None:
    for argv in (["--concurrency", "0"], ["--base-url", "https://elsewhere.example"], ["--timeout", "0"]):
        with pytest.raises(c.UsageError) as info:
            build(tmp_path, argv, {"ANTHROPIC_API_KEY": SENTINEL_KEY})
        assert SENTINEL_KEY not in str(info.value) and SENTINEL_KEY not in repr(info.value)


# --------------------------------------------------------------------------------------------
# Base URL policy (Anthropic only by default; explicit CLI opt-in; host-only messages)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("url", [None, "", "https://api.anthropic.com", "https://api.anthropic.com/",
                                 "https://API.Anthropic.com", "https://api.anthropic.com:443"])
def test_default_endpoint_forms(url) -> None:
    assert c.resolve_base_url(url, allow_custom=False) == c.BaseUrl(c.DEFAULT_BASE_URL, "api.anthropic.com", False)


@pytest.mark.parametrize(
    "url",
    [
        "https://proxy.example",
        "https://api.anthropic.com.evil.example",
        "https://evilapi.anthropic.com",
        "https://api.anthropic.com:8443",  # other port
        "https://api.anthropic.com/v2",  # other path
    ],
)
def test_non_default_endpoints_are_rejected_without_opt_in(url: str) -> None:
    with pytest.raises(c.UsageError) as info:
        c.resolve_base_url(url, allow_custom=False)
    assert info.value.exit_code == 2 and "--allow-custom-base-url" in info.value.message


@pytest.mark.parametrize(
    "url",
    ["http://api.anthropic.com", "http://proxy.example", "http://localhost:8080", "http://127.0.0.1:9/v1",
     "HTTP://api.anthropic.com", "ws://proxy.example",
     "https://user:pw@api.anthropic.com", "https://token@proxy.example", "https://api.anthropic.com/?k=v",
     "https://proxy.example/#frag", "ftp://api.anthropic.com", "api.anthropic.com", "https://", "https://[::1",
     "https://proxy.example:99999"],
)
def test_malformed_or_credential_urls_are_rejected_even_with_opt_in(url: str) -> None:
    with pytest.raises(c.UsageError):
        c.resolve_base_url(url, allow_custom=True)


def test_custom_endpoint_requires_the_explicit_flag(tmp_path: Path) -> None:
    cfg = build(tmp_path, ["--base-url", "https://proxy.example/anthropic/", "--allow-custom-base-url"])
    assert cfg.base_url == c.BaseUrl("https://proxy.example/anthropic", "proxy.example", True)


def test_plain_http_is_refused_even_with_the_opt_in(tmp_path: Path) -> None:
    for url in ("http://localhost:8080", "http://api.anthropic.com"):
        with pytest.raises(c.UsageError, match="https"):
            build(tmp_path, ["--base-url", url, "--allow-custom-base-url"], {"ANTHROPIC_API_KEY": SENTINEL_KEY})
    with pytest.raises(c.UsageError, match="https"):
        build(tmp_path, ["--allow-custom-base-url"], {"ANTHROPIC_BASE_URL": "http://proxy.example"})


@pytest.mark.parametrize("where", ["process", "dotenv"])
def test_sdk_custom_headers_variable_is_refused(tmp_path: Path, where: str) -> None:
    value = "Authorization: Bearer injected-value"
    environ = {"ANTHROPIC_CUSTOM_HEADERS": value} if where == "process" else {}
    dotenv = f"ANTHROPIC_CUSTOM_HEADERS={value}\n" if where == "dotenv" else None
    with pytest.raises(c.UsageError) as info:
        build(tmp_path, environ=environ, env_file=dotenv)
    assert "ANTHROPIC_CUSTOM_HEADERS" in info.value.message and "injected-value" not in info.value.message
    assert info.value.exit_code == 2


def test_empty_custom_headers_variable_is_harmless(tmp_path: Path) -> None:
    assert build(tmp_path, environ={"ANTHROPIC_CUSTOM_HEADERS": ""}).model == "claude-opus-5-5"


def test_environment_and_dotenv_cannot_redirect_or_opt_in(tmp_path: Path) -> None:
    for environ, dotenv in (({"ANTHROPIC_BASE_URL": "https://proxy.example"}, None),
                            ({}, "ANTHROPIC_BASE_URL=https://proxy.example\nPROOFREADER_ALLOW_CUSTOM_BASE_URL=1\n"),
                            ({"ANTHROPIC_BASE_URL": "https://proxy.example",
                              "PROOFREADER_ALLOW_CUSTOM_BASE_URL": "1", "ALLOW_CUSTOM_BASE_URL": "1"}, None)):
        with pytest.raises(c.UsageError):
            build(tmp_path, environ=environ, env_file=dotenv)


def test_environment_base_url_is_honoured_when_it_is_the_default_or_opted_in(tmp_path: Path) -> None:
    assert build(tmp_path, environ={"ANTHROPIC_BASE_URL": "https://api.anthropic.com/"}).base_url.custom is False
    cfg = build(tmp_path, ["--allow-custom-base-url"], {"ANTHROPIC_BASE_URL": "https://proxy.example"})
    assert cfg.base_url.host == "proxy.example" and cfg.base_url.custom is True
    cfg = build(tmp_path, ["--base-url", "https://api.anthropic.com"], {"ANTHROPIC_BASE_URL": "https://proxy.example"})
    assert cfg.base_url.custom is False  # CLI wins over the environment


def test_url_messages_name_the_host_only() -> None:
    secret_bits = ("secret-path", "token=abc123", "s3cr3t", "frag-secret")
    for url in ("https://proxy.example/secret-path", "https://proxy.example/x?token=abc123",
                "https://u:s3cr3t@proxy.example/", "https://proxy.example/#frag-secret"):
        with pytest.raises(c.UsageError) as info:
            c.resolve_base_url(url, allow_custom=False)
        message = info.value.message
        assert url not in message and not any(bit in message for bit in secret_bits)


# --------------------------------------------------------------------------------------------
# --lang: RFC 5646 syntax, canonical case, D8 scopes
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag, expected",
    [
        ("en", "en"), ("EN", "en"), ("en-US", "en-US"), ("en_us", "en-US"), ("EN-us", "en-US"),
        ("en-gb", "en-GB"), ("fr", "fr"), ("pt_br", "pt-BR"), ("es-419", "es-419"),
        ("zh-hant-tw", "zh-Hant-TW"), ("sr-LATN-rs", "sr-Latn-RS"), ("zh-yue-HK", "zh-yue-HK"),
        ("de-CH-1996", "de-CH-1996"), ("sl-rozaj-biske", "sl-rozaj-biske"), ("de-1901", "de-1901"),
        ("en-a-BBB-x-Priv", "en-a-bbb-x-priv"), ("en-US-u-ca-gregory", "en-US-u-ca-gregory"),
        ("en-x-a", "en-x-a"), ("haw", "haw"), ("abcde", "abcde"), ("aa-bbb-ccc-ddd", "aa-bbb-ccc-ddd"),
    ],
)
def test_lang_normalisation(tag: str, expected: str) -> None:
    assert c.normalise_lang(tag) == expected


@pytest.mark.parametrize(
    "tag",
    [
        "", "e", "engl", "abcdefghi", "x-private", "i-klingon", "en-GB-oed", "en--US", "-en", "en-", "en US", " en", "en\n",
        "en-a", "en-a-b", "en-x", "en-x-123456789", "en-1996-1996", "en-a-bb-a-cc", "en-US-US", "e1",
        "en-Latn-Latn", "aa-bbb-ccc-ddd-eee", "en-123456789", "café", "еn", "en-Us-x", "12",
        "en-" + "a" * 70, "en.US", "en/US", None, 7,
    ],
)
def test_malformed_lang_is_a_usage_error(tag) -> None:
    with pytest.raises(c.UsageError) as info:
        c.normalise_lang(tag)
    assert info.value.exit_code == 2


BCP47_EDGE_CASES = {
    # malformed extensions: singleton without subtags, subtag of 1 or 9 characters, empty subtag
    "extension": (["en-a", "en-a-b", "en-a-123456789", "en-a--bb", "en-a-x-foo", "en-US-u", "en-u-ca-b"],
                  [("en-a-bb", "en-a-bb"), ("en-US-u-ca-gregory", "en-US-u-ca-gregory"),
                   ("en-a-bbb-b-ccc", "en-a-bbb-b-ccc"), ("en-a-bbb-1996", "en-a-bbb-1996")]),
    # duplicate variants (compared case-insensitively)
    "variant": (["de-1996-1996", "sl-rozaj-ROZAJ", "en-GB-scotland-SCOTLAND", "de-CH-1901-1996-1901"],
                [("de-CH-1901-1996", "de-CH-1901-1996"), ("sl-rozaj-biske-1994", "sl-rozaj-biske-1994")]),
    # duplicate extension singletons (case-insensitive)
    "singleton": (["en-a-bbb-a-ccc", "en-a-bbb-A-ccc", "en-u-ca-gregory-u-nu-latn"],
                  [("en-a-bbb-u-ca-x-a-a", "en-a-bbb-u-ca-x-a-a")]),  # repeats after x- are private use
    # private-use sequences
    "private": (["x-whatever", "en-x", "en-x-", "en-x-123456789", "en-x--a", "x"],
                [("en-x-a", "en-x-a"), ("en-x-a-b-c", "en-x-a-b-c"), ("en-x-12345678", "en-x-12345678"),
                 ("en-X-Abc", "en-x-abc"), ("en-x-a-x-b", "en-x-a-x-b"), ("en-US-x-US", "en-US-x-us")]),
    # invalid subtag ordering
    "ordering": (["en-US-Latn", "de-1996-CH", "zh-Hant-yue", "en-Latn-Latn", "en-US-GB", "en-1996-Latn",
                  "1996-de", "Latn-en", "en-x-a-US-"],
                 [("zh-yue-Hant-HK", "zh-yue-Hant-HK"), ("sr-Latn-RS-1994", "sr-Latn-RS-1994")]),
}


@pytest.mark.parametrize("group", sorted(BCP47_EDGE_CASES))
def test_bcp47_edge_cases(group: str) -> None:
    rejected, accepted = BCP47_EDGE_CASES[group]
    for tag in rejected:
        with pytest.raises(c.UsageError):
            c.normalise_lang(tag)
    for tag, expected in accepted:
        assert c.normalise_lang(tag) == expected, tag


def test_overlong_lang_is_not_echoed() -> None:
    with pytest.raises(c.UsageError) as info:
        c.normalise_lang("en-" + "a" * 200)
    assert "a" * 70 not in info.value.message


@pytest.mark.parametrize(
    "tag, scope",
    [
        ("en", c.LocaleScope.FULL), ("en-US", c.LocaleScope.FULL), ("en-GB", c.LocaleScope.EN_GB),
        ("en-AU", c.LocaleScope.CROSS_LOCALE), ("en-Latn-US", c.LocaleScope.CROSS_LOCALE),
        ("en-US-x-test", c.LocaleScope.CROSS_LOCALE), ("en-GB-scotland", c.LocaleScope.CROSS_LOCALE),
        ("fr", c.LocaleScope.CROSS_LOCALE), ("de-CH", c.LocaleScope.CROSS_LOCALE),
    ],
)
def test_locale_scope_is_exact(tag: str, scope: c.LocaleScope) -> None:
    assert c.locale_scope(c.normalise_lang(tag)) is scope


def test_config_carries_normalised_lang_and_scope(tmp_path: Path) -> None:
    cfg = cli.parse_config(["in.xml", "--lang", "en_gb", "--style-guide", "g.docx"], environ={}, cwd=tmp_path)
    assert (cfg.lang, cfg.locale_scope) == ("en-GB", c.LocaleScope.EN_GB)


_ALPHA = st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=8)
_wellformed = st.builds(
    lambda lang, script, region, sep: sep.join(p for p in (lang, script, region) if p),
    st.one_of(st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=2, max_size=3),
              st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=5, max_size=8)),
    st.one_of(st.just(""), st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                   min_size=4, max_size=4)),
    st.one_of(st.just(""), st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                   min_size=2, max_size=2), st.text(alphabet="0123456789", min_size=3, max_size=3)),
    st.sampled_from(["-", "_"]),
)


@settings(max_examples=300, deadline=None)
@given(_wellformed)
def test_property_wellformed_tags_normalise_idempotently_and_case_insensitively(tag: str) -> None:
    normal = c.normalise_lang(tag)
    assert c.normalise_lang(normal) == normal
    assert c.normalise_lang(tag.swapcase()) == normal
    assert normal.lower() == tag.replace("_", "-").lower()


@settings(max_examples=500, deadline=None)
@given(st.text(max_size=40))
def test_property_any_text_is_normalised_or_rejected_cleanly(text: str) -> None:
    try:
        normal = c.normalise_lang(text)
    except c.UsageError:
        return
    assert normal == c.normalise_lang(normal)
    assert normal.isascii() and " " not in normal and "_" not in normal


# --------------------------------------------------------------------------------------------
# Heap metrics policy (benchmark: tracemalloc ~10x slower on large inputs)
# --------------------------------------------------------------------------------------------


def test_heap_metrics_auto_threshold_and_overrides(tmp_path: Path) -> None:
    limit = c.HEAP_METRICS_AUTO_MAX_BYTES
    assert limit == 256 * 1024
    auto = build(tmp_path)
    assert auto.heap_metrics_enabled(0) and auto.heap_metrics_enabled(limit)
    assert not auto.heap_metrics_enabled(limit + 1)
    on = build(tmp_path, ["--heap-metrics"])
    assert on.heap_metrics is c.HeapMetrics.ON and on.heap_metrics_enabled(10 * limit)
    off = build(tmp_path, ["--no-heap-metrics"])
    assert off.heap_metrics is c.HeapMetrics.OFF and not off.heap_metrics_enabled(0)


def test_build_config_validates_without_argparse(tmp_path: Path) -> None:
    """build_config is also an API: its own checks must not rely on argparse."""
    base = {"input": "a.xml", "lang": "en", "style_guide": "g.docx"}
    with pytest.raises(c.UsageError, match="--replay and --record"):
        c.build_config({**base, "replay": "r.json", "record": "s.json"}, {}, tmp_path)
    with pytest.raises(c.UsageError, match="--log-level"):
        c.build_config({**base, "log_level": "VERBOSE"}, {}, tmp_path)
    assert c.build_config({**base, "log_level": "warning"}, {}, tmp_path).log_level == "WARNING"


def test_api_key_field_is_hidden_from_the_config_repr(tmp_path: Path) -> None:
    text = repr(build(tmp_path, environ={"ANTHROPIC_API_KEY": SENTINEL_KEY}))
    assert "api_key" not in text and "ApiKey" not in text and SENTINEL_KEY not in text


def test_build_config_rejects_both_heap_switches(tmp_path: Path) -> None:
    with pytest.raises(c.UsageError):
        c.build_config({"input": "a", "lang": "en", "style_guide": "g", "heap_metrics": True,
                        "no_heap_metrics": True}, {}, tmp_path)


# --------------------------------------------------------------------------------------------
# CLI parser
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        [],  # nothing
        ["in.xml", "--lang", "en"],  # --style-guide missing
        ["in.xml", "--style-guide", "g.docx"],  # --lang missing
        ["--lang", "en", "--style-guide", "g.docx"],  # input missing
        REQUIRED + ["--replay", "r.json", "--record", "s.json"],
        REQUIRED + ["--heap-metrics", "--no-heap-metrics"],
        REQUIRED + ["--concurrency", "many"],
        REQUIRED + ["--timeout", "soon"],
        REQUIRED + ["--log-level", "VERBOSE"],
        REQUIRED + ["--style", "abbreviated.docx"],  # allow_abbrev=False
        REQUIRED + ["--unknown"],
        REQUIRED + ["extra.xml"],
    ],
)
def test_malformed_command_lines_exit_2(tmp_path: Path, argv, capsys) -> None:
    with pytest.raises(SystemExit) as info:
        cli.parse_config(argv, environ={}, cwd=tmp_path)
    assert info.value.code == 2
    capsys.readouterr()


def test_all_flags_reach_the_config(tmp_path: Path) -> None:
    cfg = build(tmp_path, ["--replay", "rec.json", "--model", "m-1", "--concurrency", "2", "--timeout", "5",
                           "--max-calls", "9", "--cache", "--force", "--log-level", "debug", "--log-file", "run.log",
                           "--metrics-json", "m.json", "--no-heap-metrics"])
    assert cfg.replay_path == Path("rec.json") and cfg.record_path is None and cfg.live is False
    assert (cfg.model, cfg.concurrency, cfg.timeout_seconds, cfg.max_calls) == ("m-1", 2, 5.0, 9)
    assert cfg.cache and cfg.force and cfg.log_level == "DEBUG"
    assert cfg.log_file == Path("run.log") and cfg.metrics_json == Path("m.json")
    assert cfg.heap_metrics is c.HeapMetrics.OFF
    rec = build(tmp_path, ["--record", "out.json"])
    assert rec.record_path == Path("out.json") and rec.live is True


def test_empty_path_arguments_are_usage_errors(tmp_path: Path) -> None:
    for argv in (["", "--lang", "en", "--style-guide", "g.docx"], REQUIRED + ["--log-file", ""]):
        with pytest.raises(c.UsageError):
            cli.parse_config(argv, environ={}, cwd=tmp_path)


def test_usage_error_is_a_proofreader_error_with_exit_2() -> None:
    from proofreader.models import ProofreaderError

    assert issubclass(c.UsageError, ProofreaderError) and c.UsageError("x").exit_code == 2
    assert issubclass(c.MissingApiKeyError, ProofreaderError) and c.MissingApiKeyError("x").exit_code == 1


# --------------------------------------------------------------------------------------------
# S5: provider selection from the model id (Gemini for "gemini-...", Anthropic otherwise)
# --------------------------------------------------------------------------------------------

GEMINI = "gemini-2.5-flash-lite"
GEMINI_KEY = "test-gemini-key-0123456789"  # invented


def test_default_model_and_provider_are_unchanged(tmp_path: Path) -> None:
    cfg = build(tmp_path, environ={"ANTHROPIC_API_KEY": SENTINEL_KEY, "GEMINI_API_KEY": GEMINI_KEY})
    assert cfg.model == "claude-opus-5-5" and cfg.provider == c.PROVIDER_ANTHROPIC == "anthropic"
    assert cfg.api_key.reveal() == SENTINEL_KEY and cfg.api_key_variable == "ANTHROPIC_API_KEY"
    assert cfg.base_url == c.BaseUrl("https://api.anthropic.com", "api.anthropic.com", False)


@pytest.mark.parametrize("how", ["flag", "environment", "dotenv"])
def test_gemini_model_selects_gemini_with_its_own_key_and_endpoint(tmp_path: Path, how: str) -> None:
    environ = {"ANTHROPIC_API_KEY": SENTINEL_KEY, "GEMINI_API_KEY": GEMINI_KEY}
    if how == "flag":
        cfg = build(tmp_path, ["--model", GEMINI], environ=environ)
    elif how == "environment":
        cfg = build(tmp_path, environ={**environ, "PROOFREADER_MODEL": GEMINI})
    else:
        cfg = build(tmp_path, environ=environ, env_file=f"PROOFREADER_MODEL={GEMINI}\n")
    assert cfg.model == GEMINI and cfg.provider == c.PROVIDER_GEMINI == "gemini"
    assert cfg.api_key.reveal() == GEMINI_KEY and cfg.api_key_variable == "GEMINI_API_KEY"
    assert cfg.base_url == c.BaseUrl("https://generativelanguage.googleapis.com",
                                     "generativelanguage.googleapis.com", False)
    assert (cfg.effort, cfg.max_tokens) == ("low", 8000)


def test_gemini_key_can_come_from_dotenv(tmp_path: Path) -> None:
    cfg = build(tmp_path, ["--model", GEMINI], env_file=f"GEMINI_API_KEY={GEMINI_KEY}\n")
    assert cfg.api_key.reveal() == GEMINI_KEY


def test_anthropic_key_is_never_used_for_gemini_and_vice_versa(tmp_path: Path) -> None:
    gem = build(tmp_path, ["--model", GEMINI], environ={"ANTHROPIC_API_KEY": SENTINEL_KEY})
    assert gem.api_key is None
    with pytest.raises(c.MissingApiKeyError) as info:
        gem.require_api_key()
    assert "GEMINI_API_KEY is not set" in str(info.value) and "ANTHROPIC" not in str(info.value)
    anth = build(tmp_path, environ={"GEMINI_API_KEY": GEMINI_KEY})
    assert anth.api_key is None
    with pytest.raises(c.MissingApiKeyError, match="ANTHROPIC_API_KEY is not set"):
        anth.require_api_key()


@pytest.mark.parametrize("bad", [" lead", "trail ", "in side", "tab\tkey", "nön-ascii"])
def test_malformed_gemini_key_is_rejected_naming_its_variable_without_echoing_it(tmp_path: Path, bad: str) -> None:
    with pytest.raises(c.UsageError) as info:
        build(tmp_path, ["--model", GEMINI], environ={"GEMINI_API_KEY": bad})
    message = str(info.value)
    assert "GEMINI_API_KEY" in message and bad.strip() not in message


def test_gemini_key_never_renders(tmp_path: Path) -> None:
    cfg = build(tmp_path, ["--model", GEMINI], environ={"GEMINI_API_KEY": GEMINI_KEY})
    for text in (repr(cfg), str(cfg), repr(cfg.api_key), f"{cfg.api_key}"):
        assert GEMINI_KEY not in text
    with pytest.raises(TypeError):
        pickle.dumps(cfg)


@pytest.mark.parametrize("model", ["gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-1.5-pro-002", "gemini-x",
                                   "gemini-2.5-flash-lite-preview-06-17", "gemini-9"])
def test_well_formed_gemini_ids_are_accepted(tmp_path: Path, model: str) -> None:
    assert build(tmp_path, ["--model", model]).provider == "gemini"


@pytest.mark.parametrize("model", [
    "gemini-", "gemini-2.5-", "gemini-2.5.", "gemini-2.5_flash", "gemini-2.5:flash", "gemini-2.5-Flash",
    "gemini--x", "gemini-.x",
])
def test_invalid_gemini_ids_are_clear_usage_errors(tmp_path: Path, model: str) -> None:
    with pytest.raises(c.UsageError, match="Gemini model id is invalid"):
        build(tmp_path, ["--model", model])


@pytest.mark.parametrize("model", ["Gemini-2.5-flash-lite", "GEMINI-2.5", "gemini", "gemini2.5", "gemini_2.5",
                                   "my-gemini-model", "claude-gemini", "x-GeMiNi"])
def test_ambiguous_gemini_ids_are_clear_usage_errors_never_routed_to_anthropic(tmp_path: Path, model: str) -> None:
    with pytest.raises(c.UsageError, match="ambiguous"):
        build(tmp_path, ["--model", model])


def test_ids_rejected_by_the_general_rule_still_fail_first(tmp_path: Path) -> None:
    for model in ("models/gemini-2.5-flash-lite", "gemini 2.5", "_gemini-x"):
        with pytest.raises(c.UsageError):
            build(tmp_path, ["--model", model])


def test_provider_for_is_exact() -> None:
    assert c.provider_for("claude-opus-5-5") == "anthropic" and c.provider_for("m-1") == "anthropic"
    assert c.provider_for("gemini-2.5-flash-lite") == "gemini"


@pytest.mark.parametrize("argv, environ, env_file", [
    (["--base-url", "https://generativelanguage.googleapis.com"], {}, None),
    (["--base-url", "https://proxy.invalid", "--allow-custom-base-url"], {}, None),
    (["--allow-custom-base-url"], {}, None),
    ([], {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"}, None),
    ([], {}, "ANTHROPIC_BASE_URL=https://redirect.invalid\n"),
])
def test_base_url_settings_are_refused_for_gemini(tmp_path: Path, argv, environ, env_file) -> None:
    with pytest.raises(c.UsageError, match="apply only to Anthropic models"):
        build(tmp_path, ["--model", GEMINI, *argv], environ=environ, env_file=env_file)


def test_empty_base_url_variable_is_harmless_for_gemini(tmp_path: Path) -> None:
    cfg = build(tmp_path, ["--model", GEMINI], environ={"ANTHROPIC_BASE_URL": ""})
    assert cfg.base_url.host == "generativelanguage.googleapis.com"


def test_custom_headers_variable_is_still_refused_for_gemini(tmp_path: Path) -> None:
    with pytest.raises(c.UsageError, match="ANTHROPIC_CUSTOM_HEADERS"):
        build(tmp_path, ["--model", GEMINI], environ={"ANTHROPIC_CUSTOM_HEADERS": "x-api-key: y"})


def test_api_key_default_variable_name_is_unchanged() -> None:
    with pytest.raises(c.UsageError, match="ANTHROPIC_API_KEY contains whitespace"):
        c.ApiKey("has space")
    with pytest.raises(c.UsageError, match="GEMINI_API_KEY is empty"):
        c.ApiKey("", "GEMINI_API_KEY")

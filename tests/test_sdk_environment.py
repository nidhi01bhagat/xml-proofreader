"""S4.1: which environment variables can still influence an explicitly configured Anthropic client?

Offline evidence for the S4.5 client-construction requirements (docs/04, "S4 SDK environment").
Every request goes to an in-memory ``httpx2.MockTransport``; credentials are dummies.
These tests pin the behaviour of the installed SDK: if a future SDK version changes it, they fail
and the policy must be reviewed.
"""

from __future__ import annotations

import json
import subprocess
import sys

import anthropic
import httpx2
import pytest

from proofreader import config as c

EXPLICIT_KEY = "explicit-dummy-key"
MARKERS = ("ENV-AUTH-TOKEN", "env-base.example", "PROFILE-SECRET", "profile.example", "FEDERATION")


def _client(seen: list) -> anthropic.Anthropic:
    """The construction S4.5 must use: every setting explicit, telemetry off, no environment trust."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={
            "id": "msg_t", "type": "message", "role": "assistant", "model": "m",
            "content": [{"type": "text", "text": "{}"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}})

    return anthropic.Anthropic(
        api_key=EXPLICIT_KEY,
        base_url=c.DEFAULT_BASE_URL,
        max_retries=0,
        timeout=5,
        open_telemetry={"traces": False, "propagation": False},
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler), trust_env=False),
    )


def _send(seen: list) -> httpx2.Request:
    _client(seen).messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "x"}])
    return seen[-1]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS",
                 "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_explicit_configuration_ignores_environment_credentials_base_url_and_profiles(clean_env, tmp_path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "hostile.json").write_text(json.dumps({"api_key": "PROFILE-SECRET", "base_url": "https://profile.example"}),
                                          encoding="utf-8")
    clean_env.setenv("ANTHROPIC_AUTH_TOKEN", "ENV-AUTH-TOKEN")
    clean_env.setenv("ANTHROPIC_BASE_URL", "https://env-base.example")
    clean_env.setenv("ANTHROPIC_PROFILE", "hostile")
    clean_env.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    clean_env.setenv("ANTHROPIC_IDENTITY_TOKEN", "FEDERATION-TOKEN")
    clean_env.setenv("ANTHROPIC_FEDERATION_RULE_ID", "FEDERATION-RULE")
    clean_env.setenv("ANTHROPIC_ORGANIZATION_ID", "FEDERATION-ORG")
    seen: list = []
    request = _send(seen)
    assert request.url.host == "api.anthropic.com" and request.url.scheme == "https"
    blob = str(request.url) + " " + " ".join(f"{k}: {v}" for k, v in request.headers.items())
    assert EXPLICIT_KEY in blob
    assert not any(marker in blob for marker in MARKERS)


def test_custom_headers_variable_reaches_requests_and_can_replace_the_key(clean_env) -> None:
    """Why config.py refuses ANTHROPIC_CUSTOM_HEADERS: explicit configuration does NOT block it."""
    clean_env.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Injected: yes\nAuthorization: Bearer INJECTED")
    seen: list = []
    request = _send(seen)
    assert request.headers.get("x-injected") == "yes"
    assert request.headers.get("authorization") == "Bearer INJECTED"
    assert EXPLICIT_KEY not in " ".join(request.headers.values())


def test_config_refuses_the_custom_headers_variable(tmp_path) -> None:
    with pytest.raises(c.UsageError, match="ANTHROPIC_CUSTOM_HEADERS"):
        c.build_config({"input": "a.xml", "lang": "en", "style_guide": "g.docx"},
                       {"ANTHROPIC_CUSTOM_HEADERS": "Authorization: Bearer x"}, tmp_path)


def test_sdk_log_variable_raises_the_sdk_logger_to_debug_at_import() -> None:
    """ANTHROPIC_LOG=debug makes the SDK log request options (paragraph text): S4.7 must pin the
    'anthropic' and 'httpx2' loggers regardless of the environment."""
    code = "import logging, anthropic; print(logging.getLogger('anthropic').level)"
    env = {"ANTHROPIC_LOG": "debug", "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert out.stdout.strip() == "10"


def test_trust_env_false_client_has_no_environment_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    """httpx2 reads HTTP(S)_PROXY / ALL_PROXY / SSL_CERT_* only when trust_env is true."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:9")
    trusting = httpx2.Client()
    strict = httpx2.Client(trust_env=False)
    try:
        assert any(m is not None for m in trusting._mounts.values())  # env proxy mounted
        assert all(m is None for m in strict._mounts.values()) or not strict._mounts
    finally:
        trusting.close()
        strict.close()

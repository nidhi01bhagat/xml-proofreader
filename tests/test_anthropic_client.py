"""S4.5: AnthropicProofreader - request shape, decoding, error classification, retries, recording.

Offline only: every request goes to an in-memory httpx2.MockTransport through the REAL anthropic SDK.
The API key is an invented dummy value; no live call is ever made.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import httpx2
import pytest

from proofreader import config as c
from proofreader import llm
from proofreader import prompt as pr
from proofreader import resolver as r
from proofreader import xml_source as xs
from proofreader.llm import (AnthropicProofreader, FailureClass, ProofreadResult, ProviderFatalError, RecordingWriter,
                             ReplayProofreader, Usage)

DUMMY_KEY = "dummy-test-key-not-real-0123456789"
SYSTEM_TEXT = "SYSTEM-PROMPT-SENTINEL"
USER_TEXT = "USER-PARAGRAPH-SENTINEL"
BODY_SECRET = "ERROR-BODY-SENTINEL"
GOOD = {"findings": [{"original": "recieve", "before": "Our team will", "after": "the final report",
                      "word_start": 3, "word_end": 3, "correction": "receive", "type": "spelling",
                      "reason": "Misspelling"}]}


@pytest.fixture
def cfg(tmp_path) -> c.RunConfig:
    return c.build_config({"input": "a.xml", "lang": "en", "style_guide": "g.docx"},
                          {"ANTHROPIC_API_KEY": DUMMY_KEY}, tmp_path)


def message(text: str | None = '{"findings": []}', *, stop: str = "end_turn", blocks: list | None = None,
            usage: dict | None = None) -> dict:
    content = blocks if blocks is not None else (
        [{"type": "thinking", "thinking": "t", "signature": "s"}] + ([{"type": "text", "text": text}] if text is not None
                                                                     else []))
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": content,
            "stop_reason": stop, "stop_sequence": None,
            "usage": usage or {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 0,
                               "cache_read_input_tokens": 0}}


def ok_response(body: dict | None = None, request_id: str = "req_ok") -> httpx2.Response:
    return httpx2.Response(200, json=body or message(), headers={"request-id": request_id})


def error_response(status: int, headers: dict | None = None) -> httpx2.Response:
    return httpx2.Response(status, json={"type": "error", "error": {"type": "x", "message": BODY_SECRET}},
                           headers={"request-id": f"req_{status}", **(headers or {})})


class Script:
    """Serves queued responses (or raises queued exceptions) and records every HTTP request."""

    def __init__(self, *items):
        self.items = list(items)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(item, BaseException):
            raise item
        return item


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self) -> float:
        self.now += 0.25
        return self.now


def client(cfg, script: Script, **kwargs) -> AnthropicProofreader:
    sleeps = kwargs.pop("sleeps", [])
    http = httpx2.Client(transport=httpx2.MockTransport(script), trust_env=False)
    return AnthropicProofreader(cfg, http_client=http, sleep=sleeps.append, clock=kwargs.pop("clock", Clock()),
                                **kwargs)


def request(cfg, paragraph_id: int = 0, user: str = USER_TEXT):
    return llm.make_request(cfg, paragraph_id, system=SYSTEM_TEXT, user=user, schema=pr.output_schema())


# --------------------------------------------------------------------------------------------
# Construction and request shape
# --------------------------------------------------------------------------------------------


def test_client_conforms_to_the_protocol(cfg) -> None:
    assert isinstance(client(cfg, Script(ok_response())), llm.Proofreader)


def test_missing_api_key_is_fatal_before_any_request(tmp_path) -> None:
    cfg = c.build_config({"input": "a.xml", "lang": "en", "style_guide": "g.docx"}, {}, tmp_path)
    with pytest.raises(c.MissingApiKeyError):
        AnthropicProofreader(cfg)


def test_request_shape_uses_the_request_and_config_exactly(cfg) -> None:
    script = Script(ok_response())
    req = request(cfg)
    client(cfg, script).proofread(req)
    (sent,) = script.requests
    body = json.loads(sent.content)
    assert sent.method == "POST" and sent.url.host == "api.anthropic.com" and sent.url.path == "/v1/messages"
    assert set(body) == {"model", "max_tokens", "system", "messages", "output_config"}
    assert body["model"] == cfg.model == "claude-opus-5-5" and body["max_tokens"] == 8000
    assert body["system"] == SYSTEM_TEXT and body["messages"] == [{"role": "user", "content": USER_TEXT}]
    assert body["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": req.schema}}
    assert body["output_config"]["format"]["schema"] == pr.output_schema()


def test_environment_cannot_redirect_or_inject(cfg, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://env-redirect.invalid")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ENV-AUTH-TOKEN")
    script = Script(ok_response())
    client(cfg, script).proofread(request(cfg))
    sent = script.requests[0]
    assert sent.url.host == "api.anthropic.com"
    assert "ENV-AUTH-TOKEN" not in " ".join(sent.headers.values())


def test_custom_https_endpoint_is_used_only_when_opted_in(tmp_path) -> None:
    cfg = c.build_config({"input": "a.xml", "lang": "en", "style_guide": "g.docx",
                          "base_url": "https://proxy.example/anthropic", "allow_custom_base_url": True},
                         {"ANTHROPIC_API_KEY": DUMMY_KEY}, tmp_path)
    script = Script(ok_response())
    client(cfg, script).proofread(request(cfg))
    assert script.requests[0].url.host == "proxy.example"


def test_default_http_client_ignores_the_environment(cfg) -> None:
    proofreader = AnthropicProofreader(cfg)
    try:
        assert proofreader._http_client.trust_env is False
    finally:
        proofreader.close()


def test_mismatched_request_settings_are_refused(cfg) -> None:
    import dataclasses

    other = llm.make_request(dataclasses.replace(cfg, model="other-model"), 0, system="S", user="U",
                             schema={"type": "object"})
    with pytest.raises(ValueError, match="do not match"):
        client(cfg, Script(ok_response())).proofread(other)


def test_invalid_constructor_arguments(cfg) -> None:
    for bad in (-1, 6, True, 1.0):
        with pytest.raises(ValueError):
            AnthropicProofreader(cfg, max_retries=bad)
    with pytest.raises(ValueError):
        AnthropicProofreader(cfg, recorder="not a writer")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------
# Successful decoding
# --------------------------------------------------------------------------------------------


def test_clean_response_and_accounting(cfg) -> None:
    script = Script(ok_response(message(usage={"input_tokens": 120, "output_tokens": 33,
                                               "cache_creation_input_tokens": 7, "cache_read_input_tokens": 5})))
    result = client(cfg, script).proofread(request(cfg))
    assert result.payload == {"findings": []} and result.failure is None
    assert result.usage == Usage(120, 33, 7, 5) and result.attempts == 1
    assert result.request_id == "req_ok" and result.stop_reason == "end_turn"
    assert result.latency_ms == pytest.approx(250.0)  # two clock reads, 0.25 s apart


def test_findings_response_is_returned_unchanged_and_resolved_by_the_resolver(cfg) -> None:
    script = Script(ok_response(message(json.dumps(GOOD))))
    result = client(cfg, script).proofread(request(cfg))
    assert result.payload == GOOD
    scan = xs.scan(b"<doc><p>Our team will recieve the final report on Monday.</p></doc>")
    (resolved,) = r.resolve_document(scan, {0: result.payload})
    assert resolved.resolved[0].annotation.correction == "receive"


def test_per_finding_problems_and_too_many_findings_are_not_schema_retried(cfg) -> None:
    bad_item = {"findings": [{"original": 5}]}
    too_many = {"findings": [GOOD["findings"][0]] * 31}
    for payload in (bad_item, too_many):
        script = Script(ok_response(message(json.dumps(payload))))
        result = client(cfg, script).proofread(request(cfg))
        assert result.failure is None and result.payload == payload and result.attempts == 1  # resolver decides


def test_thinking_blocks_are_ignored(cfg) -> None:
    blocks = [{"type": "redacted_thinking", "data": "x"}, {"type": "thinking", "thinking": "a", "signature": "s"},
              {"type": "text", "text": '{"findings": []}'}]
    assert client(cfg, Script(ok_response(message(blocks=blocks)))).proofread(request(cfg)).failure is None


# --------------------------------------------------------------------------------------------
# Unusable output: one schema retry, then SCHEMA_INVALID (never repaired)
# --------------------------------------------------------------------------------------------

UNUSABLE = {
    "invalid json": message("{not json"),
    "duplicate keys": message('{"findings": [], "findings": []}'),
    "nan": message('{"findings": [NaN]}'),
    "trailing data": message('{"findings": []} {}'),
    "not an object": message('["findings"]'),
    "missing findings": message('{"other": []}'),
    "extra top-level key": message('{"findings": [], "note": "x"}'),
    "empty text": message(""),
    "no text block": message(None),
    "no content": message(blocks=[]),
    "two text blocks": message(blocks=[{"type": "text", "text": '{"findings": []}'},
                                       {"type": "text", "text": '{"findings": []}'}]),  # each valid alone
    "tool use block": message(blocks=[{"type": "tool_use", "id": "t", "name": "x", "input": {}}]),
    "tool use beside valid text": message(blocks=[{"type": "text", "text": '{"findings": []}'},
                                                  {"type": "tool_use", "id": "t", "name": "x", "input": {}}]),
    "unexpected stop": message('{"findings": []}', stop="tool_use"),
    "unexpected structure": {"unexpected": True},
}


@pytest.mark.parametrize("name", sorted(UNUSABLE))
def test_unusable_output_is_retried_once_then_reported(cfg, name: str) -> None:
    script = Script(ok_response(UNUSABLE[name]))
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is FailureClass.SCHEMA_INVALID
    assert result.attempts == 2 and len(script.requests) == 2  # exactly one schema retry
    assert result.payload is None or isinstance(result.payload, (dict, list))


def test_envelope_invalid_payload_is_kept_for_tracing(cfg) -> None:
    script = Script(ok_response(message('{"other": 1}')))
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is FailureClass.SCHEMA_INVALID and result.payload == {"other": 1}


def test_schema_retry_success_sums_usage(cfg) -> None:
    script = Script(ok_response(message("{bad")), ok_response(message(json.dumps(GOOD)), request_id="req_second"))
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is None and result.payload == GOOD and result.attempts == 2
    assert result.usage == Usage(20, 10) and result.request_id == "req_second"


def test_decode_response_is_strict() -> None:
    assert llm.decode_response('{"findings": []}') == {"findings": []}
    for bad in ("{bad", '{"a": 1, "a": 2}', "[NaN]", "[1e400]", "{} {}", "[" * 100000 + "]" * 100000):
        with pytest.raises(ValueError):
            llm.decode_response(bad)
    with pytest.raises(ValueError):
        llm.decode_response(b"{}")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------
# Final stop reasons (never retried)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("stop, failure", [("refusal", FailureClass.REFUSAL), ("max_tokens", FailureClass.MAX_TOKENS),
                                           ("model_context_window_exceeded", FailureClass.CONTEXT_WINDOW_EXCEEDED)])
def test_refusal_and_truncation_are_final(cfg, stop: str, failure: FailureClass) -> None:
    script = Script(ok_response(message('{"findings": [', stop=stop)))
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is failure and result.stop_reason == stop and result.payload is None
    assert result.attempts == 1 and len(script.requests) == 1 and result.usage == Usage(10, 5)


# --------------------------------------------------------------------------------------------
# Transient failures: bounded retries
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item, failure",
    [
        (error_response(429), FailureClass.RATE_LIMITED), (error_response(500), FailureClass.SERVER_ERROR),
        (error_response(503), FailureClass.SERVER_ERROR), (error_response(529), FailureClass.SERVER_ERROR),
        (error_response(408), FailureClass.TIMEOUT), (error_response(409), FailureClass.SERVER_ERROR),
        (httpx2.ReadTimeout("slow"), FailureClass.TIMEOUT), (httpx2.ConnectError("down"), FailureClass.CONNECTION),
    ],
    ids=["429", "500", "503", "529", "408", "409", "timeout", "connection"],
)
def test_transient_failures_are_retried_then_reported(cfg, item, failure: FailureClass) -> None:
    script = Script(item)
    sleeps: list[float] = []
    result = client(cfg, script, sleeps=sleeps).proofread(request(cfg))
    assert result.failure is failure and result.payload is None
    assert result.attempts == 3 == len(script.requests)  # 1 + MAX_TRANSPORT_RETRIES
    assert sleeps == [0.5, 1.0]  # bounded exponential backoff
    assert llm.CallRecord.from_result(0, result).status is llm.CallStatus.PASSTHROUGH


def test_transient_then_success_counts_every_request(cfg) -> None:
    script = Script(error_response(429), httpx2.ConnectError("blip"), ok_response())
    sleeps: list[float] = []
    result = client(cfg, script, sleeps=sleeps).proofread(request(cfg))
    assert result.failure is None and result.attempts == 3 and len(sleeps) == 2


def test_retry_after_is_honoured_and_capped(cfg) -> None:
    for headers, expected in (({"retry-after": "3"}, 3.0), ({"retry-after-ms": "250"}, 0.25),
                              ({"retry-after": "9999"}, llm.RETRY_AFTER_CAP_S), ({"retry-after": "soon"}, 0.5),
                              ({"retry-after": "-5"}, 0.5), ({"retry-after": "nan"}, 0.5),
                              ({"retry-after": "inf"}, 0.5)):
        sleeps: list[float] = []
        client(cfg, Script(error_response(429, headers)), sleeps=sleeps).proofread(request(cfg))
        assert sleeps[0] == expected


def test_backoff_is_bounded() -> None:
    assert [llm._retry_delay(i, None) for i in range(8)] == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0, 8.0, 8.0]


def test_retry_limit_is_respected(cfg) -> None:
    for retries in (0, 1, 3):
        script = Script(error_response(500))
        result = client(cfg, script, max_retries=retries).proofread(request(cfg))
        assert result.attempts == retries + 1 == len(script.requests)


def test_transport_and_schema_retries_combine_with_exact_attempts(cfg) -> None:
    script = Script(error_response(503), ok_response(message("{bad")), error_response(429), ok_response())
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is None and result.attempts == 4 == len(script.requests)


# --------------------------------------------------------------------------------------------
# Paragraph-level 413 and run-fatal 4xx
# --------------------------------------------------------------------------------------------


def test_request_too_large_is_paragraph_level_and_not_retried(cfg) -> None:
    script = Script(error_response(413))
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is FailureClass.REQUEST_TOO_LARGE and result.attempts == 1 == len(script.requests)
    assert result.request_id == "req_413"


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422, 412, 431, 499])
def test_permanent_4xx_is_run_fatal_never_retried_and_redacted(cfg, status: int) -> None:
    script = Script(error_response(status))
    proofreader = client(cfg, script)
    with pytest.raises(ProviderFatalError) as info:
        proofreader.proofread(request(cfg))
    assert info.value.exit_code == 1 and isinstance(info.value, llm.RunFatalError)
    assert len(script.requests) == 1
    text = str(info.value) + repr(info.value)
    for secret in (DUMMY_KEY, BODY_SECRET, USER_TEXT, SYSTEM_TEXT):
        assert secret not in text
    assert f"HTTP {status}" in text and f"req_{status}" in text


def test_non_anthropic_exceptions_and_interrupts_propagate(cfg, monkeypatch) -> None:
    proofreader = client(cfg, Script(ok_response()))

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(proofreader._client.messages, "create", interrupted)
    with pytest.raises(KeyboardInterrupt):
        proofreader.proofread(request(cfg))


# --------------------------------------------------------------------------------------------
# Redaction and logging
# --------------------------------------------------------------------------------------------


def test_repr_and_logs_never_expose_the_key_or_text(cfg, caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="proofreader")
    script = Script(error_response(429), ok_response(message(json.dumps(GOOD))))
    proofreader = client(cfg, script)
    result = proofreader.proofread(request(cfg))
    for text in (repr(proofreader), repr(result)):
        for secret in (DUMMY_KEY, USER_TEXT, SYSTEM_TEXT, "recieve"):
            assert secret not in text
    assert [r for r in caplog.records if r.name.startswith("proofreader")] == []


def test_api_key_is_sent_only_as_the_auth_header(cfg) -> None:
    script = Script(ok_response())
    client(cfg, script).proofread(request(cfg))
    sent = script.requests[0]
    assert DUMMY_KEY not in sent.content.decode("utf-8") and DUMMY_KEY not in str(sent.url)
    assert any(DUMMY_KEY in value for value in sent.headers.values())  # authentication header only


# --------------------------------------------------------------------------------------------
# Recording integration
# --------------------------------------------------------------------------------------------


def test_recording_round_trip_through_replay(cfg, tmp_path) -> None:
    writer = RecordingWriter()
    script = Script(ok_response(message(json.dumps(GOOD))), error_response(429), error_response(429),
                    error_response(429))
    live = client(cfg, script, recorder=writer)
    first, second = request(cfg, 0, user="paragraph one"), request(cfg, 1, user="paragraph two")
    live_results = [live.proofread(first), live.proofread(second)]
    target = tmp_path / "run.recording.json"
    writer.write(target)
    replay = ReplayProofreader.from_file(target)
    assert [replay.proofread(first), replay.proofread(second)] == live_results
    assert live_results[1].failure is FailureClass.RATE_LIMITED
    text = target.read_text(encoding="utf-8")
    # Never serialised: request text, system prompt, credentials, headers, configuration (model id).
    for secret in (DUMMY_KEY, SYSTEM_TEXT, "paragraph one", "paragraph two", "claude-opus-5-5", "Authorization",
                   "x-api-key", "output_config", "max_tokens"):
        assert secret not in text
    # NOT text-free: the response payload is recorded as-is and quotes source-paragraph fragments.
    assert "recieve" in text and "Our team will" in text
    assert set(json.loads(text)["entries"][0]) == {"fingerprint", "result", "failure", "payload", "usage",
                                                   "attempts", "latency_ms", "request_id", "stop_reason"}


def test_identical_requests_are_recorded_once(cfg) -> None:
    writer = RecordingWriter()
    live = client(cfg, Script(ok_response()), recorder=writer)
    live.proofread(request(cfg, 0, user="same"))
    live.proofread(request(cfg, 7, user="same"))  # same fingerprint, different paragraph
    assert len(writer) == 1


def test_run_fatal_outcomes_are_not_recorded(cfg) -> None:
    writer = RecordingWriter()
    with pytest.raises(ProviderFatalError):
        client(cfg, Script(error_response(401)), recorder=writer).proofread(request(cfg))
    assert len(writer) == 0


def test_recorded_fingerprint_equals_the_request_fingerprint(cfg, tmp_path) -> None:
    writer = RecordingWriter()
    req = request(cfg)
    client(cfg, Script(ok_response()), recorder=writer).proofread(req)
    target = tmp_path / "fp.recording.json"
    writer.write(target)
    assert list(llm.load_recording(target)) == [req.fingerprint]
    assert req.fingerprint == pr.fingerprint_for(cfg, system=SYSTEM_TEXT, user=USER_TEXT, schema=pr.output_schema())


def test_results_are_valid_proofread_results(cfg) -> None:
    result = client(cfg, Script(ok_response())).proofread(request(cfg))
    assert isinstance(result, ProofreadResult)
    assert Path(__file__).name == "test_anthropic_client.py"


# --------------------------------------------------------------------------------------------
# S4.5 final audit: environment headers, SDK debug logging, retry bounds
# --------------------------------------------------------------------------------------------

INJECTED = "\n".join(["x-api-key: INJECTED-KEY", "Authorization: Bearer INJECTED-TOKEN", "X-Injected: yes"])


def test_custom_headers_set_after_config_validation_are_refused(cfg, monkeypatch) -> None:
    """S4.1 validates the environment once; the SDK re-reads os.environ when the client is built."""
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", INJECTED)
    script = Script(ok_response())
    with pytest.raises(c.UsageError) as info:
        client(cfg, script)
    assert info.value.exit_code == 2 and script.requests == []
    assert "INJECTED" not in str(info.value) and "ANTHROPIC_CUSTOM_HEADERS" in str(info.value)


def test_custom_headers_guard_is_needed(cfg, monkeypatch) -> None:
    """Positive control: without the guard the SDK replaces the configured key and adds headers."""
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", INJECTED)
    monkeypatch.setattr(llm, "_refuse_sdk_override_env", lambda: None)
    script = Script(ok_response())
    client(cfg, script).proofread(request(cfg))
    headers = script.requests[0].headers
    assert headers["authorization"] == "Bearer INJECTED-TOKEN" and headers["x-injected"] == "yes"
    assert DUMMY_KEY not in " ".join(headers.values())  # the configured key was replaced


def test_empty_custom_headers_variable_is_accepted_and_injects_nothing(cfg, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "")
    script = Script(ok_response())
    client(cfg, script).proofread(request(cfg))
    headers = script.requests[0].headers
    assert headers["authorization"] == f"Bearer {DUMMY_KEY}" and "x-injected" not in headers
    assert "x-api-key" not in headers


def test_config_still_refuses_custom_headers(tmp_path) -> None:
    with pytest.raises(c.UsageError):
        c.build_config({"input": "a.xml", "lang": "en", "style_guide": "g.docx"},
                       {"ANTHROPIC_API_KEY": DUMMY_KEY, "ANTHROPIC_CUSTOM_HEADERS": INJECTED}, tmp_path)


@pytest.fixture
def sdk_logger_level():
    loggers = [logging.getLogger(n) for n in ("", "anthropic", "anthropic._base_client")]
    saved = [lg.level for lg in loggers]
    yield
    for lg, level in zip(loggers, saved):
        lg.setLevel(level)


@pytest.mark.parametrize("name", ["anthropic", "anthropic._base_client", ""])
def test_sdk_debug_logging_is_refused_before_any_request(cfg, sdk_logger_level, name: str) -> None:
    logging.getLogger(name).setLevel(logging.DEBUG)
    if name:
        logging.getLogger("").setLevel(logging.WARNING)
    script = Script(ok_response())
    with pytest.raises(llm.RunFatalError) as info:
        client(cfg, script).proofread(request(cfg))
    assert info.value.exit_code == 1 and script.requests == []
    assert "ANTHROPIC_LOG" in str(info.value)


def test_sdk_debug_logging_guard_is_needed(cfg, sdk_logger_level, monkeypatch, caplog) -> None:
    """Positive control: at DEBUG the SDK logs the request body (system prompt and user message)."""
    monkeypatch.setattr(llm, "_refuse_sdk_debug_logging", lambda: None)
    caplog.set_level(logging.DEBUG, logger="anthropic")
    client(cfg, Script(ok_response())).proofread(request(cfg))
    logged = caplog.text
    assert USER_TEXT in logged and SYSTEM_TEXT in logged
    assert DUMMY_KEY not in logged  # the key is not in the logged options (headers are added later)


def test_sdk_info_logging_is_allowed_and_content_free(cfg, sdk_logger_level, caplog) -> None:
    caplog.set_level(logging.INFO)
    result = client(cfg, Script(error_response(500), ok_response())).proofread(request(cfg))
    assert result.failure is None
    for secret in (DUMMY_KEY, USER_TEXT, SYSTEM_TEXT, BODY_SECRET):
        assert secret not in caplog.text


def test_anthropic_log_environment_variable_is_refused() -> None:
    code = "\n".join([
        "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'tests')",
        "import test_anthropic_client as t, pathlib",
        "from proofreader import config as c, llm",
        "cfg = c.build_config({'input': 'a.xml', 'lang': 'en', 'style_guide': 'g.docx'},"
        " {'ANTHROPIC_API_KEY': t.DUMMY_KEY}, pathlib.Path('.'))",
        "s = t.Script(t.ok_response())",
        "try:",
        "    t.client(cfg, s).proofread(t.request(cfg))",
        "except llm.RunFatalError:",
        "    print('refused', len(s.requests))",
    ])
    env = {"ANTHROPIC_LOG": "debug", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, env=env,
                         cwd=root, timeout=120)
    assert out.stdout.strip().splitlines()[-1] == "refused 0", out.stderr[-500:]
    assert "USER-PARAGRAPH-SENTINEL" not in out.stderr + out.stdout


def test_worst_case_is_six_requests_with_exact_accounting(cfg) -> None:
    """2 transport retries per logical call, 1 schema retry: at most (1 + 2) * (1 + 1) = 6 requests."""
    script = Script(error_response(503), error_response(429), ok_response(message("{bad")),
                    httpx2.ReadTimeout("slow"), httpx2.ConnectError("down"), ok_response(message("{bad")),
                    ok_response())  # a 7th response would be served if the client over-retried
    sleeps: list[float] = []
    result = client(cfg, script, sleeps=sleeps).proofread(request(cfg))
    assert result.failure is FailureClass.SCHEMA_INVALID and result.payload is None
    assert result.attempts == 6 == len(script.requests) and len(sleeps) == 4
    assert result.usage == Usage(20, 10)  # only the two messages carried usage


def test_sdk_client_has_no_internal_retries(cfg) -> None:
    proofreader = client(cfg, Script(ok_response()))
    assert proofreader._client.max_retries == 0

"""S5: GeminiProofreader - request shape, decoding, error classification, retries, recording.

Offline only: every request goes to an in-memory httpx2.MockTransport (the session-wide network guard in
conftest.py would fail any real connection). The API key is an invented dummy value and all text is invented.
"""

from __future__ import annotations

import copy
import json
import logging

import httpx2
import pytest

from proofreader import config as c
from proofreader import gemini as g
from proofreader import llm
from proofreader import prompt as pr
from proofreader.llm import FailureClass, ProviderFatalError, RecordingWriter, ReplayProofreader, Usage

MODEL = "gemini-2.5-flash-lite"
DUMMY_KEY = "dummy-gemini-key-not-real-0123456789"
ANTHROPIC_SENTINEL = "dummy-anthropic-key-must-not-be-used-42"
SYSTEM_TEXT = "SYSTEM-PROMPT-SENTINEL"
USER_TEXT = "USER-PARAGRAPH-SENTINEL"
BODY_SECRET = "ERROR-BODY-SENTINEL"
GOOD = {"findings": [{"original": "recieve", "before": "Our team will", "after": "the final report",
                      "word_start": 3, "word_end": 3, "correction": "receive", "type": "spelling",
                      "reason": "Misspelling"}]}
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


@pytest.fixture
def cfg(tmp_path) -> c.RunConfig:
    return c.build_config({"input": "a.xml", "lang": "en", "style_guide": "g.docx", "model": MODEL},
                          {"GEMINI_API_KEY": DUMMY_KEY, "ANTHROPIC_API_KEY": ANTHROPIC_SENTINEL,
                           "PYTHON_DOTENV_DISABLED": "1"}, tmp_path)


def body(text: str | None = '{"findings": []}', *, finish: str | None = "STOP", parts: list | None = None,
         usage: dict | None = None, response_id: str = "resp_ok", **extra) -> dict:
    parts = parts if parts is not None else ([{"text": text}] if text is not None else [])
    candidate: dict = {"content": {"role": "model", "parts": parts}, "index": 0}
    if finish is not None:
        candidate["finishReason"] = finish
    result = {"candidates": [candidate], "modelVersion": MODEL, "responseId": response_id,
              "usageMetadata": usage or {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}}
    result.update(extra)
    return result


def ok_response(payload: dict | None = None) -> httpx2.Response:
    return httpx2.Response(200, json=payload if payload is not None else body())


def error_response(status: int, headers: dict | None = None) -> httpx2.Response:
    return httpx2.Response(status, json={"error": {"code": status, "message": BODY_SECRET, "status": "X"}},
                           headers=headers or {})


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


def client(cfg, script: Script, **kwargs) -> g.GeminiProofreader:
    sleeps = kwargs.pop("sleeps", [])
    http = httpx2.Client(transport=httpx2.MockTransport(script), trust_env=False)
    return g.GeminiProofreader(cfg, http_client=http, sleep=sleeps.append, clock=kwargs.pop("clock", Clock()),
                               **kwargs)


def request(cfg, paragraph_id: int = 0, user: str = USER_TEXT):
    return llm.make_request(cfg, paragraph_id, system=SYSTEM_TEXT, user=user, schema=pr.output_schema())


# --------------------------------------------------------------------------------------------
# Construction and request shape
# --------------------------------------------------------------------------------------------


def test_client_conforms_to_the_protocol(cfg) -> None:
    assert isinstance(client(cfg, Script(ok_response())), llm.Proofreader)


def test_missing_gemini_key_is_fatal_before_any_request_even_with_an_anthropic_key(tmp_path) -> None:
    cfg = c.build_config({"input": "a.xml", "lang": "en", "style_guide": "g.docx", "model": MODEL},
                         {"ANTHROPIC_API_KEY": ANTHROPIC_SENTINEL, "PYTHON_DOTENV_DISABLED": "1"}, tmp_path)
    with pytest.raises(c.MissingApiKeyError) as info:
        g.GeminiProofreader(cfg)
    assert "GEMINI_API_KEY is not set" in str(info.value) and "ANTHROPIC" not in str(info.value)


def test_an_anthropic_configuration_is_refused(tmp_path) -> None:
    cfg = c.build_config({"input": "a.xml", "lang": "en", "style_guide": "g.docx"},
                         {"ANTHROPIC_API_KEY": ANTHROPIC_SENTINEL, "PYTHON_DOTENV_DISABLED": "1"}, tmp_path)
    with pytest.raises(ValueError, match="Gemini configuration"):
        g.GeminiProofreader(cfg)


def test_request_shape_uses_the_request_and_config_exactly(cfg) -> None:
    script = Script(ok_response())
    req = request(cfg)
    client(cfg, script).proofread(req)
    (sent,) = script.requests
    sent_body = json.loads(sent.content)
    assert sent.method == "POST" and str(sent.url) == URL
    assert set(sent_body) == {"systemInstruction", "contents", "generationConfig"}
    assert sent_body["systemInstruction"] == {"parts": [{"text": SYSTEM_TEXT}]}
    assert sent_body["contents"] == [{"role": "user", "parts": [{"text": USER_TEXT}]}]
    assert sent_body["generationConfig"] == {"responseMimeType": "application/json",
                                             "responseJsonSchema": g.gemini_schema(req.schema),
                                             "maxOutputTokens": 8000}
    assert "effort" not in sent.content.decode("utf-8") and "thinkingConfig" not in sent_body["generationConfig"]


def test_api_key_is_sent_only_in_the_goog_header(cfg) -> None:
    script = Script(ok_response())
    client(cfg, script).proofread(request(cfg))
    sent = script.requests[0]
    assert sent.headers["x-goog-api-key"] == DUMMY_KEY
    assert DUMMY_KEY not in sent.content.decode("utf-8") and DUMMY_KEY not in str(sent.url)
    assert "key=" not in str(sent.url) and sent.url.query == b""
    assert [name for name, value in sent.headers.items() if DUMMY_KEY in value] == ["x-goog-api-key"]
    assert "authorization" not in sent.headers and "x-api-key" not in sent.headers
    assert ANTHROPIC_SENTINEL not in " ".join(sent.headers.values()) + sent.content.decode("utf-8")


def test_default_http_client_ignores_the_environment_and_redirects(cfg) -> None:
    proofreader = g.GeminiProofreader(cfg)
    try:
        assert proofreader._http.trust_env is False and proofreader._http.follow_redirects is False
    finally:
        proofreader.close()


def test_mismatched_request_settings_are_refused(cfg, tmp_path) -> None:
    other = c.build_config({"input": "a.xml", "lang": "en", "style_guide": "g.docx", "model": "gemini-2.0-flash"},
                           {"GEMINI_API_KEY": DUMMY_KEY, "PYTHON_DOTENV_DISABLED": "1"}, tmp_path)
    with pytest.raises(ValueError, match="do not match"):
        client(cfg, Script(ok_response())).proofread(request(other))
    with pytest.raises(TypeError):
        client(cfg, Script(ok_response())).proofread("not a request")


def test_invalid_constructor_arguments(cfg) -> None:
    for bad in (-1, 6, True, 1.0):
        with pytest.raises(ValueError):
            g.GeminiProofreader(cfg, max_retries=bad, http_client=httpx2.Client(trust_env=False))
    with pytest.raises(ValueError):
        g.GeminiProofreader(cfg, recorder=object(), http_client=httpx2.Client(trust_env=False))


# --------------------------------------------------------------------------------------------
# Schema sent to Gemini: self-contained, otherwise unchanged; the canonical schema is untouched
# --------------------------------------------------------------------------------------------


def test_gemini_schema_inlines_the_definitions_and_keeps_every_keyword() -> None:
    canonical = pr.output_schema()
    before = copy.deepcopy(canonical)
    sent = g.gemini_schema(canonical)
    assert canonical == before  # not mutated
    text = json.dumps(sent)
    assert "$ref" not in text and "$defs" not in text
    finding = canonical["$defs"]["Finding"]
    assert sent["properties"]["findings"]["items"] == finding
    assert {k: v for k, v in sent.items() if k != "properties"} == \
        {k: v for k, v in canonical.items() if k not in ("properties", "$defs")}
    assert sent["additionalProperties"] is False and sent["required"] == ["findings"]
    assert sent["properties"]["findings"]["items"]["additionalProperties"] is False


@pytest.mark.parametrize("schema", [
    {"$ref": "#/$defs/Missing", "$defs": {}},
    {"$ref": "https://example.invalid/schema"},
    {"$ref": "#/$defs/A", "extra": 1, "$defs": {"A": {}}},
    {"$defs": []},
])
def test_gemini_schema_refuses_unsupported_references(schema) -> None:
    with pytest.raises(ValueError):
        g.gemini_schema(schema)


def test_gemini_schema_refuses_reference_cycles() -> None:
    with pytest.raises(ValueError):
        g.gemini_schema({"$ref": "#/$defs/A", "$defs": {"A": {"items": {"$ref": "#/$defs/A"}}}})


# --------------------------------------------------------------------------------------------
# Successful responses
# --------------------------------------------------------------------------------------------


def test_clean_response_and_accounting(cfg) -> None:
    usage = {"promptTokenCount": 11, "candidatesTokenCount": 7, "thoughtsTokenCount": 2,
             "cachedContentTokenCount": 3, "totalTokenCount": 20}
    result = client(cfg, Script(ok_response(body(json.dumps(GOOD), usage=usage, response_id="resp_A1")))) \
        .proofread(request(cfg))
    assert result.failure is None and result.payload == GOOD and result.attempts == 1
    assert result.usage == Usage(11, 9, 0, 3)
    assert result.request_id == "resp_A1" and result.stop_reason == "stop"
    assert llm.CallRecord.from_result(0, result).status is llm.CallStatus.OK


def test_missing_or_malformed_usage_counts_as_zero(cfg) -> None:
    for usage in ({}, {"promptTokenCount": -1, "candidatesTokenCount": "5", "thoughtsTokenCount": True}):
        payload = body()
        payload["usageMetadata"] = usage
        assert client(cfg, Script(ok_response(payload))).proofread(request(cfg)).usage == Usage()


def test_unsafe_response_id_is_dropped(cfg) -> None:
    result = client(cfg, Script(ok_response(body(response_id="bad id/../x")))).proofread(request(cfg))
    assert result.failure is None and result.request_id is None


def test_thought_parts_are_ignored(cfg) -> None:
    parts = [{"text": "thinking...", "thought": True}, {"text": '{"findings": []}'}]
    assert client(cfg, Script(ok_response(body(parts=parts)))).proofread(request(cfg)).failure is None


def test_per_finding_problems_are_not_schema_retried(cfg) -> None:
    bad_finding = {"findings": [{"original": "x"}]}  # the resolver rejects it per finding, not the envelope
    script = Script(ok_response(body(json.dumps(bad_finding))))
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is None and result.payload == bad_finding and len(script.requests) == 1


# --------------------------------------------------------------------------------------------
# Unusable output: one schema retry, then SCHEMA_INVALID (never repaired)
# --------------------------------------------------------------------------------------------

UNUSABLE = {
    "invalid json": body("{not json"),
    "duplicate keys": body('{"findings": [], "findings": []}'),
    "nan": body('{"findings": [NaN]}'),
    "trailing data": body('{"findings": []} {}'),
    "not an object": body('["findings"]'),
    "missing findings": body('{"other": []}'),
    "extra top-level key": body('{"findings": [], "note": "x"}'),
    "empty text": body(""),
    "no parts": body(parts=[]),
    "two text parts": body(parts=[{"text": '{"findings": []}'}, {"text": '{"findings": []}'}]),
    "function call part": body(parts=[{"functionCall": {"name": "x", "args": {}}}]),
    "function call beside valid text": body(parts=[{"text": '{"findings": []}'},
                                                   {"functionCall": {"name": "x", "args": {}}}]),
    "text part with an extra key": body(parts=[{"text": '{"findings": []}', "inlineData": {}}]),
    "non-string text": body(parts=[{"text": 5}]),
    "only thought parts": body(parts=[{"text": '{"findings": []}', "thought": True}]),
    "unexpected finish reason": body(finish="OTHER"),
    "malformed response finish": body(finish="MALFORMED_RESPONSE"),
    "missing finish reason": body(finish=None),
    "two candidates": {**body(), "candidates": body()["candidates"] * 2},
    "no candidates and no block": {"candidates": [], "usageMetadata": {}},
    "candidates not a list": {"candidates": {"x": 1}},
    "body is a list": [1, 2],
}


@pytest.mark.parametrize("name", sorted(UNUSABLE))
def test_unusable_output_is_retried_once_then_reported(cfg, name: str) -> None:
    script = Script(ok_response(UNUSABLE[name]))
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is FailureClass.SCHEMA_INVALID
    assert result.attempts == 2 and len(script.requests) == 2  # exactly one schema retry


@pytest.mark.parametrize("content", [b"not json at all", b"", b'{"a": 1'])
def test_unparseable_http_body_is_retried_once_then_reported(cfg, content: bytes) -> None:
    script = Script(httpx2.Response(200, content=content))
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is FailureClass.SCHEMA_INVALID and result.attempts == 2 == len(script.requests)


def test_envelope_invalid_payload_is_kept_for_tracing(cfg) -> None:
    result = client(cfg, Script(ok_response(body('{"other": 1}')))).proofread(request(cfg))
    assert result.failure is FailureClass.SCHEMA_INVALID and result.payload == {"other": 1}


def test_schema_retry_success_sums_usage(cfg) -> None:
    script = Script(ok_response(body("{bad")), ok_response(body(json.dumps(GOOD), response_id="resp_second")))
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is None and result.payload == GOOD and result.attempts == 2
    assert result.usage == Usage(20, 10) and result.request_id == "resp_second"


# --------------------------------------------------------------------------------------------
# Final outcomes: truncation, safety, blocked prompts (never retried)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("finish, failure", [
    ("MAX_TOKENS", FailureClass.MAX_TOKENS), ("SAFETY", FailureClass.REFUSAL), ("RECITATION", FailureClass.REFUSAL),
    ("BLOCKLIST", FailureClass.REFUSAL), ("PROHIBITED_CONTENT", FailureClass.REFUSAL),
    ("SPII", FailureClass.REFUSAL),
])
def test_truncation_and_safety_stops_are_final(cfg, finish: str, failure: FailureClass) -> None:
    script = Script(ok_response(body(json.dumps(GOOD), finish=finish)))
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is failure and result.payload is None
    assert result.attempts == 1 == len(script.requests) and result.stop_reason == finish.lower()
    assert result.usage == Usage(10, 5)


def test_blocked_prompt_is_a_final_refusal(cfg) -> None:
    payload = {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 4}}
    script = Script(ok_response(payload))
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is FailureClass.REFUSAL and result.stop_reason == "blocked"
    assert result.attempts == 1 == len(script.requests) and result.usage == Usage(4, 0)


# --------------------------------------------------------------------------------------------
# Transport failures: bounded retries with backoff
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item, failure",
    [
        (error_response(429), FailureClass.RATE_LIMITED), (error_response(500), FailureClass.SERVER_ERROR),
        (error_response(503), FailureClass.SERVER_ERROR), (error_response(504), FailureClass.SERVER_ERROR),
        (error_response(408), FailureClass.TIMEOUT), (error_response(409), FailureClass.SERVER_ERROR),
        (httpx2.ReadTimeout("slow"), FailureClass.TIMEOUT), (httpx2.ConnectError("down"), FailureClass.CONNECTION),
    ],
    ids=["429", "500", "503", "504", "408", "409", "timeout", "connection"],
)
def test_transient_failures_are_retried_then_reported(cfg, item, failure: FailureClass) -> None:
    script = Script(item)
    sleeps: list[float] = []
    result = client(cfg, script, sleeps=sleeps).proofread(request(cfg))
    assert result.failure is failure and result.payload is None
    assert result.attempts == 3 == len(script.requests)  # 1 + MAX_TRANSPORT_RETRIES
    assert sleeps == [0.5, 1.0]
    assert llm.CallRecord.from_result(0, result).status is llm.CallStatus.PASSTHROUGH


def test_transient_then_success_counts_every_request(cfg) -> None:
    script = Script(error_response(429), httpx2.ConnectError("blip"), ok_response())
    sleeps: list[float] = []
    result = client(cfg, script, sleeps=sleeps).proofread(request(cfg))
    assert result.failure is None and result.attempts == 3 and len(sleeps) == 2


def test_retry_after_is_honoured_and_capped(cfg) -> None:
    for headers, expected in (({"retry-after": "3"}, 3.0), ({"retry-after": "9999"}, llm.RETRY_AFTER_CAP_S),
                              ({"retry-after": "soon"}, 0.5)):
        sleeps: list[float] = []
        client(cfg, Script(error_response(429, headers)), sleeps=sleeps).proofread(request(cfg))
        assert sleeps[0] == expected


def test_retry_limit_is_respected(cfg) -> None:
    for retries in (0, 1, 3):
        script = Script(error_response(500))
        result = client(cfg, script, max_retries=retries).proofread(request(cfg))
        assert result.attempts == retries + 1 == len(script.requests)


def test_transport_and_schema_retries_combine_with_exact_attempts(cfg) -> None:
    script = Script(error_response(503), ok_response(body("{bad")), error_response(429), ok_response())
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is None and result.attempts == 4 == len(script.requests)


def test_worst_case_is_six_requests(cfg) -> None:
    script = Script(error_response(503), error_response(503), ok_response(body("{bad")),
                    error_response(503), error_response(503), ok_response(body("{bad")))
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is FailureClass.SCHEMA_INVALID and result.attempts == 6 == len(script.requests)


# --------------------------------------------------------------------------------------------
# Paragraph-level 413 and run-fatal statuses
# --------------------------------------------------------------------------------------------


def test_request_too_large_is_paragraph_level_and_not_retried(cfg) -> None:
    script = Script(error_response(413))
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is FailureClass.REQUEST_TOO_LARGE and result.attempts == 1 == len(script.requests)


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422, 412, 431, 499, 301, 302, 304])
def test_permanent_statuses_are_run_fatal_never_retried_and_redacted(cfg, status: int) -> None:
    script = Script(error_response(status, {"location": "https://elsewhere.invalid/"}))
    proofreader = client(cfg, script)
    with pytest.raises(ProviderFatalError) as info:
        proofreader.proofread(request(cfg))
    assert info.value.exit_code == 1 and isinstance(info.value, llm.RunFatalError)
    assert len(script.requests) == 1
    text = str(info.value) + repr(info.value)
    for secret in (DUMMY_KEY, BODY_SECRET, USER_TEXT, SYSTEM_TEXT):
        assert secret not in text
    assert f"HTTP {status}" in text and "Gemini" in text


def test_other_client_errors_are_run_fatal_and_redacted(cfg) -> None:
    script = Script(httpx2.TooManyRedirects(f"redirect loop {DUMMY_KEY}"))
    with pytest.raises(ProviderFatalError) as info:
        client(cfg, script).proofread(request(cfg))
    assert "TooManyRedirects" in str(info.value) and DUMMY_KEY not in str(info.value)
    assert info.value.__cause__ is None and len(script.requests) == 1


def test_non_http_exceptions_and_interrupts_propagate(cfg, monkeypatch) -> None:
    proofreader = client(cfg, Script(ok_response()))

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(proofreader._http, "post", interrupted)
    with pytest.raises(KeyboardInterrupt):
        proofreader.proofread(request(cfg))


# --------------------------------------------------------------------------------------------
# Redaction, logging, recording
# --------------------------------------------------------------------------------------------


def test_repr_and_logs_never_expose_the_key_or_text(cfg, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    script = Script(error_response(429), ok_response(body(json.dumps(GOOD))))
    proofreader = client(cfg, script)
    result = proofreader.proofread(request(cfg))
    rendered = repr(proofreader) + repr(result) + caplog.text
    for secret in (DUMMY_KEY, USER_TEXT, SYSTEM_TEXT, "recieve"):
        assert secret not in rendered
    assert repr(proofreader) == f"GeminiProofreader(model='{MODEL}', host='generativelanguage.googleapis.com', " \
                                "max_retries=2)"
    assert [r for r in caplog.records if r.name.startswith("proofreader")] == []


def test_recording_round_trip_through_replay(cfg, tmp_path) -> None:
    writer = RecordingWriter()
    script = Script(ok_response(body(json.dumps(GOOD))), error_response(429), error_response(429),
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
    for secret in (DUMMY_KEY, SYSTEM_TEXT, "paragraph one", "paragraph two", MODEL, "x-goog-api-key",
                   "generationConfig", "responseJsonSchema"):
        assert secret not in text


def test_identical_requests_are_recorded_once_and_fatal_outcomes_never(cfg) -> None:
    writer = RecordingWriter()
    proofreader = client(cfg, Script(ok_response()), recorder=writer)
    proofreader.proofread(request(cfg))
    proofreader.proofread(request(cfg))
    assert len(writer) == 1
    fatal_writer = RecordingWriter()
    with pytest.raises(ProviderFatalError):
        client(cfg, Script(error_response(400)), recorder=fatal_writer).proofread(request(cfg))
    assert len(fatal_writer) == 0


def test_recordings_refuse_gemini_key_shapes(tmp_path) -> None:
    writer = RecordingWriter()
    leaked = "AIza" + "B" * 35  # invented key-shaped text
    fingerprint = "0" * 64
    writer.add(fingerprint, llm.ProofreadResult({"findings": [{"reason": leaked}]}, None, Usage(), 0.0, 1, None,
                                                "stop"))
    with pytest.raises(llm.RecordingError):
        writer.write(tmp_path / "x.recording.json")


def test_gemini_and_anthropic_requests_have_different_fingerprints(cfg, tmp_path) -> None:
    anthropic_cfg = c.build_config({"input": "a.xml", "lang": "en", "style_guide": "g.docx"},
                                   {"PYTHON_DOTENV_DISABLED": "1"}, tmp_path)
    assert request(cfg).fingerprint != request(anthropic_cfg).fingerprint


# --------------------------------------------------------------------------------------------
# thoughtSignature beside the text (observed live from gemini-3.5-flash-lite on 2026-11-23; values invented)
# --------------------------------------------------------------------------------------------

SIGNATURE = "SIGNATURE-SENTINEL-opaque+base64/value=="


def signed_response(text: str = '{"findings": []}', **part_extra) -> dict:
    """The response shape observed live: one text part carrying thoughtSignature, role "model", index 0,
    usage with prompt token details and a service tier."""
    return {"candidates": [{"content": {"parts": [{"text": text, "thoughtSignature": SIGNATURE, **part_extra}],
                                        "role": "model"}, "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 989, "candidatesTokenCount": 131, "totalTokenCount": 1120,
                              "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 989}],
                              "serviceTier": "standard"},
            "modelVersion": "gemini-3.5-flash-lite", "responseId": "resp_signed_1"}


def test_text_part_with_thought_signature_is_accepted(cfg) -> None:
    script = Script(ok_response(signed_response(json.dumps(GOOD))))
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is None and result.payload == GOOD
    assert result.attempts == 1 == len(script.requests)  # no schema retry
    assert result.usage == Usage(989, 131) and result.request_id == "resp_signed_1" and result.stop_reason == "stop"
    assert SIGNATURE not in repr(result)


@pytest.mark.parametrize("parts", [
    [{"text": '{"findings": []}'}],
    [{"text": '{"findings": []}', "thoughtSignature": SIGNATURE}],
    [{"text": "t", "thought": True, "thoughtSignature": SIGNATURE}, {"text": '{"findings": []}'}],
    [{"text": "t", "thought": True}, {"text": '{"findings": []}', "thoughtSignature": SIGNATURE}],
], ids=["plain text", "text with signature", "signed thought then text", "thought then signed text"])
def test_plain_and_signed_text_parts_are_extracted_identically(cfg, parts) -> None:
    assert g._single_text(body(parts=parts)) == '{"findings": []}'
    assert client(cfg, Script(ok_response(body(parts=parts)))).proofread(request(cfg)).failure is None


UNUSABLE_SIGNED = {
    "signature without text": body(parts=[{"thoughtSignature": SIGNATURE}]),
    "signature with null text": body(parts=[{"text": None, "thoughtSignature": SIGNATURE}]),
    "signature with non-string text": body(parts=[{"text": 5, "thoughtSignature": SIGNATURE}]),
    "non-string signature": body(parts=[{"text": '{"findings": []}', "thoughtSignature": 7}]),
    "null signature": body(parts=[{"text": '{"findings": []}', "thoughtSignature": None}]),
    "signed text with an extra field": signed_response(inlineData={}),
    "signed text with thought false": signed_response(thought=False),
    "signed text with functionCall": signed_response(functionCall={"name": "x", "args": {}}),
    "signed text beside a function call part": body(parts=[{"text": '{"findings": []}', "thoughtSignature": SIGNATURE},
                                                           {"functionCall": {"name": "x", "args": {}}}]),
    "two signed text parts": body(parts=[{"text": '{"findings": []}', "thoughtSignature": SIGNATURE}] * 2),
    "signed text that is not JSON": signed_response("not json"),
    "signed text with an invalid envelope": signed_response('{"findings": [], "extra": 1}'),
    "signed text that is empty": signed_response(""),
    "parts is not a list": {"candidates": [{"content": {"parts": {"text": "x"}}, "finishReason": "STOP"}]},
    "content is not an object": {"candidates": [{"content": "x", "finishReason": "STOP"}]},
    "candidate is not an object": {"candidates": ["x"]},
    "part is not an object": {"candidates": [{"content": {"parts": ["x"]}, "finishReason": "STOP"}]},
}


@pytest.mark.parametrize("name", sorted(UNUSABLE_SIGNED))
def test_malformed_signed_or_structural_responses_are_retried_once_then_reported(cfg, name: str) -> None:
    script = Script(ok_response(UNUSABLE_SIGNED[name]))
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is FailureClass.SCHEMA_INVALID
    assert result.attempts == 2 and len(script.requests) == 2  # exactly one schema retry, never more


def test_signature_never_reaches_a_recording(cfg, tmp_path) -> None:
    writer = RecordingWriter()
    client(cfg, Script(ok_response(signed_response(json.dumps(GOOD)))), recorder=writer).proofread(request(cfg))
    target = tmp_path / "signed.recording.json"
    writer.write(target)
    text = target.read_text(encoding="utf-8")
    assert SIGNATURE not in text and "thoughtSignature" not in text


@pytest.mark.parametrize("part", [
    {"text": '{"findings": []}', "note": "x"},
    {"text": '{"findings": []}', "thoughtSignature": SIGNATURE, "note": "x"},
    {"text": '{"findings": []}', "thought_signature": SIGNATURE},  # wrong spelling is an unknown field
    {"text": '{"findings": []}', "thoughtSignature": SIGNATURE, "mimeType": "application/json"},
], ids=["string extra field", "signed with string extra field", "misspelled signature key", "signed with mime type"])
def test_unexpected_string_valued_fields_are_rejected(cfg, part) -> None:
    """Extra fields are rejected by their names, not only by their value types (mutation M30)."""
    assert g._single_text(body(parts=[part])) is None
    script = Script(ok_response(body(parts=[part])))
    result = client(cfg, script).proofread(request(cfg))
    assert result.failure is FailureClass.SCHEMA_INVALID and len(script.requests) == 2

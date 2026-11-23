"""Gemini provider (S5): ``GeminiProofreader`` implements the same ``Proofreader`` protocol as
``AnthropicProofreader`` over the Gemini REST API (``models/{model}:generateContent``).

Selected when the model id starts with ``gemini-`` (``config.provider_for``). Everything after the HTTP
response is shared with the Anthropic client: strict JSON decoding (``llm.decode_response``), the resolver
envelope check, the failure classes, bounded backoff, the one schema retry and the recorder. The S3 resolver,
quotation protection and the verifier stay authoritative - Gemini's schema-constrained decoding is a help,
never a guarantee.

Transport: ``httpx2`` (already the Anthropic SDK's HTTP library), environment proxies / CA overrides / netrc
ignored (``trust_env=False``), no redirects, no internal retries. The key is sent only in the
``x-goog-api-key`` header - never in the URL, body, logs, reprs, errors or recordings. Nothing is logged.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from . import llm
from .config import PROVIDER_GEMINI, RunConfig
from .llm import (MAX_SCHEMA_RETRIES, MAX_TRANSPORT_RETRIES, FailureClass, ProofreadRequest, ProofreadResult,
                  ProviderFatalError, RecordingWriter, _envelope_invalid, _is_int, _retry_delay, _safe_request_id,
                  _safe_stop_reason, _Tally, decode_response)

API_VERSION_PATH = "/v1beta/models/{model}:generateContent"
_OK_FINISH = frozenset({"stop"})
# finishReason (lowercased) values that are final, never retried. Every other value is unusable output
# (one schema retry). "blocked" is our name for a prompt rejected through promptFeedback.blockReason.
_FINAL_FINISH = {
    "max_tokens": FailureClass.MAX_TOKENS,
    "safety": FailureClass.REFUSAL,
    "recitation": FailureClass.REFUSAL,
    "blocklist": FailureClass.REFUSAL,
    "prohibited_content": FailureClass.REFUSAL,
    "spii": FailureClass.REFUSAL,
    "blocked": FailureClass.REFUSAL,
}
_RUN_FATAL_HINT = "the run was stopped"


def gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """A self-contained copy of the output schema for ``responseJsonSchema``: every local ``#/$defs/<name>``
    reference is replaced by (a copy of) its definition and ``$defs`` is dropped, because the Gemini
    documentation does not state ``$defs`` support. Keywords are otherwise unchanged. The canonical schema
    (and therefore the request fingerprint) is not modified. Raises ValueError on any other reference form."""
    defs = schema.get("$defs", {})
    if not isinstance(defs, dict):
        raise ValueError("schema $defs must be an object")

    def resolve(node: Any, depth: int) -> Any:
        if depth > 32:
            raise ValueError("schema references are nested too deeply")
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                name = ref[len("#/$defs/"):] if isinstance(ref, str) and ref.startswith("#/$defs/") else None
                if name is None or name not in defs or len(node) != 1:
                    raise ValueError("unsupported schema reference")
                return resolve(defs[name], depth + 1)
            return {key: resolve(value, depth + 1) for key, value in node.items() if key != "$defs"}
        if isinstance(node, list):
            return [resolve(value, depth + 1) for value in node]
        return node

    return resolve(schema, 0)


def _usage_of(body: dict) -> llm.Usage:
    meta = body.get("usageMetadata")
    meta = meta if isinstance(meta, dict) else {}

    def count(name: str) -> int:
        value = meta.get(name)
        return value if _is_int(value) and value >= 0 else 0

    # Thinking tokens are billed as output; Gemini has no separate cache-write count.
    return llm.Usage(count("promptTokenCount"), count("candidatesTokenCount") + count("thoughtsTokenCount"), 0,
                     count("cachedContentTokenCount"))


def _finish_reason(body: dict) -> str | None:
    """Lowercased finishReason of the single candidate; "blocked" for a prompt rejected without candidates;
    None when the shape is anything else."""
    candidates = body.get("candidates")
    if not candidates:
        feedback = body.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason"):
            return "blocked"
        return None
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
        return None
    reason = candidates[0].get("finishReason")
    return _safe_stop_reason(reason.lower()) if isinstance(reason, str) else None


_TEXT_PART_KEYS = (frozenset({"text"}), frozenset({"text", "thoughtSignature"}))


def _is_text_part(part: dict) -> bool:
    """A plain text part: exactly ``text``, optionally with ``thoughtSignature`` (opaque, added by newer models
    and ignored here; observed live 2026-11-23); both values must be strings. Any other field is rejected."""
    return (frozenset(part) in _TEXT_PART_KEYS
            and all(isinstance(value, str) for value in part.values()))


def _single_text(body: dict) -> str | None:
    """The one text part of the single candidate, or None for anything else (no candidate, several text
    parts, or any non-text part). Thought parts (``"thought": true``) are ignored, like thinking blocks; a
    ``thoughtSignature`` beside the text is ignored, any other extra field is not."""
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
        return None
    content = candidates[0].get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list) or not parts:
        return None
    texts = []
    for part in parts:
        if not isinstance(part, dict):
            return None
        if part.get("thought") is True:
            continue
        if not _is_text_part(part):
            return None
        texts.append(part["text"])
    return texts[0] if len(texts) == 1 else None


class GeminiProofreader:
    """Live Gemini client (S5). Same contract as ``AnthropicProofreader``: transient failures are retried here
    (bounded), unusable output gets one schema retry, paragraph-level failures are returned and run-fatal ones
    raised as ``ProviderFatalError``."""

    def __init__(self, config: RunConfig, *, http_client: object = None, recorder: RecordingWriter | None = None,
                 max_retries: int = MAX_TRANSPORT_RETRIES, sleep=None, clock=None) -> None:
        import time

        import httpx2

        if config.provider != PROVIDER_GEMINI:
            raise ValueError("GeminiProofreader needs a Gemini configuration (model id 'gemini-...')")
        if not _is_int(max_retries) or not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be an int from 0 to 5")
        if recorder is not None and not isinstance(recorder, RecordingWriter):
            raise ValueError("recorder must be a RecordingWriter or None")
        self._key = config.require_api_key()  # MissingApiKeyError (exit 1) before any request
        self._httpx2 = httpx2
        self._model, self._effort, self._max_tokens = config.model, config.effort, config.max_tokens
        self._timeout = config.timeout_seconds
        self._host = config.base_url.host
        self._url = config.base_url.url + API_VERSION_PATH.format(model=config.model)
        self._max_retries = max_retries
        self._sleep = sleep if sleep is not None else time.sleep
        self._clock = clock if clock is not None else time.monotonic
        if http_client is None:  # no environment proxies / CA overrides / netrc, no redirects
            http_client = httpx2.Client(trust_env=False, timeout=config.timeout_seconds, follow_redirects=False)
        self._http = http_client
        self._recorder = recorder
        self._recorded: set[str] = set()
        self._record_lock = threading.Lock()

    def __repr__(self) -> str:
        return f"GeminiProofreader(model={self._model!r}, host={self._host!r}, max_retries={self._max_retries})"

    def close(self) -> None:
        self._http.close()

    # -- protocol ---------------------------------------------------------------------------

    def proofread(self, request: ProofreadRequest) -> ProofreadResult:
        if not isinstance(request, ProofreadRequest):
            raise TypeError("request must be a ProofreadRequest")
        if (request.model, request.effort, request.max_tokens) != (self._model, self._effort, self._max_tokens):
            raise ValueError("request settings do not match this client's configuration")
        start = self._clock()
        tally = _Tally()
        payload: object = None
        request_id: str | None = None
        stop_reason: str | None = None
        for _schema_round in range(MAX_SCHEMA_RETRIES + 1):
            kind, value, request_id = self._call(request, tally)
            if kind == "failure":
                return self._finish(request, None, value, tally, start, request_id, None)
            payload, stop_reason = None, None
            if kind == "message":
                tally.usage = tally.usage + _usage_of(value)
                stop_reason = _finish_reason(value)
                if stop_reason in _FINAL_FINISH:  # refusal / block / truncation: final, never retried
                    return self._finish(request, None, _FINAL_FINISH[stop_reason], tally, start, request_id,
                                        stop_reason)
                text = _single_text(value) if stop_reason in _OK_FINISH else None
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

    def _body(self, request: ProofreadRequest) -> dict[str, Any]:
        return {
            "systemInstruction": {"parts": [{"text": request.system}]},
            "contents": [{"role": "user", "parts": [{"text": request.user}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": gemini_schema(request.schema),
                "maxOutputTokens": request.max_tokens,
            },
        }

    def _send(self, request: ProofreadRequest):
        return self._http.post(self._url, json=self._body(request), timeout=self._timeout,
                               headers={"x-goog-api-key": self._key.reveal()})

    def _call(self, request: ProofreadRequest, tally: _Tally) -> tuple[str, object, str | None]:
        """One logical call with bounded transport retries. Returns ("message", body, request_id),
        ("invalid", None, None) for an unparseable 2xx body, or ("failure", FailureClass, None). Raises
        ProviderFatalError for run-fatal statuses and unexpected client errors."""
        httpx2 = self._httpx2
        failure = FailureClass.CONNECTION
        for retry_index in range(self._max_retries + 1):
            tally.attempts += 1
            headers = None
            try:
                response = self._send(request)
            except httpx2.TimeoutException:
                failure = FailureClass.TIMEOUT
            except httpx2.TransportError:
                failure = FailureClass.CONNECTION
            except httpx2.HTTPError as exc:
                raise ProviderFatalError(f"Gemini client error ({type(exc).__name__}); {_RUN_FATAL_HINT}") from None
            else:
                status = response.status_code
                if 200 <= status < 300:
                    try:
                        body = json.loads(response.content)
                    except ValueError:
                        return "invalid", None, None
                    if not isinstance(body, dict):
                        return "invalid", None, None
                    return "message", body, _safe_request_id(body.get("responseId"))
                headers = response.headers
                if status == 413:
                    return "failure", FailureClass.REQUEST_TOO_LARGE, None
                if status >= 500:
                    failure = FailureClass.SERVER_ERROR
                elif status in llm._TRANSIENT_FAILURES:
                    failure = llm._TRANSIENT_FAILURES[status]
                else:  # 400/401/403/404 and every other non-2xx: run-fatal; the body is never echoed
                    raise ProviderFatalError(f"Gemini API rejected the request (HTTP {status}); {_RUN_FATAL_HINT}")
            if retry_index < self._max_retries:
                self._sleep(_retry_delay(retry_index, headers))
        return "failure", failure, None

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

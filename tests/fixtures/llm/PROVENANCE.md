# Replay fixture provenance (S4.8b, AC-30)

| Item | Value |
|---|---|
| Fixture | `golden-e2e-1.recording.json` |
| SHA-256 | `e16f5dd0359e0b342a73871f5680447c7bc07f4e6ea4c2e3e320c80680d3096b` |
| Size | 4130 bytes, 8 entries (paragraphs 0, 1, 2, 3, 4, 5, 6 and 8 of the input) |
| Input document | `tests/fixtures/xml/golden-e2e-1-input.xml` (invented; S4.8a) |
| Expected replay output | `tests/fixtures/xml/golden-e2e-1-expected.xml` (hand-written; S4.8a) |
| Style guide | invented, built by `tests/docx_factory.py` ("House rules" / "Write per cent in running text, never the % sign.") |
| Language | `en` (full style guide selected) |
| Recording format | `proofreader-recording` version 1; `fingerprint_version` `proofreader-request-v1`; `prompt_version` `s4.3-v1` |
| Request settings | production values, unchanged: model `claude-opus-5-5`, effort `low`, `max_tokens` 8000, packaged system prompt and output schema |
| Generator | `generate_replay_fixtures.py` (this directory) |

## How it was made

`generate_replay_fixtures.py` runs the real command line (`proofreader.cli.main ... --record`) with the real
`AnthropicProofreader` and the real Anthropic SDK. Every HTTP request is answered **in memory** by an
`httpx2.MockTransport`; no request leaves the process. The responses are **hand-written, invented findings**
(`RESPONSES` in the generator), not output from any model. Request ids (`req_fixture_<paragraph>`) and token counts
(input 1000 + paragraph id, output 50 + paragraph id) are invented too. The client clock is fixed, so every recorded
latency is 0, and entries are sorted by fingerprint, so the file is byte-deterministic.

The API key used during generation is a dummy test value. The recording writer stores only whitelisted result
fields (no request, prompt, style guide, configuration, credential or header) and refuses credential-like text.

## What it contains and what it does not

- Contains: the request fingerprint of each paragraph's request, the invented parsed response, status, attempts,
  latency 0, invented usage and request ids. The responses quote fragments of the invented input document.
- Does not contain: any API key or credential, real model output, assessment material, local paths or personal
  data.

## Regenerating

    python -I -B tests/fixtures/llm/generate_replay_fixtures.py            # rewrite
    python -I -B tests/fixtures/llm/generate_replay_fixtures.py --check    # verify only

Any change to the system prompt, the schema, the request settings, the prompt or fingerprint version, the user
message format or the invented inputs changes the fingerprints. `tests/test_replay_fixture.py` then fails until the
fixture is regenerated deliberately and this file's SHA-256 is updated after review.

## Known limit

Replay matches requests by fingerprint; it does not authenticate the recorded responses. An edited response would
replay silently. For this committed fixture, the byte-for-byte regeneration test detects any edit.

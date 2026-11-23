# XML Proofreader

A command-line tool that proofreads the text of every `<p>` element in an XML document with a generative AI model
and marks each problem in place with an `<error>` tag, leaving every other byte of the document unchanged.

```xml
<p>Revenue grew 5% last year.</p>
<!-- becomes -->
<p>Revenue grew <error type="styleguide" correction="5 per cent" reason="…">5%</error> last year.</p>
```

## Status

- This repository is a clean public snapshot of a project developed in stages S0–S5 (repository foundation, XML
  mapping, annotation and verification, quotation-aware finding resolution, model clients and CLI, Gemini
  provider). The internal development history is not included.
- This snapshot was validated on a fresh installation (see [Testing](#testing)).
- Model output has been validated live only on invented text (see [Known limitations](#known-limitations)).

## Key capabilities

- **GenAI proofreading** of grammar, spelling, punctuation, capitalization, clarity and style-guide rules, guided by
  a style guide supplied as `.docx`. Providers: Anthropic (`claude-opus-5-5`, the default) or Google Gemini
  (any `gemini-…` model id).
- **Byte-exact preservation**: elements, attributes, namespaces, comments, processing instructions, CDATA, entity
  references, encoding and line endings are kept; tags are inserted without re-serialising the document.
- **Independent verification**: removing the inserted `<error>` tags must restore the original bytes exactly, so
  the text, its length and all whitespace are unchanged; a paragraph that fails is rolled back.
- **Nested paragraphs**: each `<p>` is proofread for its own text; nested `<p>` elements are handled separately.
- **Safe placement**: the model quotes the original text with surrounding words, and a finding is applied only
  where it matches exactly and unambiguously. Quoted material is never edited. Paragraphs without findings are
  left unchanged.
- **Language selection** with BCP 47 tags (`en`, `en-US`, `en-GB`, `fr`, …).
- **Runtime and memory metrics**, logging without document text or credentials, and explicit exit codes.
- **Record and replay** of model responses for reproducible, offline runs.

## Architecture

```
XML bytes ─► encoding check ─► byte-level mapping (Expat, cross-checked with ElementTree)
          ─► per-paragraph prompt (style guide + paragraph words) ─► model (bounded concurrency, retries)
          ─► strict JSON decoding ─► finding resolution (anchors, quotation protection)
          ─► tag insertion ─► independent verifier / rollback ─► atomic write of INPUT.corrected.xml
```

| Module (`src/proofreader/`) | Responsibility |
|---|---|
| `encoding.py`, `xml_source.py`, `tokens.py` | input checks, byte-exact XML mapping, word grid |
| `style_guide.py`, `prompt.py`, `prompts/` | `.docx` style-guide extraction, locale selection, prompt and JSON schema |
| `llm.py`, `gemini.py` | `Proofreader` protocol; Anthropic and Gemini clients; fake and replay clients; recordings |
| `quotes.py`, `resolver.py`, `models.py` | quotation analysis, finding validation and placement |
| `annotation.py`, `verify.py` | `<error>` rendering, independent verification, per-paragraph rollback |
| `pipeline.py`, `cli.py`, `metrics.py`, `config.py` | orchestration, command line, output, logging, metrics, configuration |

## Installation

Requires Python 3.11 or later (developed and tested with Python 3.11.9 on Windows).

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"   # equivalent: .venv\Scripts\python -m pip install -r requirements.txt
```

Runtime dependencies: `anthropic`, `pydantic`, `python-dotenv` (declared in `pyproject.toml`); development adds
`pytest`, `hypothesis` and `httpx2`.

## Configuration

Settings come from command-line flags, then the process environment, then a `.env` file in the current directory,
then defaults. Copy `.env.example` to `.env` and fill in the key for the provider you use.

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | key for Anthropic models (the default provider) |
| `GEMINI_API_KEY` | key for Gemini models |
| `PROOFREADER_MODEL` | model id; default `claude-opus-5-5`; a lowercase `gemini-…` id selects Gemini |
| `PROOFREADER_CONCURRENCY` | requests in flight, 1–64 (default 8) |
| `PROOFREADER_TIMEOUT_SECONDS` | seconds per request, 1–600 (default 60) |

Keys are needed only for live runs, not for tests or replay. Only the selected provider's key is read.

## Usage

```powershell
# Default provider (Anthropic, claude-opus-5-5)
.venv\Scripts\python -m proofreader report.xml --lang en --style-guide guide.docx

# Gemini, selected explicitly; one request at a time suits free-tier rate limits
.venv\Scripts\python -m proofreader report.xml --lang en --style-guide guide.docx --model gemini-3.5-flash-lite --concurrency 1

# Record a live run, then reproduce it offline (no key, no network)
.venv\Scripts\python -m proofreader report.xml --lang en --style-guide guide.docx --record run.recording.json
.venv\Scripts\python -m proofreader report.xml --lang en --style-guide guide.docx --replay run.recording.json --force

# Also write metrics as JSON
.venv\Scripts\python -m proofreader report.xml --lang en --style-guide guide.docx --metrics-json run.metrics.json
```

| Option | Meaning |
|---|---|
| `INPUT` | the XML file to proofread (up to 10 MiB) |
| `--lang TAG` | required BCP 47 tag; `en` and `en-US` receive the full style guide; any other tag (including `en-GB`) receives only sections approved as locale-neutral (none are currently approved) and is proofread in that language; malformed tags exit 2 |
| `--style-guide FILE` | required; the style guide (`.docx`) |
| `--model ID` | model id (default `claude-opus-5-5`); a lowercase `gemini-…` id selects Gemini; any other id that mentions Gemini exits 2 |
| `--concurrency N` | requests in flight, 1–64 (default 8) |
| `--timeout S` | seconds per request, 1–600 (default 60) |
| `--max-calls N` | refuse to run if more than N paragraphs need a model call (default 200) |
| `--record FILE` / `--replay FILE` | record a live run / serve recorded responses offline (`*.recording.json`); not both |
| `--force` | replace existing output, recording, metrics or log files (never the input) |
| `--log-level LEVEL`, `--log-file FILE` | logging level (default INFO) and an optional log file |
| `--metrics-json FILE` | write run metrics as JSON |
| `--heap-metrics` / `--no-heap-metrics` | Python heap tracing (automatic for inputs up to 256 KiB) |
| `--base-url URL`, `--allow-custom-base-url` | Anthropic models only; https only; a non-default endpoint needs the explicit flag |
| `--cache` | refused: prompt caching is not implemented |

## Input, output and exit codes

- The corrected document is written next to the input as `INPUT.corrected.xml` (only the last suffix is replaced).
  Existing files are refused unless `--force`; the input is never overwritten; files are written atomically.
- Each finding becomes `<error type="TYPE" correction="TEXT" reason="WHY">original text</error>`, where `TYPE` is
  `grammar`, `spelling`, `punctuation`, `capitalization`, `clarity` or `styleguide`. A style-guide finding without
  a reason is rejected; the model is instructed to name the rule in the reason.
- Paragraphs with no findings, CDATA sections, paragraphs that already contain `<error>`, and everything outside
  `<p>` text are left exactly as they were. Paragraphs over 50 words or 5,000 characters are not sent; they are left
  unchanged with a warning, and the run exits 3.

**Example** — invented text; actual output of a live `gemini-3.5-flash-lite` run with the one-rule style guide
"Write per cent in running text, never the % sign":

```xml
<!-- input -->
<doc><p>The comittee approved a 5% increase in the travel budget.</p></doc>
<!-- output -->
<doc><p>The comittee approved a <error type="styleguide" correction="5 per cent" reason="Write per cent in running text per house style.">5%</error> increase in the travel budget.</p></doc>
```

The model also proposed `comittee` → `committee` but quoted the wrong surrounding words, so that finding was
rejected (`ANCHOR_MISMATCH`) rather than applied at a guessed position.

**Exit codes:** `0` success · `3` output written, but some paragraphs passed through unchanged (or an auxiliary
file failed) · `1` fatal error, no output · `2` usage error or unsupported input, no output · `130` interrupted.

**Logs** contain no document text, corrections, prompts, style-guide content or keys. Every completed run ends with
a runtime summary; `--metrics-json` adds per-paragraph latency, attempts, token usage and findings by status and
rejection code. From the live run above:

```
INFO run time 2.090 s (read_input 0.005 s, load_style_guide 0.015 s, pipeline 1.783 s [prepare 0.013 s,
calls 1.767 s, resolve 0.002 s, annotate 0.001 s], write_output 0.005 s); peak heap 10.0 MiB; peak process memory 89.5 MiB
```

## Testing

```powershell
.venv\Scripts\python -m pytest
```

The suite runs offline: a session-wide guard (`tests/conftest.py`) fails any test that opens a non-loopback
connection, and the model clients are exercised through in-memory HTTP transports and a checked-in replay fixture.
It includes byte-for-byte golden tests, end-to-end CLI tests, property-based tests (Hypothesis) and an adversarial
suite (malformed XML, entity expansion, encodings, namespaces, CDATA, nested paragraphs, size limits).

Result on a fresh installation of this snapshot with a new Python 3.11.9 environment: **2,655 passed, 13 skipped**,
also with `--hypothesis-seed=77`. The 13 skips are 11 checks against private assessment files that are not part of
the repository and 2 symbolic-link tests that need symlink permission on Windows.

`scripts/evaluate.py` compares a corrected file with a reference annotation of the same document and reports
precision, recall and F1 at four matching levels (no pass threshold):

```powershell
.venv\Scripts\python scripts/evaluate.py --original IN.xml --expected GOLD.xml --predicted IN.corrected.xml
```

## Performance

Measured results only:

| Measurement | Result |
|---|---|
| Live Gemini run, one invented paragraph (`gemini-3.5-flash-lite`) | 1.77 s model call, 2.09 s total; 989 input / 178 output tokens; peak heap 10.0 MiB; peak process memory 89.5 MiB |
| Local processing in that run (mapping, resolution, annotation, verification) | about 0.02 s |
| Synthetic 10 MiB input with 1,639 annotations through mapping, annotation and verification (no model calls) | 28.9 s; 1.25 GB peak memory |

By design, paragraphs are sent concurrently (default 8 requests in flight), so model time on a multi-paragraph
document should grow with the slowest calls rather than their sum; this has not been measured on a real document.
Each paragraph uses at most 6 requests (2 transport retries and 1 retry for unusable output).

## Known limitations

- **Live validation is minimal**: Eight controlled live Gemini evaluation cases using invented text; four findings
  were applied and two were safely rejected because of invalid contextual anchors. All eight outputs passed XML
  verification. The Anthropic account used during development returned HTTP 429 for every request, so the
  Anthropic client is verified only offline (the real SDK over an in-memory transport).
- **Real-document evaluation is pending**: proofreading quality, latency and cost on real documents have not been
  measured for either provider.
- **Gemini anchoring**: Across the live Gemini diagnostics, the model sometimes returned incorrect or over-length
  surrounding-word context. In the eight-case evaluation, two valid corrections were rejected: one due to
  ANCHOR_MISMATCH and one due to INVALID_FIELD (six context words exceeded the four-word limit). Such findings are
  rejected rather than applied at guessed positions, so valid corrections can be lost, but the text is never
  corrupted.
- Straight quotation marks are not converted to curly ones (quotation marks are protected by design); headings and
  other text outside `<p>` are not proofread.
- Prompt caching is not implemented; the style guide is sent with every paragraph.
- Inputs near the 10 MiB limit need more than 1 GB of memory; large-input performance has not been optimised.

## Secrets and security

- API keys are read only from the environment or `./.env` (git-ignored), never from a command-line flag. They never
  appear in logs, error messages, metrics or recordings, and recordings refuse credential-like text.
- The Gemini key is sent only in the `x-goog-api-key` header, never in a URL. Provider endpoints are fixed unless
  an Anthropic endpoint is explicitly allowed with `--allow-custom-base-url`; only https is accepted.
- XML is parsed without fetching external DTDs or entities, and entity expansion is capped.
- Google may use input sent to the Gemini free tier to improve its products: send only non-confidential text
  through the free tier.

## Project documentation

- [`docs/README.md`](docs/README.md) — index of the design documents
- [`docs/02-xml-source-mapping.md`](docs/02-xml-source-mapping.md) — XML preservation design
- [`docs/03-final-architecture.md`](docs/03-final-architecture.md) — architecture, finding contract, acceptance criteria
- [`docs/04-decision-register.md`](docs/04-decision-register.md) — design decisions, measurements and risks

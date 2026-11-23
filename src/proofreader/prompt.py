"""Prompt construction (S4.3): fixed system prompt, deterministic JSON data block, output schema and
request fingerprint. No API calls, no response decoding (S4.5), no pipeline logic (S4.6).

* ``system_prompt()`` is the packaged ``prompts/system.md``: fixed text with no placeholders, so it is
  byte-identical for every paragraph and every run.
* ``build_user_message()`` serialises everything that varies - locale, scope, the selected style guide
  (as untrusted reference data) and the paragraph (segments, ``resolver.word_grid`` words, advisory
  protected-word hints) - as one deterministic JSON object.
* Segments are cut at the same boundaries ``resolver.word_grid`` uses (``{0, len} | barriers``), so they
  rebuild ``Paragraph.text`` exactly and every serialised word lies inside one segment. The boundary
  between two segments is the (non-text) nested-paragraph marker (U2).
* ``protected_words`` are hints only; the S3 resolver's quotation protection stays authoritative.
* ``output_schema()`` is ``anthropic.transform_schema(FindingList)`` with model-facing descriptions; every
  field, type, enum, ``required`` list and ``additionalProperties: false`` is kept unchanged.
* ``request_fingerprint()`` is a policy-free SHA-256 of every response-affecting setting; run policy
  (effort ``low``, ``max_tokens`` 8000) stays in ``config`` (S4.1) and reaches it via ``fingerprint_for()``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import anthropic

from .config import LocaleScope, RunConfig, UsageError, locale_scope, normalise_lang
from .models import MAX_ANCHOR_WORDS, Finding, FindingList, Paragraph, QuoteAnalysis
from .resolver import MAX_CLARITY_SPAN_WORDS, MAX_SPAN_WORDS, word_grid
from .style_guide import SelectedGuide

PROMPT_VERSION = "s4.3-v1"
FINGERPRINT_VERSION = "proofreader-request-v1"
MAX_CALL_PARAGRAPH_CHARS = 5_000  # defensive pre-call limit (Unicode code points)
MAX_CALL_PARAGRAPH_WORDS = 50  # assignment paragraph constraint, counted on word_grid
MAX_SCHEMA_DEPTH = 64
STYLE_GUIDE_TRUST = "untrusted reference data"
SYSTEM_PROMPT_RESOURCE = "system.md"

_JSON_DUMPS = {"ensure_ascii": False, "separators": (",", ":"), "allow_nan": False}

# Field limits come from the Pydantic models' own JSON schema (the single source of truth for validation);
# span and anchor limits come from the resolver / model constants. No number is written twice.
_FIELDS = Finding.model_json_schema()["properties"]
_MAX_FINDINGS = FindingList.model_json_schema()["properties"]["findings"]["maxItems"]
_ORIGINAL_MIN, _ORIGINAL_MAX = _FIELDS["original"]["minLength"], _FIELDS["original"]["maxLength"]
_CORRECTION_MAX = _FIELDS["correction"]["maxLength"]
_REASON_MIN, _REASON_MAX = _FIELDS["reason"]["minLength"], _FIELDS["reason"]["maxLength"]
_WORD_INDEX_MIN = _FIELDS["word_start"]["minimum"]

# Model-facing schema descriptions, keyed by their path in the transformed schema.
_DESCRIPTIONS: dict[tuple[str, ...], str] = {
    (): "Proofreading findings for one paragraph.",
    ("properties", "findings"):
        f"At most {_MAX_FINDINGS} findings; an empty list when the paragraph needs no change.",
    ("$defs", "Finding"): "One proposed correction of a span of the paragraph.",
    ("$defs", "Finding", "properties", "original"):
        f"Exact text to correct, copied verbatim from one segment; {_ORIGINAL_MIN}-{_ORIGINAL_MAX} characters; "
        f"at most {MAX_SPAN_WORDS} words ({MAX_CLARITY_SPAN_WORDS} for clarity).",
    ("$defs", "Finding", "properties", "before"):
        f"Up to {MAX_ANCHOR_WORDS} words immediately before original, verbatim from the same segment; "
        "empty only at a segment edge.",
    ("$defs", "Finding", "properties", "after"):
        f"Up to {MAX_ANCHOR_WORDS} words immediately after original, verbatim from the same segment; "
        "empty only at a segment edge.",
    ("$defs", "Finding", "properties", "word_start"):
        f"Index from words of the first word of original ({_WORD_INDEX_MIN} or more).",
    ("$defs", "Finding", "properties", "word_end"):
        "Index from words of the last word of original (inclusive; not less than word_start).",
    ("$defs", "Finding", "properties", "correction"): f"Replacement for original; at most {_CORRECTION_MAX} characters.",
    ("$defs", "Finding", "properties", "type"): "Type of the finding.",
    ("$defs", "Finding", "properties", "reason"):
        f"Short explanation, {_REASON_MIN}-{_REASON_MAX} characters; for styleguide, name the rule that applies.",
}

# Schema keywords whose value maps NAMES to sub-schemas: a key named "description" inside them is a
# property / definition name, not an annotation, and must never be removed.
_NAMED_SCHEMA_MAPS = frozenset({"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"})


# --------------------------------------------------------------------------------------------
# System prompt and schema
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def system_prompt() -> str:
    """The fixed system prompt (package data). Read once; identical for every paragraph and run."""
    text = resources.files("proofreader").joinpath("prompts", SYSTEM_PROMPT_RESOURCE).read_text(encoding="utf-8")
    if "{{" in text or "}}" in text:
        raise RuntimeError("system prompt must not contain placeholders")
    return text


def output_schema() -> dict:
    """``transform_schema(FindingList)`` with model-facing descriptions; a fresh copy on every call."""
    schema = copy.deepcopy(anthropic.transform_schema(FindingList))
    _strip_descriptions(schema)
    for path, text in _DESCRIPTIONS.items():
        node = schema
        for key in path:
            if not isinstance(node, dict) or key not in node:
                raise RuntimeError("unexpected FindingList schema shape: " + "/".join(path))
            node = node[key]
        if not isinstance(node, dict):
            raise RuntimeError("unexpected FindingList schema shape: " + "/".join(path))
        node["description"] = text
    return schema


def _strip_descriptions(node: object, *, names: bool = False) -> None:
    """Remove every ``description`` annotation. ``names`` is True for a mapping of property / definition
    names to sub-schemas, where a key called ``description`` is a name and is kept (its sub-schema is still
    cleaned)."""
    if isinstance(node, dict):
        if not names:
            node.pop("description", None)
        for key, value in node.items():
            _strip_descriptions(value, names=not names and key in _NAMED_SCHEMA_MAPS)
    elif isinstance(node, list):
        for value in node:
            _strip_descriptions(value)


# --------------------------------------------------------------------------------------------
# Paragraph serialisation
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SizeViolation:
    kind: str  # "chars" | "words"
    value: int
    limit: int


def _bounds(paragraph: Paragraph) -> tuple[int, ...]:
    """The segment edges ``resolver.word_grid`` uses: sorted unique {0, len} | in-range barriers."""
    n = len(paragraph.text)
    return tuple(sorted({0, n} | {b for b in paragraph.barriers if 0 <= b <= n}))


def paragraph_segments(paragraph: Paragraph) -> tuple[str, ...]:
    """The paragraph text split at its nested-paragraph barriers; ``"".join`` rebuilds it exactly."""
    bounds = _bounds(paragraph)
    return tuple(paragraph.text[a:b] for a, b in zip(bounds, bounds[1:]))


def segment_words(paragraph: Paragraph) -> tuple[tuple[tuple[int, str], ...], ...]:
    """``resolver.word_grid`` words grouped by segment as ``(index, text)`` pairs."""
    bounds = _bounds(paragraph)
    groups: list[list[tuple[int, str]]] = [[] for _ in range(max(len(bounds) - 1, 0))]
    segment = 0
    for word in word_grid(paragraph):
        while segment < len(groups) and not (bounds[segment] <= word.start and word.end <= bounds[segment + 1]):
            segment += 1
        if segment == len(groups):
            raise RuntimeError("word_grid word does not lie inside one segment")
        groups[segment].append((word.index, word.text))
    return tuple(tuple(group) for group in groups)


def protected_word_indices(paragraph: Paragraph, analysis: QuoteAnalysis) -> tuple[int, ...]:
    """Advisory hint: grid indices of words overlapping a protected quotation range (all words when the
    paragraph is guarded). The S3 resolver remains authoritative."""
    if analysis.paragraph_id != paragraph.id:
        raise ValueError("quotation analysis belongs to a different paragraph")
    grid = word_grid(paragraph)
    if analysis.unsafe_reason is not None:
        return tuple(w.index for w in grid)
    return tuple(w.index for w in grid if any(r.start < w.end and w.start < r.end for r in analysis.ranges))


def size_violation(paragraph: Paragraph) -> SizeViolation | None:
    """Pre-call size guard (never truncates): more than 5,000 characters or more than 50 words.

    Reports the FIRST exceeded limit only: characters are checked before words, so a paragraph over both
    limits is reported as a ``chars`` violation."""
    chars = len(paragraph.text)
    if chars > MAX_CALL_PARAGRAPH_CHARS:
        return SizeViolation("chars", chars, MAX_CALL_PARAGRAPH_CHARS)
    words = len(word_grid(paragraph))
    if words > MAX_CALL_PARAGRAPH_WORDS:
        return SizeViolation("words", words, MAX_CALL_PARAGRAPH_WORDS)
    return None


def build_user_message(paragraph: Paragraph, analysis: QuoteAnalysis, *, lang: str, scope: LocaleScope,
                       guide: SelectedGuide) -> str:
    """The deterministic JSON data block for one paragraph. Raises ValueError for caller errors."""
    if paragraph.skip_reason is not None or not paragraph.text.strip():
        raise ValueError("paragraph is not eligible for a model call")
    violation = size_violation(paragraph)
    if violation is not None:
        raise ValueError(f"paragraph exceeds the {violation.kind} limit ({violation.value} > {violation.limit})")
    try:
        normalised = normalise_lang(lang)
    except UsageError:
        raise ValueError("lang is not a well-formed BCP 47 tag") from None
    if normalised != lang:
        raise ValueError("lang must already be normalised")
    if not isinstance(scope, LocaleScope) or locale_scope(lang) is not scope:
        raise ValueError("scope does not match lang")
    if not isinstance(guide, SelectedGuide) or guide.scope is not scope:
        raise ValueError("selected style guide does not match scope")
    data = {
        "v": PROMPT_VERSION,
        "lang": lang,
        "scope": scope.value,
        "style_guide": {"trust": STYLE_GUIDE_TRUST, "text": guide.text} if guide.text else None,
        "paragraph": {
            "segments": list(paragraph_segments(paragraph)),
            "words": [[[index, text] for index, text in group] for group in segment_words(paragraph)],
            "protected_words": list(protected_word_indices(paragraph, analysis)),
        },
    }
    return json.dumps(data, **_JSON_DUMPS)


# --------------------------------------------------------------------------------------------
# Request fingerprint
# --------------------------------------------------------------------------------------------


def _validate_json(value: object, path: str, depth: int, active: set[int]) -> None:
    if depth > MAX_SCHEMA_DEPTH:
        raise ValueError(f"{path}: nested deeper than {MAX_SCHEMA_DEPTH} levels")
    if isinstance(value, dict):
        if id(value) in active:
            raise ValueError(f"{path}: self-referencing structure")
        active.add(id(value))
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: key of type {type(key).__name__} (only strings are allowed)")
            _validate_json(item, f"{path}.{key}", depth + 1, active)
        active.discard(id(value))
    elif isinstance(value, list):
        if id(value) in active:
            raise ValueError(f"{path}: self-referencing structure")
        active.add(id(value))
        for position, item in enumerate(value):
            _validate_json(item, f"{path}[{position}]", depth + 1, active)
        active.discard(id(value))
    elif value is None or isinstance(value, (str, bool, int)):
        return
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite number")
    else:
        raise ValueError(f"{path}: unsupported type {type(value).__name__}")


def validate_json_value(value: object, path: str = "value") -> None:
    """Public entry point to the recursive JSON validator used by ``request_fingerprint`` (S4.4, D6):
    string keys only, supported JSON types only, finite numbers, cycle detection, depth limit."""
    _validate_json(value, path, 0, set())


def request_fingerprint(*, model: str, effort: str, max_tokens: int, system: str, user: str,
                        schema: dict[str, object]) -> str:
    """Policy-free SHA-256 (64 lowercase hex) of every response-affecting request setting.
    Validates format only; run policy belongs to ``config`` (use ``fingerprint_for``)."""
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    if not isinstance(effort, str) or not effort:
        raise ValueError("effort must be a non-empty string")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(system, str):
        raise ValueError("system must be a string")
    if not isinstance(user, str):
        raise ValueError("user must be a string")
    if not isinstance(schema, dict):
        raise ValueError("schema must be a JSON object (dict)")
    _validate_json(schema, "schema", 0, set())
    payload = {
        "fingerprint": FINGERPRINT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "effort": effort,
        "max_tokens": max_tokens,
        "system": system,
        "user": user,
        "schema": schema,
    }
    canonical = json.dumps(payload, sort_keys=True, **_JSON_DUMPS)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fingerprint_for(config: RunConfig, *, system: str, user: str, schema: dict[str, object]) -> str:
    """The call path for later stages: model, effort and max_tokens come from the RunConfig (S4.1)."""
    return request_fingerprint(model=config.model, effort=config.effort, max_tokens=config.max_tokens,
                               system=system, user=user, schema=schema)

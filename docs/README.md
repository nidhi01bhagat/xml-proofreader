# Architecture & Requirements Analysis — Index

**Project:** GenAI-Powered XML Proofreader (assessment)
**Status (2026-11-23):** D1–D17 accepted with adjustments (see 04). All implementation stages are committed:
S0 (repository foundation), S1 (encoding check, XML source mapping, tokenization), S2 (annotation, rendering,
independent verification, rollback), S3 (quotation protection, finding resolution, tracing), S4 (style guide,
prompt, model clients, pipeline, CLI, validation layer) and S5 (Gemini provider). Test results, measurements and
known limitations are summarised in the top-level [README](../README.md); risks and measurements in detail are in
04. Where 03 and 04 disagree, **04 (decision record) wins**.

This is a public snapshot. The requirements and source analysis (document 01) and the internal status log are not
included, because they describe the confidential assessment material; references to "01" in these documents point
to that excluded analysis. The internal development history is not part of this repository.

## Files

| File | Contents |
|---|---|
| [02-xml-source-mapping.md](02-xml-source-mapping.md) | XML preservation strategy, Expat mapping algorithm, nested ownership, CDATA/entity/inline rules, encoding policy, quotation policy, verifier, evidence status |
| [03-final-architecture.md](03-final-architecture.md) | Project structure, modules, finding contract, resolver, provider choice, error policy, security, 32 acceptance criteria, test specification, implementation stages |
| [04-decision-register.md](04-decision-register.md) | Decision record for D1–D17 (with adjustments), additional requirements AR1–AR5, superseded original recommendations, risks, blockers |

## Evidence labels used throughout

- **[A]** assignment document · **[SG]** style guide · **[EX]** supplied example files · **[REC]** engineering recommendation · **[ASSUME]** assumption
- **T** — covered by the committed pytest suite (runs on a clean clone unless marked local only)
- **P-spike** — demonstrated by an experimental pre-implementation spike script (not part of this repository; its
  invented cases M1–M13 are ported to the pytest suite)
- **P-adhoc** — demonstrated by in-memory scripts during the review (not saved)
- **SPEC** — specified, not yet executed

Only **T** counts as verified; the other labels record the pre-implementation evidence.

## Superseded positions (history of the review)

| Earlier position | Final position |
|---|---|
| Schema `{original, occurrence, …}` | Anchored contract `{original, before, after, word_start, word_end, …}` |
| Fine-token index + "closest occurrence" | Anchors decide; word index only as exact tiebreaker; **no ±1 fallback** |
| Overlap: styleguide > longer > earlier | Category priority → **shorter** → earlier (D15) → **S2/S3: no priority; all overlapping or conflicting annotations rejected** (D15 disabled, decided in S3) |
| X11 "properly nested ranges, descending insertion, drop lowest priority" | Disjoint spans only, one ascending insertion pass, renderer refuses overlaps (02 X11) |
| Encoding override proves UTF-8 | Expat auto-detects UTF-16 even under a UTF-8 override → verifier classifies raw bytes itself (02 §2, §8.1) |
| Find start-tag end by scanning for `>` | Event raw range = `[index, next event index)` |
| Decode entities ourselves | Expat delivers references as atomic callbacks |
| `itertext()` for verification | Recursive own-text that excludes nested `<p>` |
| "Expat rejects unsupported encodings" | Expat **accepts** UTF-16, ISO-8859-1, windows-1252 → explicit pre-parse policy required |
| lxml "rewrites declaration/empty tags" | Unverified (lxml not installed) |

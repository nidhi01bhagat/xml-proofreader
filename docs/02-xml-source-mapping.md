# 02 — XML Source Mapping, Preservation & Verification

Evidence: **T** (committed pytest suite) · **P-spike** (saved spike script) · **P-adhoc** (in-memory, not saved) ·
**SPEC** (not executed). Sections 1–7 record the design analysis; where the S2/S3 implementation refined a rule, the
rule text below states the implemented behaviour (X9, X11, X12, §6, §8) and §10 gives the current evidence.

## 1. Strategy decision

| | A. Expat mapping + patching (bare) | B. lxml tree manipulation | **C. Hybrid (chosen)** |
|---|---|---|---|
| Untouched content | byte-identical by construction | infoset-equal; serializer normalises declaration, quoting, char refs, CRLF *(unverified — lxml not installed)* | byte-identical, **checked every run** |
| Annotation safety | silent CDATA corruption possible (P-adhoc) | well-formed by construction; CDATA marker lost when split | same as A + verifier + rollback |
| Dependencies | stdlib | lxml | stdlib |
| Risk | medium | medium | **low** |

**C = Expat atom mapping + source-byte insertion + independent verification + fail-closed per paragraph.**

## 2. Verified Expat behaviour (expat 2.6.0, Python 3.11.9)

- `buffer_text=False` (default): character data arrives as **atoms** — literal runs split at newlines; each `&amp;`/`&#8217;`/internal entity as its own callback; each `\r\n`/`\r` as its own `"\n"` callback (P-adhoc).
- `CurrentByteIndex` is the exact source byte of each event; event raw range = `[index, next event index)` (P-adhoc, P-spike).
- **`buffer_text=True` merges chunks and breaks offsets — must stay False** (P-adhoc).
- BOM is counted in offsets; ISO-8859-1 offsets refer to source bytes (P-adhoc).
- `<p/>`: end event index is the position after the tag (zero-width end) (P-adhoc).
- Entity whose replacement contains markup → events with **zero-width** raw ranges (P-adhoc) → paragraph unannotatable.
- Undeclared entity with external DTD → `SkippedEntityHandler` event (P-adhoc); without DTD → fatal parse error.
- Expat **accepts** UTF-16, `utf8` alias, ISO-8859-1, windows-1252; **rejects** UTF-32, invalid UTF-8, non-ASCII under US-ASCII, UTF-8 BOM + UTF-16 declaration (P-adhoc).
- S2.6 (T): Expat and ElementTree accept **all eight UTF-16 forms** (LE/BE × with/without BOM × with/without
  declaration) **even when told the input is UTF-8** — Expat auto-detects UTF-16 from the BOM or the NUL-byte pattern.
  BOM-less UTF-32 (with or without declaration) is rejected ("invalid token"). An encoding override therefore never
  proves the input is UTF-8; the verifier classifies the raw bytes itself (§8).

## 3. Mapping model

```
Atom      = {raw_start, raw_end, text, kind: LIT|REF|NL|CDATA|BAD, container_id, path}
Paragraph = {id, element_id, atoms[], text, barriers[], annotatable, skip_reason}
```

- **LIT** raw bytes decode to text · **REF** raw starts with `&` (atomic) · **NL** raw is `\r\n`/`\r` (atomic) · **CDATA** inside a CDATA section · **BAD** anything else → paragraph not annotatable.
- Char → byte: inside LIT `raw_start + len(text[:k].encode(enc))`; REF/NL only at their exact start/end, else `SPLITS_ATOMIC`.
- **Binding:** opening tag binds to the char at `start` (insert immediately before its bytes, i.e. after preceding markup); closing tag binds to char `end−1` (immediately after its bytes, before following markup).
- Self-checks: atoms + markup events tile each paragraph's content bytes contiguously; paragraph text equals verifier-style recursive own-text from ElementTree; mismatch → paragraph passthrough.

## 4. Nested paragraph ownership (P-adhoc ownership and barrier; not covered by the spike)

- **Each text atom is owned by the innermost open `<p>` on the Expat element stack.** Outer paragraphs never receive text of nested `<p>` (including their inline children). Ownership partitions text → no text can be annotated twice.
- **Barrier:** logical position in the outer text where a nested `<p>` sits between two consecutive owned atoms (computed from source byte ranges). Spans crossing a barrier → `CROSSES_PARAGRAPH`.
- Each nested `<p>` is proofread and verified independently.

Demonstrated:
```
<doc><p>Outer teh <p>Inner teh</p> more teh</p></doc>
outer text 'Outer teh  more teh'  atoms ('Outer teh ', 8–18), (' more teh', 34–43); barrier at 10
inner text 'Inner teh'            atoms ('Inner teh', 21–30)
span 'teh  more' WITHOUT barrier → would wrap b'teh <p>Inner teh</p> more' (the hole)
WITH barrier → REJECT CROSSES_PARAGRAPH
```

## 5. Annotation rules (conservative)

| # | Situation | Rule | Code |
|---|---|---|---|
| X1 | within one text atom | allowed | — |
| X2 | across atoms in same container (text + refs + newlines) | allowed; atomic atoms covered whole | SPLITS_ATOMIC |
| X3 | encloses complete inline elements / comments / PIs / `<br/>` | **D4: OFF by default** → reject. When enabled (only after dedicated tests): allowed, bytes unchanged, they gain an `<error>` parent | CROSSES_ELEMENT (D4 off) |
| X4 | partially crosses an inline element | **D4: OFF by default** → reject. When enabled: zero-text expansion only (move boundary over markup when no text lies between it and the element edge; repeat outwards); else reject | CROSSES_ELEMENT |
| X5 | an endpoint inside a CDATA section, or the span contains any part of a CDATA section | reject. CDATA sections are **preservation boundaries**: no `<error>` markup is inserted inside, enclosing or across them. A span that merely ends at or starts at a section boundary, without containing any part of it, is not affected. May be relaxed only by a separately tested implementation that proves XML semantics and the original-text invariant are preserved | IN_CDATA |
| X6 | paragraph contains an `<error>` (local name, any depth) | skip whole paragraph (idempotent re-runs) | PARA_HAS_ERROR |
| X7 | nested `<p>` | own-text only; no crossing; no enclosing a `<p>` | CROSSES_PARAGRAPH |
| X8 | comments/PIs | never text; may be enclosed only when D4 is enabled (see X3) | CROSSES_ELEMENT (D4 off) |
| X9 | quotation (§6, implemented in S3) | reject a span that contains a quotation mark, overlaps a protected range, or is adjacent to one; reject a correction that introduces a quotation delimiter | TOUCHES_QUOTATION / IN_QUOTATION / QUOTATION_IN_CORRECTION |
| X10 | BAD atom, skipped entity, zero-width text, tiling/cross-check failure | paragraph passthrough | PARA_UNMAPPABLE |
| X11 | overlap (S2, implemented): spans must be **disjoint**. Exact duplicates collapse into one; same span and category with a different correction **or** reason → every one involved rejected; nested, partially overlapping, or same span with different categories → every one involved rejected; adjacent spans allowed. Overlap is evaluated only among span-safe annotations (an invalid one cannot knock out a valid one). **No winner is picked** (no D15 priority in S2). Rendering inserts in one ascending pass; closing tag before opening tag at an equal offset; the renderer refuses (never repairs) overlapping or shared byte ranges | reject all involved | CONFLICT / OVERLAP |
| X12 | post-render verification (§8), implemented in `annotation.annotate()`: each round re-renders **from the original bytes** with the retained spans and verifies the exact output; paragraphs blamed by the verifier are rolled back (per paragraph id, so nested paragraphs are independent). Document-level issues, blame on an unannotated paragraph, renderer refusal or non-convergence → **fail closed** (no output) | failing paragraph rolled back | VERIFY_FAILED |

### Worked examples (all invented)

IDs match the spike's case labels (`M1`…`M13`). Sub-cases a/b/c follow the order of the spike cases that share a label.

**Evidence per column:**
- **D4 on (spike):** what the spike actually ran. The spike always maps with zero-text expansion and enclosure
  enabled, so this column is **P-spike**, except where a cell says SPEC.
- **D4 off (default):** the approved behaviour.
  - Where the D4-on cell reads "same", there is no markup to expand over, so the D4-off result is the spike
    result (**P-spike**).
  - Where the D4-off cell reads `CROSSES_ELEMENT` and the D4-on cell shows an expansion or enclosure, the rejection
    is **SPEC**: the spike has no D4-off switch.

**D4 off rejects only** spans that would need zero-text expansion or enclosure of markup (M2a, M6b, M7, M8, M9, M11).
Spans entirely inside one element (M2b, M6a), spans over references or line breaks (M3, M10, M12) and the other
same-row cases are unaffected. M6c is rejected under both settings.

**Naming:** the resolver's code for "no token-aligned occurrence" is `NOT_FOUND`. The spike's simplified `locate()`
reports the same condition as `NOT_UNIQUE(0)`.

| ID | Input | Finding | D4 off (default) | D4 on (spike) |
|---|---|---|---|---|
| M1 | `<p>We recieved teh parcel.</p>` | `recieved` | chars [3,11) → open@6, close@14 ✔ | same |
| M2a | `<p>We saw a <em>bright</em>-red kite.</p>` | `bright-red` | CROSSES_ELEMENT | start moved before `<em>` → `<error…><em>bright</em>-red</error>` ✔ |
| M2b | same | `bright` | wrapped inside `<em>` ✔ | same |
| M2c | same | `t-red` | not token-aligned → NOT_FOUND | same (spike: `NOT_UNIQUE(0)`) |
| M3a | `<p>Salt &amp; peper are on teh table.</p>` | `Salt & peper` | whole `&amp;` reference inside span ✔ | same |
| M3b | same | `&` | wraps the whole `&amp;` reference ✔ | same |
| M4 | `<p>She wrote &#8217;back soon&#8217; on teh door.</p>` | `wrote` / `back soon` | `wrote` ✔ / `back soon` → IN_QUOTATION | `wrote` ✔ (spike) / `back soon`: SPEC (quotation policy not in spike) |
| M5 | `<p><![CDATA[Teh quick fox]]></p>` | `Teh` / `Teh quick fox` | PARA_NOT_ANNOTATABLE (paragraph skipped as `CDATA_ONLY`, S1 decision); IN_CDATA applies to CDATA inside otherwise annotatable paragraphs (CD-04) | IN_CDATA / IN_CDATA (spike predates `CDATA_ONLY`) |
| M6a | `<p>Ask the <em>night-shift</em> porter.</p>` | `night-shift` | wrapped inside `<em>` ✔ | same |
| M6b | same | `night-shift porter` | CROSSES_ELEMENT | start moved before `<em>` → `<error…><em>night-shift</em> porter</error>` ✔ |
| M6c | same | `the night` | CROSSES_ELEMENT | CROSSES_ELEMENT (text `-shift` lies between end and `</em>`) |
| M7 | `<p>my pet <b>dgo</b></p>` | `pet dgo` | CROSSES_ELEMENT | close moved after `</b>` ✔ |
| M8 | `<p>a <b><i>teh</i></b> cat</p>` | `teh cat` | CROSSES_ELEMENT | two-level expansion ✔ |
| M9 | `<p>very<!--c--> <?pi x?>good</p>` | `very good` | CROSSES_ELEMENT | comment + PI enclosed ✔ |
| M10 | `<p>ice\r\ncream cone</p>` | `ice cream` (whitespace gap) | `ice\r\ncream` wrapped, `\r\n` kept ✔ | same (byte mapping only; resolver whitespace-gap match: SPEC) |
| M11 | `<p>ice<br/>cream</p>` | `icecream` | CROSSES_ELEMENT | encloses `<br/>` ✔ |
| M12 | `<!DOCTYPE p [<!ENTITY co "Acme corp">]><p>See &co; now</p>` | `Acme` / `Acme corp` | SPLITS_ATOMIC / whole `&co;` wrapped ✔ | same |
| M13 | UTF-8 BOM + `<p>teh</p>`; `<r xmlns="urn:x"><p>teh</p></r>`; US-ASCII `<p>dont go</p>` with correction `don’t` | `teh` / `teh` / `dont` | BOM kept ✔; default-namespace `<p>` matched ✔; `don&#8217;t` written ✔ | same |
| G | supplied example input/output pair (**local only**, git-ignored) | derived from the output at run time | byte-for-byte reproduction ✔ | — |

CDATA (M5): under X5 a span is rejected if it has an endpoint inside a CDATA section, or the span contains any part
of a CDATA section; this includes a span equal to the whole section content. A paragraph whose only text is CDATA
(plus whitespace) is skipped entirely (`CDATA_ONLY`), so annotations on it are rejected as PARA_NOT_ANNOTATABLE.

**Ported to pytest (S2, T, D4 off):** `tests/test_golden.py` runs every D4-off case through the full S2 path
(prepare → render → verify). Accepted cases (M1, M2b, M3a, M3b, M4 `wrote`, M6a, M10, M12 whole entity, M13) must
equal **hand-written** expected bytes; rejected cases (M2a, M5, M6b, M6c, M7, M8, M9, M11, M12 partial) must leave
the document byte-identical with the listed code. M2c (`NOT_FOUND`) and M4 `back soon` (`IN_QUOTATION`) belong to
the S3 resolver/quotation layer and are not S2 cases. The **G** row is `test_golden_b_supplied_example_pair_byte_for_byte`
(local only; skipped on a clean clone).

## 6. Quotation policy (implemented in S3.2/S3.3/S3.6: `quotes.py`, `resolver.py`; T)

Rule (D6): quoted material is never edited. Uncertainty always protects **more** text (a lost correction is
acceptable; editing a quotation is not). The analysis works on a paragraph's own logical text (CDATA, inline-element
and reference text included; comments/PIs contribute no text); quotation marks written as character references are
seen as the characters they denote.

### 6.1 Classification (decision E; ★ = S3 extension of the original table)

Context: p / n = base character before (combining marks skipped) / after the mark. **boundary** = paragraph edge,
whitespace, `( [ { — –`, or a nested-`<p>` barrier★. **punct** = Unicode category P* that is not itself a quotation
character★. L/N = letter / number.

| Character(s) | Family | Role |
|---|---|---|
| `“` `„` `‟`★ `«` `〝`★ | double | OPEN |
| `”` `»` `〞`★ `〟`★ | double | CLOSE (definite) |
| `"` and `＂`★ | double | OPEN if p boundary & n non-space; CLOSE if p non-space & n boundary/punct; else AMBIGUOUS |
| `‹` / `›` | single★ | OPEN / CLOSE (definite pair) |
| `‘` `‚`★ `‛`★ | single | OPEN (`‘` is AMBIGUOUS if letter–letter) |
| `’` `'` | single | APOSTROPHE if letter–letter (`don't`), digit–letter (`1950’s`, S4.8e) or boundary–digit (`’90s`); CLOSE-OR-APOSTROPHE if L/N★ then boundary/punct (`boys’ `, `left’.`); OPEN-CANDIDATE if boundary–letter (`'tis`); else AMBIGUOUS |
| `「` / `」`★, `『` / `』`★ | corner / white corner | OPEN / CLOSE (definite) |
| any other Unicode Pi/Pf character★ | other | AMBIGUOUS |
| primes `′` `″` | — | not quotation marks |

Closers pair only with their own openers: `”` closes `“ „ ‟`; `»` closes `«`; `›` closes `‹`; `」` closes `「`;
`』` closes `『`; `〞`/`〟` close `〝`; each straight double mark pairs only with itself. Known, accepted
consequences, all pinned by tests: a `’` directly before `”` is AMBIGUOUS
(quotation marks are not "punct"); German `„…“` never closes (`“` classifies as OPEN), so it protects to the
paragraph end.

### 6.2 Pairing and protected ranges

1. A definite closer pops the **nearest matching** opener of its family (others stay open) → range
   [opener, closer], marks included (`pair`). Without one: [paragraph start, closer] (`empty_stack_close`, open_start).
2. **Rule S** (single family, except `‹ ›`): an OPEN / OPEN-CANDIDATE protects to the **last** CLOSE-OR-APOSTROPHE
   after it, else to the paragraph end. A CLOSE-OR-APOSTROPHE is never a definite close; one left outside every
   range becomes an APOSTROPHE (final role).
3. Unclosed openers protect to the paragraph end (`unclosed`, open_end).
4. AMBIGUOUS marks protect from the mark to the paragraph end; an odd count of one straight double kind protects
   from its first unmatched occurrence to the end (`ambiguous`).
5. **Guard (decision F):** more than 64 non-apostrophe marks, or more than 8 overlapping ranges (computed before
   merging), protects the whole paragraph (`guard`).
6. Ranges are sorted and merged (touching ranges combine). Every quotation mark lies inside a protected range.

Example: `‘dogs’ and cats’ were fed.` → one range from `‘` to the **last** candidate closer `cats’` (rule S), so
`and` is protected even though a reader might take `dogs’` as the closer (documented conservative loss).

### 6.3 Nested paragraphs (decision U3; `quotes.analyse_document`)

Each paragraph is analysed on its own text; barriers count as boundaries. A child paragraph inherits
**whole-paragraph** protection (`inherited`, `inherited_from` = its direct parent) only if one of the parent's
**individual** protected ranges (evaluated before touching ranges are merged) actually spans the child's barrier:
strictly inside the range, or at a paragraph edge the range reaches only because a mark is missing (open_start at
offset 0, open_end at the paragraph end, the guard). Two separate quotations that merely touch at the barrier
(`“a”<p>…</p>“b”`) do not protect the child; the parent's final merged ranges are unaffected. Propagation is transitive (an entirely
protected child protects its own children). Siblings outside the parent's quotations keep their own analysis.
Barrier offsets are derived from S1 data (`nested_barriers`) and equal `Paragraph.barriers` on every fixture.

### 6.4 Span checks (decision G; `quotes.span_verdict`)

Checked in this order for a resolved span: the span **contains** a quotation mark (final role not APOSTROPHE) →
TOUCHES_QUOTATION; it **overlaps** a protected range → IN_QUOTATION; it is **adjacent** to one (ends where a range
starts or starts where one ends) → TOUCHES_QUOTATION. Apostrophes never block.

### 6.5 Quotation delimiters in corrections (revision 3; `resolver._introduced_apostrophe_verdict`)

Corrections are attribute values and never change source bytes; this is a content rule. The **source** quotation
structure (6.2, unchanged) is the only authority; corrections never re-pair it. In the corrected text
T' = text[:s] + correction + text[e:] (barriers after the span shifted):
- any delimiter other than `'` and `’` (including `‘`, `"` and other Pi/Pf characters) → QUOTATION_IN_CORRECTION;
- each `'` / `’` the correction introduces, classified locally in T': APOSTROPHE → allowed (`don’t`, `boy’s`);
  CLOSE-OR-APOSTROPHE → allowed (a possessive outside every completed quotation: 6.4 already guarantees the span is
  outside and not adjacent to any source range); OPEN-CANDIDATE or AMBIGUOUS → rejected (would open / create an
  ambiguous quotation);
- neighbour stability: existing quotation marks at s−1 and e must keep their local role (correcting `cats` to
  `cats ` would make the following `’` ambiguous → rejected);
- defensive: an introduced apostrophe at a position inside or next to a source range → rejected (unreachable through
  the pipeline because 6.4 runs first; unit-tested).

Examples (all invented, all executed tests): `‘dogs’ and cats’ met the boys bikes.` with `boys` → `boys’` is
**allowed** (after the completed quotation); `‘dogs’ and the boys bikes and cats’ fun.` with the same correction →
IN_QUOTATION (inside rule S's range); `tis` → `’tis` → QUOTATION_IN_CORRECTION.

Tests: Q-01…Q-21 (03 §8) are executed twice — on the quotation analysis (`tests/test_quotes.py`, including a check
that the test data matches the 03 §8 table) and through the resolver (`tests/test_resolution_pipeline.py`).

## 7. Encoding policy (expat behaviour P-adhoc; policy SPEC)

Pre-parse check (before any API call or output file):

| Input | Action |
|---|---|
| UTF-8 without BOM; declaration UTF-8/utf8 (any case) or none | accept |
| UTF-8 with BOM; declaration none or UTF-8 | accept; BOM bytes untouched |
| US-ASCII/ASCII declared | accept only if all bytes < 0x80; inserted non-ASCII as `&#N;` |
| UTF-16/UTF-32 (BOM or `3C 00 3F 00`-style patterns) | **exit 2**, no output |
| any other declared encoding (ISO-8859-1, windows-1252…) | **exit 2** (expat would accept — explicitly rejected) |
| BOM contradicts declaration | **exit 2** |
| declared UTF-8 but invalid bytes | **exit 1** (byte offset reported) |
| ASCII declared but byte ≥ 0x80 | **exit 1** |
| malformed XML | **exit 1** (line/column) |

Declaration and BOM never rewritten; source never re-encoded; documents without a declaration stay without one.

## 8. Independent verification (implemented in S2: `verify.py`, T)

Byte restoration (G1) only proves nothing else changed; it does not prove the right text was wrapped. The verifier re-parses output with ElementTree (different code path from the byte mapper):

- **own_text(el)** = `el.text` + Σ children (own_text(child) if child is not `<p>`) + `child.tail` — recursive, document order, nested `<p>` excluded. **Never use `itertext()`** (it includes nested paragraphs: `'Outer teh Inner teh more teh'`).
- **V1** flattened text of k-th inserted `<error>` == source text of k-th accepted span.
- **V2** logical start/end via document-order walk == resolved offsets.
- **V3** attributes exactly equal intended values; order type, correction, reason.
- **V4** structure: inserted-error count == accepted spans; `<p>` ancestor; no `<error>` ancestor.
- **V5** structural event stream (start-ns, tags with namespace URI, attributes, comments, PIs) of output minus inserted errors == input stream.
- **V6** paragraph own-text unchanged (I1). **G1** inserted ranges removed == input bytes.

Demonstrated (P-adhoc, re-run 2026-11-21) for source `<p>We saw a <em>bright</em>-red kite.</p>` and output
`<p>We saw a <error …><em>bright</em>-red</error> kite.</p>` (a D4-on rendering):
`<error>.text` is `None` (so `.text` alone is insufficient); flattened text `'bright-red'` == T[9:19]; attributes exact;
paragraph text unchanged. Earlier review: a namespaced variant (`x:em`, `x:id`, comment, PI) gave an identical structure stream.

Limitations: ElementTree shares the expat library (different code path, same tokenizer); CDATA markers and entity spellings are invisible to ElementTree → covered by G1 only. Optional: lxml as test-only second parser (D16).

### 8.1 Implemented verifier (S2)

`verify(original, output, applied) -> VerificationReport` uses only those three inputs: never the renderer's
offsets or byte ranges, never S1's source map (its only project import is `models`; a test enforces this).
Two parsing paths: ElementTree (own-text, error text and attributes: V1–V3, V6) and a minimal namespace-aware Expat
event recorder (tag byte positions for G1, nesting and placement for V4, the structural stream for V5). Nothing
external is ever fetched.

- **Which `<error>` elements are inserted:** those whose nearest `<p>` received annotations. Such paragraphs had no
  `<error>` in the original (X6), so every `<error>` there must match an applied annotation.
- **Namespace inheritance (S2.3 decision):** the inserted `<error>` is unprefixed and declares no namespace, so it
  inherits the default namespace in scope; it is recognised by local name. A prefix or `xmlns` on it →
  `NAMESPACE_MISMATCH`.
- **Canonical tag bytes (S2.6):** semantically equal but differently spelled tags are rejected (`TAG_NOT_CANONICAL`):
  exactly `<error type="T" correction="C"[ reason="R"]>` … `</error>`, single spaces, double quotes; values use only
  `&amp; &lt; &gt; &quot;`; decimal references without leading zeros; references never spell ASCII characters other
  than TAB/LF/CR. **Encoding-aware:** in US-ASCII documents the tag is pure ASCII and non-ASCII characters appear as
  decimal references; in UTF-8 documents non-ASCII characters are raw and references are allowed only for TAB/LF/CR.
- **Encoding classification (S2.6, independent of S1):** raw bytes first, on every path including "nothing applied":
  a UTF-16/UTF-32 byte-order mark or any NUL byte → `ENCODING_UNSUPPORTED` (a NUL byte cannot occur in well-formed
  UTF-8/US-ASCII XML but always occurs in UTF-16/UTF-32 markup). Then the declaration, read by the verifier's own
  Expat parser (on the "nothing applied" path only the declaration bytes are parsed; documents without a
  declaration are not parsed at all): `utf-8`/`utf8`/none → UTF-8; `us-ascii`/`ascii` (case-insensitive) → US-ASCII,
  but a UTF-8 BOM or any non-ASCII byte contradicts it; any other name → `ENCODING_UNSUPPORTED`. A malformed
  declaration → `ORIGINAL_UNPARSEABLE`. Input is never treated as UTF-8 by default when it cannot be classified.
- **Issue codes.** Document level (output unusable, fail closed): `ORIGINAL_UNPARSEABLE`, `ENCODING_UNSUPPORTED`,
  `OUTPUT_NOT_WELL_FORMED`, `UNCHANGED_EXPECTED`, `INVALID_ANNOTATION`, `PARAGRAPH_COUNT_CHANGED`,
  `ERROR_OUTSIDE_PARAGRAPH`, plus `BYTES_NOT_RESTORED` / `STRUCTURE_CHANGED` outside annotated paragraphs. Paragraph
  level (rolled back by X12): `BYTES_NOT_RESTORED` (G1), `STRUCTURE_CHANGED` (V5), `TEXT_CHANGED` (V6),
  `MISSING_ERROR` / `EXTRA_ERROR` / `NESTED_ERROR` (V4), `SPAN_MISMATCH` (V1/V2), `ATTRIBUTES_MISMATCH` (V3),
  `NAMESPACE_MISMATCH`, `TAG_NOT_CANONICAL`.
- **Evidence:** a property test changes one byte of a correct rendering (replace/delete/insert, 300 random cases per
  run) and requires detection; a sweep of 14,481 characters per encoding confirmed every renderer-generated tag
  passes (S2.6 review, not part of the suite).

## 9. Supported input constraints (MVP)

1. Well-formed XML 1.0, UTF-8 (BOM optional) or US-ASCII.
2. Paragraph = element with local name `p`, any namespace (D2).
3. Annotatable: spans over LIT/REF/NL atoms with both endpoints in the same element instance and no markup inside the span (D4 off, default; zero-text expansion/enclosure only if D4 is enabled), no endpoint inside a CDATA section and no part of a CDATA section contained in the span (X5), not crossing nested `<p>`, not in quotations.
4. Preserved but not annotated: CDATA content, partial custom-entity expansions, paragraphs with markup/skipped entities, paragraphs already containing `<error>`, paragraphs failing self-checks.
5. External DTDs/entities never fetched; expat ≥ 2.4 amplification protection; whole file in memory.

## 10. Evidence status

**After S2 (2026-11-21), covered by the committed pytest suite (T):** encoding pre-check, atom mapping, nested
ownership and barriers, tiling/cross-check vetoes, existing-`<error>` skip (S1); field validation, span safety
X1–X8/X10 with D4 off (including all D4-off rejections and X5 CDATA containment), X11 overlap handling, rendering
with attribute escaping in both encodings, the independent verifier (G1, V1–V6, namespace, canonical tags,
encoding classification), X12 rollback and fail-closed behaviour, golden AC-03 (a) and M1–M13 (S2).

**After S3 (2026-11-21), also covered (T):** quotation classification, pairing, guard and Q-01…Q-21 (§6), nested
quotation propagation (§6.3), the finding contract and its validation, cluster-aware matching tiers, anchors and word
tie-breaks, quotation span and correction checks, document resolution and tracing (03 §3). **Still unverified
(S4+):** prompt and LLM behaviour, anchor-mismatch / ambiguity rates on real model output, cache hits, cost, the CLI
(including Windows file writing), lxml claims. D15 priority is deliberately not implemented (04). The table below is
the pre-implementation evidence record and is kept for history.

| Demonstrated (P-spike / P-adhoc) | Still unverified (SPEC) |
|---|---|
| **P-spike:** atom offsets for LIT/REF/NL/CDATA/comment/PI/BOM; golden reproduction (local files only); expat text == ElementTree on the local example and sample files; the D4-on column of worked examples M1–M13 (except the SPEC cells); start/end/two-level expansion and comment/PI/`<br/>` enclosure (D4-on path only); CRLF in span; one-pass multi-insert incl. adjacent spans; US-ASCII char refs; attribute escaping round-trip; default-namespace `<p>`; endpoint-in-CDATA, partial-crossing and entity-split rejections. **P-adhoc:** nested ownership + barrier hole; recursive verifier V1–V6; expat encoding behaviour | D4-off rejections (no switch in the spike); CDATA containment rejection (X5); cluster tokenizer; quote detection; resolver (anchors, NFC, whitespace gaps, overlap policy); encoding pre-check; existing-`<error>` skip; tiling self-check as a gate; rollback; LLM behaviour; cache hits; cost; Windows file writing; large files; lxml claims |

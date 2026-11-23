"""Evaluation utility (AC-15): compare predicted <error> annotations with expected ones on the same document.

    python scripts/evaluate.py --original IN.xml --expected GOLD.xml --predicted OUT.xml [--json REPORT.json]
                               [--details]

Not part of the proofreader itself; no model calls, no acceptance threshold (the report only measures).

Method
------
1. Each file is parsed; its <p> elements are taken in document order (the proofreader's paragraph ids) and each
   paragraph's OWN text (nested <p> excluded, <error> markup removed) is read with the independent verifier's
   reader. The expected and predicted files must have exactly the original's paragraphs and own texts - otherwise
   they are not annotations of the same document and the run stops (exit 1).
2. Every <error> becomes an annotation: (paragraph, start, end, type, correction) with character offsets into
   that paragraph's own text, so multi-word spans are compared by their exact extent.
3. Annotations are matched at four levels:
   * span   - same paragraph and identical [start, end)
   * type   - span and the same type
   * strict - span, type and the same correction
   * overlap - same paragraph and overlapping spans, paired one-to-one by largest overlap (ties: earliest start),
               so a partly overlapping annotation can match at most one counterpart
   Exact levels count duplicates as a multiset. For each level: true positives, false positives, false negatives,
   precision = TP / predicted, recall = TP / expected, F1; "n/a" when a denominator is zero.
4. Per category (type level): recall by expected type, precision by predicted type.
5. Reasons are free text and are not scored; the report gives the share of predicted styleguide errors that
   carry a non-empty reason (required by the brief).

Matching one editor's example output measures agreement with that editor, not editorial quality as such:
equally valid corrections, different span choices or extra true errors all count as mismatches. Read the
numbers together with a manual review.

Output never contains document text: counts and, with --details, paragraph ids, offsets and types only.
Exit codes: 0 report produced; 1 inputs unreadable or not annotations of the same document; 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from proofreader.verify import _own_text_and_errors  # the verifier's independent reader (same as golden (b))

NA = "n/a"
LEVELS = ("span", "type", "strict", "overlap")


class EvaluationError(Exception):
    """The inputs cannot be compared (unreadable, malformed, or not the same document)."""


@dataclass(frozen=True)
class Ann:
    paragraph: int
    start: int
    end: int
    type: str
    correction: str
    reason: str | None


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def read(path: Path) -> tuple[list[str], list[Ann]]:
    """Own text of every <p> (document order) and every <error> as an annotation."""
    try:
        root = ET.fromstring(Path(path).read_bytes())
    except (OSError, ET.ParseError) as exc:
        raise EvaluationError(f"{Path(path).name}: cannot be read as XML ({type(exc).__name__})") from None
    texts: list[str] = []
    anns: list[Ann] = []
    for pid, para in enumerate(e for e in root.iter() if _local(e.tag) == "p"):
        text, errors = _own_text_and_errors(para)
        texts.append(text)
        anns += [Ann(pid, s, e, el.get("type") or "", el.get("correction") or "", el.get("reason"))
                 for s, e, el in errors]
    return texts, anns


def _ratio(num: int, den: int) -> float | str:
    return round(num / den, 4) if den else NA


def _scores(tp: int, n_pred: int, n_exp: int) -> dict:
    p, r = _ratio(tp, n_pred), _ratio(tp, n_exp)
    f1 = NA if NA in (p, r) or (p + r) == 0 else round(2 * p * r / (p + r), 4)  # type: ignore[operator]
    return {"tp": tp, "fp": n_pred - tp, "fn": n_exp - tp, "precision": p, "recall": r, "f1": f1}


KEYS = {
    "span": lambda a: (a.paragraph, a.start, a.end),
    "type": lambda a: (a.paragraph, a.start, a.end, a.type),
    "strict": lambda a: (a.paragraph, a.start, a.end, a.type, a.correction),
}


def match_exact(expected: list[Ann], predicted: list[Ann], level: str) -> tuple[list[Ann], list[Ann], list[Ann]]:
    """(matched predicted, unmatched expected, unmatched predicted) as multisets of the level's key."""
    key = KEYS[level]
    budget = Counter(key(a) for a in expected)
    matched, extra = [], []
    for a in predicted:
        if budget[key(a)] > 0:
            budget[key(a)] -= 1
            matched.append(a)
        else:
            extra.append(a)
    left = Counter(key(a) for a in matched)
    missing = []
    for a in expected:
        if left[key(a)] > 0:
            left[key(a)] -= 1
        else:
            missing.append(a)
    return matched, missing, extra


def match_overlap(expected: list[Ann], predicted: list[Ann]) -> tuple[list[tuple[Ann, Ann]], list[Ann], list[Ann]]:
    """One-to-one pairs of overlapping spans in the same paragraph, largest overlap first."""
    pairs = []
    for i, e in enumerate(expected):
        for j, p in enumerate(predicted):
            if e.paragraph == p.paragraph:
                overlap = min(e.end, p.end) - max(e.start, p.start)
                if overlap > 0:
                    pairs.append((-overlap, e.start, p.start, i, j))
    used_e, used_p, matched = set(), set(), []
    for _, _, _, i, j in sorted(pairs):
        if i not in used_e and j not in used_p:
            used_e.add(i)
            used_p.add(j)
            matched.append((expected[i], predicted[j]))
    return (matched, [e for i, e in enumerate(expected) if i not in used_e],
            [p for j, p in enumerate(predicted) if j not in used_p])


def evaluate(original: Path, expected: Path, predicted: Path) -> dict:
    texts, original_anns = read(original)
    exp_texts, exp = read(expected)
    pred_texts, pred = read(predicted)
    for name, other in ((Path(expected).name, exp_texts), (Path(predicted).name, pred_texts)):
        if len(other) != len(texts):
            raise EvaluationError(f"{name}: {len(other)} paragraphs, the original has {len(texts)}")
        bad = [i for i, (a, b) in enumerate(zip(texts, other)) if a != b]
        if bad:
            raise EvaluationError(f"{name}: paragraph text differs from the original in paragraph(s) {bad[:10]}")
    # annotations already present in the original are not part of either side's contribution
    exp = [a for a in exp if a not in original_anns]
    pred = [a for a in pred if a not in original_anns]
    report: dict = {"paragraphs": len(texts), "expected": len(exp), "predicted": len(pred), "levels": {},
                    "unmatched": {}}
    for level in ("span", "type", "strict"):
        matched, missing, extra = match_exact(exp, pred, level)
        report["levels"][level] = _scores(len(matched), len(pred), len(exp))
        report["unmatched"][level] = {"expected": missing, "predicted": extra}
    pairs, missing, extra = match_overlap(exp, pred)
    report["levels"]["overlap"] = _scores(len(pairs), len(pred), len(exp))
    report["unmatched"]["overlap"] = {"expected": missing, "predicted": extra}
    matched_type, _, _ = match_exact(exp, pred, "type")
    categories = sorted({a.type for a in exp} | {a.type for a in pred})
    report["by_type"] = {
        t: {"expected": sum(a.type == t for a in exp), "predicted": sum(a.type == t for a in pred),
            "matched": sum(a.type == t for a in matched_type),
            "precision": _ratio(sum(a.type == t for a in matched_type), sum(a.type == t for a in pred)),
            "recall": _ratio(sum(a.type == t for a in matched_type), sum(a.type == t for a in exp))}
        for t in categories}
    styled = [a for a in pred if a.type == "styleguide"]
    report["styleguide_reason_coverage"] = _ratio(sum(bool(a.reason and a.reason.strip()) for a in styled),
                                                  len(styled))
    return report


def _public(report: dict, details: bool) -> dict:
    out = {k: v for k, v in report.items() if k != "unmatched"}
    if details:
        out["unmatched"] = {level: {side: [{"paragraph": a.paragraph, "start": a.start, "end": a.end, "type": a.type}
                                           for a in anns] for side, anns in sides.items()}
                            for level, sides in report["unmatched"].items()}
    return out


def render(report: dict) -> str:
    lines = [f"paragraphs {report['paragraphs']}; expected annotations {report['expected']}; "
             f"predicted annotations {report['predicted']}",
             f"{'level':8} {'tp':>4} {'fp':>4} {'fn':>4} {'precision':>10} {'recall':>8} {'f1':>8}"]
    for level in LEVELS:
        s = report["levels"][level]
        lines.append(f"{level:8} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} {s['precision']!s:>10} {s['recall']!s:>8} "
                     f"{s['f1']!s:>8}")
    lines.append("by type (type level): " + ", ".join(
        f"{t} p={v['precision']} r={v['recall']} ({v['matched']}/{v['expected']} expected, {v['predicted']} predicted)"
        for t, v in report["by_type"].items()))
    lines.append(f"styleguide reason coverage: {report['styleguide_reason_coverage']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare predicted <error> annotations with expected ones.")
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--predicted", required=True, type=Path)
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    parser.add_argument("--details", action="store_true", help="list unmatched annotations (ids and offsets only)")
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.original, args.expected, args.predicted)
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    public = _public(report, args.details)
    print(render(report))
    if args.details:
        print(json.dumps(public["unmatched"], indent=2))
    if args.json is not None:
        args.json.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

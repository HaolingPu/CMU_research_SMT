#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


NONTERMINAL_UPPER_RE = re.compile(
    r"(?P<context>\b[A-Za-z][A-Za-z']*[,:;]\s+(?P<word>[A-Z][a-z]{1,})\b)"
)
REPEATED_WORD_RE = re.compile(r"\b(?P<word>[A-Za-z][A-Za-z']*)\s+(?P=word)\b", re.IGNORECASE)
LOWERCASE_I_RE = re.compile(r"(?<![A-Za-z])i(?![A-Za-z])")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+[,.!?;:]")
NO_SPACE_AFTER_PUNCT_RE = re.compile(r"[,.!?;:][A-Za-z]")
ARCHAIC_INVERSION_RE = re.compile(
    r"\b(?P<pron>me|him|her|us|them|you)\s+"
    r"(?P<aux>shouldst|wouldst|couldst|wilt|shalt|dost|didst|hast|hadst|art|wert)\s+thou\b",
    re.IGNORECASE,
)

PROBABLY_OK_AFTER_COMMA = {
    "I",
    "Mr",
    "Mrs",
    "Ms",
    "Dr",
    "St",
    "Sir",
    "Lady",
    "Lord",
}


def snippet(text: str, start: int, end: int, width: int = 90) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    return text[left:right].replace("\t", " ").replace("\n", " ")


def add_issue(
    issues: List[Dict[str, str]],
    issue_type: str,
    text: str,
    start: int,
    end: int,
    detail: str,
) -> None:
    issues.append(
        {
            "type": issue_type,
            "detail": detail,
            "snippet": snippet(text, start, end),
        }
    )


def audit_text(text: str) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    if not text:
        return issues

    for m in NONTERMINAL_UPPER_RE.finditer(text):
        word = m.group("word")
        if word in PROBABLY_OK_AFTER_COMMA:
            continue
        add_issue(
            issues,
            "uppercase_after_comma_semicolon_or_colon",
            text,
            m.start(),
            m.end(),
            m.group("context"),
        )

    for m in REPEATED_WORD_RE.finditer(text):
        word = m.group("word")
        if word.lower() in {"had", "that"}:
            # These can be legitimate across clauses often enough to be noisy.
            continue
        add_issue(issues, "repeated_word", text, m.start(), m.end(), m.group(0))

    for m in ARCHAIC_INVERSION_RE.finditer(text):
        add_issue(issues, "possible_missing_comma_before_archaic_inversion", text, m.start(), m.end(), m.group(0))

    for m in LOWERCASE_I_RE.finditer(text):
        add_issue(issues, "lowercase_standalone_i", text, m.start(), m.end(), m.group(0))

    for m in SPACE_BEFORE_PUNCT_RE.finditer(text):
        add_issue(issues, "space_before_punctuation", text, m.start(), m.end(), m.group(0))

    for m in NO_SPACE_AFTER_PUNCT_RE.finditer(text):
        add_issue(issues, "no_space_after_punctuation", text, m.start(), m.end(), m.group(0))

    stripped = text.rstrip()
    if stripped and stripped[-1] not in ".!?\"'”’":
        add_issue(issues, "missing_final_punctuation", text, max(0, len(text) - 1), len(text), stripped[-20:])

    return issues


def parse_src_text_full(value: str) -> List[str]:
    value = (value or "").strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        try:
            import ast

            parsed = ast.literal_eval(value)
        except Exception:
            return []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return []


def audit_manifest(input_tsv: Path, output_dir: Path, max_examples_per_type: int) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    suspects_tsv = output_dir / "asr_source_text_suspects.tsv"
    summary_json = output_dir / "asr_source_text_audit_summary.json"
    examples_json = output_dir / "asr_source_text_audit_examples.json"

    total = 0
    suspect_rows = 0
    issue_counts: Counter[str] = Counter()
    example_by_type: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    with input_tsv.open("r", encoding="utf-8") as fin, suspects_tsv.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(
            fout,
            delimiter="\t",
            fieldnames=["id", "issue_types", "num_issues", "details", "src_text", "src_text_full"],
        )
        writer.writeheader()

        for row in reader:
            total += 1
            uid = row.get("id", "").strip()
            text = row.get("src_text", "").strip()
            issues = audit_text(text)

            # Also catch bad segment boundaries in src_text_full, where a segment
            # starts with uppercase after the previous segment ended with comma.
            segments = parse_src_text_full(row.get("src_text_full", ""))
            for i in range(1, len(segments)):
                prev = segments[i - 1].rstrip()
                cur = segments[i].lstrip()
                if prev.endswith((",", ";", ":")) and re.match(r"[A-Z][a-z]{1,}\b", cur):
                    add_issue(
                        issues,
                        "uppercase_segment_after_nonterminal_punctuation",
                        text,
                        0,
                        min(len(text), 120),
                        f"seg{i-1} ends {prev[-30:]!r}; seg{i} starts {cur[:30]!r}",
                    )

            if not issues:
                continue

            suspect_rows += 1
            type_order = []
            for issue in issues:
                issue_type = issue["type"]
                issue_counts[issue_type] += 1
                if issue_type not in type_order:
                    type_order.append(issue_type)
                if len(example_by_type[issue_type]) < max_examples_per_type:
                    example_by_type[issue_type].append(
                        {
                            "id": uid,
                            "detail": issue["detail"],
                            "snippet": issue["snippet"],
                            "src_text": text,
                        }
                    )

            writer.writerow(
                {
                    "id": uid,
                    "issue_types": ",".join(type_order),
                    "num_issues": len(issues),
                    "details": json.dumps(issues, ensure_ascii=False),
                    "src_text": text,
                    "src_text_full": row.get("src_text_full", ""),
                }
            )

    summary = {
        "input_tsv": str(input_tsv),
        "total_rows": total,
        "suspect_rows": suspect_rows,
        "suspect_pct": (suspect_rows / total * 100.0) if total else 0.0,
        "issue_counts": dict(issue_counts.most_common()),
        "suspects_tsv": str(suspects_tsv),
        "examples_json": str(examples_json),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    examples_json.write_text(json.dumps(example_by_type, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristic audit for suspicious ASR/source text in GigaSpeech TSV.")
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples-per-type", type=int, default=50)
    args = parser.parse_args()

    summary = audit_manifest(Path(args.input_tsv), Path(args.output_dir), args.max_examples_per_type)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

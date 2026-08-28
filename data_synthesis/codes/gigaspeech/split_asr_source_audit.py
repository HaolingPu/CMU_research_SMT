#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set


WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")
ALL_CAPS_RE = re.compile(r"^[A-Z']+$")


def load_suspects(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as fin:
        return list(csv.DictReader(fin, delimiter="\t"))


def parse_details(row: Dict[str, str]) -> List[Dict[str, str]]:
    try:
        data = json.loads(row.get("details", "[]"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def write_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def issue_rows(rows: List[Dict[str, str]], issue_types: Set[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        matched = [d for d in parse_details(row) if d.get("type") in issue_types]
        if not matched:
            continue
        out.append(
            {
                "id": row["id"],
                "issue_types": ",".join(sorted({m.get("type", "") for m in matched})),
                "num_issues": str(len(matched)),
                "details": json.dumps(matched, ensure_ascii=False),
                "src_text": row["src_text"],
                "src_text_full": row["src_text_full"],
            }
        )
    return out


def collect_word_counts(manifest_tsv: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with manifest_tsv.open("r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        for row in reader:
            for word in WORD_RE.findall(row.get("src_text", "")):
                w = word.strip("'")
                if w:
                    counts[w.lower()] += 1
    return counts


def aspell_unknown(words: Iterable[str]) -> Set[str]:
    unique = sorted(set(words))
    if not unique:
        return set()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        for word in unique:
            tmp.write(word + "\n")
    try:
        proc = subprocess.run(
            ["aspell", "list"],
            input=tmp_path.read_text(encoding="utf-8"),
            text=True,
            check=True,
            stdout=subprocess.PIPE,
        )
        return {line.strip().lower() for line in proc.stdout.splitlines() if line.strip()}
    finally:
        tmp_path.unlink(missing_ok=True)


def likely_spelling_rows(manifest_tsv: Path, out_path: Path, min_len: int, max_count: int) -> Dict[str, int]:
    word_counts = collect_word_counts(manifest_tsv)
    candidates = []
    for word, count in word_counts.items():
        if len(word) < min_len:
            continue
        if count > max_count:
            continue
        if "'" in word:
            continue
        if any(ch.isdigit() for ch in word):
            continue
        candidates.append(word)

    unknown = aspell_unknown(candidates)
    unknown_counts = Counter({word: word_counts[word] for word in unknown})

    examples: Dict[str, List[str]] = {word: [] for word in unknown_counts}
    with manifest_tsv.open("r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        for row in reader:
            text = row.get("src_text", "")
            lowered = {w.strip("'").lower() for w in WORD_RE.findall(text)}
            hit_words = lowered & unknown
            for word in hit_words:
                if len(examples[word]) < 3:
                    examples[word].append(row.get("id", ""))

    with out_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            delimiter="\t",
            fieldnames=["word", "count", "example_ids"],
        )
        writer.writeheader()
        for word, count in unknown_counts.most_common():
            writer.writerow({"word": word, "count": count, "example_ids": ",".join(examples[word])})

    return {"candidate_words": len(candidates), "unknown_words": len(unknown_counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Split ASR source audit into human-friendly TSVs.")
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--manifest-tsv", required=True)
    parser.add_argument("--min-spell-len", type=int, default=6)
    parser.add_argument("--max-spell-count", type=int, default=50)
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir)
    manifest_tsv = Path(args.manifest_tsv)
    suspects_path = audit_dir / "asr_source_text_suspects.tsv"
    out_dir = audit_dir / "split"
    rows = load_suspects(suspects_path)

    common_fields = ["id", "issue_types", "num_issues", "details", "src_text", "src_text_full"]
    files = {
        "case_boundary.tsv": issue_rows(
            rows,
            {
                "uppercase_after_comma_semicolon_or_colon",
                "uppercase_segment_after_nonterminal_punctuation",
            },
        ),
        "repeated_words.tsv": issue_rows(rows, {"repeated_word"}),
        "punctuation_spacing.tsv": issue_rows(
            rows,
            {"space_before_punctuation", "no_space_after_punctuation", "missing_final_punctuation"},
        ),
        "lowercase_i.tsv": issue_rows(rows, {"lowercase_standalone_i"}),
        "archaic_inversion.tsv": issue_rows(rows, {"possible_missing_comma_before_archaic_inversion"}),
    }

    summary: Dict[str, object] = {}
    for name, split_rows in files.items():
        write_rows(out_dir / name, split_rows, common_fields)
        summary[name] = len(split_rows)

    spell_stats = likely_spelling_rows(
        manifest_tsv,
        out_dir / "possible_misspellings_words.tsv",
        args.min_spell_len,
        args.max_spell_count,
    )
    summary["possible_misspellings_words.tsv"] = spell_stats

    (out_dir / "README.txt").write_text(
        "\n".join(
            [
                "Split ASR/source audit outputs.",
                "",
                "case_boundary.tsv: comma/semicolon/colon followed by uppercase, including segment-boundary variants. High recall but noisy.",
                "repeated_words.tsv: repeated adjacent tokens such as 'it it' or 'false false'. Some are legitimate emphasis or boundary artifacts.",
                "punctuation_spacing.tsv: missing final punctuation, space before punctuation, or no space after punctuation.",
                "lowercase_i.tsv: standalone lowercase 'i'.",
                "archaic_inversion.tsv: patterns like 'me shouldst thou', often missing comma before an archaic clause.",
                "possible_misspellings_words.tsv: word-level aspell unknowns, filtered to low-frequency words. This is a candidate list, not row-level proof.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

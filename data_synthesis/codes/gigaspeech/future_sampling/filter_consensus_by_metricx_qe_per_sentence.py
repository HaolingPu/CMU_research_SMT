#!/usr/bin/env python3
"""Filter consensus JSON outputs by per-sentence MetricX QE.

Expects the per-sentence MetricX output (produced from the input jsonl emitted
by `convert_metricx_consensus_per_sentence.py`). Groups scores by utt_id; an
utterance is kept only if ALL its sentences have QE <= threshold AND no
sentences are missing a score. If any sentence fails, the whole instance is
dropped.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def iter_metricx_rows(path: Path) -> Iterable[Tuple[str, int, int, Optional[float], Path]]:
    """Yield (utt_id, sentence_idx, total_sentences, score_or_None, stream_json)."""
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                continue

            meta = obj.get("metadata") or {}
            utt_id = str(meta.get("utt_id", "")).strip()
            try:
                sentence_idx = int(meta.get("sentence_idx"))
                total_sentences = int(meta.get("total_sentences"))
            except (TypeError, ValueError):
                continue
            stream_json = Path(str(meta.get("stream_json", "")).strip())
            if not utt_id or not stream_json:
                continue

            pred = obj.get("prediction")
            score: Optional[float]
            try:
                score = float(pred)
                if math.isnan(score):
                    score = None
            except (TypeError, ValueError):
                score = None

            yield utt_id, sentence_idx, total_sentences, score, stream_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep consensus JSON outputs whose every sentence has MetricX QE <= threshold."
    )
    parser.add_argument("--metricx-output", required=True, help="Path to per-sentence metricx_output.jsonl")
    parser.add_argument("--output-dir", required=True, help="Destination directory for kept JSON files")
    parser.add_argument("--threshold", type=float, default=3.0, help="Keep if every sentence QE <= threshold")
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing JSON files in output-dir before copying",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path to write a per-utterance report (jsonl).",
    )
    args = parser.parse_args()

    metricx_output = Path(args.metricx_output).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_output:
        for path in output_dir.glob("*.json"):
            path.unlink()

    # utt_id -> list of (sentence_idx, score_or_None), expected_total, stream_json
    per_utt: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"rows": [], "total": None, "stream_json": None}
    )

    total_rows = 0
    for utt_id, s_idx, total, score, stream_json in iter_metricx_rows(metricx_output):
        total_rows += 1
        rec = per_utt[utt_id]
        rec["rows"].append((s_idx, score))
        if rec["total"] is None:
            rec["total"] = total
        rec["stream_json"] = stream_json

    kept = 0
    dropped_missing = 0
    dropped_fail = 0
    missing_json = 0
    report_rows: List[Dict[str, Any]] = []

    for utt_id, rec in per_utt.items():
        rows = rec["rows"]
        total = rec["total"] or 0
        stream_json: Path = rec["stream_json"]

        seen_idxs = {s for s, _ in rows}
        expected = set(range(total))
        is_complete = expected.issubset(seen_idxs)
        all_scored = all(score is not None for _, score in rows) and is_complete
        worst = None
        if all_scored and rows:
            worst = max(score for _, score in rows if score is not None)

        passed = bool(all_scored and worst is not None and worst <= args.threshold)

        if args.report is not None:
            report_rows.append({
                "utt_id": utt_id,
                "total_expected": total,
                "num_rows_seen": len(rows),
                "is_complete": is_complete,
                "all_scored": all_scored,
                "worst_score": worst,
                "passed": passed,
                "per_sentence": sorted(
                    [{"sentence_idx": s, "score": sc} for s, sc in rows],
                    key=lambda x: x["sentence_idx"],
                ),
            })

        if not passed:
            if not all_scored:
                dropped_missing += 1
            else:
                dropped_fail += 1
            continue

        if stream_json is None or not stream_json.is_file():
            missing_json += 1
            continue
        dst = output_dir / stream_json.name
        shutil.copy2(stream_json, dst)
        kept += 1

    if args.report is not None:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as frep:
            for rr in report_rows:
                frep.write(json.dumps(rr, ensure_ascii=False) + "\n")

    print("===== Filter Consensus by Per-Sentence MetricX QE =====")
    print(f"MetricX output    : {metricx_output}")
    print(f"Output dir        : {output_dir}")
    print(f"Threshold (<=)    : {args.threshold}")
    print(f"Total rows read   : {total_rows}")
    print(f"Utterances seen   : {len(per_utt)}")
    print(f"Kept              : {kept}")
    print(f"Dropped (any QE>thr): {dropped_fail}")
    print(f"Dropped (missing) : {dropped_missing}")
    print(f"Missing src JSON  : {missing_json}")
    if args.report is not None:
        print(f"Report            : {args.report}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Custom finalize for sub-sentence (per-input-sentence) MetricX QE on EAST data.

Reads metricx_output.jsonl (sub-sentence rows produced by SegAlign+QE on
fake-consensus EAST JSONs whose doc_id is "<utt>__<latency>"). Groups scores
by (utt, latency); a (utt, latency) pair is kept iff every sub-sentence has
QE <= threshold. Writes a metricx_filtered.jsonl in the format expected by
final_output_gigaspeech.py (one row per kept (utt, latency)) so the existing
final-output script can build streaming jsonl files.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metricx-output", required=True,
                    help="Per-sub-sentence metricx_output.jsonl")
    ap.add_argument("--consensus-format-root", required=True,
                    help="Dir containing job_east_<lang>/task_0/<doc>.json with east_stream_json field")
    ap.add_argument("--filtered-output", required=True,
                    help="Output metricx_filtered.jsonl path")
    ap.add_argument("--threshold", type=float, default=3.0,
                    help="Single threshold (used when per-latency thresholds are not set)")
    ap.add_argument("--threshold-low", type=float, default=None)
    ap.add_argument("--threshold-medium", type=float, default=None)
    ap.add_argument("--threshold-high", type=float, default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    per_lat_thr = {}
    if any(x is not None for x in (args.threshold_low, args.threshold_medium, args.threshold_high)):
        per_lat_thr = {
            "low":    args.threshold_low    if args.threshold_low    is not None else args.threshold,
            "medium": args.threshold_medium if args.threshold_medium is not None else args.threshold,
            "high":   args.threshold_high   if args.threshold_high   is not None else args.threshold,
        }
        print(f"[finalize] per-latency thresholds: {per_lat_thr}")
    else:
        print(f"[finalize] single threshold: {args.threshold}")

    # Index consensus-format JSONs to recover east_stream_json per doc_id.
    cons_idx = {}
    for fpath in Path(args.consensus_format_root).rglob("*.json"):
        try:
            d = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        doc = d.get("utt_id")
        if doc:
            cons_idx[doc] = d

    # Group scores by doc_id (= utt__lat).
    per_doc = defaultdict(list)  # doc_id -> list of (sentence_idx, score_or_None)
    per_doc_total = {}
    for line in open(args.metricx_output, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        meta = row.get("metadata") or {}
        doc = meta.get("utt_id")
        if not doc:
            continue
        try:
            s_idx = int(meta.get("sentence_idx", -1))
            total = int(meta.get("total_sentences", -1))
        except (TypeError, ValueError):
            continue
        pred = row.get("prediction")
        try:
            score = float(pred)
            if math.isnan(score):
                score = None
        except (TypeError, ValueError):
            score = None
        per_doc[doc].append((s_idx, score))
        per_doc_total[doc] = total

    # Decide kept docs.
    out_path = Path(args.filtered_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped_score = 0
    dropped_missing = 0
    dropped_no_meta = 0
    report_rows = []

    with out_path.open("w", encoding="utf-8") as fout:
        for doc, rows in per_doc.items():
            cm = cons_idx.get(doc)
            if cm is None:
                dropped_no_meta += 1
                continue
            total = per_doc_total[doc]
            scores = [s for (_idx, s) in rows]
            missing = [s for s in scores if s is None] or (len(scores) < total)
            if missing:
                dropped_missing += 1
                report_rows.append({"doc_id": doc, "kept": False, "reason": "missing_score",
                                     "n_scored": len(scores), "expected": total})
                continue
            max_score = max(scores)
            lat = cm.get("east_latency", "")
            utt = cm.get("east_utt_id", "")
            stream = cm.get("east_stream_json", "")
            thr = per_lat_thr.get(lat, args.threshold) if per_lat_thr else args.threshold
            if max_score > thr:
                dropped_score += 1
                report_rows.append({"doc_id": doc, "kept": False, "reason": "score>thr",
                                     "max_score": max_score, "threshold": thr, "latency": lat})
                continue
            kept += 1
            # Write a row in the schema final_output_gigaspeech.py expects.
            out_row = {
                "source": "",
                "hypothesis": "",
                "reference": "",
                "metadata": {
                    "utt_id": utt,
                    "latency": lat,
                    "stream_json": stream,
                    "max_subsentence_score": max_score,
                    "n_subsentences": total,
                },
                "prediction": max_score,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            report_rows.append({"doc_id": doc, "kept": True,
                                 "max_score": max_score, "n_subsentences": total})

    print(f"[finalize] docs seen        : {len(per_doc)}")
    print(f"[finalize] kept             : {kept}")
    print(f"[finalize] dropped (score>thr): {dropped_score}")
    print(f"[finalize] dropped (missing): {dropped_missing}")
    print(f"[finalize] dropped (no_meta): {dropped_no_meta}")
    print(f"[finalize] threshold        : {args.threshold}")
    print(f"[finalize] -> {out_path}")

    if args.report:
        rep = Path(args.report)
        rep.parent.mkdir(parents=True, exist_ok=True)
        with rep.open("w", encoding="utf-8") as fr:
            for r in report_rows:
                fr.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[finalize] report           : {rep}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List


def _get_path_value(data: Dict[str, Any], path: List[str], default: Any = 0) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GPT future-sampler token usage and cost per case.")
    parser.add_argument("paths", nargs="+", help="JSON files or directories containing per-utterance JSON files.")
    parser.add_argument("--csv", default="", help="Optional CSV output path.")
    args = parser.parse_args()

    files: List[str] = []
    for path in args.paths:
        if os.path.isdir(path):
            files.extend(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
        else:
            files.extend(glob.glob(path))
    files = sorted(set(p for p in files if os.path.isfile(p)))
    if not files:
        raise SystemExit("No JSON files found.")

    rows: List[Dict[str, Any]] = []
    total = {
        "calls": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "billable_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "visible_output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }
    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        summary = data.get("gpt_sampler_usage", {}) or {}
        if "summary" in summary:
            summary = summary.get("summary", {}) or {}
        if not summary:
            continue
        cost = _get_path_value(summary, ["cost_usd", "total"], 0.0)
        row = {
            "file": path,
            "utt_id": data.get("utt_id", os.path.splitext(os.path.basename(path))[0]),
            "calls": int(summary.get("calls", 0) or 0),
            "input_tokens": int(summary.get("input_tokens", 0) or 0),
            "cached_input_tokens": int(summary.get("cached_input_tokens", 0) or 0),
            "billable_input_tokens": int(summary.get("billable_input_tokens", 0) or 0),
            "output_tokens": int(summary.get("output_tokens", 0) or 0),
            "reasoning_tokens": int(summary.get("reasoning_tokens", 0) or 0),
            "visible_output_tokens": int(summary.get("visible_output_tokens", 0) or 0),
            "total_tokens": int(summary.get("total_tokens", 0) or 0),
            "cost_usd": float(cost or 0.0),
        }
        rows.append(row)
        for key in total:
            total[key] += row[key]

    fieldnames = [
        "utt_id", "calls", "input_tokens", "cached_input_tokens", "billable_input_tokens",
        "output_tokens", "reasoning_tokens", "visible_output_tokens", "total_tokens", "cost_usd", "file",
    ]
    out = sys.stdout
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        printable = dict(row)
        printable["cost_usd"] = f"{row['cost_usd']:.8f}"
        writer.writerow(printable)
    print(
        f"# TOTAL cases={len(rows)} calls={total['calls']} input={total['input_tokens']} "
        f"cached_input={total['cached_input_tokens']} output={total['output_tokens']} "
        f"reasoning={total['reasoning_tokens']} visible_output={total['visible_output_tokens']} "
        f"cost_usd={total['cost_usd']:.8f}",
        file=sys.stderr,
    )

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


if __name__ == "__main__":
    main()

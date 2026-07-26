#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from compare_metricx_qe import fmt_stats, load_metricx_scores, summarize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize MetricX QE scores for one experiment directory or JSONL file."
    )
    parser.add_argument("--input", required=True, help="Experiment dir, metricx_output.jsonl, or metricx_shards dir root.")
    parser.add_argument(
        "--latency",
        default="future_sampling",
        help="Latency slice to summarize. Default: future_sampling",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Also report the fraction with prediction <= threshold.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to dump the summary as JSON.",
    )
    args = parser.parse_args()

    scores, info = load_metricx_scores(args.input, args.latency)
    values = list(scores.values())
    stats = summarize(values)
    threshold_count = sum(value <= args.threshold for value in values)
    threshold_ratio = (threshold_count / len(values)) if values else float("nan")

    summary = {
        "input": args.input,
        "latency": args.latency,
        "threshold": args.threshold,
        "info": info,
        "stats": stats,
        "threshold_count": threshold_count,
        "threshold_ratio": threshold_ratio,
    }

    print(f"Input   : {args.input}")
    print(f"Latency : {args.latency}")
    print(fmt_stats("MetricX QE", stats))
    if values:
        print(
            f"Threshold <= {args.threshold:.2f}: {threshold_count}/{len(values)} "
            f"({threshold_ratio:.2%})"
        )
    else:
        print("No valid QE scores found after filtering.")

    if info["duplicates"]:
        print(f"Duplicate utt_ids (last one kept): {info['duplicates'][:10]}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fout:
            json.dump(summary, fout, ensure_ascii=False, indent=2)
        print(f"Wrote JSON summary to: {args.output_json}")


if __name__ == "__main__":
    main()

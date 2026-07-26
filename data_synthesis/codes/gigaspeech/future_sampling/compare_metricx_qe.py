#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _resolve_metricx_files(path_str: str) -> List[Path]:
    path = Path(path_str)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    merged = path / "metricx_output.jsonl"
    if merged.is_file():
        return [merged]

    shard_dir = path / "metricx_shards"
    shard_files = sorted(shard_dir.glob("output_*.jsonl")) if shard_dir.is_dir() else []
    if shard_files:
        return shard_files

    raise FileNotFoundError(
        f"Could not find metricx_output.jsonl or metricx_shards/output_*.jsonl under: {path}"
    )


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_metricx_scores(path_str: str, latency: str) -> Tuple[Dict[str, float], Dict[str, object]]:
    files = _resolve_metricx_files(path_str)
    scores: Dict[str, float] = {}
    duplicates: List[str] = []
    total_lines = 0
    kept_lines = 0
    skipped_latency = 0
    skipped_bad = 0

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Bad JSON in {file_path}:{line_no}: {exc}") from exc

                meta = obj.get("metadata") or {}
                if latency and str(meta.get("latency", "")) != latency:
                    skipped_latency += 1
                    continue

                utt_id = str(meta.get("utt_id", "")).strip()
                score = _safe_float(obj.get("prediction"))
                if not utt_id or score is None or math.isnan(score):
                    skipped_bad += 1
                    continue

                if utt_id in scores:
                    duplicates.append(utt_id)
                scores[utt_id] = score
                kept_lines += 1

    info = {
        "path": path_str,
        "files": [str(p) for p in files],
        "total_lines": total_lines,
        "kept_lines": kept_lines,
        "skipped_latency": skipped_latency,
        "skipped_bad": skipped_bad,
        "duplicates": sorted(set(duplicates)),
    }
    return scores, info


def summarize(values: Iterable[float]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "stdev": float("nan"),
        }
    return {
        "count": len(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def fmt_stats(label: str, stats: Dict[str, float]) -> str:
    if not stats["count"]:
        return f"{label}: count=0"
    return (
        f"{label}: count={stats['count']} mean={stats['mean']:.4f} "
        f"median={stats['median']:.4f} min={stats['min']:.4f} "
        f"max={stats['max']:.4f} stdev={stats['stdev']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MetricX QE outputs for two experiment directories."
    )
    parser.add_argument("--baseline", required=True, help="Baseline dir or metricx_output.jsonl file.")
    parser.add_argument("--candidate", required=True, help="Candidate dir or metricx_output.jsonl file.")
    parser.add_argument(
        "--latency",
        default="future_sampling",
        help="Latency slice to compare. Default: future_sampling",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Report the fraction with prediction <= threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many best/worst utterances to print.",
    )
    parser.add_argument(
        "--tie-eps",
        type=float,
        default=1e-6,
        help="Absolute tolerance for tie decisions.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to dump the summary as JSON.",
    )
    args = parser.parse_args()

    baseline_scores, baseline_info = load_metricx_scores(args.baseline, args.latency)
    candidate_scores, candidate_info = load_metricx_scores(args.candidate, args.latency)

    baseline_ids = set(baseline_scores)
    candidate_ids = set(candidate_scores)
    common_ids = sorted(baseline_ids & candidate_ids)
    baseline_only = sorted(baseline_ids - candidate_ids)
    candidate_only = sorted(candidate_ids - baseline_ids)

    baseline_common = [baseline_scores[utt_id] for utt_id in common_ids]
    candidate_common = [candidate_scores[utt_id] for utt_id in common_ids]
    deltas = [candidate_scores[utt_id] - baseline_scores[utt_id] for utt_id in common_ids]

    wins = sum(delta < -args.tie_eps for delta in deltas)
    losses = sum(delta > args.tie_eps for delta in deltas)
    ties = len(deltas) - wins - losses

    threshold_baseline = sum(score <= args.threshold for score in baseline_common)
    threshold_candidate = sum(score <= args.threshold for score in candidate_common)

    ranked = sorted(
        (
            {
                "utt_id": utt_id,
                "baseline": baseline_scores[utt_id],
                "candidate": candidate_scores[utt_id],
                "delta": candidate_scores[utt_id] - baseline_scores[utt_id],
            }
            for utt_id in common_ids
        ),
        key=lambda item: item["delta"],
    )
    improvements = ranked[: args.top_k]
    regressions = list(reversed(ranked[-args.top_k :])) if ranked else []

    summary = {
        "baseline_info": baseline_info,
        "candidate_info": candidate_info,
        "latency": args.latency,
        "threshold": args.threshold,
        "baseline_total": len(baseline_ids),
        "candidate_total": len(candidate_ids),
        "common_total": len(common_ids),
        "baseline_only_total": len(baseline_only),
        "candidate_only_total": len(candidate_only),
        "baseline_stats_common": summarize(baseline_common),
        "candidate_stats_common": summarize(candidate_common),
        "delta_stats": summarize(deltas),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "threshold_baseline": threshold_baseline,
        "threshold_candidate": threshold_candidate,
        "threshold_baseline_ratio": (threshold_baseline / len(common_ids)) if common_ids else float("nan"),
        "threshold_candidate_ratio": (threshold_candidate / len(common_ids)) if common_ids else float("nan"),
        "top_improvements": improvements,
        "top_regressions": regressions,
        "baseline_only_examples": baseline_only[: args.top_k],
        "candidate_only_examples": candidate_only[: args.top_k],
    }

    print(f"Baseline : {args.baseline}")
    print(f"Candidate: {args.candidate}")
    print(f"Latency  : {args.latency}")
    print()
    print(fmt_stats("Baseline(common)", summary["baseline_stats_common"]))
    print(fmt_stats("Candidate(common)", summary["candidate_stats_common"]))
    print(fmt_stats("Delta(candidate-baseline)", summary["delta_stats"]))
    print()
    print(
        f"Common utts={summary['common_total']} | baseline_only={summary['baseline_only_total']} | "
        f"candidate_only={summary['candidate_only_total']}"
    )
    print(
        f"Pairwise wins(lower better): candidate_better={wins} tie={ties} candidate_worse={losses}"
    )
    if common_ids:
        print(
            f"Threshold <= {args.threshold:.2f}: baseline={threshold_baseline}/{len(common_ids)} "
            f"({summary['threshold_baseline_ratio']:.2%}) | candidate={threshold_candidate}/{len(common_ids)} "
            f"({summary['threshold_candidate_ratio']:.2%})"
        )
    else:
        print("No overlapping utterances after filtering; nothing to compare.")

    if baseline_info["duplicates"]:
        print(
            f"Baseline duplicate utt_ids (last one kept): {baseline_info['duplicates'][:args.top_k]}"
        )
    if candidate_info["duplicates"]:
        print(
            f"Candidate duplicate utt_ids (last one kept): {candidate_info['duplicates'][:args.top_k]}"
        )

    print("\nTop improvements (candidate lower than baseline):")
    for item in improvements:
        print(
            f"  {item['utt_id']}: delta={item['delta']:.4f} "
            f"baseline={item['baseline']:.4f} candidate={item['candidate']:.4f}"
        )

    print("\nTop regressions (candidate higher than baseline):")
    for item in regressions:
        print(
            f"  {item['utt_id']}: delta={item['delta']:.4f} "
            f"baseline={item['baseline']:.4f} candidate={item['candidate']:.4f}"
        )

    if baseline_only:
        print(
            f"\nBaseline-only utt_ids (first {min(args.top_k, len(baseline_only))}): "
            f"{baseline_only[:args.top_k]}"
        )
    if candidate_only:
        print(
            f"Candidate-only utt_ids (first {min(args.top_k, len(candidate_only))}): "
            f"{candidate_only[:args.top_k]}"
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON summary to {out_path}")


if __name__ == "__main__":
    main()

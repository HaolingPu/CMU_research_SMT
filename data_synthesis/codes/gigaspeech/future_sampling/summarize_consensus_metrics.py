#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


def load_consensus_stats(experiment_dir: Path) -> Dict[str, float]:
    values_bleu: List[float] = []
    values_laal: List[float] = []
    json_files = sorted(experiment_dir.glob("task_*/*.json"))
    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = data.get("metrics") or {}
        bleu = metrics.get("bleu_char")
        laal = metrics.get("laal_text")
        if isinstance(bleu, (int, float)) and not math.isnan(float(bleu)):
            values_bleu.append(float(bleu))
        if isinstance(laal, (int, float)) and not math.isnan(float(laal)):
            values_laal.append(float(laal))
    return {
        "count": len(json_files),
        "bleu_avg": (sum(values_bleu) / len(values_bleu)) if values_bleu else float("nan"),
        "laal_avg": (sum(values_laal) / len(values_laal)) if values_laal else float("nan"),
        "bleu_min": min(values_bleu) if values_bleu else float("nan"),
        "bleu_max": max(values_bleu) if values_bleu else float("nan"),
        "laal_min": min(values_laal) if values_laal else float("nan"),
        "laal_max": max(values_laal) if values_laal else float("nan"),
    }


def load_metricx_stats(metricx_dir: Path) -> Optional[Dict[str, float]]:
    summary_json = metricx_dir / "summary.json"
    if not summary_json.is_file():
        return None
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    stats = data.get("stats") or {}
    return {
        "qe_mean": float(stats.get("mean", float("nan"))),
        "qe_median": float(stats.get("median", float("nan"))),
        "qe_min": float(stats.get("min", float("nan"))),
        "qe_max": float(stats.get("max", float("nan"))),
        "qe_leq3": float(data.get("threshold_ratio", float("nan"))),
    }


def fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize BLEU / LAAL / MetricX QE for one consensus experiment.")
    parser.add_argument("--experiment-dir", required=True, help="Consensus decoding experiment directory.")
    parser.add_argument("--metricx-dir", default="", help="Optional MetricX QE directory for the same experiment.")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    cons = load_consensus_stats(exp_dir)
    metricx = load_metricx_stats(Path(args.metricx_dir)) if args.metricx_dir else None

    print(f"Experiment : {exp_dir}")
    print(f"Count      : {cons['count']}")
    print(f"BLEU avg   : {fmt(cons['bleu_avg'])}")
    print(f"LAAL avg   : {fmt(cons['laal_avg'])}")
    print(f"BLEU range : {fmt(cons['bleu_min'])} .. {fmt(cons['bleu_max'])}")
    print(f"LAAL range : {fmt(cons['laal_min'])} .. {fmt(cons['laal_max'])}")
    if metricx is not None:
        print(f"QE mean    : {fmt(metricx['qe_mean'])}")
        print(f"QE median  : {fmt(metricx['qe_median'])}")
        print(f"QE range   : {fmt(metricx['qe_min'])} .. {fmt(metricx['qe_max'])}")
        print(f"QE <= 3.0  : {fmt(metricx['qe_leq3'])}")
    else:
        print("QE         : unavailable")


if __name__ == "__main__":
    main()

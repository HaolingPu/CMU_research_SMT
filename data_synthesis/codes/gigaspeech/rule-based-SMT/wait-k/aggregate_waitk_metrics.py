#!/usr/bin/env python3
"""Aggregate BLEU / LAAL / MetricX QE for wait-k configs.

Emits a table matching the LA-2 format:
  k | BLEU | LAAL | MetricX (avg) | QE<=3.0 | QE<=3.0%
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

WAITK_DIR = Path(
    "/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/wait-k"
)
OUTPUT_DIR = WAITK_DIR / "output"
K_VALUES = [3, 6, 9, 12, 15]
QE_THRESHOLD = 3.0


def _nan(x: float) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def load_bleu_laal(run_dir: Path) -> tuple[list[float], list[float]]:
    bleus, laals = [], []
    for path in sorted(run_dir.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        m = obj.get("metrics") or {}
        b, l = m.get("bleu_char"), m.get("laal_text")
        if not _nan(b):
            bleus.append(float(b))
        if not _nan(l):
            laals.append(float(l))
    return bleus, laals


def load_metricx_qe(metricx_path: Path) -> list[float]:
    scores = []
    if not metricx_path.exists():
        return scores
    with metricx_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            score = obj.get("prediction")
            if score is None:
                continue
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                continue
    return scores


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def main() -> None:
    print(f"{'k':>4}  {'BLEU':>7}  {'LAAL':>7}  {'MetricX(avg)':>13}  "
          f"{'QE<=3.0':>8}  {'QE<=3.0%':>9}")
    print("-" * 60)
    rows = []
    for k in K_VALUES:
        run_dir = OUTPUT_DIR / f"waitk{k}_100_stride1"
        metricx_out = OUTPUT_DIR / f"waitk{k}_metricx_output.jsonl"

        bleus, laals = load_bleu_laal(run_dir)
        qes = load_metricx_qe(metricx_out)

        bleu_avg = mean(bleus)
        laal_avg = mean(laals)
        qe_avg = mean(qes)
        qe_pass = sum(1 for s in qes if s <= QE_THRESHOLD)
        n_qe = len(qes)
        qe_pct = 100.0 * qe_pass / n_qe if n_qe else float("nan")

        rows.append(
            {
                "k": k,
                "n_jsons": len(bleus),
                "n_qe": n_qe,
                "bleu": bleu_avg,
                "laal": laal_avg,
                "metricx_avg": qe_avg,
                "qe_pass": qe_pass,
                "qe_pct": qe_pct,
            }
        )
        print(
            f"{k:>4}  {bleu_avg:>7.2f}  {laal_avg:>7.2f}  "
            f"{qe_avg:>13.3f}  {qe_pass:>8d}  {qe_pct:>8.2f}%"
        )

    summary_path = OUTPUT_DIR / "waitk_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

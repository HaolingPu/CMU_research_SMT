#!/usr/bin/env python3
"""Aggregate BLEU / LAAL / MetricX QE for the PA 100-case run.

Emits the same table format as wait-k:
  method | BLEU | LAAL | MetricX (avg) | QE<=3.0 | QE<=3.0%
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

PA_DIR = Path(
    "/home/haolingp/CMU_research_SMT/data_synthesis/codes/gigaspeech/rule-based-SMT/prefix_alignment"
)
OUTPUT_DIR = PA_DIR / "output"
RUN_DIR = OUTPUT_DIR / "pa_paper_100"
METRICX_OUT = OUTPUT_DIR / "pa_paper_100_metricx_output.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "pa_paper_100_summary.json"
SUMMARY_TXT = OUTPUT_DIR / "pa_paper_100_summary.txt"
QE_THRESHOLD = 3.0


def _is_nan(x) -> bool:
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
        if not _is_nan(b):
            bleus.append(float(b))
        if not _is_nan(l):
            laals.append(float(l))
    return bleus, laals


def load_qe(metricx_path: Path) -> list[float]:
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
    bleus, laals = load_bleu_laal(RUN_DIR)
    qes = load_qe(METRICX_OUT)

    bleu_avg = mean(bleus)
    laal_avg = mean(laals)
    qe_avg = mean(qes)
    qe_pass = sum(1 for s in qes if s <= QE_THRESHOLD)
    n_qe = len(qes)
    qe_pct = 100.0 * qe_pass / n_qe if n_qe else float("nan")

    method_label = "PA (paper, BERTScore zh, char)"
    header = (
        f"{'method':<32}  {'BLEU':>7}  {'LAAL':>7}  {'MetricX(avg)':>13}  "
        f"{'QE<=3.0':>8}  {'QE<=3.0%':>9}"
    )
    row = (
        f"{method_label:<32}  {bleu_avg:>7.2f}  {laal_avg:>7.2f}  "
        f"{qe_avg:>13.3f}  {qe_pass:>8d}  {qe_pct:>8.2f}%"
    )
    sep = "-" * len(header)

    print(header)
    print(sep)
    print(row)
    print()
    print(f"n_jsons={len(bleus)}  n_qe={n_qe}")

    summary = {
        "method": method_label,
        "n_jsons": len(bleus),
        "n_qe": n_qe,
        "bleu_char": bleu_avg,
        "laal_text": laal_avg,
        "metricx_avg": qe_avg,
        "qe_pass_count": qe_pass,
        "qe_pass_pct": qe_pct,
        "qe_threshold": QE_THRESHOLD,
        "run_dir": str(RUN_DIR),
        "metricx_output": str(METRICX_OUT),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    SUMMARY_TXT.write_text(
        f"{header}\n{sep}\n{row}\n\nn_jsons={len(bleus)}  n_qe={n_qe}\n",
        encoding="utf-8",
    )
    print(f"\nSummary saved:\n  {SUMMARY_PATH}\n  {SUMMARY_TXT}")


if __name__ == "__main__":
    main()

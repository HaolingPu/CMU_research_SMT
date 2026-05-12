#!/usr/bin/env python3
"""Plot LongYAAL (CU) vs each quality metric on the future-aware test set.

One PNG per metric. Each figure shows 3 models x 2 speeds (1.0 and 0.7),
with color encoding the model and linestyle encoding the speed.
"""
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

CKPT_ROOT = Path("/data/user_data/siqiouya/ckpts/infinisst-omni")
SEGS = [960, 1920, 2880, 3840]
SPEEDS = ["1.0", "0.7"]
METRICS = ["BLEU", "chrF", "COMET", "BLEURT", "TASER"]
LATENCY_KEY = "LongYAAL (CU)"

MODELS = [
    ("s_origin", "gigaspeech-zh-s_origin-bsz4/v1-20260122-055820-hf"),
    ("hibiki", "gigaspeech-zh-hibiki-s-bsz4/v0-20260326-141050-hf"),
    ("consensus-topk5_f200", "gigaspeech-zh-consensus-topk5_f200-s-bsz4/v0-20260502-125501-hf"),
]

OUT_DIR = Path(__file__).resolve().parent

SPEED_STYLE = {"1.0": "-", "0.7": "--"}
SPEED_MARKER = {"1.0": "o", "0.7": "s"}


def parse_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    if not path.exists():
        return scores
    with path.open() as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            try:
                scores[row[0]] = float(row[1])
            except ValueError:
                pass
    return scores


def collect(model_path: str, speed: str):
    yaal: list[float] = []
    metric_vals: dict[str, list[float]] = {m: [] for m in METRICS}
    for seg in SEGS:
        tsv = (
            CKPT_ROOT
            / model_path
            / "evaluation/future_aware_test/en-zh"
            / f"spd_{speed}_seg{seg}"
            / "segmentation_output"
            / "scores.tsv"
        )
        s = parse_scores(tsv)
        yaal.append(s.get(LATENCY_KEY, math.nan))
        for m in METRICS:
            metric_vals[m].append(s.get(m, math.nan))
    return yaal, metric_vals


def main():
    colors = plt.cm.tab10.colors
    model_color = {label: colors[i] for i, (label, _) in enumerate(MODELS)}

    data = {}  # (label, speed) -> (yaal, metric_vals)
    for label, path in MODELS:
        for speed in SPEEDS:
            data[(label, speed)] = collect(path, speed)

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, _ in MODELS:
            for speed in SPEEDS:
                yaal, mvals = data[(label, speed)]
                ax.plot(
                    yaal,
                    mvals[metric],
                    color=model_color[label],
                    linestyle=SPEED_STYLE[speed],
                    marker=SPEED_MARKER[speed],
                    linewidth=2,
                    markersize=7,
                    label=f"{label} (speed={speed})",
                )
        ax.set_xlabel("LongYAAL (CU) [ms]")
        ax.set_ylabel(metric)
        ax.set_title(f"future_aware_test en-zh: LongYAAL (CU) vs {metric}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = OUT_DIR / f"latency_quality_future_aware_{metric}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
